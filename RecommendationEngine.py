from sqlalchemy.exc import IntegrityError
from models import Paper

from EmbeddingService import EmbeddingService
from ArxivService import ArxivService
from PaperStorage import restore_from_bucket, write_to_bucket
from worker import get_job_queue

class RecommendationEngine:
    """
    Cohere-based recommendation system for research papers published to arXiv
    Relies on prebuilt FAISS indices for facilitating fast kNN
    """

    def __init__(self, db_session):
        self.session = db_session
        self.queue = get_job_queue()
        self.paper_index = None

    def recommend(self, query: str, k: int):
        """
        Recommend k papers using abstract embeddings

        Args:
            query (str): Paper title or equivalent arXiv ID
            k (int): Number of papers to recommend

        Returns:
            list: Recommended papers
        """

        # Lazily initialize FAISS index for faster boot
        if not self.paper_index:
            self.paper_index = restore_from_bucket()

        emb = EmbeddingService.embed_query(query)
        scores, indices_list = self.paper_index.search(emb, k)
        indices = indices_list[0].tolist()

        results = []
        # NOTE: A single filter query doesn't necessarily preserve the order of papers by ID
        # Doing an individual query per retrieved ID is probably fine, since k is likely small
        papers = [self.session.query(Paper).filter(
            Paper.id == id).one() for id in indices]
        for idx, paper in enumerate(papers):
            results.append(paper.to_dict(score=100 * scores[0][idx]))
        return results

    def get_total_papers(self, x=3):
        """
        Get approximate number of papers in database

        Args:
            x: Position to round to (i.e round to the nearest 10 ** x place)
        Returns:
            int: Number of papers in DB, rounded to some threshold
        """
        num_papers = self.session.query(Paper).count()
        thresh = 10 ** x
        return int(thresh * (num_papers // thresh))

    def add_by_title(self, title: str, abstract: str):
        """
        Add paper to database

        Args:
            title: Title of paper to add
            abstract: Abstract of paper to add
        Returns:
            None
        """

        paper_data = ArxivService.get_paper_by_title(title)
        db_id = self.get_total_papers(0) + 1
        try:
            new_paper = Paper(id=db_id, arxiv_id=paper_data['arxiv_id'], title=paper_data['title'],
                              authors=paper_data['authors'], url=paper_data['url'])
            self.session.add(new_paper)
            self.session.commit()
        except IntegrityError:
            # Likely due to a uniqueness violation
            self.session.rollback()
            raise ValueError('Title already present in database')
        except Exception as e:
            self.session.rollback()
            raise e

        emb = EmbeddingService.embed_query(
            paper_data['title'] + ' ' + paper_data['abstract'], is_document=True)
        if not self.paper_index:
            self.paper_index = restore_from_bucket()
        self.paper_index.add(emb)
        self.queue.enqueue(write_to_bucket, self.paper_index)

    def add_by_id(self, arxiv_id: str):
        """
        Add paper to database from arXiv ID

        Args:
            arxiv_id: arXiv ID of paper to add
        Returns:
            None
        """

        paper_data = ArxivService.get_paper_by_id(arxiv_id)
        db_id = self.get_total_papers(0) + 1
        # Add new paper to DB
        try:
            new_paper = Paper(id=db_id, arxiv_id=arxiv_id, title=paper_data['title'],
                              authors=paper_data['authors'], url=paper_data['url'])
            self.session.add(new_paper)
            self.session.commit()
        except IntegrityError:
            # Likely a violation of uniqueness of arXiv ID
            self.session.rollback()
            raise ValueError('Title already present in database')
        except Exception as e:
            self.session.rollback()
            raise e

        emb = EmbeddingService.embed_query(
            paper_data['title'] + ' ' + paper_data['abstract'], is_document=True)
        if not self.paper_index:
            self.paper_index = restore_from_bucket()
        self.paper_index.add(emb)
        self.queue.enqueue(write_to_bucket, self.paper_index)
