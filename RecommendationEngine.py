import os

from faiss import read_index
from models import Paper
from sqlalchemy.exc import IntegrityError

from EmbeddingService import EmbeddingService
from ArxivService import ArxivService
from job_queue import job_queue
from background_tasks import write_to_bucket

class PaperIndex:
    def __init__(self, local_faiss_path: str):
        self.local_path = local_faiss_path
        self.index = None
        self.initialized = False

    def _init_index(self):
        if not os.path.exists(self.local_path):
            raise FileNotFoundError('Missing local file')
        self.index = read_index(self.local_path)
        self.initialized = True

    def ensure_initialized(self):
        if not self.initialized:
            self._init_index()

    def add_embedding(self, new_emb):
        self.ensure_initialized()
        self.index.add(new_emb)
        # job_queue.enqueue(write_to_bucket, self.index)

    def search(self, query_vec, k):
        self.ensure_initialized()
        scores, indices = self.index.search(query_vec, k)
        return scores[0], indices[0].tolist()


class Papers:
    def __init__(self, db_session):
        self.db = db_session

    def add_paper(self, db_id, arxiv_id, title, authors, url):
        paper = Paper(id=db_id, arxiv_id=arxiv_id,
                      title=title, authors=authors, url=url)
        try:
            self.db.add(paper)
            self.db.commit()
        except IntegrityError as e:
            self.db.rollback()
            raise ValueError('Title already present in database')
        except Exception as e:
            self.db.rollback()
            raise e

    def get_papers_by_ids(self, ids):
        # NOTE: A single filter query doesn't necessarily preserve the order of papers by ID
        # Doing an individual query per retrieved ID is probably fine, since k is likely small
        return [self.db.query(Paper).filter(Paper.id == row_id).one() for row_id in ids]

    def get_total_papers(self, x=3):
        """
        Get approximate number of papers in database

        Args:
            x: Position to round to (i.e round to the nearest 10 ** x place)
        Returns:
            int: Number of papers in DB, rounded to some threshold
        """
        num_papers = self.db.query(Paper).count()
        thresh = 10 ** x
        return int(thresh * (num_papers // thresh))


class RecommendationEngine:
    """
    Cohere-based recommendation system for research papers published to arXiv
    Relies on prebuilt FAISS indices for facilitating fast kNN
    """

    def __init__(self, index: PaperIndex, papers: Papers):
        self.paper_index = index
        self.papers = papers

    def recommend(self, query: str, k: int):
        """
        Recommend k papers using abstract embeddings

        Args:
            query (str): Paper title or equivalent arXiv ID
            k (int): Number of papers to recommend

        Returns:
            list: Recommended papers
        """

        emb = EmbeddingService.embed_query(query)
        scores, indices = self.paper_index.search(emb, k)

        papers = self.papers.get_papers_by_ids(indices)
        results = []
        for idx, paper in enumerate(papers):
            results.append(paper.to_dict(score=100 * scores[idx]))
        return results

    def add_by_id(self, arxiv_id: str):
        paper_data = ArxivService.get_paper_by_id(arxiv_id=arxiv_id)
        db_id = 1 + self.papers.get_total_papers(0)
        self.papers.add_paper(db_id=db_id, arxiv_id=paper_data['arxiv_id'], title=paper_data['title'],
                              authors=paper_data['authors'], url=paper_data['url'])
        doc = paper_data['title'] + ' ' + paper_data['abstract']
        emb = EmbeddingService.embed_query(query=doc, is_document=True)
        self.paper_index.add_embedding(emb)

    def add_by_title(self, title: str, abstract: str):
        paper_data = ArxivService.get_paper_by_title(title)
        db_id = 1 + self.papers.get_total_papers(0)
        self.papers.add_paper(db_id=db_id, arxiv_id=paper_data['arxiv_id'], title=paper_data['title'],
                              authors=paper_data['authors'], url=paper_data['url'])
        doc = paper_data['title'] + ' ' + paper_data['abstract']
        emb = EmbeddingService.embed_query(query=doc, is_document=True)
        self.paper_index.add_embedding(emb)
