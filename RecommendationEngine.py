import os
import threading 
import logging

from faiss import read_index
from models import Paper
from sqlalchemy.exc import IntegrityError

from EmbeddingService import EmbeddingService
from ArxivService import ArxivService
from background_tasks import write_to_bucket

class PaperIndex:
    """
    Abstraction for FAISS-like index that holds paper embeddings
    Currently uses the faiss python library
    """
    def __init__(self, local_faiss_path: str):
        self.local_path = local_faiss_path
        self.index = None
        self.initialized = False
        
    def _init_index(self):
        if not os.path.exists(self.local_path):
            raise FileNotFoundError('Missing local file')
        logging.debug('Reading FAISS index from local file')
        self.index = read_index(self.local_path)
        self.initialized = True

    def ensure_initialized(self):
        """
        Ensure that the FAISS index is initialized before add/search
        """
        if not self.initialized:
            self._init_index()

    def add_embedding(self, new_emb):
        """
        Add embedding to the index
        
        Args:
            new_emb: New embedding to add
            
        Returns:
            None
        """
        self.ensure_initialized()
        logging.debug('Adding paper to FAISS index')
        self.index.add(new_emb)
        threading.Thread(target=write_to_bucket, args=(self.index, ), daemon=True).start()

    def search(self, query_vec, k: int):
        """
        Searches index for k closest vectors
        
        Args:
            query_vec: embedding of query
            k (int): number of vectors to return
            
        Returns:
            scores (list): Similarity scores for each retrieved embedding
            indices (list): id of retrieved embedding in database
        """
        self.ensure_initialized()
        scores, indices = self.index.search(query_vec, k)
        return scores[0], indices[0].tolist()


class Papers:
    """
    Abstraction of database containing paper metadata
    Currently uses Postgres
    """
    def __init__(self, db_session):
        self.db = db_session

    def add_paper(self, db_id: int, arxiv_id: str, title: str, authors: str, url: str):
        """
        Add paper to database
        
        Args:
            db_id (int): id/primary key of new paper
            arxiv_id (str): arXiv id of new paper
            title (str): title of new paper
            authors (str): list of author(s) of new paper
            url (str): url of new paper
            
        Returns:
            None
        """
        paper = Paper(id=db_id, arxiv_id=arxiv_id,
                      title=title, authors=authors, url=url)
        try:
            self.db.add(paper)
            self.db.commit()
            logging.info(f'Paper with title {title} successfully added')
        except IntegrityError as e:
            logging.error(f'IntegrityError: {e.msg}, rolling back transaction')
            self.db.rollback()
            raise ValueError('Title already present in database')
        except Exception as e:
            logging.error(f'Exception {e.msg} raised, rolling back transaction')
            self.db.rollback()
            raise e

    def get_papers_by_ids(self, ids):
        """
        Return paper metadata by id
        
        Args:
            ids (list[int]): List of ids for papers to return
        
        Returns:
            papers (list[Paper]): Paper ORM objects matching ids
        """
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
        Recommend k papers using title/abstract embeddings

        Args:
            query (str): Paper title or equivalent arXiv ID
            k (int): Number of papers to recommend

        Returns:
            list: Recommended papers
        """

        emb = EmbeddingService.embed_query(query)
        scores, indices = self.paper_index.search(emb, k)
        logging.info(f'Found {k} papers with ids: {indices}')
        
        papers = self.papers.get_papers_by_ids(indices)
        results = []
        for idx, paper in enumerate(papers):
            results.append(paper.to_dict(score=100 * scores[idx]))
        return results

    def add_by_id(self, arxiv_id: str):
        """
        Add paper to DB via the arXiv id
        
        Args:
            arxiv_id (str): arXiv id of paper to add
            
        Returns:
            None
        """
        paper_data = ArxivService.get_paper_by_id(arxiv_id=arxiv_id)
        db_id = self.papers.get_total_papers(0)
        self.papers.add_paper(db_id=db_id, arxiv_id=paper_data['arxiv_id'], title=paper_data['title'],
                              authors=paper_data['authors'], url=paper_data['url'])
        doc = paper_data['title'] + ' ' + paper_data['abstract']
        emb = EmbeddingService.embed_query(query=doc, is_document=True)
        self.paper_index.add_embedding(emb)

    def add_by_title(self, title: str, abstract: str, authors: str, url: str, arxiv_id: str):
        """
        Add paper to DB via metadata
        
        Args:
            db_id (int): id/primary key of new paper
            arxiv_id (str): arXiv id of new paper
            title (str): title of new paper
            authors (str): list of author(s) of new paper
            url (str): url of new paper
            
        Returns:
            None
        """
        db_id = self.papers.get_total_papers(0)
        author_list = str([author.strip() for author in authors.split(sep=',')])
        self.papers.add_paper(db_id, arxiv_id, title, author_list, url)
        doc = title + ' ' + abstract
        emb = EmbeddingService.embed_query(query=doc, is_document=True)
        self.paper_index.add_embedding(emb)
