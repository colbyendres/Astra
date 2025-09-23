import json
import re

import faiss
import numpy as np
import requests

from arxiv import Search

from config import Config
from models import Paper


class RecommendationEngine:
    """
    SPECTER-based recommendation system for research papers published to arXiv
    Relies on prebuilt FAISS indices for facilitating fast kNN
    """
    IS_ARXIV_ID = re.compile(r'[0-9]{4}\.[0-9]{5}')
    ENDPOINT = f'{Config.EMBEDDING_URL}/v1/embeddings'
    
    def __init__(self, faiss_paper_path: str, db_session):
        self.session = db_session
        self.index_path = faiss_paper_path
        self.paper_index = None
        self.initialized = False

    def _encode_query(self, query: str):            
        # Determine if we're given an arXiv ID and convert it to title
        match = RecommendationEngine.IS_ARXIV_ID.search(query)
        if match:
            arxiv_res = Search(id_list=[match.group()], max_results=1)
            query_title = next(arxiv_res.results()).title
        else:
            query_title = query

        header = {
            "Authorization": f"Bearer {Config.EMBEDDING_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": Config.EMBEDDING_MODEL_ID,
            "input": query_title,
            "input_type": "search_query",
            "allow_ignored_params": True
        }
        rsp = requests.post(url=self.ENDPOINT, headers=header, data=json.dumps(payload), timeout=10)
        rsp.raise_for_status()
        emb = np.array(rsp.json()['data'][0]['embedding']).reshape(1,-1)
        
        # Our embeddings are already normalized in the FAISS index, so match that
        return emb / np.linalg.norm(emb, ord=2)

    def _init_faiss_index(self):
        import os 
        print(f'Index path {self.index_path} exists? {os.path.exists(self.index_path)}')

        self.paper_index = faiss.read_index(self.index_path)
        self.initialized = True 
        
    def recommend(self, query: str, k: int):
        """
        Recommend k papers using abstract embeddings

        Args:
            query (str): Paper title or equivalent arXiv ID
            k (int): Papers to recommend

        Returns:
            list: Recommended papers
        """
        
        # Lazily initialize FAISS index for faster boot
        if not self.initialized:
            self._init_faiss_index()
            
        emb = self._encode_query(query)
        scores, indices_list = self.paper_index.search(emb, k)
        indices = indices_list[0].tolist()

        results = []
        # NOTE: A single filter query doesn't necessarily preserve the order of papers by ID
        # Doing an individual query per retrieved ID is probably fine, since k is likely small
        papers = [self.session.query(Paper).filter(Paper.id == id).one() for id in indices]
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
        
