from arxiv import Search
from models import Paper
import faiss
import requests
import re 

class RecommendationEngine:
    """
    SPECTER-based recommendation system for research papers published to arXiv
    Relies on prebuilt FAISS indices for facilitating fast kNN
    """
    IS_ARXIV_ID = re.compile(r'[0-9]{4}\.[0-9]{5}')

    def __init__(self, faiss_abstract_idx: str, db_session):
        self.index_abstract = faiss.read_index(faiss_abstract_idx)
        self.session = db_session

    def _encode_query(self, query: str):            
        # Determine if we're given an arXiv ID and convert it to title
        match = RecommendationEngine.IS_ARXIV_ID.search(query)
        if match:
            arxiv_res = Search(id_list=[match.group()], max_results=1)
            query_title = next(arxiv_res.results()).title
        else:
            query_title = query

        # Query the model service for the embedding of our query title
        rsp = requests.post('foo/encode', json={'query': query_title}, timeout=10)
        rsp.raise_for_status()
        emb = rsp.json()['embedding']
        
        # Our precomputed embedding are unit norm, so do the same to query
        faiss.normalize_L2(emb)
        return emb

    def recommend(self, query: str, k: int):
        """
        Recommend k papers using abstract embeddings

        Args:
            query (str): Paper title or equivalent arXiv ID
            k (int): Papers to recommend

        Returns:
            list: Recommended papers
        """
        emb = self._encode_query(query)
        scores, indices = self.index_abstract.search(emb, k)
        # These are zero-based indices, not arXiv IDs
        results = []
        papers = self.session.query(Paper).filter(Paper.id.in_(indices[0].tolist())).all()
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
        
