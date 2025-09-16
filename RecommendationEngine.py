from torch import no_grad
from arxiv import Search
from models import Paper
import faiss
import re

class RecommendationEngine:
    """
    SPECTER-based recommendation system for research papers published to arXiv
    Relies on prebuilt FAISS indices for facilitating fast kNN
    """
    IS_ARXIV_ID = re.compile(r'[0-9]{4}\.[0-9]{5}')

    def __init__(self, faiss_abstract_idx: str, db_session):
        self.session = db_session
        self.index_path = faiss_abstract_idx
        self.model = None 
        self.index_abstract = None 
        self.tokenizer = None 
        self.initialized = False 
        
    def _initialize_model(self):
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel
                
        self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        self.model.load_adapter("allenai/specter2", source="hf",
                                 load_as="specter2", set_active=True)
        self.model.eval()
        self.initialized = True
        self.index_abstract = faiss.read_index(self.index_path)

    def _encode_query(self, query: str):
        # Lazily initialize model, tokenizers, etc.
        if not self.initialized:
            self._initialize_model()
            
        # Determine if we're given an arXiv ID and convert it to title
        match = RecommendationEngine.IS_ARXIV_ID.search(query)
        if match:
            arxiv_res = Search(id_list=[match.group()], max_results=1)
            query_title = next(arxiv_res.results()).title
        else:
            query_title = query

        # Get embedding of query for search
        with no_grad():
            inputs = self.tokenizer(
                query_title,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512
            )

            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0,
                                            :].numpy().astype("float32")
        # Our embeddings in the FAISS index are unit vectors, so normalize here
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
        
