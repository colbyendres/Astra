import numpy as np
import requests
import json 

from config import Config
from ArxivService import ArxivService

class EmbeddingService:
    """
    Wrapper class for the Cohere v3 embedding model
    """
    ENDPOINT = f'{Config.EMBEDDING_URL}/v1/embeddings'
    HEADER = {
        'Authorization': f'Bearer {Config.EMBEDDING_KEY}',
        'Content-Type': 'application/json'
    }
    
    @staticmethod
    def embed_query(query: str, is_document=False):
        """
        Embed query in Cohere's embedding space
        
        Args:
            query (str): either an arxiv_id or raw text to be embedded
            is_document (bool): query corresponds to a document (default: False)
        """
        
        if ArxivService.is_valid_arxiv_id(query):
            paper_data = ArxivService.get_paper_by_id(query)
            input_data = paper_data['title'] + ' ' + paper_data['abstract']
        else:
            input_data = query
            
        query_type = "search_document" if is_document else "search_query"
        payload = {
            "model": Config.EMBEDDING_MODEL_ID,
            "input": input_data,
            "input_type": query_type,
            "allow_ignored_params": True
        }
        rsp = requests.post(url=EmbeddingService.ENDPOINT, headers=EmbeddingService.HEADER,
                            data=json.dumps(payload), timeout=10)
        rsp.raise_for_status()
        emb = np.array(rsp.json()['data'][0]['embedding']).reshape(1, -1)

        # Our embeddings are already normalized in the FAISS index, so match that
        return emb / np.linalg.norm(emb, ord=2)