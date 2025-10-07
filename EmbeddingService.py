import numpy as np
import requests
import json 

from config import Config
from ArxivService import ArxivService
from pylatexenc.latex2text import LatexNodes2Text

class EmbeddingService:
    """
    Wrapper class for the Cohere v3 embedding model
    """
    ENDPOINT = f'{Config.EMBEDDING_URL}/v1/embeddings'
    HEADER = {
        'Authorization': f'Bearer {Config.EMBEDDING_KEY}',
        'Content-Type': 'application/json'
    }
    LATEX_CONV = LatexNodes2Text()
    
    @staticmethod
    def _preprocess(text: str, is_document: bool):
        """
        Preprocess query to abide by Cohere's recommendations for embedding
        
        Args:
            text (str): raw text to be embedded
            is_document (bool): query corresponds to a document (default: False)
            
        Returns:
            new_text (str): preprocessed text suitable for embedding    
        """
        # LaTeX commands can mess up the POST request to the Embed API
        # Replace these with their natural language equivalent
        if is_document:
            text = EmbeddingService.LATEX_CONV.latex_to_text(text)
        
        # The Embed API has a 2048 character/512 token limit
        # TODO: Add compliance to the latter requirement
        # NOTE: Will require envelope math, since we don't want to pull in HF tokenizer
        return text[:2048]
    
    @staticmethod
    def embed_query(query: str, is_document=False):
        """
        Embed query in Cohere's embedding space
        
        Args:
            query (str): either an arxiv_id or raw text to be embedded
            is_document (bool): query corresponds to a document (default: False)
            
        Returns:
            emb (np.array): L2-normalized embedding of query/document 
        """
        
        if ArxivService.is_valid_arxiv_id(query):
            paper_data = ArxivService.get_paper_by_id(query)
            input_data = paper_data['title'] + ' ' + paper_data['abstract']
        else:
            input_data = query
            
        input_data = EmbeddingService._preprocess(input_data, is_document)
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