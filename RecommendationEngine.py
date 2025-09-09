from ast import literal_eval
import regex as re
import torch
import faiss
import pandas as pd
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from titlecase import titlecase
import arxiv


class RecommendationEngine:
    """
    SPECTER-based recommendation system for research papers published to arXiv
    Relies on prebuilt FAISS indices for facilitating fast kNN
    """
    IS_ARXIV_ID = re.compile(r'[0-9]{4}\.[0-9]{5}')

    def __init__(self, faiss_abstract_idx: str, faiss_title_idx: str, paper_metadata: str):
        self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        self.model.load_adapter("allenai/specter2", source="hf",
                                load_as="specter2", set_active=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.index_abstract = faiss.read_index(faiss_abstract_idx)
        self.index_title = faiss.read_index(faiss_title_idx)
        self.paper_metadata = pd.read_csv(paper_metadata)

    def _encode_query(self, query: str):
        # Determine if we're given an arXiv ID and convert it to title
        match = RecommendationEngine.IS_ARXIV_ID.search(query)
        if match:
            arxiv_res = arxiv.Search(id_list=[match.group()], max_results=1)
            query_title = next(arxiv_res.results()).title
        else:
            query_title = query

        # Get embedding of query for search
        with torch.no_grad():
            inputs = self.tokenizer(
                query_title,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512
            ).to(self.device)

            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0,
                                            :].cpu().numpy().astype("float32")
        # Our embeddings in the FAISS index are unit vectors, so normalize here
        faiss.normalize_L2(emb)
        return emb

    def recommend_from_title(self, query: str, k: int):
        """
        Recommend k papers using the title-only embeddings

        Args:
            query (str): Paper title or equivalent arXiv ID
            k (int): Papers to recommend

        Returns:
            list: Recommended papers
        """
        emb = self._encode_query(query)
        scores, idxs = self.index_title.search(emb, k)
        return self._format_results(scores[0], idxs[0])

    def recommend_from_abstract(self, query: str, k: int):
        """
        Recommend k papers using the abstract-only embeddings

        Args:
            query (str): Paper title or equivalent arXiv ID
            k (int): Papers to recommend

        Returns:
            list: Recommended papers
        """
        emb = self._encode_query(query)
        scores, idxs = self.index_abstract.search(emb, k)
        return self._format_results(scores[0], idxs[0])

    def _format_results(self, scores: list[float], indices: list[int]):
        results = []
        for idx, paper_idx in enumerate(indices):
            paper = self.paper_metadata.iloc[paper_idx]
            author_list = literal_eval(paper['authors'])
            if len(author_list) > 2:
                authors = ', '.join(author_list[:2]).title() + ' et al.'
            else:
                authors = ', '.join(author_list).title()
            results.append(
                {
                    'title': titlecase(paper['title']),
                    'authors': authors,
                    'url': paper['url'],
                    'score': 100 * scores[idx]
                }
            )
        return results
    
    def get_total_papers(self, x=3):
        """
        Get approximate number of papers in database
        
        Args:
            x: Position to round to (i.e round to the nearest 10 ** x place)
        Returns:
            int: Number of papers in DB, rounded to some threshold
        """
        num_papers = len(self.paper_metadata)
        thresh = 10 ** x
        # Round to nearest thousand
        return int(thresh * (num_papers // thresh))
        
