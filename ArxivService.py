from arxiv import Search
import re

class ArxivService:
    """
    Wrapper class for querying arXiv
    """
    IS_ARXIV_ID = re.compile(r'[0-9]{4}\.[0-9]{5}')
     
    @staticmethod
    def is_valid_arxiv_id(arxiv_id: str):
        """
        Validates arXiv ID
        
        Args:
            arxiv_id: Presumed arxiv_id
            
        Returns:
            bool
        """
        return bool(ArxivService.IS_ARXIV_ID.search(arxiv_id))
    
    @staticmethod
    def get_paper_by_id(arxiv_id: str):
        """
        Retrieve paper from arXiv by ID
        
        Args:
            arxiv_id: arXiv ID of paper to retrieve
            
        Returns:
            paper (dict): Relevant attributes of paper 
        """
        if not ArxivService.is_valid_arxiv_id(arxiv_id):
            raise TypeError('An invalid arXiv ID was supplied')
        arxiv_res = Search(id_list=[arxiv_id], max_results=1)
        data = next(arxiv_res.results())
        authors = [str(a) for a in data.authors]
        return {
            'arxiv_id': arxiv_id,
            'title': data.title,
            'authors': authors,
            'url': data.entry_id,
            'abstract': data.summary
        }
        
    @staticmethod
    def get_paper_by_title(title: str):
        """
        Retrieve paper from arXiv by title
        
        Args:
            title: title of paper to retrieve
            
        Returns:
            paper (dict): Relevant attributes of paper 
        """
        arxiv_res = Search(query=title, max_results=1)
        data = next(arxiv_res.results())
        authors = [str(a) for a in data.authors]
        # The API doesn't directly expose the arXiv ID, but we can snag it from the url
        arxiv_id = ArxivService.IS_ARXIV_ID.search(data.entry_id).group(0)
        return {
            'arxiv_id': arxiv_id,
            'title': data.title,
            'authors': authors,
            'url': data.entry_id,
            'abstract': data.summary
        }