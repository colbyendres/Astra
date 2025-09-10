from flask_sqlalchemy import SQLAlchemy
from titlecase import titlecase
from ast import literal_eval

db = SQLAlchemy()

class Paper(db.Model):
    """
    Paper ORM class
    """
    __tablename__ = "papers"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text)
    url = db.Column(db.Text)
    arxiv_id = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f"<Paper {self.arxiv_id}: {self.title[:40]}>"
    
    def to_dict(self, score=0):
        """
        Return paper as a dictionary
        """
        author_list = literal_eval(self.authors)
        if len(author_list) > 2:
            authors = ', '.join(author_list[:2]).title() + ' et al.'
        else:
            authors = ', '.join(author_list).title()
        return {
            'arxiv_id': self.arxiv_id,
            'title': titlecase(self.title),
            'authors': authors,
            'url': self.url,
            'score': score
        }
