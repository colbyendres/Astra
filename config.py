import os 
from dotenv import load_dotenv


class Config:
    load_dotenv()
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DB_URI = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
    EMBEDDINGS_INDEX = os.environ.get('EMBEDDINGS_INDEX')
    EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID')
    EMBEDDING_URL = os.environ.get('EMBEDDING_URL')
    EMBEDDING_KEY = os.environ.get('EMBEDDING_KEY')
    SESSION_KEY = os.environ.get('SESSION_KEY')
    SQLALCHEMY_TRACK_MODIFICATIONS = False 
    