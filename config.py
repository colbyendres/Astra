import os 
from dotenv import load_dotenv

class Config:
    load_dotenv()
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DB_URI = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
    EMBEDDINGS_INDEX = os.environ.get('EMBEDDINGS_INDEX')
    SQLALCHEMY_TRACK_MODIFICATIONS = False 
    