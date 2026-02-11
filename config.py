import os 
import logging
from dotenv import load_dotenv

class Config:
    load_dotenv()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DB_URI = os.environ.get('DATABASE_URL')
    if DB_URI and 'postgres://' in DB_URI:
        logging.debug('Using remote Postgres database')
        DB_URI = DB_URI.replace('postgres://', 'postgresql://')
    elif DB_URI and 'sqlite:///' in DB_URI:
        # Assume that the local db is at ROOT_DIR/data/db_name.db
        # Dynamically generate absolute path, since we can't hardcode that for Heroku
        db_name = DB_URI.strip('sqlite:///')
        full_path = os.path.join(os.getcwd(), 'data', db_name)
        DB_URI = f'sqlite:///{full_path}'
        logging.debug('Using local SQLite database at abs path: %s', full_path)
        
    EMBEDDINGS_INDEX = os.environ.get('EMBEDDINGS_INDEX')
    EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID')
    EMBEDDING_URL = os.environ.get('EMBEDDING_URL')
    EMBEDDING_KEY = os.environ.get('EMBEDDING_KEY')
    SESSION_KEY = os.environ.get('SESSION_KEY')
    BUCKETEER_AWS_ACCESS_KEY_ID = os.environ.get('BUCKETEER_AWS_ACCESS_KEY_ID')
    BUCKETEER_AWS_REGION = os.environ.get('BUCKETEER_AWS_REGION')
    BUCKETEER_BUCKET_NAME = os.environ.get('BUCKETEER_BUCKET_NAME')
    BUCKETEER_AWS_SECRET_ACCESS_KEY = os.environ.get('BUCKETEER_AWS_SECRET_ACCESS_KEY')
    SQLALCHEMY_TRACK_MODIFICATIONS = False 
    LOCAL_FAISS_PATH = '/tmp/paper_index.faiss'
    S3_FAISS_PATH = 'data/paper_index.faiss'

    