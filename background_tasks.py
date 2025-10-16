import os
import logging 

from faiss import write_index
from boto3 import Session
from config import Config

def cache_faiss_index():
    """
    Pull in FAISS index from S3 bucket into /tmp
    """
    if not os.path.exists(Config.LOCAL_FAISS_PATH):
        session = Session(
            aws_access_key_id=Config.BUCKETEER_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.BUCKETEER_AWS_SECRET_ACCESS_KEY,   
            region_name=Config.BUCKETEER_AWS_REGION
        )
        client = session.client('s3')
        try:
            client.download_file(Config.BUCKETEER_BUCKET_NAME, Config.S3_FAISS_PATH, Config.LOCAL_FAISS_PATH)
            logging.info('FAISS file downloaded from S3 bucket')
        except Exception as e:
            logging.error(f'Failed to download file from {Config.S3_FAISS_PATH} in S3 bucket')

def write_to_bucket(new_index):
    """
    Write new FAISS index back to bucket to maintain consistency
    """
    session = Session(
        aws_access_key_id=Config.BUCKETEER_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.BUCKETEER_AWS_SECRET_ACCESS_KEY,   
        region_name=Config.BUCKETEER_AWS_REGION
    )
    client = session.client('s3')
    write_index(new_index, Config.LOCAL_FAISS_PATH)
    try:
        client.upload_file(Config.LOCAL_FAISS_PATH, Config.BUCKETEER_BUCKET_NAME, Config.S3_FAISS_PATH)
        logging.info('FAISS file uploaded to S3 bucket')
    except Exception as e:
        logging.warning(f'Failed to upload index to S3 bucket')