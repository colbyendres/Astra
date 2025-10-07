import os
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
        client.download_file(Config.BUCKETEER_BUCKET_NAME, Config.S3_FAISS_PATH, Config.LOCAL_FAISS_PATH)
        print('FAISS file downloaded from S3 bucket')

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
    client.upload_file(Config.LOCAL_FAISS_PATH, Config.BUCKETEER_BUCKET_NAME, Config.S3_FAISS_PATH)
    print('FAISS file uploaded to S3 bucket')

    
def s3_failure(job, connection, type, value, traceback):
    """
    Callback for if S3 pull fails
    """
    raise RuntimeError('Failed to pull FAISS index in from S3')