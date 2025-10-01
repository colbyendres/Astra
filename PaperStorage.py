import os 
from faiss import read_index, write_index 
from boto3 import Session 
from config import Config

S3_FAISS_PATH = 'data/paper_index.faiss'
TMP_FAISS_PATH = '/tmp/paper_index.faiss'
    
session = Session(
    aws_access_key_id=Config.BUCKETEER_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=Config.BUCKETEER_AWS_SECRET_ACCESS_KEY,
    region_name=Config.BUCKETEER_AWS_REGION
)
client = session.client('s3')
bucket_name = Config.BUCKETEER_BUCKET_NAME
                
def restore_from_bucket():
    """
    Construct FAISS index from S3 bucket
    """
    if not os.path.exists(TMP_FAISS_PATH):
        client.download_file(bucket_name, S3_FAISS_PATH, TMP_FAISS_PATH)
    return read_index(TMP_FAISS_PATH)

def write_to_bucket(new_index):
    """
    Write new FAISS index back to bucket to maintain consistency
    """
    write_index(new_index, TMP_FAISS_PATH)
    client.upload_file(TMP_FAISS_PATH, bucket_name, S3_FAISS_PATH)