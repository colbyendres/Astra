import os
import logging

from faiss import write_index
from boto3 import Session
from botocore.exceptions import ClientError
from config import Config


def start_s3_client():
    """
    Create client that connects to S3 bucket
    """
    if not Config.BUCKETEER_AWS_ACCESS_KEY_ID or not Config.BUCKETEER_AWS_SECRET_ACCESS_KEY:
        logging.error(
            'Missing Bucketeer credentials: ensure BUCKETEER_AWS_ACCESS_KEY_ID and BUCKETEER_AWS_SECRET_ACCESS_KEY env vars are set')
        return None
    session = Session(
        aws_access_key_id=Config.BUCKETEER_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=Config.BUCKETEER_AWS_SECRET_ACCESS_KEY,
        region_name=Config.BUCKETEER_AWS_REGION
    )
    return session.client('s3')


def cache_faiss_index():
    """
    Pull in FAISS index from S3 bucket into /tmp
    """
    if os.path.exists(Config.LOCAL_FAISS_PATH):
        return
    client = start_s3_client()
    try:
        client.download_file(Config.BUCKETEER_BUCKET_NAME,
                             Config.S3_FAISS_PATH, Config.LOCAL_FAISS_PATH)
        logging.info('FAISS file downloaded from S3 bucket')
    except Exception as e:
        logging.error(
            'Failed to download file from %s in S3 bucket', Config.S3_FAISS_PATH)
        raise e


def write_to_bucket(new_index):
    """
    Write new FAISS index back to bucket to maintain consistency
    """
    client = start_s3_client()
    write_index(new_index, Config.LOCAL_FAISS_PATH)
    try:
        client.upload_file(Config.LOCAL_FAISS_PATH,
                           Config.BUCKETEER_BUCKET_NAME, Config.S3_FAISS_PATH)
        logging.info('FAISS file uploaded to S3 bucket')
    except ClientError as e:
        logging.warning('S3 write back failed, msg: %s', e)
