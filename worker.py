import redis
from rq import Worker, Queue
from config import Config
from urllib.parse import urlparse 

def get_job_queue() -> Queue:
    if not Config.REDIS_URL:
        raise RuntimeError('Missing config variable REDIS_URL, set this in your .env file')

    url = urlparse(Config.REDIS_URL)
    conn = redis.Redis(
        host=url.hostname,
        port=url.port,
        password=url.password,
        ssl=(url.scheme == 'rediss'),
        ssl_cert_reqs=None
    )
    job_queue = Queue(name='job_queue', connection=conn)
    return job_queue