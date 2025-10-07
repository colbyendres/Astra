import redis 

from urllib.parse import urlparse
from rq import Queue 
from config import Config

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