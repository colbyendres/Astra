from rq import Worker
from job_queue import job_queue, conn 

if __name__ == '__main__':
    worker = Worker(queues=job_queue, name='queue_worker', connection=conn)
    worker.work()
