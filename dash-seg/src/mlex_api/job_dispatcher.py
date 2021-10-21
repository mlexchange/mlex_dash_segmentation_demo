import json
import uuid
import requests
import threading, logging, time
import subprocess
import os
import sys
import kq
from kafka import KafkaProducer
from kq import Queue
from kq.job import Job

class simpleJob():
    """ Simple job made for segmentation demo.
    Create a job and then deploy it as part of a simple docker-compose service
    """

    def __init__(self,
                 job_description,
                 job_type,
                 deploy_location,
                 docker_uri,
                 docker_cmd,
                 kw_args,
                 # job_queue,
                 db_collection,
                 GPU=False #,
                 # job_id,  # if no id, create
                 ):
        self.job_description = job_description
        self.job_type = job_type
        self.deploy_location = deploy_location
        self.docker_uri = docker_uri
        self.docker_cmd = docker_cmd
        self.kw_args = kw_args
        self.gpu = GPU
        # self.queue = job_queue
        self.collection = db_collection
        self.response = None
        self.job_id = str(uuid.uuid4())

        # create json payload
        payload = {'job_id': self.job_id,
                   'docker_uri': self.docker_uri,
                   'job_type': self.job_type,
                   'docker_cmd': self.docker_cmd,
                   'gpu': self.gpu,
                   'kw_args': self.kw_args,
                   }
        self.js_payload = json.dumps(payload)

    def launchJob(self):
        """
        Send the job to a simple kafka queue and update job status in mongodb database collection
        """
        DATA_DIR = os.environ['DATA_DIR']
        ### Initialize Connection to Work Queue
        TOPIC = 'seg-demo'
        producer = KafkaProducer(bootstrap_servers='kafka:9092', api_version=(0,9))
        job_queue = Queue(topic=TOPIC, producer=producer)
        job_collection = self.collection
        payload = json.loads(self.js_payload)
        cmds = ''
        s = " "
        cmd = s.join(['docker', 'run', *cmds.split(), '-v', '{}:/data'.format(DATA_DIR), payload['docker_uri'],
               *payload['docker_cmd'].split(), *payload['kw_args'].split() ])
        dict = {'text':True, 'check':True, 'stdout':subprocess.PIPE, 'shell':True}
        job = Job(id = self.job_id, func = subprocess.run, args = [cmd], kwargs = dict)
        job_queue.enqueue(job)
        job_collection.insert_one({"job_id": str(self.job_id),
                                   "job_type": str(self.job_type),
                                   "status": "sent to queue"})
        print('send job to queue')

