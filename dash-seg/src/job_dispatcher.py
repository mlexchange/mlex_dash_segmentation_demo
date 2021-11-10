import json
import helper_utils
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


class SimpleJob:
   """
   Create a job and then deploy it as part of a simple docker-compose service
   """

   def __init__(self,
                user,
                job_type,
                description,
                deploy_location,
                gpu,
                data_uri,
                container_uri,
                container_cmd,
                container_kwargs,
                mlex_app = 'seg-demo',
                ):
       self.user = user
       self.mlex_app = mlex_app
       self.job_type = job_type
       self.description = description
       self.deploy_location = deploy_location
       self.gpu = gpu
       self.data_uri = data_uri
       self.container_uri = container_uri
       self.container_cmd = container_cmd
       self.container_kwargs = container_kwargs

   def launch_job(self):
       """
       Send the job to a simple kafka queue and update job status in mongodb database collection
       """
       return helper_utils.post_job(self)
