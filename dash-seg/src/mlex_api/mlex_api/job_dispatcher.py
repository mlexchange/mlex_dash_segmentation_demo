#job_dispatcher.py

from abc import ABCMeta, abstractmethod
import json
import pika
import uuid
import hashlib


class JobInterface:
    __metaclass__ = ABCMeta

    def __init__(self, job_description: dict, container_image_uri: str, code_to_execute_uri: str):
        self.job_location = job_location
        self.container_image_uri = container_image_uri
        self.code_to_executre_uri = code_to_execture_uri
        super().__init__()

    @abstractmethod
    def createJob():
        """
        Create a Job object (assuming that we are within a kubernetes environment, Job is a k-Job)
        TODO: if we build up a good job description, we could choose the appropriate container
        for the user
        
        Args:
            container_image_uri, str: path to docker/singularity image on registry
            code_to_execture_uri, str: path to github/bitbucket for the code you want to execute
            job_template, Path: path to job_template.yaml file which defines default for Job
        
        Return:
            Kubernetes Job Object
        """
        pass

    @abstractmethod
    def createJobResources():
        """
        Create/assign the compute resources needed to complete the job.
        For example, if on a local kluster, create a Job object to run the code.
        Or, if on a local kluster, create a dask cluster to run parallel jobs

        Args:
            job_description, dict:
                            { location: nersc, local, gcp
                              parallel_dask: True/False
                              }

        Returns:
            tuple(int, json): (1 if successful 0 if not, json containing any logs from creation)

        """
        pass

    @abstractmethod
    def putJob(job, job_resources):
        """
        Launch created Job on provisioned resources

        Args:
            job: Kubernetes Job Object
            job_resources: server/local connection to computing resources
        """
        pass
    
class workQueue():
    def __init__(self, connection_url):
        self.url = connection_url

        ### assume amqp for now
        ### establish connection to be used for the rest of
        ### the session
        self.params = pika.URLParameters(self.url)
        self.connection = pika.BlockingConnection(self.params)
        self.channel = self.connection.channel()

        ### setup work queue and results queue
        self.ml_queue_name = 'ml_tasks'
        self.channel.queue_declare(
                queue=self.ml_queue_name, durable =True)
        self.results_queue_name = 'results'
        results = self.channel.queue_declare(
                queue=self.results_queue_name, durable=True)
        self.results_queue = results.method.queue
        # register callback for basic consume
    
    def _on_response(self, ch, method, props, body):
        print('calling')
        print(method.delivery_tag)
        print(props.correlation_id)
        print(body)
        channel.basic_ack(delivery_tag=method.delivery_tag)
        return body
    def get_logs(self):
        method_frame, properties, body = self.channel.basic_get(self.results_queue, auto_ack = True) 
        # result will tuple (pika.spec.Basic.GetOK(), pika.spec.Basic.Properties, message body)
        if body is not None:
           print('results recieved')
           print(body)
           print(properties)
           print(properties.correlation_id)
           return (method_frame, properties, body)
        else:
            print('channel empty')
            return None
    def put_job(self, payload, job_id):
        self.channel.basic_publish(
                exchange='',
                routing_key=self.ml_queue_name,
                body=payload,
                properties=pika.BasicProperties(
                    reply_to = self.results_queue,
                    correlation_id=job_id,
                    )
                )


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
            work_queue,
            GPU = False,
            corr_id = str(uuid.uuid4()), # if no id, create
            ):
        self.job_description = job_description
        self.job_type = job_type
        self.deploy_location = deploy_location
        self.docker_uri = docker_uri
        self.docker_cmd = docker_cmd
        self.kw_args = kw_args
        self.gpu = GPU
        self.queue = work_queue
        self.response = None
        self.corr_id = corr_id 

        # create json payload
        payload = {'docker_uri':self.docker_uri,
                'job_type': self.job_type,
                'docker_cmd':self.docker_cmd,
                'gpu': self.gpu,
                'kw_args':self.kw_args,
                }
        self.js_payload = json.dumps(payload)
        #params = pika.URLParameters(amqp_url)
        #self.connection = pika.BlockingConnection(params)
        #self.channel = self.connection.channel()
        #result =self.channel.queue_declare(queue='results', durable=True)
        #self.channel.queue_declare(queue='ml_tasks', durable=True)
        #self.callback_queue = result.method.queue 
        #self.channel.basic_consume(
        #        queue=self.callback_queue,
        #        on_message_callback=self._on_response,
        #        )


    def launchJob(self):
        """
        Send the job to a simple amqp message queue
        wait for response
        """
        self.queue.put_job(self.js_payload, self.corr_id)
        print('send job to queue')

    def _on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def monitorJob(self):
        """
        Blocking connection, return when job is finished
        """
        while self.response is None:
            self.connection.process_data_events()
        self.connection.close()
        return self.response
        pass
