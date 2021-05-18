#job_dispatcher.py

from abc import ABCMeta, abstractmethod


class JobInterface:
    __metaclass__ = ABCMeta

    def __init__(self, job_description: dict, container_image_uri: str, code_to_execute_uri: str):
        self.job_location = job_location
        self.container_image_uri = container_image_uri
        self.code_to_executre_uri = code_to_execture_uri
        super().__init__()

    @abstactmethod
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
    
        
