import logging
from kafka import KafkaConsumer
#from kq import Worker
from kf_worker import Worker

def job_def(cmd):
    logs = subprocess.run(cmd, text=True, check=True, stdout=subprocess.PIPE)
    print(logs)

# Set up logging.
formatter = logging.Formatter('[%(levelname)s] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger = logging.getLogger('kq.worker')
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

# Set up a Kafka consumer.
consumer = KafkaConsumer(
    bootstrap_servers='kafka:9092',
    group_id='group',
    auto_offset_reset='latest',
    api_version=(0, 9)
)

# Set up a worker.
worker = Worker(topic='seg-demo', consumer=consumer)
worker.start(max_messages=1000)

