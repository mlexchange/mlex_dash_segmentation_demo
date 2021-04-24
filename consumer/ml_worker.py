import pika
import subprocess
import json
import os
import sys

def main():
    AMQP_URL = os.environ['AMQP_URL']
    DATA_DIR = os.environ['DATA_DIR']
    params = pika.URLParameters(AMQP_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    def callback(ch, method, properties, body):
        """pika callback function-- run when message is recieved
        """
        print("Recieved message: {}".format(body))
        payload = json.loads(body)
        #logs = subprocess.run(sub_commands, text=True, check=True)
        #logs = subprocess.run(['docker', 'run','-v', '{}:/data'.format(DATA_DIR), payload['docker_uri'], payload['docker_cmd'], *payload['kw_args'].split() ], text=True, check=True, stdout=subprocess.PIPE)
        logs = subprocess.run(['docker', 'run','-v', '{}:/data'.format(DATA_DIR), payload['docker_uri'], *payload['docker_cmd'].split(), *payload['kw_args'].split() ], text=True, check=True, stdout=subprocess.PIPE)
        print(logs.stdout)

        #now send message back
        ch.basic_publish(exchange='',
                routing_key = properties.reply_to,
                properties = pika.BasicProperties(correlation_id = properties.correlation_id),
                body=str(logs.stdout),
                )
        print('send logs back')

    channel.basic_consume(queue="ml_tasks", on_message_callback=callback, auto_ack=True)
    print("Worker up. Waiting for tasks...")
    channel.start_consuming()


if __name__ == '__main__':
    
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupt!")

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


    

