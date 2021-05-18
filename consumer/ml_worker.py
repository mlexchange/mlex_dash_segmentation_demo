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
    result =channel.queue_declare(queue='ml_tasks', durable=True)

    def callback(ch, method, properties, body):
        """pika callback function-- run when message is recieved
        """
        print("Recieved message: {}".format(body))
        payload = json.loads(body)
        #logs = subprocess.run(sub_commands, text=True, check=True)
        #logs = subprocess.run(['docker', 'run','-v', '{}:/data'.format(DATA_DIR), payload['docker_uri'], payload['docker_cmd'], *payload['kw_args'].split() ], text=True, check=True, stdout=subprocess.PIPE)
        try:
            logs = subprocess.run(['docker', 'run','-v', '{}:/data'.format(DATA_DIR), payload['docker_uri'], *payload['docker_cmd'].split(), *payload['kw_args'].split() ], text=True, check=True, stdout=subprocess.PIPE)
            print(logs.stdout)
        except:
            print('task completely failed')
            print(logs.stdout)

        #now send message back
        ch.basic_publish(exchange='',
                routing_key = properties.reply_to,
                properties = pika.BasicProperties(correlation_id = properties.correlation_id),
                body=str(logs.stdout),
                )
        print('send logs back')

    channel.basic_consume(queue="ml_tasks", on_message_callback=callback)
    print("Worker up. Waiting for tasks...")
    channel.start_consuming()
    return (channel, connection)


if __name__ == '__main__':
    
    AMQP_URL = os.environ['AMQP_URL']
    DATA_DIR = os.environ['DATA_DIR']
    params = pika.URLParameters(AMQP_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    result =channel.queue_declare(queue='ml_tasks', durable=True)

    def callback(ch, method, properties, body):
        """pika callback function-- run when message is recieved
        """
        print("Recieved message: {}".format(body))
        payload = json.loads(body)
        #logs = subprocess.run(sub_commands, text=True, check=True)
        #logs = subprocess.run(['docker', 'run','-v', '{}:/data'.format(DATA_DIR), payload['docker_uri'], payload['docker_cmd'], *payload['kw_args'].split() ], text=True, check=True, stdout=subprocess.PIPE)
        cmds = ''
        if payload['gpu'] == True:
            cmds = '--gpus all'
        else:
            cmds = ''
        try:
            logs = subprocess.run(['docker', 'run', *cmds.split(), '-v', '{}:/data'.format(DATA_DIR), payload['docker_uri'], *payload['docker_cmd'].split(), *payload['kw_args'].split() ], text=True, check=True, stdout=subprocess.PIPE)
            print(logs.stdout)
            print(logs)
            ch.basic_ack(delivery_tag = method.delivery_tag)

        #now send message back
            payload_back = {
                    'job_type': payload['job_type'],
                    'job_status':'complete', 
                    'logs': str(logs.stdout)}
            payload_back = json.dumps(payload)
            ch.basic_publish(exchange='',
                    routing_key = properties.reply_to,
                    properties = pika.BasicProperties(correlation_id = properties.correlation_id),
                    body=payload_back,
                    )
            print('send logs back')
        except Exception as e:
            print('task failed, check logs')
            print(e)
            print(str(e.stdout))
            payload = {'error':e,
                    'log':str(e.stdout)
                    }
            ch.basic_ack(delivery_tag = method.delivery_tag)
            ch.basic_publish(exchange='',
                    routing_key = properties.reply_to,
                    properties = pika.BasicProperties(correlation_id = properties.correlation_id),
                    body=payload,
                    ) #need to return something, or the dash app will hang and throw an error when the callback doesn't finish
    channel.basic_consume(queue="ml_tasks", on_message_callback=callback)
    print("Worker up. Waiting for tasks...")
    try:
        channel.start_consuming()


    except KeyboardInterrupt:
        print("Interrupt!")
        channel.stop_consuming()
        connection.close()

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


    

