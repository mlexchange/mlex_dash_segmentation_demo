import pika
import os

EXCHANGE = 'exchange'

ROUTING_KEY = 'exchange.ml_tasks'

DELAY = 5

def main():
    ampq_url = os.environ['AMQP_URL']
    print("rabbit url: {}".format(ampq_url))


    #get parameters from connection:
    params = pika.URLParameters(ampq_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue='ml_tasks', durable=True)

    message="Hello World"
    channel.basic_publish(
            exchange='',
            routing_key='ml_tasks',
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2,

            ))
    print("sent message: {}".format(message)) 
    connection.close()

if __name__=='__main__':
    main()
