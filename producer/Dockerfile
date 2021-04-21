FROM python:3.7

RUN pip install pika

ENV PYTHONBUFFERED=1

WORKDIR /work/app/

COPY producer.py .

CMD ["sleep", "1d"]
