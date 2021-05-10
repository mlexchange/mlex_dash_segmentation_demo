from ubuntu:20.04
MAINTAINER ADAM GREEN


RUN apt-get update && apt-get install -y curl\
    ca-certificates\
    amqp-tools\
    dnsutils\
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
RUN  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

RUN echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io

RUN apt-get install -y python3-pip\
    python3
RUN curl -s https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.deb.sh | bash &&\
    apt-get install -y rabbitmq-server
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt 
WORKDIR /app
COPY ml_worker.sh ml_worker.sh
COPY ml_worker.py ml_worker.py
ENV PYTHONUNBUFFERED=1
#CMD ["sleep", "1d"]

CMD python3 ml_worker.py
#CMD ["bash", "ml_worker.sh"]
