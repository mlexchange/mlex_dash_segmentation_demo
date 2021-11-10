FROM python:3.7.10
MAINTAINER MLEXCHANGE TEAM

#COPY requirements.txt requirements.txt
# if run with docker context in main folder
COPY docker/requirements.txt requirements.txt 

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    python3-pip\
    ffmpeg\
    libsm6\
    libxext6 

RUN pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt &&\
    pip3 install jupyter 

WORKDIR /app/notebook
ENV HOME /app/notebook
ENV PYTHONUNBUFFERED=1

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
#ENV TINI_VERSION v0.6.0
#ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
#RUN chmod +x /usr/bin/tini
#ENTRYPOINT ["/usr/bin/tini", "--"]


COPY model_validation.py model_validation.py
COPY kmeans.py kmeans.py
COPY segment.py segment.py
CMD ["echo", "running kmeans"]
