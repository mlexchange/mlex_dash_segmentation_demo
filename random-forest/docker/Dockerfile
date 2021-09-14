FROM python:3.7.10
MAINTAINER TANNY CHAVEZ

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
    pip3 install -r requirements.txt 

WORKDIR /app/work
ENV HOME /app/work
ENV PYTHONUNBUFFERED=1

COPY feature_generation.py feature_generation.py
COPY random_forest.py random_forest.py
COPY segment.py segment.py
CMD ["echo", "running random forest"]
