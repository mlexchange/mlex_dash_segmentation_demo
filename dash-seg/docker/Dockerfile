FROM python:3.9
MAINTAINER TANNY CHAVEZ

RUN ls
COPY docker/requirements.txt requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    python3-pip\
    ffmpeg\
    libsm6\
    libxext6 

RUN pip3 install --upgrade pip &&\
    pip3 install --timeout=2000 -r requirements.txt 

#CMD ["bash","call_defect.sh"]
EXPOSE 8000

WORKDIR /app/work
ENV HOME /app/work
COPY src src
ENV PYTHONUNBUFFERED=1


CMD ["bash"]
