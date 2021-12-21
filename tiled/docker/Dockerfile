FROM ghcr.io/bluesky/tiled:main as base

COPY docker/requirements.txt docker/entrypoint.sh docker/tree.py /deploy/
COPY docker/gunicorn_config.py /app/docker/
COPY docker/config.yml /deploy/config/
ENV TILED_CONFIG=/deploy/config
RUN pip install -r requirements.txt
