# syntax=docker/dockerfile:1.0.0-experimental

FROM pytorch/pytorch as base 

SHELL ["/bin/bash", "-c"]

ADD ./wheels/  /app/code/wheels
ADD ./standard/  /app/code/standard
#COPY ./wheels/ ./standard/ /app/code/
WORKDIR /app/code/

RUN  /app/code/standard/install_dependencies.sh \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install --upgrade --no-input virtualenv poetry

RUN python3 -m pip install /app/code/wheels/*whl

ENTRYPOINT ["/app/code/standard/entrypoint.sh"]
