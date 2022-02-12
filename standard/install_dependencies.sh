#!/bin/bash
set -euxo pipefail


apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-wheel \
    build-essential \
    python3-setuptools \
    python3-venv 
    #curl \
    #netcat \
    #rsync

