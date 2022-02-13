#!/bin/bash
set -Eeuo pipefail

IMAGE_NAME="neural_networks_training"

#CURRENT_GIT_HASH=`git rev-parse --short HEAD`
buildId=${1:-latest}


CURRENT_SCRIPT_DIR=$(readlink -f "$(dirname $0)")
BASE_DIR=$(readlink -f "${CURRENT_SCRIPT_DIR}/..")

cd $BASE_DIR

echo "========="
echo $BASE_DIR
echo $CLOUD_COMPUTE_DIR
echo "========="

printf "\n== Building Neural Nets Training container ==\n\n"

DOCKER_BUILDKIT=1 docker build --progress=plain \
                            -t $IMAGE_NAME:$buildId -t $IMAGE_NAME:latest \
                            -f $BASE_DIR/Dockerfile $BASE_DIR

#docker tag $IMAGE_NAME:$buildId $IMAGE_NAME:latest
