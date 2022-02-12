#!/bin/bash
set -Eeuo pipefail

IMAGE_NAME="neural_networks_memorization"

#CURRENT_GIT_HASH=`git rev-parse --short HEAD`
buildId=${1:-latest}


CURRENT_SCRIPT_DIR=$(readlink -f "$(dirname $0)")
BASE_DIR=$(readlink -f "${CURRENT_SCRIPT_DIR}/..")
CLOUD_COMPUTE_DIR="${BASE_DIR}/cloud_compute_utils"

cd $BASE_DIR


bash $BASE_DIR/standard/build_whls.sh

echo "========="
echo $BASE_DIR
echo $CLOUD_COMPUTE_DIR
echo "========="

printf "\n== Building Neural Nets Memorization container ==\n\n"

DOCKER_BUILDKIT=1 docker build --progress=plain \
                            -t $IMAGE_NAME:$buildId -t $IMAGE_NAME:latest \
                            -f $BASE_DIR/Dockerfile $BASE_DIR

#docker tag $IMAGE_NAME:$buildId $IMAGE_NAME:latest
