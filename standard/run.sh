#!/bin/bash
IMAGE_NAME="neural_networks_training"
CONTAINER_NAME="neural_networks_training_container"
WSDIR="${1:-./${CONTAINER_NAME}-ws}"
BUILD_ID=${2:-'latest'}

# Make sure it is exact path
if [[ "$WSDIR" != /* ]]; then
    WSDIR=`pwd`/$WSDIR
fi

if [[ ! -d "${WSDIR}" ]]; then
    echo "Creating workspace at \`${wsDir}\`"
    mkdir -p "${wsDir}"
fi

docker run --mount type=bind,source=$WSDIR,target=/app/home/ \
    --name $CONTAINER_NAME \
    $IMAGE_NAME:$BUILD_ID
