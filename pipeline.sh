#!/bin/bash

REGISTRY="172.24.198.42:5000"
IMAGE_NAME="dvml_gruppe1"
COMMIT_HASH=$(git rev-parse --short HEAD)

# Build Docker Container (tests run during build)
docker build --network=host --build-arg COMMIT_HASH=${COMMIT_HASH} -t ${IMAGE_NAME}:${COMMIT_HASH} -f Dockerfile .

# Tag for registry
docker tag ${IMAGE_NAME}:${COMMIT_HASH} ${REGISTRY}/${IMAGE_NAME}:${COMMIT_HASH}
docker tag ${IMAGE_NAME}:${COMMIT_HASH} ${REGISTRY}/${IMAGE_NAME}:latest

# Push to registry
docker push ${REGISTRY}/${IMAGE_NAME}:${COMMIT_HASH}
docker push ${REGISTRY}/${IMAGE_NAME}:latest

# Run Docker Container
docker run --gpus all ${IMAGE_NAME}:${COMMIT_HASH}

echo "\nEnd of pipeline"
