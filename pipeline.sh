#!/bin/bash
set -e

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
docker run --gpus all -e BUILD_NUMBER=${BUILD_NUMBER} ${IMAGE_NAME}:${COMMIT_HASH}

# Merge dev to main on successful pipeline run
git remote set-url origin "https://${GIT_USER}:${GIT_PASS}@github.com/WaerWaters/MLOps.git"
git checkout main
git merge dev --ff-only
git push origin main
git checkout dev

echo "\nEnd of pipeline"
