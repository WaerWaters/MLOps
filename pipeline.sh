#!/bin/bash

# Build Docker Container
docker build -t pytorch-container .

# Run Tests
sh tests/check_cuda.sh

# End of pipeline
echo "\nEnd of pipeline"
