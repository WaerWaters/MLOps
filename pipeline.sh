#!/bin/bash

# Build Docker Container
docker build -t pytorch-container -f Dockerfile .



# End of pipeline
echo "\nEnd of pipeline"
