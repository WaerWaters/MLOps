# Build Docker Container
docker build --network=host -t dvml_gruppe_docker -f Dockerfile .

# Run Docker Container
docker run --gpus all dvml_gruppe_docker

# End of pipeline
echo "\nEnd of pipeline"
