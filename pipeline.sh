# Build Docker Container
docker build --network=host -t dvml_gruppe_docker -f Dockerfile .

# End of pipeline
echo "\nEnd of pipeline"
