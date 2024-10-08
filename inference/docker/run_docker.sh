#!/bin/bash

# Define the container name
NAME=biodiversity_docker

# Define the absolute path to the main directory
MAIN_DIR=$(realpath "`dirname $0`/../")

# Set the network configuration
NETWORK="host"

# Set default command and working directory
CMD="/bin/bash"
WORKDIR="/home/work"

# Check if the 'start-service' command is provided
if [ "$1" == "start-service" ]; then
    CMD="python3 microservice.py --device cpu"
    WORKDIR="/home/work/infer"  # Set the working directory to the 'infer' folder
fi

# Run Docker command
docker run -it --rm --gpus all \
    -e PLATFORM=$ENV_PLATFORM \
    -p 5000:5000 \
    --network $NETWORK \
    -v $MAIN_DIR:/home/work \
    --shm-size 16G \
    --workdir $WORKDIR \
    --name $NAME \
    $NAME \
    $CMD
