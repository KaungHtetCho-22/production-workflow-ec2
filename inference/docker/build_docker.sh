#!/bin/bash
NAME=biodiversity_docker
cd "`dirname $0`"

docker build . -t $NAME

