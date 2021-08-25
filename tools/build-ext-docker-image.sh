#!/usr/bin/env bash

# This script builds a Docker image that contains LPM's third-party libraries
# in /opt/lpm, on top of a recent Ubuntu image. Run it like so:
#
# `./build-ext-docker-image.sh <build-type> <precision>`
#
# The arguments are:
# <build-type> - Debug or Release (determines build optimization level)
# <precision> - single or double (floating point precision)
#
# For this script to work, Docker must be installed on your machine.
BUILD_TYPE=$1
PRECISION=$2

if [[ "$1" == "" || "$2" == "" ]]; then
  echo "Usage: $0 <build-type> <precision>"
  exit
fi

TAG=$BUILD_TYPE-$PRECISION

# Build the image locally.
cp Dockerfile.ext ../Dockerfile
docker build -t lpm-tpl:$TAG --network=host ..
rm ../Dockerfile

# Tag the image.
docker image tag lpm-tpl:$TAG pbosler/lpm-tpl:$TAG

echo "To upload this image to DockerHub, use the following:"
echo "docker login"
echo "docker image push pbosler/lpm-tpl:$TAG"
echo "docker logout"
