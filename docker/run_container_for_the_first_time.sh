#!/bin/bash

# Remove the existing container if there it is
docker rm -f bundlesdf

# プロジェクトディレクトリ
DIR=$(pwd)/../
echo "DIR to be mounted: $DIR"

# Do not use the following command since it may too strong and unsafe to be used on the shared server
# xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name bundlesdf  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v /home:/home -v /tmp:/tmp -v /mnt:/mnt -v $DIR:$DIR  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE nvcr.io/nvidian/bundlesdf:latest bash

docker run --name bundlesdf \
  --interactive \
  --tty \
  --network=host \
  --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --env GIT_INDEX_FILE \
  --env DISPLAY=$DISPLAY \
  --env XAUTHORITY=/root/.Xauthority \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --volume /tmp:/tmp \
  --volume /mnt:/mnt \
  --volume $DIR:$DIR \
  --volume ~/.Xauthority:/root/.Xauthority:rw \
  nvcr.io/nvidian/bundlesdf:latest \
  bash
