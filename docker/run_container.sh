#!/bin/bash

# コンテナ名を指定
CONTAINER_NAME="prebuild-bundlesdf-blackwell"
IMAGE_NAME="bundlesdf-blackwell"

# マウント用のディレクトリ設定
DIR=$(pwd)/../

# 画面描画の許可 (共通して必要)
xhost +

# コンテナが存在するか確認
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "--------------------------------------------------"
    echo "Existing container '${CONTAINER_NAME}' was found."
    echo "Resume it..."
    echo "--------------------------------------------------"
    
    # 既存のコンテナをインタラクティブモードで再開
    docker start -ai ${CONTAINER_NAME}

else
    echo "--------------------------------------------------"
    echo "Containers named '${CONTAINER_NAME}' was not found."
    echo "Make and run a container withe the name."
    echo "--------------------------------------------------"

    # 新規作成コマンド (--name を変更し、rm は除去)
    docker run \
        --name ${CONTAINER_NAME} \
        --gpus all \
        --env NVIDIA_DISABLE_REQUIRE=1 \
        --interactive \
        --tty \
        --network=host \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --volume /home:/home \
        --volume /tmp:/tmp \
        --volume /mnt:/mnt \
        --volume $DIR:$DIR \
        --env DISPLAY=${DISPLAY} \
        --env GIT_INDEX_FILE \
        ${IMAGE_NAME} bash
fi