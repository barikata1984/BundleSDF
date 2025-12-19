#!/bin/bash

# スクリプトの場所に関わらず、dockerディレクトリをカレントにする
cd "$(dirname "$0")"

# 設定
IMAGE_NAME="bundlesdf-blackwell"
TAG="latest"
DOCKERFILE="dockerfile"
CONTEXT_PATH=".." # プロジェクトルート（third_partyが含まれる場所）をコンテキストに指定

echo "--------------------------------------------------"
echo "Starting Docker build for ${IMAGE_NAME}:${TAG}"
echo "Context path: $(realpath ${CONTEXT_PATH})"
echo "--------------------------------------------------"

# Xauthority ファイルの準備（GUI用：ビルド前に行っておくと確実）
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - 2>/dev/null

# Docker Build の実行
# --no-cache を付けたい場合は引数で調整可能にする
docker build \
    -t ${IMAGE_NAME}:${TAG} \
    -f ${DOCKERFILE} \
    ${CONTEXT_PATH}

# ビルド結果の確認
if [ $? -eq 0 ]; then
    echo "--------------------------------------------------"
    echo "Successfully built ${IMAGE_NAME}:${TAG}"
    echo "You can now start the container using:"
    echo "  docker compose up -d"
    echo "--------------------------------------------------"
else
    echo "--------------------------------------------------"
    echo "Build FAILED. Please check the error messages above."
    echo "--------------------------------------------------"
    exit 1
fi