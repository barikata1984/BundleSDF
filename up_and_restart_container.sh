#!/bin/bash

# このスクリプトは、PC再起動やログアウト/ログイン後に実行してください。
# X11の認証情報（Magic Cookie）を更新し、コンテナを再起動して反映させます。

echo "--------------------------------------------------"
echo "Updating X11 authentication..."
echo "--------------------------------------------------"

# 1. 認証情報を更新
# 既存のファイルを空にして作り直す（権限エラー回避のため sudo が必要な場合もあるが、通常は所有者が自分なら不要）
rm -f /tmp/.docker.xauth
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
chmod 644 /tmp/.docker.xauth

echo "Xauthority file updated at /tmp/.docker.xauth"

# 2. コンテナを起動・再起動
echo "--------------------------------------------------"
echo "Restarting container to apply changes..."
echo "--------------------------------------------------"

# もしコンテナがなければ作成、あれば設定反映（通常は up -d で十分だが、bind mountの中身変更を確実に反映させる意図で restart も併用）
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml restart bundlesdf-blackwell

echo "--------------------------------------------------"
echo "Done. You can now login using:"
echo "  docker exec -it bundlesdf-blackwell bash"
echo "--------------------------------------------------"
