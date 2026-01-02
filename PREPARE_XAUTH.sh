#!/bin/bash

# Prepare Xauthority data
XAUTH=/tmp/.docker.xauth
if [ -d "$XAUTH" ]; then
    echo "Removing directory $XAUTH"
    sudo rm -rf "$XAUTH"
fi
# Always recreate the file to ensure permissions and content are correct
rm -f $XAUTH
touch $XAUTH

# Use current DISPLAY, fallback to :0 if not set
DISPLAY_NUM=${DISPLAY:-:0}
xauth_list=$(xauth nlist $DISPLAY_NUM | sed -e 's/^..../ffff/')
if [ ! -z "$xauth_list" ]; then
    echo $xauth_list | xauth -f $XAUTH nmerge -
fi
chmod a+r $XAUTH

xhost +local:docker  # Docker からの接続
