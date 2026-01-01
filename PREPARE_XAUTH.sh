#!/bin/bash

# Prepare Xauthority data
XAUTH=/tmp/.docker.xauth
if [ -d "$XAUTH" ]; then
    echo "Removing directory $XAUTH"
    sudo rm -rf "$XAUTH"
fi
if [ ! -f $XAUTH ]; then
    touch $XAUTH
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]; then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    fi
    chmod a+r $XAUTH
fi
