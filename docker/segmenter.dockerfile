# SAM 3 segmenter node (separate container from the BundleSDF core).
# Runtime-only CUDA base: the node runs inference, nothing is compiled here.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ROS One (noble) apt repository maintained by TechFak Bielefeld.
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSL https://ros.packages.techfak.net/gpg.key -o /etc/apt/keyrings/ros-one-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net noble main" \
        > /etc/apt/sources.list.d/ros1.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-one-ros-base \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init && \
    echo "yaml https://ros.packages.techfak.net/ros-one.yaml one" > /etc/ros/rosdep/sources.list.d/1-ros-one.list && \
    rosdep update --rosdistro one

# Python deps (system python3.12). torch/torchvision from the CUDA 12.8 index.
# transformers >= 5.11 ships the sam3_video streaming API (verified against 5.10.2 sources).
RUN pip3 install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.1 torchvision==0.24.1 && \
    pip3 install --break-system-packages --no-cache-dir \
        "transformers==5.12.1" \
        "huggingface_hub>=0.35" \
        accelerate \
        numpy

# Hugging Face cache is expected to be a mounted volume so the ~3.45GB sam3
# weights are not baked into the image. Mount a host dir at /hf_cache.
ENV HF_HOME=/hf_cache
VOLUME /hf_cache

# Source ROS on every login shell so the node can import rospy.
RUN echo "source /opt/ros/one/setup.bash" >> /etc/bash.bashrc
SHELL ["/bin/bash", "-lc"]

CMD ["bash"]
