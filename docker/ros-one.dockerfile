# syntax=docker/dockerfile:1
# BundleSDF on Ubuntu 22.04 (jammy) + ROS One, CUDA 12.9, OpenCV 4.13 CUDA build.
# Repo is bind-mounted at runtime (see run_container.sh); nothing from the source
# tree (including model weights) is copied into the image.

ARG CUDA_ARCH_BIN=12.0

########################################
# Stage 1: build OpenCV with CUDA
########################################
FROM nvidia/cuda:12.9.2-devel-ubuntu22.04 AS opencv-builder

ARG OPENCV_VERSION=4.13.0
ARG CUDA_ARCH_BIN=12.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    libeigen3-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git /opencv && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git /opencv_contrib

RUN cmake -S /opencv -B /opencv/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/opencv-cuda \
      -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
      -DCUDA_ARCH_PTX="" \
      -DBUILD_LIST=core,imgproc,calib3d,features2d,highgui,cudev,cudaarithm,cudafilters,cudawarping,cudafeatures2d,cudaimgproc,cudaoptflow \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_python3=OFF \
      -DBUILD_JAVA=OFF && \
    cmake --build /opencv/build --target install && \
    rm -rf /opencv /opencv_contrib

########################################
# Stage 2: runtime / build environment
########################################
FROM nvidia/cuda:12.9.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Build tools and C++ dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    time \
    python3-pip \
    python3-dev \
    libeigen3-dev \
    pybind11-dev \
    libyaml-cpp-dev \
    libpcl-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-serialization-dev \
    libzmq3-dev \
    freeglut3-dev \
    libblas-dev \
    liblapack-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# ROS One (jammy)
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSL https://ros.packages.techfak.net/gpg.key -o /etc/apt/keyrings/ros-one-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net jammy main" \
      > /etc/apt/sources.list.d/ros1.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-one-ros-base \
    ros-one-tf2-ros \
    ros-one-message-filters \
    ros-one-catkin \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# OpenCV (CUDA) from build stage
COPY --from=opencv-builder /opt/opencv-cuda /opt/opencv-cuda
ENV OpenCV_DIR=/opt/opencv-cuda/lib/cmake/opencv4
ENV LD_LIBRARY_PATH=/opt/opencv-cuda/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Python packages. numpy is pinned last because kaolin pulls a newer numpy.
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
      --index-url https://download.pytorch.org/whl/cu129 && \
    pip3 install --no-cache-dir kaolin==0.18.0 \
      -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu129.html && \
    pip3 install --no-cache-dir \
      trimesh open3d kornia einops loguru transformations imageio scikit-image \
      wandb matplotlib tqdm ruamel.yaml sacred pymongo pyrender jupyterlab ninja \
      "Cython>=0.29.37" yacs scipy scikit-learn opencv-python pytorch_lightning \
      awscli-plugin-endpoint gputil xatlas pymeshlab rtree dearpygui \
      pytinyrenderer PyQt5 cython-npm chardet openpyxl && \
    pip3 install --no-cache-dir "transformers==5.12.1" "huggingface_hub>=0.35" accelerate && \
    pip3 install --no-cache-dir numpy==1.26.4

ENV CUDA_HOME=/usr/local/cuda
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTHONUNBUFFERED=1
# SAM3 weights are gated + large; keep the HF cache on a mounted volume
ENV HF_HOME=/hf_cache

RUN imageio_download_bin freeimage || true

# Source ROS One in interactive shells. catkin_ws is a local build artifact
# (gitignored, created via `catkin_make` after the repo is mounted), so guard
# on its existence rather than failing shells that haven't built it yet.
RUN echo "source /opt/ros/one/setup.bash" >> /root/.bashrc && \
    echo '[ -f /workspace/catkin_ws/devel/setup.bash ] && source /workspace/catkin_ws/devel/setup.bash' >> /root/.bashrc

# CUDA MPS: start the control daemon before CMD, quit it gracefully on stop.
COPY docker/mps-entrypoint.sh /usr/local/bin/mps-entrypoint.sh
RUN chmod +x /usr/local/bin/mps-entrypoint.sh

WORKDIR /home
ENTRYPOINT ["/usr/local/bin/mps-entrypoint.sh"]
CMD ["bash"]
