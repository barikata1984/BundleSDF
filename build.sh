#!/bin/bash

# スクリプトのあるディレクトリに移動
cd "$(dirname "$0")"
ROOT=$(pwd)

# Python バージョンを動的に取得
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# 1. 環境変数の設定
export LD_LIBRARY_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib:$LD_LIBRARY_PATH"
export TORCH_LIBRARIES="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;12.0"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
export PYTHONPATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages:$PYTHONPATH"

# 2. デバッグ情報の表示
echo "--------------------------------------------------"
echo "Testing PyTorch environment..."
echo "Python version: ${PYTHON_VERSION}"
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "--------------------------------------------------"

# 3. 追加：Python パッケージを編集可能モードで登録
echo "Registering third_party packages..."
pip3 install -e ${ROOT}/third_party/sam3
pip3 install -e ${ROOT}/third_party/Cutie

# 4. mycuda のビルドとインストール
echo "Building mycuda..."
cd ${ROOT}/mycuda && rm -rf build *egg* && python3 -m pip install -e . 
# Use the following when build artifacts are left
#cd ${ROOT}/mycuda && python3 -m pip install -e .  

# 5. BundleTrack (C++ライブラリ) のビルド
echo "Building BundleTrack C++ libraries..."
cd ${ROOT}/third_party/BundleTrack && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
# Use the following when build artifacts are left
#cd ${ROOT}/third_party/BundleTrack && mkdir -p build && cd build && cmake .. && make -j$(nproc)
 
echo "--------------------------------------------------"
echo "Build process completed successfully."
echo "--------------------------------------------------"