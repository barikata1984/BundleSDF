#!/bin/bash
ROOT=$(pwd)

# 1. 環境変数の設定
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"
export TORCH_LIBRARIES="/usr/local/lib/python3.12/dist-packages/torch/lib"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;12.0"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:$PYTHONPATH"

# 2. デバッグ情報の表示
echo "--------------------------------------------------"
echo "Testing PyTorch environment..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "--------------------------------------------------"

# 3. 追加：Python パッケージを編集可能モードで登録
echo "Registering third_party packages..."
# --break-system-packages を付けてシステムPython環境への干渉エラーを回避
pip3 install --break-system-packages -e ${ROOT}/third_party/sam3
pip3 install --break-system-packages -e ${ROOT}/third_party/Cutie

# 4. mycuda のビルドとインストール
echo "Building mycuda..."
cd ${ROOT}/mycuda && rm -rf build *egg* && python3 -m pip install --break-system-packages -e . 

# 5. BundleTrack (C++ライブラリ) のビルド
echo "Building BundleTrack C++ libraries..."
cd ${ROOT}/third_party/BundleTrack && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)

echo "--------------------------------------------------"
echo "Build process completed successfully."
echo "--------------------------------------------------"