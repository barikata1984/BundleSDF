ROOT=$(pwd)

# Set PyTorch library path
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"
export TORCH_LIBRARIES="/usr/local/lib/python3.12/dist-packages/torch/lib"

# Additional PyTorch environment variables
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;12.0"  # Added 8.9 and 12.0 for Ada Lovelace and Blackwell, respectively
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"

# Ensure PyTorch can be found
export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:$PYTHONPATH"

# Print debug info
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "Testing PyTorch import..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

cd ${ROOT}/mycuda && rm -rf build *egg* && python3 -m pip install -e . 
cd ${ROOT}/third_party/BundleTrack && rm -rf build && mkdir build && cd build && cmake .. && make -j11