#!/bin/bash
# CUDA Toolkit 12.8 Installation Script for Ubuntu 24.04
# GPU: RTX 5070 Laptop with Driver 570.172.08

set -e

echo "=========================================="
echo "CUDA Toolkit 12.8 Installation"
echo "=========================================="
echo ""

# Check if already downloaded
if [ ! -f cuda-keyring_1.1-1_all.deb ]; then
    echo "[1/5] Downloading CUDA keyring..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
else
    echo "[1/5] CUDA keyring already downloaded"
fi

echo ""
echo "[2/5] Installing CUDA keyring..."
sudo dpkg -i cuda-keyring_1.1-1_all.deb

echo ""
echo "[3/5] Updating package list..."
sudo apt-get update

echo ""
echo "[4/5] Installing CUDA Toolkit 12.8..."
echo "This may take 5-10 minutes..."
sudo apt-get install -y cuda-toolkit-12-8

echo ""
echo "[5/5] Setting up environment variables..."

# Add to current session
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc for persistence
if ! grep -q "cuda-12.8" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA 12.8 Environment" >> ~/.bashrc
    echo "export PATH=/usr/local/cuda-12.8/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "Added CUDA to ~/.bashrc"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
nvcc --version
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""
echo "Next steps:"
echo "1. Source your bashrc: source ~/.bashrc"
echo "2. Rebuild project: cd /home/diddy/Desktop/DARPA-DEMO && cargo clean && cargo build --release"
echo "3. Run GPU tests: cargo test --release test_performance_contract -- --nocapture"
echo ""
