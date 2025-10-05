#!/bin/bash
# CUDA Toolkit Installation Script for PRISM-AI
# Installs CUDA 12.3 with all necessary components

set -e

echo "=== PRISM-AI CUDA Toolkit Installation ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Detect system
detect_system() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    else
        echo -e "${RED}Cannot detect operating system${NC}"
        exit 1
    fi

    echo -e "${GREEN}Detected: $OS $VER${NC}"

    # Check if Ubuntu 22.04 or 20.04
    if [[ "$OS" == "ubuntu" ]]; then
        if [[ "$VER" == "22.04" ]]; then
            CUDA_REPO="cuda-repo-ubuntu2204-12-3-local_12.3.2-545.23.08-1_amd64.deb"
            CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/$CUDA_REPO"
        elif [[ "$VER" == "20.04" ]]; then
            CUDA_REPO="cuda-repo-ubuntu2004-12-3-local_12.3.2-545.23.08-1_amd64.deb"
            CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/$CUDA_REPO"
        else
            echo -e "${RED}Unsupported Ubuntu version: $VER${NC}"
            echo "Supported versions: 20.04, 22.04"
            exit 1
        fi
    else
        echo -e "${RED}This script currently only supports Ubuntu${NC}"
        exit 1
    fi
}

# Check for existing CUDA installation
check_existing_cuda() {
    echo -e "${YELLOW}Checking for existing CUDA installation...${NC}"

    if [ -d /usr/local/cuda ]; then
        if [ -f /usr/local/cuda/version.json ]; then
            EXISTING_VERSION=$(python3 -c "import json; print(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'])" 2>/dev/null || echo "unknown")
            echo -e "${YELLOW}Found existing CUDA version: $EXISTING_VERSION${NC}"
            read -p "Continue with installation? This will override existing CUDA. (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
}

# Check NVIDIA driver
check_nvidia_driver() {
    echo -e "${YELLOW}Checking NVIDIA driver...${NC}"

    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}NVIDIA driver not found!${NC}"
        echo "Please install NVIDIA driver first:"
        echo "  sudo apt update"
        echo "  sudo apt install nvidia-driver-545"
        exit 1
    fi

    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo -e "${GREEN}NVIDIA Driver version: $DRIVER_VERSION${NC}"

    # Check if driver version is sufficient for CUDA 12.3 (need >= 545)
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -lt 545 ]; then
        echo -e "${RED}Driver version $DRIVER_VERSION is too old for CUDA 12.3${NC}"
        echo "Please update to driver version 545 or newer"
        exit 1
    fi
}

# Install CUDA
install_cuda() {
    echo -e "${YELLOW}Installing CUDA 12.3...${NC}"

    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    # Download CUDA installer
    echo "Downloading CUDA installer (3.4GB)..."
    wget -q --show-progress "$CUDA_URL"

    # Install
    echo "Installing CUDA..."
    sudo dpkg -i "$CUDA_REPO"

    # Copy keyring
    sudo cp /var/cuda-repo-*/cuda-*-keyring.gpg /usr/share/keyrings/

    # Update and install
    sudo apt-get update
    sudo apt-get -y install cuda-12-3

    # Install additional libraries
    echo "Installing additional CUDA libraries..."
    sudo apt-get -y install \
        libcudnn8 \
        libcudnn8-dev \
        libcublas-12-3 \
        libcusparse-12-3 \
        libcufft-12-3 \
        libcurand-12-3 \
        libcusolver-12-3 \
        libnvjitlink-12-3

    # Cleanup
    cd /
    rm -rf "$TEMP_DIR"

    echo -e "${GREEN}CUDA 12.3 installed successfully${NC}"
}

# Setup environment variables
setup_cuda_env() {
    echo -e "${YELLOW}Setting up CUDA environment...${NC}"

    PROFILE_FILE="$HOME/.bashrc"
    CUDA_HOME="/usr/local/cuda-12.3"

    # Add to profile if not already present
    if ! grep -q "CUDA_HOME" "$PROFILE_FILE"; then
        echo "" >> "$PROFILE_FILE"
        echo "# CUDA Configuration for PRISM-AI" >> "$PROFILE_FILE"
        echo "export CUDA_HOME=$CUDA_HOME" >> "$PROFILE_FILE"
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> "$PROFILE_FILE"
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> "$PROFILE_FILE"
        echo "export CUDA_VISIBLE_DEVICES=0" >> "$PROFILE_FILE"
    fi

    # Export for current session
    export CUDA_HOME=$CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    echo -e "${GREEN}CUDA environment configured${NC}"
}

# Verify CUDA installation
verify_cuda() {
    echo -e "${YELLOW}Verifying CUDA installation...${NC}"

    # Check nvcc
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
        echo -e "${GREEN}✓ nvcc version $NVCC_VERSION${NC}"
    else
        echo -e "${RED}✗ nvcc not found${NC}"
        return 1
    fi

    # Test compilation
    echo "Testing CUDA compilation..."
    cat << 'EOF' > /tmp/test_cuda.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel() {
    printf("CUDA kernel executed successfully\n");
}

int main() {
    test_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("Found %d CUDA devices\n", device_count);

    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    return 0;
}
EOF

    if nvcc /tmp/test_cuda.cu -o /tmp/test_cuda 2>/dev/null; then
        echo -e "${GREEN}✓ CUDA compilation successful${NC}"
        /tmp/test_cuda
    else
        echo -e "${RED}✗ CUDA compilation failed${NC}"
        return 1
    fi

    rm -f /tmp/test_cuda.cu /tmp/test_cuda

    echo -e "${GREEN}CUDA verification complete!${NC}"
}

# Install cuDNN (for neural network operations)
install_cudnn() {
    echo -e "${YELLOW}Installing cuDNN...${NC}"

    # Download and install cuDNN
    wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
    sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cudnn-cuda-12

    rm cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb

    echo -e "${GREEN}cuDNN installed${NC}"
}

# Main execution
main() {
    echo "Starting CUDA 12.3 installation for PRISM-AI..."
    echo

    detect_system
    check_existing_cuda
    check_nvidia_driver
    install_cuda
    setup_cuda_env
    install_cudnn
    verify_cuda

    echo
    echo -e "${GREEN}=== CUDA Toolkit Installation Complete ===${NC}"
    echo "Please run: source ~/.bashrc"
    echo "Or restart your terminal to apply environment changes"
    echo
    echo "Next steps:"
    echo "1. Run: ./scripts/setup_python_validation.sh"
    echo "2. Build project: cargo build --release --features cuda"
}

# Run if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi