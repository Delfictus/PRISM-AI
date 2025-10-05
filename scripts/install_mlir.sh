#!/bin/bash
# MLIR/LLVM Installation Script for PRISM-AI
# Installs LLVM 17 with MLIR and NVPTX backend for GPU compilation

set -e  # Exit on error

echo "=== PRISM-AI MLIR/LLVM Installation ==="
echo "This will install LLVM 17 with MLIR and CUDA support"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check system requirements
check_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"

    # Check for required tools
    for tool in cmake ninja-build python3 git; do
        if ! command -v $tool &> /dev/null; then
            echo -e "${RED}ERROR: $tool is not installed${NC}"
            echo "Please install with: sudo apt-get install $tool"
            exit 1
        fi
    done

    # Check for CUDA
    if ! command -v nvcc &> /dev/null; then
        echo -e "${YELLOW}WARNING: CUDA not detected. GPU features will be limited.${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
        echo -e "${GREEN}CUDA $CUDA_VERSION detected${NC}"
    fi

    # Check available disk space (need at least 20GB)
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 20 ]; then
        echo -e "${RED}ERROR: Insufficient disk space. Need at least 20GB, have ${AVAILABLE_SPACE}GB${NC}"
        exit 1
    fi

    echo -e "${GREEN}System requirements met${NC}"
}

# Install system dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing system dependencies...${NC}"

    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        clang \
        lld \
        zlib1g-dev \
        libedit-dev \
        libxml2-dev \
        python3-dev \
        python3-pip

    echo -e "${GREEN}Dependencies installed${NC}"
}

# Clone and build LLVM/MLIR
build_mlir() {
    echo -e "${YELLOW}Building LLVM/MLIR...${NC}"

    LLVM_DIR="$HOME/llvm-project"
    BUILD_DIR="$LLVM_DIR/build"

    # Clone if not exists
    if [ ! -d "$LLVM_DIR" ]; then
        echo "Cloning LLVM project..."
        git clone --depth 1 --branch llvmorg-17.0.6 \
            https://github.com/llvm/llvm-project.git "$LLVM_DIR"
    else
        echo "LLVM project already cloned"
    fi

    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Configure with CMake
    echo "Configuring build..."
    cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="mlir;clang" \
        -DLLVM_TARGETS_TO_BUILD="NVPTX;X86" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_ENABLE_CUDA_RUNNER=ON \
        -DMLIR_ENABLE_ROCM_RUNNER=OFF \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_BUILD_TESTS=OFF \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_USE_LINKER=lld

    # Build (use all available cores)
    echo "Building LLVM/MLIR (this will take 20-30 minutes)..."
    NPROC=$(nproc)
    ninja -j$NPROC

    # Install
    echo "Installing LLVM/MLIR..."
    sudo ninja install

    echo -e "${GREEN}LLVM/MLIR built and installed successfully${NC}"
}

# Setup environment variables
setup_environment() {
    echo -e "${YELLOW}Setting up environment...${NC}"

    PROFILE_FILE="$HOME/.bashrc"

    # Add to profile if not already present
    if ! grep -q "MLIR_DIR" "$PROFILE_FILE"; then
        echo "" >> "$PROFILE_FILE"
        echo "# MLIR/LLVM Configuration for PRISM-AI" >> "$PROFILE_FILE"
        echo "export MLIR_DIR=/usr/local" >> "$PROFILE_FILE"
        echo "export LLVM_DIR=/usr/local" >> "$PROFILE_FILE"
        echo "export PATH=\$MLIR_DIR/bin:\$PATH" >> "$PROFILE_FILE"
        echo "export LD_LIBRARY_PATH=\$MLIR_DIR/lib:\$LD_LIBRARY_PATH" >> "$PROFILE_FILE"
    fi

    # Source for current session
    export MLIR_DIR=/usr/local
    export LLVM_DIR=/usr/local
    export PATH=$MLIR_DIR/bin:$PATH
    export LD_LIBRARY_PATH=$MLIR_DIR/lib:$LD_LIBRARY_PATH

    echo -e "${GREEN}Environment configured${NC}"
}

# Verify installation
verify_installation() {
    echo -e "${YELLOW}Verifying installation...${NC}"

    # Check mlir-opt
    if command -v mlir-opt &> /dev/null; then
        echo -e "${GREEN}✓ mlir-opt found$(mlir-opt --version | head -1)${NC}"
    else
        echo -e "${RED}✗ mlir-opt not found${NC}"
        return 1
    fi

    # Check mlir-translate
    if command -v mlir-translate &> /dev/null; then
        echo -e "${GREEN}✓ mlir-translate found${NC}"
    else
        echo -e "${RED}✗ mlir-translate not found${NC}"
        return 1
    fi

    # Test NVPTX backend
    echo "Testing NVPTX backend..."
    cat << 'EOF' > /tmp/test_nvptx.mlir
module {
  func.func @test(%arg0: f32) -> f32 {
    %0 = arith.mulf %arg0, %arg0 : f32
    return %0 : f32
  }
}
EOF

    if mlir-opt /tmp/test_nvptx.mlir -o /tmp/test_nvptx_opt.mlir 2>/dev/null; then
        echo -e "${GREEN}✓ MLIR compilation working${NC}"
    else
        echo -e "${RED}✗ MLIR compilation failed${NC}"
        return 1
    fi

    rm -f /tmp/test_nvptx.mlir /tmp/test_nvptx_opt.mlir

    echo -e "${GREEN}Installation verified successfully!${NC}"
}

# Main execution
main() {
    echo "Starting MLIR/LLVM installation for PRISM-AI..."
    echo "This process will take approximately 30-45 minutes"
    echo

    check_requirements
    install_dependencies
    build_mlir
    setup_environment
    verify_installation

    echo
    echo -e "${GREEN}=== MLIR/LLVM Installation Complete ===${NC}"
    echo "Please run: source ~/.bashrc"
    echo "Or restart your terminal to apply environment changes"
    echo
    echo "Next steps:"
    echo "1. Run: ./scripts/install_cuda_toolkit.sh"
    echo "2. Run: ./scripts/setup_python_validation.sh"
    echo "3. Build project: cargo build --release --features mlir"
}

# Run if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi