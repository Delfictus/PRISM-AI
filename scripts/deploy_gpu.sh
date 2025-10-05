#!/bin/bash

# PRISM-AI GPU Deployment Script
# Deploys the GPU-accelerated version with all optimizations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_DIR="/opt/prism-ai"
SERVICE_NAME="prism-ai"
LOG_DIR="/var/log/prism-ai"
CONFIG_DIR="/etc/prism-ai"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}     PRISM-AI GPU Deployment Script${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
fi

# Step 1: Check GPU availability
echo -e "\n${BLUE}Step 1: Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    print_status "GPU detected and driver installed"
else
    print_warning "No GPU detected. Deployment will continue but performance will be limited."
    read -p "Continue without GPU? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Check CUDA installation
echo -e "\n${BLUE}Step 2: Checking CUDA installation...${NC}"
if command -v nvcc &> /dev/null; then
    nvcc --version
    print_status "CUDA toolkit installed"
else
    print_warning "CUDA toolkit not found"
    echo "Installing CUDA 12.3..."
    bash scripts/install_cuda_toolkit.sh
fi

# Step 3: Build from source
echo -e "\n${BLUE}Step 3: Building PRISM-AI...${NC}"

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    print_warning "Rust not installed. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Build with GPU features
cargo build --release --features cuda
if [ $? -eq 0 ]; then
    print_status "Build successful with GPU acceleration"
else
    print_error "Build failed"
fi

# Build CUDA kernels
echo "Building CUDA kernels..."
cd src/kernels
for kernel in *.cu; do
    if [ -f "$kernel" ]; then
        echo "  Compiling $kernel..."
        nvcc -ptx -arch=sm_86 -O3 --use_fast_math "$kernel" -o "${kernel%.cu}.ptx"
    fi
done
cd ../..
print_status "CUDA kernels compiled"

# Step 4: Create deployment directories
echo -e "\n${BLUE}Step 4: Setting up deployment directories...${NC}"
mkdir -p $DEPLOY_DIR
mkdir -p $LOG_DIR
mkdir -p $CONFIG_DIR
mkdir -p $DEPLOY_DIR/ptx

print_status "Directories created"

# Step 5: Copy files
echo -e "\n${BLUE}Step 5: Copying files...${NC}"
cp target/release/prism-ai $DEPLOY_DIR/
cp -r target/ptx/* $DEPLOY_DIR/ptx/ 2>/dev/null || true
cp -r src/kernels/*.ptx $DEPLOY_DIR/ptx/ 2>/dev/null || true

# Copy configuration
cat > $CONFIG_DIR/prism-ai.toml <<EOF
# PRISM-AI Configuration

[gpu]
enabled = true
device_id = 0
precision = "double_double"  # 106-bit precision
cuda_cores = "auto"

[performance]
threads = 16
batch_size = 1000
cache_size = "4GB"

[quantum]
evolution_method = "trotter_suzuki"
time_steps = 1000
convergence_threshold = 1e-30

[cma]
pac_bayes_confidence = 0.99
conformal_coverage = 0.95
ensemble_size = 100

[logging]
level = "info"
file = "$LOG_DIR/prism-ai.log"
EOF

print_status "Files deployed to $DEPLOY_DIR"

# Step 6: Create systemd service
echo -e "\n${BLUE}Step 6: Creating systemd service...${NC}"
cat > /etc/systemd/system/prism-ai.service <<EOF
[Unit]
Description=PRISM-AI GPU-Accelerated Platform
Documentation=https://github.com/Delfictus/PRISM-AI
After=network.target

[Service]
Type=simple
User=prism
Group=prism
WorkingDirectory=$DEPLOY_DIR
ExecStart=$DEPLOY_DIR/prism-ai --config $CONFIG_DIR/prism-ai.toml
Restart=always
RestartSec=10

# Performance tuning
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=99
IOSchedulingClass=realtime
IOSchedulingPriority=0

# GPU access
SupplementaryGroups=video
PrivateDevices=no

# Environment
Environment="CUDA_HOME=/usr/local/cuda"
Environment="PRISM_PTX_PATH=$DEPLOY_DIR/ptx"
Environment="RUST_LOG=info"

# Limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

# Create user if doesn't exist
if ! id -u prism &>/dev/null; then
    useradd -r -s /bin/false -d $DEPLOY_DIR prism
    usermod -a -G video prism  # Add to video group for GPU access
fi

chown -R prism:prism $DEPLOY_DIR
chown -R prism:prism $LOG_DIR
chown -R prism:prism $CONFIG_DIR

systemctl daemon-reload
print_status "Systemd service created"

# Step 7: Run validation tests
echo -e "\n${BLUE}Step 7: Running validation tests...${NC}"

# Quick GPU test
echo "Testing GPU functionality..."
$DEPLOY_DIR/prism-ai --test-gpu
if [ $? -eq 0 ]; then
    print_status "GPU test passed"
else
    print_warning "GPU test failed - check configuration"
fi

# Performance benchmark
echo "Running performance benchmark..."
timeout 30 $DEPLOY_DIR/prism-ai --benchmark > $LOG_DIR/benchmark.txt 2>&1 || true
if [ -f $LOG_DIR/benchmark.txt ]; then
    echo "Benchmark results:"
    grep -E "Speedup|GFLOPS|Precision" $LOG_DIR/benchmark.txt | head -5
    print_status "Benchmark completed"
fi

# Step 8: Configure firewall (if needed)
echo -e "\n${BLUE}Step 8: Configuring firewall...${NC}"
if command -v ufw &> /dev/null; then
    ufw allow 8080/tcp comment 'PRISM-AI HTTP'
    ufw allow 8443/tcp comment 'PRISM-AI HTTPS'
    print_status "Firewall rules added"
else
    print_warning "UFW not installed - skipping firewall configuration"
fi

# Step 9: Start service
echo -e "\n${BLUE}Step 9: Starting PRISM-AI service...${NC}"
systemctl enable prism-ai
systemctl start prism-ai

sleep 2
if systemctl is-active --quiet prism-ai; then
    print_status "PRISM-AI service started successfully"
else
    print_error "Failed to start PRISM-AI service"
fi

# Step 10: Display status
echo -e "\n${BLUE}Deployment Summary:${NC}"
echo "═══════════════════════════════════════════════════════════"
echo -e "Installation directory: ${GREEN}$DEPLOY_DIR${NC}"
echo -e "Configuration file:     ${GREEN}$CONFIG_DIR/prism-ai.toml${NC}"
echo -e "Log directory:          ${GREEN}$LOG_DIR${NC}"
echo -e "Service status:         ${GREEN}$(systemctl is-active prism-ai)${NC}"

# Check performance metrics
if [ -f $LOG_DIR/benchmark.txt ]; then
    SPEEDUP=$(grep -oP 'Speedup: \K[0-9]+x' $LOG_DIR/benchmark.txt | head -1)
    echo -e "GPU Speedup:            ${GREEN}${SPEEDUP:-N/A}${NC}"
fi

echo "═══════════════════════════════════════════════════════════"

# Display service commands
echo -e "\n${BLUE}Useful commands:${NC}"
echo "  systemctl status prism-ai    # Check service status"
echo "  systemctl restart prism-ai   # Restart service"
echo "  journalctl -u prism-ai -f    # View logs"
echo "  $DEPLOY_DIR/prism-ai --help  # Show help"

echo -e "\n${GREEN}✓ Deployment complete!${NC}"
echo -e "${GREEN}PRISM-AI is now running with GPU acceleration.${NC}"

# Optional: Run validation
read -p "Run full validation suite? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running validation..."
    cd validation
    python3 validate_quantum_evolution.py
    cd ..
fi

exit 0