#!/bin/bash
# PRISM-AI Adaptive World-Record Demo Runner
# Auto-detects GPU (RTX 5070 vs H100) and runs optimized benchmark

set -e

echo "🌈 PRISM-AI Adaptive World-Record Demo Setup 🌈"
echo ""

# Detect GPU
if nvidia-smi | grep -q "H100"; then
    export GPU_TYPE="H100"
    echo "✓ Detected: NVIDIA H100"
    echo "  Optimal problem size: 100 dimensions"
    echo "  Expected speedup: 1000x+ on graph coloring"
elif nvidia-smi | grep -q "RTX"; then
    export GPU_TYPE="RTX5070"
    echo "✓ Detected: NVIDIA RTX GPU"
    echo "  Optimal problem size: 50 dimensions"
    echo "  Expected speedup: 300x+ on graph coloring"
else
    export GPU_TYPE="CPU"
    echo "⚠ No NVIDIA GPU detected"
    echo "  Running in CPU mode (slower)"
fi

echo ""
echo "Creating standalone runner project..."

# Create runner directory
RUNNER_DIR="$HOME/prism-adaptive-runner"
mkdir -p "$RUNNER_DIR/src"
cd "$RUNNER_DIR"

# Create Cargo.toml
cat > Cargo.toml << 'EOF'
[package]
name = "prism-adaptive-runner"
version = "0.1.0"
edition = "2021"

[dependencies]
prism-ai = { path = "/home/diddy/Desktop/PRISM-AI" }
prct-core = { path = "/home/diddy/Desktop/PRISM-AI/src/prct-core" }
shared-types = { path = "/home/diddy/Desktop/PRISM-AI/src/shared-types" }
ndarray = "0.15"
anyhow = "1.0"
colored = "2.1"
EOF

# Copy demo code
cp /home/diddy/Desktop/PRISM-AI/examples/adaptive_world_record_demo.rs src/main.rs

echo "✓ Runner project created at: $RUNNER_DIR"
echo ""
echo "Building (this may take a few minutes)..."
echo ""

# Build
cargo build --release 2>&1 | grep -E "Compiling prism-ai|Finished|error"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ BUILD SUCCESSFUL!"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Running Adaptive World-Record Demo..."
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    # Run the demo
    cargo run --release

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Demo Complete!"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo ""
    echo "❌ Build failed. Check errors above."
    exit 1
fi
