#!/bin/bash
# Download and compile LKH-3 for head-to-head comparison

set -e

echo "════════════════════════════════════════════════════════"
echo "  LKH-3 Setup - Head-to-Head TSP Benchmark"
echo "════════════════════════════════════════════════════════"
echo ""

# Create benchmark directory
mkdir -p benchmarks/lkh
cd benchmarks/lkh

# Download LKH-3
echo "📥 Downloading LKH-3..."
if [ ! -f "LKH-3.0.9.tgz" ]; then
    wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.9.tgz
fi

# Extract
echo "📦 Extracting LKH-3..."
tar xzf LKH-3.0.9.tgz

# Compile
echo "🔨 Compiling LKH-3..."
cd LKH-3.0.9
make

echo ""
echo "✅ LKH-3 installed successfully!"
echo ""
echo "Binary location: benchmarks/lkh/LKH-3.0.9/LKH"
echo ""
echo "Next steps:"
echo "  1. Download TSPLIB instances: ./scripts/download_tsplib.sh"
echo "  2. Run head-to-head benchmark: ./run_lkh_comparison.sh"
echo ""
