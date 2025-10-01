#!/bin/bash
# Automated head-to-head comparison: GPU 2-opt vs LKH-3

set -e

echo "════════════════════════════════════════════════════════════════════"
echo "  HEAD-TO-HEAD: GPU 2-opt vs LKH-3"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check if LKH-3 is installed
if [ ! -f "benchmarks/lkh/LKH-3.0.9/LKH" ]; then
    echo "📦 LKH-3 not found. Setting up..."
    echo ""
    chmod +x scripts/setup_lkh.sh
    ./scripts/setup_lkh.sh
    echo ""
fi

# Check if TSPLIB instances are downloaded
if [ ! -d "benchmarks/tsplib" ] || [ -z "$(ls -A benchmarks/tsplib)" ]; then
    echo "📥 TSPLIB instances not found. Downloading..."
    echo ""
    chmod +x scripts/download_tsplib.sh
    ./scripts/download_tsplib.sh
    echo ""
fi

# Build the benchmark
echo "🔨 Building head-to-head benchmark..."
cargo build --release --example lkh_comparison_benchmark

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STARTING HEAD-TO-HEAD COMPARISON"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Run the benchmark
cargo run --release --example lkh_comparison_benchmark

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  HEAD-TO-HEAD COMPARISON COMPLETE"
echo "════════════════════════════════════════════════════════════════════"
echo ""
