#!/bin/bash
# Head-to-head comparison: GPU 2-opt vs LKH-3 on LARGE-SCALE problems
# Tests usa13509 (13k cities), d15112 (15k cities), d18512 (18k cities)

set -e

echo "════════════════════════════════════════════════════════════════════"
echo "  LARGE-SCALE HEAD-TO-HEAD: GPU vs LKH-3"
echo "  Testing 13k-18k city problems"
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

# Build the benchmark
echo "🔨 Building large-scale comparison benchmark..."
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cargo build --release --example lkh_large_scale_comparison

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STARTING LARGE-SCALE COMPARISON"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "⚠️  WARNING: This will take 1-3 HOURS!"
echo ""
echo "Expected per instance:"
echo "  • GPU: 40-100 seconds"
echo "  • LKH: 10-60 minutes (1 hour time limit)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 0
fi

# Run the benchmark
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cargo run --release --example lkh_large_scale_comparison

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  LARGE-SCALE COMPARISON COMPLETE"
echo "════════════════════════════════════════════════════════════════════"
echo ""
