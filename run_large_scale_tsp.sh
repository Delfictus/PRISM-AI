#!/bin/bash
# Large-Scale TSP Demo Runner
# Tests GPU TSP on famous TSPLIB instances (10k-18k cities)

echo "════════════════════════════════════════════════════════"
echo "  Large-Scale TSP Demo - Research Grade Problems"
echo "════════════════════════════════════════════════════════"
echo ""

# Set WSL2 CUDA library path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "⚠️  WARNING: This will test large problem instances"
echo "   • usa13509: 13,509 cities (~3 GB GPU memory)"
echo "   • d15112: 15,112 cities (~4 GB GPU memory)"
echo "   • d18512: 18,512 cities (~6 GB GPU memory)"
echo ""
echo "   Expected runtime: 2-10 minutes total"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Building large-scale demo..."
echo "────────────────────────────────────────────────────────"
cargo build --release --example large_scale_tsp_demo
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "Running large-scale benchmarks..."
echo "────────────────────────────────────────────────────────"
echo "y" | cargo run --release --example large_scale_tsp_demo

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Large-Scale TSP Demo Complete"
echo "════════════════════════════════════════════════════════"
