#!/bin/bash
# GPU TSP Test Runner - Run outside of Claude
# Usage: ./run_gpu_tsp_tests.sh

echo "════════════════════════════════════════════════════════"
echo "  GPU TSP Test Suite - Standalone Runner"
echo "════════════════════════════════════════════════════════"
echo ""

# Set WSL2 CUDA library path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "Step 1: Building with CUDA kernels..."
echo "────────────────────────────────────────────────────────"
cargo build --release --example test_gpu_tsp
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "Step 2: Running GPU TSP minimal tests..."
echo "────────────────────────────────────────────────────────"
cargo run --release --example test_gpu_tsp
TEST_STATUS=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [ $TEST_STATUS -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
else
    echo "❌ TESTS FAILED"
fi
echo "════════════════════════════════════════════════════════"

exit $TEST_STATUS
