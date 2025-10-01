#!/bin/bash
# GPU TSP Benchmark Runner - Run outside of Claude
# Usage: ./run_gpu_tsp_benchmarks.sh

echo "════════════════════════════════════════════════════════"
echo "  GPU TSP TSPLIB Benchmark Suite - Standalone Runner"
echo "════════════════════════════════════════════════════════"
echo ""

# Set WSL2 CUDA library path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "Step 1: Building benchmark runner..."
echo "────────────────────────────────────────────────────────"
cargo build --release --example tsp_benchmark_runner_gpu
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "Step 2: Running TSPLIB benchmarks (9 instances)..."
echo "────────────────────────────────────────────────────────"
cargo run --release --example tsp_benchmark_runner_gpu
BENCH_STATUS=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [ $BENCH_STATUS -eq 0 ]; then
    echo "✅ ALL BENCHMARKS COMPLETED"
else
    echo "❌ BENCHMARKS FAILED"
fi
echo "════════════════════════════════════════════════════════"

exit $BENCH_STATUS
