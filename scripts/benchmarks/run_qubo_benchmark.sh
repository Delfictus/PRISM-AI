#!/bin/bash
# QUBO Benchmark Runner - Direct Competition with Intel Loihi 2
# Maximum Independent Set (MIS) Problems - 105 instances

echo "════════════════════════════════════════════════════════"
echo "  QUBO Benchmark: Competing with Intel Loihi 2"
echo "  Maximum Independent Set - 105 Problem Instances"
echo "════════════════════════════════════════════════════════"
echo ""

echo "Building QUBO benchmark..."
echo "────────────────────────────────────────────────────────"
cargo build --release --example qubo_loihi_benchmark
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "Running 105 QUBO benchmarks..."
echo "Expected runtime: 10-20 minutes"
echo "────────────────────────────────────────────────────────"
echo ""

cargo run --release --example qubo_loihi_benchmark

echo ""
echo "════════════════════════════════════════════════════════"
echo "  QUBO Benchmark Complete"
echo "════════════════════════════════════════════════════════"
