#!/bin/bash
# GPU-Accelerated DIMACS Benchmark Runner
# Sets up WSL2 CUDA environment and runs benchmarks

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "╔════════════════════════════════════════════════════╗"
echo "║   GPU-Accelerated DIMACS Benchmark Runner          ║"
echo "║   NVIDIA RTX 5070 Laptop GPU (8GB)                 ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

cargo run --release --example dimacs_benchmark_runner_gpu
