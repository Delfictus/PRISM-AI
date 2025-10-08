#!/bin/bash
set -e

echo "==================================================================="
echo "PRISM-AI H100 Benchmark Runner"
echo "==================================================================="
echo ""

# Detect GPU
nvidia-smi || { echo "ERROR: No GPU detected!"; exit 1; }

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

cd /prism-ai

# Run system validation test
echo "=== Test 1: System Validation ==="
echo "Running test_full_gpu to verify baseline performance..."
timeout 60 cargo run --release --features cuda --example test_full_gpu 2>&1 | tee /tmp/test_full_gpu.log
echo ""

# Run world record dashboard
echo "=== Test 2: World Record Dashboard ==="
echo "Running 4 benchmark scenarios..."
timeout 300 cargo run --release --features cuda --example world_record_dashboard 2>&1 | tee /tmp/world_record.log
echo ""

# Test MTX parser
echo "=== Test 3: MTX Parser ==="
echo "Testing official DIMACS instance loading..."
timeout 30 cargo run --release --features cuda --example test_mtx_parser 2>&1 | tee /tmp/mtx_parser.log
echo ""

echo "==================================================================="
echo "Benchmark Results Saved:"
echo "  - /tmp/test_full_gpu.log"
echo "  - /tmp/world_record.log"
echo "  - /tmp/mtx_parser.log"
echo "==================================================================="

# Extract key metrics
echo ""
echo "=== PERFORMANCE SUMMARY ==="
grep "Total Latency:" /tmp/test_full_gpu.log || echo "Latency not found"
grep "vs World Record:" /tmp/world_record.log | head -4 || echo "Speedups not found"
grep "Successfully parsed" /tmp/mtx_parser.log || echo "Parser status not found"
