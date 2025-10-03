#!/bin/bash
# Quick test: Run small LKH comparison first to verify setup

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "Testing LKH setup with small benchmark..."
echo ""

# Run the small benchmark which we know works
cargo run --release --example lkh_comparison_benchmark 2>&1 | grep -A 300 "Starting head-to-head"
