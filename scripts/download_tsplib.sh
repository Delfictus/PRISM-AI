#!/bin/bash
# Download standard TSPLIB instances for head-to-head comparison

set -e

echo "════════════════════════════════════════════════════════"
echo "  TSPLIB Instance Downloader"
echo "════════════════════════════════════════════════════════"
echo ""

mkdir -p benchmarks/tsplib
cd benchmarks/tsplib

# Download individual instances
instances=(
    "berlin52.tsp"
    "eil51.tsp"
    "eil76.tsp"
    "kroA100.tsp"
    "kroB100.tsp"
    "rd100.tsp"
    "eil101.tsp"
    "pr152.tsp"
    "kroA200.tsp"
    "pr264.tsp"
    "pr299.tsp"
    "pr439.tsp"
    "pcb442.tsp"
    "d493.tsp"
    "u574.tsp"
    "rat575.tsp"
    "p654.tsp"
    "d657.tsp"
    "pr1002.tsp"
    "u1060.tsp"
    "pr2392.tsp"
)

echo "📥 Downloading TSPLIB instances..."
echo ""

BASE_URL="http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp"

for instance in "${instances[@]}"; do
    if [ ! -f "$instance" ]; then
        echo "  Downloading $instance..."
        wget -q "$BASE_URL/$instance" || echo "  ⚠️  Failed to download $instance"
    else
        echo "  ✓ $instance already exists"
    fi
done

echo ""
echo "✅ TSPLIB instances downloaded!"
echo ""
echo "Location: benchmarks/tsplib/"
echo "Total instances: ${#instances[@]}"
echo ""
