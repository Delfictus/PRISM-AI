#!/bin/bash
# Download official DIMACS COLOR benchmark graphs
# Source: https://mat.tepper.cmu.edu/COLOR/instances.html

set -e

BENCHMARK_DIR="benchmarks"
mkdir -p "$BENCHMARK_DIR"

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║          DIMACS COLOR Benchmark Download Script                          ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Base URL for DIMACS COLOR benchmarks
BASE_URL="https://mat.tepper.cmu.edu/COLOR/instances"

# Function to download and extract
download_benchmark() {
    local name=$1
    local url=$2
    local expected_chi=$3

    echo "[*] Downloading $name (χ ≈ $expected_chi)..."

    if [ -f "$BENCHMARK_DIR/${name}.col" ]; then
        echo "    ✓ Already exists, skipping"
        return
    fi

    # Download compressed file
    if ! wget -q "$url" -O "$BENCHMARK_DIR/${name}.col.gz"; then
        echo "    ✗ Download failed, trying alternative source..."
        return 1
    fi

    # Extract
    gunzip -f "$BENCHMARK_DIR/${name}.col.gz"

    if [ -f "$BENCHMARK_DIR/${name}.col" ]; then
        local vertices=$(grep "^p edge" "$BENCHMARK_DIR/${name}.col" | awk '{print $3}')
        local edges=$(grep "^p edge" "$BENCHMARK_DIR/${name}.col" | awk '{print $4}')
        echo "    ✓ Downloaded: $vertices vertices, $edges edges"
    else
        echo "    ✗ Extraction failed"
        return 1
    fi
}

echo "Downloading SMALL benchmarks (< 100 vertices):"
echo "───────────────────────────────────────────────"
download_benchmark "myciel3" "$BASE_URL/myciel3.col.gz" "4"
download_benchmark "myciel4" "$BASE_URL/myciel4.col.gz" "5"
download_benchmark "queen5_5" "$BASE_URL/queen5_5.col.gz" "5"
download_benchmark "jean" "$BASE_URL/jean.col.gz" "10"

echo ""
echo "Downloading MEDIUM benchmarks (100-500 vertices):"
echo "──────────────────────────────────────────────────"
download_benchmark "queen8_8" "$BASE_URL/queen8_8.col.gz" "9"
download_benchmark "huck" "$BASE_URL/huck.col.gz" "11"
download_benchmark "david" "$BASE_URL/david.col.gz" "11"
download_benchmark "anna" "$BASE_URL/anna.col.gz" "11"
download_benchmark "games120" "$BASE_URL/games120.col.gz" "9"
download_benchmark "miles250" "$BASE_URL/miles250.col.gz" "8"
download_benchmark "miles500" "$BASE_URL/miles500.col.gz" "20"

echo ""
echo "Downloading LARGE benchmarks (500+ vertices, HARD):"
echo "────────────────────────────────────────────────────"
download_benchmark "dsjc125.1" "$BASE_URL/dsjc125.1.col.gz" "5"
download_benchmark "dsjc125.5" "$BASE_URL/dsjc125.5.col.gz" "17"
download_benchmark "dsjc250.1" "$BASE_URL/dsjc250.1.col.gz" "8"
download_benchmark "dsjc250.5" "$BASE_URL/dsjc250.5.col.gz" "28"
download_benchmark "dsjc500.1" "$BASE_URL/dsjc500.1.col.gz" "12"
download_benchmark "dsjc500.5" "$BASE_URL/dsjc500.5.col.gz" "48"

echo ""
echo "Downloading REGISTER ALLOCATION benchmarks:"
echo "────────────────────────────────────────────"
download_benchmark "fpsol2.i.1" "$BASE_URL/fpsol2.i.1.col.gz" "65"
download_benchmark "fpsol2.i.2" "$BASE_URL/fpsol2.i.2.col.gz" "30"
download_benchmark "fpsol2.i.3" "$BASE_URL/fpsol2.i.3.col.gz" "30"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                          DOWNLOAD COMPLETE                                ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Downloaded benchmarks are in: $BENCHMARK_DIR/"
echo ""
echo "To run tests:"
echo "  cargo run --release --example dimacs_benchmark_runner"
echo ""
echo "Graph categories:"
echo "  • Small (< 100 vertices): Good for quick testing"
echo "  • Medium (100-500 vertices): Standard difficulty"
echo "  • Large (500+ vertices): Challenging, world-class algorithms struggle here"
echo "  • DSJC: Random graphs with varying density"
echo "  • Queen: Queen graph (chess queens placement)"
echo "  • Register: Real-world register allocation problems"
echo ""
