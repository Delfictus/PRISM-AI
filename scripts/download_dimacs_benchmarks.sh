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

# Base URLs for DIMACS COLOR benchmarks (multiple sources)
BASE_URL_PRIMARY="https://cedric.cnam.fr/~porumbed/graphs"
BASE_URL_FALLBACK="http://mat.gsia.cmu.edu/COLOR/instances"

# Function to download and extract
download_benchmark() {
    local name=$1
    local expected_chi=$2

    echo "[*] Downloading $name (χ ≈ $expected_chi)..."

    if [ -f "$BENCHMARK_DIR/${name}.col" ]; then
        echo "    ✓ Already exists, skipping"
        return
    fi

    # Try primary source
    local url1="${BASE_URL_PRIMARY}/${name}.col.gz"
    local url2="${BASE_URL_PRIMARY}/${name}.col"

    # Try .col.gz first
    if wget -q "$url1" -O "$BENCHMARK_DIR/${name}.col.gz" 2>/dev/null; then
        gunzip -f "$BENCHMARK_DIR/${name}.col.gz" 2>/dev/null
    # Try uncompressed .col
    elif wget -q "$url2" -O "$BENCHMARK_DIR/${name}.col" 2>/dev/null; then
        true  # Already downloaded
    else
        echo "    ⚠ Not available from primary source, trying fallback..."
        # Try fallback source
        if wget -q "${BASE_URL_FALLBACK}/${name}.col.gz" -O "$BENCHMARK_DIR/${name}.col.gz" 2>/dev/null; then
            gunzip -f "$BENCHMARK_DIR/${name}.col.gz" 2>/dev/null
        else
            echo "    ✗ Download failed from all sources"
            return 1
        fi
    fi

    if [ -f "$BENCHMARK_DIR/${name}.col" ]; then
        local vertices=$(grep "^p edge" "$BENCHMARK_DIR/${name}.col" | awk '{print $3}')
        local edges=$(grep "^p edge" "$BENCHMARK_DIR/${name}.col" | awk '{print $4}')
        echo "    ✓ Downloaded: $vertices vertices, $edges edges"
    else
        echo "    ✗ Extraction failed"
        return 1
    fi
}

echo "Downloading MEDIUM benchmarks (100-250 vertices):"
echo "──────────────────────────────────────────────────"
download_benchmark "dsjc125.1" "5"
download_benchmark "r250.5" "65"
download_benchmark "dsjc250.5" "28"

echo ""
echo "Downloading LARGE benchmarks (500+ vertices, HARD):"
echo "────────────────────────────────────────────────────"
download_benchmark "dsjc500.1" "12"
download_benchmark "dsjc500.5" "48"
download_benchmark "dsjc500.9" "126"
download_benchmark "dsjr500.1c" "85"
download_benchmark "dsjr500.5" "122"

echo ""
echo "Downloading EXTRA LARGE benchmarks (1000+ vertices):"
echo "─────────────────────────────────────────────────────"
download_benchmark "dsjc1000.1" "20"
download_benchmark "dsjc1000.5" "83"
download_benchmark "r1000.1c" "98"
download_benchmark "r1000.5" "234"

echo ""
echo "Downloading FLAT and LE benchmarks:"
echo "────────────────────────────────────"
download_benchmark "flat300_28_0" "31"
download_benchmark "le450_25c" "25"
download_benchmark "flat1000_50_0" "50"


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
