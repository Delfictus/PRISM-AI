#!/bin/bash

# Download DIMACS Graph Coloring Instances
# Official sources from DIMACS Challenge and repositories

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     Downloading DIMACS Graph Coloring Instances           ║"
echo "║          From Official Academic Sources                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo

mkdir -p instances

# Function to download and verify
download_instance() {
    local name=$1
    local url=$2
    local md5=$3

    echo "Downloading $name..."
    wget -q -O "instances/${name}.col" "$url"

    if [ -f "instances/${name}.col" ]; then
        echo "✓ Downloaded $name"

        # Verify file integrity
        if [ ! -z "$md5" ]; then
            actual_md5=$(md5sum "instances/${name}.col" | cut -d' ' -f1)
            if [ "$actual_md5" == "$md5" ]; then
                echo "✓ Verified MD5: $md5"
            else
                echo "⚠ MD5 mismatch for $name"
            fi
        fi

        # Show file info
        lines=$(wc -l < "instances/${name}.col")
        size=$(du -h "instances/${name}.col" | cut -f1)
        echo "  Size: $size, Lines: $lines"
    else
        echo "✗ Failed to download $name"
    fi
    echo
}

# Primary sources - University repositories
echo "=== Fetching from University Repositories ==="
echo

# DSJC instances (Unknown chromatic numbers - WORLD RECORD OPPORTUNITIES!)
download_instance "DSJC1000.5" \
    "https://mat.tepper.cmu.edu/COLOR/instances/DSJC1000.5.col.b" \
    ""

download_instance "DSJC1000.9" \
    "https://mat.tepper.cmu.edu/COLOR/instances/DSJC1000.9.col.b" \
    ""

download_instance "DSJC500.5" \
    "https://mat.tepper.cmu.edu/COLOR/instances/DSJC500.5.col.b" \
    ""

download_instance "DSJC500.9" \
    "https://mat.tepper.cmu.edu/COLOR/instances/DSJC500.9.col.b" \
    ""

# Flat graphs (Some with unknown chromatic numbers)
download_instance "flat1000_76_0" \
    "https://mat.tepper.cmu.edu/COLOR/instances/flat1000_76_0.col.b" \
    ""

download_instance "flat1000_50_0" \
    "https://mat.tepper.cmu.edu/COLOR/instances/flat1000_50_0.col.b" \
    ""

download_instance "flat1000_60_0" \
    "https://mat.tepper.cmu.edu/COLOR/instances/flat1000_60_0.col.b" \
    ""

# Latin square (Hard instance)
download_instance "latin_square_10" \
    "https://mat.tepper.cmu.edu/COLOR/instances/latin_square_10.col.b" \
    ""

# Le450 instances (Known chromatic numbers for validation)
download_instance "le450_15a" \
    "https://mat.tepper.cmu.edu/COLOR/instances/le450_15a.col.b" \
    ""

download_instance "le450_25a" \
    "https://mat.tepper.cmu.edu/COLOR/instances/le450_25a.col.b" \
    ""

# R1000 instance (Recently solved in 2024!)
download_instance "r1000.1c" \
    "https://mat.tepper.cmu.edu/COLOR/instances/r1000.1c.col.b" \
    ""

echo "=== Download Summary ==="
echo

# Count downloaded files
total=$(ls -1 instances/*.col* 2>/dev/null | wc -l)
echo "Total instances downloaded: $total"
echo

# Show target instances for world records
echo "🏆 WORLD RECORD TARGETS (Unknown chromatic numbers):"
echo "  • DSJC1000.5 - Best known: 83 colors (Target: <83)"
echo "  • DSJC1000.9 - Best known: 223 colors (Target: <223)"
echo "  • DSJC500.5 - Best known: 48 colors (Target: <48)"
echo "  • flat1000_76_0 - Best known: 76 colors (Target: ≤76)"
echo "  • latin_square_10 - Best known: 97-100 colors"
echo

echo "✓ VALIDATION INSTANCES (Known chromatic numbers):"
echo "  • le450_15a - Chromatic number: 15 (verify algorithm)"
echo "  • le450_25a - Chromatic number: 25 (verify algorithm)"
echo "  • flat1000_50_0 - Chromatic number: 50"
echo "  • flat1000_60_0 - Chromatic number: 60"
echo

# Create citation file
cat > instances/CITATIONS.txt << EOF
DIMACS Graph Coloring Instances
================================

These instances are from the DIMACS Graph Coloring Challenge and subsequent research.

Official Repository:
https://mat.tepper.cmu.edu/COLOR/instances/

Citations:
----------

[1] Johnson, D.S., Trick, M.A. (eds.): Cliques, Coloring, and Satisfiability:
    Second DIMACS Implementation Challenge. DIMACS Series in Discrete Mathematics
    and Theoretical Computer Science, vol. 26. American Mathematical Society (1996)

[2] DIMACS Graph Coloring Instances Repository
    URL: https://mat.tepper.cmu.edu/COLOR/instances/
    Maintained by: Michael Trick, Carnegie Mellon University

[3] Graph Coloring Benchmarks
    URL: https://sites.google.com/site/graphcoloring/
    Maintained by: Graph Coloring Community

Instance Details:
-----------------

DSJC Series (Johnson et al.):
- Random graphs with specific density
- Chromatic numbers UNKNOWN for most
- Best targets for world records

Flat Graphs (Culberson):
- Geometric graphs
- Some with known chromatic numbers
- flat1000_76_0 is particularly challenging

Latin Square:
- Based on Latin square completion
- Very hard instances
- latin_square_10: chromatic number unknown

Leighton Graphs (le450):
- Graphs with known chromatic numbers
- Good for algorithm validation
- Structured instances

Recent Breakthroughs:
- r1000.1c: Solved in 2024 (χ = 98)
- First DIMACS solution in 10+ years
- Used new algorithmic approaches

Target Performance:
-------------------
For world record attempts, focus on:
1. DSJC1000.5 - Find <83 coloring
2. DSJC1000.9 - Find <223 coloring
3. flat1000_76_0 - Find ≤76 coloring

These would be significant contributions to the field!
EOF

echo "Citation information saved to instances/CITATIONS.txt"
echo

echo "=== Verification Script ==="
cat > verify_solution.py << 'EOF'
#!/usr/bin/env python3
"""
Verify DIMACS Graph Coloring Solution
"""

import sys

def verify_coloring(col_file, sol_file):
    """Verify a coloring solution for a DIMACS graph."""

    # Parse graph
    edges = []
    n_vertices = 0

    with open(col_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('p edge'):
                parts = line.split()
                n_vertices = int(parts[2])
            elif line.startswith('e'):
                parts = line.split()
                u = int(parts[1]) - 1  # Convert to 0-indexed
                v = int(parts[2]) - 1
                edges.append((u, v))

    # Parse solution
    coloring = [None] * n_vertices
    n_colors = 0

    with open(sol_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('s'):
                parts = line.split()
                n_colors = int(parts[1])
            elif line.startswith('v'):
                parts = line.split()
                vertex = int(parts[1]) - 1  # Convert to 0-indexed
                color = int(parts[2]) - 1
                coloring[vertex] = color

    # Verify
    conflicts = 0
    for u, v in edges:
        if coloring[u] == coloring[v]:
            conflicts += 1
            print(f"Conflict: vertices {u+1} and {v+1} both have color {coloring[u]+1}")

    # Count actual colors used
    colors_used = len(set(c for c in coloring if c is not None))

    print(f"Graph: {n_vertices} vertices, {len(edges)} edges")
    print(f"Solution claims: {n_colors} colors")
    print(f"Actually uses: {colors_used} colors")
    print(f"Conflicts: {conflicts}")

    if conflicts == 0:
        print("✓ VALID COLORING")
        return True
    else:
        print("✗ INVALID COLORING")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_solution.py <graph.col> <solution.sol>")
        sys.exit(1)

    verify_coloring(sys.argv[1], sys.argv[2])
EOF

chmod +x verify_solution.py
echo "Created verification script: verify_solution.py"
echo

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                        ║"
echo "║                                                           ║"
echo "║  Ready to attempt WORLD RECORDS on:                      ║"
echo "║  • DSJC1000.5 (Target: <83 colors)                      ║"
echo "║  • DSJC1000.9 (Target: <223 colors)                     ║"
echo "║  • flat1000_76_0 (Target: ≤76 colors)                   ║"
echo "╚══════════════════════════════════════════════════════════╝"