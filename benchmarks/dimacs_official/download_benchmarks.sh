#!/bin/bash
# Download official DIMACS graph coloring benchmarks
# Source: Network Repository (nrvis.com)
# Priority order for world-record validation

echo "Downloading DIMACS official benchmarks..."
echo "Source: https://nrvis.com/download/data/dimacs/"

# Priority 1 - DSJC1000.5 (1000 vertices, best known: 82-83 colors)
echo "=== Priority 1: DSJC1000.5 ==="
wget -O DSJC1000-5.zip https://nrvis.com/download/data/dimacs/DSJC1000-5.zip
unzip -o DSJC1000-5.zip

# Priority 2 - DSJC500.5 (500 vertices, best known: 47-48 colors)
echo "=== Priority 2: DSJC500.5 ==="
wget -O DSJC500-5.zip https://nrvis.com/download/data/dimacs/DSJC500-5.zip
unzip -o DSJC500-5.zip

# Priority 3 - DSJC1000.9 (1000 vertices, very dense, best: 222-223 colors)
echo "=== Priority 3: DSJC1000.9 ==="
wget -O DSJC1000-9.zip https://nrvis.com/download/data/dimacs/DSJC1000-9.zip
unzip -o DSJC1000-9.zip

# Priority 4 - C2000.5 (2000 vertices, best: 145 colors)
echo "=== Priority 4: C2000.5 ==="
wget -O C2000-5.zip https://nrvis.com/download/data/dimacs/C2000-5.zip
unzip -o C2000-5.zip

# Priority 5 - C4000.5 (4000 vertices - THE BIG ONE, best: 259 colors)
echo "=== Priority 5: C4000.5 ==="
wget -O C4000-5.zip https://nrvis.com/download/data/dimacs/C4000-5.zip
unzip -o C4000-5.zip

# Priority 6 - flat1000_76_0 (known optimal: 76, best found: 81-82)
echo "=== Priority 6: flat1000_76_0 ==="
wget -O flat1000-76-0.zip https://nrvis.com/download/data/dimacs/flat1000-76-0.zip
unzip -o flat1000-76-0.zip

# Priority 7 - C2000.9 (2000 vertices, very dense, best: 400 colors)
echo "=== Priority 7: C2000.9 ==="
wget -O C2000-9.zip https://nrvis.com/download/data/dimacs/C2000-9.zip
unzip -o C2000-9.zip

# Validation benchmarks
echo "=== Validation: DSJC250.5 ==="
wget -O DSJC250-5.zip https://nrvis.com/download/data/dimacs/DSJC250-5.zip
unzip -o DSJC250-5.zip

echo "=== Validation: DSJC125.5 ==="
wget -O DSJC125-5.zip https://nrvis.com/download/data/dimacs/DSJC125-5.zip
unzip -o DSJC125-5.zip

echo "=== Validation: R1000.5 ==="
wget -O R1000-5.zip https://nrvis.com/download/data/dimacs/R1000-5.zip
unzip -o R1000-5.zip

# Recently solved
echo "=== Recently Solved: r1000.1c ==="
wget -O r1000-1c.zip https://nrvis.com/download/data/dimacs/r1000-1c.zip
unzip -o r1000-1c.zip

echo ""
echo "Download complete!"
echo "Files downloaded:"
ls -lh *.col 2>/dev/null | wc -l
echo ""
echo "Organizing by priority..."
