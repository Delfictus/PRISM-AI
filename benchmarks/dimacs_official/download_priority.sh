#!/bin/bash
echo "Downloading highest priority 2025 DIMACS instances..."

# Priority 1: Recently solved (2024)
wget -O r1000-1c.zip https://nrvis.com/download/data/dimacs/r1000-1c.zip
unzip -o r1000-1c.zip

# Priority 2: Known optimal with gap
wget -O flat1000-76-0.zip https://nrvis.com/download/data/dimacs/flat1000-76-0.zip
unzip -o flat1000-76-0.zip

# Priority 3: Dense graph challenge
wget -O DSJC1000-9.zip https://nrvis.com/download/data/dimacs/DSJC1000-9.zip
unzip -o DSJC1000-9.zip

echo "Complete! Priority instances:"
ls -lh *.mtx | tail -5
