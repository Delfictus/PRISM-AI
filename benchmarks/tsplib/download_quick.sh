#!/bin/bash
# Quick download of essential TSPLIB instances from GitHub

BASE_URL="https://raw.githubusercontent.com/mastqe/tsplib/master"

cd /home/is/neuromorphic-quantum-platform-clean/benchmarks/tsplib

echo "Downloading essential TSPLIB instances..."

for instance in eil51 eil76 kroA100 kroB100 rd100 eil101 pr152 kroA200; do
    if [ ! -f "${instance}.tsp" ]; then
        echo "  Downloading ${instance}.tsp..."
        curl -s -L -m 30 "${BASE_URL}/${instance}.tsp" -o "${instance}.tsp" &
    fi
done

wait

echo "Download complete!"
ls -lh *.tsp
