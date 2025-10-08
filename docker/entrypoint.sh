#!/bin/bash
set -e

echo "PRISM-AI Container Starting..."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not detected')"
echo ""

# Default: Run benchmarks
if [ "$1" = "benchmark" ]; then
    exec /usr/local/bin/run_benchmarks.sh
elif [ "$1" = "shell" ]; then
    exec /bin/bash
else
    exec "$@"
fi
