#!/bin/bash
# Automatic launcher for 8× H200 SXM parallel TSP benchmark

set -e

IMAGE="delfictus/prism-ai-tsp-h100:h200"
TOTAL_CITIES=85900
NUM_GPUS=8
CITIES_PER_GPU=$((TOTAL_CITIES / NUM_GPUS))

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PRISM-AI 8× H200 SXM Parallel Launcher                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 Configuration:"
echo "  Image:           $IMAGE"
echo "  Total cities:    $TOTAL_CITIES"
echo "  GPUs:            $NUM_GPUS"
echo "  Cities per GPU:  $CITIES_PER_GPU"
echo ""

# Create output directories
mkdir -p output
for GPU in {0..7}; do
  mkdir -p output/gpu$GPU
done

echo "🚀 Launching containers on 8× H200 GPUs..."
echo ""

# Launch one container per GPU
for GPU in {0..7}; do
  START=$((GPU * CITIES_PER_GPU))
  NAME="prism-tsp-h200-gpu$GPU"

  echo "  GPU $GPU: Cities $START-$((START + CITIES_PER_GPU - 1))"

  docker run -d \
    --name $NAME \
    --gpus "device=$GPU" \
    -e NUM_CITIES=$CITIES_PER_GPU \
    -e RUST_LOG=info \
    -v $(pwd)/output/gpu$GPU:/output \
    $IMAGE

  sleep 0.5  # Brief delay between launches
done

echo ""
echo "✅ All 8 containers launched!"
echo ""
echo "📊 Monitor progress:"
echo "  docker logs -f prism-tsp-h200-gpu0"
echo "  docker logs -f prism-tsp-h200-gpu1"
echo "  ... (gpu0-gpu7)"
echo ""
echo "📈 Check all container status:"
echo "  docker ps | grep prism-tsp-h200"
echo ""
echo "🎮 Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "📁 Results will be in:"
echo "  output/gpu0/benchmark.log"
echo "  output/gpu1/benchmark.log"
echo "  ... (through gpu7)"
echo ""
echo "⏱️  Expected completion: ~10 minutes"
echo ""
echo "🛑 Stop all containers:"
echo "  docker stop \$(docker ps -q --filter name=prism-tsp-h200)"
echo ""
