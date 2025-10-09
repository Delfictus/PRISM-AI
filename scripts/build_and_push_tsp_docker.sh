#!/bin/bash
# Build and push PRISM-AI TSP H100 Docker image to Docker Hub

set -e

# Configuration
IMAGE_NAME="prism-ai-tsp-h100"
DOCKER_USERNAME="${DOCKER_USERNAME:-delfictus}"
VERSION="1.0.0"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
LATEST_TAG="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PRISM-AI TSP H100 Docker Build & Push                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if logged in to Docker Hub
echo "🔐 Checking Docker Hub authentication..."
if ! docker info | grep -q "Username"; then
    echo "⚠️  Not logged in to Docker Hub"
    echo "📝 Please run: docker login"
    exit 1
fi

echo "✅ Authenticated to Docker Hub"
echo ""

# Build Docker image
echo "🔨 Building Docker image..."
echo "  Image: $FULL_IMAGE_NAME"
echo "  Platform: linux/amd64 (H100 compatible)"
echo ""

cd "$(dirname "$0")/.."

docker build \
    --platform linux/amd64 \
    -f Dockerfile.tsp-h100 \
    -t "$FULL_IMAGE_NAME" \
    -t "$LATEST_TAG" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo ""
echo "✅ Docker image built successfully"
echo ""

# Show image size
echo "📦 Image information:"
docker images "$DOCKER_USERNAME/$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo ""

# Test the image (basic smoke test)
echo "🧪 Running smoke test..."
if docker run --rm "$FULL_IMAGE_NAME" echo "Container starts successfully"; then
    echo "✅ Smoke test passed"
else
    echo "❌ Smoke test failed"
    exit 1
fi
echo ""

# Push to Docker Hub
echo "📤 Pushing to Docker Hub..."
echo "  Pushing: $FULL_IMAGE_NAME"
docker push "$FULL_IMAGE_NAME"

echo "  Pushing: $LATEST_TAG"
docker push "$LATEST_TAG"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ SUCCESS - Image pushed to Docker Hub                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Pull commands:"
echo "  docker pull $FULL_IMAGE_NAME"
echo "  docker pull $LATEST_TAG"
echo ""
echo "🚀 RunPod usage:"
echo "  1. Create new pod on RunPod with H100 GPU"
echo "  2. Use Docker image: $LATEST_TAG"
echo "  3. Set environment variables:"
echo "     NUM_CITIES=10000      # Number of cities to solve"
echo "     MAX_ITER=5000         # Max optimization iterations"
echo "  4. Mount volume to /output for results"
echo ""
echo "🎯 Run locally with GPU:"
echo "  docker run --gpus all -e NUM_CITIES=1000 -e MAX_ITER=1000 \\"
echo "    -v \$(pwd)/output:/output $LATEST_TAG"
echo ""
echo "🎮 Full run (85,900 cities):"
echo "  docker run --gpus all -e NUM_CITIES=85900 -e MAX_ITER=10000 \\"
echo "    -v \$(pwd)/output:/output $LATEST_TAG"
echo ""
