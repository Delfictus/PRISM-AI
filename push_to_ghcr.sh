#!/bin/bash
# Push PRISM-AI H100 benchmark image to GitHub Container Registry
# For RunPod deployment

set -e

IMAGE_NAME="ghcr.io/delfictus/prism-ai-h100-benchmark:latest"

echo "==================================================================="
echo "Pushing PRISM-AI to GitHub Container Registry"
echo "==================================================================="
echo "Image: ${IMAGE_NAME}"
echo ""

# Check if already tagged
if ! docker images | grep -q "ghcr.io/delfictus/prism-ai-h100-benchmark"; then
    echo "Tagging image..."
    docker tag prism-ai-h100-benchmark:latest ${IMAGE_NAME}
fi

echo "Pushing to GHCR..."
echo "Note: You need to be logged in to GHCR first:"
echo "  echo \$GITHUB_TOKEN | docker login ghcr.io -u Delfictus --password-stdin"
echo ""

docker push ${IMAGE_NAME}

echo ""
echo "==================================================================="
echo "Image pushed successfully!"
echo "==================================================================="
echo ""
echo "To use on RunPod:"
echo "1. Go to https://www.runpod.io/console/pods"
echo "2. Deploy → Select H100 or A100"
echo "3. Container Image: ${IMAGE_NAME}"
echo "4. Docker Command: benchmark"
echo "5. Deploy"
echo ""
echo "Image is public, no authentication needed on RunPod"
