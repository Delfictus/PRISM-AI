#!/bin/bash
# Flexible H100 deployment - tries multiple zones and configurations
# Project: aresedge-engine

set -e

PROJECT_ID="aresedge-engine"
INSTANCE_NAME="prism-h100-benchmark"
IMAGE_NAME="gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest"

echo "==================================================================="
echo "PRISM-AI H100 Flexible Deployment"
echo "==================================================================="
echo ""

# Try different zones in order
ZONES=("us-central1-b" "us-central1-a" "us-central1-c" "us-east1-c" "us-east4-c")

for ZONE in "${ZONES[@]}"; do
    echo "Trying zone: ${ZONE}..."

    # Try creating instance
    if gcloud compute instances create ${INSTANCE_NAME} \
        --project=${PROJECT_ID} \
        --zone=${ZONE} \
        --machine-type=a3-highgpu-1g \
        --accelerator=type=nvidia-h100-80gb,count=1 \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=200GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --maintenance-policy=TERMINATE \
        --metadata=startup-script='#!/bin/bash
set -e
echo "Installing NVIDIA drivers and Docker..."
apt-get update
apt-get install -y nvidia-driver-535 docker.io

# Install NVIDIA container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list | \
    sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Authenticate and pull
gcloud auth configure-docker gcr.io --quiet
docker pull gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest

# Run benchmarks
docker run --gpus all --name prism-benchmark gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest benchmark > /var/log/prism-benchmark.log 2>&1
echo "Benchmarks complete!"
' 2>/dev/null; then
        echo ""
        echo "SUCCESS! Instance created in ${ZONE}"
        echo "Instance: ${INSTANCE_NAME}"
        echo "Zone: ${ZONE}"
        echo ""
        echo "Monitor with:"
        echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='tail -f /var/log/syslog'"
        echo ""
        echo "Get results:"
        echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='cat /var/log/prism-benchmark.log'"
        echo ""
        exit 0
    else
        echo "Zone ${ZONE} unavailable, trying next..."
        echo ""
    fi
done

echo "ERROR: No H100 availability in any zone. Try again later or request quota increase."
exit 1
