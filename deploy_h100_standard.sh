#!/bin/bash
# Deploy PRISM-AI to H100 using standard Ubuntu VM (not deprecated container-optimized)
# Project: aresedge-engine

set -e

PROJECT_ID="aresedge-engine"
ZONE="us-central1-b"  # Zone b has H100 availability
INSTANCE_NAME="prism-h100-benchmark"
IMAGE_NAME="gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest"

echo "==================================================================="
echo "PRISM-AI H100 Deployment (Standard Ubuntu VM)"
echo "==================================================================="
echo "Project: ${PROJECT_ID}"
echo "Zone: ${ZONE}"
echo "Instance: ${INSTANCE_NAME}"
echo "Image: ${IMAGE_NAME}"
echo ""

# Create H100 instance with Ubuntu (not container-optimized)
echo "Creating H100 instance..."
gcloud compute instances create ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=a3-highgpu-1g \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=startup-script='#!/bin/bash
set -e

echo "Installing NVIDIA drivers..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-driver-535 nvidia-container-toolkit docker.io

# Configure Docker for NVIDIA
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Authenticate with GCR
gcloud auth configure-docker gcr.io --quiet

# Pull and run container
echo "Pulling PRISM-AI container..."
docker pull '${IMAGE_NAME}'

echo "Running benchmarks on H100..."
docker run --gpus all --name prism-benchmark '${IMAGE_NAME}' benchmark > /var/log/prism-benchmark.log 2>&1

echo "Benchmarks complete. Results in /var/log/prism-benchmark.log"
'

echo ""
echo "Instance created. Startup script will:"
echo "  1. Install NVIDIA drivers (~5 min)"
echo "  2. Install Docker + NVIDIA toolkit (~2 min)"
echo "  3. Pull container from GCR (~5-10 min)"
echo "  4. Run benchmarks (~5-10 min)"
echo ""
echo "Total time: ~20-30 minutes"
echo ""
echo "Monitor progress:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='tail -f /var/log/syslog'"
echo ""
echo "Get results:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='cat /var/log/prism-benchmark.log'"
echo ""
echo "Cost: ~\$3.67/hour for a3-highgpu-1g (1x H100)"
echo ""
echo "IMPORTANT: Delete instance when done to stop charges:"
echo "  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
