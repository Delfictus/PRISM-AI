#!/bin/bash
# Deploy PRISM-AI to GCP H100 Instance
# Project: aresedge-engine

set -e

PROJECT_ID="aresedge-engine"
ZONE="us-central1-a"  # Change if H100s are in different zone
INSTANCE_NAME="prism-ai-h100-benchmark"
IMAGE_NAME="gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest"

echo "==================================================================="
echo "PRISM-AI H100 Deployment to GCP"
echo "==================================================================="
echo "Project: ${PROJECT_ID}"
echo "Zone: ${ZONE}"
echo "Instance: ${INSTANCE_NAME}"
echo ""

# Step 1: Configure Docker for GCR
echo "=== Step 1: Configure Docker for Google Container Registry ==="
gcloud auth configure-docker
echo ""

# Step 2: Tag and push image
echo "=== Step 2: Push Docker image to GCR ==="
docker tag prism-ai-h100-benchmark:latest ${IMAGE_NAME}
docker push ${IMAGE_NAME}
echo ""

# Step 3: Check H100 availability
echo "=== Step 3: Check H100 GPU Availability ==="
echo "Checking quota for nvidia-h100 in ${ZONE}..."
gcloud compute project-info describe --project=${PROJECT_ID} | grep -i "h100" || echo "No H100 quota info found"
echo ""

# Step 4: Create H100 instance
echo "=== Step 4: Create H100 VM Instance ==="
read -p "Create H100 instance? This will cost ~\$3-4/hour. (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
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
        --provisioning-model=STANDARD \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata=install-nvidia-driver=True \
        --tags=prism-ai-benchmark

    echo ""
    echo "Instance created. Waiting for GPU driver installation (2-5 minutes)..."
    sleep 120
    echo ""
fi

# Step 5: Setup Docker on VM
echo "=== Step 5: Setup Docker and NVIDIA Container Toolkit on VM ==="
read -p "SSH and setup Docker? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Create setup script
    cat > /tmp/setup_docker.sh << 'SETUP'
#!/bin/bash
set -e

echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "Verifying GPU access..."
sudo docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi

echo "Docker setup complete!"
SETUP

    # Upload and execute setup script
    gcloud compute scp /tmp/setup_docker.sh ${INSTANCE_NAME}:~/ --zone=${ZONE}
    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="bash ~/setup_docker.sh"

    echo ""
    echo "Docker setup complete!"
    echo ""
fi

# Step 6: Run benchmarks
echo "=== Step 6: Run Benchmarks on H100 ==="
read -p "Run benchmarks now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
        sudo docker pull ${IMAGE_NAME} && \
        sudo docker run --gpus all --name prism-benchmark ${IMAGE_NAME} benchmark
    "

    echo ""
    echo "Benchmarks complete!"
    echo ""

    # Step 7: Copy results
    echo "=== Step 7: Copy Results ==="
    read -p "Copy results to local machine? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        mkdir -p ./h100_results
        gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
            sudo docker cp prism-benchmark:/tmp/test_full_gpu.log /tmp/ && \
            sudo docker cp prism-benchmark:/tmp/world_record.log /tmp/ && \
            sudo docker cp prism-benchmark:/tmp/mtx_parser.log /tmp/
        "
        gcloud compute scp ${INSTANCE_NAME}:/tmp/*.log ./h100_results/ --zone=${ZONE}

        echo ""
        echo "Results copied to ./h100_results/"
        ls -lh ./h100_results/
        echo ""
    fi
fi

# Step 8: Cleanup
echo "=== Step 8: Cleanup ==="
echo "IMPORTANT: H100 instance costs ~\$3-4/hour"
echo "Current instance: ${INSTANCE_NAME}"
echo ""
read -p "Delete H100 instance to stop charges? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --quiet
    echo "Instance deleted. Charges stopped."
else
    echo "Instance still running. Remember to delete when done:"
    echo "  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE}"
fi

echo ""
echo "==================================================================="
echo "Deployment Complete!"
echo "==================================================================="
