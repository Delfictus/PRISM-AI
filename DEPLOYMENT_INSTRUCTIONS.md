# H100 Deployment Instructions - Ready to Deploy

**Status:** ✅ Docker image built and ready
**Image:** `prism-ai-h100-benchmark:latest` (12.4GB)
**Tagged for GCR:** `gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest`
**Project:** aresedge-engine

---

## ⚠️ gcloud CLI Not Installed on This Machine

**You'll need to deploy from a machine with `gcloud` CLI installed.**

**Option 1: Install gcloud CLI Here**
```bash
# Install gcloud
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
gcloud auth configure-docker

# Then run deployment
./deploy_to_gcp_h100.sh
```

**Option 2: Save Image and Deploy Elsewhere**
```bash
# Save Docker image to tar
docker save prism-ai-h100-benchmark:latest | gzip > prism-ai-h100.tar.gz

# Transfer to machine with gcloud CLI
# Then: docker load < prism-ai-h100.tar.gz
# Then: ./deploy_to_gcp_h100.sh
```

**Option 3: Manual Deployment (Detailed Below)**

---

## Manual Deployment Steps

### Step 1: Push Image to GCR

**On a machine with gcloud CLI:**

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project aresedge-engine
gcloud auth configure-docker

# Push image
docker push gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest
```

---

### Step 2: Create H100 VM Instance

**Using gcloud CLI:**

```bash
# Create instance
gcloud compute instances create prism-ai-h100-benchmark \
    --project=aresedge-engine \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-1g \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=install-nvidia-driver=True

# Wait 2-5 minutes for driver installation
```

**Or using GCP Console:**
1. Go to: https://console.cloud.google.com/compute/instances
2. Create Instance
3. Name: prism-ai-h100-benchmark
4. Region: us-central1, Zone: us-central1-a
5. Machine type: a3-highgpu-1g
6. GPUs: 1x NVIDIA H100 80GB
7. Boot disk: Ubuntu 22.04 LTS, 200GB
8. Check: "Install NVIDIA GPU driver automatically"
9. Create

**Cost:** ~$3.67/hour

---

### Step 3: SSH and Setup Docker

**SSH to instance:**
```bash
gcloud compute ssh prism-ai-h100-benchmark --zone=us-central1-a
```

**On the VM, install Docker:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
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

# Verify GPU
sudo docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

**Expected:** Should show NVIDIA H100 80GB

---

### Step 4: Pull and Run Container

**On the VM:**

```bash
# Pull from GCR
sudo docker pull gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest

# Run benchmarks
sudo docker run --gpus all \
    --name prism-benchmark \
    gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest benchmark
```

**This will automatically:**
1. Test system validation (test_full_gpu)
2. Run world record dashboard (4 scenarios)
3. Test MTX parser on DSJC500-5
4. Save results to /tmp/*.log

**Expected time:** 5-10 minutes

---

### Step 5: Collect Results

**Copy results from container:**

```bash
# Extract logs from container
sudo docker cp prism-benchmark:/tmp/test_full_gpu.log ./
sudo docker cp prism-benchmark:/tmp/world_record.log ./
sudo docker cp prism-benchmark:/tmp/mtx_parser.log ./

# View results
cat test_full_gpu.log | grep "Total Latency"
cat world_record.log | grep "vs World Record"
```

**Expected H100 Performance:**
- Baseline: 1-2ms (vs 4ms on RTX 5070)
- Speedup: 2-4x faster than current
- Total speedup vs original: 200-300x

**Copy back to local machine:**
```bash
# From your local machine
gcloud compute scp prism-ai-h100-benchmark:/home/YOUR_USERNAME/*.log ./h100_results/ --zone=us-central1-a
```

---

### Step 6: Cleanup (IMPORTANT!)

**Delete instance to stop charges:**

```bash
# From local machine
gcloud compute instances delete prism-ai-h100-benchmark \
    --zone=us-central1-a \
    --project=aresedge-engine
```

**Or from GCP Console:**
1. Go to Compute Engine → VM Instances
2. Select prism-ai-h100-benchmark
3. Delete

**⚠️ Don't forget this step! H100 costs $3-4/hour**

---

## Quick Reference

**Image Info:**
- Local tag: `prism-ai-h100-benchmark:latest`
- GCR tag: `gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest`
- Size: 12.4GB
- Built: 2025-10-08

**GCP Info:**
- Project: aresedge-engine
- Zone: us-central1-a (or check availability)
- Machine: a3-highgpu-1g (1x H100 80GB)
- Cost: ~$3.67/hour on-demand

**Commands Summary:**
```bash
# Push image
docker push gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest

# Create VM
gcloud compute instances create prism-ai-h100-benchmark --machine-type=a3-highgpu-1g ...

# Run benchmarks
sudo docker run --gpus all gcr.io/aresedge-engine/prism-ai-h100-benchmark:latest benchmark

# Delete VM
gcloud compute instances delete prism-ai-h100-benchmark --zone=us-central1-a
```

---

## What Happens When You Deploy

**Benchmarks that will run:**
1. System validation (4.07ms baseline → expect 1-2ms on H100)
2. World record dashboard (4 scenarios)
3. MTX parser test (DSJC500-5 loading)

**Results you'll get:**
- Total pipeline latency on H100
- Speedup vs world records (updated for H100)
- Verification that all systems work on H100
- Official benchmark readiness confirmation

**Next after results:**
- Analyze H100 performance vs RTX 5070
- Update world-record claims with H100 numbers
- Run official DIMACS instances (DSJC500.5, DSJC1000.5, etc.)

---

## Status

✅ Docker image: Built (12.4GB)
✅ GCR tagged: Ready to push
✅ Deployment script: Ready
✅ Instructions: Complete
⏸️ Waiting: gcloud CLI for automated deployment

**Next:** Install gcloud CLI or deploy manually from machine that has it

---

**See also:**
- `GCP_H100_DEPLOYMENT.md` - Detailed deployment guide
- `deploy_to_gcp_h100.sh` - Automated deployment script
- All committed and pushed (commit: 410f9ff)
