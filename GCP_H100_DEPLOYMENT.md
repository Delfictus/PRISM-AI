# GCP H100 Deployment Guide

**Purpose:** Deploy PRISM-AI to GCP H100 instance for official benchmark validation
**Target GPU:** NVIDIA H100 (80GB, SM 9.0)
**Container:** Docker with CUDA 12.3
**Goal:** Run official DIMACS benchmarks at maximum performance

---

## Prerequisites

### GCP Setup

**Required:**
- GCP account with billing enabled
- Project with Compute Engine API enabled
- Access to H100 GPUs (quota approval may be needed)
- gcloud CLI installed locally

**Check quotas:**
```bash
gcloud compute project-info describe --project=YOUR_PROJECT_ID
# Look for: nvidia-h100 GPU quota
```

**Request quota increase if needed:**
- Navigate to: IAM & Admin → Quotas
- Filter: "GPUs (all regions)" and "NVIDIA H100"
- Request increase to at least 1 GPU

---

## Step 1: Build Docker Container Locally (Optional Testing)

**Build the container:**
```bash
cd /home/diddy/Desktop/PRISM-AI

# Build H100-specific benchmark container
docker build -f Dockerfile.h100-benchmark -t prism-ai-h100-benchmark:latest .

# Test locally (if you have GPU)
docker run --gpus all prism-ai-h100-benchmark:latest benchmark

# Or get a shell
docker run --gpus all -it prism-ai-h100-benchmark:latest shell
```

**Expected build time:** 20-30 minutes (includes Rust compilation)

---

## Step 2: Push to Google Container Registry

**Configure Docker for GCR:**
```bash
# Authenticate
gcloud auth configure-docker

# Tag for GCR
PROJECT_ID="your-gcp-project-id"
docker tag prism-ai-h100-benchmark:latest \
    gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest

# Push to GCR
docker push gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest
```

**Alternative: Build on GCP Cloud Build (Recommended)**
```bash
# Submit build to Cloud Build (builds in GCP)
gcloud builds submit --config cloudbuild.yaml --timeout=1h

# cloudbuild.yaml (create this):
```
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'Dockerfile.h100-benchmark'
      - '-t'
      - 'gcr.io/$PROJECT_ID/prism-ai-h100-benchmark:latest'
      - '.'
timeout: 3600s
images:
  - 'gcr.io/$PROJECT_ID/prism-ai-h100-benchmark:latest'
options:
  machineType: 'N1_HIGHCPU_32'  # Fast build machine
```

---

## Step 3: Create H100 VM Instance

**Option A: Using gcloud CLI (Recommended)**

```bash
# Set variables
PROJECT_ID="your-gcp-project-id"
ZONE="us-central1-a"  # H100 availability varies by zone
INSTANCE_NAME="prism-ai-h100-benchmark"

# Create H100 instance
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
    --metadata=install-nvidia-driver=True

# Wait for instance to be ready (2-5 minutes)
gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE}
```

**H100 Instance Types:**
- `a3-highgpu-1g` - 1x H100 80GB (26 vCPUs, 208GB RAM)
- `a3-highgpu-2g` - 2x H100 80GB
- `a3-highgpu-4g` - 4x H100 80GB
- `a3-highgpu-8g` - 8x H100 80GB

**Cost:** ~$3-4/hour for 1x H100

**Option B: Using GCP Console**
1. Navigate to Compute Engine → VM Instances
2. Create Instance
3. Machine type: a3-highgpu-1g
4. GPU: 1x NVIDIA H100 80GB
5. Boot disk: Ubuntu 22.04 LTS, 200GB
6. Check: "Install NVIDIA GPU driver"

---

## Step 4: Setup VM and Install Docker

**SSH into instance:**
```bash
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}
```

**On the VM, install Docker with NVIDIA runtime:**
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

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

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

**Expected output:** NVIDIA H100 80GB detected

---

## Step 5: Run Benchmarks on H100

**Pull and run container:**
```bash
# Pull from GCR
docker pull gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest

# Run benchmarks
docker run --gpus all \
    --name prism-benchmark-run \
    gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest benchmark

# Or run interactively
docker run --gpus all -it \
    gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest shell

# Inside container:
cd /prism-ai
cargo run --release --features cuda --example world_record_dashboard
```

**Copy results out of container:**
```bash
# Copy logs from container
docker cp prism-benchmark-run:/tmp/test_full_gpu.log ./
docker cp prism-benchmark-run:/tmp/world_record.log ./
docker cp prism-benchmark-run:/tmp/mtx_parser.log ./

# Or mount volume for results
docker run --gpus all -v $(pwd)/results:/results \
    gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark:latest benchmark
```

---

## Step 6: Run Official DIMACS Benchmarks

**Inside container or VM:**

```bash
cd /prism-ai

# Test MTX parser on official benchmark
cargo run --release --features cuda --example test_mtx_parser

# Expected:
# - Loads DSJC500-5 in <5ms
# - Shows: 500 vertices, 125,248 edges
# - Ready to run solver

# Run solver on DSJC500-5 (TODO: need to create benchmark runner)
# For now, test with world_record_dashboard
cargo run --release --features cuda --example world_record_dashboard

# Collect results
grep "Total Latency" /tmp/*.log
grep "Speedup" /tmp/*.log
```

---

## Performance Expectations

### On H100 vs Current Hardware

**Current (RTX 5070 or similar):**
- Pipeline: 4.07ms
- Policy evaluation: 1.04ms
- Neuromorphic: 0.131ms

**Expected on H100:**
- Pipeline: 1-2ms (2-4x faster due to higher compute, memory bandwidth)
- Policy evaluation: 0.3-0.5ms (2-3x faster)
- Neuromorphic: 0.05-0.1ms (2-3x faster)
- **Total speedup: ~200-300x from original baseline**

**H100 Advantages:**
- 3x faster FP64 vs consumer GPUs
- 3TB/s memory bandwidth (vs 600GB/s)
- 80GB HBM3 memory (vs 8-12GB)
- Can handle C4000.5 (4000 vertices) easily

---

## Benchmark Strategy on H100

### Priority Order

**1. DSJC500.5 (Warm-up)**
- 500 vertices, should be fast
- Target: Beat 47 colors
- Verify: Solution correctness
- Estimated: <2ms on H100

**2. DSJC1000.5 (Main Challenge)**
- 1000 vertices, best known: 82-83
- Target: Beat 82 colors
- This is the main validation target
- Estimated: <5ms on H100

**3. C2000.5 (Scalability)**
- 2000 vertices, best: 145
- Tests scalability
- May push H100 limits
- Estimated: 10-20ms

**4. C4000.5 (Ultimate Challenge)**
- 4000 vertices, best: 259
- Requires H100's 80GB memory
- May not be feasible on smaller GPUs
- Estimated: 50-100ms (if works)

---

## Cost Estimation

### GCP H100 Pricing

**On-Demand:**
- a3-highgpu-1g: ~$3.67/hour
- For benchmark run: ~1-2 hours
- **Total: ~$7-8**

**Preemptible/Spot:**
- a3-highgpu-1g: ~$1.10/hour
- Risk: May be interrupted
- **Total: ~$2-3**

**Committed Use:**
- 1-year: ~$2.50/hour
- Only if running continuously

**Recommendation:** Use on-demand for initial testing, spot for large batch runs

---

## Deployment Checklist

### Before Deploying

- [x] Dockerfile.h100-benchmark created
- [ ] Test Dockerfile builds locally
- [ ] Push to GCR (Google Container Registry)
- [ ] Verify GCP project has H100 quota
- [ ] Set up billing alerts

### Deployment Steps

- [ ] Create H100 VM instance
- [ ] Install Docker + NVIDIA runtime
- [ ] Pull container from GCR
- [ ] Run benchmark suite
- [ ] Collect results
- [ ] Copy results back to local
- [ ] Shut down instance (save costs!)

### After Benchmarks

- [ ] Analyze results vs best known
- [ ] Verify solution correctness
- [ ] Document speedups
- [ ] Compare H100 vs local GPU performance
- [ ] Update world-record claims with H100 results

---

## Troubleshooting

### Issue: GPU Not Detected

```bash
# Check driver installation
nvidia-smi

# If not working, install manually:
sudo apt-get install -y nvidia-driver-535
sudo reboot

# Verify after reboot
nvidia-smi
```

### Issue: Docker Can't Access GPU

```bash
# Reinstall NVIDIA container toolkit
sudo apt-get purge -y nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.3.1-base nvidia-smi
```

### Issue: Build Fails in Container

```bash
# Check CUDA version compatibility
nvcc --version  # Should be 12.3.x

# Check Rust version
rustc --version  # Should be 1.75+

# Build with verbose output
cargo build --release --features cuda --verbose
```

---

## Quick Start Commands

**Full deployment in one session:**

```bash
# 1. Build container locally
docker build -f Dockerfile.h100-benchmark -t prism-ai-h100-benchmark .

# 2. Push to GCR
PROJECT_ID="your-project-id"
docker tag prism-ai-h100-benchmark gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark
docker push gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark

# 3. Create H100 VM
gcloud compute instances create prism-h100 \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-1g \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --metadata=install-nvidia-driver=True

# 4. SSH and setup
gcloud compute ssh prism-h100 --zone=us-central1-a

# 5. On VM: Install Docker + NVIDIA runtime (see Step 4 above)

# 6. Run benchmarks
docker run --gpus all gcr.io/${PROJECT_ID}/prism-ai-h100-benchmark benchmark

# 7. Copy results
docker cp $(docker ps -lq):/tmp/world_record.log ./

# 8. Cleanup
exit
gcloud compute instances delete prism-h100 --zone=us-central1-a
```

**Total time:** 30-60 minutes
**Total cost:** $3-8

---

## Expected Results on H100

**Predicted Performance:**
- System baseline: 1-2ms (vs 4ms on RTX 5070)
- DSJC500.5: <2ms
- DSJC1000.5: <5ms
- C2000.5: 10-20ms
- C4000.5: 50-100ms (if memory allows)

**Speedup vs Baselines:**
- If we beat best-known colorings: WORLD-RECORD
- If we match: Competitive performance
- Even if worse: Sub-10ms on 1000+ vertex graphs is excellent

---

## Next Steps After H100 Results

### If Results Are Excellent (< Best Known Colors)

1. Document official results
2. Write academic paper
3. Submit to DIMACS organization
4. Publish in journal
5. Claim world-record status

### If Results Are Good (Close to Best Known)

1. Document competitive performance
2. Write conference paper
3. Present approach and results
4. Publish as novel method

### If Results Need Work

1. Analyze where system struggles
2. Optimize for specific graph types
3. Iterate and improve
4. Focus on strengths

---

## Files

**Dockerfile:** `Dockerfile.h100-benchmark`
**Deployment Guide:** This file (`GCP_H100_DEPLOYMENT.md`)
**Benchmark Results:** Will be in `/tmp/*.log` in container

---

**Status:** Ready to build and deploy
**Next:** Build container and test locally, or deploy directly to GCP
**Estimated Time:** 1-2 hours for full deployment and benchmark run
**Estimated Cost:** $5-10 for H100 runtime
