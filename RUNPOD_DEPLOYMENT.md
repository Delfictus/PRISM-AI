# RunPod Deployment Guide

**Alternative:** RunPod.io for H100/A100 benchmarking
**Advantage:** Better GPU availability, easier setup, pay-per-minute
**Cost:** ~$2-4/hour for H100, ~$1-2/hour for A100

---

## Option 1: RunPod with Docker Hub (Easiest)

### Step 1: Push Image to Docker Hub

```bash
# Tag for Docker Hub
docker tag prism-ai-h100-benchmark:latest YOUR_DOCKERHUB_USERNAME/prism-ai-h100-benchmark:latest

# Login to Docker Hub
docker login

# Push
docker push YOUR_DOCKERHUB_USERNAME/prism-ai-h100-benchmark:latest
```

### Step 2: Deploy on RunPod

1. Go to https://www.runpod.io/
2. Sign up/Login
3. Click "Deploy"
4. Select GPU: H100 80GB or A100 80GB
5. Container Settings:
   - **Container Image:** `YOUR_DOCKERHUB_USERNAME/prism-ai-h100-benchmark:latest`
   - **Docker Command:** `benchmark`
   - **Expose HTTP Ports:** None (optional)
6. Deploy

### Step 3: Access and Monitor

- **Console access:** Available in RunPod web interface
- **Logs:** Streamed in real-time
- **Results:** Will be in `/tmp/*.log` inside container

---

## Option 2: RunPod with Public GitHub Container Registry

### Push to GHCR (Alternative to Docker Hub)

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Tag for GHCR
docker tag prism-ai-h100-benchmark:latest ghcr.io/YOUR_GITHUB_USERNAME/prism-ai-h100-benchmark:latest

# Push
docker push ghcr.io/YOUR_GITHUB_USERNAME/prism-ai-h100-benchmark:latest
```

Then use `ghcr.io/YOUR_GITHUB_USERNAME/prism-ai-h100-benchmark:latest` in RunPod.

---

## Option 3: RunPod with Manual Setup

**If you don't want to push image publicly:**

1. **Deploy base Ubuntu + CUDA on RunPod**
   - GPU: H100 or A100
   - Image: `nvidia/cuda:12.3.1-devel-ubuntu22.04`

2. **SSH into RunPod instance**

3. **Clone repo and build:**
```bash
git clone https://github.com/Delfictus/PRISM-AI.git
cd PRISM-AI
cargo build --release --features cuda
```

4. **Run benchmarks:**
```bash
cargo run --release --features cuda --example test_full_gpu
cargo run --release --features cuda --example world_record_dashboard
cargo run --release --features cuda --example test_mtx_parser
```

---

## RunPod Advantages

**vs GCP:**
- ✅ Better H100/A100 availability
- ✅ Easier setup (no IAM/permission issues)
- ✅ Pay per minute (not hour)
- ✅ Instant start
- ✅ Web-based console access

**Pricing:**
- H100 80GB: ~$2.49/hour spot, ~$3.99/hour on-demand
- A100 80GB: ~$1.29/hour spot, ~$2.29/hour on-demand
- Billed per minute

**For 30-minute benchmark run:**
- H100: ~$1-2
- A100: ~$0.50-1

**Much cheaper than GCP for short runs!**

---

## Quick Start Commands

**Push to Docker Hub:**
```bash
docker tag prism-ai-h100-benchmark:latest YOUR_USERNAME/prism-ai-h100:latest
docker login
docker push YOUR_USERNAME/prism-ai-h100:latest
```

**Deploy on RunPod:**
1. https://www.runpod.io/console/pods
2. Deploy → Select GPU (H100 or A100)
3. Container: YOUR_USERNAME/prism-ai-h100:latest
4. Command: benchmark
5. Deploy

**Monitor:**
- Web console shows real-time output
- Logs show all benchmark results
- Can SSH in to check /tmp/*.log files

---

## Expected Results on RunPod

**H100:**
- Baseline: 1-2ms (vs 4ms current)
- Speedup: 2-4x faster
- Total: 200-300x vs original

**A100:**
- Baseline: 2-3ms (vs 4ms current)
- Speedup: 1.5-2x faster
- Total: 100-140x vs original

**Both excellent for world-record validation!**

---

**Status:** Ready to deploy to RunPod
**Next:** Push image to Docker Hub or GHCR, then deploy on RunPod
