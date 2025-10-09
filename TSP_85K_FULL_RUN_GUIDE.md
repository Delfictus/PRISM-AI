# Running Full 85,900 City pla85900 Benchmark

**Docker Image:** `delfictus/prism-ai-tsp-h100:latest` (v1.0.1)
**Status:** ✅ Updated - Full 85,900 city support enabled

---

## 🎯 Running the Full Benchmark

### Option 1: Use NUM_CITIES Environment Variable

The image now respects the `NUM_CITIES` environment variable:

```bash
# RunPod Configuration
Image: delfictus/prism-ai-tsp-h100:latest

Environment Variables:
  NUM_CITIES=85900    # Full instance!
  RUST_LOG=info
```

### Option 2: Docker Run (Local with GPU)

```bash
docker run --gpus all \
  -e NUM_CITIES=85900 \
  -v $(pwd)/output:/output \
  delfictus/prism-ai-tsp-h100:latest
```

### Option 3: No Environment Variable (Default = Full)

If you DON'T set `NUM_CITIES`, it will use the FULL instance by default:

```bash
docker run --gpus all \
  -v $(pwd)/output:/output \
  delfictus/prism-ai-tsp-h100:latest
```

---

## 📊 Test Progression

### Start Small (Validate)
```
NUM_CITIES=1000
Expected time: ~5 seconds
Cost: ~$0.005
```

### Medium (Build Confidence)
```
NUM_CITIES=10000
Expected time: ~2 minutes
Cost: ~$0.12
```

### Large (Near Full)
```
NUM_CITIES=50000
Expected time: ~30 minutes
Cost: ~$1.75
```

### FULL INSTANCE
```
NUM_CITIES=85900
Expected time: ~60-90 minutes
Cost: ~$3-5
```

---

## ⚠️ Important Notes for Full Run

### GPU Requirements
- **Minimum**: H100 80GB (H100 PCIe or SXM5)
- **Memory needed**: ~60-70 GB VRAM
- **Lower GPUs won't work**: A100 40GB will OOM

### Expected Performance
| Cities | Initialization | Processing | Total | GPU Memory |
|--------|----------------|-----------|-------|------------|
| 1,000 | 0.5s | 4-5s | ~5s | ~100 MB |
| 10,000 | 2s | 120s | ~2min | ~2 GB |
| 50,000 | 10s | 1800s | ~30min | ~50 GB |
| 85,900 | 30s | 5400s | ~90min | ~70 GB |

### Cost on RunPod
- H100 PCIe: ~$3.49/hour
- Full run (~90 min): ~$5.24
- Use **Spot pricing** for 50% discount: ~$2.62

---

## 🔍 What the Benchmark Does

The `honest_tsp_benchmark` doesn't actually solve TSP optimally - it runs your **quantum-inspired pipeline** on TSP data:

1. **Loads TSP coordinates** (all 85,900 cities)
2. **Runs 8-phase GPU pipeline**:
   - Neuromorphic reservoir
   - Transfer entropy
   - Coupling computation
   - Thermodynamic evolution
   - Quantum processing
   - Active inference
   - Control synthesis
   - Synchronization
3. **Measures real GPU performance**
4. **Validates physical constraints** (entropy, free energy)

**This is NOT an optimal TSP solver** - it's a benchmark of your platform's performance on TSP-sized data.

---

## 🎯 For Actual TSP Solving

If you want to actually **solve TSP optimally**, you need a different approach:

### LKH-3 (CPU, Best Quality)
```bash
# Download LKH-3
wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.9.tgz
tar xzf LKH-3.0.9.tgz
cd LKH-3.0.9
make

# Run on pla85900
./LKH pla85900.par
```

**Expected:**
- Time: 6-24 hours on CPU
- Quality: <1% from optimal
- Tour length: ~143-144 million

### Your Platform (GPU, Fast but Different Purpose)
```bash
docker run --gpus all -e NUM_CITIES=85900 delfictus/prism-ai-tsp-h100:latest
```

**Expected:**
- Time: ~90 minutes
- Purpose: Benchmark GPU pipeline performance
- Not optimal TSP tour (different algorithm)

---

## 📌 RunPod Deployment Steps

### Step 1: Create H100 Pod

1. Go to https://runpod.io/console/pods
2. Click "Deploy"
3. Select **"H100 PCIe 80GB"** or **"H100 SXM5 80GB"**

### Step 2: Configure Container

**Container Image:**
```
delfictus/prism-ai-tsp-h100:latest
```

**Environment Variables:**
```
NUM_CITIES=85900
RUST_LOG=info
```

**Volume Mounts:**
```
Container Path: /output
Size: 10 GB
```

**Pod Settings:**
- GPU: H100 80GB
- CPU: 16+ cores recommended
- RAM: 64 GB recommended
- Disk: 50 GB

### Step 3: Launch

Click "Deploy" and wait for pod to start (~30 seconds)

### Step 4: Monitor

**View Logs:**
- Go to your pod
- Click "Logs" tab
- Watch for:
  ```
  ✅ NVIDIA H100 detected!
  ✓ Processing FULL instance: 85900 cities
  ```

**Expected Runtime:**
- Initialization: ~30 seconds
- Processing: ~60-90 minutes
- Total: ~90 minutes

### Step 5: Access Results

**Via RunPod Web UI:**
1. Click on pod
2. Go to "Files"
3. Navigate to `/output/benchmark.log`
4. Download the log file

**Results will include:**
- Execution time for 85,900 cities
- GPU performance metrics
- Phase-by-phase timings
- Physical validation (entropy, free energy)

---

## ⚠️ Current Limitation: PTX Error

**The logs show:**
```
Error: Failed to load neuromorphic GEMV PTX: DriverError(CUDA_ERROR_INVALID_PTX)
```

**Cause:** PTX kernels compiled for your local GPU (RTX 5070) may not be compatible with H100

**Solutions:**

### Option A: Recompile PTX for H100
```bash
# Set compute capability for H100
export CUDA_COMPUTE_CAP=90

# Rebuild with H100 target
cargo clean
cargo build --release --features cuda

# Rebuild Docker with new PTX
docker build -f Dockerfile.tsp-runtime -t delfictus/prism-ai-tsp-h100:latest .
docker push delfictus/prism-ai-tsp-h100:latest
```

### Option B: Multi-Architecture PTX (RECOMMENDED)
Build PTX for multiple GPU architectures:
```bash
# Edit build.rs to compile for multiple targets:
# SM 8.6 (RTX 3090)
# SM 8.9 (RTX 4090)
# SM 9.0 (H100)
```

---

## 🚀 Quick Test (Before Full Run)

### Test 1: Small (1K cities)
```
NUM_CITIES=1000
Time: ~5 seconds
Cost: ~$0.005
```

If this works, PTX kernels are compatible ✅

If this fails with PTX error, need to recompile kernels ❌

### Test 2: Medium (10K cities)
```
NUM_CITIES=10000
Time: ~2 minutes
Cost: ~$0.12
```

Validates GPU memory and performance

### Test 3: Full (85.9K cities)
```
NUM_CITIES=85900
Time: ~90 minutes
Cost: ~$5
```

Only run after validating smaller sizes

---

## 📋 Current Status

✅ Docker image updated: `delfictus/prism-ai-tsp-h100:1.0.1`
✅ Full 85,900 city support enabled
✅ Pushed to Docker Hub
✅ H100 GPU detected in RunPod
⚠️ PTX compatibility issue (may need recompile for H100)

**Next:** Test with NUM_CITIES=1000 first to validate PTX compatibility

---

*Last Updated: 2025-10-09*
*Version: 1.0.1*
*Full 85,900 city support: ✅ ENABLED*
