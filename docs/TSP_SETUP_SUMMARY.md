# TSP H100 Docker Setup - Summary

**Date:** 2025-10-09
**Target:** pla85900.tsp (85,900 cities)
**Platform:** RunPod H100 GPU
**Status:** ✅ Ready for build & deploy

---

## What Was Created

### 1. TSP Benchmark Runner
**File:** `examples/run_tsp_pla85900.rs`

GPU-accelerated TSP solver specifically for pla85900 benchmark:
- Parses TSPLIB format
- Computes coupling matrix from coordinates
- Uses GPU for distance matrix and 2-opt optimization
- Configurable problem size and iterations
- Outputs tour and performance metrics

### 2. H100-Optimized Dockerfile
**File:** `Dockerfile.tsp-h100`

Multi-stage Docker image:
- Base: NVIDIA CUDA 12.3 (H100 compatible)
- Rust toolchain with CUDA support
- Compiles CUDA kernels for SM 9.0 (H100)
- Includes pla85900.tsp benchmark
- Configurable via environment variables
- Saves results to `/output` volume

### 3. Build & Push Script
**File:** `scripts/build_and_push_tsp_docker.sh`

Automated build and deployment:
- Builds Docker image for linux/amd64
- Tags with version and latest
- Pushes to Docker Hub: `delfictus/prism-ai-tsp-h100`
- Includes smoke tests

### 4. Documentation

**`docs/TSP_H100_DOCKER_GUIDE.md`** - Complete technical guide:
- Pull and run commands
- Environment variables
- Expected performance
- Troubleshooting
- Memory requirements

**`RUNPOD_QUICK_START.md`** - Quick start for RunPod:
- Step-by-step pod creation
- Configuration settings
- Test scenarios (1K to 85K cities)
- Cost estimates
- Monitoring and results access

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NUM_CITIES` | 1000 | Number of cities to solve |
| `MAX_ITER` | 1000 | Max optimization iterations |
| `TSP_FILE` | benchmarks/tsp/pla85900.tsp | Input file |
| `OUTPUT_FILE` | /output/tour.txt | Output file |

### Docker Image

**Image Name:** `delfictus/prism-ai-tsp-h100`
**Tags:**
- `1.0.0` - Version tag
- `latest` - Latest build

---

## Build Process

### Current Status

Docker build in progress:
- Downloading CUDA base image (1.3GB)
- Will compile Rust project (~10-15 minutes)
- Will build CUDA kernels for H100
- Final image size: ~5-8GB

### Build Command

```bash
./scripts/build_and_push_tsp_docker.sh
```

Or manually:
```bash
docker build \
  --platform linux/amd64 \
  -f Dockerfile.tsp-h100 \
  -t delfictus/prism-ai-tsp-h100:1.0.0 \
  -t delfictus/prism-ai-tsp-h100:latest \
  .
```

### Push to Docker Hub

```bash
docker push delfictus/prism-ai-tsp-h100:1.0.0
docker push delfictus/prism-ai-tsp-h100:latest
```

---

## Usage on RunPod

### 1. Create H100 Pod

1. Go to runpod.io
2. Select H100 PCIe or H100 SXM5
3. Use Docker image: `delfictus/prism-ai-tsp-h100:latest`

### 2. Configure Environment

```
NUM_CITIES=10000
MAX_ITER=5000
RUST_LOG=info
```

### 3. Mount Output Volume

- Container path: `/output`
- Size: 10GB

### 4. Launch and Monitor

Watch logs for:
```
✅ NVIDIA H100 detected!
🔧 Initializing GPU TSP Solver for 10000 cities...
🔄 Running GPU 2-opt optimization...
```

---

## Performance Expectations

### H100 PCIe 80GB

| Cities | Estimated Time | GPU Memory | Cost (@$3.49/hr) |
|--------|---------------|------------|------------------|
| 1,000 | ~5 seconds | ~100 MB | ~$0.005 |
| 5,000 | ~1 minute | ~500 MB | ~$0.06 |
| 10,000 | ~3 minutes | ~2 GB | ~$0.17 |
| 20,000 | ~15 minutes | ~8 GB | ~$0.87 |
| 50,000 | ~90 minutes | ~50 GB | ~$5.24 |
| **85,900** | **~4 hours** | **~80 GB** | **~$13.96** |

---

## Algorithm Details

### What Runs on GPU:
- Distance matrix computation (O(n²))
- 2-opt swap evaluation (parallel)
- Distance normalization
- Min/max reductions

### What Runs on CPU:
- TSPLIB file parsing
- Coupling matrix computation
- Nearest-neighbor initial tour
- Swap application
- Tour validation

### GPU Kernels Used:
1. `compute_distance_matrix` - Computes all pairwise distances
2. `find_max_distance` - Parallel reduction for normalization
3. `normalize_distances` - Scale distances to [0,1]
4. `evaluate_2opt_swaps` - Test all possible 2-opt swaps
5. `find_min_delta` - Find best improvement

---

## Benchmark Information

### pla85900.tsp

- **Type:** Real TSPLIB benchmark
- **Source:** Microchip layout (programmed logic array)
- **Cities:** 85,900
- **Coordinates:** 2D Euclidean
- **Known Best:** 142,382,641 (solved 2024)
- **Solver:** Concorde TSP (exact solver, took days/weeks)

### Goal

1. **Speed:** Solve in <4 hours (vs days for exact solver)
2. **Quality:** Within 5% of optimal (142M to 149M tour length)
3. **Scalability:** Demonstrate GPU acceleration effectiveness

---

## Testing Strategy

### Phase 1: Validation (1-5K cities)
- Verify algorithm correctness
- Test GPU acceleration
- Baseline performance metrics
- Cost: ~$0.10

### Phase 2: Scaling (10-20K cities)
- Test scalability
- Monitor GPU utilization
- Optimize parameters
- Cost: ~$1.00

### Phase 3: Large Scale (50K cities)
- Near-full-scale test
- Memory usage validation
- Performance projections
- Cost: ~$5.00

### Phase 4: Full Run (85,900 cities)
- Complete benchmark
- Record tour length
- Compare with known best
- Cost: ~$14.00

**Total estimated cost for full testing: ~$20**

---

## Next Steps

### 1. Wait for Docker Build
Current build in progress (~15 minutes remaining)

### 2. Push to Docker Hub
```bash
docker push delfictus/prism-ai-tsp-h100:latest
```

### 3. Test Locally (if GPU available)
```bash
docker run --gpus all \
  -e NUM_CITIES=1000 \
  -v $(pwd)/output:/output \
  delfictus/prism-ai-tsp-h100:latest
```

### 4. Deploy to RunPod
- Create H100 pod
- Configure environment
- Run test suite
- Scale to full instance

### 5. Document Results
- Record tour lengths
- Compare with known best
- Analyze GPU utilization
- Calculate cost efficiency

---

## Support Files

All files created and ready:

✅ `examples/run_tsp_pla85900.rs` - Benchmark runner
✅ `Dockerfile.tsp-h100` - H100-optimized image
✅ `scripts/build_and_push_tsp_docker.sh` - Build script
✅ `docs/TSP_H100_DOCKER_GUIDE.md` - Complete guide
✅ `RUNPOD_QUICK_START.md` - Quick start
✅ `docs/TSP_SETUP_SUMMARY.md` - This file

---

## Comparison: TSP vs Graph Coloring

You now have two GPU-accelerated benchmarks ready:

### Graph Coloring (Completed)
- ✅ DIMACS benchmarks integrated
- ✅ GPU kernel fixed and validated
- ✅ Results: 72 colors (optimal for approach)
- ✅ Ready for publication

### TSP (In Progress)
- 🔄 Docker image building
- ⏳ Ready for RunPod deployment
- 🎯 Target: pla85900 (85,900 cities)
- 📊 Goal: <4 hours, <5% from optimal

Both demonstrate your novel quantum-inspired GPU acceleration approach!

---

**Status:** Docker build in progress
**Next:** Push to Docker Hub when build completes
**Timeline:** Ready to run on RunPod within ~20 minutes
