# PRISM-AI TSP H100 Docker Guide

**GPU-Accelerated TSP Solver for pla85900 (85,900 cities)**

Optimized for NVIDIA H100 GPUs on RunPod

---

## Quick Start

### Pull the Image

```bash
docker pull your-dockerhub-username/prism-ai-tsp-h100:latest
```

### Run Small Test (1,000 cities)

```bash
docker run --gpus all \
  -e NUM_CITIES=1000 \
  -e MAX_ITER=1000 \
  -v $(pwd)/output:/output \
  your-dockerhub-username/prism-ai-tsp-h100:latest
```

### Run Large Scale (10,000 cities)

```bash
docker run --gpus all \
  -e NUM_CITIES=10000 \
  -e MAX_ITER=5000 \
  -v $(pwd)/output:/output \
  your-dockerhub-username/prism-ai-tsp-h100:latest
```

### Run Full Instance (85,900 cities)

```bash
docker run --gpus all \
  -e NUM_CITIES=85900 \
  -e MAX_ITER=10000 \
  -v $(pwd)/output:/output \
  your-dockerhub-username/prism-ai-tsp-h100:latest
```

---

## RunPod Setup

### 1. Create H100 Pod

1. Go to [RunPod.io](https://runpod.io)
2. Click "Deploy" → "GPU Instances"
3. Select "H100 PCIe" or "H100 SXM5"
4. Choose Docker deployment

### 2. Configure Pod

**Docker Image:**
```
your-dockerhub-username/prism-ai-tsp-h100:latest
```

**Environment Variables:**
```
NUM_CITIES=10000
MAX_ITER=5000
RUST_LOG=info
```

**Volume Mounts:**
- Container Path: `/output`
- This is where tour results will be saved

### 3. Start Pod

Click "Deploy" and wait for pod to start

### 4. View Logs

```bash
# Via RunPod web interface
Click on pod → "Logs" tab

# Or via RunPod CLI
runpod logs <pod-id>
```

### 5. Download Results

```bash
# Results saved in /output/tour.txt
runpod exec <pod-id> cat /output/tour.txt > tour.txt
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_CITIES` | 1000 | Number of cities to solve |
| `MAX_ITER` | 1000 | Maximum optimization iterations |
| `TSP_FILE` | benchmarks/tsp/pla85900.tsp | TSP instance file |
| `OUTPUT_FILE` | /output/tour.txt | Output file for tour |
| `RUST_LOG` | info | Log level (error, warn, info, debug, trace) |

---

## Expected Performance

### H100 PCIe (80GB)

| Cities | Coupling | Init | Optimization | Total |
|--------|----------|------|--------------|-------|
| 1,000 | 0.5s | 0.3s | 2-5s | ~3-6s |
| 5,000 | 10s | 1s | 30-60s | ~40-70s |
| 10,000 | 40s | 2s | 120-180s | ~160-220s |
| 20,000 | 160s | 5s | 480-720s | ~650-900s |
| 50,000 | 1000s | 15s | 3000-5000s | ~4000-6000s |
| 85,900 | 3000s | 30s | 8000-12000s | ~11000-15000s |

**Full instance (85,900 cities) estimated time: 3-4 hours on H100**

---

## Output Format

### Console Output

```
╔══════════════════════════════════════════════════════════════╗
║  PRISM-AI TSP BENCHMARK: pla85900                           ║
║  85,900 Cities - TSPLIB Official Benchmark                  ║
║  GPU-Accelerated with H100 Optimization                     ║
╚══════════════════════════════════════════════════════════════╝

✅ NVIDIA H100 detected!

📊 Instance Information:
  Name:        pla85900
  Dimension:   85900 cities
  Type:        Real TSPLIB benchmark
  Known best:  142,382,641 (as of 2024)

🎯 Running on 10000 cities (huge)

...

═══════════════════════════════════════════════════════════════
  RESULTS
═══════════════════════════════════════════════════════════════

Instance:            pla85900
Cities processed:    10000 / 85900
Known best (full):   142,382,641

PERFORMANCE:
  Coupling setup:    40.23s
  Initialization:    2.15s
  Optimization:      156.78s
  Total time:        199.16s

SOLUTION:
  Final tour length: 123456.78
  Tour valid:        ✓ YES

💾 Tour saved to: /output/tour.txt
```

### Tour File Format

```
0
142
87
...
(one city index per line)
```

---

## Building the Image

### Prerequisites

- Docker installed
- NVIDIA Docker runtime configured
- Docker Hub account

### Build Command

```bash
cd /path/to/PRISM-AI
./scripts/build_and_push_tsp_docker.sh
```

Or manually:

```bash
docker build \
  --platform linux/amd64 \
  -f Dockerfile.tsp-h100 \
  -t your-dockerhub-username/prism-ai-tsp-h100:1.0.0 \
  .
```

### Push to Docker Hub

```bash
docker login
docker push your-dockerhub-username/prism-ai-tsp-h100:1.0.0
docker push your-dockerhub-username/prism-ai-tsp-h100:latest
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check if nvidia-smi works in container
docker run --gpus all your-dockerhub-username/prism-ai-tsp-h100:latest nvidia-smi

# If fails, check Docker runtime
docker info | grep nvidia
```

### Out of Memory

```bash
# Reduce number of cities
-e NUM_CITIES=1000

# Or use smaller batch size (modify code)
```

### Slow Performance

```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Should show high GPU utilization during optimization
```

### CUDA Errors

```bash
# Check CUDA version compatibility
docker run --gpus all your-dockerhub-username/prism-ai-tsp-h100:latest nvcc --version

# H100 requires CUDA 12.x
```

---

## Technical Details

### GPU Acceleration

**What runs on GPU:**
- Distance matrix computation (O(n²))
- 2-opt swap evaluation (massively parallel)
- Distance normalization

**What runs on CPU:**
- Initial nearest-neighbor tour construction
- Swap application
- Tour validation

### Algorithm

1. **Initialization**
   - Parse TSPLIB file (CPU)
   - Compute coupling matrix from coordinates (CPU)
   - Compute distance matrix (GPU)
   - Build initial tour with nearest-neighbor (CPU)

2. **Optimization (GPU 2-opt)**
   - Evaluate all n×(n-3)/2 possible swaps (GPU)
   - Find best improvement (GPU reduction)
   - Apply best swap (CPU)
   - Repeat until convergence or max iterations

3. **Output**
   - Validate tour (CPU)
   - Save results to file

### Memory Requirements

| Cities | Distance Matrix | Total GPU Memory |
|--------|----------------|------------------|
| 1,000 | 4 MB | ~100 MB |
| 5,000 | 100 MB | ~500 MB |
| 10,000 | 400 MB | ~2 GB |
| 20,000 | 1.6 GB | ~8 GB |
| 50,000 | 10 GB | ~50 GB |
| 85,900 | 30 GB | ~80 GB |

**H100 80GB can handle full 85,900 city instance**

---

## Benchmarking

### Run Multiple Sizes

```bash
#!/bin/bash
for cities in 1000 2000 5000 10000 20000; do
  echo "Testing $cities cities..."
  docker run --gpus all \
    -e NUM_CITIES=$cities \
    -e MAX_ITER=1000 \
    -v $(pwd)/output:/output \
    your-dockerhub-username/prism-ai-tsp-h100:latest \
    2>&1 | tee output/benchmark_${cities}.log
done
```

### Compare with LKH-3

```bash
# Run PRISM-AI
time docker run --gpus all -e NUM_CITIES=10000 ...

# Run LKH-3 (CPU)
time LKH pla85900.par

# Compare results
```

---

## World Record Attempt

### Known Best for pla85900

**142,382,641** (as of 2024)

Solved by Concorde TSP solver after extensive computation time.

### Goal

1. **Match or beat time**: Solve full instance in <3 hours (vs days for Concorde)
2. **Competitive quality**: Within 5% of optimal (142M to 149M)
3. **Validate approach**: GPU acceleration for large-scale TSP

### Strategy

1. Start with small subsets (1K, 5K, 10K cities)
2. Verify quality vs known results
3. Scale up incrementally
4. Run full instance (85,900 cities)
5. Compare tour length with known best

---

## Citation

If you use this work, please cite:

```
@software{prism_ai_tsp_2025,
  title = {PRISM-AI GPU-Accelerated TSP Solver},
  author = {PRISM-AI Team},
  year = {2025},
  url = {https://github.com/yourusername/PRISM-AI}
}
```

---

## License

See main repository for license information.

---

## Support

- Issues: GitHub Issues
- Documentation: `/docs`
- Examples: `/examples`

---

**Last Updated:** 2025-10-09
**Version:** 1.0.0
**Target GPU:** NVIDIA H100
**Benchmark:** pla85900 (85,900 cities)
