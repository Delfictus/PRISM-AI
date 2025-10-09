# RunPod Quick Start - PRISM-AI TSP H100

## Step 1: Pull Docker Image

Once the image is pushed to Docker Hub:

```
delfictus/prism-ai-tsp-h100:latest
```

## Step 2: Create RunPod Pod

1. Go to https://runpod.io
2. Navigate to "Pods" → "Deploy"
3. Select GPU: **H100 PCIe** or **H100 SXM5**

## Step 3: Configure Pod

### Container Configuration

**Docker Image:**
```
delfictus/prism-ai-tsp-h100:latest
```

**Environment Variables:**
```
NUM_CITIES=10000
MAX_ITER=5000
RUST_LOG=info
```

**Volume Mounts:**
- Container Path: `/output`
- Size: 10 GB (for results)

### Pod Settings

- **GPU**: H100 (80GB recommended)
- **CPU**: 8+ cores recommended
- **RAM**: 32 GB+ recommended
- **Disk**: 50 GB+

## Step 4: Launch & Monitor

### Start Pod

Click "Deploy" button

### View Logs (Real-time)

Click on your pod → "Logs" tab

You should see:
```
╔══════════════════════════════════════════════════════════════╗
║  PRISM-AI TSP BENCHMARK: pla85900                           ║
║  85,900 Cities - TSPLIB Official Benchmark                  ║
║  GPU-Accelerated with H100 Optimization                     ║
╚══════════════════════════════════════════════════════════════╝

✅ NVIDIA H100 detected!
```

### Monitor Progress

Watch for:
- `🔧 Initializing GPU TSP Solver...`
- `🔄 Running GPU 2-opt optimization...`
- Progress updates every 10 iterations

## Step 5: Test Different Scales

### Quick Test (1K cities, ~5 seconds)
```
NUM_CITIES=1000
MAX_ITER=1000
```

### Medium Test (5K cities, ~1 minute)
```
NUM_CITIES=5000
MAX_ITER=2000
```

### Large Test (10K cities, ~3 minutes)
```
NUM_CITIES=10000
MAX_ITER=5000
```

### Massive Test (20K cities, ~15 minutes)
```
NUM_CITIES=20000
MAX_ITER=10000
```

### FULL INSTANCE (85,900 cities, ~3-4 hours)
```
NUM_CITIES=85900
MAX_ITER=10000
```

## Step 6: Access Results

### View Output File

```bash
# Via RunPod terminal
cat /output/tour.txt

# Download via RunPod web interface
Navigate to "Files" → /output → Download tour.txt
```

### Interpret Results

Example output:
```
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
```

## Expected Runtimes on H100

| Cities | Time | Cost (RunPod H100 @ $3.49/hr) |
|--------|------|-------------------------------|
| 1,000 | ~5s | ~$0.005 |
| 5,000 | ~60s | ~$0.06 |
| 10,000 | ~3min | ~$0.17 |
| 20,000 | ~15min | ~$0.87 |
| 50,000 | ~90min | ~$5.24 |
| 85,900 | ~4hrs | ~$13.96 |

## Troubleshooting

### Container Won't Start

- Check GPU selection (must be H100 or compatible)
- Verify Docker image name is correct
- Check RunPod account has GPU credits

### Out of Memory

- Reduce NUM_CITIES
- Ensure H100 80GB model selected
- Full 85,900 cities needs 80GB GPU

### Slow Performance

- Verify H100 is actually being used: Check logs for "H100 detected"
- Monitor GPU utilization in RunPod dashboard
- Should see >90% GPU utilization during optimization

### No Output File

- Check `/output` volume is mounted
- Verify write permissions
- Look for errors in logs

## Cost Optimization

### Spot Instances

Use RunPod "Spot" pricing for 50-70% discount:
- Good for: Long runs, full instance
- Risk: May be interrupted
- Recommendation: Use for experiments, not production

### On-Demand Instances

Use "On-Demand" for guaranteed availability:
- Good for: Important runs, time-sensitive
- Cost: ~$3.49/hour for H100

### Strategy

1. Test with small sizes on Spot (cheap)
2. Scale up incrementally
3. Run full instance on On-Demand (guaranteed)

## Next Steps

### After Successful Run

1. **Download results**: Save tour.txt
2. **Validate tour**: Check tour length vs known best
3. **Scale up**: Try larger problem sizes
4. **Compare**: Run LKH-3 for comparison
5. **Publish**: Document your findings

### For World Record Attempt

1. Start with 10K cities (validate approach)
2. Scale to 20K, 50K (test scalability)
3. Run full 85,900 cities
4. Compare with known best: 142,382,641
5. Document methodology and results

## Support

**Issues**: GitHub Issues on PRISM-AI repository
**Documentation**: See `docs/TSP_H100_DOCKER_GUIDE.md`
**Image**: `delfictus/prism-ai-tsp-h100:latest`

---

**Ready to run? Pull the image and launch your pod!**

```bash
# Image will be available at:
docker pull delfictus/prism-ai-tsp-h100:latest
```
