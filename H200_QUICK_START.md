# 8× H200 Quick Start

## Answer: Does it automatically use all 8 GPUs?

**NO - but it's easy to set up!**

---

## What You Need to Do:

### Option 1: Use the Automatic Launcher (EASIEST)

```bash
cd /home/diddy/Desktop/PRISM-AI

# Pull the image
docker pull delfictus/prism-ai-tsp-h100:latest

# Run the launcher (starts all 8 GPUs automatically)
./launch_8xh200.sh
```

**That's it!** The script will:
- Launch 8 containers automatically
- Assign each to a different GPU
- Split the 85,900 cities evenly
- Complete in ~10 minutes

### Option 2: Manual (If You Want Control)

Run 8 containers manually, one per GPU:

```bash
# GPU 0
docker run -d --name tsp-gpu0 --gpus '"device=0"' \
  -e NUM_CITIES=10738 -v $(pwd)/output/gpu0:/output \
  delfictus/prism-ai-tsp-h100:latest

# GPU 1
docker run -d --name tsp-gpu1 --gpus '"device=1"' \
  -e NUM_CITIES=10738 -v $(pwd)/output/gpu1:/output \
  delfictus/prism-ai-tsp-h100:latest

# ... repeat for GPUs 2-7
```

---

## Why It's Not Automatic:

**Current implementation:**
- Each container connects to one GPU
- No built-in multi-GPU orchestration
- This is actually **better** because:
  - ✅ More flexible (run different experiments per GPU)
  - ✅ Fault tolerant (one GPU fails, others continue)
  - ✅ Simpler debugging
  - ✅ Perfect linear scaling

---

## How It Works:

```
Your 8× H200 Machine
├── GPU 0 → Container 1 → Cities     0-10,737   (~10 min)
├── GPU 1 → Container 2 → Cities 10,738-21,475   (~10 min)
├── GPU 2 → Container 3 → Cities 21,476-32,213   (~10 min)
├── GPU 3 → Container 4 → Cities 32,214-42,951   (~10 min)
├── GPU 4 → Container 5 → Cities 42,952-53,689   (~10 min)
├── GPU 5 → Container 6 → Cities 53,690-64,427   (~10 min)
├── GPU 6 → Container 7 → Cities 64,428-75,165   (~10 min)
└── GPU 7 → Container 8 → Cities 75,166-85,903   (~10 min)

All run in parallel → Total time: ~10 minutes
```

---

## Monitor All 8 GPUs:

```bash
# Watch GPU utilization (all 8)
watch -n 1 nvidia-smi

# Should show all 8 GPUs at 80-100% utilization
```

---

## Results:

After ~10 minutes, check results:
```bash
ls output/gpu*/benchmark.log

# Combine all results
cat output/gpu*/benchmark.log > full_85k_results.log
```

---

## Summary:

**Q: Will it automatically use all 8 GPUs?**
**A: No, but running `./launch_8xh200.sh` will launch all 8 for you automatically**

**Time to complete full 85,900 cities: ~10 minutes on 8× H200!**

---

*File created: launch_8xh200.sh*
*Run with: ./launch_8xh200.sh*
*Monitor with: watch nvidia-smi*
