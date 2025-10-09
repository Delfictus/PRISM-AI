# RunPod Single Instance with 8× H200 - Configuration

**Your Setup:** One RunPod instance with 8× H200 GPUs built-in

---

## ❌ Current Problem

**Issue 1:** Only using 1,000 cities (not 85,900)
**Issue 2:** Only using GPU 0 (not all 8 GPUs)

---

## ✅ Solution

### Fix #1: Set NUM_CITIES in RunPod

**In RunPod Pod Configuration:**

Environment Variables section, set:
```
NUM_CITIES=85900
```

Currently it's showing:
```
📊 Using 1000 cities (requested via NUM_CITIES env var)
```

This means RunPod has `NUM_CITIES=1000` set. **Change it to 85900**.

### Fix #2: Enable All GPUs in Docker

**In RunPod Pod Configuration:**

Docker Options → GPU Settings:
```
--gpus all
```

Or in environment:
```
NVIDIA_VISIBLE_DEVICES=all
```

This will let the container see all 8 GPUs instead of just GPU 0.

---

## 🔧 Current Code Limitation

**The platform code currently only uses GPU 0:**

```rust
// src/integration/unified_platform.rs:189
let cuda_context = CudaContext::new(0)  // Hardcoded to device 0
```

**To use all 8 GPUs, we need to modify the code.**

Do you want me to:

### Option A: Modify Code for True Multi-GPU (30-60 min work)
- Update UnifiedPlatform to accept device_id
- Create multi-GPU coordinator
- Split work across 8 GPUs in single process
- Rebuild and push new Docker image

### Option B: Quick Fix for Full 85,900 Cities (5 min)
- Just set NUM_CITIES=85900 in RunPod
- Run on single GPU (GPU 0)
- Takes ~90 minutes but uses full instance
- Other 7 GPUs idle

### Option C: Docker Compose Multi-Container (15 min)
- Create docker-compose file for your instance
- Launch 8 containers on your instance
- Each uses different GPU
- Works with current image

---

## Recommended: Option C (Works Now)

**Create on your RunPod instance:**

`docker-compose.yml`:
```yaml
version: '3.8'

services:
  tsp-gpu0:
    image: delfictus/prism-ai-tsp-h100:h200
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NUM_CITIES=10738
      - RUST_LOG=info
    volumes:
      - ./output/gpu0:/output

  tsp-gpu1:
    image: delfictus/prism-ai-tsp-h100:h200
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - NUM_CITIES=10738
      - RUST_LOG=info
    volumes:
      - ./output/gpu1:/output

  # ... repeat for gpu2-gpu7 ...
```

Then run:
```bash
docker-compose up
```

All 8 GPUs will work in parallel!

---

## What Would You Like?

1. **Quick fix** - Just set NUM_CITIES=85900 (uses 1 GPU, ~90 min)
2. **Option C** - Docker compose setup (uses all 8 GPUs, ~10 min)
3. **Option A** - Modify code for true multi-GPU in single process (takes longer to implement)

Let me know and I'll implement it!
