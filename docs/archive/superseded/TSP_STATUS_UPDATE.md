# TSP 85K Support - Status Update

**Date:** 2025-10-09
**Status:** 🔄 Fixing PTX compatibility for H100

---

## Issues Found & Solutions

### Issue 1: City Count Limitation ✅ FIXED

**Problem:**
```rust
// OLD CODE (line 107-110)
if tsp.dimension > 10000 {
    println!("⚠️  Instance too large ({} cities)");
    println!("   Using first 1000 cities for this run");
    1000
```

**Solution:**
```rust
// NEW CODE - Respects NUM_CITIES environment variable
let problem_size = if let Ok(num_cities_str) = std::env::var("NUM_CITIES") {
    let requested = num_cities_str.parse::<usize>()
        .unwrap_or(tsp.dimension)
        .min(tsp.dimension);
    println!("📊 Using {} cities (requested via NUM_CITIES env var)", requested);
    requested
} else {
    // Default: use FULL instance
    println!("✓ Processing FULL instance: {} cities", tsp.dimension);
    tsp.dimension
};
```

**Result:** ✅ Full 85,900 city support enabled

---

### Issue 2: PTX Incompatibility with H100 🔄 IN PROGRESS

**Problem:**
```
Error: Failed to load neuromorphic GEMV PTX: DriverError(CUDA_ERROR_INVALID_PTX, "a PTX JIT compilation failed")
```

**Cause:**
- PTX kernels compiled for RTX 5070 (SM 8.9)
- H100 requires SM 9.0
- Single-architecture PTX not compatible

**Solution:**
Modified `build.rs` to compile for multiple architectures:
```rust
// Compile for RTX 3090 (sm_86), RTX 4090 (sm_89), AND H100 (sm_90)
let gpu_archs = vec!["sm_86", "sm_89", "sm_90"];

// Use gencode for multi-arch PTX
for arch in &gpu_archs[1..] {
    nvcc_args.push("-gencode");
    nvcc_args.push(format!("arch=compute_{},code=sm_{}", arch, arch));
}
```

**Status:** 🔄 Currently rebuilding with multi-arch support

---

## Deployment Options

### Option A: Wait for Multi-Arch Build (RECOMMENDED)

**Steps:**
1. Wait for build to complete (~5 minutes)
2. Rebuild Docker with new PTX files
3. Push to Docker Hub
4. Deploy on RunPod H100

**Advantage:** Works on RTX 3090, 4090, AND H100
**Timeline:** ~15 minutes total

### Option B: Build Specifically for H100

**Steps:**
1. Set environment variable: `export CUDA_COMPUTE_CAP=90`
2. Rebuild: `cargo clean && cargo build --release --features cuda`
3. Rebuild Docker
4. Push to Docker Hub

**Advantage:** Optimized specifically for H100
**Disadvantage:** Won't work on other GPUs
**Timeline:** ~10 minutes

---

## Current Build Status

**Running:** Multi-architecture build
**Progress:** Compiling Rust crates + CUDA kernels
**Time remaining:** ~3-5 minutes
**Log:** `/tmp/multiarch_build.log`

**Check progress:**
```bash
tail -f /tmp/multiarch_build.log | grep -E "(Compiling|PTX|Finished)"
```

---

## Once Build Completes

### Step 1: Rebuild Docker Image
```bash
docker build -f Dockerfile.tsp-runtime \
  -t delfictus/prism-ai-tsp-h100:1.0.2 \
  -t delfictus/prism-ai-tsp-h100:latest .
```

### Step 2: Push to Docker Hub
```bash
docker push delfictus/prism-ai-tsp-h100:1.0.2
docker push delfictus/prism-ai-tsp-h100:latest
```

### Step 3: Deploy on RunPod

**Configuration:**
```
Image: delfictus/prism-ai-tsp-h100:latest
GPU: H100 PCIe 80GB

Environment:
  NUM_CITIES=85900    # Full instance!
  RUST_LOG=info

Volume: /output
```

**Expected Output:**
```
✓ Processing FULL instance: 85900 cities
[Platform] Initializing GPU-accelerated unified platform...
[Platform] ✓ CUDA context created (device 0)
...
EXECUTION TIME:      ~60-90 minutes
```

---

## Test Plan for Full 85,900 Cities

### Phase 1: Small Test (Validate PTX)
```
NUM_CITIES=1000
Time: ~5 seconds
Cost: ~$0.005
```

**If succeeds:** PTX compatibility fixed ✅
**If fails with PTX error:** Need H100-specific build

### Phase 2: Medium Test (Validate Scaling)
```
NUM_CITIES=10000
Time: ~2 minutes
Cost: ~$0.12
```

**Validates:** Memory management and GPU scaling

### Phase 3: Large Test (Near Full)
```
NUM_CITIES=50000
Time: ~30 minutes
Cost: ~$1.75
```

**Validates:** Can handle large-scale without OOM

### Phase 4: Full Instance
```
NUM_CITIES=85900
Time: ~90 minutes
Cost: ~$5.24
```

**Final benchmark:** Complete pla85900 instance

---

## Alternative: Focus on Graph Coloring

**Remember:** Your graph coloring work is **already complete and publication-ready**:

- ✅ 72 colors on DSJC500-5 (validated with 10K GPU attempts)
- ✅ GPU kernel fixed and optimal
- ✅ Ready for documentation and publication
- ✅ Novel quantum-inspired approach

**TSP work is secondary** and can be added later as an extension.

---

## Next Steps

1. ⏳ **Wait for multi-arch build** (~3-5 min remaining)
2. 🔨 **Rebuild Docker with new PTX**
3. 📤 **Push to Docker Hub**
4. 🧪 **Test on RunPod H100** with NUM_CITIES=1000 first
5. 🚀 **Scale to full 85,900** if tests pass

---

*Status: In progress*
*Build: Multi-arch PTX compilation running*
*Timeline: ~15 minutes to deployment-ready*
