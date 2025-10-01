# GPU-Accelerated TSP Testing Guide

## Quick Start

### Run Basic Tests (3 test cases, ~5 seconds)
```bash
./run_gpu_tsp_tests.sh
```

### Run Full Benchmarks (9 TSPLIB instances, ~3 seconds)
```bash
./run_gpu_tsp_benchmarks.sh
```

## Manual Execution

### Basic Tests
```bash
# Set CUDA library path (required for WSL2)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Build and run
cargo build --release --example test_gpu_tsp
cargo run --release --example test_gpu_tsp
```

### Benchmarks
```bash
# Set CUDA library path (required for WSL2)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Build and run
cargo build --release --example tsp_benchmark_runner_gpu
cargo run --release --example tsp_benchmark_runner_gpu
```

## Test Coverage

### `test_gpu_tsp` - Minimal Test Suite
- **Test 1**: Complete graph K_5 (5 cities)
- **Test 2**: Grid coupling (8 cities, geometric layout)
- **Test 3**: Random coupling (10 cities)

**Expected Output:**
```
✅ ALL TESTS PASSED
- Test 1: Valid tour, optimized length
- Test 2: ~30% improvement
- Test 3: ~65% improvement
```

### `tsp_benchmark_runner_gpu` - TSPLIB Benchmark Suite
Simulates 9 standard TSPLIB benchmark instances:
- **berlin52** (52 cities)
- **eil51** (51 cities)
- **eil76** (76 cities)
- **kroA100** (100 cities)
- **kroB100** (100 cities)
- **rd100** (100 cities)
- **eil101** (101 cities)
- **pr152** (152 cities)
- **kroA200** (200 cities)

**Expected Output:**
```
✅ ALL BENCHMARKS COMPLETED
Completed: 9/9 (100.0%)
Average Improvement: ~17.6%
Average Time: ~0.34s
Total Time: ~3.22s
```

## What's Being Tested

### GPU Features
1. **Distance Matrix Computation** - Parallel O(n²) on GPU
2. **2-opt Swap Evaluation** - Parallel evaluation of all possible swaps
3. **Parallel Reduction** - Finding best improvement across GPU threads
4. **Memory Transfers** - Host-to-Device (H2D) and Device-to-Host (D2H)
5. **CUDA Kernel Compilation** - PTX loading and execution

### Algorithms
1. **Nearest Neighbor** - Greedy tour construction (CPU)
2. **2-opt Local Search** - Iterative improvement (GPU-accelerated)
3. **Tour Validation** - Ensures all cities visited exactly once

## System Requirements

- **GPU**: NVIDIA RTX 5070 (or compatible CUDA device)
- **CUDA**: NVIDIA drivers with CUDA support
- **WSL2**: Windows Subsystem for Linux 2 (if on Windows)
- **nvcc**: CUDA compiler for kernel compilation

## Troubleshooting

### Error: "Failed to initialize CUDA device 0"
```bash
# Check GPU is visible
nvidia-smi

# Check WSL2 GPU access
ls -la /dev/dxg

# Set library path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Error: "PTX file not found"
```bash
# Clean and rebuild
cargo clean
cargo build --release --example test_gpu_tsp
```

### Error: "nvcc not found"
```bash
# Check CUDA compiler
nvcc --version

# If missing, install CUDA toolkit
# https://developer.nvidia.com/cuda-downloads
```

## Performance Expectations

### Basic Tests (~5 seconds total)
- Test 1 (5 cities): < 1 second
- Test 2 (8 cities): < 1 second  
- Test 3 (10 cities): < 1 second

### Benchmarks (~3 seconds total)
- 50 cities: ~0.06s
- 100 cities: ~0.06s
- 150 cities: ~0.07s
- 200 cities: ~0.07s

**Note**: First benchmark (berlin52) may take 2-3s due to GPU warmup.

## Files

### Test Files
- `examples/test_gpu_tsp.rs` - Minimal test suite
- `examples/tsp_benchmark_runner_gpu.rs` - TSPLIB benchmarks

### Runner Scripts
- `run_gpu_tsp_tests.sh` - Basic test runner
- `run_gpu_tsp_benchmarks.sh` - Benchmark runner

### Source Code
- `src/quantum/src/gpu_tsp.rs` - GPU TSP solver implementation
- `cuda/tsp_solver.cu` - CUDA kernels
- `build_cuda.rs` - CUDA compilation script

## Success Criteria

✅ **All tests pass** - Valid tours with improvements
✅ **All benchmarks complete** - 9/9 instances solved
✅ **GPU utilized** - CUDA kernels execute successfully
✅ **Performance validated** - Sub-second latency per benchmark

---

**Last Updated**: Based on successful run achieving:
- 100% test pass rate (3/3)
- 100% benchmark completion (9/9)
- 17.6% average improvement
- 3.22s total benchmark time
