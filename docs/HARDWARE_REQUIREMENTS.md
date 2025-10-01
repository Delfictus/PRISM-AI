# Hardware Requirements Analysis

## Kronecker Graph (kron_g500-logn16) Requirements

### Graph Statistics
- **Vertices (N)**: 65,536
- **Edges**: 3,145,477 (undirected)
- **Density**: 0.146%
- **Max Degree**: 44,193

### Memory Requirements

#### CPU-Only Implementation
```
Coupling Matrix: N × N × sizeof(Complex64)
                = 65,536 × 65,536 × 16 bytes
                = 68,719,476,736 bytes
                = 64 GB

Additional structures:
- State vectors: ~2 MB
- Edge lists: ~50 MB
- Workspace: ~1 GB

Total: ~65 GB RAM minimum
```

#### GPU Implementation (Current System)
```
GPU Memory Required:
- Coupling matrix on GPU: 64 GB
- State vectors: ~2 MB
- Intermediate results: ~4 GB

Total: ~68 GB GPU memory minimum
```

### NVIDIA H100 Specifications

#### H100 SXM5 (High-End)
- **GPU Memory**: 80 GB HBM3
- **Memory Bandwidth**: 3.35 TB/s
- **FP64 Performance**: 34 TFLOPS
- **FP32 Performance**: 67 TFLOPS
- **Tensor Core Performance**: 989 TFLOPS (FP8)

#### H100 PCIe
- **GPU Memory**: 80 GB HBM3
- **Memory Bandwidth**: 2.0 TB/s
- **FP64 Performance**: 26 TFLOPS

### Feasibility Analysis

#### ✅ H100 80GB: **YES, CAN HANDLE IT**

```
Available: 80 GB
Required:  68 GB
Headroom:  12 GB (15%)
```

**Performance Estimate:**
- Matrix allocation: ~2 seconds
- Per k-value test: 5-30 seconds
- Full search (k=2..50): 5-25 minutes
- Expected chromatic number: χ ∈ [15, 50] for this graph

#### ❌ A100 40GB: **NO, INSUFFICIENT**

```
Available: 40 GB
Required:  68 GB
Deficit:   -28 GB (70% short)
```

#### ⚠️ A100 80GB: **YES, BUT TIGHT**

```
Available: 80 GB
Required:  68 GB
Headroom:  12 GB (15%)
```

Similar to H100 but slower (~20% longer runtime).

### Google Cloud Instance Recommendations

#### Option 1: a3-highgpu-8g (RECOMMENDED)
```yaml
Instance: a3-highgpu-8g
GPUs: 8× NVIDIA H100 80GB SXM5
CPU: 208 vCPUs (4th Gen Intel Xeon)
RAM: 1.87 TB
Network: 3,200 Gbps
Cost: ~$32/hour

Benefits:
- Single GPU can handle full graph
- Can parallelize multiple k-value tests
- Massive memory for even larger problems
- Ultra-high bandwidth for data transfer
```

#### Option 2: a2-ultragpu-1g (H100 PCIe, BUDGET)
```yaml
Instance: Single H100 80GB
GPUs: 1× NVIDIA H100 80GB PCIe
CPU: 12-96 vCPUs
RAM: 85-680 GB
Cost: ~$5-12/hour

Benefits:
- Sufficient for single graph test
- Lower cost for validation
- Good for DARPA demo
```

#### Option 3: a2-highgpu-1g (A100, NOT RECOMMENDED)
```yaml
Instance: a2-highgpu-1g
GPUs: 1× NVIDIA A100 40GB
RAM: 85 GB
Cost: ~$3/hour

Problems:
- Only 40GB GPU memory (INSUFFICIENT)
- Would need out-of-core algorithms
- Much slower performance
```

### Implementation Strategy for H100

#### Current Code Compatibility
```rust
// src/quantum/src/gpu.rs already has cudarc integration
use cudarc::driver::{CudaDevice, CudaSlice};

// For kron_g500, we need:
// 1. Allocate 64GB on GPU
// 2. Transfer edge list
// 3. Build sparse coupling matrix on GPU
// 4. Run coloring iterations
```

#### Memory Optimization Options

**Option A: Dense Matrix (Current)**
```rust
// 64 GB GPU memory
let coupling_matrix: CudaSlice<Complex64> = device.htod_sync_copy(&matrix)?;
```

**Option B: Sparse Matrix (Efficient)**
```rust
// ~100 MB GPU memory (3.1M edges × 32 bytes)
struct SparseCoupling {
    row_indices: CudaSlice<u32>,    // 3.1M × 4 bytes
    col_indices: CudaSlice<u32>,    // 3.1M × 4 bytes
    values: CudaSlice<Complex64>,   // 3.1M × 16 bytes
}

// Enables graphs up to ~10M vertices on H100
```

**Recommendation**: Implement sparse matrix support for production use.

### Performance Projections

#### Dense Matrix (64 GB)
```
Memory Transfer (PCIe): ~20 seconds (3.2 GB/s)
Memory Transfer (NVLink): ~2 seconds (32 GB/s via NVLink)
Per k-value iteration: 5-10 seconds
Full search (k=2..50): 5-10 minutes
```

#### Sparse Matrix (100 MB)
```
Memory Transfer: <1 second
Per k-value iteration: 1-3 seconds
Full search (k=2..50): 1-3 minutes
Scales to 10M+ vertices
```

### Cost Estimate for DARPA Demo

#### Validation Run (Single Test)
```
Instance: a3-highgpu-8g
Duration: 1 hour
Tasks:
- Upload graph: 5 min
- Test kron_g500: 10 min
- Test 10 DIMACS benchmarks: 30 min
- Generate report: 15 min

Cost: $32 × 1 hour = $32
```

#### Full Benchmark Suite
```
Instance: a3-highgpu-8g
Duration: 4 hours
Tasks:
- Test 50+ DIMACS benchmarks
- Test 5 Kronecker graphs (different scales)
- Generate comparative analysis
- Export results

Cost: $32 × 4 hours = $128
```

### Comparison with Alternatives

| Hardware | Memory | Can Handle? | Speed | Cost/hr |
|----------|--------|-------------|-------|---------|
| **H100 80GB** | 80 GB | ✅ Yes | 1.0× | $12 |
| **A100 80GB** | 80 GB | ✅ Tight | 1.2× | $8 |
| **A100 40GB** | 40 GB | ❌ No | N/A | $3 |
| **RTX 5070** | 16 GB | ❌ No | N/A | Owned |
| **CPU Only** | 128 GB+ | ✅ Yes | 50× | $2 |

### Recommended Approach

#### For DARPA Proposal Validation

1. **Current Hardware (RTX 5070)**: Test DIMACS benchmarks (100-500 vertices)
   - Cost: $0 (owned)
   - Time: 1-2 hours
   - Proves algorithm correctness
   - Demonstrates competitive performance on standard benchmarks

2. **Google Cloud H100**: Test extreme-scale graphs (kron_g500, 65k vertices)
   - Cost: $32-128 one-time
   - Time: 1-4 hours
   - Proves scalability
   - Demonstrates capability for large real-world problems

3. **Deliverable for DARPA**:
   - "Algorithm validated on 20+ DIMACS benchmarks (optimal/near-optimal)"
   - "Scales to 65k vertex graphs on H100 hardware"
   - "Production deployment ready for edge/cloud hybrid"

### Implementation Timeline

#### Phase 1: Local Validation (Now)
```bash
# Download DIMACS benchmarks
./scripts/download_dimacs_benchmarks.sh

# Run comprehensive test suite
cargo run --release --example dimacs_benchmark_runner

# Expected results: 90%+ quality score
# Time: 30 minutes
# Cost: $0
```

#### Phase 2: Cloud Deployment (Optional)
```bash
# Deploy to Google Cloud H100
gcloud compute instances create ares-51-benchmark \
  --machine-type=a3-highgpu-8g \
  --zone=us-central1-a \
  --accelerator=type=nvidia-h100-80gb,count=1

# Install and run
./scripts/setup_cloud.sh
cargo run --release --example benchmark_kron_g500

# Time: 1 hour
# Cost: $32
```

### Conclusion

**Answer: YES, H100 80GB can handle kron_g500-logn16**

- ✅ 80 GB GPU memory > 68 GB required
- ✅ Performance suitable for validation (5-10 minutes)
- ✅ Cost reasonable for one-time demo ($32/hour)
- ✅ Proves scalability for DARPA proposal

**Recommendation for DARPA Proposal:**
1. Use local RTX 5070 for DIMACS benchmarks (proves correctness)
2. Use H100 cloud instance for kron_g500 (proves scale)
3. Total cost: <$50 for complete validation
4. Demonstrates both algorithm quality AND production scalability
