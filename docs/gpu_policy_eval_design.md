# GPU Policy Evaluation Design Document

**Task:** 1.1.1.1 - Design GPU-friendly policy representation
**Date:** 2025-10-06
**Status:** 🟡 Design Phase - Under Review

---

## Executive Summary

This document designs GPU-friendly data structures and memory layout for accelerating policy evaluation in Active Inference. The goal is to move the 231ms CPU bottleneck to GPU, achieving <10ms target.

---

## Current System Parameters

### Discovered from Codebase Analysis

**Policy Configuration** (from `src/integration/adapters.rs:364`):
- `horizon`: 3 (3-step lookahead)
- `n_policies`: 5 (optimized from 10)
- `preferred_observations`: 100 dimensions

**State Dimensions** (from `hierarchical_model.rs`):
- `n_windows`: 900 (30×30 window array) - **ACTUAL STATE SPACE**
- `obs_dim`: 100 (measurement space, reduced from 900)
- `state_dim`: 900 (Level 1 window phases)

**Data Structures:**
- `Policy`: Contains `Vec<ControlAction>` + `expected_free_energy` + `id`
- `ControlAction`: Contains `Array1<f64>` (900 dims) + `Vec<usize>` (measurement pattern)
- `GaussianBelief`: mean (900), variance (900), precision (900)

---

## Problem Analysis

### Current CPU Implementation

```rust
// src/active_inference/policy_selection.rs:125-147
pub fn select_policy(&self, model: &HierarchicalModel) -> Policy {
    let policies = self.generate_policies(model);  // 5 policies

    let evaluated: Vec<_> = policies
        .into_iter()
        .map(|mut policy| {
            // THIS LOOP: 5 iterations × ~46ms = 230ms
            policy.expected_free_energy =
                self.compute_expected_free_energy(model, &policy);
            policy
        })
        .collect();

    evaluated.into_iter().min_by(...).unwrap()
}
```

### What `compute_expected_free_energy()` Does

For EACH of 5 policies:

1. **`multi_step_prediction()`** - 3-step trajectory simulation
   - Input: Initial state (900 dims), 3 actions (each 900 dims)
   - Output: 3 future states (each 900 dims)
   - Operations: Matrix-vector products, belief updates
   - Cost: ~20ms per policy on CPU

2. **Observation prediction** - At each of 3 future states
   - Input: Future state (900 dims)
   - Output: Predicted observations (100 dims)
   - Operations: 100×900 matrix-vector multiply
   - Cost: ~10ms per policy on CPU

3. **EFE components** - Risk, ambiguity, novelty
   - Risk: `(predicted_obs - preferred_obs)²` sum over 100 dims
   - Ambiguity: Variance propagation through observation model
   - Novelty: Entropy difference (prior vs posterior)
   - Cost: ~16ms per policy on CPU

**Total: 5 policies × 46ms = 230ms on CPU**

---

## GPU-Friendly Design

### Design Goals

1. **Flatten nested structures** for GPU
2. **Minimize CPU-GPU transfers** (ideally once per evaluation)
3. **Maximize parallelism** (5 policies, 3 steps, 900 dimensions)
4. **Keep memory coalesced** for GPU efficiency

### Memory Layout Strategy

#### Option A: Flatten Everything (Chosen)

**Rationale:** GPUs prefer contiguous arrays over nested structures.

```
Policies (5):
├─ Policy 0: [action_0_step_0 (900) | action_0_step_1 (900) | action_0_step_2 (900)]
├─ Policy 1: [action_1_step_0 (900) | action_1_step_1 (900) | action_1_step_2 (900)]
├─ Policy 2: ...
├─ Policy 3: ...
└─ Policy 4: [action_4_step_0 (900) | action_4_step_1 (900) | action_4_step_2 (900)]

Flattened: [5 × 3 × 900] = 13,500 f64 values
```

**Pros:**
- Single contiguous buffer
- Easy GPU indexing: `actions[policy_idx * horizon * state_dim + step * state_dim + dim]`
- Coalesced memory access

**Cons:**
- Ignores measurement pattern (assumes full 900-dim actions)
- Need to handle sparse patterns separately if needed

#### Option B: Separate Arrays (Discarded)

Separate `phase_corrections` and `measurement_patterns` arrays.

**Rejected because:** Measurement patterns are CPU-side selection, not needed for GPU trajectory prediction.

---

## GPU Data Structures

### 1. Input Data (CPU → GPU Upload)

```rust
// Flatten all policies into single buffer
struct GpuPolicyInputs {
    // Initial state (current belief)
    initial_mean: Vec<f64>,           // [state_dim] = [900]
    initial_variance: Vec<f64>,       // [state_dim] = [900]

    // All policy actions (flattened)
    actions: Vec<f64>,                // [n_policies × horizon × state_dim]
                                      // = [5 × 3 × 900] = 13,500

    // Transition dynamics (reused for all policies)
    transition_matrix: Vec<f64>,      // [state_dim × state_dim] = [900 × 900]
                                      // = 810,000 (LARGE!)

    // Observation model (for prediction)
    observation_matrix: Vec<f64>,     // [obs_dim × state_dim] = [100 × 900]
                                      // = 90,000
    observation_noise: Vec<f64>,      // [obs_dim] = [100]

    // Goal state
    preferred_observations: Vec<f64>, // [obs_dim] = [100]

    // Prior for novelty calculation
    prior_mean: Vec<f64>,             // [state_dim] = [900]
    prior_variance: Vec<f64>,         // [state_dim] = [900]
}
```

**Total upload size:** ~918KB per evaluation
- Initial state: 1.8KB × 2 = 3.6KB
- Actions: 13.5K × 8 bytes = 108KB
- Transition matrix: 810K × 8 bytes = 6.48MB ← **LARGE**
- Observation matrix: 90K × 8 bytes = 720KB
- Other: <10KB

**Total: ~7.3MB per evaluation**

⚠️ **CONCERN:** Transition matrix is 6.48MB. Need to verify if this can be cached or simplified.

---

### 2. Intermediate Data (GPU Computation)

```rust
struct GpuTrajectoryData {
    // Future states for all policies × all steps
    future_means: CudaSlice<f64>,     // [n_policies × horizon × state_dim]
                                      // = [5 × 3 × 900] = 13,500

    future_variances: CudaSlice<f64>, // [n_policies × horizon × state_dim]
                                      // = 13,500

    // Predicted observations at each future state
    predicted_obs: CudaSlice<f64>,    // [n_policies × horizon × obs_dim]
                                      // = [5 × 3 × 100] = 1,500

    // Observation uncertainty
    obs_variances: CudaSlice<f64>,    // [n_policies × horizon × obs_dim]
                                      // = 1,500
}
```

**GPU memory required:** ~230KB (persistent allocation)

---

### 3. Output Data (GPU → CPU Download)

```rust
struct GpuPolicyOutputs {
    // EFE components for each policy
    risk: Vec<f64>,                   // [n_policies] = [5]
    ambiguity: Vec<f64>,              // [n_policies] = [5]
    novelty: Vec<f64>,                // [n_policies] = [5]

    // Total EFE = risk + ambiguity - novelty
    total_efe: Vec<f64>,              // [n_policies] = [5]
}
```

**Total download size:** 160 bytes (negligible)

---

## Kernel Design

### Kernel 1: Trajectory Prediction

**Purpose:** Simulate 3-step future given initial state + policy actions

```cuda
__global__ void trajectory_prediction_kernel(
    // Inputs
    const double* initial_mean,        // [state_dim]
    const double* initial_variance,    // [state_dim]
    const double* actions,             // [n_policies × horizon × state_dim]
    const double* transition_matrix,   // [state_dim × state_dim]
    const double process_noise,        // scalar

    // Outputs
    double* future_means,              // [n_policies × horizon × state_dim]
    double* future_variances,          // [n_policies × horizon × state_dim]

    // Dimensions
    int n_policies,
    int horizon,
    int state_dim
)
```

**Parallelization Strategy:**

**Option A: Parallel over (policy, step)**
- Grid: `(n_policies × horizon, 1, 1)` = `(15, 1, 1)`
- Block: `(state_dim, 1, 1)` = `(900, 1, 1)` ← **TOO LARGE** (max 1024 threads)
- **Problem:** 900 threads > 1024 max threads per block

**Option B: Parallel over (policy × step), chunked state_dim**
- Grid: `(n_policies × horizon, 1, 1)` = `(15, 1, 1)`
- Block: `(256, 1, 1)` ← Safe thread count
- Each block handles multiple grid cells, each thread handles chunk of state_dim
- **Chosen:** This approach

**Pseudocode:**
```cuda
int ps_idx = blockIdx.x;  // Policy-step combined index
int policy_idx = ps_idx / horizon;
int step_idx = ps_idx % horizon;

// Each thread handles a chunk of state dimensions
for (int dim = threadIdx.x; dim < state_dim; dim += blockDim.x) {
    const double* prev_mean = (step_idx == 0)
        ? initial_mean
        : &future_means[(policy_idx * horizon + (step_idx-1)) * state_dim];

    const double* prev_variance = (step_idx == 0)
        ? initial_variance
        : &future_variances[(policy_idx * horizon + (step_idx-1)) * state_dim];

    const double* action = &actions[(policy_idx * horizon + step_idx) * state_dim];

    // State evolution: x_{t+1} = A * x_t + B * u_t + noise
    double next_mean = 0.0;
    for (int i = 0; i < state_dim; i++) {
        next_mean += transition_matrix[dim * state_dim + i] * prev_mean[i];
    }
    next_mean += action[dim];

    // Variance propagation: Σ_{t+1} = A * Σ_t * A^T + Q
    // Simplified (diagonal): σ²_{t+1} = σ²_t + process_noise
    double next_variance = prev_variance[dim] + process_noise;

    // Write output
    int out_idx = (policy_idx * horizon + step_idx) * state_dim + dim;
    future_means[out_idx] = next_mean;
    future_variances[out_idx] = next_variance;
}
```

⚠️ **CONCERN:** Inner loop over `state_dim` (900 iterations) is sequential. Matrix-vector multiply not optimized.

**Optimization needed:** Use cuBLAS for matrix-vector products instead.

---

### Kernel 2: Observation Prediction

**Purpose:** Predict observations from future states

```cuda
__global__ void observation_prediction_kernel(
    // Inputs
    const double* future_means,        // [n_policies × horizon × state_dim]
    const double* future_variances,    // [n_policies × horizon × state_dim]
    const double* observation_matrix,  // [obs_dim × state_dim]
    const double* observation_noise,   // [obs_dim]

    // Outputs
    double* predicted_obs,             // [n_policies × horizon × obs_dim]
    double* obs_variances,             // [n_policies × horizon × obs_dim]

    // Dimensions
    int n_policies,
    int horizon,
    int state_dim,
    int obs_dim
)
```

**Parallelization:**
- Grid: `(n_policies × horizon, 1, 1)` = `(15, 1, 1)`
- Block: `(obs_dim, 1, 1)` = `(100, 1, 1)` ← Safe

**Pseudocode:**
```cuda
int ps_idx = blockIdx.x;
int obs_idx = threadIdx.x;

if (obs_idx >= obs_dim) return;

int policy_idx = ps_idx / horizon;
int step_idx = ps_idx % horizon;

const double* state_mean = &future_means[(ps_idx) * state_dim];
const double* state_var = &future_variances[(ps_idx) * state_dim];

// Observation prediction: o = C * x
double pred_obs = 0.0;
for (int j = 0; j < state_dim; j++) {
    pred_obs += observation_matrix[obs_idx * state_dim + j] * state_mean[j];
}

// Variance: σ²_o = C * Σ_x * C^T + R
// Simplified (diagonal): σ²_o = Σ_j (C_ij)² * σ²_x,j + R_i
double pred_var = observation_noise[obs_idx];
for (int j = 0; j < state_dim; j++) {
    double c_ij = observation_matrix[obs_idx * state_dim + j];
    pred_var += c_ij * c_ij * state_var[j];
}

// Write output
int out_idx = ps_idx * obs_dim + obs_idx;
predicted_obs[out_idx] = pred_obs;
obs_variances[out_idx] = pred_var;
```

⚠️ **CONCERN:** Inner loop over `state_dim` (900) again sequential.

---

### Kernel 3: EFE Computation

**Purpose:** Compute risk, ambiguity, novelty for each policy

```cuda
__global__ void compute_efe_kernel(
    // Inputs
    const double* predicted_obs,       // [n_policies × horizon × obs_dim]
    const double* obs_variances,       // [n_policies × horizon × obs_dim]
    const double* preferred_obs,       // [obs_dim]
    const double* future_means,        // [n_policies × horizon × state_dim]
    const double* future_variances,    // [n_policies × horizon × state_dim]
    const double* prior_mean,          // [state_dim]
    const double* prior_variance,      // [state_dim]

    // Outputs
    double* risk_out,                  // [n_policies]
    double* ambiguity_out,             // [n_policies]
    double* novelty_out,               // [n_policies]

    // Dimensions
    int n_policies,
    int horizon,
    int state_dim,
    int obs_dim
)
```

**Parallelization:**
- Grid: `(n_policies, 1, 1)` = `(5, 1, 1)`
- Block: `(256, 1, 1)` ← Multiple threads per policy

**Pseudocode:**
```cuda
int policy_idx = blockIdx.x;
int thread_idx = threadIdx.x;

__shared__ double shared_risk;
__shared__ double shared_ambiguity;
__shared__ double shared_novelty;

if (thread_idx == 0) {
    shared_risk = 0.0;
    shared_ambiguity = 0.0;
    shared_novelty = 0.0;
}
__syncthreads();

// Accumulate over trajectory steps and dimensions
for (int step = 0; step < horizon; step++) {
    int ps_idx = policy_idx * horizon + step;

    // Risk: Σ (o_pred - o_pref)²
    for (int i = thread_idx; i < obs_dim; i += blockDim.x) {
        double error = predicted_obs[ps_idx * obs_dim + i] - preferred_obs[i];
        atomicAdd(&shared_risk, error * error);
    }

    // Ambiguity: Σ σ²_o
    for (int i = thread_idx; i < obs_dim; i += blockDim.x) {
        atomicAdd(&shared_ambiguity, obs_variances[ps_idx * obs_dim + i]);
    }

    // Novelty: H(prior) - H(posterior)
    // H(Gaussian) = 0.5 * ln((2πe)^n * |Σ|)
    // For diagonal: ln|Σ| = Σ ln(σ²)
    for (int i = thread_idx; i < state_dim; i += blockDim.x) {
        double prior_var = prior_variance[i];
        double post_var = future_variances[ps_idx * state_dim + i];
        atomicAdd(&shared_novelty, 0.5 * (log(prior_var) - log(post_var)));
    }
}

__syncthreads();

// Normalize by horizon and write output
if (thread_idx == 0) {
    risk_out[policy_idx] = shared_risk / horizon;
    ambiguity_out[policy_idx] = shared_ambiguity / horizon;
    novelty_out[policy_idx] = shared_novelty / horizon;
}
```

---

## Memory Requirements

### GPU Memory Allocation

**Persistent buffers (reused across evaluations):**
- Trajectory data: 13,500 × 2 × 8 bytes = 216KB
- Observation predictions: 1,500 × 2 × 8 bytes = 24KB
- EFE outputs: 5 × 3 × 8 bytes = 120 bytes
- **Total: ~240KB persistent**

**Per-evaluation upload:**
- Initial state: 1.8KB × 2 = 3.6KB
- Actions: 108KB
- Transition matrix: 6.48MB ← **LARGE**
- Observation matrix: 720KB
- Other: ~10KB
- **Total: ~7.3MB upload**

**Per-evaluation download:**
- EFE values: 120 bytes
- **Total: ~120 bytes download**

### Bandwidth Analysis

**PCIe 3.0 x16:** ~15 GB/s theoretical, ~5-10 GB/s practical

**Upload time:** 7.3MB / 5GB/s = 1.46ms
**Download time:** 120 bytes / 5GB/s = 0.024µs (negligible)

**Kernel execution estimate:**
- Trajectory: ~1ms (15 grid cells × matrix ops)
- Observation: ~0.5ms (15 grid cells × lighter ops)
- EFE: ~0.2ms (5 grid cells × reductions)
- **Total: ~1.7ms kernel time**

**End-to-end estimate:**
- Upload: 1.5ms
- Kernels: 1.7ms
- Download: <0.01ms
- **Total: ~3.2ms**

**Compared to CPU:** 231ms → 3.2ms = **72x speedup**

---

## Critical Issues & Concerns

### 🔴 Issue 1: Transition Matrix Size

**Problem:** 900×900 matrix = 6.48MB upload every evaluation

**Options:**
1. **Cache on GPU:** Upload once, reuse (if matrix doesn't change)
2. **Sparse representation:** If transition matrix is sparse
3. **Simplified dynamics:** Use diagonal or block-diagonal approximation
4. **Implicit operator:** Compute A*x on-the-fly instead of storing A

**Recommendation:** Check if transition matrix changes between evaluations. If not, cache it.

### 🔴 Issue 2: Sequential Matrix-Vector Products

**Problem:** Inner loops in kernels are sequential (900 iterations)

**Options:**
1. **Use cuBLAS:** Call `cublasDgemv` for matrix-vector products
2. **Shared memory:** Cache matrix rows in shared memory
3. **Warp-level primitives:** Use warp shuffle for reductions

**Recommendation:** Replace trajectory prediction with cuBLAS calls.

### 🟡 Issue 3: Atomic Operations

**Problem:** `atomicAdd` in EFE kernel may serialize threads

**Impact:** Moderate (only 256 threads, 3 atomics per step)

**Options:**
1. **Thread-local accumulation:** Each thread accumulates, then reduce
2. **Warp-level reduction:** Use warp shuffle before atomicAdd

**Recommendation:** Implement if profiling shows atomics are bottleneck.

### 🟡 Issue 4: State Dimensions (900) High

**Problem:** 900 state dimensions requires chunking threads

**Impact:** Moderate complexity, but manageable

**Recommendation:** Proceed with chunked approach, optimize if needed.

---

## Validation Strategy

### Phase 1: Correctness

1. **Unit test:** Single policy, single step
   - Input: Known state + action
   - Output: Check trajectory matches CPU
   - Tolerance: <1e-6 error

2. **Integration test:** 5 policies, 3 steps
   - Compare GPU vs CPU EFE values
   - Tolerance: <1% relative error

### Phase 2: Performance

1. **Profile kernels:** `nsys` to measure actual GPU time
2. **Check memory:** Verify no excessive allocations
3. **Bandwidth test:** Measure actual upload/download time

### Phase 3: End-to-End

1. **Replace CPU path:** Wire GPU into `PolicySelector`
2. **Run full pipeline:** `test_full_gpu` example
3. **Verify latency:** Phase 6 should be <10ms

---

## Decision: Proceed or Pivot?

### ✅ Reasons to Proceed

1. **Memory footprint reasonable:** ~7.3MB upload, ~240KB GPU memory
2. **Bandwidth not limiting:** ~1.5ms transfer time acceptable
3. **Expected speedup high:** 231ms → 3.2ms (72x) if estimates hold
4. **Architecture well-defined:** Clear kernel boundaries
5. **Validation plan clear:** Can verify correctness at each step

### ⚠️ Risks to Mitigate

1. **Transition matrix size:** Need to verify if cacheable
2. **Matrix-vector products:** Should use cuBLAS, not naive loops
3. **Complexity:** 3 kernels + Rust wrapper = significant effort (34 hours)

### 🎯 Recommendation

**PROCEED** with GPU implementation, but with modifications:

1. **Start simple:** Implement trajectory prediction with cuBLAS first
2. **Profile early:** Measure actual kernel times after Task 1.1.2
3. **Validate incrementally:** Test each kernel in isolation
4. **Keep CPU fallback:** For validation and debugging

**Next Step:** Task 1.1.1.2 - Identify parallelization strategy (cuBLAS vs custom kernels)

---

## Appendix: Alternative Approaches Considered

### Alternative A: Parallel Policy Evaluation on CPU

**Approach:** Use Rayon to evaluate 5 policies in parallel on CPU

**Pros:**
- Simple (5 lines of code)
- Fast to implement (30 minutes)
- Guaranteed correct (same algorithm)

**Cons:**
- Limited speedup: 231ms / 5 = ~46ms (5x vs 72x for GPU)
- Doesn't leverage GPU

**Verdict:** Quick win, but not ultimate solution. Could implement as intermediate step.

### Alternative B: Reduce Policy Count

**Approach:** Evaluate 1-2 policies instead of 5

**Pros:**
- Trivial to implement (change constant)
- Immediate 2-5x speedup

**Cons:**
- May degrade control quality
- Doesn't scale to harder problems

**Verdict:** Not a real solution. Band-aid fix.

### Alternative C: Simplify EFE Calculation

**Approach:** Use cheaper approximation (e.g., skip novelty term)

**Pros:**
- Reduces computation per policy

**Cons:**
- Changes algorithm behavior
- May impact performance

**Verdict:** Scientifically questionable. Avoid unless desperate.

---

**Document Status:** ✅ Design Complete - Ready for Review
**Next Task:** 1.1.1.2 - Identify parallelization strategy (confirm cuBLAS approach)
**Estimated GPU implementation effort:** 34 hours (confirmed)
**Expected speedup:** 72x (231ms → 3.2ms)
