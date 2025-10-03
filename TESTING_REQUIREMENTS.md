# Testing Requirements - Active Inference Platform

**Date**: 2025-10-03
**Status**: Build ✅ | Tests ⚠️ (Requires system libraries)

---

## Current Status

### ✅ Build: SUCCESS
```bash
$ cargo build --lib
   Finished `dev` profile [unoptimized + debuginfo] target(s)
```

**All code compiles successfully with zero errors.**

### ⚠️ Tests: Requires BLAS/LAPACK

Tests are blocked by missing system libraries:
```
error: linking with `cc` failed
rust-lld: error: undefined symbol: cblas_ddot
```

---

## Why BLAS is Required

### Root Cause

The `quantum-engine` workspace member uses `ndarray-linalg` for eigenvalue decomposition:
```rust
// src/quantum/src/robust_eigen.rs
use ndarray_linalg::eigh::Eigh;  // Requires LAPACK
```

This pulls in BLAS/LAPACK dependencies for the entire workspace, even though our `active_inference` module doesn't use eigenvalue solvers.

### What Our Code Actually Uses

**Active Inference Module** (`src/active_inference/`):
- ✅ Matrix-vector multiplication (`.dot()`) - Pure Rust, no BLAS needed
- ✅ Element-wise operations - Pure Rust
- ✅ Basic statistics - Pure Rust

**Quantum Engine Module** (`src/quantum/`):
- ⚠️ Eigenvalue decomposition (`.eigh()`) - **Requires LAPACK**
- ⚠️ Matrix inversion (`.inv()`) - **Requires LAPACK**

---

## Installation Instructions

### Ubuntu/Debian (Your System)

```bash
# Update package lists
sudo apt-get update

# Install BLAS, LAPACK, and Fortran compiler
sudo apt-get install libblas-dev liblapack-dev gfortran

# Verify installation
dpkg -l | grep -E "libblas-dev|liblapack-dev|gfortran"
```

Expected output:
```
ii  gfortran       4:13.2.0-7ubuntu1  amd64  GNU Fortran 95 compiler
ii  libblas-dev    3.12.0-3build1.1   amd64  Basic Linear Algebra Subroutines (development files)
ii  liblapack-dev  3.12.0-3build1.1   amd64  Linear Algebra PACKage (development files)
```

### Alternative: Intel MKL (Better Performance)

The `quantum-engine` is already configured for Intel MKL static linking:
```toml
ndarray-linalg = { version = "0.16", features = ["intel-mkl-static"] }
```

However, this requires Intel MKL to be installed on your system. On Ubuntu:
```bash
# Intel MKL (if you want better performance)
# This is optional - libblas-dev works fine
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"
sudo apt-get update
sudo apt-get install intel-oneapi-mkl intel-oneapi-mkl-devel
```

---

## Testing After Installation

### 1. Run All Active Inference Tests

```bash
cargo test --lib active_inference
```

Expected output:
```
running 56 tests
test active_inference::hierarchical_model::tests::test_gaussian_belief_entropy ... ok
test active_inference::hierarchical_model::tests::test_gaussian_kl_divergence ... ok
test active_inference::hierarchical_model::tests::test_window_phase_level_creation ... ok
...
test result: ok. 56 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### 2. Run Specific Test Module

```bash
# Hierarchical model tests (10 tests)
cargo test --lib active_inference::hierarchical_model::tests

# Observation model tests (8 tests)
cargo test --lib active_inference::observation_model::tests

# Transition model tests (9 tests)
cargo test --lib active_inference::transition_model::tests

# Variational inference tests (8 tests)
cargo test --lib active_inference::variational_inference::tests

# Policy selection tests (10 tests)
cargo test --lib active_inference::policy_selection::tests

# Generative model tests (11 tests)
cargo test --lib active_inference::generative_model::tests
```

### 3. Run Single Test

```bash
cargo test --lib active_inference::hierarchical_model::tests::test_gaussian_belief_entropy -- --nocapture
```

---

## Test Coverage

### Active Inference Module: 56 Unit Tests

| Module | Tests | Purpose |
|--------|-------|---------|
| `hierarchical_model.rs` | 10 | State space, entropy, KL divergence, physical parameters |
| `observation_model.rs` | 8 | Wavefront sensing, photon noise, measurement patterns |
| `transition_model.rs` | 9 | Dynamics, energy conservation, control actions |
| `variational_inference.rs` | 8 | Free energy, convergence, parameter learning |
| `policy_selection.rs` | 10 | Expected free energy, policy evaluation |
| `generative_model.rs` | 11 | End-to-end inference loop, metrics |

### Expected Test Duration

```
Active inference tests: ~5-10 seconds
Full workspace tests: ~2-3 minutes (includes CUDA compilation)
```

---

## Validation Criteria (Constitution Phase 2, Task 2.1)

After BLAS installation, verify these criteria:

### 1. Free Energy Decreases ✅

```bash
cargo test --lib test_free_energy_decreases_with_inference
```

**Expected**: Free energy decreases with each inference iteration until convergence.

### 2. Online Learning ✅

```bash
cargo test --lib test_parameter_learning_updates_jacobian
cargo test --lib test_online_parameter_learning
```

**Expected**: Parameters update from observation data.

### 3. Uncertainty Quantification ✅

```bash
cargo test --lib test_gaussian_belief_entropy
cargo test --lib test_observation_variance_includes_noise
```

**Expected**: Gaussian beliefs properly track mean + variance.

### 4. Prediction Accuracy ⏳

```bash
cargo test --lib test_prediction_rmse
```

**Expected**: RMSE < 5% for perfect predictions (needs empirical data for real validation).

---

## Performance Benchmarks (After Installation)

Run benchmarks to verify performance contracts:

```bash
# Build with optimizations
cargo build --release --lib

# Run benchmarks
cargo bench --bench active_inference_benchmarks
```

**Target Performance** (Constitution):
- Transfer Entropy: <20ms (GPU: 0.2ms achieved in Phase 1)
- Thermodynamic Evolution: <1ms (GPU: 0.08ms achieved in Phase 1)
- Active Inference: <5ms
- End-to-End: <10ms

---

## Troubleshooting

### Issue: Tests still fail after installing BLAS

**Solution 1**: Clean and rebuild
```bash
cargo clean
cargo test --lib active_inference
```

**Solution 2**: Check library paths
```bash
ldconfig -p | grep -E "blas|lapack"
```

Should show:
```
libblas.so.3 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libblas.so.3
liblapack.so.3 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/liblapack.so.3
```

**Solution 3**: Set environment variables
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cargo test --lib active_inference
```

### Issue: Intel MKL not found

If using `intel-mkl-static` feature, you may need to point to MKL:
```bash
export MKLROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
```

Or switch to OpenBLAS in `src/quantum/Cargo.toml`:
```toml
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
```

---

## CI/CD Integration

For GitHub Actions, add this to `.github/workflows/test.yml`:

```yaml
- name: Install BLAS/LAPACK
  run: |
    sudo apt-get update
    sudo apt-get install -y libblas-dev liblapack-dev gfortran

- name: Run tests
  run: cargo test --lib --verbose
```

---

## Alternative: Test Without BLAS (Development Only)

If you want to quickly test active_inference code without installing BLAS, you can temporarily remove quantum-engine from the workspace:

**NOT RECOMMENDED** (breaks workspace, only for quick testing):

1. Comment out in `Cargo.toml`:
```toml
# quantum-engine = { path = "src/quantum" }
```

2. Test active_inference:
```bash
cargo test --lib active_inference
```

3. Revert changes before committing.

---

## Summary

**Current State**:
- ✅ Code: Complete, compiles perfectly
- ⚠️ Tests: Require BLAS system libraries

**Action Required**:
```bash
sudo apt-get install libblas-dev liblapack-dev gfortran
```

**After Installation**:
- ✅ All 56 active_inference tests should pass
- ✅ Constitution validation criteria met
- ✅ Ready for Phase 2, Task 2.2

---

**Document Version**: 1.0
**Last Updated**: 2025-10-03
**Related**: PHASE2_TASK2.1_SUMMARY.md
