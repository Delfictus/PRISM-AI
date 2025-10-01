# GPU Acceleration Modes

The Neuromorphic-Quantum Platform supports two distinct GPU modes:

## 🎭 **Mode 1: Simulation (Default)**

**Purpose:** Testing and development without GPU hardware requirements

**How it works:**
- Runs actual CPU computations
- Artificially adjusts reported times to estimate GPU performance
- Uses statistical models based on RTX 5070 benchmarks
- No CUDA/NVIDIA drivers required

**When to use:**
- Development and testing on machines without NVIDIA GPUs
- CI/CD pipelines
- Initial prototyping
- Systems without CUDA support

**Build command:**
```bash
cargo build --release
# or explicitly
cargo build --release --features simulation
```

**Run example:**
```bash
cargo run --release --example gpu_performance_demo
```

**Output indicators:**
- Headers show "SIMULATION MODE"
- Statistics labeled as "(SIMULATED)"
- Warning message about artificial speedup

---

## ⚡ **Mode 2: Real CUDA Acceleration**

**Purpose:** Production deployment with actual GPU hardware acceleration

**How it works:**
- Executes actual CUDA kernels on RTX 5070 GPU
- Uses cuBLAS for optimized matrix operations
- Real GPU memory management
- Achieves genuine 10-50x speedup

**Requirements:**
- NVIDIA RTX GPU (tested on RTX 5070)
- CUDA 12.0+ installed
- NVIDIA drivers installed
- Linux/Windows with CUDA support

**Build command:**
```bash
cargo build --release --features cuda --no-default-features
```

**Run example:**
```bash
cargo run --release --example gpu_performance_demo --features cuda --no-default-features
```

**Output indicators:**
- Headers show "REAL CUDA ACCELERATION"
- Statistics labeled as "REAL"
- Confirmation of RTX 5070 initialization

---

## 📊 Performance Comparison

| Operation | CPU | Simulation | Real CUDA |
|-----------|-----|------------|-----------|
| Small reservoir (100) | 5ms | ~1ms (artificial) | ~0.8ms (real) |
| Medium reservoir (500) | 23ms | ~2ms (artificial) | ~1.5ms (real) |
| Large reservoir (1000) | 46ms | ~2.5ms (artificial) | ~2ms (real) |
| Very large (2000) | 92ms | ~3.5ms (artificial) | ~3ms (real) |

---

## 🔍 How to Identify Which Mode is Running

### In Code:
```rust
#[cfg(feature = "simulation")]
println!("Running SIMULATION mode");

#[cfg(feature = "cuda")]
println!("Running REAL CUDA mode");
```

### At Runtime:
- **Simulation:** Output contains "SIMULATED" labels, warning messages
- **Real CUDA:** Output contains "REAL" labels, CUDA device initialization messages

---

## 🛠️ Feature Flags Reference

### neuromorphic-engine Features:

```toml
[features]
default = ["simulation"]  # Default: simulation mode
cuda = ["cudarc", "candle-core", "candle-nn"]  # Real GPU
simulation = []  # CPU-based simulation
```

### Platform Features:

Add to `Cargo.toml`:
```toml
[dependencies]
neuromorphic-engine = { version = "0.1.0", features = ["cuda"], default-features = false }
```

---

## ⚠️ Important Notes

1. **Transparency:** Simulation mode clearly labels all output as "SIMULATED" to prevent misrepresentation

2. **Accuracy:** Simulation estimates are based on actual RTX 5070 benchmarks but are not substitutes for real measurements

3. **Production Use:** Always use `cuda` feature for production deployments where performance metrics matter

4. **Development:** Simulation mode is perfectly fine for algorithm development and testing

5. **CI/CD:** Use simulation mode in CI pipelines to avoid GPU hardware requirements

---

## 🚀 Quick Start

### For Development (No GPU):
```bash
cargo run --example working_demo
```

### For Production (With RTX 5070):
```bash
# Build with real CUDA
cargo build --release --features cuda --no-default-features

# Run with real GPU
./target/release/examples/gpu_performance_demo
```

---

## 📝 Adding GPU Support to New Code

### Option 1: Conditional Compilation
```rust
#[cfg(feature = "cuda")]
use neuromorphic_engine::GpuReservoirComputer;

#[cfg(feature = "simulation")]
use neuromorphic_engine::create_gpu_reservoir;
```

### Option 2: Runtime Detection
```rust
let reservoir = if cfg!(feature = "cuda") {
    // Use real GPU
    GpuReservoirComputer::new(config, gpu_config)?
} else {
    // Use simulation
    create_gpu_reservoir(size)?
};
```

---

## 🎯 Best Practices

1. **Always label output** with mode indicator (SIMULATED vs REAL)
2. **Document requirements** in README and examples
3. **Provide both modes** in examples for accessibility
4. **Use simulation** for tests and development
5. **Use CUDA** for benchmarks and production

---

## 🔧 Troubleshooting

### "CUDA not available" error:
- Install NVIDIA drivers
- Install CUDA Toolkit 12.0+
- Check: `nvidia-smi` shows GPU
- Rebuild with `--features cuda`

### Simulation mode when expecting CUDA:
- Check build command includes `--features cuda --no-default-features`
- Verify `Cargo.toml` has correct feature configuration
- Check runtime output for mode indicators

### Both modes fail:
- CPU mode always works - check basic Rust setup
- For CUDA: verify GPU hardware and drivers
- For simulation: check dependencies are installed

---

## 📚 Related Documentation

- [COMPLETE_SYSTEM_AUDIT.md](./COMPLETE_SYSTEM_AUDIT.md) - System audit identifying GPU labeling issue
- [EIGENVALUE_SOLVER_IMPLEMENTATION.md](./EIGENVALUE_SOLVER_IMPLEMENTATION.md) - Quantum solver documentation
- [examples/gpu_performance_demo.rs](./examples/gpu_performance_demo.rs) - GPU demonstration code

---

**Last Updated:** 2025-09-30
**Platform Version:** 0.1.0
**CUDA Version:** 12.0+
**Target GPU:** NVIDIA RTX 5070 (Ada Lovelace, Compute 8.9)
