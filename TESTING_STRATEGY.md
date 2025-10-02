# Comprehensive Testing Strategy
## Active Inference Platform

**Constitution Reference**: Phase 0, Task 0.2
**Version**: 1.0.0
**Last Updated**: 2024-01-28

---

## Testing Pyramid (Per Constitution)

```
                    🔺
                   /  \
                  / E2E\           < 5% - End-to-end scenarios
                 /______\
                /        \
               / Integration\      < 15% - Cross-module tests
              /____________\
             /              \
            / Property-Based \     < 20% - Generative testing
           /________________\
          /                  \
         /    Unit Tests       \   < 60% - Focused component tests
        /______________________\
```

---

## Level 1: Mathematical Proofs (Foundation)

### Purpose
Every algorithm must have formal mathematical proof of correctness.

### Implementation
```rust
// src/mathematics/proof_system.rs
pub trait MathematicalStatement {
    /// LaTeX representation for documentation
    fn latex(&self) -> String;

    /// Computational verification
    fn verify(&self) -> ProofResult;

    /// Required assumptions
    fn assumptions(&self) -> Vec<Assumption>;
}
```

### Examples

**Theorem: Entropy Production Non-Negativity**
```rust
#[test]
fn prove_entropy_production_non_negative() {
    let theorem = EntropyProductionTheorem::new();

    // Analytical proof
    let analytical = theorem.derive_from_first_principles();
    assert!(analytical.verified);

    // Numerical verification over 1M iterations
    let numerical = theorem.verify_numerically(1_000_000);
    assert!(numerical.entropy_production >= -1e-10);  // Numerical tolerance

    // Property: dS/dt >= 0 for ALL valid states
    assert!(theorem.verify().passed);
}
```

**Required Coverage**:
- [ ] Every algorithm has proof
- [ ] Both analytical and numerical verification
- [ ] Edge cases proven correct

---

## Level 2: Unit Tests (60% of test suite)

### Purpose
Test individual functions and components in isolation.

### Standards
- **Coverage Target**: >95% per module
- **Assertion Quality**: Test mathematical properties, not just values
- **Error Paths**: Test all error conditions

### Template
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basic_functionality() {
        // Arrange
        let input = create_test_input();

        // Act
        let result = function_under_test(input).unwrap();

        // Assert - mathematical properties
        assert!(result.satisfies_invariant());
        assert_relative_eq!(result.value, expected, epsilon = 1e-9);
    }

    #[test]
    fn test_edge_cases() {
        // Test: Zero input
        assert!(function(0.0).is_ok());

        // Test: Maximum input
        assert!(function(f64::MAX).is_err());

        // Test: Negative input (if applicable)
        // Test: Boundary conditions
    }

    #[test]
    #[should_panic(expected = "Invalid input")]
    fn test_error_conditions() {
        function_under_test(invalid_input()).unwrap();
    }
}
```

### Required Tests Per Module

**For Transfer Entropy Module**:
```rust
#[test]
fn test_known_causal_system() {
    // Y(t+5) = X(t) + noise
    // Should detect: TE_{X→Y}(τ=5) >> TE_{Y→X}
}

#[test]
fn test_independent_series() {
    // Uncorrelated series
    // Should detect: TE ≈ 0
}

#[test]
fn test_statistical_significance() {
    // With surrogate testing
    // Should detect: p-value for spurious causality
}

#[test]
fn test_gpu_cpu_consistency() {
    // Results should match within ε = 1e-5
}
```

---

## Level 3: Property-Based Tests (20% of test suite)

### Purpose
Test mathematical properties across randomly generated inputs.

### Implementation
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn thermodynamic_consistency(
        state in valid_thermodynamic_state(),
        coupling in valid_coupling_matrix()
    ) {
        let mut network = ThermodynamicNetwork::new(state);
        let initial_entropy = network.compute_entropy();

        network.evolve(&coupling, 0.01)?;

        let final_entropy = network.compute_entropy();

        // Property: Entropy never decreases
        prop_assert!(final_entropy >= initial_entropy - 1e-10);
    }
}
```

### Strategy Generators
```rust
// Generate valid system states
fn valid_thermodynamic_state() -> impl Strategy<Value = SystemState> {
    (
        prop::collection::vec(0.0..2.0*PI, 10..1000),  // phases
        prop::collection::vec(0.1..10.0, 10..1000),    // frequencies
        0.1..5.0,  // temperature (must be > 0)
    ).prop_map(|(phases, freqs, temp)| {
        SystemState { phases, frequencies: freqs, temperature: temp }
    })
}
```

### Required Properties

**Information Theory**:
- Transfer entropy ≥ 0
- Mutual information ≥ 0
- Data processing inequality holds

**Thermodynamics**:
- Entropy production ≥ 0
- Free energy decreases or stable
- Equilibrium distribution correct

**Quantum Mechanics**:
- Density matrix trace = 1
- Density matrix positive semi-definite
- Unitarity preserved

---

## Level 4: Integration Tests (15% of test suite)

### Purpose
Test interactions between modules and subsystems.

### Implementation
```rust
#[test]
fn test_neuromorphic_quantum_coupling() {
    // Setup both subsystems
    let neuro = NeuromorphicEngine::new_for_test()?;
    let quantum = QuantumEngine::new_for_test()?;
    let bridge = CrossDomainBridge::new();

    // Process input through both domains
    let input = generate_test_input();
    let neuro_output = neuro.process(&input)?;
    let quantum_output = quantum.process(&input)?;

    // Test bidirectional coupling
    let sync_result = bridge.synchronize(&neuro_output, &quantum_output)?;

    // Validate cross-domain properties
    assert!(sync_result.mutual_information > 0.5);
    assert!(sync_result.phase_coherence > 0.8);
    assert!(sync_result.causal_consistency_maintained());
}
```

### Required Integration Tests

**Cross-Domain Information Flow**:
```rust
#[test]
fn test_information_flow_neuromorphic_to_quantum() {
    // Information should transfer without loss > 10%
}

#[test]
fn test_information_flow_quantum_to_neuromorphic() {
    // Bidirectional should be symmetric in information content
}
```

**Phase Synchronization**:
```rust
#[test]
fn test_phase_synchronization_convergence() {
    // Phases should converge within 1000 iterations
}
```

**Active Inference Loop**:
```rust
#[test]
fn test_complete_inference_cycle() {
    // Observation → Inference → Action → New Observation
    // Free energy should decrease
}
```

---

## Level 5: End-to-End Tests (5% of test suite)

### Purpose
Test complete system workflows in realistic scenarios.

### Implementation
```rust
#[test]
#[ignore]  // Long-running test
fn test_complete_pipeline_1000_iterations() {
    let mut platform = UnifiedPlatform::new()?;

    let mut free_energies = Vec::new();

    for i in 0..1000 {
        let input = generate_realistic_input(i);
        let output = platform.process_integrated(&input)?;

        // Validate output quality
        assert!(!output.has_nan());
        assert!(output.thermodynamic_valid());
        assert!(output.information_preserved());

        free_energies.push(output.free_energy);
    }

    // Free energy should generally decrease (learning)
    let trend = linear_regression(&free_energies);
    assert!(trend.slope < 0.0);  // Negative slope = decreasing
}
```

---

## Performance Testing

### Benchmark Strategy
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_transfer_entropy(c: &mut Criterion) {
    let series = generate_time_series(4096, 2);

    c.bench_function("transfer_entropy_4096_samples", |b| {
        b.iter(|| {
            let analyzer = TransferEntropyEngine::new(config);
            analyzer.compute_te_matrix(black_box(&series))
        });
    });

    // Assert performance contract
    let measurement = c.benchmark_group("transfer_entropy").sample_size(100);
    assert!(measurement.mean_time() < Duration::from_millis(20));
}

criterion_group!(benches, bench_transfer_entropy);
criterion_main!(benches);
```

### Performance Contracts (From Constitution)

| Component | Max Latency | Min Throughput | Must Pass |
|-----------|-------------|----------------|-----------|
| Transfer Entropy | 20ms | 10K pairs/s | ✅ Required |
| Thermodynamic Evolution | 1ms | 1024 osc/step | ✅ Required |
| Active Inference | 5ms | 200 infer/s | ✅ Required |
| Cross-Domain Bridge | 1ms | 1K xfer/s | ✅ Required |
| End-to-End Pipeline | 10ms | 100 fps | ✅ Required |

---

## GPU Testing

### GPU Kernel Validation
```rust
#[cfg(feature = "cuda")]
#[test]
fn test_kernel_cpu_gpu_consistency() {
    let input = generate_test_data(1024);

    // CPU reference implementation
    let cpu_result = compute_cpu(&input);

    // GPU implementation
    let gpu_result = compute_gpu(&input)?;

    // Copy GPU result to host
    let gpu_host = gpu_result.to_host()?;

    // Assert consistency
    for i in 0..1024 {
        assert_relative_eq!(
            cpu_result[i],
            gpu_host[i],
            epsilon = 1e-5
        );
    }
}
```

### GPU Performance Tests
```rust
#[bench]
#[cfg(feature = "cuda")]
fn bench_gpu_kernel_occupancy(b: &mut Bencher) {
    let device = CudaDevice::new(0)?;

    b.iter(|| {
        // Launch kernel
        // Measure occupancy
    });

    // Assert > 80% occupancy per constitution
    assert!(measured_occupancy > 0.80);
}
```

---

## Test Data Management

### Golden Test Files
```
tests/
├── golden/
│   ├── transfer_entropy_known_system.json
│   ├── thermodynamic_equilibrium.json
│   ├── quantum_ground_state.json
│   └── active_inference_trace.json
└── generators/
    ├── synthetic_data.rs
    ├── chaotic_systems.rs
    └── noise_models.rs
```

### Synthetic Data Generators
```rust
// tests/generators/synthetic_data.rs
pub fn generate_causal_timeseries(
    length: usize,
    lag: usize,
    noise_level: f64
) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; length];
    let mut y = vec![0.0; length];

    // X is random
    for t in 0..length {
        x[t] = rand::random();
    }

    // Y depends on X with delay
    for t in lag..length {
        y[t] = x[t - lag] + noise_level * rand::random();
    }

    (x, y)
}
```

---

## Validation Gates (Blocking)

### Gate 1: Mathematical Correctness
```bash
cargo test --package mathematics --features prove
```
**Blocks if**: Any mathematical proof fails

### Gate 2: Scientific Accuracy
```bash
cargo test --package validation --features scientific
```
**Blocks if**: Thermodynamic laws violated, information bounds broken

### Gate 3: Code Quality
```bash
cargo clippy -- -D warnings
cargo fmt --check
```
**Blocks if**: Warnings present, formatting incorrect

### Gate 4: Performance Contracts
```bash
cargo bench --no-fail-fast
```
**Blocks if**: Any benchmark exceeds latency contract

### Gate 5: Test Coverage
```bash
cargo tarpaulin --out Stdout --fail-under 95
```
**Blocks if**: Coverage < 95%

---

## Continuous Integration Pipeline

### On Every Push
```yaml
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Check constitution integrity
        run: sha256sum -c IMPLEMENTATION_CONSTITUTION.md.sha256

      - name: Run unit tests
        run: cargo test --all

      - name: Check code quality
        run: cargo clippy -- -D warnings

      - name: Verify formatting
        run: cargo fmt --check
```

### On Pull Request
```yaml
jobs:
  full_validation:
    runs-on: [self-hosted, gpu]
    steps:
      - name: Run mathematical proofs
        run: cargo test --package mathematics --features prove

      - name: Run property tests
        run: cargo test --release -- --ignored

      - name: Run benchmarks
        run: cargo bench --no-fail-fast

      - name: Measure coverage
        run: cargo tarpaulin --fail-under 95

      - name: GPU kernel validation
        run: cargo test --features cuda
```

---

## Test Organization

### Directory Structure
```
tests/
├── unit/                      # Module-level tests
│   ├── transfer_entropy_test.rs
│   ├── thermodynamic_network_test.rs
│   └── active_inference_test.rs
├── integration/               # Cross-module tests
│   ├── neuromorphic_quantum_integration.rs
│   ├── information_flow_test.rs
│   └── end_to_end_pipeline.rs
├── property/                  # Property-based tests
│   ├── thermodynamic_properties.rs
│   ├── information_properties.rs
│   └── quantum_properties.rs
├── benchmarks/                # Performance tests
│   ├── transfer_entropy_bench.rs
│   ├── gpu_kernel_bench.rs
│   └── end_to_end_bench.rs
├── golden/                    # Reference data
│   └── *.json
└── generators/                # Test data generators
    └── *.rs
```

---

## Test Naming Conventions

```rust
// Unit tests
#[test]
fn test_<component>_<behavior>_<expected_outcome>()

// Examples:
fn test_transfer_entropy_detects_known_causality()
fn test_oscillator_respects_thermodynamic_laws()
fn test_inference_decreases_free_energy()

// Property tests
#[proptest]
fn prop_<property>_<condition>(inputs: Strategy)

// Examples:
fn prop_entropy_never_decreases(state: ValidState)
fn prop_information_preserved(data: TimeSeries)

// Integration tests
#[test]
fn integration_<subsystem1>_<subsystem2>_<interaction>()

// Examples:
fn integration_neuromorphic_quantum_information_transfer()
fn integration_inference_control_feedback_loop()
```

---

## Assertion Strategies

### Mathematical Properties
```rust
// Good: Test invariants
assert!(entropy_production >= 0.0);
assert_relative_eq!(trace, 1.0, epsilon = 1e-12);
assert!(mutual_info >= 0.0 && mutual_info <= min_entropy);

// Bad: Test specific values without justification
assert_eq!(result, 42.0);  // Why 42.0?
```

### Error Handling
```rust
// Good: Test specific error types
assert!(matches!(
    result,
    Err(ThermodynamicError::EntropyDecrease(_))
));

// Bad: Just check for error
assert!(result.is_err());
```

### Floating Point Comparisons
```rust
// Good: Use appropriate epsilon
assert_relative_eq!(a, b, epsilon = 1e-9);
assert_abs_diff_eq!(a, b, epsilon = 1e-12);

// Bad: Direct equality
assert_eq!(0.1 + 0.2, 0.3);  // Fails due to floating point
```

---

## Test Fixtures and Utilities

### Common Test Utilities
```rust
// tests/common/mod.rs
pub fn create_test_network(n: usize) -> ThermodynamicNetwork {
    let mut rng = StdRng::seed_from_u64(42);  // Reproducible
    ThermodynamicNetwork::new_random(n, &mut rng)
}

pub fn generate_chaotic_lorenz_attractor(steps: usize) -> Vec<Vec<f64>> {
    // Lorenz system for testing chaos handling
}

pub fn assert_thermodynamic_consistency(network: &ThermodynamicNetwork) {
    assert!(network.entropy_production_rate() >= 0.0);
    assert!(network.free_energy().is_finite());
    assert!(network.temperature() > 0.0);
}
```

---

## GPU Testing Strategy

### CPU Reference Implementation Required
Every GPU kernel must have CPU reference for validation:

```rust
// CPU reference (slow but correct)
fn transfer_entropy_cpu(x: &[f64], y: &[f64], lag: usize) -> f64 {
    // Straightforward implementation
}

// GPU implementation (fast)
fn transfer_entropy_gpu(x: &CudaSlice<f64>, y: &CudaSlice<f64>, lag: usize) -> f64 {
    // Optimized CUDA kernel
}

#[test]
fn test_gpu_matches_cpu() {
    let (x, y) = generate_test_series();

    let cpu_result = transfer_entropy_cpu(&x, &y, 5);
    let gpu_result = transfer_entropy_gpu(&x_gpu, &y_gpu, 5);

    assert_relative_eq!(cpu_result, gpu_result, epsilon = 1e-5);
}
```

---

## Regression Testing

### Prevent Known Bugs from Returning
```rust
#[test]
fn regression_quantum_nan_for_n_greater_than_3() {
    // This bug was fixed in Project Vulcan
    // This test ensures it never returns

    for n in 4..=10 {
        let result = quantum_solver.solve(n)?;
        assert!(!result.has_nan(), "NaN bug returned for n={}", n);
    }
}
```

### Golden File Testing
```rust
#[test]
fn test_against_golden_data() {
    let golden = load_golden_test("transfer_entropy_known_system.json");
    let result = analyzer.compute(&golden.input)?;

    assert_relative_eq!(
        result.te_matrix,
        golden.expected_output,
        epsilon = 1e-4
    );
}
```

---

## Test Execution Strategy

### Local Development
```bash
# Fast feedback loop
cargo test --lib                # Unit tests only
cargo test --package <module>   # Specific module

# Before commit
cargo test --all                # All tests
cargo clippy -- -D warnings     # Linting
```

### CI/CD Pipeline
```bash
# Full validation
cargo test --all --release
cargo test -- --ignored         # Long-running tests
cargo bench                     # Performance validation
cargo tarpaulin                 # Coverage measurement
```

### Pre-Release
```bash
# Comprehensive validation
cargo test --all-features --release
cargo bench --no-fail-fast
./scripts/run_validation_suite.sh  # All gates
```

---

## Coverage Requirements

### By Module Type

**Core Algorithms (95%+ coverage required)**:
- Transfer entropy
- Thermodynamic network
- Active inference
- Quantum solver

**Integration Layer (90%+ coverage required)**:
- Cross-domain bridge
- Information channels
- Synchronization

**Infrastructure (85%+ coverage required)**:
- Error handling
- Validation gates
- Compliance engine

### Measuring Coverage
```bash
cargo tarpaulin --out Stdout --fail-under 95
cargo tarpaulin --out Html
# View: target/tarpaulin/index.html
```

---

## Test Maintenance

### When to Update Tests

**Always**:
- New feature added → New tests required
- Bug fixed → Regression test added
- Performance improved → Benchmark updated
- Constitution amended → Validation updated

**Review Cycle**:
- Weekly: Check for flaky tests
- Monthly: Audit test coverage
- Per phase: Update test strategy

---

## Mutation Testing (Advanced)

### Purpose
Test the quality of tests themselves.

```bash
cargo mutants --test-timeout 300
```

### Expectation
- 90%+ mutations should be caught by tests
- If mutation survives, tests are insufficient

---

## Troubleshooting Tests

### Flaky Tests
```rust
// Bad: Non-deterministic
let random_value = rand::random();

// Good: Seeded randomness
let mut rng = StdRng::seed_from_u64(42);
let value = rng.gen();
```

### Slow Tests
```rust
// Use #[ignore] for long-running tests
#[test]
#[ignore]
fn test_million_iteration_convergence() {
    // Only run in CI or with --ignored flag
}
```

### GPU Tests Without GPU
```rust
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_specific() {
    // Only runs when CUDA feature enabled
}
```

---

## Documentation Testing

### Doc Tests
```rust
/// Compute transfer entropy
///
/// # Examples
/// ```
/// use crate::TransferEntropyEngine;
///
/// let analyzer = TransferEntropyEngine::new(config);
/// let result = analyzer.compute(&x, &y, lag);
/// assert!(result > 0.0);
/// ```
pub fn compute_transfer_entropy() {}
```

### Run Doc Tests
```bash
cargo test --doc
```

---

## Phase-Specific Testing

### Phase 1: Mathematical Foundation
**Focus**: Proofs, numerical stability, mathematical correctness

### Phase 2: Active Inference
**Focus**: Free energy convergence, learning behavior

### Phase 3: Integration
**Focus**: Information flow, synchronization, end-to-end

### Phase 4: Production
**Focus**: Error recovery, performance, reliability

### Phase 5: Validation
**Focus**: DARPA demo scenarios, scientific validation

---

## Success Criteria

Before ANY phase can be marked complete:

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] All property tests passing
- [ ] All benchmarks meeting contracts
- [ ] Coverage > 95%
- [ ] No clippy warnings
- [ ] Documentation tests passing
- [ ] GPU-CPU consistency verified
- [ ] Scientific validation passed

---

## Test Execution Commands

```bash
# Quick validation
cargo test

# Full validation
cargo test --all --release

# With coverage
cargo tarpaulin --fail-under 95

# Performance
cargo bench

# GPU tests
cargo test --features cuda

# Long-running tests
cargo test -- --ignored

# Specific test
cargo test test_name -- --nocapture

# Documentation tests
cargo test --doc
```

---

**Testing is not optional. It is mandatory per the constitution.**

Every component must have comprehensive tests before it can be considered complete.
