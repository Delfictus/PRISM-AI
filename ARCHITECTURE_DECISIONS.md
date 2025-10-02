# Architecture Decision Records (ADRs)
## Active Inference Platform

**Format**: [MADR](https://adr.github.io/madr/) - Markdown Architectural Decision Records
**Status**: Living Document
**Last Updated**: 2024-01-28

---

## ADR-001: GPU-First Architecture with No CPU Fallbacks

**Date**: 2024-01-28
**Status**: ✅ Accepted
**Constitution Reference**: Phase 0, Core Principles #4
**Deciders**: Project Team

### Context and Problem Statement

The platform requires high-performance computation for real-time active inference. We need to decide: GPU-mandatory or GPU-optional with CPU fallback?

### Decision Drivers

- Performance requirements (< 10ms end-to-end latency)
- Simplicity (avoiding dual codepaths)
- Constitution mandate (GPU-first architecture)
- DARPA Narcissus real-time requirements

### Considered Options

**Option 1**: GPU-mandatory (chosen)
**Option 2**: GPU-optional with CPU fallback
**Option 3**: Hybrid (GPU for compute, CPU for control)

### Decision Outcome

**Chosen**: Option 1 - GPU-Mandatory Architecture

**Rationale**:
- Simplifies codebase (single execution path)
- Guarantees performance characteristics
- Aligns with constitution mandate
- NVIDIA GPUs widely available
- No performance surprises in production

### Consequences

**Positive**:
- Clear performance guarantees
- Simpler code maintenance
- No CPU optimization needed
- GPU-specific optimizations possible

**Negative**:
- Requires NVIDIA GPU for development
- Cannot run on CPU-only systems
- CI/CD requires GPU runners
- Limits portability

**Mitigation**:
- Provide clear GPU requirements documentation
- Use cloud GPU instances for CI/CD
- Docker containers with GPU support

### Implementation

```rust
// System initialization
pub fn initialize() -> Result<Platform, InitError> {
    let device = CudaDevice::new(0)
        .map_err(|e| InitError::NoGpuAvailable(e))?;

    // No CPU fallback - fail fast
    Ok(Platform::new_gpu_only(device))
}
```

### Validation

- [x] Constitution compliance: ✅
- [x] Performance targets achievable: ✅
- [x] Team consensus: ✅

---

## ADR-002: Information-Theoretic Cross-Domain Coupling

**Date**: 2024-01-28
**Status**: ✅ Accepted
**Constitution Reference**: Phase 3, Task 3.1
**Deciders**: Scientific Advisor, Technical Lead

### Context and Problem Statement

Need to couple neuromorphic and quantum computational domains. How to achieve bidirectional information flow without simulating physical quantum-neural interaction?

### Decision Drivers

- Scientific rigor (no pseudoscience)
- Computational efficiency
- Theoretical soundness
- Measurable coupling strength

### Considered Options

**Option 1**: Information-theoretic coupling (chosen)
**Option 2**: Direct state coupling
**Option 3**: Energy-based coupling
**Option 4**: Physical simulation

### Decision Outcome

**Chosen**: Option 1 - Information-Theoretic Coupling using Transfer Entropy and Mutual Information

**Rationale**:
- Grounded in established information theory
- No need for physical simulation
- Computationally efficient (GPU-parallelizable)
- Quantifiable coupling strength
- Respects causality (time-lagged analysis)

### Mathematical Foundation

```math
Coupling Strength: C_ij = J_ij · tanh(α·TE_{i→j}) · exp(-τ_ij/τ_0)

Where:
- TE_{i→j}: Transfer entropy (information flow i → j)
- τ_ij: Optimal time delay
- α, τ_0: Tunable parameters
```

### Implementation

```rust
pub struct InformationGatedCoupling {
    pub fn compute_coupling(&self,
        te_analysis: &CausalAnalysis,
        phases: &[f64]
    ) -> Array2<f64> {
        // Use transfer entropy to gate coupling strength
    }
}
```

### Consequences

**Positive**:
- Scientifically sound
- No arbitrary formulas
- Measurable and testable
- Efficient computation

**Negative**:
- Requires time series data
- Transfer entropy computation overhead
- Need to tune α, τ_0 parameters

### Validation

- [x] Mathematical proof: ✅ Information theory established
- [x] Computational efficiency: ✅ O(N²) parallelizable
- [x] Scientific accuracy: ✅ No physical violations

---

## ADR-003: Active Inference via Free Energy Minimization

**Date**: 2024-01-28
**Status**: ✅ Accepted
**Constitution Reference**: Phase 2, Task 2.1-2.3
**Deciders**: Scientific Advisor

### Context and Problem Statement

System needs to adapt and learn from data. What learning framework to use that is scientifically grounded and applicable to this architecture?

### Decision Drivers

- Scientific foundation (no arbitrary learning rules)
- Handles uncertainty naturally
- Applicable to both neuromorphic and quantum domains
- Computationally tractable

### Considered Options

**Option 1**: Active Inference / Free Energy Principle (chosen)
**Option 2**: Standard reinforcement learning (Q-learning, PPO)
**Option 3**: Evolutionary algorithms
**Option 4**: Bayesian optimization

### Decision Outcome

**Chosen**: Option 1 - Active Inference with Variational Free Energy Minimization

**Rationale**:
- Established neuroscience framework (Friston et al.)
- Handles both perception and action
- Uncertainty quantification built-in
- Applicable across domains
- Thermodynamically interpretable

### Mathematical Foundation

```math
Variational Free Energy:
F = E_q[log q(x) - log p(x,o)]
  = Surprise + Complexity

Inference: Minimize F w.r.t. q(x)  (perception)
Control: Minimize E[F_future]      (action)
```

### Implementation

```rust
pub struct ActiveInferenceEngine {
    generative_model: GenerativeModel,
    recognition_model: RecognitionModel,

    pub fn minimize_free_energy(&mut self,
        observation: &Observation
    ) -> InferenceResult {
        // Variational inference to minimize F
    }
}
```

### Consequences

**Positive**:
- Principled learning framework
- Handles exploration-exploitation naturally
- Uncertainty quantification
- Theoretical foundation

**Negative**:
- Complex implementation
- Requires generative model
- Parameter tuning needed
- Computational overhead

### Validation

- [x] Theoretical soundness: ✅
- [x] Computational feasibility: ✅
- [ ] Empirical validation: Pending Phase 2

---

## ADR-004: Thermodynamically Consistent Dynamics

**Date**: 2024-01-28
**Status**: ✅ Accepted
**Constitution Reference**: Phase 1, Task 1.3
**Deciders**: Scientific Advisor

### Context and Problem Statement

Oscillator network dynamics must respect physical laws. Previous DRPP implementation violated thermodynamics.

### Decision Drivers

- Scientific correctness (2nd law of thermodynamics)
- Numerical stability
- Physical plausibility
- Computational efficiency

### Considered Options

**Option 1**: Langevin dynamics with thermal noise (chosen)
**Option 2**: Pure Kuramoto (no noise)
**Option 3**: Dissipative Kuramoto
**Option 4**: Original DRPP (rejected - thermodynamic violation)

### Decision Outcome

**Chosen**: Option 1 - Langevin Dynamics with Fluctuation-Dissipation Theorem

**Rationale**:
- Respects 2nd law of thermodynamics
- Fluctuation-dissipation theorem satisfied
- Physically realistic
- Numerically stable

### Mathematical Foundation

```math
dθ_i/dt = ω_i + Σ_j C_ij sin(θ_j - θ_i) - γ·∂S/∂θ_i + √(2γk_BT)·η(t)

Where:
- γ: Damping coefficient (energy dissipation)
- k_B: Boltzmann constant
- T: Temperature
- η(t): White noise (Gaussian)
```

### Implementation

```rust
pub struct ThermodynamicOscillatorNetwork {
    pub fn evolve(&mut self, dt: f64) -> Result<()> {
        // Coupling force
        // Damping term
        // Thermal noise (FDT)
        // Validate: dS/dt >= 0
    }
}
```

### Consequences

**Positive**:
- Thermodynamically consistent
- Entropy production guaranteed non-negative
- Equilibrium distribution correct
- Physical meaning clear

**Negative**:
- Requires temperature parameter
- Noise adds computational cost
- More complex than pure Kuramoto

### Validation

- [x] Thermodynamic consistency: ✅
- [x] Fluctuation-dissipation: ✅
- [ ] Numerical validation: Pending Phase 1

---

## ADR-005: Sparse Matrix Representation for Quantum Systems

**Date**: 2024-01-28 (from Project Vulcan)
**Status**: ✅ Accepted
**Constitution Reference**: Phase 1 (Quantum Stability Fix)
**Deciders**: Technical Lead

### Context and Problem Statement

Dense Hamiltonian matrices cause NaN for N>3 vertices due to numerical instability in high-dimensional spaces.

### Decision Drivers

- Fix critical NaN bug
- Scalability to larger systems
- Numerical stability
- Computational efficiency

### Considered Options

**Option 1**: Sparse matrices with Krylov methods (chosen)
**Option 2**: Dense with better conditioning
**Option 3**: Quantum circuit decomposition
**Option 4**: Approximate methods

### Decision Outcome

**Chosen**: Option 1 - Sparse Matrix + Lanczos/Arnoldi Eigensolvers

**Rationale**:
- Quantum Hamiltonians are naturally sparse (local interactions)
- Krylov methods scale to large systems
- Industry standard for large eigenproblems
- Numerical stability improved

### Implementation

```rust
use sprs::CsMat;  // Compressed sparse row matrix

pub struct SparseHamiltonian {
    matrix: CsMat<Complex64>,

    pub fn diagonalize(&self) -> Result<Eigenvalues> {
        // Use Lanczos iteration for sparse hermitian
    }
}
```

### Consequences

**Positive**:
- Fixes NaN bug
- Scales to N > 10
- Memory efficient
- Faster for sparse systems

**Negative**:
- More complex indexing
- Requires sparse matrix library
- Different API than dense

### Validation

- [ ] NaN fix verified: Pending Phase 1
- [ ] Scalability test: Pending Phase 1
- [ ] Performance benchmark: Pending Phase 1

---

## ADR-006: Hierarchical Active Inference Architecture

**Date**: 2024-01-28
**Status**: ✅ Accepted
**Constitution Reference**: Phase 2, Task 2.1
**Deciders**: Scientific Advisor, Technical Lead

### Context and Problem Statement

Need to handle multi-scale temporal dynamics (microseconds to seconds). Single-scale inference insufficient for complex systems.

### Decision Drivers

- Multi-scale nature of real-world data
- Computational efficiency (parallel levels)
- Theoretical foundation (hierarchical FEP)
- Narcissus application (multiple spatial/temporal scales)

### Considered Options

**Option 1**: Hierarchical multi-level inference (chosen)
**Option 2**: Single-scale inference
**Option 3**: Ensemble of independent models

### Decision Outcome

**Chosen**: Option 1 - Hierarchical Active Inference with Message Passing

**Rationale**:
- Matches neuroscience (cortical hierarchy)
- Computationally efficient (parallel levels)
- Handles multi-scale naturally
- Theoretical foundation (Friston's hierarchical FEP)

### Architecture

```
Level 3: Global Context (seconds)
    ↕ (top-down predictions, bottom-up errors)
Level 2: Meso-scale (milliseconds)
    ↕
Level 1: Local Dynamics (microseconds)
```

### Implementation

```rust
pub struct HierarchicalGenerativeModel {
    levels: Vec<GenerativeLevel>,

    pub fn hierarchical_inference(&mut self) -> Result<()> {
        // Bottom-up pass
        // Top-down pass
        // Lateral pass
        // Converge
    }
}
```

### Consequences

**Positive**:
- Handles multi-scale data
- Parallel computation per level
- Better generalization
- Matches biology

**Negative**:
- More complex implementation
- More parameters to tune
- Higher memory usage

### Validation

- [ ] Multi-scale convergence: Pending Phase 2
- [ ] Performance vs single-scale: Pending Phase 2

---

## ADR Template (For Future Decisions)

```markdown
## ADR-XXX: [Decision Title]

**Date**: YYYY-MM-DD
**Status**: [Proposed/Accepted/Deprecated/Superseded]
**Constitution Reference**: Phase X, Task Y.Z
**Deciders**: [Names/Roles]

### Context and Problem Statement
[Describe the context and problem]

### Decision Drivers
- [Driver 1]
- [Driver 2]

### Considered Options
- Option 1: [Description]
- Option 2: [Description]

### Decision Outcome
**Chosen**: Option X - [Name]

**Rationale**: [Why this option]

### Consequences
**Positive**: [Good results]
**Negative**: [Risks and trade-offs]

### Implementation
```rust
// Code sketch
```

### Validation
- [ ] Criterion 1
- [ ] Criterion 2
```

---

## Decision Index

| ADR | Title | Status | Phase | Impact |
|-----|-------|--------|-------|--------|
| 001 | GPU-First Architecture | Accepted | 0 | High |
| 002 | Information-Theoretic Coupling | Accepted | 3 | High |
| 003 | Active Inference Framework | Accepted | 2 | High |
| 004 | Thermodynamic Dynamics | Accepted | 1 | High |
| 005 | Sparse Quantum Matrices | Accepted | 1 | Medium |
| 006 | Hierarchical Inference | Accepted | 2 | Medium |

---

## Superseded Decisions

### ~~Arbitrary Phase-Causal Matrix Formula~~
**Original**: `Φ_ij = κ·cos(θ_i - θ_j) + β·TE(i→j)`
**Superseded By**: ADR-002 (Information-Theoretic Coupling)
**Reason**: Lacked theoretical justification
**Date Superseded**: 2024-01-28

### ~~DRPP Evolution Without Thermodynamic Terms~~
**Original**: `dθ/dt = ω + Σ Φ·sin(θ_j - θ_i)`
**Superseded By**: ADR-004 (Thermodynamic Dynamics)
**Reason**: Violated 2nd law of thermodynamics
**Date Superseded**: 2024-01-28

---

## Amendment Process

To add new ADR:

1. Copy ADR template
2. Fill in all sections
3. Get technical review
4. Update decision index
5. Commit to git

To supersede existing ADR:

1. Create new ADR
2. Mark old ADR as "Superseded by ADR-XXX"
3. Document migration path
4. Update affected code

---

**All architectural decisions must be documented here before implementation.**
