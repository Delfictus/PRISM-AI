# Clean Architecture - Hexagonal Pattern

## Overview

This project implements **Hexagonal Architecture** (Ports & Adapters) to achieve:
- **Zero circular dependencies** via `shared-types`
- **Infrastructure-agnostic domain logic** in `prct-core`
- **Pluggable implementations** via `prct-adapters`

## Architecture Layers

```
┌────────────────────────────────────────────────────────┐
│                  DOMAIN LAYER (prct-core)              │
│  • PRCTAlgorithm - Pure business logic                │
│  • Ports (interfaces): NeuromorphicPort, QuantumPort  │
│  • Phase-guided coloring & TSP algorithms             │
│  • NO infrastructure dependencies                      │
└─────────────────────┬──────────────────────────────────┘
                      │ depends on
                      ▼
┌────────────────────────────────────────────────────────┐
│            SHARED TYPES (shared-types)                 │
│  • Graph, QuantumState, NeuroState, SpikePattern      │
│  • ColoringSolution, TSPSolution, PRCTSolution        │
│  • ZERO dependencies - pure data types                │
└────────────────────────────────────────────────────────┘
                      ▲
                      │ implements ports for
                      │
┌─────────────────────┴──────────────────────────────────┐
│         INFRASTRUCTURE LAYER (prct-adapters)           │
│  • NeuromorphicAdapter → neuromorphic-engine           │
│  • QuantumAdapter → quantum-engine                     │
│  • CouplingAdapter → platform-foundation               │
│  • Implements port interfaces from prct-core           │
└────────────────────────────────────────────────────────┘
```

## Key Components

### 1. `shared-types` - Foundation
**Purpose:** Zero-dependency data types preventing circular deps

**Contents:**
- `neuro_types.rs` - NeuroState, SpikePattern, Spike
- `quantum_types.rs` - QuantumState, HamiltonianState, PhaseField
- `coupling_types.rs` - CouplingStrength, KuramotoState, TransferEntropy
- `graph_types.rs` - Graph, ColoringSolution, TSPSolution, PRCTSolution

**Guarantees:**
- NO external dependencies (not even `std` - uses `core` + `alloc`)
- Pure data structures only
- Optional serde feature for serialization

### 2. `prct-core` - Domain Logic
**Purpose:** Infrastructure-agnostic PRCT algorithm

**Key Files:**
- `algorithm.rs` - PRCTAlgorithm with dependency injection
- `ports.rs` - Port trait definitions (NeuromorphicPort, QuantumPort, PhysicsCouplingPort)
- `coloring.rs` - Phase-guided graph coloring
- `tsp.rs` - Phase-guided TSP optimization
- `errors.rs` - Domain error types

**Port Interfaces:**
```rust
pub trait NeuromorphicPort: Send + Sync {
    fn encode_graph_as_spikes(&self, graph: &Graph, params: &NeuromorphicEncodingParams) -> Result<SpikePattern>;
    fn process_and_detect_patterns(&self, spikes: &SpikePattern) -> Result<NeuroState>;
    fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>>;
}

pub trait QuantumPort: Send + Sync {
    fn build_hamiltonian(&self, graph: &Graph, params: &EvolutionParams) -> Result<HamiltonianState>;
    fn evolve_state(&self, hamiltonian: &HamiltonianState, initial: &QuantumState, time: f64) -> Result<QuantumState>;
    fn get_phase_field(&self, state: &QuantumState) -> Result<PhaseField>;
    fn compute_ground_state(&self, hamiltonian: &HamiltonianState) -> Result<QuantumState>;
}

pub trait PhysicsCouplingPort: Send + Sync {
    fn compute_coupling(&self, neuro: &NeuroState, quantum: &QuantumState) -> Result<CouplingStrength>;
    fn update_kuramoto_sync(&self, neuro_phases: &[f64], quantum_phases: &[f64], dt: f64) -> Result<KuramotoState>;
    fn calculate_transfer_entropy(&self, source: &[f64], target: &[f64], lag: f64) -> Result<TransferEntropy>;
    fn get_bidirectional_coupling(&self, neuro: &NeuroState, quantum: &QuantumState) -> Result<BidirectionalCoupling>;
}
```

### 3. `prct-adapters` - Infrastructure
**Purpose:** Connect domain logic to real engines

**Adapters:**
- `NeuromorphicAdapter` - Wraps `neuromorphic-engine`
  - Uses SpikeEncoder for encoding
  - Uses ReservoirComputer for processing
  - Converts between shared-types and engine types

- `QuantumAdapter` - Wraps `quantum-engine`
  - Uses Hamiltonian for quantum mechanics
  - Uses ForceFieldParams for physics
  - Handles PhaseResonanceField

- `CouplingAdapter` - Wraps `platform-foundation`
  - Uses PhysicsCoupling for bidirectional info flow
  - Implements Kuramoto synchronization
  - Computes transfer entropy

## Dependency Flow

```
neuromorphic-quantum-platform (main crate)
├─> shared-types (zero deps)
├─> prct-core
│   └─> shared-types
├─> prct-adapters
│   ├─> prct-core (for port traits)
│   ├─> shared-types (for data types)
│   ├─> neuromorphic-engine
│   ├─> quantum-engine
│   └─> platform-foundation
├─> neuromorphic-engine
├─> quantum-engine
└─> platform-foundation
```

**Key:** NO circular dependencies - dependency graph is a DAG

## Usage Example

```rust
use prct_core::{PRCTAlgorithm, PRCTConfig};
use prct_adapters::{NeuromorphicAdapter, QuantumAdapter, CouplingAdapter};
use shared_types::Graph;
use std::sync::Arc;

// 1. Create graph
let graph = Graph { /* ... */ };

// 2. Create adapters (infrastructure layer)
let neuro_adapter = Arc::new(NeuromorphicAdapter::new()?);
let quantum_adapter = Arc::new(QuantumAdapter::new());
let coupling_adapter = Arc::new(CouplingAdapter::new());

// 3. Create PRCT with dependency injection (domain layer)
let config = PRCTConfig::default();
let prct = PRCTAlgorithm::new(
    neuro_adapter,
    quantum_adapter,
    coupling_adapter,
    config,
);

// 4. Solve
let solution = prct.solve(&graph)?;

// 5. Use solution
println!("Colors used: {}", solution.coloring.chromatic_number);
println!("Phase coherence: {:.4}", solution.phase_coherence);
println!("Kuramoto order: {:.4}", solution.kuramoto_order);
```

## Benefits of This Architecture

1. **Testability**
   - Domain logic easily tested with mock adapters
   - No need to instantiate real engines for unit tests

2. **Flexibility**
   - Swap adapters without changing domain logic
   - E.g., replace neuromorphic-engine with different implementation

3. **Maintainability**
   - Clear separation of concerns
   - Changes to infrastructure don't affect business logic

4. **No Circular Dependencies**
   - shared-types prevents circular refs
   - Clean dependency graph

5. **Type Safety**
   - Port traits enforce contracts
   - Compile-time verification of adapter compliance

## Testing

Run integration tests:
```bash
# Full PRCT pipeline test (comprehensive but slow)
cargo run --example test_prct_architecture

# Simple adapter verification (quick)
cargo run --example test_adapters_simple
```

Run unit tests:
```bash
# Test domain logic with mocks
cargo test -p prct-core

# Test adapters individually
cargo test -p prct-adapters
```

## Next Steps

### Phase 1 (Current - Completion)
- [x] Create shared-types crate (zero deps)
- [x] Create prct-core with ports
- [x] Create prct-adapters implementing ports
- [x] Fix adapter API mismatches
- [x] Create integration tests
- [ ] Optimize performance for small graphs
- [ ] Add comprehensive unit tests
- [ ] Run DIMACS benchmarks through clean architecture

### Phase 2 (Future - C-Logic Integration)
- [ ] Port DRPP (Dynamic Resonance Pattern Processor) from ARES-51
- [ ] Port ADP (Adaptive Decision Processor) from ARES-51
- [ ] Port EGC (Emergent Governance Controller)
- [ ] Port EMS (Emotional Modeling System)
- [ ] Port CSF-Bus (Phase Coherence Bus messaging)
- [ ] Port ChronoPath/CSF-Time (temporal processing, HLC)
- [ ] Port CSF-Kernel (real-time scheduler)
- [ ] Integrate C-Logic with PRCT pipeline

## References

- **Hexagonal Architecture:** Alistair Cockburn (2005)
- **Ports & Adapters:** Alternative name for hexagonal architecture
- **Clean Architecture:** Robert C. Martin (2017)
- **PRCT Algorithm:** Phase Resonance Chromatic-TSP (this project)
