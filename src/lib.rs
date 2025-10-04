//! Active Inference Platform
//!
//! Constitution: Phase 1-4 - Mathematical Foundation, Active Inference, Integration & Production Hardening
//! Pure software implementation of neuromorphic and quantum domain analogues
//! with information-theoretic coupling, active inference, and enterprise-grade resilience.

pub mod mathematics;
pub mod information_theory;
pub mod statistical_mechanics;
pub mod active_inference;
pub mod integration;
pub mod resilience;
pub mod optimization;
pub mod cma; // Phase 6: Causal Manifold Annealing

// Re-export key components
pub use mathematics::{
    MathematicalStatement, ProofResult, Assumption,
};

pub use information_theory::{
    TransferEntropy, TransferEntropyResult, CausalDirection,
    detect_causal_direction,
};

pub use statistical_mechanics::{
    ThermodynamicNetwork, ThermodynamicState, NetworkConfig,
    ThermodynamicMetrics, EvolutionResult,
};

pub use active_inference::{
    GenerativeModel, HierarchicalModel, StateSpaceLevel,
    ObservationModel, TransitionModel, VariationalInference,
    PolicySelector, ActiveInferenceController,
};

pub use integration::{
    CrossDomainBridge, DomainState, CouplingStrength,
    InformationChannel, PhaseSynchronizer,
};

pub use resilience::{
    HealthMonitor, ComponentHealth, HealthStatus, SystemState,
    CircuitBreaker, CircuitState, CircuitBreakerConfig, CircuitBreakerError,
    CheckpointManager, Checkpointable, CheckpointMetadata, StorageBackend, CheckpointError,
};

pub use optimization::{
    PerformanceTuner, TuningProfile, SearchAlgorithm, SearchSpace, PerformanceMetrics,
    KernelTuner, GpuProperties, KernelConfig, OccupancyInfo,
};

/// Platform version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Active Inference Platform";
pub const DESCRIPTION: &str = "Scientifically rigorous active inference platform with GPU acceleration";

