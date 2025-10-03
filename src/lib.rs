//! Active Inference Platform
//!
//! Constitution: Phase 1-2 - Mathematical Foundation & Active Inference
//! Pure software implementation of neuromorphic and quantum domain analogues
//! with information-theoretic coupling and active inference.

pub mod mathematics;
pub mod information_theory;
pub mod statistical_mechanics;
pub mod active_inference;

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

/// Platform version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Active Inference Platform";
pub const DESCRIPTION: &str = "Scientifically rigorous active inference platform with GPU acceleration";

