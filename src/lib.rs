//! Active Inference Platform
//!
//! Constitution: Phase 1 - Mathematical Foundation & Proof System
//! Pure software implementation of neuromorphic and quantum domain analogues
//! with information-theoretic coupling and active inference.

pub mod mathematics;
pub mod information_theory;

// Re-export key components
pub use mathematics::{
    MathematicalStatement, ProofResult, Assumption,
};

pub use information_theory::{
    TransferEntropy, TransferEntropyResult, CausalDirection,
    detect_causal_direction,
};

/// Platform version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Active Inference Platform";
pub const DESCRIPTION: &str = "Scientifically rigorous active inference platform with GPU acceleration";

