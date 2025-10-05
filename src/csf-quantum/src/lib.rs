//! CSF-Quantum: High-level quantum computing simulation library powered by csf-mlir.

// Public modules that form the usable API
pub mod algorithms;
pub mod circuits;
pub mod error_correction;
pub mod gates;
pub mod optimization;
pub mod simulation;
pub mod state;

// Re-export key types for easier use
pub use state::QuantumState;
pub use simulation::QuantumSimulation;