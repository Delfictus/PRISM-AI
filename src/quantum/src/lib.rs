//! Quantum-Inspired Computing Engine
//!
//! World's first software-based quantum Hamiltonian operator for optimization
//! Implements complete PRCT (Phase Resonance Chromatic-TSP) Algorithm

pub mod types;
pub mod hamiltonian;
pub mod security;
pub mod robust_eigen;

// Re-export main types
pub use types::*;
pub use hamiltonian::{Hamiltonian, calculate_ground_state};
pub use security::{SecurityValidator, SecurityError};
pub use robust_eigen::{RobustEigenSolver, RobustEigenConfig, EigenDiagnostics, SolverMethod};