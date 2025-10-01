//! Quantum-Inspired Computing Engine
//!
//! World's first software-based quantum Hamiltonian operator for optimization
//! Implements complete PRCT (Phase Resonance Chromatic-TSP) Algorithm

pub mod types;
pub mod hamiltonian;
pub mod security;
pub mod robust_eigen;
pub mod prct_coloring;
pub mod prct_tsp;

// Re-export main types
pub use types::*;
pub use hamiltonian::{Hamiltonian, calculate_ground_state, PhaseResonanceField, PRCTDiagnostics};
pub use security::{SecurityValidator, SecurityError};
pub use robust_eigen::{RobustEigenSolver, RobustEigenConfig, EigenDiagnostics, SolverMethod};
pub use prct_coloring::ChromaticColoring;
pub use prct_tsp::TSPPathOptimizer;