//! MLIR dialect definitions for quantum operations

pub mod quantum {
    /// Quantum dialect operations
    #[derive(Debug, Clone)]
    pub enum QuantumOp {
        /// Create a quantum state
        CreateState { num_qubits: usize },
        /// Build Hamiltonian from graph
        BuildHamiltonian { dimension: usize },
        /// Evolve quantum state
        Evolve { time: f64 },
        /// Measure quantum state
        Measure,
    }
}