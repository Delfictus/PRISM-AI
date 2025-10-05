//! Quantum circuit definitions and builders

use crate::state::QuantumState;
use anyhow::Result;

/// Quantum circuit builder
pub struct CircuitBuilder {
    operations: Vec<Operation>,
}

#[derive(Clone, Debug)]
enum Operation {
    Hadamard(usize),
    CNOT(usize, usize),
    RZ(usize, f64),
    RX(usize, f64),
    RY(usize, f64),
}

impl CircuitBuilder {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn h(&mut self, qubit: usize) -> &mut Self {
        self.operations.push(Operation::Hadamard(qubit));
        self
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.operations.push(Operation::CNOT(control, target));
        self
    }

    pub fn rz(&mut self, qubit: usize, angle: f64) -> &mut Self {
        self.operations.push(Operation::RZ(qubit, angle));
        self
    }

    pub fn build(&self) -> Circuit {
        Circuit {
            operations: self.operations.clone(),
        }
    }
}

/// Compiled quantum circuit
pub struct Circuit {
    operations: Vec<Operation>,
}

impl Circuit {
    pub fn apply(&self, state: &QuantumState) -> Result<QuantumState> {
        // In real implementation, would generate MLIR and execute on GPU
        Ok(state.clone())
    }
}