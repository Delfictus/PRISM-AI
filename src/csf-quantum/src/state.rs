//! Quantum state management

use csf_mlir::runtime::MlirJit;
use std::sync::Arc;
use anyhow::{Result, anyhow};

/// Represents a quantum state managed by the MLIR runtime.
///
/// This struct is a handle to a quantum state that lives on the GPU.
/// All operations on this state are translated into MLIR and JIT-compiled.
#[derive(Clone)]
pub struct QuantumState {
    /// A unique identifier for the state buffer on the device.
    pub(crate) id: String,
    /// The number of qubits represented by this state.
    pub(crate) num_qubits: usize,
    /// The dimension of the state vector (2^num_qubits).
    pub(crate) dimension: usize,
    /// A shared reference to the MLIR JIT compilation and execution engine.
    pub(crate) jit: Arc<MlirJit>,
}

impl QuantumState {
    /// Creates a new quantum state initialized to the |0...0> state.
    ///
    /// This is the primary entry point for creating a new quantum simulation.
    ///
    /// # Arguments
    /// * `jit` - A shared reference to the MLIR JIT engine.
    /// * `num_qubits` - The number of qubits in the system.
    ///
    /// # Returns
    /// A new `QuantumState` handle.
    pub fn new(jit: Arc<MlirJit>, num_qubits: usize) -> Result<Self> {
        let dimension = 1 << num_qubits;
        let id = format!("state_{}", uuid::Uuid::new_v4());

        // MLIR generation to allocate and initialize the state on the GPU
        let mlir_module = format!(
            r#"
            module {{
                func.func @create_state() -> () {{
                    %state = "quantum.create_state"() {{
                        num_qubits = {} : i32,
                        id = "{}"
                    }} : () -> tensor<{}xcomplex<f64>>
                    return
                }}
            }}
            "#,
            num_qubits, id, dimension
        );

        jit.execute(&mlir_module)?;

        Ok(Self {
            id,
            num_qubits,
            dimension,
            jit,
        })
    }

    /// Returns the number of qubits in the state.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the dimension of the state vector.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Creates a quantum state from existing amplitudes on the GPU
    pub fn from_amplitudes(
        jit: Arc<MlirJit>,
        amplitudes: Vec<(f64, f64)>,
    ) -> Result<Self> {
        let dimension = amplitudes.len();
        let num_qubits = (dimension as f64).log2() as usize;

        if dimension != (1 << num_qubits) {
            return Err(anyhow!("Invalid state dimension: must be power of 2"));
        }

        let id = format!("state_{}", uuid::Uuid::new_v4());

        // In real implementation, would upload amplitudes to GPU
        // For now, just create the state handle

        Ok(Self {
            id,
            num_qubits,
            dimension,
            jit,
        })
    }

    /// Get the amplitudes of the quantum state
    pub fn get_amplitudes(&self) -> Result<Vec<(f64, f64)>> {
        // In real implementation, would download from GPU
        Ok(vec![(1.0, 0.0); self.dimension])
    }
}