//! CUDA Kernel FFI Bindings for Quantum MLIR
//!
//! Direct bindings to the GPU kernels with native complex number support

use std::ffi::c_void;
use std::os::raw::{c_int, c_double};
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

/// CUDA complex number type matching cuDoubleComplex
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaComplex {
    pub real: f64,
    pub imag: f64,
}

impl CudaComplex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self { real: 0.0, imag: 0.0 }
    }

    pub fn one() -> Self {
        Self { real: 1.0, imag: 0.0 }
    }
}

// Implement required traits for CudaComplex to work with cudarc
unsafe impl DeviceRepr for CudaComplex {}
unsafe impl ValidAsZeroBits for CudaComplex {}

// FFI bindings to CUDA kernels
extern "C" {
    // Initialize quantum state to |00...0>
    pub fn quantum_init_state(
        state: *mut CudaComplex,
        dimension: c_int,
    ) -> c_int;

    // Apply Hadamard gate
    pub fn quantum_hadamard(
        state: *mut CudaComplex,
        qubit: c_int,
        num_qubits: c_int,
    ) -> c_int;

    // Apply CNOT gate
    pub fn quantum_cnot(
        state: *mut CudaComplex,
        control: c_int,
        target: c_int,
        num_qubits: c_int,
    ) -> c_int;

    // Time evolution
    pub fn quantum_evolve(
        state: *mut CudaComplex,
        hamiltonian: *const CudaComplex,
        time: c_double,
        dimension: c_int,
        trotter_steps: c_int,
    ) -> c_int;

    // Quantum Fourier Transform
    pub fn quantum_qft(
        state: *mut CudaComplex,
        num_qubits: c_int,
        inverse: bool,
    ) -> c_int;

    // VQE ansatz
    pub fn quantum_vqe_ansatz(
        state: *mut CudaComplex,
        parameters: *const c_double,
        num_qubits: c_int,
        num_layers: c_int,
    ) -> c_int;

    // Measure quantum state
    pub fn quantum_measure(
        state: *const CudaComplex,
        probabilities: *mut c_double,
        dimension: c_int,
    ) -> c_int;
}

/// Safe Rust wrapper for quantum GPU operations
pub struct QuantumGpuKernels;

impl QuantumGpuKernels {
    /// Initialize quantum state on GPU
    pub fn init_state(state_ptr: *mut CudaComplex, dimension: usize) -> anyhow::Result<()> {
        unsafe {
            let result = quantum_init_state(state_ptr, dimension as c_int);
            if result == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!("CUDA error: {}", result))
            }
        }
    }

    /// Apply Hadamard gate on GPU
    pub fn hadamard(
        state_ptr: *mut CudaComplex,
        qubit: usize,
        num_qubits: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            let result = quantum_hadamard(state_ptr, qubit as c_int, num_qubits as c_int);
            if result == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!("CUDA error: {}", result))
            }
        }
    }

    /// Apply CNOT gate on GPU
    pub fn cnot(
        state_ptr: *mut CudaComplex,
        control: usize,
        target: usize,
        num_qubits: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            let result = quantum_cnot(
                state_ptr,
                control as c_int,
                target as c_int,
                num_qubits as c_int,
            );
            if result == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!("CUDA error: {}", result))
            }
        }
    }

    /// Evolve quantum state on GPU
    pub fn evolve(
        state_ptr: *mut CudaComplex,
        hamiltonian_ptr: *const CudaComplex,
        time: f64,
        dimension: usize,
        trotter_steps: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            let result = quantum_evolve(
                state_ptr,
                hamiltonian_ptr,
                time,
                dimension as c_int,
                trotter_steps as c_int,
            );
            if result == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!("CUDA error: {}", result))
            }
        }
    }

    /// Apply Quantum Fourier Transform on GPU
    pub fn qft(
        state_ptr: *mut CudaComplex,
        num_qubits: usize,
        inverse: bool,
    ) -> anyhow::Result<()> {
        unsafe {
            let result = quantum_qft(state_ptr, num_qubits as c_int, inverse);
            if result == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!("CUDA error: {}", result))
            }
        }
    }

    /// Apply VQE ansatz on GPU
    pub fn vqe_ansatz(
        state_ptr: *mut CudaComplex,
        parameters: &[f64],
        num_qubits: usize,
        num_layers: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            let result = quantum_vqe_ansatz(
                state_ptr,
                parameters.as_ptr(),
                num_qubits as c_int,
                num_layers as c_int,
            );
            if result == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!("CUDA error: {}", result))
            }
        }
    }

    /// Measure quantum state on GPU
    pub fn measure(
        state_ptr: *const CudaComplex,
        probabilities: &mut [f64],
        dimension: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            let result = quantum_measure(
                state_ptr,
                probabilities.as_mut_ptr(),
                dimension as c_int,
            );
            if result == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!("CUDA error: {}", result))
            }
        }
    }
}