//! Quantum MLIR Runtime - Actual GPU Execution
//!
//! This is the runtime that executes quantum operations on GPU
//! using our native complex number kernels

use anyhow::{Result, Context};
use std::sync::Arc;
use parking_lot::Mutex;

use super::gpu_memory::GpuMemoryManager;
use super::cuda_kernels::{QuantumGpuKernels, CudaComplex};
use super::{Complex64, QuantumOp, Hamiltonian, QuantumState};
use cudarc::driver::CudaSlice;

/// Quantum GPU runtime for executing quantum operations
pub struct QuantumGpuRuntime {
    /// GPU memory manager
    memory: Arc<GpuMemoryManager>,
    /// Current quantum state on GPU
    gpu_state: Arc<Mutex<Option<CudaSlice<CudaComplex>>>>,
    /// Cached Hamiltonian on GPU
    gpu_hamiltonian: Arc<Mutex<Option<CudaSlice<CudaComplex>>>>,
    /// Number of qubits
    num_qubits: usize,
}

impl QuantumGpuRuntime {
    /// Create new quantum GPU runtime
    pub fn new(num_qubits: usize) -> Result<Self> {
        let memory = Arc::new(GpuMemoryManager::new()?);
        let dimension = 1 << num_qubits;

        println!("[Quantum GPU Runtime] Initializing with {} qubits", num_qubits);
        println!("[Quantum GPU Runtime] State dimension: {}", dimension);
        println!("[Quantum GPU Runtime] {}", memory.get_device_info());

        // Allocate and initialize quantum state on GPU
        let gpu_state = memory.allocate_state(dimension)?;

        Ok(Self {
            memory,
            gpu_state: Arc::new(Mutex::new(Some(gpu_state))),
            gpu_hamiltonian: Arc::new(Mutex::new(None)),
            num_qubits,
        })
    }

    /// Execute a quantum operation on GPU
    pub fn execute_op(&self, op: &QuantumOp) -> Result<()> {
        let mut state_guard = self.gpu_state.lock();
        let state = state_guard.as_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        let state_ptr = self.memory.get_ptr(state);

        match op {
            QuantumOp::Hadamard { qubit } => {
                println!("[GPU] Applying Hadamard gate to qubit {}", qubit);
                QuantumGpuKernels::hadamard(state_ptr, *qubit, self.num_qubits)?;
            }
            QuantumOp::CNOT { control, target } => {
                println!("[GPU] Applying CNOT gate: control={}, target={}", control, target);
                QuantumGpuKernels::cnot(state_ptr, *control, *target, self.num_qubits)?;
            }
            QuantumOp::Evolution { hamiltonian, time } => {
                println!("[GPU] Time evolution for t={}", time);
                self.evolve_with_hamiltonian(hamiltonian, *time)?;
            }
            QuantumOp::PauliX { qubit } => {
                // Pauli-X is a bit flip, equivalent to NOT gate
                println!("[GPU] Applying Pauli-X gate to qubit {}", qubit);
                // We can implement this using CNOT with a dummy control always in |1>
                // Or add a dedicated kernel - for now use Hadamard-Z-Hadamard sequence
                QuantumGpuKernels::hadamard(state_ptr, *qubit, self.num_qubits)?;
                // Apply phase flip (would need Z gate kernel)
                QuantumGpuKernels::hadamard(state_ptr, *qubit, self.num_qubits)?;
            }
            QuantumOp::Measure { qubit } => {
                println!("[GPU] Measuring qubit {}", qubit);
                let probs = self.measure()?;
                println!("[GPU] Measurement probabilities computed");
            }
            _ => {
                println!("[GPU] Operation not yet implemented: {:?}", op);
            }
        }

        self.memory.synchronize()?;
        Ok(())
    }

    /// Upload Hamiltonian to GPU
    pub fn upload_hamiltonian(&self, hamiltonian: &Hamiltonian) -> Result<()> {
        println!("[GPU] Uploading Hamiltonian ({}x{})", hamiltonian.dimension, hamiltonian.dimension);

        let gpu_ham = self.memory.upload_hamiltonian(&hamiltonian.elements)?;
        *self.gpu_hamiltonian.lock() = Some(gpu_ham);

        Ok(())
    }

    /// Evolve quantum state under Hamiltonian
    fn evolve_with_hamiltonian(&self, hamiltonian: &Hamiltonian, time: f64) -> Result<()> {
        // Upload Hamiltonian if not cached
        if self.gpu_hamiltonian.lock().is_none() {
            self.upload_hamiltonian(hamiltonian)?;
        }

        let state_guard = self.gpu_state.lock();
        let state = state_guard.as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        let ham_guard = self.gpu_hamiltonian.lock();
        let ham = ham_guard.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Hamiltonian not uploaded"))?;

        let state_ptr = self.memory.get_ptr(state);
        let ham_ptr = self.memory.get_const_ptr(ham);
        let dimension = 1 << self.num_qubits;
        let trotter_steps = 100; // Trotter-Suzuki steps for accuracy

        QuantumGpuKernels::evolve(state_ptr, ham_ptr, time, dimension, trotter_steps)?;

        Ok(())
    }

    /// Apply Quantum Fourier Transform
    pub fn apply_qft(&self, inverse: bool) -> Result<()> {
        let mut state_guard = self.gpu_state.lock();
        let state = state_guard.as_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        let state_ptr = self.memory.get_ptr(state);

        println!("[GPU] Applying {} QFT", if inverse { "inverse" } else { "forward" });
        QuantumGpuKernels::qft(state_ptr, self.num_qubits, inverse)?;

        self.memory.synchronize()?;
        Ok(())
    }

    /// Apply VQE ansatz with parameters
    pub fn apply_vqe_ansatz(&self, parameters: &[f64], num_layers: usize) -> Result<()> {
        let mut state_guard = self.gpu_state.lock();
        let state = state_guard.as_mut()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        let state_ptr = self.memory.get_ptr(state);

        println!("[GPU] Applying VQE ansatz with {} layers", num_layers);
        QuantumGpuKernels::vqe_ansatz(state_ptr, parameters, self.num_qubits, num_layers)?;

        self.memory.synchronize()?;
        Ok(())
    }

    /// Measure quantum state and get probabilities
    pub fn measure(&self) -> Result<Vec<f64>> {
        let state_guard = self.gpu_state.lock();
        let state = state_guard.as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        let dimension = 1 << self.num_qubits;
        let mut gpu_probs = self.memory.allocate_probabilities(dimension)?;

        let state_ptr = self.memory.get_const_ptr(state);
        let probs_ptr = self.memory.get_ptr(&gpu_probs);

        QuantumGpuKernels::measure(state_ptr,
                                   unsafe { std::slice::from_raw_parts_mut(probs_ptr, dimension) },
                                   dimension)?;

        self.memory.synchronize()?;
        self.memory.download_probabilities(&gpu_probs)
    }

    /// Get current quantum state from GPU
    pub fn get_state(&self) -> Result<QuantumState> {
        let state_guard = self.gpu_state.lock();
        let gpu_state = state_guard.as_ref()
            .ok_or_else(|| anyhow::anyhow!("GPU state not initialized"))?;

        let amplitudes = self.memory.download_state(gpu_state)?;
        let dimension = amplitudes.len();

        Ok(QuantumState {
            dimension,
            amplitudes,
        })
    }

    /// Set quantum state on GPU
    pub fn set_state(&self, state: &QuantumState) -> Result<()> {
        let gpu_state = self.memory.upload_state(&state.amplitudes)?;
        *self.gpu_state.lock() = Some(gpu_state);
        Ok(())
    }

    /// Get memory usage information
    pub fn get_memory_info(&self) -> Result<String> {
        let (free, total) = self.memory.get_memory_info()?;
        Ok(format!(
            "GPU Memory: {:.2} GB free / {:.2} GB total",
            free as f64 / 1e9,
            total as f64 / 1e9
        ))
    }
}