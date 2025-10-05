//! Quantum Engine Adapter - GPU Accelerated via CSF-Quantum
//!
//! Wraps GPU-accelerated quantum Hamiltonian evolution using the
//! csf-quantum library which handles all MLIR compilation and GPU execution.
//!
//! Constitutional Compliance:
//! - Article II, Section B: No direct FFI, uses csf-quantum API
//! - Article I, Principle 3: Compiler-centric abstraction via MLIR

use prct_core::ports::QuantumPort;
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use csf_quantum::{QuantumState as CsfQuantumState, QuantumSimulation};
use csf_quantum::algorithms::Hamiltonian as CsfHamiltonian;
use csf_mlir::runtime::MlirJit;
use std::sync::Arc;
use parking_lot::Mutex;

/// Adapter connecting PRCT domain to GPU-accelerated quantum engine via CSF-Quantum
pub struct QuantumAdapter {
    /// MLIR JIT runtime for compilation and execution
    jit: Arc<MlirJit>,
    /// Quantum simulation environment
    simulation: Arc<QuantumSimulation>,
    /// Cached Hamiltonian for reuse
    cached_hamiltonian: Arc<Mutex<Option<CsfHamiltonian>>>,
    /// Whether GPU is available and initialized
    gpu_available: bool,
}

impl QuantumAdapter {
    /// Create new GPU-accelerated quantum adapter using CSF-Quantum
    pub fn new() -> Self {
        // Initialize MLIR JIT runtime
        let jit = match MlirJit::new() {
            Ok(jit) => {
                println!("✓ CSF-MLIR JIT initialized successfully");
                jit
            }
            Err(e) => {
                eprintln!("⚠ CSF-MLIR initialization failed: {}. Performance will be limited.", e);
                // Create a fallback JIT instance
                Arc::new(MlirJit::new().expect("Failed to create fallback JIT"))
            }
        };

        // Check GPU availability through the JIT runtime
        let gpu_available = jit.cuda_context().is_some();
        if gpu_available {
            println!("✓ GPU detected and ready for quantum acceleration");
        } else {
            println!("⚠ No GPU detected, using CPU fallback");
        }

        // Create quantum simulation environment
        let simulation = Arc::new(QuantumSimulation::new(jit.clone()));

        Self {
            jit,
            simulation,
            cached_hamiltonian: Arc::new(Mutex::new(None)),
            gpu_available,
        }
    }
}

impl QuantumPort for QuantumAdapter {
    /// Build Hamiltonian using CSF-Quantum GPU primitives
    /// Constitutional: Article II, Section B - Uses csf-quantum API
    fn build_hamiltonian(&self, graph: &Graph, _params: &EvolutionParams) -> Result<HamiltonianState> {
        println!("[CSF-Quantum] Building Hamiltonian from graph with {} vertices", graph.num_vertices);

        // Use CSF-Quantum to build Hamiltonian directly on GPU
        let hamiltonian = CsfHamiltonian::from_graph(&self.jit, graph)
            .map_err(|e| PRCTError::QuantumFailed(format!("Hamiltonian construction failed: {}", e)))?;

        let dimension = hamiltonian.dimension();

        // Cache the Hamiltonian for reuse
        *self.cached_hamiltonian.lock() = Some(hamiltonian);

        // For now, return placeholder matrix elements
        // In full implementation, would retrieve from GPU
        let matrix_elements: Vec<(f64, f64)> = vec![(0.0, 0.0); dimension * dimension];

        Ok(HamiltonianState {
            matrix_elements,
            eigenvalues: vec![0.0; dimension],
            ground_state_energy: -1.0,
            dimension,
        })
    }

    /// Evolve quantum state using CSF-Quantum high-level API
    /// Constitutional: Article I, Principle 3 - Compiler-centric abstraction
    fn evolve_state(
        &self,
        hamiltonian_state: &HamiltonianState,
        initial_state: &QuantumState,
        evolution_time: f64,
    ) -> Result<QuantumState> {
        println!("[CSF-Quantum] Evolving state for {}s", evolution_time);

        // Get cached Hamiltonian or error
        let hamiltonian_guard = self.cached_hamiltonian.lock();
        let hamiltonian = hamiltonian_guard.as_ref()
            .ok_or_else(|| PRCTError::QuantumFailed("Hamiltonian not initialized".into()))?;

        // Convert shared-types QuantumState to CSF-Quantum state
        let csf_state = CsfQuantumState::from_amplitudes(
            self.jit.clone(),
            initial_state.amplitudes.clone(),
        ).map_err(|e| PRCTError::QuantumFailed(format!("State creation failed: {}", e)))?;

        // Use CSF-Quantum's evolve function - all GPU operations handled internally
        let evolved_state = self.simulation.evolve(&csf_state, hamiltonian, evolution_time)
            .map_err(|e| PRCTError::QuantumFailed(format!("Evolution failed: {}", e)))?;

        // Retrieve amplitudes from GPU
        let amplitudes = evolved_state.get_amplitudes()
            .map_err(|e| PRCTError::QuantumFailed(format!("Failed to retrieve amplitudes: {}", e)))?;

        // Calculate phase coherence
        let phase_coherence = self.calculate_phase_coherence(&amplitudes);

        // Calculate energy
        let energy = self.calculate_energy(&amplitudes, hamiltonian_state.dimension);

        Ok(QuantumState {
            amplitudes,
            phase_coherence,
            energy,
            entanglement: 0.0, // Placeholder
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }

    /// Get phase field from quantum state
    fn get_phase_field(&self, state: &QuantumState) -> Result<PhaseField> {
        use num_complex::Complex64;

        // Extract phases from quantum state amplitudes
        let phases: Vec<f64> = state.amplitudes.iter()
            .map(|&(re, im)| Complex64::new(re, im).arg())
            .collect();

        let n = phases.len();

        // Compute phase coherence matrix
        let mut coherence_matrix = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let phase_diff = phases[i] - phases[j];
                let coherence = phase_diff.cos().powi(2);
                coherence_matrix[i * n + j] = coherence;
            }
        }

        // Compute order parameter
        let sum_real: f64 = phases.iter().map(|p| p.cos()).sum();
        let sum_imag: f64 = phases.iter().map(|p| p.sin()).sum();
        let order_parameter = ((sum_real / n as f64).powi(2) + (sum_imag / n as f64).powi(2)).sqrt();

        Ok(PhaseField {
            phases,
            coherence_matrix,
            order_parameter,
            resonance_frequency: 50.0, // Default frequency
        })
    }

    /// Compute ground state using CSF-Quantum VQE algorithm
    fn compute_ground_state(&self, hamiltonian_state: &HamiltonianState) -> Result<QuantumState> {
        println!("[CSF-Quantum] Computing ground state using VQE");

        let hamiltonian_guard = self.cached_hamiltonian.lock();
        let hamiltonian = hamiltonian_guard.as_ref()
            .ok_or_else(|| PRCTError::QuantumFailed("Hamiltonian not initialized".into()))?;

        // Create initial state for VQE
        let num_qubits = (hamiltonian_state.dimension as f64).log2() as usize;
        let initial_state = CsfQuantumState::new(self.jit.clone(), num_qubits)
            .map_err(|e| PRCTError::QuantumFailed(format!("Initial state creation failed: {}", e)))?;

        // Use VQE to find ground state
        let vqe = csf_quantum::algorithms::algorithms::VQE::new(self.jit.clone());
        let (ground_state, ground_energy) = vqe.find_ground_state(hamiltonian, &initial_state)
            .map_err(|e| PRCTError::QuantumFailed(format!("VQE failed: {}", e)))?;

        // Get amplitudes from ground state
        let amplitudes = ground_state.get_amplitudes()
            .map_err(|e| PRCTError::QuantumFailed(format!("Failed to retrieve ground state amplitudes: {}", e)))?;

        let phase_coherence = self.calculate_phase_coherence(&amplitudes);

        Ok(QuantumState {
            amplitudes,
            phase_coherence,
            energy: ground_energy,
            entanglement: 0.0,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }
}

// Helper methods
impl QuantumAdapter {
    /// Calculate phase coherence from amplitudes
    fn calculate_phase_coherence(&self, amplitudes: &[(f64, f64)]) -> f64 {
        use num_complex::Complex64;

        let n = amplitudes.len() as f64;
        let avg_phase: f64 = amplitudes.iter()
            .map(|&(re, im)| Complex64::new(re, im).arg())
            .sum::<f64>() / n;

        let phase_variance: f64 = amplitudes.iter()
            .map(|&(re, im)| {
                let phase = Complex64::new(re, im).arg();
                (phase - avg_phase).powi(2)
            })
            .sum::<f64>() / n;

        // Coherence is inversely related to phase variance
        (-phase_variance).exp()
    }

    /// Calculate energy expectation value
    fn calculate_energy(&self, amplitudes: &[(f64, f64)], dimension: usize) -> f64 {
        // Placeholder energy calculation
        // In full implementation, would use Hamiltonian matrix
        let norm: f64 = amplitudes.iter()
            .map(|&(re, im)| re * re + im * im)
            .sum();

        -norm.sqrt() * (dimension as f64).ln()
    }
}

impl Default for QuantumAdapter {
    fn default() -> Self {
        Self::new()
    }
}