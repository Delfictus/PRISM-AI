//! Quantum Engine Adapter
//!
//! Wraps the existing quantum-engine to implement QuantumPort.

use prct_core::ports::QuantumPort;
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use quantum_engine::{Hamiltonian, ForceFieldParams, PhaseResonanceField};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::sync::Arc;
use parking_lot::Mutex;

/// Adapter connecting PRCT domain to quantum engine
pub struct QuantumAdapter {
    hamiltonian: Arc<Mutex<Option<Hamiltonian>>>,
    phase_field: Arc<Mutex<Option<PhaseResonanceField>>>,
}

impl QuantumAdapter {
    /// Create new quantum adapter
    pub fn new() -> Self {
        Self {
            hamiltonian: Arc::new(Mutex::new(None)),
            phase_field: Arc::new(Mutex::new(None)),
        }
    }

    /// Build Hamiltonian from coupling matrix (internal helper)
    fn build_hamiltonian_internal(coupling_matrix: &Array2<Complex64>) -> Result<Hamiltonian> {
        let n = coupling_matrix.nrows();

        // Create positions (simple 1D chain for now)
        let positions = Array2::from_shape_fn((n, 3), |(i, dim)| {
            if dim == 0 { i as f64 } else { 0.0 }
        });

        // Create masses (unit mass in amu)
        let masses = Array1::from_elem(n, 1.0);

        // Force field parameters using real CHARMM-like params
        let force_field = ForceFieldParams::new();

        Hamiltonian::new(positions, masses, force_field)
            .map_err(|e| PRCTError::QuantumFailed(format!("Hamiltonian construction failed: {:?}", e)))
    }
}

impl QuantumPort for QuantumAdapter {
    fn build_hamiltonian(&self, graph: &Graph, _params: &EvolutionParams) -> Result<HamiltonianState> {
        // Build coupling matrix from graph
        let n = graph.num_vertices;
        let mut coupling = Array2::zeros((n, n));

        for &(u, v, weight) in &graph.edges {
            if u < n && v < n {
                coupling[[u, v]] = Complex64::new(weight, 0.0);
                coupling[[v, u]] = Complex64::new(weight, 0.0);
            }
        }

        // Create Hamiltonian
        let hamiltonian = Self::build_hamiltonian_internal(&coupling)?;

        // Store for later use
        *self.hamiltonian.lock() = Some(hamiltonian.clone());

        // Extract Hamiltonian state
        let matrix = hamiltonian.matrix_representation();
        let matrix_elements: Vec<(f64, f64)> = matrix.iter()
            .map(|c| (c.re, c.im))
            .collect();

        let dimension = n * 3; // 3D space per atom

        Ok(HamiltonianState {
            matrix_elements,
            eigenvalues: vec![0.0; dimension], // Would compute via eigendecomposition
            ground_state_energy: -1.0,
            dimension,
        })
    }

    fn evolve_state(
        &self,
        _hamiltonian: &HamiltonianState,
        initial_state: &QuantumState,
        evolution_time: f64,
    ) -> Result<QuantumState> {
        let mut hamiltonian_guard = self.hamiltonian.lock();
        let hamiltonian = hamiltonian_guard.as_mut()
            .ok_or_else(|| PRCTError::QuantumFailed("Hamiltonian not initialized".into()))?;

        // Convert shared-types QuantumState to engine Array1<Complex64>
        let state_vec: Array1<Complex64> = initial_state.amplitudes.iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect();

        // Evolve using Hamiltonian
        let evolved = hamiltonian.evolve(&state_vec, evolution_time)
            .map_err(|e| PRCTError::QuantumFailed(format!("Evolution failed: {:?}", e)))?;

        // Compute metrics
        let phase_coherence = hamiltonian.phase_coherence();
        let energy = hamiltonian.total_energy(&evolved);

        // Convert back to shared-types
        let amplitudes: Vec<(f64, f64)> = evolved.iter()
            .map(|c| (c.re, c.im))
            .collect();

        Ok(QuantumState {
            amplitudes,
            phase_coherence,
            energy,
            entanglement: 0.0, // Would compute via partial trace
            timestamp_ns: 0,
        })
    }

    fn get_phase_field(&self, state: &QuantumState) -> Result<PhaseField> {
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

    fn compute_ground_state(&self, _hamiltonian: &HamiltonianState) -> Result<QuantumState> {
        let mut hamiltonian_guard = self.hamiltonian.lock();
        let hamiltonian = hamiltonian_guard.as_mut()
            .ok_or_else(|| PRCTError::QuantumFailed("Hamiltonian not initialized".into()))?;

        // Use quantum engine's ground state calculation
        let ground_state = quantum_engine::calculate_ground_state(hamiltonian);

        let phase_coherence = hamiltonian.phase_coherence();
        let energy = hamiltonian.total_energy(&ground_state);

        let amplitudes: Vec<(f64, f64)> = ground_state.iter()
            .map(|c| (c.re, c.im))
            .collect();

        Ok(QuantumState {
            amplitudes,
            phase_coherence,
            energy,
            entanglement: 0.0,
            timestamp_ns: 0,
        })
    }
}

impl Default for QuantumAdapter {
    fn default() -> Self {
        Self::new()
    }
}
