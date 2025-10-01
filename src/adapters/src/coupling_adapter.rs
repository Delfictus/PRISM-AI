//! Physics Coupling Adapter
//!
//! Wraps PhysicsCoupling from platform-foundation (commit 35963c6) to implement PhysicsCouplingPort.

use prct_core::ports::PhysicsCouplingPort;
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use platform_foundation::{PhysicsCoupling, KuramotoSync};
use num_complex::Complex64;
use nalgebra::DMatrix;

/// Adapter connecting PRCT domain to physics coupling service
pub struct CouplingAdapter {
    // Stateless adapter - PhysicsCoupling is constructed on demand
}

impl CouplingAdapter {
    /// Create new coupling adapter
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicsCouplingPort for CouplingAdapter {
    fn compute_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<CouplingStrength> {
        // Build coupling matrix from quantum state
        let n = quantum_state.amplitudes.len();
        let mut coupling_matrix = DMatrix::from_element(n, n, Complex64::new(0.0, 0.0));

        for i in 0..n {
            for j in 0..n {
                let phase_i = Complex64::new(quantum_state.amplitudes[i].0, quantum_state.amplitudes[i].1).arg();
                let phase_j = Complex64::new(quantum_state.amplitudes[j].0, quantum_state.amplitudes[j].1).arg();
                let phase_diff = (phase_i - phase_j).cos();
                coupling_matrix[(i, j)] = Complex64::new(phase_diff, 0.0);
            }
        }

        // Use PhysicsCoupling from commit 35963c6
        let quantum_vec: Vec<Complex64> = quantum_state.amplitudes.iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect();

        let physics = PhysicsCoupling::from_system_state(
            &neuro_state.neuron_states,
            &neuro_state.spike_pattern.iter().map(|&s| s as f64).collect::<Vec<_>>(),
            &quantum_vec,
            &coupling_matrix,
        ).map_err(|e| PRCTError::CouplingFailed(e.to_string()))?;

        Ok(CouplingStrength {
            neuro_to_quantum: physics.neuro_to_quantum.pattern_to_hamiltonian,
            quantum_to_neuro: physics.quantum_to_neuro.energy_to_learning_rate,
            bidirectional_coherence: physics.phase_sync.order_parameter,
            timestamp_ns: 0,
        })
    }

    fn update_kuramoto_sync(
        &self,
        neuro_phases: &[f64],
        quantum_phases: &[f64],
        dt: f64,
    ) -> Result<KuramotoState> {
        // Build coupling matrix between neuromorphic and quantum oscillators
        let n = neuro_phases.len().max(quantum_phases.len());
        let mut coupling_matrix = DMatrix::from_element(n, n, Complex64::new(0.5, 0.0));

        // Simple Kuramoto update (can be enhanced with full PhysicsCoupling later)
        let mut phases = neuro_phases.to_vec();
        phases.extend_from_slice(quantum_phases);

        let natural_frequencies = vec![1.0; n];
        let coupling_strength = 0.5;

        // Kuramoto step
        let mut new_phases = phases.clone();
        for i in 0..n {
            let mut coupling_sum = 0.0;
            for j in 0..n {
                if i != j {
                    coupling_sum += (phases[j] - phases[i]).sin();
                }
            }
            let dphase = natural_frequencies[i] + (coupling_strength / n as f64) * coupling_sum;
            new_phases[i] = (phases[i] + dphase * dt) % (2.0 * core::f64::consts::PI);
        }

        // Compute order parameter
        let sum_real: f64 = new_phases.iter().map(|p| p.cos()).sum();
        let sum_imag: f64 = new_phases.iter().map(|p| p.sin()).sum();
        let order_parameter = ((sum_real / n as f64).powi(2) + (sum_imag / n as f64).powi(2)).sqrt();

        Ok(KuramotoState {
            phases: new_phases,
            natural_frequencies,
            coupling_matrix: vec![0.5; n * n],
            order_parameter,
            mean_phase: new_phases.iter().sum::<f64>() / n as f64,
        })
    }

    fn calculate_transfer_entropy(
        &self,
        source: &[f64],
        target: &[f64],
        _lag: f64,
    ) -> Result<TransferEntropy> {
        if source.len() != target.len() {
            return Err(PRCTError::CouplingFailed("Source and target length mismatch".into()));
        }

        // Simplified transfer entropy calculation
        let n = source.len();
        let mut te = 0.0;

        for i in 1..n {
            let dy = target[i] - target[i - 1];
            let x_prev = source[i - 1];
            te += (dy * x_prev).abs();
        }

        te /= (n - 1) as f64;

        Ok(TransferEntropy {
            entropy_bits: te,
            confidence: 0.9,
            lag_ms: 10.0,
        })
    }

    fn get_bidirectional_coupling(
        &self,
        neuro_state: &NeuroState,
        quantum_state: &QuantumState,
    ) -> Result<BidirectionalCoupling> {
        // Compute coupling strength
        let coupling_strength = self.compute_coupling(neuro_state, quantum_state)?;

        // Compute transfer entropy in both directions
        let neuro_to_quantum_te = self.calculate_transfer_entropy(
            &neuro_state.neuron_states,
            &quantum_state.amplitudes.iter().map(|&(re, _)| re).collect::<Vec<_>>(),
            10.0,
        )?;

        let quantum_to_neuro_te = self.calculate_transfer_entropy(
            &quantum_state.amplitudes.iter().map(|&(re, _)| re).collect::<Vec<_>>(),
            &neuro_state.neuron_states,
            10.0,
        )?;

        // Extract phases for Kuramoto
        let neuro_phases: Vec<f64> = neuro_state.neuron_states.iter()
            .map(|&x| (x * core::f64::consts::TAU) % core::f64::consts::TAU)
            .collect();

        let quantum_phases: Vec<f64> = quantum_state.amplitudes.iter()
            .map(|&(re, im)| Complex64::new(re, im).arg())
            .collect();

        let kuramoto_state = self.update_kuramoto_sync(&neuro_phases, &quantum_phases, 0.01)?;

        Ok(BidirectionalCoupling {
            neuro_to_quantum_entropy: neuro_to_quantum_te,
            quantum_to_neuro_entropy: quantum_to_neuro_te,
            kuramoto_state,
            coupling_quality: coupling_strength.bidirectional_coherence,
        })
    }
}

impl Default for CouplingAdapter {
    fn default() -> Self {
        Self::new()
    }
}
