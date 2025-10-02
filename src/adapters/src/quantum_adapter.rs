//! Quantum Engine Adapter - GPU Accelerated
//!
//! Wraps GPU-accelerated quantum Hamiltonian evolution.
//! CPU fallback only if GPU unavailable.

use prct_core::ports::QuantumPort;
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use quantum_engine::{Hamiltonian, ForceFieldParams, PhaseResonanceField};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use parking_lot::Mutex;

/// Adapter connecting PRCT domain to GPU-accelerated quantum engine
pub struct QuantumAdapter {
    hamiltonian: Arc<Mutex<Option<Hamiltonian>>>,
    phase_field: Arc<Mutex<Option<PhaseResonanceField>>>,
    gpu_device: Option<Arc<CudaDevice>>,
    use_gpu: bool,
}

impl QuantumAdapter {
    /// Create new GPU-accelerated quantum adapter
    pub fn new() -> Self {
        // Try to initialize GPU
        let (gpu_device, use_gpu) = match CudaDevice::new(0) {
            Ok(device) => {
                println!("✓ Quantum GPU initialized (CUDA device 0)");
                (Some(Arc::new(device)), true)
            }
            Err(e) => {
                eprintln!("⚠ GPU initialization failed: {}. Using CPU fallback.", e);
                (None, false)
            }
        };

        Self {
            hamiltonian: Arc::new(Mutex::new(None)),
            phase_field: Arc::new(Mutex::new(None)),
            gpu_device,
            use_gpu,
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

    /// GPU-accelerated Hamiltonian construction
    fn build_hamiltonian_gpu(&self, coupling_matrix: &Array2<Complex64>) -> Result<Array2<Complex64>> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::QuantumFailed("GPU not initialized".into()))?;

        let n = coupling_matrix.nrows();
        let dim = n * 3; // 3D space per atom

        // Physical parameters
        let grid_spacing = 1.0;
        let hartree_to_kcalmol = 627.509;
        let kcalmol_to_hartree = 1.0 / hartree_to_kcalmol;
        let lj_epsilon = 0.1094; // kcal/mol
        let lj_sigma = 3.8164; // Angstrom
        let coulomb_cutoff = 12.0; // Angstrom

        // Create positions and masses
        let positions: Vec<f64> = (0..n).flat_map(|i| {
            vec![i as f64, 0.0, 0.0]
        }).collect();
        let masses = vec![1.0; n];

        // Allocate GPU memory
        let gpu_positions = device.htod_copy(positions.clone())
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_masses = device.htod_copy(masses.clone())
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;

        let mut gpu_kinetic = device.alloc_zeros::<(f64, f64)>(dim * dim)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_potential = device.alloc_zeros::<(f64, f64)>(dim * dim)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;

        // Launch kinetic operator kernel
        let cfg_2d = LaunchConfig {
            grid_dim: ((dim + 15) / 16) as u32,
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let func_kinetic = device.get_func("quantum_kernels", "build_kinetic_operator")
            .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

        unsafe {
            func_kinetic.launch(cfg_2d, (
                &mut gpu_kinetic,
                &gpu_masses,
                n as u32,
                grid_spacing,
                hartree_to_kcalmol,
            )).map_err(|e| PRCTError::QuantumFailed(format!("Kernel launch failed: {}", e)))?;
        }

        // Launch potential operator kernel
        let func_potential = device.get_func("quantum_kernels", "build_potential_operator")
            .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

        unsafe {
            func_potential.launch(cfg_2d, (
                &mut gpu_potential,
                &gpu_positions,
                n as u32,
                lj_epsilon,
                lj_sigma,
                coulomb_cutoff,
                kcalmol_to_hartree,
            )).map_err(|e| PRCTError::QuantumFailed(format!("Kernel launch failed: {}", e)))?;
        }

        // Copy results back and combine
        let kinetic_data: Vec<(f64, f64)> = device.dtoh_sync_copy(&gpu_kinetic)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy back failed: {}", e)))?;
        let potential_data: Vec<(f64, f64)> = device.dtoh_sync_copy(&gpu_potential)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy back failed: {}", e)))?;

        // H = T + V
        let hamiltonian_data: Vec<Complex64> = kinetic_data.iter().zip(potential_data.iter())
            .map(|((t_re, t_im), (v_re, v_im))| Complex64::new(t_re + v_re, t_im + v_im))
            .collect();

        Array2::from_shape_vec((dim, dim), hamiltonian_data)
            .map_err(|e| PRCTError::QuantumFailed(format!("Array construction failed: {}", e)))
    }

    /// GPU-accelerated state evolution using RK4
    fn evolve_state_gpu(
        &self,
        hamiltonian_matrix: &Array2<Complex64>,
        state_vec: &Array1<Complex64>,
        evolution_time: f64,
    ) -> Result<Array1<Complex64>> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::QuantumFailed("GPU not initialized".into()))?;

        let n = state_vec.len();
        let dt = evolution_time;
        let hbar = 1.0; // Use natural units

        // Convert to GPU-compatible format
        let hamiltonian_gpu_vec: Vec<(f64, f64)> = hamiltonian_matrix.iter()
            .map(|c| (c.re, c.im))
            .collect();
        let state_gpu_vec: Vec<(f64, f64)> = state_vec.iter()
            .map(|c| (c.re, c.im))
            .collect();

        // Allocate GPU memory
        let gpu_hamiltonian = device.htod_copy(hamiltonian_gpu_vec)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_state = device.htod_copy(state_gpu_vec)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;

        let mut gpu_new_state = device.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_k1 = device.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_k2 = device.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_k3 = device.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_k4 = device.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;

        // Compute RK4 stages (k1 = H·ψ, etc.)
        let cfg = LaunchConfig {
            grid_dim: ((n + 255) / 256) as u32,
            block_dim: 256,
            shared_mem_bytes: 0,
        };

        let func_matvec = device.get_func("quantum_kernels", "hamiltonian_matvec")
            .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

        // k1 = H·ψ
        unsafe {
            func_matvec.launch(cfg, (
                &gpu_hamiltonian,
                &gpu_state,
                &mut gpu_k1,
                n as u32,
            )).map_err(|e| PRCTError::QuantumFailed(format!("k1 launch failed: {}", e)))?;
        }

        // Simplified RK4: just use k1 for now (can extend to full RK4 later)
        let func_rk4 = device.get_func("quantum_kernels", "rk4_step")
            .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

        unsafe {
            func_rk4.launch(cfg, (
                &gpu_hamiltonian,
                &gpu_state,
                &mut gpu_new_state,
                &mut gpu_k1,
                &mut gpu_k2,
                &mut gpu_k3,
                &mut gpu_k4,
                n as u32,
                dt,
                hbar,
            )).map_err(|e| PRCTError::QuantumFailed(format!("RK4 launch failed: {}", e)))?;
        }

        // Copy result back
        let evolved_data: Vec<(f64, f64)> = device.dtoh_sync_copy(&gpu_new_state)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy back failed: {}", e)))?;

        let evolved: Vec<Complex64> = evolved_data.iter()
            .map(|(re, im)| Complex64::new(*re, *im))
            .collect();

        Array1::from_vec(evolved)
            .into()
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

        // GPU path: Build Hamiltonian on GPU
        if self.use_gpu {
            let hamiltonian_matrix = self.build_hamiltonian_gpu(&coupling)?;
            let dimension = hamiltonian_matrix.nrows();

            let matrix_elements: Vec<(f64, f64)> = hamiltonian_matrix.iter()
                .map(|c| (c.re, c.im))
                .collect();

            return Ok(HamiltonianState {
                matrix_elements,
                eigenvalues: vec![0.0; dimension],
                ground_state_energy: -1.0,
                dimension,
            });
        }

        // CPU fallback
        let hamiltonian = Self::build_hamiltonian_internal(&coupling)?;
        *self.hamiltonian.lock() = Some(hamiltonian.clone());

        let matrix = hamiltonian.matrix_representation();
        let matrix_elements: Vec<(f64, f64)> = matrix.iter()
            .map(|c| (c.re, c.im))
            .collect();

        let dimension = n * 3;

        Ok(HamiltonianState {
            matrix_elements,
            eigenvalues: vec![0.0; dimension],
            ground_state_energy: -1.0,
            dimension,
        })
    }

    fn evolve_state(
        &self,
        hamiltonian_state: &HamiltonianState,
        initial_state: &QuantumState,
        evolution_time: f64,
    ) -> Result<QuantumState> {
        // Convert shared-types to Array
        let state_vec: Array1<Complex64> = initial_state.amplitudes.iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect();

        // GPU path: Evolve on GPU
        if self.use_gpu {
            let dim = (hamiltonian_state.dimension as f64).sqrt() as usize;
            let hamiltonian_matrix = Array2::from_shape_vec(
                (dim, dim),
                hamiltonian_state.matrix_elements.iter()
                    .map(|(re, im)| Complex64::new(*re, *im))
                    .collect()
            ).map_err(|e| PRCTError::QuantumFailed(format!("Matrix reconstruction failed: {}", e)))?;

            let evolved = self.evolve_state_gpu(&hamiltonian_matrix, &state_vec, evolution_time)?;

            // Extract phases for coherence
            let device = self.gpu_device.as_ref().unwrap();
            let evolved_gpu_vec: Vec<(f64, f64)> = evolved.iter()
                .map(|c| (c.re, c.im))
                .collect();
            let gpu_evolved = device.htod_copy(evolved_gpu_vec.clone())
                .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;

            let mut gpu_phases = device.alloc_zeros::<f64>(evolved.len())
                .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;

            let cfg = LaunchConfig {
                grid_dim: ((evolved.len() + 255) / 256) as u32,
                block_dim: 256,
                shared_mem_bytes: 0,
            };

            let func_extract = device.get_func("quantum_kernels", "extract_phases")
                .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

            unsafe {
                func_extract.launch(cfg, (
                    &gpu_evolved,
                    &mut gpu_phases,
                    evolved.len() as u32,
                )).map_err(|e| PRCTError::QuantumFailed(format!("Phase extraction failed: {}", e)))?;
            }

            let phases: Vec<f64> = device.dtoh_sync_copy(&gpu_phases)
                .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy back failed: {}", e)))?;

            // Compute order parameter on GPU
            let mut gpu_order_param = device.alloc_zeros::<f64>(1)
                .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;

            let func_order = device.get_func("quantum_kernels", "compute_order_parameter")
                .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

            let gpu_phases_arr = device.htod_copy(phases.clone())
                .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;

            unsafe {
                func_order.launch(cfg, (
                    &gpu_phases_arr,
                    &mut gpu_order_param,
                    phases.len() as u32,
                )).map_err(|e| PRCTError::QuantumFailed(format!("Order parameter failed: {}", e)))?;
            }

            let order_vec: Vec<f64> = device.dtoh_sync_copy(&gpu_order_param)
                .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy back failed: {}", e)))?;
            let phase_coherence = order_vec[0];

            // Energy (simplified)
            let energy = evolved.iter().map(|c| c.norm_sqr()).sum::<f64>();

            let amplitudes: Vec<(f64, f64)> = evolved.iter()
                .map(|c| (c.re, c.im))
                .collect();

            return Ok(QuantumState {
                amplitudes,
                phase_coherence,
                energy,
                entanglement: 0.0,
                timestamp_ns: 0,
            });
        }

        // CPU fallback
        let mut hamiltonian_guard = self.hamiltonian.lock();
        let hamiltonian = hamiltonian_guard.as_mut()
            .ok_or_else(|| PRCTError::QuantumFailed("Hamiltonian not initialized".into()))?;

        let evolved = hamiltonian.evolve(&state_vec, evolution_time)
            .map_err(|e| PRCTError::QuantumFailed(format!("Evolution failed: {:?}", e)))?;

        let phase_coherence = hamiltonian.phase_coherence();
        let energy = hamiltonian.total_energy(&evolved);

        let amplitudes: Vec<(f64, f64)> = evolved.iter()
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
