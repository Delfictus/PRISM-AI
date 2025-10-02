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
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, CudaFunction, CudaModule, PushKernelArg};
use std::sync::Arc;
use parking_lot::Mutex;

/// Adapter connecting PRCT domain to GPU-accelerated quantum engine
pub struct QuantumAdapter {
    hamiltonian: Arc<Mutex<Option<Hamiltonian>>>,
    phase_field: Arc<Mutex<Option<PhaseResonanceField>>>,
    gpu_device: Option<Arc<CudaContext>>,
    gpu_module: Option<Arc<CudaModule>>,
    use_gpu: bool,
}

impl QuantumAdapter {
    /// Create new GPU-accelerated quantum adapter
    pub fn new() -> Self {
        // Try to initialize GPU
        let (gpu_device, gpu_module, use_gpu) = match CudaContext::new(0) {
            Ok(device_arc) => {
                // cudarc 0.17 returns Arc<CudaContext> directly
                // Try to load GPU kernels
                match Self::load_gpu_module(&device_arc) {
                    Ok(module) => {
                        println!("✓ Quantum GPU initialized (CUDA device 0)");
                        (Some(device_arc), Some(module), true)
                    }
                    Err(e) => {
                        eprintln!("⚠ GPU kernel load failed: {}. Using CPU fallback.", e);
                        (None, None, false)
                    }
                }
            }
            Err(e) => {
                eprintln!("⚠ GPU initialization failed: {}. Using CPU fallback.", e);
                (None, None, false)
            }
        };

        Self {
            hamiltonian: Arc::new(Mutex::new(None)),
            phase_field: Arc::new(Mutex::new(None)),
            gpu_device,
            gpu_module,
            use_gpu,
        }
    }

    /// Load GPU module for quantum operations
    fn load_gpu_module(device: &Arc<CudaContext>) -> std::result::Result<Arc<CudaModule>, String> {
        // Load PTX from runtime location
        let ptx_path = "target/ptx/quantum_kernels.ptx";
        let ptx = std::fs::read_to_string(ptx_path)
            .map_err(|e| format!("Failed to load PTX: {}", e))?;

        // Load module using cudarc 0.17 API (returns Arc<CudaModule>)
        let module = device.load_module(ptx.into())
            .map_err(|e| format!("PTX load failed: {}", e))?;

        Ok(module)
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
    /// TODO: Implement with proper complex number handling (separate real/imag buffers)
    /// Currently disabled due to cudarc not supporting (f64,f64) tuple types
    #[allow(dead_code)]
    fn build_hamiltonian_gpu(&self, _coupling_matrix: &Array2<Complex64>) -> Result<Array2<Complex64>> {
        // Placeholder - needs implementation with separate real/imag buffers
        Err(PRCTError::QuantumFailed("GPU Hamiltonian construction not yet implemented".into()))
    }

    /*
    fn build_hamiltonian_gpu_impl(&self, coupling_matrix: &Array2<Complex64>) -> Result<Array2<Complex64>> {
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

        // Allocate GPU memory using stream-based API
        let stream = device.default_stream();
        let gpu_positions = stream.memcpy_stod(&positions)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_masses = stream.memcpy_stod(&masses)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;

        let mut gpu_kinetic = stream.alloc_zeros::<(f64, f64)>(dim * dim)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_potential = stream.alloc_zeros::<(f64, f64)>(dim * dim)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;

        // Launch kinetic operator kernel
        let cfg_2d = LaunchConfig {
            grid_dim: (((dim + 15) / 16) as u32, ((dim + 15) / 16) as u32, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        // Get module and functions
        let module = self.gpu_module.as_ref().ok_or_else(||
            PRCTError::QuantumFailed("GPU module not loaded".into()))?;
        let func_kinetic = module.load_function("build_kinetic_operator")
            .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

        let n_u32 = n as u32;

        let mut launch_args1 = stream.launch_builder(&func_kinetic);
        launch_args1.arg(&mut gpu_kinetic);
        launch_args1.arg(&gpu_masses);
        launch_args1.arg(&n_u32);
        launch_args1.arg(&grid_spacing);
        launch_args1.arg(&hartree_to_kcalmol);

        unsafe {
            launch_args1.launch(cfg_2d).map_err(|e| PRCTError::QuantumFailed(format!("Kernel launch failed: {}", e)))?;
        }

        // Launch potential operator kernel
        let func_potential = module.load_function("build_potential_operator")
            .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

        let n_u32_2 = n as u32;

        let mut launch_args2 = stream.launch_builder(&func_potential);
        launch_args2.arg(&mut gpu_potential);
        launch_args2.arg(&gpu_positions);
        launch_args2.arg(&n_u32_2);
        launch_args2.arg(&lj_epsilon);
        launch_args2.arg(&lj_sigma);
        launch_args2.arg(&coulomb_cutoff);
        launch_args2.arg(&kcalmol_to_hartree);

        unsafe {
            launch_args2.launch(cfg_2d).map_err(|e| PRCTError::QuantumFailed(format!("Kernel launch failed: {}", e)))?;
        }

        // Copy results back and combine using stream
        let kinetic_data: Vec<(f64, f64)> = stream.memcpy_dtov(&gpu_kinetic)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy back failed: {}", e)))?;
        let potential_data: Vec<(f64, f64)> = stream.memcpy_dtov(&gpu_potential)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy back failed: {}", e)))?;

        // H = T + V
        let hamiltonian_data: Vec<Complex64> = kinetic_data.iter().zip(potential_data.iter())
            .map(|((t_re, t_im), (v_re, v_im))| Complex64::new(t_re + v_re, t_im + v_im))
            .collect();

        Array2::from_shape_vec((dim, dim), hamiltonian_data)
            .map_err(|e| PRCTError::QuantumFailed(format!("Array construction failed: {}", e)))
    }
    */

    /// GPU-accelerated state evolution using RK4
    /// TODO: Implement with proper complex number handling (separate real/imag buffers)
    /// Currently disabled due to cudarc not supporting (f64,f64) tuple types
    #[allow(dead_code)]
    fn evolve_state_gpu(
        &self,
        _hamiltonian_matrix: &Array2<Complex64>,
        _state_vec: &Array1<Complex64>,
        _evolution_time: f64,
    ) -> Result<Array1<Complex64>> {
        // Placeholder - needs implementation with separate real/imag buffers
        Err(PRCTError::QuantumFailed("GPU state evolution not yet implemented".into()))
    }

    /*
    fn evolve_state_gpu_impl(
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

        // Allocate GPU memory using stream-based API
        let stream = device.default_stream();
        let gpu_hamiltonian = stream.memcpy_stod(&hamiltonian_gpu_vec)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;
        let gpu_state = stream.memcpy_stod(&state_gpu_vec)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy failed: {}", e)))?;

        let mut gpu_new_state = stream.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_k1 = stream.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_k2 = stream.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_k3 = stream.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_k4 = stream.alloc_zeros::<(f64, f64)>(n)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU alloc failed: {}", e)))?;

        // Compute RK4 stages (k1 = H·ψ, etc.)
        let cfg = LaunchConfig {
            grid_dim: (((n + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let module = self.gpu_module.as_ref().ok_or_else(||
            PRCTError::QuantumFailed("GPU module not loaded".into()))?;
        let func_matvec = module.load_function("hamiltonian_matvec")
            .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

        let n_u32 = n as u32;

        // k1 = H·ψ
        let mut launch_args1 = stream.launch_builder(&func_matvec);
        launch_args1.arg(&gpu_hamiltonian);
        launch_args1.arg(&gpu_state);
        launch_args1.arg(&mut gpu_k1);
        launch_args1.arg(&n_u32);

        unsafe {
            launch_args1.launch(cfg).map_err(|e| PRCTError::QuantumFailed(format!("k1 launch failed: {}", e)))?;
        }

        // Simplified RK4: just use k1 for now (can extend to full RK4 later)
        let func_rk4 = module.load_function("rk4_step")
            .map_err(|e| PRCTError::QuantumFailed(format!("Kernel not found: {}", e)))?;

        let n_u32_2 = n as u32;

        let mut launch_args2 = stream.launch_builder(&func_rk4);
        launch_args2.arg(&gpu_hamiltonian);
        launch_args2.arg(&gpu_state);
        launch_args2.arg(&mut gpu_new_state);
        launch_args2.arg(&mut gpu_k1);
        launch_args2.arg(&mut gpu_k2);
        launch_args2.arg(&mut gpu_k3);
        launch_args2.arg(&mut gpu_k4);
        launch_args2.arg(&n_u32_2);
        launch_args2.arg(&dt);
        launch_args2.arg(&hbar);

        unsafe {
            launch_args2.launch(cfg).map_err(|e| PRCTError::QuantumFailed(format!("RK4 launch failed: {}", e)))?;
        }

        // Copy result back using stream
        let evolved_data: Vec<(f64, f64)> = stream.memcpy_dtov(&gpu_new_state)
            .map_err(|e| PRCTError::QuantumFailed(format!("GPU copy back failed: {}", e)))?;

        let evolved: Vec<Complex64> = evolved_data.iter()
            .map(|(re, im)| Complex64::new(*re, *im))
            .collect();

        Array1::from_vec(evolved)
            .into()
    }
    */
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

        // GPU path: TODO - implement with proper complex number handling
        // if self.use_gpu {
        //     let hamiltonian_matrix = self.build_hamiltonian_gpu(&coupling)?;
        //     ...
        // }

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

        // GPU path: TODO - implement with proper complex number handling
        // if self.use_gpu {
        //     let evolved = self.evolve_state_gpu(...)?;
        //     ...
        // }

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
