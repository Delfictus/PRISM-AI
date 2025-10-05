//! GPU-Accelerated Thermodynamic Network
//!
//! Constitutional Compliance:
//! - Article V: Uses shared CUDA context
//! - Article VI: Data stays on GPU during evolution
//! - Article VII: PTX runtime loading (no FFI linking)
//!
//! Implements damped coupled oscillator dynamics on GPU:
//! - Langevin equation: dx/dt = v, dv/dt = F - γv + √(2γkT)*η(t)
//! - Coupling forces from network topology
//! - Entropy production tracking (2nd Law verification)

use std::sync::Arc;
use ndarray::{Array1, Array2};
use anyhow::{Result, anyhow, Context};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, DeviceRepr, ValidAsZeroBits};

use super::{ThermodynamicState, NetworkConfig};

/// GPU-accelerated thermodynamic network
///
/// Evolves coupled oscillators on GPU with thermal noise
pub struct ThermodynamicGpu {
    context: Arc<CudaContext>,

    // Kernels loaded from PTX
    init_kernel: Arc<CudaFunction>,
    forces_kernel: Arc<CudaFunction>,
    evolve_kernel: Arc<CudaFunction>,
    energy_kernel: Arc<CudaFunction>,
    entropy_kernel: Arc<CudaFunction>,
    order_kernel: Arc<CudaFunction>,

    // Configuration
    config: NetworkConfig,

    // GPU state
    positions: CudaSlice<f64>,
    velocities: CudaSlice<f64>,
    phases: CudaSlice<f64>,
    coupling_matrix: CudaSlice<f64>,

    // History for entropy tracking
    entropy_history: Vec<f64>,
    iteration: usize,
}

impl ThermodynamicGpu {
    /// Create new GPU thermodynamic network
    ///
    /// # Arguments
    /// * `context` - Shared CUDA context (Article V compliance)
    /// * `config` - Network configuration
    pub fn new(context: Arc<CudaContext>, config: NetworkConfig) -> Result<Self> {
        // Load PTX module (Article VII compliance)
        let ptx_path = "target/ptx/thermodynamic.ptx";

        if !std::path::Path::new(ptx_path).exists() {
            return Err(anyhow!("Thermodynamic PTX not found at: {}", ptx_path));
        }

        let ptx = cudarc::nvrtc::Ptx::from_file(ptx_path);
        let module = context.load_module(ptx)?;

        // Load all kernel functions
        let init_kernel = Arc::new(module.get_function("initialize_oscillators_kernel")?);
        let forces_kernel = Arc::new(module.get_function("compute_coupling_forces_kernel")?);
        let evolve_kernel = Arc::new(module.get_function("evolve_oscillators_kernel")?);
        let energy_kernel = Arc::new(module.get_function("compute_energy_kernel")?);
        let entropy_kernel = Arc::new(module.get_function("compute_entropy_kernel")?);
        let order_kernel = Arc::new(module.get_function("compute_order_parameter_kernel")?);

        let stream = context.default_stream();
        let n = config.n_oscillators;

        // Allocate GPU memory
        let mut positions: CudaSlice<f64> = stream.alloc_zeros(n)?;
        let mut velocities: CudaSlice<f64> = stream.alloc_zeros(n)?;
        let mut phases: CudaSlice<f64> = stream.alloc_zeros(n)?;

        // Initialize coupling matrix (identity for now, will be updated)
        let coupling_vec: Vec<f64> = (0..n*n).map(|i| {
            if i / n == i % n { 1.0 } else { 0.0 }
        }).collect();
        let coupling_matrix: CudaSlice<f64> = stream.memcpy_stod(&coupling_vec)?;

        // Initialize oscillators with random states
        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let seed = config.seed as u64;

        let mut launch_init = stream.launch_builder(&init_kernel);
        launch_init.arg(&mut positions);
        launch_init.arg(&mut velocities);
        launch_init.arg(&mut phases);
        launch_init.arg(&n_i32);
        launch_init.arg(&seed);
        unsafe { launch_init.launch(cfg)?; }

        Ok(Self {
            context,
            init_kernel,
            forces_kernel,
            evolve_kernel,
            energy_kernel,
            entropy_kernel,
            order_kernel,
            config,
            positions,
            velocities,
            phases,
            coupling_matrix,
            entropy_history: vec![],
            iteration: 0,
        })
    }

    /// Update coupling matrix from information flow
    pub fn update_coupling(&mut self, coupling: &Array2<f64>) -> Result<()> {
        let n = self.config.n_oscillators;

        if coupling.dim() != (n, n) {
            return Err(anyhow!("Coupling matrix dimension mismatch"));
        }

        let stream = self.context.default_stream();

        // Flatten and upload
        let coupling_flat: Vec<f64> = coupling.iter().cloned().collect();
        self.coupling_matrix = stream.memcpy_stod(&coupling_flat)?;

        Ok(())
    }

    /// Evolve network for one time step on GPU
    ///
    /// Article VI Compliance: All computation on GPU
    pub fn evolve_step(&mut self) -> Result<ThermodynamicState> {
        let stream = self.context.default_stream();
        let n = self.config.n_oscillators;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Allocate force buffer
        let mut forces: CudaSlice<f64> = stream.alloc_zeros(n)?;

        // Step 1: Compute coupling forces
        let n_i32 = n as i32;
        let coupling_strength = self.config.coupling_strength;

        let mut launch_forces = stream.launch_builder(&self.forces_kernel);
        launch_forces.arg(&self.positions);
        launch_forces.arg(&self.coupling_matrix);
        launch_forces.arg(&mut forces);
        launch_forces.arg(&n_i32);
        launch_forces.arg(&coupling_strength);
        unsafe { launch_forces.launch(cfg)?; }

        // Step 2: Evolve oscillators (Langevin dynamics)
        let dt = self.config.dt;
        let damping = self.config.damping;
        let temperature = self.config.temperature;
        let seed = (self.config.seed as u64) + self.iteration as u64;
        let iter_i32 = self.iteration as i32;

        let mut launch_evolve = stream.launch_builder(&self.evolve_kernel);
        launch_evolve.arg(&mut self.positions);
        launch_evolve.arg(&mut self.velocities);
        launch_evolve.arg(&mut self.phases);
        launch_evolve.arg(&forces);
        launch_evolve.arg(&dt);
        launch_evolve.arg(&damping);
        launch_evolve.arg(&temperature);
        launch_evolve.arg(&n_i32);
        launch_evolve.arg(&seed);
        launch_evolve.arg(&iter_i32);
        unsafe { launch_evolve.launch(cfg)?; }

        self.iteration += 1;

        // Step 3: Compute observables
        let state = self.get_state()?;

        // Track entropy for 2nd law verification
        self.entropy_history.push(state.entropy);

        Ok(state)
    }

    /// Get current thermodynamic state from GPU
    pub fn get_state(&self) -> Result<ThermodynamicState> {
        let stream = self.context.default_stream();
        let n = self.config.n_oscillators;

        // Compute energy on GPU
        let mut energy_components: CudaSlice<f64> = stream.alloc_zeros(3)?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_i32 = n as i32;
        let coupling_strength = self.config.coupling_strength;

        let mut launch_energy = stream.launch_builder(&self.energy_kernel);
        launch_energy.arg(&self.positions);
        launch_energy.arg(&self.velocities);
        launch_energy.arg(&self.coupling_matrix);
        launch_energy.arg(&mut energy_components);
        launch_energy.arg(&n_i32);
        launch_energy.arg(&coupling_strength);
        unsafe { launch_energy.launch(cfg)?; }

        // Compute entropy on GPU
        let mut entropy_result: CudaSlice<f64> = stream.alloc_zeros(1)?;
        let temperature = self.config.temperature;

        let mut launch_entropy = stream.launch_builder(&self.entropy_kernel);
        launch_entropy.arg(&self.positions);
        launch_entropy.arg(&self.velocities);
        launch_entropy.arg(&mut entropy_result);
        launch_entropy.arg(&n_i32);
        launch_entropy.arg(&temperature);
        unsafe { launch_entropy.launch(cfg)?; }

        // Compute order parameter on GPU
        let mut order_real: CudaSlice<f64> = stream.alloc_zeros(1)?;
        let mut order_imag: CudaSlice<f64> = stream.alloc_zeros(1)?;

        let mut launch_order = stream.launch_builder(&self.order_kernel);
        launch_order.arg(&self.phases);
        launch_order.arg(&mut order_real);
        launch_order.arg(&mut order_imag);
        launch_order.arg(&n_i32);
        unsafe { launch_order.launch(cfg)?; }

        // Download results
        let energy_vec = self.context.memcpy_dtoh_async(&energy_components)?;
        let entropy_vec = self.context.memcpy_dtoh_async(&entropy_result)?;
        let order_real_vec = self.context.memcpy_dtoh_async(&order_real)?;
        let order_imag_vec = self.context.memcpy_dtoh_async(&order_imag)?;
        let phases_vec = self.context.memcpy_dtoh_async(&self.phases)?;

        let total_energy = energy_vec[0] + energy_vec[1] + energy_vec[2];
        let entropy = entropy_vec[0];

        // Order parameter: r = |⟨e^(iθ)⟩| / N
        let order_r = (order_real_vec[0]*order_real_vec[0] + order_imag_vec[0]*order_imag_vec[0]).sqrt() / (n as f64);

        Ok(ThermodynamicState {
            energy: total_energy,
            entropy,
            temperature: self.config.temperature,
            phases: Array1::from_vec(phases_vec),
            order_parameter: order_r,
        })
    }

    /// Compute entropy production rate (dS/dt)
    ///
    /// Constitutional: Must be ≥ 0 (2nd Law)
    pub fn entropy_production(&self) -> f64 {
        if self.entropy_history.len() < 2 {
            return 0.0;
        }

        let n = self.entropy_history.len();
        let delta_s = self.entropy_history[n-1] - self.entropy_history[n-2];
        delta_s / self.config.dt
    }

    /// Get entropy history
    pub fn entropy_history(&self) -> &[f64] {
        &self.entropy_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermodynamic_gpu_creation() {
        if let Ok(context) = CudaContext::new(0) {
            let config = NetworkConfig {
                n_oscillators: 10,
                temperature: 1.0,
                damping: 0.1,
                dt: 0.001,
                coupling_strength: 0.5,
                enable_information_gating: false,
                seed: 42,
            };

            let thermo_gpu = ThermodynamicGpu::new(context, config);
            assert!(thermo_gpu.is_ok());
        }
    }

    #[test]
    fn test_thermodynamic_evolution() {
        if let Ok(context) = CudaContext::new(0) {
            let config = NetworkConfig {
                n_oscillators: 10,
                temperature: 1.0,
                damping: 0.1,
                dt: 0.001,
                coupling_strength: 0.5,
                enable_information_gating: false,
                seed: 42,
            };

            if let Ok(mut thermo_gpu) = ThermodynamicGpu::new(context, config) {
                // Evolve for 10 steps
                for _ in 0..10 {
                    let state = thermo_gpu.evolve_step();
                    assert!(state.is_ok());

                    let s = state.unwrap();
                    assert!(s.energy.is_finite());
                    assert!(s.entropy.is_finite());
                    assert!(s.order_parameter >= 0.0 && s.order_parameter <= 1.0);
                }

                // Check 2nd law: entropy production ≥ 0
                let ds_dt = thermo_gpu.entropy_production();
                assert!(ds_dt >= -1e-10); // Allow tiny numerical error

                println!("GPU Thermodynamic evolution: dS/dt = {:.6}", ds_dt);
            }
        }
    }
}
