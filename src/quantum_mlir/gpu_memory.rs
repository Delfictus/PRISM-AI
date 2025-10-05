//! GPU Memory Management for Quantum MLIR
//!
//! Handles GPU memory allocation and data transfer using cudarc

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use cudarc::driver::result::DriverError;
use std::sync::Arc;
use anyhow::{Result, Context};

use super::cuda_kernels::CudaComplex;
use super::Complex64;

/// GPU memory manager for quantum states
pub struct GpuMemoryManager {
    device: Arc<CudaDevice>,
}

impl GpuMemoryManager {
    /// Create new GPU memory manager
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)
            .context("Failed to initialize CUDA device")?;

        Ok(Self {
            device: Arc::new(device),
        })
    }

    /// Allocate GPU memory for quantum state
    pub fn allocate_state(&self, dimension: usize) -> Result<CudaSlice<CudaComplex>> {
        let mut state = unsafe {
            self.device.alloc::<CudaComplex>(dimension)
                .context("Failed to allocate GPU memory for quantum state")?
        };

        // Initialize to |00...0> state
        let mut init = vec![CudaComplex::zero(); dimension];
        init[0] = CudaComplex::one();

        self.device.htod_copy_into(&init, &mut state)
            .context("Failed to copy initial state to GPU")?;

        Ok(state)
    }

    /// Allocate GPU memory for Hamiltonian matrix
    pub fn allocate_hamiltonian(&self, dimension: usize) -> Result<CudaSlice<CudaComplex>> {
        let size = dimension * dimension;
        unsafe {
            self.device.alloc::<CudaComplex>(size)
                .context("Failed to allocate GPU memory for Hamiltonian")
        }
    }

    /// Copy quantum state from host to device
    pub fn upload_state(&self, host_state: &[Complex64]) -> Result<CudaSlice<CudaComplex>> {
        let cuda_state: Vec<CudaComplex> = host_state.iter()
            .map(|c| CudaComplex::new(c.real, c.imag))
            .collect();

        self.device.htod_sync_copy(&cuda_state)
            .context("Failed to upload quantum state to GPU")
    }

    /// Copy quantum state from device to host
    pub fn download_state(&self, device_state: &CudaSlice<CudaComplex>) -> Result<Vec<Complex64>> {
        let cuda_state = self.device.dtoh_sync_copy(device_state)
            .context("Failed to download quantum state from GPU")?;

        Ok(cuda_state.into_iter()
            .map(|c| Complex64 { real: c.real, imag: c.imag })
            .collect())
    }

    /// Upload Hamiltonian matrix to GPU
    pub fn upload_hamiltonian(&self, hamiltonian: &[Complex64]) -> Result<CudaSlice<CudaComplex>> {
        let cuda_ham: Vec<CudaComplex> = hamiltonian.iter()
            .map(|c| CudaComplex::new(c.real, c.imag))
            .collect();

        self.device.htod_sync_copy(&cuda_ham)
            .context("Failed to upload Hamiltonian to GPU")
    }

    /// Allocate GPU memory for measurement probabilities
    pub fn allocate_probabilities(&self, dimension: usize) -> Result<CudaSlice<f64>> {
        unsafe {
            self.device.alloc::<f64>(dimension)
                .context("Failed to allocate GPU memory for probabilities")
        }
    }

    /// Download probabilities from GPU
    pub fn download_probabilities(&self, device_probs: &CudaSlice<f64>) -> Result<Vec<f64>> {
        self.device.dtoh_sync_copy(device_probs)
            .context("Failed to download probabilities from GPU")
    }

    /// Get raw pointer to GPU memory (for FFI)
    pub fn get_ptr<T>(&self, slice: &CudaSlice<T>) -> *mut T {
        slice.device_ptr() as *mut T
    }

    /// Get const pointer to GPU memory (for FFI)
    pub fn get_const_ptr<T>(&self, slice: &CudaSlice<T>) -> *const T {
        slice.device_ptr() as *const T
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
            .context("Failed to synchronize GPU")
    }

    /// Get device properties
    pub fn get_device_info(&self) -> String {
        format!(
            "CUDA Device: {} (Compute Capability: {}.{})",
            self.device.name().unwrap_or("Unknown".to_string()),
            self.device.compute_capability().0,
            self.device.compute_capability().1
        )
    }

    /// Check available memory
    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        let (free, total) = self.device.memory_info()
            .context("Failed to get GPU memory info")?;
        Ok((free, total))
    }
}