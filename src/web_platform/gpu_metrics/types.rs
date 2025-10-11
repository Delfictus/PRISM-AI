/// GPU metrics types
use serde::{Deserialize, Serialize};

/// Comprehensive GPU metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub device_id: u32,
    pub name: String,
    pub uuid: String,
    pub temperature: GpuTemperature,
    pub utilization: GpuUtilization,
    pub memory: GpuMemory,
    pub power: GpuPower,
    pub clocks: GpuClocks,
    pub pcie: GpuPcie,
    pub processes: Vec<GpuProcess>,
    pub timestamp: u64,
}

/// GPU temperature metrics (Celsius)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTemperature {
    pub gpu: f64,
    pub memory: Option<f64>,
    pub threshold_shutdown: f64,
    pub threshold_slowdown: f64,
}

/// GPU utilization metrics (percentage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUtilization {
    pub gpu: f64,
    pub memory: f64,
    pub encoder: Option<f64>,
    pub decoder: Option<f64>,
}

/// GPU memory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemory {
    pub total_mb: u64,
    pub used_mb: u64,
    pub free_mb: u64,
    pub utilization_percent: f64,
}

/// GPU power metrics (Watts)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPower {
    pub current_watts: f64,
    pub limit_watts: f64,
    pub default_limit_watts: f64,
    pub min_limit_watts: f64,
    pub max_limit_watts: f64,
    pub utilization_percent: f64,
}

/// GPU clock speeds (MHz)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuClocks {
    pub graphics_mhz: u32,
    pub sm_mhz: u32,
    pub memory_mhz: u32,
    pub video_mhz: u32,
}

/// PCIe bus information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPcie {
    pub bus_id: String,
    pub link_gen_current: u32,
    pub link_gen_max: u32,
    pub link_width_current: u32,
    pub link_width_max: u32,
    pub throughput_rx_mbps: f64,
    pub throughput_tx_mbps: f64,
}

/// Process running on GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
    pub memory_used_mb: u64,
    pub gpu_utilization: Option<f64>,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub uuid: String,
    pub architecture: String,
    pub cuda_compute_capability: (u32, u32),
    pub total_memory_mb: u64,
    pub pcie_bus_id: String,
    pub driver_version: String,
    pub cuda_version: String,
}

/// GPU metrics collection error
#[derive(Debug, Clone)]
pub enum GpuError {
    NvmlNotAvailable(String),
    NvmlInitFailed(String),
    DeviceNotFound(u32),
    MetricQueryFailed(String),
    PermissionDenied(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NvmlNotAvailable(msg) => write!(f, "NVML not available: {}", msg),
            GpuError::NvmlInitFailed(msg) => write!(f, "NVML initialization failed: {}", msg),
            GpuError::DeviceNotFound(id) => write!(f, "GPU device {} not found", id),
            GpuError::MetricQueryFailed(msg) => write!(f, "Metric query failed: {}", msg),
            GpuError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}
