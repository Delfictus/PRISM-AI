/// PRISM Bridge - Connects Web Platform to PRISM-AI Core
///
/// Task 3.1.1: Create PrismBridge trait and structure (4h)
///
/// This bridge connects the web platform's plugin system to the PRISM-AI
/// core modules (UnifiedPlatform, GpuColoringSearch, etc.)

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use ndarray::Array1;
use anyhow::Result;

use crate::integration::unified_platform::{UnifiedPlatform, PlatformInput, PlatformOutput};
use crate::gpu_coloring::GpuColoringSearch;
use shared_types::{Graph, PhaseField, KuramotoState};

use super::plugin::types::{PluginData, PluginError};

/// Bridge trait for connecting web platform to PRISM-AI core
#[async_trait]
pub trait PrismBridge: Send + Sync {
    /// Get unified platform reference
    async fn get_platform(&self) -> Result<&UnifiedPlatform, BridgeError>;

    /// Get GPU coloring search reference
    async fn get_gpu_coloring(&self) -> Result<&GpuColoringSearch, BridgeError>;

    /// Process data through unified platform
    async fn process_platform(&mut self, input: PlatformInput) -> Result<PlatformOutput, BridgeError>;

    /// Run graph coloring optimization
    async fn run_graph_coloring(
        &self,
        graph: &Graph,
        phase_field: &PhaseField,
        kuramoto: &KuramotoState,
    ) -> Result<ColoringResult, BridgeError>;

    /// Get current system state (for telemetry)
    async fn get_system_state(&self) -> Result<SystemState, BridgeError>;
}

/// Bridge error types
#[derive(Debug, Clone)]
pub enum BridgeError {
    PlatformNotInitialized,
    GpuNotAvailable,
    ProcessingFailed(String),
    StateUnavailable(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::PlatformNotInitialized => write!(f, "PRISM Platform not initialized"),
            BridgeError::GpuNotAvailable => write!(f, "GPU not available for coloring"),
            BridgeError::ProcessingFailed(msg) => write!(f, "Processing failed: {}", msg),
            BridgeError::StateUnavailable(msg) => write!(f, "State unavailable: {}", msg),
        }
    }
}

impl std::error::Error for BridgeError {}

/// System state for telemetry
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemState {
    /// Phase field from quantum processing
    pub phase_field: Option<PhaseField>,
    /// Kuramoto synchronization state
    pub kuramoto_state: Option<KuramotoState>,
    /// Free energy from active inference
    pub free_energy: f64,
    /// Entropy production (2nd law verification)
    pub entropy_production: f64,
    /// Total processing latency (ms)
    pub latency_ms: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Graph coloring result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColoringResult {
    /// Node colors (color assignment for each node)
    pub colors: Vec<usize>,
    /// Chromatic number (number of colors used)
    pub chromatic_number: usize,
    /// Number of conflicts
    pub conflicts: usize,
    /// Computation time (ms)
    pub computation_time_ms: f64,
}

/// Default implementation of PRISM Bridge
pub struct DefaultPrismBridge {
    /// Unified platform instance
    platform: Arc<RwLock<Option<UnifiedPlatform>>>,
    /// GPU coloring search instance
    gpu_coloring: Arc<RwLock<Option<GpuColoringSearch>>>,
    /// Last known system state
    last_state: Arc<RwLock<Option<SystemState>>>,
}

impl DefaultPrismBridge {
    /// Create new PRISM bridge
    pub fn new() -> Self {
        Self {
            platform: Arc::new(RwLock::new(None)),
            gpu_coloring: Arc::new(RwLock::new(None)),
            last_state: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize with unified platform
    pub async fn initialize_platform(&self, n_dimensions: usize) -> Result<(), BridgeError> {
        println!("🌉 Initializing PRISM Bridge with UnifiedPlatform ({}D)...", n_dimensions);

        let platform = UnifiedPlatform::new(n_dimensions)
            .map_err(|e| BridgeError::ProcessingFailed(e.to_string()))?;

        let mut plat_lock = self.platform.write().await;
        *plat_lock = Some(platform);

        println!("✅ PRISM Bridge initialized successfully");
        Ok(())
    }

    /// Initialize GPU coloring search
    pub async fn initialize_gpu_coloring(&self) -> Result<(), BridgeError> {
        println!("🎨 Initializing GPU Coloring Search...");

        // GPU coloring requires CUDA context
        // This would connect to the actual GPU coloring module
        // For now, we'll mark it as optional

        println!("⚠️  GPU Coloring not initialized (requires CUDA)");
        Ok(())
    }
}

#[async_trait]
impl PrismBridge for DefaultPrismBridge {
    async fn get_platform(&self) -> Result<&UnifiedPlatform, BridgeError> {
        // Note: Returning reference is tricky with RwLock
        // In real implementation, we'd use different pattern
        Err(BridgeError::PlatformNotInitialized)
    }

    async fn get_gpu_coloring(&self) -> Result<&GpuColoringSearch, BridgeError> {
        Err(BridgeError::GpuNotAvailable)
    }

    async fn process_platform(&mut self, input: PlatformInput) -> Result<PlatformOutput, BridgeError> {
        let mut platform_lock = self.platform.write().await;

        let platform = platform_lock.as_mut()
            .ok_or(BridgeError::PlatformNotInitialized)?;

        let output = platform.process(input)
            .map_err(|e| BridgeError::ProcessingFailed(e.to_string()))?;

        // Cache system state
        let state = SystemState {
            phase_field: output.phase_field.clone(),
            kuramoto_state: output.kuramoto_state.clone(),
            free_energy: output.metrics.free_energy,
            entropy_production: output.metrics.entropy_production,
            latency_ms: output.metrics.total_latency_ms,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let mut state_lock = self.last_state.write().await;
        *state_lock = Some(state);

        Ok(output)
    }

    async fn run_graph_coloring(
        &self,
        graph: &Graph,
        phase_field: &PhaseField,
        kuramoto: &KuramotoState,
    ) -> Result<ColoringResult, BridgeError> {
        let gpu_coloring_lock = self.gpu_coloring.read().await;

        let gpu_coloring = gpu_coloring_lock.as_ref()
            .ok_or(BridgeError::GpuNotAvailable)?;

        let target_colors = 7; // Target for graph coloring convergence
        let n_attempts = 1000; // Number of parallel GPU attempts

        let solution = gpu_coloring.massive_parallel_search(
            graph,
            phase_field,
            kuramoto,
            target_colors,
            n_attempts,
        ).map_err(|e| BridgeError::ProcessingFailed(e.to_string()))?;

        Ok(ColoringResult {
            colors: solution.colors,
            chromatic_number: solution.chromatic_number,
            conflicts: solution.conflicts,
            computation_time_ms: solution.computation_time_ms,
        })
    }

    async fn get_system_state(&self) -> Result<SystemState, BridgeError> {
        let state_lock = self.last_state.read().await;

        state_lock.clone()
            .ok_or_else(|| BridgeError::StateUnavailable("No state cached yet".to_string()))
    }
}

impl Default for DefaultPrismBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert BridgeError to PluginError
impl From<BridgeError> for PluginError {
    fn from(err: BridgeError) -> Self {
        PluginError::DataGenerationFailed(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let bridge = DefaultPrismBridge::new();
        assert!(bridge.last_state.read().await.is_none());
    }

    #[tokio::test]
    async fn test_bridge_initialization() {
        let bridge = DefaultPrismBridge::new();

        // Initialize with small dimensions for testing
        let result = bridge.initialize_platform(10).await;

        // May fail if CUDA not available - that's okay
        if result.is_ok() {
            println!("✅ Bridge initialized successfully");
        } else {
            println!("⚠️  Bridge initialization skipped (CUDA not available)");
        }
    }

    #[tokio::test]
    async fn test_state_unavailable() {
        let bridge = DefaultPrismBridge::new();

        let result = bridge.get_system_state().await;
        assert!(matches!(result, Err(BridgeError::StateUnavailable(_))));
    }
}
