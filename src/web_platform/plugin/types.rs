/// Core types for plugin system
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Unique identifier for plugins
pub type PluginId = String;

/// Plugin capabilities enum
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Plugin can generate telemetry data
    TelemetryGenerator,
    /// Plugin can process metrics
    MetricsProcessor,
    /// Plugin supports real-time streaming
    RealtimeStreaming,
    /// Plugin can handle GPU data
    GpuMonitoring,
    /// Plugin supports orbital mechanics
    OrbitalMechanics,
    /// Plugin can calculate transfer entropy
    TransferEntropy,
    /// Custom capability
    Custom(String),
}

/// Plugin data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginData {
    pub plugin_id: PluginId,
    pub timestamp: u64,
    pub data_type: String,
    pub payload: serde_json::Value,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    pub enabled: bool,
    pub update_interval_ms: u64,
    pub auto_restart: bool,
    pub max_retries: u32,
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval_ms: 1000,
            auto_restart: true,
            max_retries: 3,
            custom_params: HashMap::new(),
        }
    }
}

/// Plugin error types
#[derive(Debug, Clone)]
pub enum PluginError {
    InitializationFailed(String),
    StartupFailed(String),
    ShutdownFailed(String),
    DataGenerationFailed(String),
    ConfigurationError(String),
    HealthCheckFailed(String),
    NotFound(String),
    AlreadyRunning(String),
    NotRunning(String),
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginError::InitializationFailed(msg) => write!(f, "Initialization failed: {}", msg),
            PluginError::StartupFailed(msg) => write!(f, "Startup failed: {}", msg),
            PluginError::ShutdownFailed(msg) => write!(f, "Shutdown failed: {}", msg),
            PluginError::DataGenerationFailed(msg) => write!(f, "Data generation failed: {}", msg),
            PluginError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            PluginError::HealthCheckFailed(msg) => write!(f, "Health check failed: {}", msg),
            PluginError::NotFound(msg) => write!(f, "Plugin not found: {}", msg),
            PluginError::AlreadyRunning(msg) => write!(f, "Plugin already running: {}", msg),
            PluginError::NotRunning(msg) => write!(f, "Plugin not running: {}", msg),
        }
    }
}

impl std::error::Error for PluginError {}

/// Core plugin trait that all plugins must implement
#[async_trait]
pub trait PrismPlugin: Send + Sync {
    /// Get plugin unique identifier
    fn id(&self) -> &str;

    /// Get plugin display name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Get plugin description
    fn description(&self) -> &str;

    /// Get plugin capabilities
    fn capabilities(&self) -> Vec<Capability>;

    /// Initialize plugin (called once on registration)
    async fn initialize(&mut self) -> Result<(), PluginError>;

    /// Start plugin data generation (called when enabled)
    async fn start(&mut self) -> Result<(), PluginError>;

    /// Stop plugin data generation (called when disabled)
    async fn stop(&mut self) -> Result<(), PluginError>;

    /// Generate plugin data (called at configured intervals)
    /// Note: &mut self to allow stateful plugins that interact with PRISM-AI core
    async fn generate_data(&mut self) -> Result<PluginData, PluginError>;

    /// Perform health check
    fn health_check(&self) -> super::health::HealthStatus;

    /// Reconfigure plugin at runtime
    async fn reconfigure(&mut self, config: PluginConfig) -> Result<(), PluginError>;

    /// Get current configuration
    fn get_config(&self) -> &PluginConfig;
}

/// Plugin metadata for registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub id: PluginId,
    pub name: String,
    pub version: String,
    pub description: String,
    pub capabilities: Vec<Capability>,
    pub enabled: bool,
    pub running: bool,
}

impl PluginMetadata {
    pub fn from_plugin(plugin: &dyn PrismPlugin, running: bool) -> Self {
        Self {
            id: plugin.id().to_string(),
            name: plugin.name().to_string(),
            version: plugin.version().to_string(),
            description: plugin.description().to_string(),
            capabilities: plugin.capabilities(),
            enabled: plugin.get_config().enabled,
            running,
        }
    }
}
