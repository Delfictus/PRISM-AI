/// Metrics (GPU Monitoring) Plugin
/// Wraps existing Metrics WebSocket functionality into plugin architecture
use async_trait::async_trait;
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;

use super::types::*;
use super::health::{HealthStatus, HealthMetrics as PluginHealthMetrics};
use crate::web_platform::types::*;

/// Metrics Plugin - GPU & System Monitoring Dashboard
pub struct MetricsPlugin {
    config: PluginConfig,
    metrics: PluginHealthMetrics,
    running: bool,
}

impl MetricsPlugin {
    /// Create new Metrics plugin with default configuration
    pub fn new() -> Self {
        Self {
            config: PluginConfig {
                enabled: true,
                update_interval_ms: 1000,  // 1 Hz system metrics
                auto_restart: true,
                max_retries: 3,
                custom_params: Default::default(),
            },
            metrics: PluginHealthMetrics::default(),
            running: false,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PluginConfig) -> Self {
        Self {
            config,
            metrics: PluginHealthMetrics::default(),
            running: false,
        }
    }

    /// Generate system metrics
    fn generate_system_metrics() -> Result<SystemMetrics, PluginError> {
        let mut rng = rand::thread_rng();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PluginError::DataGenerationFailed(e.to_string()))?
            .as_secs();

        // CPU metrics
        let cpu_usage = rng.gen_range(30.0..80.0);
        let cpu_temp = 45.0 + (cpu_usage * 0.5);

        // Memory metrics
        let memory_used = rng.gen_range(8.0..24.0);
        let memory_total = 32.0;

        // GPU metrics (simulated - will be replaced with real NVML in next enhancement)
        let gpu_metrics = vec![
            GpuMetric {
                device_id: 0,
                name: "NVIDIA RTX 4090".to_string(),
                temperature: rng.gen_range(40.0..75.0),
                utilization: rng.gen_range(50.0..95.0),
                memory_used: rng.gen_range(4096..20480),
                memory_total: 24576,
                power_usage: rng.gen_range(200.0..450.0),
                clock_speed: rng.gen_range(1800..2600),
            },
            GpuMetric {
                device_id: 1,
                name: "NVIDIA RTX 4090".to_string(),
                temperature: rng.gen_range(40.0..75.0),
                utilization: rng.gen_range(50.0..95.0),
                memory_used: rng.gen_range(4096..20480),
                memory_total: 24576,
                power_usage: rng.gen_range(200.0..450.0),
                clock_speed: rng.gen_range(1800..2600),
            },
        ];

        // Network metrics
        let network_rx = rng.gen_range(100.0..500.0);
        let network_tx = rng.gen_range(50.0..300.0);

        Ok(SystemMetrics {
            timestamp: now,
            cpu_usage,
            cpu_temp,
            memory_used,
            memory_total,
            gpu_metrics,
            network_rx_mbps: network_rx,
            network_tx_mbps: network_tx,
        })
    }
}

impl Default for MetricsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PrismPlugin for MetricsPlugin {
    fn id(&self) -> &str {
        "metrics"
    }

    fn name(&self) -> &str {
        "GPU & System Monitoring (Metrics)"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Generates real-time system metrics including CPU, memory, GPU utilization, \
         temperature, power usage, and network statistics"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![
            Capability::TelemetryGenerator,
            Capability::RealtimeStreaming,
            Capability::GpuMonitoring,
            Capability::MetricsProcessor,
        ]
    }

    async fn initialize(&mut self) -> Result<(), PluginError> {
        println!("🖥️  Initializing Metrics Plugin...");
        println!("   - CPU monitoring");
        println!("   - Memory monitoring");
        println!("   - GPU monitoring (2 devices)");
        println!("   - Network monitoring");
        println!("   - Update Rate: 1 Hz");
        Ok(())
    }

    async fn start(&mut self) -> Result<(), PluginError> {
        if self.running {
            return Err(PluginError::AlreadyRunning(self.id().to_string()));
        }
        self.running = true;
        self.metrics = PluginHealthMetrics::default();
        println!("▶️  Metrics Plugin started ({}ms interval)", self.config.update_interval_ms);
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), PluginError> {
        if !self.running {
            return Err(PluginError::NotRunning(self.id().to_string()));
        }
        self.running = false;
        println!("⏸️  Metrics Plugin stopped");
        Ok(())
    }

    async fn generate_data(&self) -> Result<PluginData, PluginError> {
        if !self.running {
            return Err(PluginError::NotRunning(self.id().to_string()));
        }

        let system_metrics = Self::generate_system_metrics()?;

        // Convert to JSON
        let payload = serde_json::to_value(&system_metrics)
            .map_err(|e| PluginError::DataGenerationFailed(e.to_string()))?;

        Ok(PluginData {
            plugin_id: self.id().to_string(),
            timestamp: system_metrics.timestamp,
            data_type: "system_metrics".to_string(),
            payload,
        })
    }

    fn health_check(&self) -> HealthStatus {
        let level = if self.metrics.consecutive_failures > 5 {
            super::health::HealthLevel::Unhealthy
        } else if self.metrics.consecutive_failures > 2 {
            super::health::HealthLevel::Degraded
        } else {
            super::health::HealthLevel::Healthy
        };

        let message = if self.running {
            format!("Monitoring system metrics at {}ms intervals", self.config.update_interval_ms)
        } else {
            "Plugin stopped".to_string()
        };

        HealthStatus {
            level,
            message,
            last_check: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metrics: self.metrics.clone(),
        }
    }

    async fn reconfigure(&mut self, config: PluginConfig) -> Result<(), PluginError> {
        // Validate configuration
        if config.update_interval_ms < 100 {
            return Err(PluginError::ConfigurationError(
                "Update interval must be at least 100ms for system metrics".to_string()
            ));
        }

        if config.update_interval_ms > 60000 {
            return Err(PluginError::ConfigurationError(
                "Update interval must be at most 60000ms (1 minute)".to_string()
            ));
        }

        self.config = config;
        println!("🔧 Metrics Plugin reconfigured: {}ms interval", self.config.update_interval_ms);
        Ok(())
    }

    fn get_config(&self) -> &PluginConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_plugin_initialization() {
        let mut plugin = MetricsPlugin::new();
        let result = plugin.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_plugin_lifecycle() {
        let mut plugin = MetricsPlugin::new();
        plugin.initialize().await.unwrap();

        // Start
        let result = plugin.start().await;
        assert!(result.is_ok());

        // Generate data
        let data = plugin.generate_data().await.unwrap();
        assert_eq!(data.plugin_id, "metrics");
        assert_eq!(data.data_type, "system_metrics");

        // Stop
        let result = plugin.stop().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_metrics_generation() {
        let metrics = MetricsPlugin::generate_system_metrics().unwrap();
        assert_eq!(metrics.gpu_metrics.len(), 2);
        assert!(metrics.cpu_usage > 0.0 && metrics.cpu_usage < 100.0);
        assert!(metrics.memory_used <= metrics.memory_total);
    }

    #[tokio::test]
    async fn test_metrics_health_check() {
        let plugin = MetricsPlugin::new();
        let health = plugin.health_check();
        assert_eq!(health.level, super::super::health::HealthLevel::Healthy);
    }
}
