/// Telecom (Network Optimization) Plugin
/// Integrates with PRISM-AI GPU graph coloring for real network optimization
use async_trait::async_trait;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::Arc;
use rand::Rng;

use super::types::*;
use super::health::{HealthStatus, HealthMetrics};
use crate::web_platform::types::*;
use crate::web_platform::prism_bridge::{PrismBridge, DefaultPrismBridge, BridgeError};
use shared_types::{Graph, PhaseField, KuramotoState};

/// Telecom Plugin - Network Optimization Dashboard
///
/// Integrates with PRISM-AI GPU graph coloring to show real network optimization:
/// - Generates network topology
/// - Runs GPU-accelerated graph coloring
/// - Shows convergence from initial to optimal coloring
/// - Uses phase field and Kuramoto state for quantum-inspired ordering
pub struct TelecomPlugin {
    config: PluginConfig,
    metrics: HealthMetrics,
    running: bool,
    iteration_count: u64,
    /// PRISM-AI bridge for GPU graph coloring
    bridge: Arc<DefaultPrismBridge>,
    /// Use real GPU coloring (vs synthetic animation)
    use_real_coloring: bool,
    /// Cached network graph for consistent topology
    cached_graph: Option<Graph>,
}

impl TelecomPlugin {
    /// Create new Telecom plugin with default configuration
    pub fn new() -> Self {
        Self {
            config: PluginConfig {
                enabled: true,
                update_interval_ms: 100,  // 10 Hz network updates
                auto_restart: true,
                max_retries: 3,
                custom_params: Default::default(),
            },
            metrics: HealthMetrics::default(),
            running: false,
            iteration_count: 0,
            bridge: Arc::new(DefaultPrismBridge::new()),
            use_real_coloring: false,
            cached_graph: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PluginConfig) -> Self {
        Self {
            config,
            metrics: HealthMetrics::default(),
            running: false,
            iteration_count: 0,
            bridge: Arc::new(DefaultPrismBridge::new()),
            use_real_coloring: false,
            cached_graph: None,
        }
    }

    /// Create with PRISM-AI bridge for real GPU coloring
    pub fn with_bridge(bridge: Arc<DefaultPrismBridge>) -> Self {
        Self {
            config: PluginConfig {
                enabled: true,
                update_interval_ms: 100,
                auto_restart: true,
                max_retries: 3,
                custom_params: Default::default(),
            },
            metrics: HealthMetrics::default(),
            running: false,
            iteration_count: 0,
            bridge,
            use_real_coloring: true,
            cached_graph: None,
        }
    }

    /// Generate network topology graph (50 nodes, random edges)
    fn generate_network_graph() -> Graph {
        let mut rng = rand::thread_rng();
        let n = 50;

        // Create adjacency matrix
        let mut adjacency = vec![false; n * n];
        for i in 0..n {
            let num_neighbors = rng.gen_range(2..=4);
            for _ in 0..num_neighbors {
                let j = rng.gen_range(0..n);
                if i != j {
                    // Symmetric adjacency
                    adjacency[i * n + j] = true;
                    adjacency[j * n + i] = true;
                }
            }
        }

        Graph {
            num_vertices: n,
            num_edges: adjacency.iter().filter(|&&b| b).count() / 2,
            adjacency,
        }
    }

    /// Generate network update using real GPU graph coloring
    async fn generate_gpu_network_update(&mut self) -> Result<TelecomUpdate, PluginError> {
        let mut rng = rand::thread_rng();

        // Get or create cached graph
        let graph = if let Some(ref g) = self.cached_graph {
            g.clone()
        } else {
            let g = Self::generate_network_graph();
            self.cached_graph = Some(g.clone());
            g
        };

        // Get system state from PRISM bridge (for phase field and Kuramoto)
        let system_state = self.bridge.get_system_state().await
            .map_err(|e| PluginError::DataGenerationFailed(format!("Bridge error: {}", e)))?;

        // Use phase field and Kuramoto state from PRISM-AI core
        let phase_field = system_state.phase_field
            .ok_or_else(|| PluginError::DataGenerationFailed("No phase field available".to_string()))?;

        let kuramoto = system_state.kuramoto_state
            .ok_or_else(|| PluginError::DataGenerationFailed("No Kuramoto state available".to_string()))?;

        // Run GPU graph coloring
        let coloring_result = self.bridge.run_graph_coloring(&graph, &phase_field, &kuramoto).await
            .map_err(|e| PluginError::DataGenerationFailed(format!("Graph coloring failed: {}", e)))?;

        // Generate nodes with GPU-computed colors
        let mut nodes = Vec::new();
        for i in 0..graph.num_vertices {
            let color = if i < coloring_result.colors.len() {
                coloring_result.colors[i]
            } else {
                1  // Fallback
            };

            nodes.push(TelecomNode {
                id: format!("NODE-{:02}", i),
                x: rng.gen_range(0.0..1000.0),
                y: rng.gen_range(0.0..800.0),
                color,
                load: rng.gen_range(0.1..1.0),
            });
        }

        // Generate edges from graph adjacency
        let mut edges = Vec::new();
        for i in 0..graph.num_vertices {
            for j in (i + 1)..graph.num_vertices {
                if graph.adjacency[i * graph.num_vertices + j] {
                    edges.push(TelecomEdge {
                        source: format!("NODE-{:02}", i),
                        target: format!("NODE-{:02}", j),
                        bandwidth: rng.gen_range(100.0..1000.0),
                    });
                }
            }
        }

        let network = TelecomNetwork { nodes, edges };

        // Metrics from real GPU coloring
        let metrics = TelecomMetrics {
            iteration: self.iteration_count,
            num_colors: coloring_result.chromatic_number as u32,
            convergence_rate: 1.0 - (coloring_result.conflicts as f64 / graph.num_edges as f64),
            total_bandwidth: edges.iter().map(|e| e.bandwidth).sum(),
        };

        Ok(TelecomUpdate { network, metrics })
    }

    /// Generate telecom network update (synthetic fallback)
    fn generate_network_update(iteration: u64) -> Result<TelecomUpdate, PluginError> {
        let mut rng = rand::thread_rng();

        // Generate 50-node network
        let mut nodes = Vec::new();
        for i in 0..50 {
            // Convergence animation: start with 15 colors, converge to 7
            let color = if iteration < 100 {
                // Early: random colors (15 options)
                (i % 15) + 1
            } else {
                // Late: converged (7 colors)
                (i % 7) + 1
            };

            nodes.push(TelecomNode {
                id: format!("NODE-{:02}", i),
                x: rng.gen_range(0.0..1000.0),
                y: rng.gen_range(0.0..800.0),
                color,
                load: rng.gen_range(0.1..1.0),
            });
        }

        // Generate edges (each node connects to 2-4 neighbors)
        let mut edges = Vec::new();
        for i in 0..50 {
            let num_edges = rng.gen_range(2..=4);
            for _ in 0..num_edges {
                let target = rng.gen_range(0..50);
                if target != i {
                    edges.push(TelecomEdge {
                        source: format!("NODE-{:02}", i),
                        target: format!("NODE-{:02}", target),
                        bandwidth: rng.gen_range(100.0..1000.0),
                    });
                }
            }
        }

        let network = TelecomNetwork { nodes, edges };

        // Optimization metrics improve over time
        let convergence = if iteration < 100 {
            iteration as f64 / 100.0
        } else {
            1.0
        };

        let metrics = TelecomMetrics {
            iteration,
            num_colors: if iteration < 100 {
                15 - ((iteration / 10) as u32)
            } else {
                7
            },
            convergence_rate: convergence,
            total_bandwidth: 45000.0 + (convergence * 5000.0),
        };

        Ok(TelecomUpdate { network, metrics })
    }
}

impl Default for TelecomPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PrismPlugin for TelecomPlugin {
    fn id(&self) -> &str {
        "telecom"
    }

    fn name(&self) -> &str {
        "Network Optimization (Telecom)"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Generates real-time network optimization visualization showing \
         graph coloring convergence from 15 colors to 7 colors"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![
            Capability::TelemetryGenerator,
            Capability::RealtimeStreaming,
            Capability::Custom("graph_optimization".to_string()),
            Capability::Custom("convergence_animation".to_string()),
        ]
    }

    async fn initialize(&mut self) -> Result<(), PluginError> {
        println!("📡 Initializing Telecom Plugin...");
        println!("   - Network Size: 50 nodes");

        // Try to initialize PRISM-AI GPU coloring
        if self.use_real_coloring {
            println!("   - Connecting to PRISM-AI GPU Graph Coloring...");

            // Initialize UnifiedPlatform first (for phase field and Kuramoto)
            match self.bridge.initialize_platform(50).await {  // 50D for network graph
                Ok(_) => {
                    println!("   ✅ UnifiedPlatform connected (for quantum state)");
                }
                Err(e) => {
                    println!("   ⚠️  UnifiedPlatform unavailable ({})", e);
                    self.use_real_coloring = false;
                }
            }

            // Initialize GPU coloring
            if self.use_real_coloring {
                match self.bridge.initialize_gpu_coloring().await {
                    Ok(_) => {
                        println!("   ✅ Real GPU graph coloring enabled");
                        self.use_real_coloring = true;
                    }
                    Err(e) => {
                        println!("   ⚠️  GPU coloring unavailable ({}), using synthetic animation", e);
                        self.use_real_coloring = false;
                    }
                }
            }
        } else {
            println!("   - Using synthetic convergence animation (15 → 7 colors)");
        }

        println!("   - Update Rate: 10 Hz");
        Ok(())
    }

    async fn start(&mut self) -> Result<(), PluginError> {
        if self.running {
            return Err(PluginError::AlreadyRunning(self.id().to_string()));
        }
        self.running = true;
        self.iteration_count = 0;
        self.metrics = HealthMetrics::default();
        println!("▶️  Telecom Plugin started ({}ms interval)", self.config.update_interval_ms);
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), PluginError> {
        if !self.running {
            return Err(PluginError::NotRunning(self.id().to_string()));
        }
        self.running = false;
        println!("⏸️  Telecom Plugin stopped at iteration {}", self.iteration_count);
        Ok(())
    }

    async fn generate_data(&mut self) -> Result<PluginData, PluginError> {
        if !self.running {
            return Err(PluginError::NotRunning(self.id().to_string()));
        }

        // Use real GPU coloring if available, otherwise use synthetic animation
        let update = if self.use_real_coloring {
            self.generate_gpu_network_update().await?
        } else {
            Self::generate_network_update(self.iteration_count)?
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PluginError::DataGenerationFailed(e.to_string()))?
            .as_secs();

        // Convert to JSON
        let payload = serde_json::to_value(&update)
            .map_err(|e| PluginError::DataGenerationFailed(e.to_string()))?;

        // Increment iteration
        self.iteration_count += 1;

        Ok(PluginData {
            plugin_id: self.id().to_string(),
            timestamp: now,
            data_type: "telecom_update".to_string(),
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
            format!("Network optimization at iteration {} ({}ms intervals)",
                    self.iteration_count, self.config.update_interval_ms)
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
        if config.update_interval_ms < 50 {
            return Err(PluginError::ConfigurationError(
                "Update interval must be at least 50ms for network updates".to_string()
            ));
        }

        if config.update_interval_ms > 5000 {
            return Err(PluginError::ConfigurationError(
                "Update interval must be at most 5000ms".to_string()
            ));
        }

        self.config = config;
        println!("🔧 Telecom Plugin reconfigured: {}ms interval", self.config.update_interval_ms);
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
    async fn test_telecom_plugin_initialization() {
        let mut plugin = TelecomPlugin::new();
        let result = plugin.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_telecom_plugin_lifecycle() {
        let mut plugin = TelecomPlugin::new();
        plugin.initialize().await.unwrap();

        // Start
        let result = plugin.start().await;
        assert!(result.is_ok());

        // Generate data
        let data = plugin.generate_data().await.unwrap();
        assert_eq!(data.plugin_id, "telecom");
        assert_eq!(data.data_type, "telecom_update");

        // Stop
        let result = plugin.stop().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_network_convergence() {
        // Early iteration - many colors
        let early_update = TelecomPlugin::generate_network_update(10).unwrap();
        assert!(early_update.metrics.num_colors > 7);

        // Late iteration - converged
        let late_update = TelecomPlugin::generate_network_update(150).unwrap();
        assert_eq!(late_update.metrics.num_colors, 7);
        assert_eq!(late_update.metrics.convergence_rate, 1.0);
    }
}
