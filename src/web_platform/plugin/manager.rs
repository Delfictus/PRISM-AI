/// Plugin manager - orchestrates plugin lifecycle
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::types::*;
use super::event_bus::{EventBus, PluginEvent};
use super::health::HealthStatus;

/// Plugin container with runtime state
struct PluginContainer {
    plugin: Box<dyn PrismPlugin>,
    running: bool,
    retry_count: u32,
}

/// Plugin manager - central registry and orchestrator
pub struct PluginManager {
    /// Registered plugins
    plugins: Arc<RwLock<HashMap<PluginId, PluginContainer>>>,
    /// Event bus for plugin communication
    event_bus: Arc<EventBus>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            event_bus: Arc::new(EventBus::default()),
        }
    }

    /// Create plugin manager with custom event bus capacity
    pub fn with_event_capacity(capacity: usize) -> Self {
        Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            event_bus: Arc::new(EventBus::new(capacity)),
        }
    }

    /// Register a new plugin
    pub async fn register_plugin(&self, mut plugin: Box<dyn PrismPlugin>) -> Result<(), PluginError> {
        let plugin_id = plugin.id().to_string();

        // Initialize plugin
        plugin.initialize().await?;

        let container = PluginContainer {
            plugin,
            running: false,
            retry_count: 0,
        };

        // Add to registry
        let mut plugins = self.plugins.write().await;
        plugins.insert(plugin_id.clone(), container);
        drop(plugins);

        // Publish event
        self.event_bus.publish(PluginEvent::PluginRegistered {
            plugin_id: plugin_id.clone(),
        }).await;

        println!("✅ Plugin registered: {}", plugin_id);
        Ok(())
    }

    /// Unregister a plugin
    pub async fn unregister_plugin(&self, plugin_id: &str) -> Result<(), PluginError> {
        // Stop if running
        self.stop_plugin(plugin_id).await.ok();

        // Remove from registry
        let mut plugins = self.plugins.write().await;
        plugins.remove(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        println!("✅ Plugin unregistered: {}", plugin_id);
        Ok(())
    }

    /// Start a plugin
    pub async fn start_plugin(&self, plugin_id: &str) -> Result<(), PluginError> {
        let mut plugins = self.plugins.write().await;

        let container = plugins.get_mut(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        if container.running {
            return Err(PluginError::AlreadyRunning(plugin_id.to_string()));
        }

        // Start plugin
        container.plugin.start().await?;
        container.running = true;
        container.retry_count = 0;

        drop(plugins);

        // Publish event
        self.event_bus.publish(PluginEvent::PluginStarted {
            plugin_id: plugin_id.to_string(),
        }).await;

        println!("▶️  Plugin started: {}", plugin_id);
        Ok(())
    }

    /// Stop a plugin
    pub async fn stop_plugin(&self, plugin_id: &str) -> Result<(), PluginError> {
        let mut plugins = self.plugins.write().await;

        let container = plugins.get_mut(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        if !container.running {
            return Err(PluginError::NotRunning(plugin_id.to_string()));
        }

        // Stop plugin
        container.plugin.stop().await?;
        container.running = false;

        drop(plugins);

        // Publish event
        self.event_bus.publish(PluginEvent::PluginStopped {
            plugin_id: plugin_id.to_string(),
        }).await;

        println!("⏸️  Plugin stopped: {}", plugin_id);
        Ok(())
    }

    /// Start all enabled plugins
    pub async fn start_all(&self) -> Vec<(PluginId, Result<(), PluginError>)> {
        let plugins = self.plugins.read().await;
        let plugin_ids: Vec<PluginId> = plugins.keys()
            .filter(|id| {
                if let Some(container) = plugins.get(*id) {
                    container.plugin.get_config().enabled && !container.running
                } else {
                    false
                }
            })
            .cloned()
            .collect();
        drop(plugins);

        let mut results = Vec::new();
        for plugin_id in plugin_ids {
            let result = self.start_plugin(&plugin_id).await;
            results.push((plugin_id, result));
        }
        results
    }

    /// Stop all running plugins
    pub async fn stop_all(&self) -> Vec<(PluginId, Result<(), PluginError>)> {
        let plugins = self.plugins.read().await;
        let plugin_ids: Vec<PluginId> = plugins.keys()
            .filter(|id| {
                if let Some(container) = plugins.get(*id) {
                    container.running
                } else {
                    false
                }
            })
            .cloned()
            .collect();
        drop(plugins);

        let mut results = Vec::new();
        for plugin_id in plugin_ids {
            let result = self.stop_plugin(&plugin_id).await;
            results.push((plugin_id, result));
        }
        results
    }

    /// Generate data from a specific plugin
    pub async fn generate_data(&self, plugin_id: &str) -> Result<PluginData, PluginError> {
        let plugins = self.plugins.read().await;

        let container = plugins.get(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        if !container.running {
            return Err(PluginError::NotRunning(plugin_id.to_string()));
        }

        let data = container.plugin.generate_data().await?;

        drop(plugins);

        // Publish event
        let data_size = serde_json::to_string(&data.payload)
            .map(|s| s.len())
            .unwrap_or(0);

        self.event_bus.publish(PluginEvent::DataGenerated {
            plugin_id: plugin_id.to_string(),
            data_size,
        }).await;

        Ok(data)
    }

    /// Get health status for a plugin
    pub async fn get_health(&self, plugin_id: &str) -> Result<HealthStatus, PluginError> {
        let plugins = self.plugins.read().await;

        let container = plugins.get(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        let status = container.plugin.health_check();
        Ok(status)
    }

    /// Get health status for all plugins
    pub async fn get_all_health(&self) -> HashMap<PluginId, HealthStatus> {
        let plugins = self.plugins.read().await;

        plugins.iter()
            .map(|(id, container)| (id.clone(), container.plugin.health_check()))
            .collect()
    }

    /// Reconfigure a plugin
    pub async fn reconfigure_plugin(&self, plugin_id: &str, config: PluginConfig) -> Result<(), PluginError> {
        let mut plugins = self.plugins.write().await;

        let container = plugins.get_mut(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        container.plugin.reconfigure(config).await?;

        drop(plugins);

        // Publish event
        self.event_bus.publish(PluginEvent::ConfigurationChanged {
            plugin_id: plugin_id.to_string(),
        }).await;

        Ok(())
    }

    /// Get plugin metadata
    pub async fn get_plugin_metadata(&self, plugin_id: &str) -> Result<PluginMetadata, PluginError> {
        let plugins = self.plugins.read().await;

        let container = plugins.get(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        Ok(PluginMetadata::from_plugin(container.plugin.as_ref(), container.running))
    }

    /// List all registered plugins
    pub async fn list_plugins(&self) -> Vec<PluginMetadata> {
        let plugins = self.plugins.read().await;

        plugins.values()
            .map(|container| PluginMetadata::from_plugin(container.plugin.as_ref(), container.running))
            .collect()
    }

    /// Get event bus reference
    pub fn event_bus(&self) -> Arc<EventBus> {
        self.event_bus.clone()
    }

    /// Check if plugin exists
    pub async fn has_plugin(&self, plugin_id: &str) -> bool {
        let plugins = self.plugins.read().await;
        plugins.contains_key(plugin_id)
    }

    /// Check if plugin is running
    pub async fn is_running(&self, plugin_id: &str) -> bool {
        let plugins = self.plugins.read().await;
        plugins.get(plugin_id)
            .map(|c| c.running)
            .unwrap_or(false)
    }

    /// Get count of registered plugins
    pub async fn plugin_count(&self) -> usize {
        let plugins = self.plugins.read().await;
        plugins.len()
    }

    /// Get count of running plugins
    pub async fn running_count(&self) -> usize {
        let plugins = self.plugins.read().await;
        plugins.values().filter(|c| c.running).count()
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct MockPlugin {
        id: String,
        config: PluginConfig,
        initialized: bool,
        running: bool,
    }

    impl MockPlugin {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                config: PluginConfig::default(),
                initialized: false,
                running: false,
            }
        }
    }

    #[async_trait]
    impl PrismPlugin for MockPlugin {
        fn id(&self) -> &str { &self.id }
        fn name(&self) -> &str { "Mock Plugin" }
        fn version(&self) -> &str { "1.0.0" }
        fn description(&self) -> &str { "Test plugin" }
        fn capabilities(&self) -> Vec<Capability> { vec![Capability::TelemetryGenerator] }

        async fn initialize(&mut self) -> Result<(), PluginError> {
            self.initialized = true;
            Ok(())
        }

        async fn start(&mut self) -> Result<(), PluginError> {
            self.running = true;
            Ok(())
        }

        async fn stop(&mut self) -> Result<(), PluginError> {
            self.running = false;
            Ok(())
        }

        async fn generate_data(&self) -> Result<PluginData, PluginError> {
            Ok(PluginData {
                plugin_id: self.id.clone(),
                timestamp: 1234567890,
                data_type: "test".to_string(),
                payload: serde_json::json!({"value": 42}),
            })
        }

        fn health_check(&self) -> HealthStatus {
            HealthStatus::healthy("OK")
        }

        async fn reconfigure(&mut self, config: PluginConfig) -> Result<(), PluginError> {
            self.config = config;
            Ok(())
        }

        fn get_config(&self) -> &PluginConfig {
            &self.config
        }
    }

    #[tokio::test]
    async fn test_plugin_registration() {
        let manager = PluginManager::new();
        let plugin = Box::new(MockPlugin::new("test_plugin"));

        let result = manager.register_plugin(plugin).await;
        assert!(result.is_ok());
        assert_eq!(manager.plugin_count().await, 1);
    }

    #[tokio::test]
    async fn test_plugin_start_stop() {
        let manager = PluginManager::new();
        let plugin = Box::new(MockPlugin::new("test_plugin"));

        manager.register_plugin(plugin).await.unwrap();

        // Start plugin
        let result = manager.start_plugin("test_plugin").await;
        assert!(result.is_ok());
        assert!(manager.is_running("test_plugin").await);

        // Stop plugin
        let result = manager.stop_plugin("test_plugin").await;
        assert!(result.is_ok());
        assert!(!manager.is_running("test_plugin").await);
    }

    #[tokio::test]
    async fn test_data_generation() {
        let manager = PluginManager::new();
        let plugin = Box::new(MockPlugin::new("test_plugin"));

        manager.register_plugin(plugin).await.unwrap();
        manager.start_plugin("test_plugin").await.unwrap();

        let data = manager.generate_data("test_plugin").await.unwrap();
        assert_eq!(data.plugin_id, "test_plugin");
        assert_eq!(data.data_type, "test");
    }
}
