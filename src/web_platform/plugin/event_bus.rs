/// Event bus for plugin communication
use tokio::sync::{broadcast, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

use super::types::PluginId;

/// Event types in the plugin system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginEvent {
    /// Plugin registered with the system
    PluginRegistered { plugin_id: PluginId },
    /// Plugin started
    PluginStarted { plugin_id: PluginId },
    /// Plugin stopped
    PluginStopped { plugin_id: PluginId },
    /// Plugin health changed
    PluginHealthChanged { plugin_id: PluginId, healthy: bool },
    /// Plugin data generated
    DataGenerated { plugin_id: PluginId, data_size: usize },
    /// Plugin error occurred
    PluginError { plugin_id: PluginId, error: String },
    /// Plugin configuration changed
    ConfigurationChanged { plugin_id: PluginId },
    /// Custom event
    Custom { event_type: String, data: serde_json::Value },
}

/// Event bus for broadcasting plugin events
pub struct EventBus {
    /// Global event channel
    sender: broadcast::Sender<PluginEvent>,
    /// Per-plugin event channels
    plugin_senders: Arc<RwLock<HashMap<PluginId, broadcast::Sender<PluginEvent>>>>,
}

impl EventBus {
    /// Create a new event bus with specified capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            plugin_senders: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Publish an event to all subscribers
    pub async fn publish(&self, event: PluginEvent) {
        // Send to global channel
        let _ = self.sender.send(event.clone());

        // Send to plugin-specific channel if applicable
        if let Some(plugin_id) = self.extract_plugin_id(&event) {
            let senders = self.plugin_senders.read().await;
            if let Some(plugin_sender) = senders.get(&plugin_id) {
                let _ = plugin_sender.send(event);
            }
        }
    }

    /// Subscribe to all events
    pub fn subscribe(&self) -> broadcast::Receiver<PluginEvent> {
        self.sender.subscribe()
    }

    /// Subscribe to events for a specific plugin
    pub async fn subscribe_to_plugin(&self, plugin_id: &str) -> broadcast::Receiver<PluginEvent> {
        let mut senders = self.plugin_senders.write().await;
        let plugin_sender = senders.entry(plugin_id.to_string())
            .or_insert_with(|| {
                let (tx, _) = broadcast::channel(100);
                tx
            });
        plugin_sender.subscribe()
    }

    /// Get number of global subscribers
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Extract plugin ID from event
    fn extract_plugin_id(&self, event: &PluginEvent) -> Option<String> {
        match event {
            PluginEvent::PluginRegistered { plugin_id } => Some(plugin_id.clone()),
            PluginEvent::PluginStarted { plugin_id } => Some(plugin_id.clone()),
            PluginEvent::PluginStopped { plugin_id } => Some(plugin_id.clone()),
            PluginEvent::PluginHealthChanged { plugin_id, .. } => Some(plugin_id.clone()),
            PluginEvent::DataGenerated { plugin_id, .. } => Some(plugin_id.clone()),
            PluginEvent::PluginError { plugin_id, .. } => Some(plugin_id.clone()),
            PluginEvent::ConfigurationChanged { plugin_id } => Some(plugin_id.clone()),
            PluginEvent::Custom { .. } => None,
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_bus_publish_subscribe() {
        let bus = EventBus::new(10);
        let mut receiver = bus.subscribe();

        let event = PluginEvent::PluginStarted {
            plugin_id: "test_plugin".to_string(),
        };

        bus.publish(event.clone()).await;

        let received = receiver.recv().await.unwrap();
        match received {
            PluginEvent::PluginStarted { plugin_id } => {
                assert_eq!(plugin_id, "test_plugin");
            }
            _ => panic!("Unexpected event type"),
        }
    }

    #[tokio::test]
    async fn test_plugin_specific_subscription() {
        let bus = EventBus::new(10);
        let mut receiver = bus.subscribe_to_plugin("plugin_a").await;

        // Publish event for plugin_a
        bus.publish(PluginEvent::PluginStarted {
            plugin_id: "plugin_a".to_string(),
        }).await;

        // Publish event for plugin_b
        bus.publish(PluginEvent::PluginStarted {
            plugin_id: "plugin_b".to_string(),
        }).await;

        // Should receive plugin_a event
        let received = receiver.recv().await.unwrap();
        match received {
            PluginEvent::PluginStarted { plugin_id } => {
                assert_eq!(plugin_id, "plugin_a");
            }
            _ => panic!("Unexpected event type"),
        }
    }
}
