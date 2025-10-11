/// Projections - read models built from event streams
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

use super::types::*;
use super::store::EventStore;

/// Projection trait - builds read models from events
pub trait Projection: Send + Sync {
    /// Handle an event to update the projection
    fn handle(&mut self, event: &DomainEvent);

    /// Reset projection state
    fn reset(&mut self);
}

/// Plugin status projection - current state of all plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginStatus {
    pub plugin_id: String,
    pub running: bool,
    pub start_count: u32,
    pub stop_count: u32,
    pub data_generation_count: u64,
    pub last_event_timestamp: u64,
}

pub struct PluginStatusProjection {
    statuses: Arc<RwLock<HashMap<String, PluginStatus>>>,
}

impl PluginStatusProjection {
    pub fn new() -> Self {
        Self {
            statuses: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get status for a plugin
    pub async fn get_status(&self, plugin_id: &str) -> Option<PluginStatus> {
        let statuses = self.statuses.read().await;
        statuses.get(plugin_id).cloned()
    }

    /// Get all plugin statuses
    pub async fn get_all_statuses(&self) -> Vec<PluginStatus> {
        let statuses = self.statuses.read().await;
        statuses.values().cloned().collect()
    }

    /// Get running plugins count
    pub async fn running_count(&self) -> usize {
        let statuses = self.statuses.read().await;
        statuses.values().filter(|s| s.running).count()
    }

    /// Rebuild projection from event store
    pub async fn rebuild(&mut self, event_store: &EventStore) -> Result<(), EventStoreError> {
        self.reset();

        event_store.replay_events(|event| {
            self.handle(event);
        }).await?;

        Ok(())
    }
}

impl Default for PluginStatusProjection {
    fn default() -> Self {
        Self::new()
    }
}

impl Projection for PluginStatusProjection {
    fn handle(&mut self, event: &DomainEvent) {
        if event.aggregate_type != "Plugin" {
            return;
        }

        let statuses = self.statuses.clone();
        let event_clone = event.clone();

        tokio::spawn(async move {
            let mut statuses = statuses.write().await;
            let status = statuses.entry(event_clone.aggregate_id.clone())
                .or_insert_with(|| PluginStatus {
                    plugin_id: event_clone.aggregate_id.clone(),
                    running: false,
                    start_count: 0,
                    stop_count: 0,
                    data_generation_count: 0,
                    last_event_timestamp: 0,
                });

            status.last_event_timestamp = event_clone.timestamp;

            match event_clone.event_type.as_str() {
                "PluginStarted" => {
                    status.running = true;
                    status.start_count += 1;
                }
                "PluginStopped" => {
                    status.running = false;
                    status.stop_count += 1;
                }
                "DataGenerated" => {
                    status.data_generation_count += 1;
                }
                _ => {}
            }
        });
    }

    fn reset(&mut self) {
        let statuses = self.statuses.clone();
        tokio::spawn(async move {
            let mut statuses = statuses.write().await;
            statuses.clear();
        });
    }
}

/// Event statistics projection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStatistics {
    pub total_events: u64,
    pub events_by_type: HashMap<String, u64>,
    pub events_by_aggregate_type: HashMap<String, u64>,
    pub first_event_timestamp: Option<u64>,
    pub last_event_timestamp: Option<u64>,
}

pub struct EventStatisticsProjection {
    stats: Arc<RwLock<EventStatistics>>,
}

impl EventStatisticsProjection {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(RwLock::new(EventStatistics {
                total_events: 0,
                events_by_type: HashMap::new(),
                events_by_aggregate_type: HashMap::new(),
                first_event_timestamp: None,
                last_event_timestamp: None,
            })),
        }
    }

    /// Get current statistics
    pub async fn get_statistics(&self) -> EventStatistics {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Rebuild projection from event store
    pub async fn rebuild(&mut self, event_store: &EventStore) -> Result<(), EventStoreError> {
        self.reset();

        event_store.replay_events(|event| {
            self.handle(event);
        }).await?;

        Ok(())
    }
}

impl Default for EventStatisticsProjection {
    fn default() -> Self {
        Self::new()
    }
}

impl Projection for EventStatisticsProjection {
    fn handle(&mut self, event: &DomainEvent) {
        let stats = self.stats.clone();
        let event_clone = event.clone();

        tokio::spawn(async move {
            let mut stats = stats.write().await;

            stats.total_events += 1;

            *stats.events_by_type.entry(event_clone.event_type.clone())
                .or_insert(0) += 1;

            *stats.events_by_aggregate_type.entry(event_clone.aggregate_type.clone())
                .or_insert(0) += 1;

            if stats.first_event_timestamp.is_none() {
                stats.first_event_timestamp = Some(event_clone.timestamp);
            }

            stats.last_event_timestamp = Some(event_clone.timestamp);
        });
    }

    fn reset(&mut self) {
        let stats = self.stats.clone();
        tokio::spawn(async move {
            let mut stats = stats.write().await;
            stats.total_events = 0;
            stats.events_by_type.clear();
            stats.events_by_aggregate_type.clear();
            stats.first_event_timestamp = None;
            stats.last_event_timestamp = None;
        });
    }
}

/// Projection manager - coordinates multiple projections
pub struct ProjectionManager {
    plugin_status: PluginStatusProjection,
    statistics: EventStatisticsProjection,
    event_store: Arc<EventStore>,
}

impl ProjectionManager {
    pub fn new(event_store: Arc<EventStore>) -> Self {
        Self {
            plugin_status: PluginStatusProjection::new(),
            statistics: EventStatisticsProjection::new(),
            event_store,
        }
    }

    /// Handle new event (update all projections)
    pub async fn handle_event(&mut self, event: &DomainEvent) {
        self.plugin_status.handle(event);
        self.statistics.handle(event);
    }

    /// Rebuild all projections
    pub async fn rebuild_all(&mut self) -> Result<(), EventStoreError> {
        self.plugin_status.rebuild(&self.event_store).await?;
        self.statistics.rebuild(&self.event_store).await?;
        Ok(())
    }

    /// Get plugin status projection
    pub fn plugin_status(&self) -> &PluginStatusProjection {
        &self.plugin_status
    }

    /// Get statistics projection
    pub fn statistics(&self) -> &EventStatisticsProjection {
        &self.statistics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_plugin_status_projection() {
        let mut projection = PluginStatusProjection::new();

        let event1 = DomainEvent::new(
            "plugin-1".to_string(),
            "Plugin",
            "PluginStarted",
            1,
            serde_json::json!({}),
        );

        projection.handle(&event1);
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await; // Wait for async handle

        let status = projection.get_status("plugin-1").await.unwrap();
        assert!(status.running);
        assert_eq!(status.start_count, 1);
    }

    #[tokio::test]
    async fn test_event_statistics_projection() {
        let mut projection = EventStatisticsProjection::new();

        let event1 = DomainEvent::new(
            "plugin-1".to_string(),
            "Plugin",
            "PluginStarted",
            1,
            serde_json::json!({}),
        );

        let event2 = DomainEvent::new(
            "plugin-1".to_string(),
            "Plugin",
            "DataGenerated",
            2,
            serde_json::json!({}),
        );

        projection.handle(&event1);
        projection.handle(&event2);

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await; // Wait for async handles

        let stats = projection.get_statistics().await;
        assert_eq!(stats.total_events, 2);
        assert_eq!(*stats.events_by_type.get("PluginStarted").unwrap(), 1);
        assert_eq!(*stats.events_by_type.get("DataGenerated").unwrap(), 1);
    }
}
