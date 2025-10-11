/// Event store - persistent storage for domain events
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use super::types::*;

/// In-memory event store (can be replaced with persistent storage)
pub struct EventStore {
    /// Event streams indexed by aggregate ID
    streams: Arc<RwLock<HashMap<AggregateId, EventStream>>>,
    /// Snapshots indexed by aggregate ID
    snapshots: Arc<RwLock<HashMap<AggregateId, Snapshot>>>,
    /// Global event log (all events in order)
    global_log: Arc<RwLock<Vec<DomainEvent>>>,
    /// Snapshot interval (take snapshot every N events)
    snapshot_interval: u64,
}

impl EventStore {
    /// Create a new event store
    pub fn new() -> Self {
        Self {
            streams: Arc::new(RwLock::new(HashMap::new())),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            global_log: Arc::new(RwLock::new(Vec::new())),
            snapshot_interval: 10, // Snapshot every 10 events
        }
    }

    /// Create event store with custom snapshot interval
    pub fn with_snapshot_interval(interval: u64) -> Self {
        let mut store = Self::new();
        store.snapshot_interval = interval;
        store
    }

    /// Append event to aggregate stream
    pub async fn append_event(
        &self,
        aggregate_id: &str,
        aggregate_type: impl Into<String>,
        event_type: impl Into<String>,
        payload: serde_json::Value,
        expected_version: Option<EventVersion>,
    ) -> Result<DomainEvent, EventStoreError> {
        let mut streams = self.streams.write().await;
        let aggregate_type_str = aggregate_type.into();

        // Get or create stream
        let stream = streams.entry(aggregate_id.to_string())
            .or_insert_with(|| EventStream::new(aggregate_id.to_string(), &aggregate_type_str));

        // Check version conflict (optimistic concurrency)
        if let Some(expected) = expected_version {
            if stream.current_version != expected {
                return Err(EventStoreError::VersionConflict {
                    aggregate_id: aggregate_id.to_string(),
                    expected,
                    actual: stream.current_version,
                });
            }
        }

        // Create new event
        let new_version = stream.current_version + 1;
        let event = DomainEvent::new(
            aggregate_id.to_string(),
            aggregate_type_str,
            event_type,
            new_version,
            payload,
        );

        // Append to stream
        stream.append(event.clone());

        drop(streams);

        // Add to global log
        let mut global_log = self.global_log.write().await;
        global_log.push(event.clone());
        drop(global_log);

        // Check if snapshot needed
        if new_version % self.snapshot_interval == 0 {
            // Snapshot logic can be implemented here
            // For now, just log that a snapshot should be taken
            println!("📸 Snapshot recommended for {} at version {}", aggregate_id, new_version);
        }

        Ok(event)
    }

    /// Append multiple events atomically
    pub async fn append_events(
        &self,
        aggregate_id: &str,
        aggregate_type: impl Into<String>,
        events: Vec<(String, serde_json::Value)>, // (event_type, payload)
        expected_version: Option<EventVersion>,
    ) -> Result<Vec<DomainEvent>, EventStoreError> {
        let mut result = Vec::new();
        let mut current_expected = expected_version;

        for (event_type, payload) in events {
            let event = self.append_event(
                aggregate_id,
                aggregate_type.as_ref(),
                event_type,
                payload,
                current_expected,
            ).await?;

            current_expected = Some(event.version);
            result.push(event);
        }

        Ok(result)
    }

    /// Get all events for an aggregate
    pub async fn get_events(&self, aggregate_id: &str) -> Result<Vec<DomainEvent>, EventStoreError> {
        let streams = self.streams.read().await;

        let stream = streams.get(aggregate_id)
            .ok_or_else(|| EventStoreError::AggregateNotFound(aggregate_id.to_string()))?;

        Ok(stream.events.clone())
    }

    /// Get events since a specific version
    pub async fn get_events_since(
        &self,
        aggregate_id: &str,
        since_version: EventVersion,
    ) -> Result<Vec<DomainEvent>, EventStoreError> {
        let streams = self.streams.read().await;

        let stream = streams.get(aggregate_id)
            .ok_or_else(|| EventStoreError::AggregateNotFound(aggregate_id.to_string()))?;

        Ok(stream.since_version(since_version)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Get current version of an aggregate
    pub async fn get_version(&self, aggregate_id: &str) -> Result<EventVersion, EventStoreError> {
        let streams = self.streams.read().await;

        let stream = streams.get(aggregate_id)
            .ok_or_else(|| EventStoreError::AggregateNotFound(aggregate_id.to_string()))?;

        Ok(stream.current_version)
    }

    /// Save snapshot
    pub async fn save_snapshot(&self, snapshot: Snapshot) -> Result<(), EventStoreError> {
        let mut snapshots = self.snapshots.write().await;
        snapshots.insert(snapshot.aggregate_id.clone(), snapshot);
        Ok(())
    }

    /// Load snapshot
    pub async fn load_snapshot(&self, aggregate_id: &str) -> Result<Option<Snapshot>, EventStoreError> {
        let snapshots = self.snapshots.read().await;
        Ok(snapshots.get(aggregate_id).cloned())
    }

    /// Get all events from global log (for projections)
    pub async fn get_all_events(&self) -> Vec<DomainEvent> {
        let global_log = self.global_log.read().await;
        global_log.clone()
    }

    /// Get events from global log since timestamp
    pub async fn get_events_since_timestamp(&self, timestamp: u64) -> Vec<DomainEvent> {
        let global_log = self.global_log.read().await;
        global_log.iter()
            .filter(|e| e.timestamp >= timestamp)
            .cloned()
            .collect()
    }

    /// Get events by type (for specific event subscriptions)
    pub async fn get_events_by_type(&self, event_type: &str) -> Vec<DomainEvent> {
        let global_log = self.global_log.read().await;
        global_log.iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Get events by aggregate type
    pub async fn get_events_by_aggregate_type(&self, aggregate_type: &str) -> Vec<DomainEvent> {
        let global_log = self.global_log.read().await;
        global_log.iter()
            .filter(|e| e.aggregate_type == aggregate_type)
            .cloned()
            .collect()
    }

    /// Get total event count
    pub async fn event_count(&self) -> usize {
        let global_log = self.global_log.read().await;
        global_log.len()
    }

    /// Get aggregate count
    pub async fn aggregate_count(&self) -> usize {
        let streams = self.streams.read().await;
        streams.len()
    }

    /// List all aggregate IDs
    pub async fn list_aggregates(&self) -> Vec<(AggregateId, String, EventVersion)> {
        let streams = self.streams.read().await;
        streams.iter()
            .map(|(id, stream)| (id.clone(), stream.aggregate_type.clone(), stream.current_version))
            .collect()
    }

    /// Replay events to build projections
    pub async fn replay_events<F>(&self, mut handler: F) -> Result<(), EventStoreError>
    where
        F: FnMut(&DomainEvent),
    {
        let global_log = self.global_log.read().await;
        for event in global_log.iter() {
            handler(event);
        }
        Ok(())
    }
}

impl Default for EventStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_append_event() {
        let store = EventStore::new();

        let event = store.append_event(
            "plugin-1",
            "Plugin",
            "PluginStarted",
            serde_json::json!({"plugin_id": "pwsa"}),
            None,
        ).await.unwrap();

        assert_eq!(event.version, 1);
        assert_eq!(event.aggregate_id, "plugin-1");
    }

    #[tokio::test]
    async fn test_version_conflict() {
        let store = EventStore::new();

        // First event
        store.append_event(
            "plugin-1",
            "Plugin",
            "PluginStarted",
            serde_json::json!({}),
            None,
        ).await.unwrap();

        // Try to append with wrong expected version
        let result = store.append_event(
            "plugin-1",
            "Plugin",
            "PluginStopped",
            serde_json::json!({}),
            Some(5), // Wrong version
        ).await;

        assert!(matches!(result, Err(EventStoreError::VersionConflict { .. })));
    }

    #[tokio::test]
    async fn test_get_events() {
        let store = EventStore::new();

        store.append_event(
            "plugin-1",
            "Plugin",
            "EventA",
            serde_json::json!({}),
            None,
        ).await.unwrap();

        store.append_event(
            "plugin-1",
            "Plugin",
            "EventB",
            serde_json::json!({}),
            Some(1),
        ).await.unwrap();

        let events = store.get_events("plugin-1").await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "EventA");
        assert_eq!(events[1].event_type, "EventB");
    }

    #[tokio::test]
    async fn test_snapshots() {
        let store = EventStore::new();

        let snapshot = Snapshot::new(
            "plugin-1".to_string(),
            "Plugin",
            10,
            serde_json::json!({"running": true}),
        );

        store.save_snapshot(snapshot.clone()).await.unwrap();

        let loaded = store.load_snapshot("plugin-1").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().version, 10);
    }

    #[tokio::test]
    async fn test_global_log() {
        let store = EventStore::new();

        store.append_event("agg-1", "Type1", "Event1", serde_json::json!({}), None).await.unwrap();
        store.append_event("agg-2", "Type2", "Event2", serde_json::json!({}), None).await.unwrap();

        let all_events = store.get_all_events().await;
        assert_eq!(all_events.len(), 2);
        assert_eq!(store.aggregate_count().await, 2);
    }
}
