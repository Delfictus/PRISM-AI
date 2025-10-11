/// Aggregate root pattern for event sourcing
use async_trait::async_trait;
use super::types::*;
use super::store::EventStore;
use std::sync::Arc;

/// Aggregate trait - domain entities that emit events
#[async_trait]
pub trait Aggregate: Send + Sync {
    /// Get aggregate ID
    fn aggregate_id(&self) -> &str;

    /// Get aggregate type
    fn aggregate_type(&self) -> &str;

    /// Get current version
    fn version(&self) -> EventVersion;

    /// Apply event to update aggregate state
    fn apply_event(&mut self, event: &DomainEvent);

    /// Validate state (business rules)
    fn validate(&self) -> Result<(), String>;
}

/// Aggregate root - manages aggregate lifecycle with event store
pub struct AggregateRoot<T: Aggregate> {
    /// The aggregate instance
    pub aggregate: T,
    /// Event store reference
    event_store: Arc<EventStore>,
    /// Uncommitted events (for transaction-like behavior)
    uncommitted_events: Vec<DomainEvent>,
}

impl<T: Aggregate> AggregateRoot<T> {
    /// Create new aggregate root
    pub fn new(aggregate: T, event_store: Arc<EventStore>) -> Self {
        Self {
            aggregate,
            event_store,
            uncommitted_events: Vec::new(),
        }
    }

    /// Load aggregate from event store
    pub async fn load(
        mut aggregate: T,
        event_store: Arc<EventStore>,
    ) -> Result<Self, EventStoreError> {
        let aggregate_id = aggregate.aggregate_id();

        // Try to load from snapshot first
        if let Some(snapshot) = event_store.load_snapshot(aggregate_id).await? {
            // Deserialize snapshot state into aggregate
            // (In real implementation, aggregate would implement FromSnapshot trait)
            println!("📸 Loaded snapshot for {} at version {}", aggregate_id, snapshot.version);

            // Load events since snapshot
            let events = event_store.get_events_since(aggregate_id, snapshot.version).await?;
            for event in events {
                aggregate.apply_event(&event);
            }
        } else {
            // No snapshot - replay all events
            let events = event_store.get_events(aggregate_id).await?;
            for event in events {
                aggregate.apply_event(&event);
            }
        }

        Ok(Self {
            aggregate,
            event_store,
            uncommitted_events: Vec::new(),
        })
    }

    /// Record an event (not yet persisted)
    pub fn record_event(
        &mut self,
        event_type: impl Into<String>,
        payload: serde_json::Value,
    ) -> Result<(), String> {
        let new_version = self.aggregate.version() + 1;

        let event = DomainEvent::new(
            self.aggregate.aggregate_id().to_string(),
            self.aggregate.aggregate_type(),
            event_type,
            new_version,
            payload,
        );

        // Apply to aggregate
        self.aggregate.apply_event(&event);

        // Validate business rules
        self.aggregate.validate()?;

        // Add to uncommitted
        self.uncommitted_events.push(event);

        Ok(())
    }

    /// Commit all uncommitted events to event store
    pub async fn commit(&mut self) -> Result<(), EventStoreError> {
        if self.uncommitted_events.is_empty() {
            return Ok(());
        }

        // Get expected version (version before first uncommitted event)
        let expected_version = if self.uncommitted_events.is_empty() {
            self.aggregate.version()
        } else {
            self.uncommitted_events[0].version - 1
        };

        // Persist events
        for event in &self.uncommitted_events {
            self.event_store.append_event(
                &event.aggregate_id,
                &event.aggregate_type,
                &event.event_type,
                event.payload.clone(),
                Some(expected_version),
            ).await?;
        }

        // Clear uncommitted
        self.uncommitted_events.clear();

        Ok(())
    }

    /// Get uncommitted events
    pub fn uncommitted_events(&self) -> &[DomainEvent] {
        &self.uncommitted_events
    }

    /// Has uncommitted changes
    pub fn has_uncommitted_changes(&self) -> bool {
        !self.uncommitted_events.is_empty()
    }

    /// Get reference to aggregate
    pub fn get(&self) -> &T {
        &self.aggregate
    }

    /// Get mutable reference to aggregate
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.aggregate
    }
}

/// Example plugin aggregate
pub struct PluginAggregate {
    pub id: String,
    pub version: EventVersion,
    pub running: bool,
    pub start_count: u32,
    pub stop_count: u32,
}

impl PluginAggregate {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: 0,
            running: false,
            start_count: 0,
            stop_count: 0,
        }
    }
}

#[async_trait]
impl Aggregate for PluginAggregate {
    fn aggregate_id(&self) -> &str {
        &self.id
    }

    fn aggregate_type(&self) -> &str {
        "Plugin"
    }

    fn version(&self) -> EventVersion {
        self.version
    }

    fn apply_event(&mut self, event: &DomainEvent) {
        self.version = event.version;

        match event.event_type.as_str() {
            "PluginStarted" => {
                self.running = true;
                self.start_count += 1;
            }
            "PluginStopped" => {
                self.running = false;
                self.stop_count += 1;
            }
            "DataGenerated" => {
                // Track data generation (could increment counter)
            }
            _ => {
                println!("⚠️  Unknown event type: {}", event.event_type);
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        // Business rule: can't have more stops than starts
        if self.stop_count > self.start_count {
            return Err("Invalid state: more stops than starts".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_aggregate_root_record_and_commit() {
        let event_store = Arc::new(EventStore::new());
        let aggregate = PluginAggregate::new("plugin-1");
        let mut root = AggregateRoot::new(aggregate, event_store.clone());

        // Record event
        root.record_event("PluginStarted", serde_json::json!({"plugin_id": "pwsa"}))
            .unwrap();

        assert!(root.has_uncommitted_changes());
        assert_eq!(root.aggregate.version, 1);
        assert!(root.aggregate.running);

        // Commit
        root.commit().await.unwrap();
        assert!(!root.has_uncommitted_changes());

        // Verify stored
        let events = event_store.get_events("plugin-1").await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "PluginStarted");
    }

    #[tokio::test]
    async fn test_aggregate_load() {
        let event_store = Arc::new(EventStore::new());

        // Create and persist events
        event_store.append_event(
            "plugin-1",
            "Plugin",
            "PluginStarted",
            serde_json::json!({}),
            None,
        ).await.unwrap();

        event_store.append_event(
            "plugin-1",
            "Plugin",
            "PluginStopped",
            serde_json::json!({}),
            Some(1),
        ).await.unwrap();

        // Load aggregate
        let aggregate = PluginAggregate::new("plugin-1");
        let root = AggregateRoot::load(aggregate, event_store.clone())
            .await
            .unwrap();

        assert_eq!(root.aggregate.version, 2);
        assert!(!root.aggregate.running);
        assert_eq!(root.aggregate.start_count, 1);
        assert_eq!(root.aggregate.stop_count, 1);
    }

    #[tokio::test]
    async fn test_validation() {
        let event_store = Arc::new(EventStore::new());
        let aggregate = PluginAggregate::new("plugin-1");
        let mut root = AggregateRoot::new(aggregate, event_store.clone());

        // Try to stop without starting (should fail validation)
        let result = root.record_event("PluginStopped", serde_json::json!({}));
        assert!(result.is_err());
    }
}
