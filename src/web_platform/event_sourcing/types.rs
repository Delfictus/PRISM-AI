/// Core types for event sourcing
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Unique identifier for aggregates
pub type AggregateId = String;

/// Event sequence number
pub type EventVersion = u64;

/// Domain event - the fundamental unit of change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainEvent {
    /// Unique event ID
    pub event_id: Uuid,
    /// Aggregate ID this event belongs to
    pub aggregate_id: AggregateId,
    /// Aggregate type (e.g., "Plugin", "WebSocket", "Dashboard")
    pub aggregate_type: String,
    /// Event type/name (e.g., "PluginStarted", "DataGenerated")
    pub event_type: String,
    /// Event sequence number for this aggregate
    pub version: EventVersion,
    /// Unix timestamp (seconds)
    pub timestamp: u64,
    /// Event payload (serialized domain-specific data)
    pub payload: serde_json::Value,
    /// Optional metadata (user ID, correlation ID, etc.)
    pub metadata: EventMetadata,
}

/// Event metadata for tracing and correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// User or system that triggered this event
    pub caused_by: String,
    /// Correlation ID for request tracing
    pub correlation_id: Option<String>,
    /// Additional context
    pub context: serde_json::Value,
}

impl Default for EventMetadata {
    fn default() -> Self {
        Self {
            caused_by: "system".to_string(),
            correlation_id: None,
            context: serde_json::json!({}),
        }
    }
}

impl DomainEvent {
    /// Create a new domain event
    pub fn new(
        aggregate_id: AggregateId,
        aggregate_type: impl Into<String>,
        event_type: impl Into<String>,
        version: EventVersion,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            aggregate_id,
            aggregate_type: aggregate_type.into(),
            event_type: event_type.into(),
            version,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            payload,
            metadata: EventMetadata::default(),
        }
    }

    /// Create event with metadata
    pub fn with_metadata(mut self, metadata: EventMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Create event with correlation ID
    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.metadata.correlation_id = Some(correlation_id.into());
        self
    }
}

/// Event stream - a sequence of events for an aggregate
#[derive(Debug, Clone)]
pub struct EventStream {
    pub aggregate_id: AggregateId,
    pub aggregate_type: String,
    pub events: Vec<DomainEvent>,
    pub current_version: EventVersion,
}

impl EventStream {
    /// Create new empty event stream
    pub fn new(aggregate_id: AggregateId, aggregate_type: impl Into<String>) -> Self {
        Self {
            aggregate_id,
            aggregate_type: aggregate_type.into(),
            events: Vec::new(),
            current_version: 0,
        }
    }

    /// Append event to stream
    pub fn append(&mut self, event: DomainEvent) {
        self.current_version = event.version;
        self.events.push(event);
    }

    /// Get events since version
    pub fn since_version(&self, version: EventVersion) -> Vec<&DomainEvent> {
        self.events.iter()
            .filter(|e| e.version > version)
            .collect()
    }
}

/// Event store errors
#[derive(Debug, Clone)]
pub enum EventStoreError {
    /// Aggregate not found
    AggregateNotFound(AggregateId),
    /// Version conflict (optimistic concurrency)
    VersionConflict {
        aggregate_id: AggregateId,
        expected: EventVersion,
        actual: EventVersion,
    },
    /// Serialization error
    SerializationError(String),
    /// Storage error
    StorageError(String),
    /// Invalid event
    InvalidEvent(String),
}

impl fmt::Display for EventStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventStoreError::AggregateNotFound(id) => {
                write!(f, "Aggregate not found: {}", id)
            }
            EventStoreError::VersionConflict { aggregate_id, expected, actual } => {
                write!(f, "Version conflict for {}: expected {}, got {}",
                       aggregate_id, expected, actual)
            }
            EventStoreError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            EventStoreError::StorageError(msg) => {
                write!(f, "Storage error: {}", msg)
            }
            EventStoreError::InvalidEvent(msg) => {
                write!(f, "Invalid event: {}", msg)
            }
        }
    }
}

impl std::error::Error for EventStoreError {}

/// Snapshot for aggregate state optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub aggregate_id: AggregateId,
    pub aggregate_type: String,
    pub version: EventVersion,
    pub timestamp: u64,
    pub state: serde_json::Value,
}

impl Snapshot {
    /// Create a new snapshot
    pub fn new(
        aggregate_id: AggregateId,
        aggregate_type: impl Into<String>,
        version: EventVersion,
        state: serde_json::Value,
    ) -> Self {
        Self {
            aggregate_id,
            aggregate_type: aggregate_type.into(),
            version,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_event_creation() {
        let event = DomainEvent::new(
            "plugin-1".to_string(),
            "Plugin",
            "PluginStarted",
            1,
            serde_json::json!({"plugin_id": "pwsa"}),
        );

        assert_eq!(event.aggregate_id, "plugin-1");
        assert_eq!(event.aggregate_type, "Plugin");
        assert_eq!(event.event_type, "PluginStarted");
        assert_eq!(event.version, 1);
    }

    #[test]
    fn test_event_stream() {
        let mut stream = EventStream::new("test-1".to_string(), "Test");

        let event1 = DomainEvent::new(
            "test-1".to_string(),
            "Test",
            "EventA",
            1,
            serde_json::json!({}),
        );

        let event2 = DomainEvent::new(
            "test-1".to_string(),
            "Test",
            "EventB",
            2,
            serde_json::json!({}),
        );

        stream.append(event1);
        stream.append(event2);

        assert_eq!(stream.events.len(), 2);
        assert_eq!(stream.current_version, 2);

        let recent = stream.since_version(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].event_type, "EventB");
    }
}
