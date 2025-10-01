//! Real-Time Data Ingestion Framework
//!
//! Provides async, high-throughput data ingestion from multiple sources
//! with buffering, backpressure handling, and <10ms latency targets.

pub mod buffer;
pub mod engine;
pub mod types;

pub use buffer::CircularBuffer;
pub use engine::{IngestionEngine, IngestionStats};
pub use types::{DataPoint, DataSource, SourceInfo};
