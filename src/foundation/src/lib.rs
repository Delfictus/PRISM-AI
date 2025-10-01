//! Platform Foundation
//!
//! Unified API for the world's first software-based neuromorphic-quantum computing platform

pub mod platform;
pub mod types;
pub mod ingestion;
pub mod adapters;

// Re-export main components
pub use platform::NeuromorphicQuantumPlatform;
pub use types::*;
pub use ingestion::{
    CircuitBreaker, CircuitBreakerState, DataPoint, DataSource, IngestionEngine, IngestionError,
    IngestionStats, RetryPolicy, SourceInfo,
};
pub use adapters::{AlpacaMarketDataSource, OpticalSensorArray, SyntheticDataSource};