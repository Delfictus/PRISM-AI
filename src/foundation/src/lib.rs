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
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState, ComponentHealth, DataPoint,
    DataSource, EngineConfig, HealthMetrics, HealthReport, HealthStatus, IngestionConfig,
    IngestionEngine, IngestionError, IngestionStats, RetryConfig, RetryPolicy, SourceConfig,
    SourceInfo,
};
pub use adapters::{AlpacaMarketDataSource, OpticalSensorArray, SyntheticDataSource};