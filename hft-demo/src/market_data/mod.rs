//! Market Data Module
//!
//! Provides data loading, simulation, feature extraction, and validation
//! for high-frequency trading backtesting.
//!
//! # Modules
//!
//! - `loader`: Historical data loading from CSV, Parquet, and APIs
//! - `simulator`: Market simulation with historical replay and synthetic generation
//! - `features`: Feature extraction from tick data
//! - `validation`: Data quality checks and outlier detection
//!
//! # ARES Compliance
//!
//! All modules follow ARES anti-drift standards:
//! - No hardcoded market data
//! - All metrics computed from actual data
//! - Anti-drift tests validate behavior varies with input

pub mod loader;
// pub mod simulator;    // Task 1.2
// pub mod features;     // Task 1.3
// pub mod validation;   // Task 1.4

// Re-exports
pub use loader::{
    CsvDataLoader,
    MarketTick,
    OrderBookSnapshot,
};
