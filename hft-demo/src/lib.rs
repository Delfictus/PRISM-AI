//! HFT Backtesting Demo - PRISM-AI
//!
//! High-frequency trading backtesting demonstration using neuromorphic-quantum
//! prediction strategies with GPU acceleration.
//!
//! # Overview
//!
//! This library provides a complete HFT backtesting framework including:
//! - Market data loading and simulation
//! - Feature extraction from tick data
//! - Neuromorphic spike encoding for price prediction
//! - GPU-accelerated backtesting engine
//! - Performance metrics and strategy comparison
//!
//! # ARES Anti-Drift Compliance
//!
//! This implementation follows the ARES (Anti-drift, Real-time, Exact, Specific)
//! standards to ensure all metrics are computed from actual data rather than
//! hardcoded values.
//!
//! ## Forbidden Patterns
//! ```rust,ignore
//! // ❌ NEVER do this
//! pub fn predict(&self) -> f64 { 0.85 }  // Hardcoded confidence
//! pub fn sharpe_ratio(&self) -> f64 { 2.4 }  // Hardcoded metric
//! ```
//!
//! ## Required Patterns
//! ```rust,ignore
//! // ✅ Always do this
//! pub fn predict(&self, features: &MarketFeatures) -> TradingSignal {
//!     let spike_train = self.encode_features(features);
//!     let activations = self.process_spikes(&spike_train);
//!     self.decode_to_signal(activations)  // COMPUTED from actual processing
//! }
//! ```
//!
//! # Examples
//!
//! ## Basic Usage
//! ```rust,no_run
//! use hft_demo::market_data::{CsvDataLoader, MarketSimulator};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load historical data
//! let loader = CsvDataLoader::new(
//!     "data/sample_aapl_1hour.csv".to_string(),
//!     "AAPL".to_string()
//! );
//! let ticks = loader.load_all()?;
//!
//! // Create simulator
//! let mut simulator = MarketSimulator::new_historical(
//!     "data/sample_aapl_1hour.csv".to_string(),
//!     "AAPL".to_string(),
//!     1000.0  // 1000x speed
//! )?;
//!
//! // Process ticks
//! while let Some(tick) = simulator.next_tick().await? {
//!     println!("Tick: {} @ ${:.2}", tick.symbol, tick.price);
//! }
//! # Ok(())
//! # }
//! ```

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(dead_code)]  // Allow during development

pub mod market_data;

// Re-exports for convenience
pub use market_data::{
    CsvDataLoader,
    MarketTick,
    OrderBookSnapshot,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if anti-drift validation is enabled
pub fn is_anti_drift_enabled() -> bool {
    cfg!(test) || std::env::var("HFT_DEMO_STRICT_VALIDATION").is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
