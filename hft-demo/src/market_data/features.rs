//! Feature Extraction Module
//!
//! Extracts technical indicators and market features from tick data
//! for use in trading strategies and machine learning models.
//!
//! # ARES Anti-Drift Compliance
//!
//! All features are COMPUTED from actual market data, never hardcoded:
//!
//! ## Forbidden Patterns
//! ```rust,ignore
//! // ❌ NEVER do this
//! pub fn rsi(&self) -> f64 { 65.0 }  // Hardcoded RSI
//! pub fn volatility(&self) -> f64 { 0.25 }  // Fake volatility
//! ```
//!
//! ## Required Patterns
//! ```rust,ignore
//! // ✅ Always do this
//! pub fn rsi(&self, window: usize) -> f64 {
//!     let gains_losses = self.compute_price_changes();
//!     let avg_gain = gains_losses.gains.mean();
//!     let avg_loss = gains_losses.losses.mean();
//!     100.0 - (100.0 / (1.0 + avg_gain / avg_loss))  // COMPUTED
//! }
//! ```

use crate::market_data::MarketTick;
use anyhow::{Result, bail};
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

/// Market features extracted from tick data
///
/// All fields are COMPUTED from actual market data, ensuring ARES compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFeatures {
    /// Timestamp of feature computation (nanoseconds)
    pub timestamp_ns: u64,

    /// Symbol being analyzed
    pub symbol: String,

    // === Price-based features ===

    /// Log return over the window (dimensionless)
    pub log_return: f64,

    /// Simple return over the window (percentage)
    pub simple_return: f64,

    /// Realized volatility (standard deviation of returns)
    pub volatility: f64,

    /// Price momentum (average return over window)
    pub momentum: f64,

    /// Current price
    pub price: f64,

    // === Technical indicators ===

    /// Relative Strength Index (0-100)
    pub rsi: f64,

    /// MACD line (12-period EMA - 26-period EMA)
    pub macd_line: f64,

    /// MACD signal line (9-period EMA of MACD)
    pub macd_signal: f64,

    /// MACD histogram (MACD - Signal)
    pub macd_histogram: f64,

    /// Bollinger Band upper bound
    pub bb_upper: f64,

    /// Bollinger Band middle (SMA)
    pub bb_middle: f64,

    /// Bollinger Band lower bound
    pub bb_lower: f64,

    /// Bollinger Band width (normalized)
    pub bb_width: f64,

    // === Microstructure features ===

    /// Bid-ask spread in basis points
    pub spread_bps: f64,

    /// Order flow imbalance (-1 to 1)
    pub order_flow_imbalance: f64,

    /// Volume-weighted average price over window
    pub vwap: f64,

    /// Trade volume (shares)
    pub volume: u64,

    /// Volume momentum (current vs average)
    pub volume_momentum: f64,

    // === Normalization statistics ===

    /// Mean price over window (for normalization)
    pub price_mean: f64,

    /// Price standard deviation over window
    pub price_std: f64,

    /// Mean volume over window
    pub volume_mean: f64,

    /// Volume standard deviation over window
    pub volume_std: f64,
}

impl MarketFeatures {
    /// Normalize price-based features to [-1, 1] range for neural networks
    ///
    /// Uses z-score normalization: (x - μ) / σ, then clip to [-3, 3] and scale to [-1, 1]
    pub fn normalize_price(&self, price: f64) -> f64 {
        if self.price_std > 0.0 {
            let z_score = (price - self.price_mean) / self.price_std;
            z_score.clamp(-3.0, 3.0) / 3.0  // Clip to 3 std devs, scale to [-1, 1]
        } else {
            0.0
        }
    }

    /// Normalize volume to [0, 1] range
    pub fn normalize_volume(&self, volume: u64) -> f64 {
        if self.volume_std > 0.0 {
            let z_score = (volume as f64 - self.volume_mean) / self.volume_std;
            (z_score.clamp(-3.0, 3.0) / 3.0 + 1.0) / 2.0  // Map [-1, 1] to [0, 1]
        } else {
            0.5
        }
    }

    /// Get all features as a normalized vector for ML models
    pub fn to_normalized_vector(&self) -> Vec<f64> {
        vec![
            self.normalize_price(self.price),
            self.log_return,
            self.volatility,
            self.momentum,
            (self.rsi - 50.0) / 50.0,  // RSI to [-1, 1]
            self.macd_histogram / self.price_std.max(1.0),  // Normalize MACD
            self.bb_width,
            (self.spread_bps - 5.0) / 5.0,  // Typical spread ~5 bps
            self.order_flow_imbalance,
            self.normalize_volume(self.volume as u64),
            self.volume_momentum,
        ]
    }
}

/// Feature extractor with rolling window
///
/// Maintains a sliding window of market ticks and computes features
/// incrementally as new data arrives.
pub struct FeatureExtractor {
    /// Symbol being analyzed
    symbol: String,

    /// Rolling window of recent ticks
    tick_window: VecDeque<MarketTick>,

    /// Maximum window size
    max_window_size: usize,

    /// Minimum ticks needed before computing features
    min_ticks: usize,

    // === Cached intermediate calculations ===

    /// EMA state for MACD calculation (12-period)
    ema_12: Option<f64>,

    /// EMA state for MACD calculation (26-period)
    ema_26: Option<f64>,

    /// EMA state for MACD signal line (9-period)
    macd_signal_ema: Option<f64>,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol to analyze
    /// * `window_size` - Number of ticks to maintain in rolling window
    pub fn new(symbol: String, window_size: usize) -> Self {
        Self {
            symbol,
            tick_window: VecDeque::with_capacity(window_size),
            max_window_size: window_size,
            min_ticks: 30,  // Need at least 30 ticks for meaningful features
            ema_12: None,
            ema_26: None,
            macd_signal_ema: None,
        }
    }

    /// Add a new tick to the rolling window
    pub fn add_tick(&mut self, tick: MarketTick) {
        // Only accept ticks for our symbol
        if tick.symbol != self.symbol {
            return;
        }

        self.tick_window.push_back(tick);

        // Maintain window size
        if self.tick_window.len() > self.max_window_size {
            self.tick_window.pop_front();
        }
    }

    /// Extract features from current window
    ///
    /// Returns None if insufficient data available
    pub fn extract_features(&mut self) -> Result<Option<MarketFeatures>> {
        if self.tick_window.len() < self.min_ticks {
            return Ok(None);
        }

        // Extract tick info first (before mutable borrows)
        let timestamp_ns = self.tick_window.back()
            .ok_or_else(|| anyhow::anyhow!("Empty tick window"))?.timestamp_ns;
        let price = self.tick_window.back().unwrap().price;

        // Compute all features
        let (log_return, simple_return) = self.compute_returns()?;
        let volatility = self.compute_volatility()?;
        let momentum = self.compute_momentum()?;
        let rsi = self.compute_rsi(14)?;
        let (macd_line, macd_signal, macd_histogram) = self.compute_macd()?;
        let (bb_upper, bb_middle, bb_lower, bb_width) = self.compute_bollinger_bands(20, 2.0)?;
        let spread_bps = self.compute_spread_bps()?;
        let order_flow_imbalance = self.compute_order_flow_imbalance()?;
        let vwap = self.compute_vwap()?;
        let (volume, volume_momentum) = self.compute_volume_features()?;
        let (price_mean, price_std) = self.compute_price_statistics()?;
        let (volume_mean, volume_std) = self.compute_volume_statistics()?;

        Ok(Some(MarketFeatures {
            timestamp_ns,
            symbol: self.symbol.clone(),
            log_return,
            simple_return,
            volatility,
            momentum,
            price,
            rsi,
            macd_line,
            macd_signal,
            macd_histogram,
            bb_upper,
            bb_middle,
            bb_lower,
            bb_width,
            spread_bps,
            order_flow_imbalance,
            vwap,
            volume,
            volume_momentum,
            price_mean,
            price_std,
            volume_mean,
            volume_std,
        }))
    }

    /// Compute log and simple returns
    fn compute_returns(&self) -> Result<(f64, f64)> {
        if self.tick_window.len() < 2 {
            bail!("Need at least 2 ticks to compute returns");
        }

        let first_price = self.tick_window.front().unwrap().price;
        let last_price = self.tick_window.back().unwrap().price;

        let simple_return = (last_price - first_price) / first_price;
        let log_return = (last_price / first_price).ln();

        Ok((log_return, simple_return))
    }

    /// Compute realized volatility (standard deviation of returns)
    fn compute_volatility(&self) -> Result<f64> {
        if self.tick_window.len() < 2 {
            return Ok(0.0);
        }

        // Compute tick-to-tick returns
        let returns: Vec<f64> = self.tick_window
            .iter()
            .zip(self.tick_window.iter().skip(1))
            .map(|(prev, curr)| (curr.price / prev.price).ln())
            .collect();

        if returns.is_empty() {
            return Ok(0.0);
        }

        // Compute standard deviation
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }

    /// Compute momentum (average return over window)
    fn compute_momentum(&self) -> Result<f64> {
        if self.tick_window.len() < 2 {
            return Ok(0.0);
        }

        let returns: Vec<f64> = self.tick_window
            .iter()
            .zip(self.tick_window.iter().skip(1))
            .map(|(prev, curr)| (curr.price - prev.price) / prev.price)
            .collect();

        Ok(returns.iter().sum::<f64>() / returns.len() as f64)
    }

    /// Compute Relative Strength Index
    ///
    /// RSI = 100 - (100 / (1 + RS))
    /// where RS = Average Gain / Average Loss over period
    fn compute_rsi(&self, period: usize) -> Result<f64> {
        if self.tick_window.len() < period + 1 {
            return Ok(50.0);  // Neutral RSI if insufficient data
        }

        // Compute price changes
        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 1..self.tick_window.len() {
            let change = self.tick_window[i].price - self.tick_window[i-1].price;
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        // Take last 'period' changes
        let start_idx = gains.len().saturating_sub(period);
        let recent_gains = &gains[start_idx..];
        let recent_losses = &losses[start_idx..];

        let avg_gain: f64 = recent_gains.iter().sum::<f64>() / period as f64;
        let avg_loss: f64 = recent_losses.iter().sum::<f64>() / period as f64;

        if avg_loss == 0.0 {
            return Ok(100.0);  // All gains
        }

        let rs = avg_gain / avg_loss;
        Ok(100.0 - (100.0 / (1.0 + rs)))
    }

    /// Compute MACD (Moving Average Convergence Divergence)
    ///
    /// Returns (MACD line, Signal line, Histogram)
    fn compute_macd(&mut self) -> Result<(f64, f64, f64)> {
        if self.tick_window.is_empty() {
            return Ok((0.0, 0.0, 0.0));
        }

        let current_price = self.tick_window.back().unwrap().price;

        // Update EMAs
        let alpha_12 = 2.0 / (12.0 + 1.0);
        let alpha_26 = 2.0 / (26.0 + 1.0);
        let alpha_9 = 2.0 / (9.0 + 1.0);

        self.ema_12 = Some(match self.ema_12 {
            Some(prev) => alpha_12 * current_price + (1.0 - alpha_12) * prev,
            None => current_price,
        });

        self.ema_26 = Some(match self.ema_26 {
            Some(prev) => alpha_26 * current_price + (1.0 - alpha_26) * prev,
            None => current_price,
        });

        let macd_line = self.ema_12.unwrap() - self.ema_26.unwrap();

        self.macd_signal_ema = Some(match self.macd_signal_ema {
            Some(prev) => alpha_9 * macd_line + (1.0 - alpha_9) * prev,
            None => macd_line,
        });

        let macd_signal = self.macd_signal_ema.unwrap();
        let macd_histogram = macd_line - macd_signal;

        Ok((macd_line, macd_signal, macd_histogram))
    }

    /// Compute Bollinger Bands
    ///
    /// Returns (upper, middle, lower, width)
    fn compute_bollinger_bands(&self, period: usize, num_std: f64) -> Result<(f64, f64, f64, f64)> {
        if self.tick_window.len() < period {
            let current_price = self.tick_window.back().unwrap().price;
            return Ok((current_price, current_price, current_price, 0.0));
        }

        // Use last 'period' prices
        let start_idx = self.tick_window.len().saturating_sub(period);
        let prices: Vec<f64> = self.tick_window.iter()
            .skip(start_idx)
            .map(|t| t.price)
            .collect();

        // Compute SMA (middle band)
        let sma: f64 = prices.iter().sum::<f64>() / prices.len() as f64;

        // Compute standard deviation
        let variance: f64 = prices.iter()
            .map(|p| (p - sma).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std = variance.sqrt();

        let upper = sma + num_std * std;
        let lower = sma - num_std * std;
        let width = (upper - lower) / sma;  // Normalized width

        Ok((upper, sma, lower, width))
    }

    /// Compute bid-ask spread in basis points
    fn compute_spread_bps(&self) -> Result<f64> {
        if self.tick_window.is_empty() {
            return Ok(0.0);
        }

        let current_tick = self.tick_window.back().unwrap();

        if current_tick.bid > 0.0 && current_tick.ask > 0.0 {
            let spread = current_tick.ask - current_tick.bid;
            Ok((spread / current_tick.bid) * 10_000.0)  // Convert to basis points
        } else {
            Ok(0.0)
        }
    }

    /// Compute order flow imbalance
    ///
    /// Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
    fn compute_order_flow_imbalance(&self) -> Result<f64> {
        if self.tick_window.is_empty() {
            return Ok(0.0);
        }

        let current_tick = self.tick_window.back().unwrap();

        let bid_volume = current_tick.bid_size as f64;
        let ask_volume = current_tick.ask_size as f64;
        let total_volume = bid_volume + ask_volume;

        if total_volume > 0.0 {
            Ok((bid_volume - ask_volume) / total_volume)
        } else {
            Ok(0.0)
        }
    }

    /// Compute Volume-Weighted Average Price
    fn compute_vwap(&self) -> Result<f64> {
        if self.tick_window.is_empty() {
            return Ok(0.0);
        }

        let mut total_pv = 0.0;
        let mut total_volume = 0.0;

        for tick in &self.tick_window {
            total_pv += tick.price * tick.volume as f64;
            total_volume += tick.volume as f64;
        }

        if total_volume > 0.0 {
            Ok(total_pv / total_volume)
        } else {
            Ok(self.tick_window.back().unwrap().price)
        }
    }

    /// Compute volume features
    fn compute_volume_features(&self) -> Result<(u64, f64)> {
        if self.tick_window.is_empty() {
            return Ok((0, 0.0));
        }

        let current_volume = self.tick_window.back().unwrap().volume as u64;

        // Compute average volume
        let avg_volume: f64 = self.tick_window.iter()
            .map(|t| t.volume as f64)
            .sum::<f64>() / self.tick_window.len() as f64;

        let volume_momentum = if avg_volume > 0.0 {
            (current_volume as f64 / avg_volume) - 1.0
        } else {
            0.0
        };

        Ok((current_volume, volume_momentum))
    }

    /// Compute price statistics for normalization
    fn compute_price_statistics(&self) -> Result<(f64, f64)> {
        if self.tick_window.is_empty() {
            return Ok((0.0, 1.0));
        }

        let prices: Vec<f64> = self.tick_window.iter().map(|t| t.price).collect();

        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance: f64 = prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std = variance.sqrt().max(1e-10);  // Avoid division by zero

        Ok((mean, std))
    }

    /// Compute volume statistics for normalization
    fn compute_volume_statistics(&self) -> Result<(f64, f64)> {
        if self.tick_window.is_empty() {
            return Ok((0.0, 1.0));
        }

        let volumes: Vec<f64> = self.tick_window.iter()
            .map(|t| t.volume as f64)
            .collect();

        let mean: f64 = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let variance: f64 = volumes.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / volumes.len() as f64;
        let std = variance.sqrt().max(1e-10);

        Ok((mean, std))
    }

    /// Reset the extractor (clear window and cached state)
    pub fn reset(&mut self) {
        self.tick_window.clear();
        self.ema_12 = None;
        self.ema_26 = None;
        self.macd_signal_ema = None;
    }

    /// Get current window size
    pub fn window_size(&self) -> usize {
        self.tick_window.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_data::MarketTick;

    fn create_test_tick(timestamp_ns: u64, price: f64, volume: u32) -> MarketTick {
        MarketTick {
            timestamp_ns,
            symbol: "TEST".to_string(),
            price,
            volume,
            bid: price - 0.01,
            ask: price + 0.01,
            bid_size: volume / 2,
            ask_size: volume / 2,
            exchange: "TEST".to_string(),
            conditions: vec![],
        }
    }

    #[test]
    fn test_feature_extractor_basic() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Not enough data initially
        assert_eq!(extractor.window_size(), 0);

        // Add 50 ticks
        for i in 0..50u64 {
            let tick = create_test_tick(
                i * 1_000_000_000,
                100.0 + i as f64 * 0.1,
                (100 + i) as u32,
            );
            extractor.add_tick(tick);
        }

        assert_eq!(extractor.window_size(), 50);

        // Should be able to extract features now
        let features = extractor.extract_features().unwrap();
        assert!(features.is_some());

        let features = features.unwrap();
        assert_eq!(features.symbol, "TEST");
        assert!(features.price > 100.0);
    }

    #[test]
    fn test_returns_calculation() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add ticks with known price movement
        for i in 0..50 {
            let price = 100.0 * (1.0 + 0.01 * i as f64);  // 1% growth per tick
            let tick = create_test_tick(i * 1_000_000_000, price, 100);
            extractor.add_tick(tick);
        }

        let features = extractor.extract_features().unwrap().unwrap();

        // Should have positive returns
        assert!(features.log_return > 0.0, "Expected positive log return");
        assert!(features.simple_return > 0.0, "Expected positive simple return");
        assert!(features.momentum > 0.0, "Expected positive momentum");
    }

    #[test]
    fn test_volatility_calculation() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add ticks with high volatility (zigzag pattern)
        for i in 0..50 {
            let price = if i % 2 == 0 { 100.0 } else { 110.0 };
            let tick = create_test_tick(i * 1_000_000_000, price, 100);
            extractor.add_tick(tick);
        }

        let features_volatile = extractor.extract_features().unwrap().unwrap();

        // Reset and create low volatility data
        extractor.reset();
        for i in 0..50 {
            let price = 100.0 + 0.01 * i as f64;  // Smooth trend
            let tick = create_test_tick(i * 1_000_000_000, price, 100);
            extractor.add_tick(tick);
        }

        let features_stable = extractor.extract_features().unwrap().unwrap();

        // High volatility should be detected
        assert!(features_volatile.volatility > features_stable.volatility,
                "Zigzag pattern should have higher volatility than smooth trend");
    }

    #[test]
    fn test_rsi_calculation() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add ticks with strong upward movement
        for i in 0..50 {
            let price = 100.0 + i as f64 * 2.0;  // Strong uptrend
            let tick = create_test_tick(i * 1_000_000_000, price, 100);
            extractor.add_tick(tick);
        }

        let features = extractor.extract_features().unwrap().unwrap();

        // RSI should be high (>70 indicates overbought)
        assert!(features.rsi > 70.0, "Strong uptrend should produce high RSI");
        assert!(features.rsi <= 100.0, "RSI should not exceed 100");
    }

    #[test]
    fn test_macd_updates() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add ticks and verify MACD changes over time
        let mut macd_values = Vec::new();

        for i in 0..50 {
            let price = 100.0 + (i as f64 / 10.0).sin() * 5.0;  // Sine wave
            let tick = create_test_tick(i * 1_000_000_000, price, 100);
            extractor.add_tick(tick);

            if let Some(features) = extractor.extract_features().unwrap() {
                macd_values.push(features.macd_histogram);
            }
        }

        // MACD histogram should vary (not constant)
        let unique_values: std::collections::HashSet<String> = macd_values.iter()
            .map(|v| format!("{:.6}", v))
            .collect();

        assert!(unique_values.len() > 5, "MACD should vary with price changes");
    }

    #[test]
    fn test_bollinger_bands() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add 50 ticks
        for i in 0..50 {
            let price = 100.0 + (i as f64 / 5.0).sin() * 2.0;
            let tick = create_test_tick(i * 1_000_000_000, price, 100);
            extractor.add_tick(tick);
        }

        let features = extractor.extract_features().unwrap().unwrap();

        // Upper band should be above middle, which should be above lower
        assert!(features.bb_upper > features.bb_middle);
        assert!(features.bb_middle > features.bb_lower);

        // Width should be positive
        assert!(features.bb_width > 0.0);
    }

    #[test]
    fn test_spread_calculation() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add ticks with known spread
        for i in 0..50 {
            let mut tick = create_test_tick(i * 1_000_000_000, 100.0, 100);
            tick.bid = 100.0;
            tick.ask = 100.10;  // 10 basis points spread
            extractor.add_tick(tick);
        }

        let features = extractor.extract_features().unwrap().unwrap();

        // Should detect ~10 bps spread
        assert!((features.spread_bps - 10.0).abs() < 0.5,
                "Expected spread around 10 bps, got {}", features.spread_bps);
    }

    #[test]
    fn test_order_flow_imbalance() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add tick with bid-heavy order book
        let mut tick = create_test_tick(0, 100.0, 100);
        tick.bid_size = 1000;  // Heavy bid
        tick.ask_size = 500;   // Light ask
        extractor.add_tick(tick);

        // Add more ticks to reach minimum
        for i in 1..50 {
            let mut tick = create_test_tick(i * 1_000_000_000, 100.0, 100);
            tick.bid_size = 1000;
            tick.ask_size = 500;
            extractor.add_tick(tick);
        }

        let features = extractor.extract_features().unwrap().unwrap();

        // Should detect positive imbalance (more bids than asks)
        assert!(features.order_flow_imbalance > 0.0,
                "Expected positive imbalance with more bid volume");
    }

    #[test]
    fn test_vwap_calculation() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add ticks: high volume at 100, low volume at 110
        for i in 0..25 {
            let tick = create_test_tick(i * 1_000_000_000, 100.0, 1000);
            extractor.add_tick(tick);
        }

        for i in 25..50 {
            let tick = create_test_tick(i * 1_000_000_000, 110.0, 100);
            extractor.add_tick(tick);
        }

        let features = extractor.extract_features().unwrap().unwrap();

        // VWAP should be closer to 100 than 110 due to volume weighting
        assert!(features.vwap < 105.0,
                "VWAP should be weighted toward high-volume price");
    }

    #[test]
    fn test_normalization() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 100);

        // Add ticks
        for i in 0..50 {
            let tick = create_test_tick(i * 1_000_000_000, 100.0 + i as f64, 100);
            extractor.add_tick(tick);
        }

        let features = extractor.extract_features().unwrap().unwrap();

        // Test normalization
        let normalized_current = features.normalize_price(features.price);
        assert!(normalized_current >= -1.0 && normalized_current <= 1.0,
                "Normalized price should be in [-1, 1]");

        let normalized_vol = features.normalize_volume(100);
        assert!(normalized_vol >= 0.0 && normalized_vol <= 1.0,
                "Normalized volume should be in [0, 1]");

        // Test feature vector
        let vector = features.to_normalized_vector();
        assert_eq!(vector.len(), 11, "Feature vector should have 11 elements");

        // All features should be in reasonable range
        for &val in &vector {
            assert!(val >= -5.0 && val <= 5.0,
                    "Feature value {} out of reasonable range", val);
        }
    }

    #[test]
    fn test_anti_drift_features_vary() {
        // ARES: Verify features vary with different inputs

        let mut extractor1 = FeatureExtractor::new("TEST".to_string(), 100);
        let mut extractor2 = FeatureExtractor::new("TEST".to_string(), 100);

        // First dataset: uptrend
        for i in 0..50 {
            let tick = create_test_tick(i * 1_000_000_000, 100.0 + i as f64, 100);
            extractor1.add_tick(tick);
        }

        // Second dataset: downtrend
        for i in 0..50 {
            let tick = create_test_tick(i * 1_000_000_000, 150.0 - i as f64, 100);
            extractor2.add_tick(tick);
        }

        let features1 = extractor1.extract_features().unwrap().unwrap();
        let features2 = extractor2.extract_features().unwrap().unwrap();

        // Features should be different
        assert_ne!(features1.log_return, features2.log_return,
                   "Returns should differ between uptrend and downtrend");
        assert_ne!(features1.momentum, features2.momentum,
                   "Momentum should differ");
        assert_ne!(features1.rsi, features2.rsi,
                   "RSI should differ");

        // Momentum should be positive for uptrend, negative for downtrend
        assert!(features1.momentum > 0.0, "Uptrend should have positive momentum");
        assert!(features2.momentum < 0.0, "Downtrend should have negative momentum");
    }

    #[test]
    fn test_window_management() {
        let mut extractor = FeatureExtractor::new("TEST".to_string(), 20);

        // Add more ticks than window size
        for i in 0..50 {
            let tick = create_test_tick(i * 1_000_000_000, 100.0, 100);
            extractor.add_tick(tick);
        }

        // Window should be capped at max size
        assert_eq!(extractor.window_size(), 20);

        // Reset should clear window
        extractor.reset();
        assert_eq!(extractor.window_size(), 0);
    }

    #[test]
    fn test_ignore_wrong_symbol() {
        let mut extractor = FeatureExtractor::new("AAPL".to_string(), 100);

        // Add tick for different symbol
        let mut tick = create_test_tick(0, 100.0, 100);
        tick.symbol = "MSFT".to_string();
        extractor.add_tick(tick);

        // Should not be added
        assert_eq!(extractor.window_size(), 0);

        // Add correct symbol
        let mut tick = create_test_tick(0, 100.0, 100);
        tick.symbol = "AAPL".to_string();
        extractor.add_tick(tick);

        assert_eq!(extractor.window_size(), 1);
    }
}
