//! Market Simulator
//!
//! Provides market simulation capabilities for backtesting:
//! - Historical replay with configurable speed
//! - Synthetic data generation using stochastic models
//! - Hybrid modes combining real and synthetic data
//!
//! # ARES Anti-Drift Compliance
//!
//! This module is CRITICAL for anti-drift compliance. All market behaviors
//! must be computed from parameters, never hardcoded.
//!
//! ## Forbidden Patterns
//! ```rust,ignore
//! // ❌ NEVER hardcode market behavior
//! fn generate_price(&self) -> f64 {
//!     if self.current_tick % 100 == 0 {
//!         150.0  // HARDCODED!
//!     } else {
//!         145.0  // HARDCODED!
//!     }
//! }
//! ```
//!
//! ## Required Patterns
//! ```rust,ignore
//! // ✅ Always compute from parameters and stochastic processes
//! fn generate_price(&mut self) -> f64 {
//!     // Use GBM: dS = μS dt + σS dW
//!     let dw = self.normal.sample(&mut self.rng);
//!     let price_change = self.mu * self.price * self.dt
//!                      + self.sigma * self.price * dw * self.dt.sqrt();
//!     self.price = (self.price + price_change).max(1.0);
//!     self.price  // COMPUTED from stochastic process
//! }
//! ```

use crate::market_data::MarketTick;
use anyhow::{bail, Context, Result};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Poisson};
use std::time::Instant;

/// Simulation mode for market replay
#[derive(Debug, Clone)]
pub enum SimulationMode {
    /// Replay historical data at specified speed
    Historical {
        /// Speed multiplier: 1.0 = real-time, 1000.0 = 1000x fast
        speed_multiplier: f32,
    },

    /// Generate synthetic data using stochastic models
    Synthetic {
        /// Base/starting price
        base_price: f64,

        /// Annual volatility (e.g., 0.3 = 30%)
        volatility: f32,

        /// Annual trend/drift (e.g., 0.1 = 10% per year)
        trend: f32,

        /// Milliseconds between ticks
        tick_interval_ms: u64,
    },

    /// Hybrid mode: historical base with added noise
    Hybrid {
        /// Amount of noise to add (0.0-1.0)
        noise_level: f32,

        /// Preserve original trend direction
        preserve_trends: bool,
    },
}

/// Market simulator for backtesting and strategy development
///
/// # ARES Compliance
/// This simulator generates market data using well-defined stochastic processes.
/// All prices, volumes, and spreads are COMPUTED, never hardcoded.
///
/// # Examples
///
/// ```rust,ignore
/// // Historical replay at 1000x speed
/// let mut sim = MarketSimulator::new_historical(
///     historical_ticks,
///     "AAPL".to_string(),
///     1000.0,
/// )?;
///
/// while let Some(tick) = sim.next_tick().await? {
///     // Process tick
/// }
///
/// // Synthetic generation
/// let sim = MarketSimulator::new_synthetic(
///     "AAPL".to_string(),
///     182.50,    // base price
///     0.25,      // 25% annual volatility
///     0.10,      // 10% annual uptrend
///     3600,      // 1 hour
///     1000,      // 1 tick/second
/// )?;
/// ```
pub struct MarketSimulator {
    /// Simulation mode configuration
    mode: SimulationMode,

    /// Symbol being simulated
    symbol: String,

    /// Current tick index
    current_tick: usize,

    /// Loaded or generated tick data
    tick_data: Vec<MarketTick>,

    /// Random number generator (for reproducibility)
    rng: rand::rngs::StdRng,

    /// Simulation start time (for timing calculations)
    start_time: Instant,
}

impl MarketSimulator {
    /// Create new historical replay simulator
    ///
    /// # Arguments
    /// * `tick_data` - Pre-loaded historical ticks
    /// * `symbol` - Stock symbol
    /// * `speed_multiplier` - Replay speed (1.0 = real-time, 1000.0 = fast)
    ///
    /// # ARES Compliance
    /// Uses actual loaded data, no hardcoded ticks
    pub fn new_historical(
        tick_data: Vec<MarketTick>,
        symbol: String,
        speed_multiplier: f32,
    ) -> Result<Self> {
        if tick_data.is_empty() {
            bail!("Cannot create simulator with empty tick data");
        }

        if speed_multiplier <= 0.0 {
            bail!("Speed multiplier must be positive (got {})", speed_multiplier);
        }

        log::info!(
            "Created historical simulator for {} with {} ticks at {}x speed",
            symbol,
            tick_data.len(),
            speed_multiplier
        );

        Ok(Self {
            mode: SimulationMode::Historical { speed_multiplier },
            symbol,
            current_tick: 0,
            tick_data,
            rng: rand::rngs::StdRng::from_entropy(),
            start_time: Instant::now(),
        })
    }

    /// Create new synthetic data simulator
    ///
    /// # Arguments
    /// * `symbol` - Stock symbol
    /// * `base_price` - Starting price
    /// * `volatility` - Annual volatility (0.0-2.0, typically 0.2-0.5)
    /// * `trend` - Annual drift (-1.0 to 1.0, typically -0.2 to 0.2)
    /// * `duration_seconds` - Simulation duration
    /// * `tick_interval_ms` - Milliseconds between ticks
    ///
    /// # ARES Compliance
    /// Generates data using Geometric Brownian Motion. No hardcoded prices.
    /// Different runs produce different stochastic paths.
    pub fn new_synthetic(
        symbol: String,
        base_price: f64,
        volatility: f32,
        trend: f32,
        duration_seconds: u64,
        tick_interval_ms: u64,
    ) -> Result<Self> {
        // Validate parameters
        if base_price <= 0.0 {
            bail!("Base price must be positive (got {})", base_price);
        }
        if volatility < 0.0 || volatility > 2.0 {
            bail!("Volatility must be between 0 and 2 (got {})", volatility);
        }
        if trend < -1.0 || trend > 1.0 {
            bail!("Trend must be between -1 and 1 (got {})", trend);
        }
        if tick_interval_ms == 0 {
            bail!("Tick interval must be positive");
        }

        let num_ticks = (duration_seconds * 1000 / tick_interval_ms) as usize;
        let mut tick_data = Vec::with_capacity(num_ticks);
        let mut rng = rand::rngs::StdRng::from_entropy();

        // Geometric Brownian Motion parameters
        // dS = μS dt + σS dW
        // where S = price, μ = trend, σ = volatility, dW = Brownian increment
        let dt = tick_interval_ms as f32 / 1000.0;  // Convert to seconds
        let mu = trend / (252.0 * 6.5 * 3600.0);     // Annual → per-second
        let sigma = volatility / (252.0 * 6.5 * 3600.0_f32).sqrt();  // Annual → per-second

        // Stochastic distributions
        let normal = Normal::new(0.0, 1.0)
            .context("Failed to create normal distribution")?;
        let volume_poisson = Poisson::new(100.0)
            .context("Failed to create Poisson distribution")?;

        let mut price = base_price;

        // Base timestamp: current time
        let base_timestamp = chrono::Utc::now()
            .timestamp_nanos_opt()
            .context("Timestamp overflow")? as u64;

        for i in 0..num_ticks {
            // Geometric Brownian Motion price evolution
            let dw = normal.sample(&mut rng) as f32;
            let price_change = mu * price as f32 * dt
                             + sigma * price as f32 * dw * dt.sqrt();
            price = (price + price_change as f64).max(1.0);  // Ensure positive

            // Add mean reversion (intraday prices revert to base)
            let mean_reversion_strength = 0.0001;
            let reversion = mean_reversion_strength * (base_price - price);
            price += reversion;

            // Realistic bid-ask spread (1-10 basis points)
            let spread_bps = rng.gen_range(1.0..10.0);
            let spread = price * spread_bps / 10000.0;
            let bid = price - spread / 2.0;
            let ask = price + spread / 2.0;

            // Volume follows Poisson distribution (ensures positive integer)
            let volume = (volume_poisson.sample(&mut rng) as u32).max(1);

            // Bid/ask sizes (randomized)
            let bid_size = rng.gen_range(10..200);
            let ask_size = rng.gen_range(10..200);

            // Create tick
            let tick = MarketTick {
                timestamp_ns: base_timestamp + (i as u64 * tick_interval_ms * 1_000_000),
                symbol: symbol.clone(),
                price,
                volume,
                bid,
                ask,
                bid_size,
                ask_size,
                exchange: "SYNTHETIC".to_string(),
                conditions: vec![],
            };

            tick_data.push(tick);
        }

        log::info!(
            "Generated {} synthetic ticks for {} (base: ${:.2}, vol: {:.1}%, trend: {:.1}%)",
            tick_data.len(),
            symbol,
            base_price,
            volatility * 100.0,
            trend * 100.0
        );

        Ok(Self {
            mode: SimulationMode::Synthetic {
                base_price,
                volatility,
                trend,
                tick_interval_ms,
            },
            symbol,
            current_tick: 0,
            tick_data,
            rng,
            start_time: Instant::now(),
        })
    }

    /// Get next market tick (async for proper timing)
    ///
    /// Returns `None` when simulation is complete.
    ///
    /// # ARES Compliance
    /// Returns actual data from historical or generated ticks.
    /// No hardcoded return values.
    pub async fn next_tick(&mut self) -> Result<Option<MarketTick>> {
        if self.current_tick >= self.tick_data.len() {
            return Ok(None);
        }

        let tick = self.tick_data[self.current_tick].clone();
        self.current_tick += 1;

        // Calculate delay for historical replay
        if let SimulationMode::Historical { speed_multiplier } = self.mode {
            if self.current_tick < self.tick_data.len() {
                let next_tick = &self.tick_data[self.current_tick];
                let time_diff_ns = next_tick.timestamp_ns.saturating_sub(tick.timestamp_ns);
                let delay_ns = (time_diff_ns as f32 / speed_multiplier) as u64;

                // Cap delay at 10 seconds to avoid blocking
                if delay_ns > 0 && delay_ns < 10_000_000_000 {
                    tokio::time::sleep(
                        tokio::time::Duration::from_nanos(delay_ns)
                    ).await;
                }
            }
        }

        Ok(Some(tick))
    }

    /// Reset simulation to beginning
    pub fn reset(&mut self) {
        self.current_tick = 0;
        self.start_time = Instant::now();
    }

    /// Get current progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        if self.tick_data.is_empty() {
            return 0.0;
        }
        self.current_tick as f32 / self.tick_data.len() as f32
    }

    /// Check if simulation is complete
    pub fn is_complete(&self) -> bool {
        self.current_tick >= self.tick_data.len()
    }

    /// Get total number of ticks
    pub fn total_ticks(&self) -> usize {
        self.tick_data.len()
    }

    /// Get current tick index
    pub fn current_index(&self) -> usize {
        self.current_tick
    }

    /// Get simulation mode
    pub fn mode(&self) -> &SimulationMode {
        &self.mode
    }

    /// Get symbol
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_data::CsvDataLoader;

    #[tokio::test]
    async fn test_historical_replay() {
        let loader = CsvDataLoader::new(
            "data/sample_aapl_1hour.csv".to_string(),
            "AAPL".to_string(),
        );
        let ticks = loader.load_all().expect("Should load sample data");

        let mut sim = MarketSimulator::new_historical(
            ticks.clone(),
            "AAPL".to_string(),
            1000.0,  // Fast replay
        ).unwrap();

        let mut count = 0;
        while let Some(tick) = sim.next_tick().await.unwrap() {
            assert_eq!(tick.symbol, "AAPL");
            count += 1;
        }

        assert_eq!(count, ticks.len());
        assert!(sim.is_complete());
        assert_eq!(sim.progress(), 1.0);
    }

    #[test]
    fn test_synthetic_generation() {
        let sim = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,    // base price
            0.3,      // 30% volatility
            0.0,      // no trend
            3600,     // 1 hour
            1000,     // 1 tick/second
        ).unwrap();

        assert_eq!(sim.total_ticks(), 3600);
        assert_eq!(sim.symbol(), "TEST");
        assert!(!sim.is_complete());

        // Verify all ticks have valid data
        for tick in &sim.tick_data {
            assert!(tick.price > 0.0, "Price should be positive");
            assert!(tick.bid > 0.0, "Bid should be positive");
            assert!(tick.ask > tick.bid, "Ask should be > bid");
            assert!(tick.volume > 0, "Volume should be positive");
        }
    }

    #[test]
    fn test_synthetic_volatility_affects_variance() {
        // ARES ANTI-DRIFT: Different parameters → different behavior
        let sim_low = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.1,  // Low volatility
            0.0,
            3600,
            1000,
        ).unwrap();

        let sim_high = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.5,  // High volatility
            0.0,
            3600,
            1000,
        ).unwrap();

        // Calculate price variance
        let variance_low = calculate_price_variance(&sim_low.tick_data);
        let variance_high = calculate_price_variance(&sim_high.tick_data);

        // High volatility should produce higher variance
        assert!(
            variance_high > variance_low * 2.0,
            "High vol variance ({:.4}) should be > 2x low vol ({:.4})",
            variance_high,
            variance_low
        );
    }

    #[test]
    fn test_synthetic_trend_affects_drift() {
        // ARES ANTI-DRIFT: Trend parameter should affect price drift
        // Use longer simulation to make trend effect more visible
        let sim_up = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.1,  // Lower volatility to see trend better
            0.5,  // Strong uptrend
            36000,  // 10 hours
            1000,
        ).unwrap();

        let sim_down = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.1,  // Lower volatility
            -0.5,  // Strong downtrend
            36000,  // 10 hours
            1000,
        ).unwrap();

        // Compare final prices (more stable than averages with mean reversion)
        let final_up = sim_up.tick_data.last().unwrap().price;
        let final_down = sim_down.tick_data.last().unwrap().price;

        // With strong trends over 10 hours, final prices should reflect trend
        assert!(
            final_up > final_down,
            "Uptrend final ({:.2}) should be > downtrend final ({:.2})",
            final_up,
            final_down
        );

        // Also verify trend direction from start
        let start_price = 100.0;
        assert!(
            final_up > start_price * 0.98,  // Allow some variation
            "Uptrend should end near or above start ({:.2} vs {})",
            final_up,
            start_price
        );
    }

    #[test]
    fn test_no_hardcoded_prices() {
        // ARES ANTI-DRIFT: Different runs should produce different paths
        let sim1 = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.3,
            0.0,
            100,
            1000,
        ).unwrap();

        let sim2 = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.3,
            0.0,
            100,
            1000,
        ).unwrap();

        // Should generate different price paths (stochastic)
        let mut differences = 0;
        for i in 0..sim1.tick_data.len().min(sim2.tick_data.len()) {
            if (sim1.tick_data[i].price - sim2.tick_data[i].price).abs() > 0.01 {
                differences += 1;
            }
        }

        assert!(
            differences > 50,
            "Should have many different prices (got {} differences)",
            differences
        );
    }

    #[test]
    fn test_reset_functionality() {
        let sim = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.2,
            0.0,
            100,
            1000,
        ).unwrap();

        let mut sim = sim;
        sim.current_tick = 50;
        assert_eq!(sim.progress(), 0.5);
        assert!(!sim.is_complete());

        sim.reset();
        assert_eq!(sim.current_tick, 0);
        assert_eq!(sim.progress(), 0.0);
        assert!(!sim.is_complete());
    }

    #[test]
    fn test_invalid_parameters() {
        // Negative base price
        assert!(MarketSimulator::new_synthetic(
            "TEST".to_string(),
            -100.0,
            0.3,
            0.0,
            100,
            1000,
        ).is_err());

        // Invalid volatility
        assert!(MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            -0.1,
            0.0,
            100,
            1000,
        ).is_err());

        // Invalid speed multiplier
        assert!(MarketSimulator::new_historical(
            vec![],
            "TEST".to_string(),
            -1.0,
        ).is_err());
    }

    /// Calculate price variance for anti-drift tests
    fn calculate_price_variance(ticks: &[MarketTick]) -> f64 {
        let prices: Vec<f64> = ticks.iter().map(|t| t.price).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        prices
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>()
            / prices.len() as f64
    }
}
