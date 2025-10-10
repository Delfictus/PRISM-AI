# HFT Demo - Task 1.2: Market Simulator - Detailed Implementation Plan

**Task Duration:** 4 hours
**Priority:** HIGH
**Status:** 🔵 Starting
**File:** `hft-demo/src/market_data/simulator.rs`

---

## 🎯 Task 1.2 Overview

Build a market simulator that can replay historical data or generate synthetic tick data with configurable parameters. Must be fully ARES anti-drift compliant.

### Goals
1. Historical replay with speed multipliers (1x, 10x, 100x, 1000x)
2. Synthetic data generation using Geometric Brownian Motion
3. Realistic bid-ask spread dynamics
4. Volume simulation using stochastic processes
5. Event-driven architecture for backtesting

### ARES Compliance Requirements
- ❌ NO hardcoded market behaviors
- ❌ NO hardcoded price movements
- ❌ NO hardcoded volume patterns
- ✅ ALL behaviors computed from parameters
- ✅ Different inputs → different outputs
- ✅ Randomness properly seeded and varied

---

## 📋 Subtask Breakdown

### Subtask 1.2.1: Define Simulation Modes (30 min)
**Priority:** BLOCKER
**Estimated Time:** 30 minutes

**Requirements:**
- [ ] Define `SimulationMode` enum with three variants
- [ ] Historical: Replay loaded data with speed control
- [ ] Synthetic: Generate realistic data on-the-fly
- [ ] Hybrid: Mix historical base with synthetic noise
- [ ] Define `MarketSimulator` struct
- [ ] Add configuration parameters
- [ ] Define `MicrostructureModel` enum
- [ ] Add RNG state management

**Data Structures:**
```rust
#[derive(Debug, Clone)]
pub enum SimulationMode {
    Historical {
        speed_multiplier: f32,  // 1.0 = real-time, 1000.0 = fast
    },
    Synthetic {
        base_price: f64,
        volatility: f32,        // Annual volatility (e.g., 0.3 for 30%)
        trend: f32,             // Annual drift (e.g., 0.1 for 10% per year)
        tick_interval_ms: u64,  // Milliseconds between ticks
    },
    Hybrid {
        noise_level: f32,       // How much noise to add (0.0-1.0)
        preserve_trends: bool,  // Keep original trend direction
    },
}

pub struct MarketSimulator {
    mode: SimulationMode,
    symbol: String,
    current_tick: usize,
    tick_data: Vec<MarketTick>,
    rng: rand::rngs::StdRng,  // Use StdRng for reproducibility
    start_time: std::time::Instant,
}
```

**Success Criteria:**
- Compiles without errors
- All enums have proper Debug, Clone derives
- RNG properly initialized with seed

---

### Subtask 1.2.2: Historical Replay (45 min)
**Priority:** HIGH
**Estimated Time:** 45 minutes

**Requirements:**
- [ ] Implement `MarketSimulator::new_historical()`
- [ ] Load tick data from CSV via existing loader
- [ ] Implement `next_tick()` async method
- [ ] Support speed multiplier (1x, 10x, 100x, 1000x)
- [ ] Calculate real-time delays based on timestamps
- [ ] Handle end-of-data gracefully (return None)
- [ ] Add `reset()` method to restart simulation
- [ ] Track simulation progress (current tick / total ticks)

**Implementation:**
```rust
impl MarketSimulator {
    pub fn new_historical(
        tick_data: Vec<MarketTick>,
        symbol: String,
        speed_multiplier: f32,
    ) -> Result<Self> {
        if tick_data.is_empty() {
            bail!("Cannot create simulator with empty tick data");
        }

        Ok(Self {
            mode: SimulationMode::Historical { speed_multiplier },
            symbol,
            current_tick: 0,
            tick_data,
            rng: rand::rngs::StdRng::from_entropy(),
            start_time: std::time::Instant::now(),
        })
    }

    /// Get next market tick (async for proper delay handling)
    pub async fn next_tick(&mut self) -> Result<Option<MarketTick>> {
        if self.current_tick >= self.tick_data.len() {
            return Ok(None);
        }

        let tick = self.tick_data[self.current_tick].clone();
        self.current_tick += 1;

        // Calculate delay for next tick
        if let SimulationMode::Historical { speed_multiplier } = self.mode {
            if self.current_tick < self.tick_data.len() {
                let next_tick = &self.tick_data[self.current_tick];
                let time_diff_ns = next_tick.timestamp_ns - tick.timestamp_ns;
                let delay_ns = (time_diff_ns as f32 / speed_multiplier) as u64;

                if delay_ns > 0 && delay_ns < 10_000_000_000 {  // Cap at 10 seconds
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
        self.start_time = std::time::Instant::now();
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
}
```

**Success Criteria:**
- Can load sample data and replay at 1x speed
- Can replay at 1000x speed (fast)
- progress() returns correct value
- reset() allows replaying from start
- No panics on end-of-data

---

### Subtask 1.2.3: Synthetic Data Generation (1.5 hours)
**Priority:** HIGH
**Estimated Time:** 90 minutes

**Requirements:**
- [ ] Implement `MarketSimulator::new_synthetic()`
- [ ] Use Geometric Brownian Motion for price evolution
- [ ] Add realistic bid-ask spread dynamics (1-10 bps)
- [ ] Simulate volume using Poisson distribution
- [ ] Add mean reversion for intraday behavior
- [ ] Support configurable volatility and trend
- [ ] Generate ticks on-demand (lazy generation)
- [ ] Add microstructure effects (clustering, momentum)
- [ ] NO HARDCODED BEHAVIORS (ARES CRITICAL!)

**Implementation:**
```rust
impl MarketSimulator {
    pub fn new_synthetic(
        symbol: String,
        base_price: f64,
        volatility: f32,
        trend: f32,
        duration_seconds: u64,
        tick_interval_ms: u64,
    ) -> Result<Self> {
        if base_price <= 0.0 {
            bail!("Base price must be positive");
        }
        if volatility < 0.0 || volatility > 2.0 {
            bail!("Volatility must be between 0 and 2");
        }

        let num_ticks = (duration_seconds * 1000 / tick_interval_ms) as usize;
        let mut tick_data = Vec::with_capacity(num_ticks);
        let mut rng = rand::rngs::StdRng::from_entropy();

        // Geometric Brownian Motion parameters
        // dS = μS dt + σS dW
        let dt = tick_interval_ms as f32 / 1000.0;  // Convert to seconds
        let mu = trend / (252.0 * 6.5 * 3600.0);     // Annual → per-second
        let sigma = volatility / (252.0 * 6.5 * 3600.0_f32).sqrt();

        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let poisson = rand_distr::Poisson::new(100.0).unwrap();
        let mut price = base_price;

        // Base timestamp
        let base_timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap() as u64;

        for i in 0..num_ticks {
            // Geometric Brownian Motion
            let dw = normal.sample(&mut rng) as f32;
            let price_change = mu * price as f32 * dt
                             + sigma * price as f32 * dw * dt.sqrt();
            price = (price + price_change as f64).max(1.0);

            // Add mean reversion (prices revert to base intraday)
            let mean_reversion_strength = 0.0001;
            let reversion = mean_reversion_strength * (base_price - price);
            price += reversion;

            // Realistic bid-ask spread (1-10 basis points)
            let spread_bps = rng.gen_range(1.0..10.0);
            let spread = price * spread_bps / 10000.0;
            let bid = price - spread / 2.0;
            let ask = price + spread / 2.0;

            // Volume follows Poisson distribution
            let volume = poisson.sample(&mut rng).max(1) as u32;

            // Bid/ask sizes
            let bid_size = rng.gen_range(10..200);
            let ask_size = rng.gen_range(10..200);

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
            start_time: std::time::Instant::now(),
        })
    }
}
```

**Success Criteria:**
- Generates realistic price paths
- Volatility parameter affects price variance
- Trend parameter affects drift direction
- Different runs produce different paths (randomness)
- No hardcoded price values
- Bid < Ask always holds

---

### Subtask 1.2.4: Anti-Drift Validation & Tests (30 min)
**Priority:** CRITICAL
**Estimated Time:** 30 minutes

**Requirements:**
- [ ] Test historical replay works correctly
- [ ] Test speed multiplier affects timing
- [ ] Test synthetic generation with different parameters
- [ ] Verify high volatility → high variance
- [ ] Verify positive trend → price increase
- [ ] Verify different runs → different outputs
- [ ] Test reset() functionality
- [ ] Test progress() tracking
- [ ] ANTI-DRIFT: No hardcoded values

**Test Template:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_historical_replay() {
        let loader = CsvDataLoader::new(
            "data/sample_aapl_1hour.csv".to_string(),
            "AAPL".to_string(),
        );
        let ticks = loader.load_all().unwrap();

        let mut sim = MarketSimulator::new_historical(
            ticks.clone(),
            "AAPL".to_string(),
            1000.0,  // Fast replay
        ).unwrap();

        let mut count = 0;
        while let Some(_tick) = sim.next_tick().await.unwrap() {
            count += 1;
        }

        assert_eq!(count, ticks.len());
        assert!(sim.is_complete());
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
        assert!(variance_high > variance_low * 2.0,
                "High vol variance ({}) should be > 2x low vol ({})",
                variance_high, variance_low);
    }

    #[test]
    fn test_synthetic_trend_affects_drift() {
        // ARES ANTI-DRIFT: Trend parameter should affect price drift
        let sim_up = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.2,
            0.2,  // 20% annual uptrend
            3600,
            1000,
        ).unwrap();

        let sim_down = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.2,
            -0.2,  // 20% annual downtrend
            3600,
            1000,
        ).unwrap();

        let final_price_up = sim_up.tick_data.last().unwrap().price;
        let final_price_down = sim_down.tick_data.last().unwrap().price;

        // Uptrend should end higher than downtrend
        assert!(final_price_up > final_price_down);
    }

    #[test]
    fn test_no_hardcoded_prices() {
        // ARES ANTI-DRIFT: Different runs should produce different paths
        let sim1 = MarketSimulator::new_synthetic(
            "TEST".to_string(), 100.0, 0.3, 0.0, 100, 1000
        ).unwrap();

        let sim2 = MarketSimulator::new_synthetic(
            "TEST".to_string(), 100.0, 0.3, 0.0, 100, 1000
        ).unwrap();

        // Should generate different price paths (stochastic)
        let mut differences = 0;
        for i in 0..sim1.tick_data.len().min(sim2.tick_data.len()) {
            if (sim1.tick_data[i].price - sim2.tick_data[i].price).abs() > 0.01 {
                differences += 1;
            }
        }

        assert!(differences > 50, "Should have many different prices (got {})", differences);
    }

    #[test]
    fn test_reset_functionality() {
        let sim = MarketSimulator::new_synthetic(
            "TEST".to_string(), 100.0, 0.2, 0.0, 100, 1000
        ).unwrap();

        let mut sim = sim;
        sim.current_tick = 50;
        assert_eq!(sim.progress(), 0.5);

        sim.reset();
        assert_eq!(sim.current_tick, 0);
        assert_eq!(sim.progress(), 0.0);
    }

    fn calculate_price_variance(ticks: &[MarketTick]) -> f64 {
        let prices: Vec<f64> = ticks.iter().map(|t| t.price).collect();
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / prices.len() as f64
    }
}
```

**Success Criteria:**
- All tests pass
- Anti-drift tests verify variability
- Different parameters produce measurably different results
- No hardcoded values in any computation

---

## 📊 Progress Tracking

| Subtask | Duration | Status | Notes |
|---------|----------|--------|-------|
| 1.2.1 Simulation Modes | 30 min | 🔵 Pending | Enums and structs |
| 1.2.2 Historical Replay | 45 min | 🔵 Pending | Speed multipliers |
| 1.2.3 Synthetic Generation | 90 min | 🔵 Pending | GBM, ARES critical |
| 1.2.4 Anti-Drift Tests | 30 min | 🔵 Pending | Validation |
| **Total** | **4 hours** | **0%** | |

---

## ✅ Success Criteria

Task 1.2 is complete when:
1. ✅ Can replay historical data at 1x, 10x, 100x, 1000x speed
2. ✅ Can generate synthetic data with configurable parameters
3. ✅ Volatility parameter demonstrably affects price variance
4. ✅ Trend parameter demonstrably affects price drift
5. ✅ Different runs produce different stochastic paths
6. ✅ All tests passing (including anti-drift tests)
7. ✅ No hardcoded values in any computation
8. ✅ Can reset and replay simulation

**Performance Targets:**
- Generate 3,600 synthetic ticks: <100ms
- Replay 3,600 ticks at 1000x: <5 seconds
- Memory usage: <10MB for 10,000 ticks

---

## 🚀 Implementation Order

1. **Start with Subtask 1.2.1** - Define all data structures
2. **Then Subtask 1.2.2** - Implement historical replay (easier)
3. **Then Subtask 1.2.3** - Implement synthetic generation (complex)
4. **Finally Subtask 1.2.4** - Add comprehensive tests

---

*Document created: 2025-10-10*
*Status: Ready to begin implementation*
*Next: Create simulator.rs file*
