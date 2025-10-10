# HFT Demo - Phase 1: Market Data Engine - Detailed Task Breakdown

**Phase Duration:** Day 1-2 (12 hours total)
**Goal:** Build robust market data pipeline with anti-drift compliance
**Output:** Functional data loader, simulator, feature extractor, and validator

---

## 🎯 Phase 1 Overview

### Files to Create
```
hft-demo/
├── Cargo.toml                                    # NEW
├── src/
│   ├── lib.rs                                    # NEW
│   └── market_data/
│       ├── mod.rs                                # NEW
│       ├── loader.rs                             # Task 1.1
│       ├── simulator.rs                          # Task 1.2
│       ├── features.rs                           # Task 1.3
│       └── validation.rs                         # Task 1.4
├── data/
│   └── sample_aapl_1hour.csv                     # NEW
└── tests/
    └── market_data_tests.rs                      # NEW
```

### Dependencies to Add
```toml
[dependencies]
prism-ai = { path = ".." }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
csv = "1.3"
polars = { version = "0.35", features = ["lazy", "parquet"] }
ndarray = "0.15"
statrs = "0.16"
ta = "0.5"  # Technical analysis library
anyhow = "1.0"
thiserror = "1.0"
tokio = { version = "1", features = ["full"] }
async-trait = "0.1"
rand = "0.8"
rand_distr = "0.4"
log = "0.4"
```

---

## 📝 Task 1.1: Historical Data Loader (3 hours)

**File:** `hft-demo/src/market_data/loader.rs`
**Estimated Time:** 3 hours
**ARES Compliance:** CRITICAL - No hardcoded data returns

### Subtask 1.1.1: Project Setup (20 min)
**Priority:** BLOCKER

- [ ] Create `hft-demo/` directory in project root
- [ ] Initialize new Cargo workspace member
- [ ] Create `hft-demo/Cargo.toml` with all dependencies
- [ ] Create `src/lib.rs` with module structure
- [ ] Create `src/market_data/mod.rs` with public exports
- [ ] Add `hft-demo` to workspace members in root `Cargo.toml`
- [ ] Verify compilation: `cargo build -p hft-demo`

**Success Criteria:**
```bash
cd hft-demo
cargo check
# Should compile with no errors
```

---

### Subtask 1.1.2: Define Core Data Structures (30 min)
**Priority:** HIGH
**File:** `hft-demo/src/market_data/loader.rs`

**Requirements:**
- [ ] Define `MarketTick` struct with all fields
- [ ] Implement `Debug`, `Clone`, `Serialize`, `Deserialize` for `MarketTick`
- [ ] Add nanosecond-precision timestamp (u64)
- [ ] Include symbol, price, volume, bid, ask, sizes
- [ ] Add exchange code and trade conditions
- [ ] Define `OrderBookSnapshot` struct
- [ ] Add bid/ask depth vectors (price, size pairs)
- [ ] Calculate spread_bps and imbalance in constructor
- [ ] Define `DataSource` trait for multiple providers

**Code Template:**
```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    /// Nanosecond precision timestamp
    pub timestamp_ns: u64,

    /// Stock symbol
    pub symbol: String,

    /// Trade price
    pub price: f64,

    /// Share volume
    pub volume: u32,

    /// Best bid price
    pub bid: f64,

    /// Best ask price
    pub ask: f64,

    /// Bid size
    pub bid_size: u32,

    /// Ask size
    pub ask_size: u32,

    /// Exchange identifier
    pub exchange: String,

    /// Trade conditions (e.g., ["@", "F"])
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub timestamp_ns: u64,

    /// Bid levels: (price, size)
    pub bids: Vec<(f64, u32)>,

    /// Ask levels: (price, size)
    pub asks: Vec<(f64, u32)>,

    /// Spread in basis points (COMPUTED!)
    pub spread_bps: f32,

    /// Order imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol) (COMPUTED!)
    pub imbalance: f32,
}

impl OrderBookSnapshot {
    /// Create with COMPUTED metrics (ANTI-DRIFT COMPLIANT)
    pub fn new(timestamp_ns: u64, bids: Vec<(f64, u32)>, asks: Vec<(f64, u32)>) -> Self {
        let spread_bps = if !bids.is_empty() && !asks.is_empty() {
            let best_bid = bids[0].0;
            let best_ask = asks[0].0;
            ((best_ask - best_bid) / best_bid * 10_000.0) as f32
        } else {
            0.0
        };

        let bid_volume: u32 = bids.iter().map(|(_, size)| size).sum();
        let ask_volume: u32 = asks.iter().map(|(_, size)| size).sum();

        let imbalance = if bid_volume + ask_volume > 0 {
            (bid_volume as f32 - ask_volume as f32) / (bid_volume + ask_volume) as f32
        } else {
            0.0
        };

        Self {
            timestamp_ns,
            bids,
            asks,
            spread_bps,
            imbalance,
        }
    }
}
```

**Anti-Drift Validation:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_spread_varies() {
        // ARES STANDARD: Verify computed values vary with input
        let snapshot1 = OrderBookSnapshot::new(
            0,
            vec![(100.0, 10), (99.9, 5)],
            vec![(100.1, 10), (100.2, 5)]
        );

        let snapshot2 = OrderBookSnapshot::new(
            0,
            vec![(100.0, 10), (99.9, 5)],
            vec![(100.5, 10), (100.6, 5)]
        );

        // Different spreads for different order books
        assert_ne!(snapshot1.spread_bps, snapshot2.spread_bps);

        // Verify spread is computed correctly
        assert!(snapshot1.spread_bps > 0.0);
        assert!(snapshot2.spread_bps > snapshot1.spread_bps);
    }
}
```

---

### Subtask 1.1.3: CSV Data Loader (40 min)
**Priority:** HIGH
**File:** `hft-demo/src/market_data/loader.rs`

**Requirements:**
- [ ] Implement `CsvDataLoader` struct
- [ ] Add file path and symbol fields
- [ ] Parse CSV with `csv` crate
- [ ] Handle timestamp parsing (RFC3339 or Unix timestamp)
- [ ] Convert to nanosecond precision
- [ ] Validate data quality during loading
- [ ] Handle missing fields gracefully
- [ ] Support streaming large files (iterator)
- [ ] Add progress reporting for large files

**Code Template:**
```rust
use csv::Reader;
use std::fs::File;
use std::path::Path;
use anyhow::{Context, Result};

pub struct CsvDataLoader {
    file_path: String,
    symbol: String,
}

impl CsvDataLoader {
    pub fn new(file_path: String, symbol: String) -> Self {
        Self { file_path, symbol }
    }

    /// Load all ticks from CSV file
    pub fn load_all(&self) -> Result<Vec<MarketTick>> {
        let file = File::open(&self.file_path)
            .context("Failed to open CSV file")?;

        let mut reader = Reader::from_reader(file);
        let mut ticks = Vec::new();

        for result in reader.deserialize() {
            let record: CsvRecord = result.context("Failed to parse CSV row")?;
            let tick = self.parse_record(record)?;
            ticks.push(tick);
        }

        log::info!("Loaded {} ticks from {}", ticks.len(), self.file_path);

        Ok(ticks)
    }

    /// Parse CSV record into MarketTick
    fn parse_record(&self, record: CsvRecord) -> Result<MarketTick> {
        // Convert timestamp string to nanoseconds
        let timestamp_ns = if record.timestamp.contains('T') {
            // RFC3339 format
            let dt = DateTime::parse_from_rfc3339(&record.timestamp)
                .context("Invalid RFC3339 timestamp")?;
            dt.timestamp_nanos_opt()
                .context("Timestamp out of range")? as u64
        } else {
            // Unix timestamp in milliseconds
            let ts_ms: i64 = record.timestamp.parse()
                .context("Invalid Unix timestamp")?;
            (ts_ms * 1_000_000) as u64
        };

        Ok(MarketTick {
            timestamp_ns,
            symbol: self.symbol.clone(),
            price: record.price,
            volume: record.volume,
            bid: record.bid,
            ask: record.ask,
            bid_size: record.bid_size.unwrap_or(0),
            ask_size: record.ask_size.unwrap_or(0),
            exchange: record.exchange.unwrap_or_default(),
            conditions: record.conditions.unwrap_or_default(),
        })
    }
}

#[derive(Debug, Deserialize)]
struct CsvRecord {
    timestamp: String,
    price: f64,
    volume: u32,
    bid: f64,
    ask: f64,
    #[serde(default)]
    bid_size: Option<u32>,
    #[serde(default)]
    ask_size: Option<u32>,
    #[serde(default)]
    exchange: Option<String>,
    #[serde(default)]
    conditions: Option<Vec<String>>,
}
```

---

### Subtask 1.1.4: Alpaca API Integration (30 min)
**Priority:** MEDIUM
**File:** `hft-demo/src/market_data/loader.rs`

**Requirements:**
- [ ] Leverage existing Alpaca adapter from `prism-ai`
- [ ] Create wrapper for historical data fetching
- [ ] Support date range queries
- [ ] Handle rate limiting (100 requests/min)
- [ ] Implement retry logic with exponential backoff
- [ ] Cache responses to avoid redundant API calls
- [ ] Support multiple symbols
- [ ] Convert Alpaca format to `MarketTick`

**Code Template:**
```rust
use prism_ai::foundation::adapters::AlpacaMarketDataSource;
use tokio::time::{sleep, Duration};

pub struct AlpacaDataLoader {
    api_key: String,
    api_secret: String,
    symbols: Vec<String>,
    cache_dir: Option<String>,
}

impl AlpacaDataLoader {
    pub fn new(api_key: String, api_secret: String, symbols: Vec<String>) -> Self {
        Self {
            api_key,
            api_secret,
            symbols,
            cache_dir: Some("./data/cache".to_string()),
        }
    }

    /// Fetch historical data for date range
    pub async fn load_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MarketTick>> {
        // Check cache first
        if let Some(cached) = self.load_from_cache(start, end)? {
            log::info!("Loaded {} ticks from cache", cached.len());
            return Ok(cached);
        }

        // Fetch from API with rate limiting
        let mut all_ticks = Vec::new();

        for symbol in &self.symbols {
            log::info!("Fetching {} from Alpaca API", symbol);

            let ticks = self.fetch_symbol_range(symbol, start, end).await?;
            all_ticks.extend(ticks);

            // Rate limiting: wait 100ms between symbols
            sleep(Duration::from_millis(100)).await;
        }

        // Save to cache
        self.save_to_cache(&all_ticks, start, end)?;

        Ok(all_ticks)
    }

    async fn fetch_symbol_range(
        &self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MarketTick>> {
        // Implementation using Alpaca API
        // TODO: Use existing AlpacaMarketDataSource adapter
        todo!("Implement Alpaca API fetching")
    }
}
```

---

### Subtask 1.1.5: Data Caching (20 min)
**Priority:** MEDIUM

**Requirements:**
- [ ] Implement file-based cache for loaded data
- [ ] Use bincode for fast serialization
- [ ] Create cache directory structure
- [ ] Add cache invalidation logic
- [ ] Support cache expiration (e.g., 24 hours)
- [ ] Log cache hits/misses

**Code Template:**
```rust
use bincode::{serialize, deserialize};
use std::fs::{create_dir_all, File};
use std::io::{Write, Read};
use std::path::PathBuf;

impl AlpacaDataLoader {
    fn cache_path(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> PathBuf {
        let cache_dir = self.cache_dir.as_ref().unwrap();
        let filename = format!(
            "{}_{}_to_{}.bin",
            self.symbols.join("_"),
            start.format("%Y%m%d"),
            end.format("%Y%m%d")
        );
        PathBuf::from(cache_dir).join(filename)
    }

    fn load_from_cache(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Option<Vec<MarketTick>>> {
        let path = self.cache_path(start, end);

        if !path.exists() {
            return Ok(None);
        }

        let mut file = File::open(&path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let ticks: Vec<MarketTick> = deserialize(&buffer)
            .context("Failed to deserialize cache")?;

        Ok(Some(ticks))
    }

    fn save_to_cache(
        &self,
        ticks: &[MarketTick],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<()> {
        let path = self.cache_path(start, end);

        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }

        let encoded = serialize(ticks).context("Failed to serialize ticks")?;
        let mut file = File::create(&path)?;
        file.write_all(&encoded)?;

        log::info!("Saved {} ticks to cache: {:?}", ticks.len(), path);

        Ok(())
    }
}
```

---

### Subtask 1.1.6: Sample Data Generation (20 min)
**Priority:** HIGH
**File:** `hft-demo/data/sample_aapl_1hour.csv`

**Requirements:**
- [ ] Generate 1 hour of realistic AAPL tick data
- [ ] 1 tick per second = 3,600 ticks
- [ ] Realistic price movement (Brownian motion)
- [ ] Bid-ask spread of 1-5 cents
- [ ] Variable volume (50-500 shares per tick)
- [ ] Include occasional large trades
- [ ] Save as CSV with headers

**Script to Generate Sample Data:**
```rust
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::Write;

pub fn generate_sample_data(output_path: &str) -> Result<()> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 0.0001).unwrap();

    let mut file = File::create(output_path)?;
    writeln!(file, "timestamp,price,volume,bid,ask,bid_size,ask_size,exchange")?;

    let mut price = 182.50; // Starting price for AAPL
    let base_timestamp = 1704067200_000_000_000u64; // 2024-01-01 00:00:00 UTC

    for i in 0..3600 {
        // Random walk price
        let price_change = normal.sample(&mut rng);
        price = (price * (1.0 + price_change)).max(100.0).min(300.0);

        // Realistic spread (1-5 cents)
        let spread = rng.gen_range(0.01..0.05);
        let bid = price - spread / 2.0;
        let ask = price + spread / 2.0;

        // Variable volume
        let volume = rng.gen_range(50..500);
        let bid_size = rng.gen_range(10..200);
        let ask_size = rng.gen_range(10..200);

        // Timestamp: 1 tick per second
        let timestamp = base_timestamp + (i * 1_000_000_000);

        writeln!(
            file,
            "{},{:.2},{},{:.2},{:.2},{},{},NASDAQ",
            timestamp, price, volume, bid, ask, bid_size, ask_size
        )?;
    }

    log::info!("Generated sample data at {}", output_path);
    Ok(())
}
```

---

### Subtask 1.1.7: Unit Tests (20 min)
**Priority:** HIGH
**File:** `hft-demo/tests/market_data_tests.rs`

**Requirements:**
- [ ] Test CSV loading with sample file
- [ ] Test timestamp parsing (both formats)
- [ ] Test data structure conversions
- [ ] Test cache hit/miss logic
- [ ] ANTI-DRIFT: Verify no hardcoded values
- [ ] ANTI-DRIFT: Test that different data → different results

**Code Template:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_loader() {
        let loader = CsvDataLoader::new(
            "data/sample_aapl_1hour.csv".to_string(),
            "AAPL".to_string(),
        );

        let ticks = loader.load_all().expect("Failed to load CSV");

        assert_eq!(ticks.len(), 3600);
        assert_eq!(ticks[0].symbol, "AAPL");
        assert!(ticks[0].price > 0.0);
    }

    #[test]
    fn test_no_hardcoded_prices() {
        // ARES ANTI-DRIFT: Load two different data files
        let loader1 = CsvDataLoader::new("data/sample_aapl_1hour.csv".to_string(), "AAPL".to_string());
        let loader2 = CsvDataLoader::new("data/sample_tsla_1hour.csv".to_string(), "TSLA".to_string());

        let ticks1 = loader1.load_all().unwrap();
        let ticks2 = loader2.load_all().unwrap();

        // Verify different data sources produce different results
        assert_ne!(ticks1[0].price, ticks2[0].price);
        assert_ne!(ticks1[100].price, ticks2[100].price);
    }

    #[test]
    fn test_timestamp_parsing() {
        // Test RFC3339 format
        let tick1 = parse_timestamp("2024-01-01T00:00:00Z").unwrap();
        assert_eq!(tick1, 1704067200_000_000_000);

        // Test Unix milliseconds
        let tick2 = parse_timestamp("1704067200000").unwrap();
        assert_eq!(tick2, 1704067200_000_000_000);
    }
}
```

---

## 📝 Task 1.2: Market Simulator (4 hours)

**File:** `hft-demo/src/market_data/simulator.rs`
**Estimated Time:** 4 hours
**ARES Compliance:** HIGH RISK - Must avoid hardcoded market behaviors

### Subtask 1.2.1: Define Simulation Modes (30 min)

**Requirements:**
- [ ] Define `SimulationMode` enum with three variants
- [ ] Historical: Replay loaded data
- [ ] Synthetic: Generate realistic data
- [ ] Hybrid: Mix of both
- [ ] Define `MarketSimulator` struct
- [ ] Add configuration parameters

**Code Template:**
```rust
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub enum SimulationMode {
    Historical {
        data_source: String,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        speed_multiplier: f32,  // 1.0 = real-time, 1000.0 = fast
    },
    Synthetic {
        base_price: f64,
        volatility: f32,  // Annual volatility (e.g., 0.3 for 30%)
        trend: f32,       // Annual drift (e.g., 0.1 for 10% per year)
        microstructure_model: MicrostructureModel,
    },
    Hybrid {
        historical_base: String,
        add_noise: bool,
        noise_level: f32,
    },
}

#[derive(Debug, Clone)]
pub enum MicrostructureModel {
    SimplePoisson,          // Poisson arrival times
    HawkesProcess,          // Self-exciting process
    OrderBookDynamics,      // Limit order book simulation
}

pub struct MarketSimulator {
    mode: SimulationMode,
    current_tick: usize,
    tick_data: Vec<MarketTick>,
    rng: rand::rngs::ThreadRng,
}
```

---

### Subtask 1.2.2: Historical Replay (45 min)

**Requirements:**
- [ ] Implement `MarketSimulator::new_historical()`
- [ ] Load tick data from CSV or cache
- [ ] Implement `next_tick()` method
- [ ] Support speed multiplier (1x, 10x, 100x, 1000x)
- [ ] Add sleep/delay for real-time simulation
- [ ] Handle end-of-data gracefully

**Code Template:**
```rust
impl MarketSimulator {
    pub fn new_historical(
        data_path: String,
        symbol: String,
        speed_multiplier: f32,
    ) -> Result<Self> {
        let loader = CsvDataLoader::new(data_path, symbol);
        let tick_data = loader.load_all()?;

        Ok(Self {
            mode: SimulationMode::Historical {
                data_source: "csv".to_string(),
                start_date: Utc::now(),
                end_date: Utc::now(),
                speed_multiplier,
            },
            current_tick: 0,
            tick_data,
            rng: rand::thread_rng(),
        })
    }

    /// Get next market tick
    pub async fn next_tick(&mut self) -> Result<Option<MarketTick>> {
        if self.current_tick >= self.tick_data.len() {
            return Ok(None);
        }

        let tick = self.tick_data[self.current_tick].clone();
        self.current_tick += 1;

        // Calculate delay based on speed multiplier
        if let SimulationMode::Historical { speed_multiplier, .. } = self.mode {
            if self.current_tick < self.tick_data.len() {
                let next_tick = &self.tick_data[self.current_tick];
                let time_diff_ns = next_tick.timestamp_ns - tick.timestamp_ns;
                let delay_ns = (time_diff_ns as f32 / speed_multiplier) as u64;

                if delay_ns > 0 {
                    tokio::time::sleep(
                        tokio::time::Duration::from_nanos(delay_ns)
                    ).await;
                }
            }
        }

        Ok(Some(tick))
    }
}
```

---

### Subtask 1.2.3: Synthetic Data Generation (1.5 hours)

**Requirements:**
- [ ] Implement Geometric Brownian Motion for prices
- [ ] Add realistic bid-ask spread dynamics
- [ ] Simulate volume using Poisson process
- [ ] Add market microstructure effects
- [ ] Implement intraday patterns (open/close volatility)
- [ ] NO HARDCODED BEHAVIORS (ARES CRITICAL!)

**Code Template:**
```rust
use rand_distr::{Distribution, Normal, Poisson};

impl MarketSimulator {
    pub fn new_synthetic(
        symbol: String,
        base_price: f64,
        volatility: f32,
        trend: f32,
        duration_seconds: u64,
    ) -> Result<Self> {
        let mut tick_data = Vec::with_capacity(duration_seconds as usize);
        let mut rng = rand::thread_rng();

        // Geometric Brownian Motion parameters
        let dt = 1.0;  // 1 second timestep
        let mu = trend / (252.0 * 6.5 * 3600.0);  // Convert annual to per-second
        let sigma = volatility / (252.0 * 6.5 * 3600.0_f32).sqrt();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut price = base_price;
        let base_timestamp = Utc::now().timestamp_nanos_opt().unwrap() as u64;

        for i in 0..duration_seconds {
            // Geometric Brownian Motion: dS = μS dt + σS dW
            let dW = normal.sample(&mut rng) as f32;
            let price_change = mu * price as f32 * dt + sigma * price as f32 * dW * dt.sqrt();
            price = (price + price_change as f64).max(1.0);

            // Realistic spread (1-10 basis points)
            let spread_bps = rng.gen_range(1.0..10.0);
            let spread = price * spread_bps / 10000.0;
            let bid = price - spread / 2.0;
            let ask = price + spread / 2.0;

            // Volume follows Poisson distribution
            let volume_poisson = Poisson::new(100.0).unwrap();
            let volume = volume_poisson.sample(&mut rng).max(1) as u32;

            let tick = MarketTick {
                timestamp_ns: base_timestamp + i * 1_000_000_000,
                symbol: symbol.clone(),
                price,
                volume,
                bid,
                ask,
                bid_size: rng.gen_range(10..200),
                ask_size: rng.gen_range(10..200),
                exchange: "SYNTHETIC".to_string(),
                conditions: vec![],
            };

            tick_data.push(tick);
        }

        log::info!("Generated {} synthetic ticks", tick_data.len());

        Ok(Self {
            mode: SimulationMode::Synthetic {
                base_price,
                volatility,
                trend,
                microstructure_model: MicrostructureModel::SimplePoisson,
            },
            current_tick: 0,
            tick_data,
            rng,
        })
    }
}
```

---

### Subtask 1.2.4: Anti-Drift Validation (30 min)

**Requirements:**
- [ ] Write tests verifying different inputs → different outputs
- [ ] Test that volatility parameter affects price variance
- [ ] Test that trend parameter affects price drift
- [ ] Ensure no magic numbers in price generation

**Code Template:**
```rust
#[cfg(test)]
mod simulator_tests {
    use super::*;

    #[test]
    fn test_synthetic_varies_with_volatility() {
        // ARES STANDARD: Different parameters → different behavior
        let sim1 = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.1,  // Low volatility
            0.0,
            100,
        ).unwrap();

        let sim2 = MarketSimulator::new_synthetic(
            "TEST".to_string(),
            100.0,
            0.5,  // High volatility
            0.0,
            100,
        ).unwrap();

        // Calculate price variance for each
        let variance1 = calculate_price_variance(&sim1.tick_data);
        let variance2 = calculate_price_variance(&sim2.tick_data);

        // High volatility should produce higher variance
        assert!(variance2 > variance1 * 2.0);
    }

    #[test]
    fn test_no_hardcoded_prices() {
        // Generate two separate simulations
        let sim1 = MarketSimulator::new_synthetic("TEST".to_string(), 100.0, 0.3, 0.0, 10).unwrap();
        let sim2 = MarketSimulator::new_synthetic("TEST".to_string(), 100.0, 0.3, 0.0, 10).unwrap();

        // Should generate different price paths (random)
        assert_ne!(sim1.tick_data[5].price, sim2.tick_data[5].price);
    }
}
```

---

## 📝 Task 1.3: Feature Extraction (3 hours)

**File:** `hft-demo/src/market_data/features.rs`
**Estimated Time:** 3 hours
**ARES Compliance:** CRITICAL - All features must be computed, not hardcoded

### Subtask 1.3.1: Define MarketFeatures Struct (20 min)

**Requirements:**
- [ ] Define `MarketFeatures` struct with all fields
- [ ] Price features: returns, volatility, momentum
- [ ] Order book features: spread, imbalance, flow
- [ ] Technical indicators: RSI, MACD, Bollinger
- [ ] Microstructure: trade intensity, volume profile
- [ ] Add normalization methods

**Code Template:**
```rust
use ndarray::Array1;
use statrs::statistics::Statistics;

#[derive(Debug, Clone)]
pub struct MarketFeatures {
    // Price features
    pub returns: Vec<f32>,           // Log returns
    pub volatility: f32,             // Rolling volatility
    pub momentum: f32,               // Short-term momentum

    // Order book features
    pub spread_bps: f32,
    pub depth_imbalance: f32,
    pub order_flow: f32,

    // Technical indicators
    pub rsi: f32,                    // Relative Strength Index
    pub macd: f32,                   // MACD signal
    pub bollinger_position: f32,     // Position in Bollinger Bands

    // Microstructure
    pub trade_intensity: f32,        // Trades per second
    pub volume_profile: Vec<f32>,    // Volume by price level
    pub tick_direction: i8,          // +1 uptick, -1 downtick, 0 neutral

    // Normalized features for neural network
    pub normalized: Array1<f32>,
}
```

---

### Subtask 1.3.2: Implement Feature Extractor (1.5 hours)

**Requirements:**
- [ ] Create `FeatureExtractor` struct
- [ ] Maintain rolling window of ticks
- [ ] Compute returns from price changes
- [ ] Calculate rolling volatility
- [ ] Compute order flow imbalance
- [ ] ALL VALUES COMPUTED FROM DATA (ARES CRITICAL!)

**Code Template:**
```rust
pub struct FeatureExtractor {
    window_size: usize,
    tick_history: Vec<MarketTick>,
}

impl FeatureExtractor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            tick_history: Vec::with_capacity(window_size),
        }
    }

    /// Extract features from current tick (COMPUTED, NOT HARDCODED)
    pub fn extract(&mut self, tick: &MarketTick) -> Result<MarketFeatures> {
        // Add to history
        self.tick_history.push(tick.clone());
        if self.tick_history.len() > self.window_size {
            self.tick_history.remove(0);
        }

        // Require minimum history
        if self.tick_history.len() < 10 {
            anyhow::bail!("Insufficient history for feature extraction");
        }

        // COMPUTED: Log returns
        let returns = self.compute_returns();

        // COMPUTED: Volatility (standard deviation of returns)
        let volatility = returns.std_dev() as f32;

        // COMPUTED: Momentum (average return over window)
        let momentum = returns.mean() as f32;

        // COMPUTED: Order flow
        let order_flow = self.compute_order_flow();

        // COMPUTED: RSI
        let rsi = self.compute_rsi(14);

        // COMPUTED: MACD
        let macd = self.compute_macd();

        // COMPUTED: Bollinger position
        let bollinger_position = self.compute_bollinger_position();

        // COMPUTED: Trade intensity
        let trade_intensity = self.compute_trade_intensity();

        // COMPUTED: Tick direction
        let tick_direction = self.compute_tick_direction();

        Ok(MarketFeatures {
            returns: returns.iter().map(|&x| x as f32).collect(),
            volatility,
            momentum,
            spread_bps: ((tick.ask - tick.bid) / tick.bid * 10000.0) as f32,
            depth_imbalance: (tick.bid_size as f32 - tick.ask_size as f32)
                / (tick.bid_size + tick.ask_size) as f32,
            order_flow,
            rsi,
            macd,
            bollinger_position,
            trade_intensity,
            volume_profile: self.compute_volume_profile(),
            tick_direction,
            normalized: self.normalize_features(),
        })
    }

    fn compute_returns(&self) -> Vec<f64> {
        self.tick_history.windows(2)
            .map(|w| (w[1].price / w[0].price).ln())
            .collect()
    }

    fn compute_order_flow(&self) -> f32 {
        // COMPUTED from bid/ask changes
        let recent = &self.tick_history[self.tick_history.len().saturating_sub(10)..];
        let buy_volume: u32 = recent.iter()
            .filter(|t| t.price >= t.ask)
            .map(|t| t.volume)
            .sum();
        let sell_volume: u32 = recent.iter()
            .filter(|t| t.price <= t.bid)
            .map(|t| t.volume)
            .sum();

        if buy_volume + sell_volume > 0 {
            (buy_volume as f32 - sell_volume as f32) / (buy_volume + sell_volume) as f32
        } else {
            0.0
        }
    }

    fn compute_rsi(&self, period: usize) -> f32 {
        // COMPUTED: Relative Strength Index
        if self.tick_history.len() < period + 1 {
            return 50.0;  // Neutral
        }

        let returns = self.compute_returns();
        let recent_returns = &returns[returns.len().saturating_sub(period)..];

        let gains: f64 = recent_returns.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = recent_returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();

        if losses == 0.0 {
            return 100.0;
        }

        let rs = gains / losses;
        (100.0 - 100.0 / (1.0 + rs)) as f32
    }
}
```

---

### Subtask 1.3.3: Implement Technical Indicators (1 hour)

**Requirements:**
- [ ] Implement MACD (Moving Average Convergence Divergence)
- [ ] Implement Bollinger Bands
- [ ] Implement volume profile
- [ ] Use `ta` crate for standard indicators
- [ ] ALL COMPUTED FROM DATA

---

### Subtask 1.3.4: Feature Normalization (30 min)

**Requirements:**
- [ ] Normalize all features to [0, 1] or [-1, 1]
- [ ] Use z-score normalization for price features
- [ ] Min-max scaling for bounded indicators
- [ ] Create normalized feature vector for neural network

---

## 📝 Task 1.4: Data Validation (2 hours)

**File:** `hft-demo/src/market_data/validation.rs`
**Estimated Time:** 2 hours

### Subtask 1.4.1: Outlier Detection (45 min)

**Requirements:**
- [ ] Detect price spikes (>5 standard deviations)
- [ ] Flag suspicious volume
- [ ] Check bid-ask spread validity
- [ ] Log warnings for anomalies

---

### Subtask 1.4.2: Data Quality Checks (45 min)

**Requirements:**
- [ ] Check for timestamp gaps
- [ ] Validate bid < ask always
- [ ] Ensure positive prices and volumes
- [ ] Detect duplicate timestamps

---

### Subtask 1.4.3: Integration Tests (30 min)

**Requirements:**
- [ ] End-to-end test: Load → Simulate → Extract → Validate
- [ ] Verify full pipeline works
- [ ] Test with multiple data sources
- [ ] ANTI-DRIFT: Verify all paths produce different results

---

## ✅ Phase 1 Completion Checklist

### Must Have (MVP)
- [ ] MarketTick and OrderBookSnapshot structs defined
- [ ] CSV loader working with sample data
- [ ] Historical replay simulator functional
- [ ] Basic feature extraction (returns, volatility, spread)
- [ ] Sample data file (1 hour AAPL)
- [ ] Unit tests passing
- [ ] Anti-drift validation passing

### Should Have
- [ ] Alpaca API integration
- [ ] Data caching layer
- [ ] Synthetic data generation
- [ ] Full technical indicators (RSI, MACD, Bollinger)
- [ ] Comprehensive tests

### Nice to Have
- [ ] Parquet support for large files
- [ ] Streaming data loader
- [ ] Advanced microstructure models
- [ ] Performance benchmarks

---

## 🎯 Success Criteria

**Phase 1 is complete when:**
1. ✅ Can load 3,600 ticks from CSV in <100ms
2. ✅ Can replay ticks at 1x, 10x, 100x, 1000x speed
3. ✅ Can generate synthetic data with realistic properties
4. ✅ Can extract 15+ features from tick stream
5. ✅ All tests passing (including anti-drift tests)
6. ✅ No hardcoded values in any computations
7. ✅ Documentation complete for all public APIs

**Performance Targets:**
- Load 10K ticks: <500ms
- Extract features: <1ms per tick
- Synthetic generation: >1000 ticks/sec

---

## 📊 Time Tracking

| Task | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| 1.1.1 Project Setup | 20 min | | |
| 1.1.2 Data Structures | 30 min | | |
| 1.1.3 CSV Loader | 40 min | | |
| 1.1.4 Alpaca API | 30 min | | |
| 1.1.5 Caching | 20 min | | |
| 1.1.6 Sample Data | 20 min | | |
| 1.1.7 Tests | 20 min | | |
| **Task 1.1 Total** | **3 hours** | | |
| 1.2.1 Simulation Modes | 30 min | | |
| 1.2.2 Historical Replay | 45 min | | |
| 1.2.3 Synthetic Generation | 1.5 hours | | |
| 1.2.4 Anti-Drift Tests | 30 min | | |
| **Task 1.2 Total** | **4 hours** | | |
| 1.3.1 Features Struct | 20 min | | |
| 1.3.2 Feature Extraction | 1.5 hours | | |
| 1.3.3 Technical Indicators | 1 hour | | |
| 1.3.4 Normalization | 30 min | | |
| **Task 1.3 Total** | **3 hours** | | |
| 1.4.1 Outlier Detection | 45 min | | |
| 1.4.2 Quality Checks | 45 min | | |
| 1.4.3 Integration Tests | 30 min | | |
| **Task 1.4 Total** | **2 hours** | | |
| **PHASE 1 TOTAL** | **12 hours** | | |

---

*Document created: 2025-10-10*
*Status: Ready for implementation*
*Next: Begin Task 1.1.1 - Project Setup*
