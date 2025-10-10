//! Market Data Loader
//!
//! Loads historical tick data from various sources including CSV files,
//! Parquet files, and market data APIs.
//!
//! # ARES Anti-Drift Compliance
//!
//! This module is CRITICAL for anti-drift compliance. All data must be
//! loaded from actual sources, never hardcoded.
//!
//! ## Forbidden
//! ```rust,ignore
//! // ❌ NEVER return hardcoded data
//! fn load_ticks(&self) -> Vec<MarketTick> {
//!     vec![MarketTick { price: 100.0, volume: 1000, ... }]  // HARDCODED!
//! }
//! ```
//!
//! ## Required
//! ```rust,ignore
//! // ✅ Always load from actual source
//! fn load_ticks(&self) -> Result<Vec<MarketTick>> {
//!     let file = File::open(&self.file_path)?;
//!     let mut reader = csv::Reader::from_reader(file);
//!     // ... parse actual data
//! }
//! ```

use chrono::{DateTime, NaiveDateTime};
use serde::{Deserialize, Serialize};
use anyhow::{Context, Result, bail};
use std::fs::File;
use std::path::Path;

// Module will be implemented in Task 1.1.2 and beyond
// This is just the skeleton for compilation

/// Market tick data with nanosecond precision
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

/// Order book snapshot with computed metrics
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
        // COMPUTE spread from actual bid/ask
        let spread_bps = if !bids.is_empty() && !asks.is_empty() {
            let best_bid = bids[0].0;
            let best_ask = asks[0].0;
            ((best_ask - best_bid) / best_bid * 10_000.0) as f32
        } else {
            0.0
        };

        // COMPUTE imbalance from actual volumes
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

/// CSV data loader with streaming support
pub struct CsvDataLoader {
    file_path: String,
    symbol: String,
}

/// CSV record format for parsing
#[derive(Debug, Deserialize)]
struct CsvRecord {
    timestamp: String,
    symbol: String,
    price: f64,
    volume: u32,
    bid: f64,
    ask: f64,
    bid_size: u32,
    ask_size: u32,
    exchange: String,
    #[serde(default)]
    conditions: String,
}

impl CsvDataLoader {
    /// Create new CSV loader
    pub fn new(file_path: String, symbol: String) -> Self {
        Self { file_path, symbol }
    }

    /// Load all ticks from CSV file
    ///
    /// # ARES Compliance
    /// This function MUST load from actual file, never return hardcoded data
    ///
    /// # Timestamp Formats Supported
    /// - RFC3339: "2024-01-15T09:30:00.123456789Z"
    /// - Unix seconds: "1705314600"
    /// - Unix milliseconds: "1705314600123"
    /// - Unix nanoseconds: "1705314600123456789"
    ///
    /// # CSV Format
    /// Expected columns: timestamp,symbol,price,volume,bid,ask,bid_size,ask_size,exchange,conditions
    pub fn load_all(&self) -> Result<Vec<MarketTick>> {
        // Verify file exists
        if !Path::new(&self.file_path).exists() {
            bail!("CSV file not found: {}", self.file_path);
        }

        let file = File::open(&self.file_path)
            .with_context(|| format!("Failed to open CSV file: {}", self.file_path))?;

        let mut reader = csv::Reader::from_reader(file);
        let mut ticks = Vec::new();
        let mut line_num = 1; // Start at 1 (header is line 0)

        for result in reader.deserialize() {
            line_num += 1;

            let record: CsvRecord = result
                .with_context(|| format!("Failed to parse CSV line {}", line_num))?;

            // Validate data quality
            if record.price <= 0.0 {
                log::warn!("Line {}: Invalid price {} (must be > 0), skipping", line_num, record.price);
                continue;
            }

            if record.bid <= 0.0 || record.ask <= 0.0 {
                log::warn!("Line {}: Invalid bid/ask ({}/{}), skipping", line_num, record.bid, record.ask);
                continue;
            }

            if record.ask < record.bid {
                log::warn!("Line {}: Ask {} < Bid {} (crossed market), skipping",
                          line_num, record.ask, record.bid);
                continue;
            }

            // Parse timestamp to nanoseconds
            let timestamp_ns = Self::parse_timestamp(&record.timestamp)
                .with_context(|| format!("Line {}: Failed to parse timestamp '{}'",
                                        line_num, record.timestamp))?;

            // Parse conditions (comma-separated string to Vec)
            let conditions: Vec<String> = if record.conditions.is_empty() {
                Vec::new()
            } else {
                record.conditions.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            };

            ticks.push(MarketTick {
                timestamp_ns,
                symbol: record.symbol,
                price: record.price,
                volume: record.volume,
                bid: record.bid,
                ask: record.ask,
                bid_size: record.bid_size,
                ask_size: record.ask_size,
                exchange: record.exchange,
                conditions,
            });
        }

        if ticks.is_empty() {
            bail!("No valid ticks loaded from {}", self.file_path);
        }

        log::info!("Loaded {} ticks from {} (symbol: {})",
                  ticks.len(), self.file_path, self.symbol);

        Ok(ticks)
    }

    /// Parse timestamp from various formats to nanoseconds
    fn parse_timestamp(timestamp_str: &str) -> Result<u64> {
        let timestamp_str = timestamp_str.trim();

        // Try RFC3339 format first (ISO 8601)
        if timestamp_str.contains('T') || timestamp_str.contains('-') {
            if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp_str) {
                let nanos = dt.timestamp_nanos_opt()
                    .ok_or_else(|| anyhow::anyhow!("Timestamp out of range"))?;
                return Ok(nanos as u64);
            }

            // Try naive datetime (no timezone)
            if let Ok(ndt) = NaiveDateTime::parse_from_str(timestamp_str, "%Y-%m-%d %H:%M:%S%.f") {
                return Ok((ndt.and_utc().timestamp_nanos_opt()
                    .ok_or_else(|| anyhow::anyhow!("Timestamp out of range"))?) as u64);
            }

            if let Ok(ndt) = NaiveDateTime::parse_from_str(timestamp_str, "%Y-%m-%d %H:%M:%S") {
                return Ok((ndt.and_utc().timestamp_nanos_opt()
                    .ok_or_else(|| anyhow::anyhow!("Timestamp out of range"))?) as u64);
            }
        }

        // Try parsing as numeric timestamp
        if let Ok(num) = timestamp_str.parse::<u64>() {
            // Detect format by magnitude
            if num > 1_000_000_000_000_000_000 {
                // Already in nanoseconds (> ~2001 in nanos)
                return Ok(num);
            } else if num > 1_000_000_000_000 {
                // Milliseconds (> ~2001 in millis)
                return Ok(num * 1_000_000);
            } else if num > 1_000_000_000 {
                // Seconds (> ~2001 in seconds)
                return Ok(num * 1_000_000_000);
            } else {
                bail!("Timestamp {} too small to determine format", num);
            }
        }

        bail!("Unable to parse timestamp: '{}'", timestamp_str)
    }

    /// Load ticks with progress reporting for large files
    pub fn load_with_progress<F>(&self, mut progress_callback: F) -> Result<Vec<MarketTick>>
    where
        F: FnMut(usize),
    {
        if !Path::new(&self.file_path).exists() {
            bail!("CSV file not found: {}", self.file_path);
        }

        let file = File::open(&self.file_path)
            .with_context(|| format!("Failed to open CSV file: {}", self.file_path))?;

        let mut reader = csv::Reader::from_reader(file);
        let mut ticks = Vec::new();
        let mut line_num = 1;

        for result in reader.deserialize() {
            line_num += 1;

            let record: CsvRecord = result
                .with_context(|| format!("Failed to parse CSV line {}", line_num))?;

            if record.price <= 0.0 || record.bid <= 0.0 || record.ask <= 0.0 {
                continue;
            }

            if record.ask < record.bid {
                continue;
            }

            let timestamp_ns = Self::parse_timestamp(&record.timestamp)
                .with_context(|| format!("Line {}: Failed to parse timestamp", line_num))?;

            let conditions: Vec<String> = if record.conditions.is_empty() {
                Vec::new()
            } else {
                record.conditions.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            };

            ticks.push(MarketTick {
                timestamp_ns,
                symbol: record.symbol,
                price: record.price,
                volume: record.volume,
                bid: record.bid,
                ask: record.ask,
                bid_size: record.bid_size,
                ask_size: record.ask_size,
                exchange: record.exchange,
                conditions,
            });

            // Report progress every 1000 records
            if ticks.len() % 1000 == 0 {
                progress_callback(ticks.len());
            }
        }

        if ticks.is_empty() {
            bail!("No valid ticks loaded from {}", self.file_path);
        }

        progress_callback(ticks.len()); // Final progress report
        log::info!("Loaded {} ticks from {}", ticks.len(), self.file_path);

        Ok(ticks)
    }

    /// Get iterator for streaming large files (memory-efficient)
    pub fn iter(&self) -> Result<CsvTickIterator> {
        CsvTickIterator::new(&self.file_path)
    }
}

/// Iterator for streaming CSV ticks without loading all into memory
pub struct CsvTickIterator {
    inner: csv::DeserializeRecordsIntoIter<File, CsvRecord>,
    line_num: usize,
}

impl CsvTickIterator {
    fn new(file_path: &str) -> Result<Self> {
        if !Path::new(file_path).exists() {
            bail!("CSV file not found: {}", file_path);
        }

        let file = File::open(file_path)
            .with_context(|| format!("Failed to open CSV file: {}", file_path))?;

        let reader = csv::Reader::from_reader(file);
        let inner = reader.into_deserialize();

        Ok(Self {
            inner,
            line_num: 1,
        })
    }
}

impl Iterator for CsvTickIterator {
    type Item = Result<MarketTick>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.line_num += 1;

            match self.inner.next() {
                Some(Ok(record)) => {
                    // Validate data quality
                    if record.price <= 0.0 || record.bid <= 0.0 || record.ask <= 0.0 {
                        continue; // Skip invalid records
                    }

                    if record.ask < record.bid {
                        continue; // Skip crossed markets
                    }

                    // Parse timestamp
                    let timestamp_ns = match CsvDataLoader::parse_timestamp(&record.timestamp) {
                        Ok(ts) => ts,
                        Err(e) => return Some(Err(e.context(format!("Line {}", self.line_num)))),
                    };

                    // Parse conditions
                    let conditions: Vec<String> = if record.conditions.is_empty() {
                        Vec::new()
                    } else {
                        record.conditions.split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect()
                    };

                    return Some(Ok(MarketTick {
                        timestamp_ns,
                        symbol: record.symbol,
                        price: record.price,
                        volume: record.volume,
                        bid: record.bid,
                        ask: record.ask,
                        bid_size: record.bid_size,
                        ask_size: record.ask_size,
                        exchange: record.exchange,
                        conditions,
                    }));
                }
                Some(Err(e)) => return Some(Err(anyhow::Error::from(e).context(format!("Line {}", self.line_num)))),
                None => return None, // End of file
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_timestamp_rfc3339() {
        // RFC3339 format
        let ts = CsvDataLoader::parse_timestamp("2024-01-15T09:30:00.123456789Z").unwrap();
        assert!(ts > 0);

        // With timezone
        let ts2 = CsvDataLoader::parse_timestamp("2024-01-15T09:30:00-05:00").unwrap();
        assert!(ts2 > 0);
    }

    #[test]
    fn test_parse_timestamp_unix() {
        // Unix seconds
        let ts_sec = CsvDataLoader::parse_timestamp("1705314600").unwrap();
        assert_eq!(ts_sec, 1705314600 * 1_000_000_000);

        // Unix milliseconds
        let ts_ms = CsvDataLoader::parse_timestamp("1705314600123").unwrap();
        assert_eq!(ts_ms, 1705314600123 * 1_000_000);

        // Unix nanoseconds
        let ts_ns = CsvDataLoader::parse_timestamp("1705314600123456789").unwrap();
        assert_eq!(ts_ns, 1705314600123456789);
    }

    #[test]
    fn test_parse_timestamp_naive() {
        // Naive datetime formats
        let ts1 = CsvDataLoader::parse_timestamp("2024-01-15 09:30:00").unwrap();
        assert!(ts1 > 0);

        let ts2 = CsvDataLoader::parse_timestamp("2024-01-15 09:30:00.123456").unwrap();
        assert!(ts2 > 0);
    }

    #[test]
    fn test_parse_timestamp_invalid() {
        assert!(CsvDataLoader::parse_timestamp("not-a-timestamp").is_err());
        assert!(CsvDataLoader::parse_timestamp("123").is_err()); // Too small
        assert!(CsvDataLoader::parse_timestamp("").is_err());
    }

    #[test]
    fn test_csv_loader_basic() -> Result<()> {
        // Create temporary CSV file
        let mut tmp_file = NamedTempFile::new()?;
        writeln!(tmp_file, "timestamp,symbol,price,volume,bid,ask,bid_size,ask_size,exchange,conditions")?;
        writeln!(tmp_file, "2024-01-15T09:30:00Z,AAPL,150.25,100,150.24,150.26,50,50,NASDAQ,\"\"")?;
        writeln!(tmp_file, "2024-01-15T09:30:01Z,AAPL,150.30,200,150.29,150.31,60,40,NASDAQ,\"@\"")?;
        writeln!(tmp_file, "2024-01-15T09:30:02Z,AAPL,150.28,150,150.27,150.29,55,45,NASDAQ,\"@,F\"")?;
        tmp_file.flush()?;

        let loader = CsvDataLoader::new(
            tmp_file.path().to_str().unwrap().to_string(),
            "AAPL".to_string()
        );

        let ticks = loader.load_all()?;

        // ARES ANTI-DRIFT: Verify actual data loaded
        assert_eq!(ticks.len(), 3);
        assert_eq!(ticks[0].symbol, "AAPL");
        assert_eq!(ticks[0].price, 150.25);
        assert_eq!(ticks[0].volume, 100);
        assert_eq!(ticks[1].price, 150.30);
        assert_eq!(ticks[2].conditions.len(), 2);
        assert_eq!(ticks[2].conditions[0], "@");
        assert_eq!(ticks[2].conditions[1], "F");

        Ok(())
    }

    #[test]
    fn test_csv_loader_validation() -> Result<()> {
        // Create CSV with invalid data
        let mut tmp_file = NamedTempFile::new()?;
        writeln!(tmp_file, "timestamp,symbol,price,volume,bid,ask,bid_size,ask_size,exchange,conditions")?;
        writeln!(tmp_file, "2024-01-15T09:30:00Z,AAPL,150.25,100,150.24,150.26,50,50,NASDAQ,\"\"")?; // Valid
        writeln!(tmp_file, "2024-01-15T09:30:01Z,AAPL,-10.0,200,150.29,150.31,60,40,NASDAQ,\"\"")?;  // Invalid price
        writeln!(tmp_file, "2024-01-15T09:30:02Z,AAPL,150.28,150,150.30,150.29,55,45,NASDAQ,\"\"")?; // Crossed market
        writeln!(tmp_file, "2024-01-15T09:30:03Z,AAPL,150.32,100,150.31,150.33,50,50,NASDAQ,\"\"")?; // Valid
        tmp_file.flush()?;

        let loader = CsvDataLoader::new(
            tmp_file.path().to_str().unwrap().to_string(),
            "AAPL".to_string()
        );

        let ticks = loader.load_all()?;

        // ARES ANTI-DRIFT: Should only load 2 valid ticks
        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].price, 150.25);
        assert_eq!(ticks[1].price, 150.32);

        Ok(())
    }

    #[test]
    fn test_csv_loader_progress() -> Result<()> {
        let mut tmp_file = NamedTempFile::new()?;
        writeln!(tmp_file, "timestamp,symbol,price,volume,bid,ask,bid_size,ask_size,exchange,conditions")?;

        // Write 2500 records to trigger multiple progress callbacks
        for i in 0..2500 {
            writeln!(tmp_file, "2024-01-15T09:30:{:02}Z,AAPL,{:.2},100,{:.2},{:.2},50,50,NASDAQ,\"\"",
                    i % 60, 150.0 + (i as f64 * 0.01), 149.99 + (i as f64 * 0.01), 150.01 + (i as f64 * 0.01))?;
        }
        tmp_file.flush()?;

        let loader = CsvDataLoader::new(
            tmp_file.path().to_str().unwrap().to_string(),
            "AAPL".to_string()
        );

        let mut progress_updates = Vec::new();
        let ticks = loader.load_with_progress(|count| {
            progress_updates.push(count);
        })?;

        // ARES ANTI-DRIFT: Verify actual data and progress
        assert_eq!(ticks.len(), 2500);
        assert!(progress_updates.len() > 2); // Should have multiple updates
        assert_eq!(*progress_updates.last().unwrap(), 2500);

        Ok(())
    }

    #[test]
    fn test_csv_iterator() -> Result<()> {
        let mut tmp_file = NamedTempFile::new()?;
        writeln!(tmp_file, "timestamp,symbol,price,volume,bid,ask,bid_size,ask_size,exchange,conditions")?;
        writeln!(tmp_file, "2024-01-15T09:30:00Z,AAPL,150.25,100,150.24,150.26,50,50,NASDAQ,\"\"")?;
        writeln!(tmp_file, "2024-01-15T09:30:01Z,AAPL,150.30,200,150.29,150.31,60,40,NASDAQ,\"\"")?;
        tmp_file.flush()?;

        let loader = CsvDataLoader::new(
            tmp_file.path().to_str().unwrap().to_string(),
            "AAPL".to_string()
        );

        let mut count = 0;
        for result in loader.iter()? {
            let tick = result?;
            assert_eq!(tick.symbol, "AAPL");
            count += 1;
        }

        // ARES ANTI-DRIFT: Verify iteration works
        assert_eq!(count, 2);

        Ok(())
    }

    #[test]
    fn test_csv_loader_nonexistent_file() {
        let loader = CsvDataLoader::new(
            "/nonexistent/path/to/file.csv".to_string(),
            "AAPL".to_string()
        );

        assert!(loader.load_all().is_err());
    }

    #[test]
    fn test_csv_loader_empty_file() {
        let mut tmp_file = NamedTempFile::new().unwrap();
        writeln!(tmp_file, "timestamp,symbol,price,volume,bid,ask,bid_size,ask_size,exchange,conditions").unwrap();
        tmp_file.flush().unwrap();

        let loader = CsvDataLoader::new(
            tmp_file.path().to_str().unwrap().to_string(),
            "AAPL".to_string()
        );

        // Should error because no valid ticks
        assert!(loader.load_all().is_err());
    }

    #[test]
    fn test_order_book_spread_computed() {
        // ARES ANTI-DRIFT: Verify spread is computed, not hardcoded
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

        // Different order books → different spreads
        assert_ne!(snapshot1.spread_bps, snapshot2.spread_bps);

        // Verify spread is positive and reasonable
        assert!(snapshot1.spread_bps > 0.0);
        assert!(snapshot2.spread_bps > snapshot1.spread_bps);
    }

    #[test]
    fn test_order_book_imbalance_computed() {
        // ARES ANTI-DRIFT: Verify imbalance is computed
        let snapshot_buy_heavy = OrderBookSnapshot::new(
            0,
            vec![(100.0, 100), (99.9, 50)],  // 150 bid volume
            vec![(100.1, 50)]                 // 50 ask volume
        );

        let snapshot_sell_heavy = OrderBookSnapshot::new(
            0,
            vec![(100.0, 50)],                // 50 bid volume
            vec![(100.1, 100), (100.2, 50)]  // 150 ask volume
        );

        // Buy-heavy should have positive imbalance
        assert!(snapshot_buy_heavy.imbalance > 0.0);

        // Sell-heavy should have negative imbalance
        assert!(snapshot_sell_heavy.imbalance < 0.0);

        // Verify they're different
        assert_ne!(snapshot_buy_heavy.imbalance, snapshot_sell_heavy.imbalance);
    }
}
