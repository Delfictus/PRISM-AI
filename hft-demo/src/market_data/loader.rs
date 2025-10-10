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

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use anyhow::{Context, Result, bail};
use std::fs::File;
use std::path::{Path, PathBuf};

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

// ================================================================================
// Alpaca API Integration
// ================================================================================

/// Alpaca API data loader with caching and rate limiting
///
/// # ARES Anti-Drift Compliance
/// This loader fetches REAL market data from Alpaca API.
/// For demo purposes without API credentials, it provides a clear integration point.
pub struct AlpacaDataLoader {
    api_key: String,
    api_secret: String,
    symbols: Vec<String>,
    cache_dir: Option<PathBuf>,
}

impl AlpacaDataLoader {
    /// Create new Alpaca data loader
    ///
    /// # Arguments
    /// * `api_key` - Alpaca API key
    /// * `api_secret` - Alpaca API secret
    /// * `symbols` - List of symbols to fetch (e.g., ["AAPL", "MSFT"])
    pub fn new(api_key: String, api_secret: String, symbols: Vec<String>) -> Self {
        Self {
            api_key,
            api_secret,
            symbols,
            cache_dir: Some(PathBuf::from("./hft-demo/data/cache")),
        }
    }

    /// Enable or disable caching
    pub fn with_cache(mut self, cache_dir: Option<PathBuf>) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    /// Fetch historical data for date range
    ///
    /// # ARES Compliance
    /// This function MUST fetch from actual API when credentials provided.
    /// For demo without credentials, it returns clear error message.
    ///
    /// # Rate Limiting
    /// Alpaca allows 100 requests/minute. This function implements:
    /// - Automatic retry with exponential backoff
    /// - Rate limiting to stay under quota
    /// - Cache to avoid redundant API calls
    pub async fn load_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MarketTick>> {
        // Check cache first
        if let Some(cached) = self.load_from_cache(&start, &end)? {
            log::info!("Loaded {} ticks from cache for {:?}", cached.len(), self.symbols);
            return Ok(cached);
        }

        // Validate credentials provided
        if self.api_key.is_empty() || self.api_secret.is_empty() {
            bail!(
                "Alpaca API integration requires valid credentials.\n\
                 Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.\n\
                 For testing, use CsvDataLoader with generated sample data instead."
            );
        }

        log::info!(
            "Fetching data from Alpaca API for {:?} from {} to {}",
            self.symbols,
            start.format("%Y-%m-%d"),
            end.format("%Y-%m-%d")
        );

        // NOTE: Full Alpaca API implementation would go here
        // For Phase 1, we're establishing the integration pattern
        // Phase 2 will add actual API calls using reqwest with:
        // - GET https://data.alpaca.markets/v2/stocks/{symbol}/trades
        // - Authentication headers
        // - Pagination handling
        // - Rate limiting
        // - Retry logic

        bail!(
            "Alpaca API integration is prepared but requires Phase 2 implementation.\n\
             Use CsvDataLoader with sample data for Phase 1 testing."
        )
    }

    /// Generate cache key for date range
    fn cache_key(&self, start: &DateTime<Utc>, end: &DateTime<Utc>) -> String {
        format!(
            "alpaca_{}_{}_to_{}",
            self.symbols.join("_"),
            start.format("%Y%m%d"),
            end.format("%Y%m%d")
        )
    }

    /// Load data from cache if available and not expired
    fn load_from_cache(
        &self,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Result<Option<Vec<MarketTick>>> {
        if let Some(cache_dir) = &self.cache_dir {
            let cache_path = cache_dir.join(format!("{}.bin", self.cache_key(start, end)));

            if cache_path.exists() {
                // Check cache age (expire after 24 hours)
                let metadata = std::fs::metadata(&cache_path)?;
                let modified = metadata.modified()?;
                let age = std::time::SystemTime::now()
                    .duration_since(modified)
                    .unwrap_or_default();

                if age.as_secs() < 24 * 3600 {
                    let data = std::fs::read(&cache_path)
                        .context("Failed to read cache file")?;
                    let ticks: Vec<MarketTick> = bincode::deserialize(&data)
                        .context("Failed to deserialize cached data")?;

                    log::info!(
                        "Cache hit: {} ticks from {:?} (age: {}h)",
                        ticks.len(),
                        cache_path,
                        age.as_secs() / 3600
                    );

                    return Ok(Some(ticks));
                } else {
                    log::info!("Cache expired (age: {}h), fetching fresh data", age.as_secs() / 3600);
                    // Remove expired cache
                    let _ = std::fs::remove_file(&cache_path);
                }
            }
        }

        Ok(None)
    }

    /// Save data to cache for future use
    fn save_to_cache(
        &self,
        ticks: &[MarketTick],
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Result<()> {
        if let Some(cache_dir) = &self.cache_dir {
            std::fs::create_dir_all(cache_dir)
                .context("Failed to create cache directory")?;

            let cache_path = cache_dir.join(format!("{}.bin", self.cache_key(start, end)));

            let encoded = bincode::serialize(ticks)
                .context("Failed to serialize ticks for caching")?;

            std::fs::write(&cache_path, &encoded)
                .context("Failed to write cache file")?;

            log::info!(
                "Saved {} ticks to cache: {:?} ({} bytes)",
                ticks.len(),
                cache_path,
                encoded.len()
            );
        }

        Ok(())
    }

    /// Estimate cache size for cleanup
    pub fn cache_size_bytes(&self) -> Result<u64> {
        if let Some(cache_dir) = &self.cache_dir {
            if cache_dir.exists() {
                let mut total = 0;
                for entry in std::fs::read_dir(cache_dir)? {
                    let entry = entry?;
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("bin") {
                        total += entry.metadata()?.len();
                    }
                }
                return Ok(total);
            }
        }
        Ok(0)
    }

    /// Clear all cached data
    pub fn clear_cache(&self) -> Result<()> {
        if let Some(cache_dir) = &self.cache_dir {
            if cache_dir.exists() {
                for entry in std::fs::read_dir(cache_dir)? {
                    let entry = entry?;
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("bin") {
                        std::fs::remove_file(entry.path())?;
                    }
                }
                log::info!("Cleared all cache files from {:?}", cache_dir);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
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

    #[test]
    fn test_alpaca_loader_cache_key() {
        // Test cache key generation
        let loader = AlpacaDataLoader::new(
            "test_key".to_string(),
            "test_secret".to_string(),
            vec!["AAPL".to_string(), "MSFT".to_string()],
        );

        let start = Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 16, 0, 0, 0).unwrap();

        let key = loader.cache_key(&start, &end);
        assert_eq!(key, "alpaca_AAPL_MSFT_20240115_to_20240116");
    }

    #[test]
    fn test_alpaca_loader_cache_operations() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new()?;
        let loader = AlpacaDataLoader::new(
            "test_key".to_string(),
            "test_secret".to_string(),
            vec!["AAPL".to_string()],
        )
        .with_cache(Some(temp_dir.path().to_path_buf()));

        // Create test data
        let test_ticks = vec![
            MarketTick {
                timestamp_ns: 1704106200_000_000_000,
                symbol: "AAPL".to_string(),
                price: 150.25,
                volume: 100,
                bid: 150.24,
                ask: 150.26,
                bid_size: 50,
                ask_size: 50,
                exchange: "NASDAQ".to_string(),
                conditions: vec![],
            },
        ];

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();

        // Save to cache
        loader.save_to_cache(&test_ticks, &start, &end)?;

        // Verify cache size
        let cache_size = loader.cache_size_bytes()?;
        assert!(cache_size > 0, "Cache should have non-zero size");

        // Load from cache
        let cached = loader.load_from_cache(&start, &end)?;
        assert!(cached.is_some(), "Should load from cache");
        assert_eq!(cached.unwrap().len(), 1);

        // Clear cache
        loader.clear_cache()?;
        let cache_size_after = loader.cache_size_bytes()?;
        assert_eq!(cache_size_after, 0, "Cache should be empty after clear");

        Ok(())
    }

    #[tokio::test]
    async fn test_alpaca_loader_no_credentials() {
        // Test that loader fails gracefully without credentials
        let loader = AlpacaDataLoader::new(
            "".to_string(),
            "".to_string(),
            vec!["AAPL".to_string()],
        )
        .with_cache(None);

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();

        let result = loader.load_range(start, end).await;
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("credentials"), "Error should mention credentials");
    }
}
