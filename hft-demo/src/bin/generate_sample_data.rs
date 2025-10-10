//! Generate Sample Market Data
//!
//! Generates realistic sample tick data for testing and development.
//! Uses Geometric Brownian Motion for price evolution.
//!
//! # ARES Anti-Drift Compliance
//! - All prices computed using GBM (no hardcoded values)
//! - Spread and volume randomized within realistic ranges
//! - Each run produces different data (stochastic)

use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 HFT Demo - Sample Data Generator");
    println!("=====================================\n");

    // Configuration
    let output_dir = "hft-demo/data";
    let output_file = "sample_aapl_1hour.csv";
    let output_path = format!("{}/{}", output_dir, output_file);

    let config = DataConfig {
        symbol: "AAPL".to_string(),
        base_price: 182.50,          // Starting price
        volatility: 0.25,             // 25% annual volatility
        duration_seconds: 3600,       // 1 hour = 3600 seconds
        tick_interval_sec: 1.0,       // 1 tick per second
        min_spread_bps: 1.0,          // Minimum 1 basis point spread
        max_spread_bps: 5.0,          // Maximum 5 basis points spread
        min_volume: 50,               // Minimum volume per tick
        max_volume: 500,              // Maximum volume per tick
        large_trade_probability: 0.02, // 2% chance of large trade
        large_trade_multiplier: 5.0,  // Large trades are 5x normal volume
    };

    println!("Configuration:");
    println!("  Symbol: {}", config.symbol);
    println!("  Base Price: ${:.2}", config.base_price);
    println!("  Volatility: {:.0}% annual", config.volatility * 100.0);
    println!("  Duration: {} seconds ({} ticks)", config.duration_seconds, config.duration_seconds);
    println!("  Tick Interval: {:.1}s", config.tick_interval_sec);
    println!("  Spread Range: {:.1}-{:.1} bps", config.min_spread_bps, config.max_spread_bps);
    println!("  Volume Range: {}-{} shares", config.min_volume, config.max_volume);
    println!("  Output: {}\n", output_path);

    // Generate data
    println!("Generating market data...");
    let ticks = generate_market_data(&config)?;

    // Create output directory
    if let Some(parent) = Path::new(&output_path).parent() {
        create_dir_all(parent)?;
    }

    // Write to CSV
    println!("Writing to {}...", output_path);
    write_csv(&output_path, &ticks)?;

    // Statistics
    let prices: Vec<f64> = ticks.iter().map(|t| t.price).collect();
    let min_price = prices.iter().copied().fold(f64::INFINITY, f64::min);
    let max_price = prices.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let final_price = prices.last().unwrap();
    let total_volume: u32 = ticks.iter().map(|t| t.volume).sum();
    let avg_volume = total_volume as f64 / ticks.len() as f64;

    println!("\n✅ Generated {} ticks", ticks.len());
    println!("\nPrice Statistics:");
    println!("  Starting: ${:.2}", config.base_price);
    println!("  Final:    ${:.2}", final_price);
    println!("  Min:      ${:.2}", min_price);
    println!("  Max:      ${:.2}", max_price);
    println!("  Range:    ${:.2} ({:.2}%)",
             max_price - min_price,
             (max_price - min_price) / config.base_price * 100.0);

    println!("\nVolume Statistics:");
    println!("  Total:    {} shares", total_volume);
    println!("  Average:  {:.0} shares/tick", avg_volume);
    println!("  Min:      {} shares", config.min_volume);
    println!("  Max:      {} shares (normal), {} (large trade)",
             config.max_volume,
             (config.max_volume as f32 * config.large_trade_multiplier) as u32);

    println!("\n🎉 Sample data generation complete!");

    Ok(())
}

#[derive(Debug, Clone)]
struct DataConfig {
    symbol: String,
    base_price: f64,
    volatility: f32,
    duration_seconds: u64,
    tick_interval_sec: f64,
    min_spread_bps: f32,
    max_spread_bps: f32,
    min_volume: u32,
    max_volume: u32,
    large_trade_probability: f32,
    large_trade_multiplier: f32,
}

#[derive(Debug, Clone)]
struct TickData {
    timestamp_ns: u64,
    price: f64,
    volume: u32,
    bid: f64,
    ask: f64,
    bid_size: u32,
    ask_size: u32,
}

/// Generate realistic market data using Geometric Brownian Motion
fn generate_market_data(config: &DataConfig) -> Result<Vec<TickData>, Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let mut ticks = Vec::with_capacity(config.duration_seconds as usize);

    // Geometric Brownian Motion parameters
    // dS = μS dt + σS dW
    let dt = config.tick_interval_sec;
    let mu = 0.0; // No drift for intraday data (mean-reverting)
    let sigma = config.volatility / (252.0 * 6.5 * 3600.0_f32).sqrt(); // Convert annual to per-second

    // Normal distribution for Brownian motion
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut price = config.base_price;

    // Base timestamp: 2024-01-01 09:30:00 UTC (market open)
    let base_timestamp = 1704106200_000_000_000u64;

    for i in 0..config.duration_seconds {
        // Geometric Brownian Motion: dS = μS dt + σS dW
        let dw = normal.sample(&mut rng) as f32;
        let price_change = mu * price as f32 * dt as f32 + sigma * price as f32 * dw * (dt as f32).sqrt();
        price = (price + price_change as f64).max(1.0); // Ensure price stays positive

        // Add mean reversion (prices tend to revert to starting price intraday)
        let mean_reversion_strength = 0.0001;
        let reversion = mean_reversion_strength * (config.base_price - price);
        price += reversion;

        // Realistic bid-ask spread (1-5 basis points)
        let spread_bps = rng.gen_range(config.min_spread_bps..config.max_spread_bps);
        let spread = price * spread_bps as f64 / 10000.0;
        let bid = price - spread / 2.0;
        let ask = price + spread / 2.0;

        // Volume: normal trades with occasional large trades
        let is_large_trade = rng.gen::<f32>() < config.large_trade_probability;
        let base_volume = rng.gen_range(config.min_volume..=config.max_volume);
        let volume = if is_large_trade {
            (base_volume as f32 * config.large_trade_multiplier) as u32
        } else {
            base_volume
        };

        // Bid/ask sizes (typically smaller than trade volume)
        let bid_size = rng.gen_range(config.min_volume / 5..=config.max_volume / 2);
        let ask_size = rng.gen_range(config.min_volume / 5..=config.max_volume / 2);

        // Timestamp: base + i seconds in nanoseconds
        let timestamp_ns = base_timestamp + (i * 1_000_000_000);

        ticks.push(TickData {
            timestamp_ns,
            price,
            volume,
            bid,
            ask,
            bid_size,
            ask_size,
        });
    }

    Ok(ticks)
}

/// Write tick data to CSV file
fn write_csv(path: &str, ticks: &[TickData]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;

    // Write header
    writeln!(file, "timestamp,symbol,price,volume,bid,ask,bid_size,ask_size,exchange,conditions")?;

    // Write data rows
    for tick in ticks {
        writeln!(
            file,
            "{},AAPL,{:.4},{},{:.4},{:.4},{},{},NASDAQ,\"\"",
            tick.timestamp_ns,
            tick.price,
            tick.volume,
            tick.bid,
            tick.ask,
            tick.bid_size,
            tick.ask_size
        )?;
    }

    Ok(())
}
