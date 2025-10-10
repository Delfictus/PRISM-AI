//! Integration tests for HFT Demo
//!
//! Tests the full pipeline: data generation → loading → validation

use hft_demo::market_data::CsvDataLoader;

#[test]
fn test_load_generated_sample_data() {
    // Test loading the generated sample data file
    let loader = CsvDataLoader::new(
        "data/sample_aapl_1hour.csv".to_string(),
        "AAPL".to_string(),
    );

    let ticks = loader.load_all()
        .expect("Should load sample data file");

    // ARES ANTI-DRIFT: Verify actual data loaded
    assert_eq!(ticks.len(), 3600, "Should have 3600 ticks (1 hour at 1 tick/sec)");

    // Verify first tick
    assert_eq!(ticks[0].symbol, "AAPL");
    assert_eq!(ticks[0].exchange, "NASDAQ");
    assert!(ticks[0].price > 0.0);
    assert!(ticks[0].volume > 0);

    // Verify bid < ask for all ticks
    for (i, tick) in ticks.iter().enumerate() {
        assert!(tick.bid < tick.ask,
                "Tick {}: bid ({}) should be < ask ({})",
                i, tick.bid, tick.ask);
        assert!(tick.price > 0.0, "Tick {}: price should be positive", i);
        assert!(tick.volume > 0, "Tick {}: volume should be positive", i);
    }

    // Verify timestamps are sequential
    for i in 1..ticks.len() {
        assert!(ticks[i].timestamp_ns > ticks[i-1].timestamp_ns,
                "Timestamps should be strictly increasing");
    }

    // Verify realistic price range (shouldn't vary wildly)
    let prices: Vec<f64> = ticks.iter().map(|t| t.price).collect();
    let min_price = prices.iter().copied().fold(f64::INFINITY, f64::min);
    let max_price = prices.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let price_range_pct = (max_price - min_price) / min_price * 100.0;

    assert!(price_range_pct < 10.0,
            "Intraday price range should be < 10% (got {:.2}%)",
            price_range_pct);

    println!("✅ Loaded {} ticks from sample data", ticks.len());
    println!("   Price range: ${:.2} - ${:.2} ({:.2}%)",
             min_price, max_price, price_range_pct);
}

#[test]
fn test_sample_data_anti_drift() {
    // ARES ANTI-DRIFT: Verify generated data is not hardcoded
    // Load the sample data twice to verify it contains real data
    let loader = CsvDataLoader::new(
        "data/sample_aapl_1hour.csv".to_string(),
        "AAPL".to_string(),
    );

    let ticks1 = loader.load_all().expect("Should load data");
    let ticks2 = loader.load_all().expect("Should load data again");

    // Same file should load identical data
    assert_eq!(ticks1.len(), ticks2.len());
    assert_eq!(ticks1[0].price, ticks2[0].price);
    assert_eq!(ticks1[100].price, ticks2[100].price);

    // But prices should vary within the file (not all the same)
    let unique_prices: std::collections::HashSet<String> = ticks1.iter()
        .map(|t| format!("{:.4}", t.price))
        .collect();

    assert!(unique_prices.len() > 100,
            "Should have many unique prices (got {}), not hardcoded",
            unique_prices.len());

    println!("✅ Anti-drift check passed: {} unique prices", unique_prices.len());
}
