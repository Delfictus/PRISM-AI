//! Real-Time Data Ingestion Streaming Demo
//!
//! Demonstrates the complete ingestion pipeline with multiple data sources

use platform_foundation::{
    IngestionEngine, SyntheticDataSource,
};
use anyhow::Result;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🌊 Real-Time Data Ingestion Streaming Demo");
    println!("==========================================\n");

    // Create ingestion engine
    let mut engine = IngestionEngine::new(
        1000,   // Channel buffer size
        10000,  // History buffer size
    );

    println!("📡 Starting data sources...\n");

    // Source 1: Sine wave (simulating sensor data)
    let source1 = Box::new(SyntheticDataSource::sine_wave(10, 2.0));
    engine.start_source(source1).await?;
    println!("  ✅ Started sine wave source (10 dimensions, 2 Hz)");

    // Source 2: Random walk (simulating market data)
    let source2 = Box::new(SyntheticDataSource::random_walk(5));
    engine.start_source(source2).await?;
    println!("  ✅ Started random walk source (5 dimensions)");

    // Source 3: Gaussian noise (simulating sensor noise)
    let source3 = Box::new(SyntheticDataSource::gaussian(3));
    engine.start_source(source3).await?;
    println!("  ✅ Started Gaussian noise source (3 dimensions)\n");

    println!("⏱️  Collecting data for 3 seconds...\n");
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Test 1: Get batch with timeout
    println!("📊 Test 1: Batch Ingestion");
    println!("──────────────────────────");

    let start = std::time::Instant::now();
    let batch = engine.get_batch(20, Duration::from_millis(500)).await?;
    let latency = start.elapsed();

    println!("  • Batch size: {} points", batch.len());
    println!("  • Latency: {:.2}ms", latency.as_micros() as f64 / 1000.0);
    println!("  • First point dimensions: {}", batch[0].dimension());
    println!("  • Data sources in batch:");

    let mut source_counts = std::collections::HashMap::new();
    for point in &batch {
        let source = point.metadata.get("source").map(|s| s.as_str()).unwrap_or("unknown");
        *source_counts.entry(source).or_insert(0) += 1;
    }

    for (source, count) in &source_counts {
        println!("    - {}: {} points", source, count);
    }

    // Test 2: Streaming processing
    println!("\n📈 Test 2: Streaming Processing");
    println!("────────────────────────────────");

    for i in 0..5 {
        let start = std::time::Instant::now();
        let batch = engine.get_batch(10, Duration::from_millis(200)).await?;
        let latency = start.elapsed();

        println!(
            "  Batch {} | Size: {:2} | Latency: {:5.2}ms | Buffer: {} points",
            i + 1,
            batch.len(),
            latency.as_micros() as f64 / 1000.0,
            engine.buffer_size().await
        );

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Test 3: Statistics
    println!("\n📊 Test 3: Ingestion Statistics");
    println!("────────────────────────────────");

    let stats = engine.get_stats().await;
    println!("  • Total points ingested: {}", stats.total_points);
    println!("  • Active sources: {}", stats.active_sources);
    println!("  • Average rate: {:.1} points/sec", stats.average_rate_hz);
    println!("  • Error count: {}", stats.error_count);

    // Test 4: Historical data
    println!("\n🕐 Test 4: Historical Data Buffer");
    println!("──────────────────────────────────");

    let history = engine.get_history(50).await;
    println!("  • Historical points available: {}", history.len());

    if !history.is_empty() {
        println!("  • Oldest timestamp: {}", history.first().unwrap().timestamp);
        println!("  • Newest timestamp: {}", history.last().unwrap().timestamp);

        let time_span = history.last().unwrap().timestamp - history.first().unwrap().timestamp;
        println!("  • Time span: {}ms", time_span);
    }

    // Test 5: Latency validation (<10ms target)
    println!("\n⚡ Test 5: Latency Validation (<10ms target)");
    println!("─────────────────────────────────────────────");

    let mut latencies = Vec::new();
    for _ in 0..20 {
        let start = std::time::Instant::now();
        let _ = engine.get_batch(5, Duration::from_millis(50)).await?;
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64 / 1000.0);
    }

    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let min_latency = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_latency = latencies.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("  • Average latency: {:.2}ms", avg_latency);
    println!("  • Min latency: {:.2}ms", min_latency);
    println!("  • Max latency: {:.2}ms", max_latency);

    if avg_latency < 10.0 {
        println!("  ✅ PASSED: Average latency < 10ms target");
    } else {
        println!("  ⚠️  WARNING: Average latency exceeds 10ms target");
    }

    // Final statistics
    println!("\n📈 Final Statistics");
    println!("───────────────────");

    let final_stats = engine.get_stats().await;
    println!("  • Total points processed: {}", final_stats.total_points);
    println!("  • Processing rate: {:.1} points/sec", final_stats.average_rate_hz);
    println!("  • Buffer utilization: {} points", engine.buffer_size().await);

    println!("\n✅ Real-Time Data Ingestion Demo Complete!");
    println!("════════════════════════════════════════════\n");

    Ok(())
}
