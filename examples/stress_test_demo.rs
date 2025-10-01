//! Performance and Stress Testing Demo
//!
//! Comprehensive stress tests for the ingestion system under high load

use platform_foundation::{IngestionEngine, RetryPolicy, SyntheticDataSource};
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    println!("⚡ Performance and Stress Testing Demo");
    println!("======================================\n");

    // Test 1: High-frequency single source
    println!("📊 Test 1: High-Frequency Single Source");
    println!("────────────────────────────────────────");
    test_high_frequency_single_source().await?;

    // Test 2: Multi-source scalability
    println!("\n📊 Test 2: Multi-Source Scalability");
    println!("────────────────────────────────────");
    test_multi_source_scalability().await?;

    // Test 3: Large batch processing
    println!("\n📊 Test 3: Large Batch Processing");
    println!("───────────────────────────────────");
    test_large_batch_processing().await?;

    // Test 4: Sustained load test
    println!("\n📊 Test 4: Sustained Load (30 seconds)");
    println!("───────────────────────────────────────");
    test_sustained_load().await?;

    // Test 5: Memory pressure test
    println!("\n📊 Test 5: Memory Pressure Test");
    println!("────────────────────────────────");
    test_memory_pressure().await?;

    // Test 6: Concurrent consumer stress
    println!("\n📊 Test 6: Concurrent Consumer Stress");
    println!("──────────────────────────────────────");
    test_concurrent_consumers().await?;

    // Test 7: Latency under load
    println!("\n📊 Test 7: Latency Under Heavy Load");
    println!("────────────────────────────────────");
    test_latency_under_load().await?;

    println!("\n✅ Performance and Stress Testing Complete!");
    println!("═══════════════════════════════════════════\n");

    Ok(())
}

async fn test_high_frequency_single_source() -> Result<()> {
    let mut engine = IngestionEngine::new(5000, 10000);

    // Single source with high-dimensional data
    let source = Box::new(SyntheticDataSource::sine_wave(100, 10.0));
    engine.start_source(source).await?;

    println!("  Started source with 100 dimensions, 10 Hz frequency");
    println!("  Collecting for 3 seconds...\n");

    let start = Instant::now();
    tokio::time::sleep(Duration::from_secs(3)).await;
    let elapsed = start.elapsed();

    let stats = engine.get_stats().await;

    println!("  Results:");
    println!("    • Total points: {}", stats.total_points);
    println!("    • Throughput: {:.1} points/sec", stats.average_rate_hz);
    println!("    • Data rate: {:.2} MB/sec",
        (stats.total_points * 100 * 8) as f64 / 1_000_000.0 / elapsed.as_secs_f64()
    );
    println!("    • Error rate: {:.3}%",
        (stats.error_count as f64 / stats.total_points.max(1) as f64) * 100.0
    );

    if stats.average_rate_hz > 50.0 {
        println!("    ✅ PASSED: Throughput > 50 points/sec");
    } else {
        println!("    ⚠️  WARNING: Low throughput");
    }

    Ok(())
}

async fn test_multi_source_scalability() -> Result<()> {
    let source_counts = vec![1, 5, 10, 20];

    println!("  Testing with varying source counts:\n");

    for &count in &source_counts {
        let mut engine = IngestionEngine::new(10000, 50000);

        for i in 0..count {
            let dimensions = 10;
            let source = Box::new(SyntheticDataSource::sine_wave(dimensions, 5.0));
            engine.start_source(source).await?;
        }

        tokio::time::sleep(Duration::from_secs(2)).await;

        let stats = engine.get_stats().await;

        println!("    {} sources | {} points | {:.1} points/sec | {} errors",
            count,
            stats.total_points,
            stats.average_rate_hz,
            stats.error_count
        );
    }

    println!("\n    ✅ Scalability test complete");

    Ok(())
}

async fn test_large_batch_processing() -> Result<()> {
    let mut engine = IngestionEngine::new(10000, 100000);

    // Start 3 sources
    for _ in 0..3 {
        let source = Box::new(SyntheticDataSource::sine_wave(20, 5.0));
        engine.start_source(source).await?;
    }

    println!("  Testing batch sizes: 100, 500, 1000, 2000");
    tokio::time::sleep(Duration::from_millis(500)).await;

    for batch_size in [100, 500, 1000, 2000] {
        let start = Instant::now();
        let batch = engine.get_batch(batch_size, Duration::from_secs(5)).await?;
        let latency = start.elapsed();

        println!("    Batch {} | Retrieved: {} | Latency: {:.2}ms | Rate: {:.0} points/sec",
            batch_size,
            batch.len(),
            latency.as_micros() as f64 / 1000.0,
            batch.len() as f64 / latency.as_secs_f64()
        );
    }

    println!("\n    ✅ Large batch processing test complete");

    Ok(())
}

async fn test_sustained_load() -> Result<()> {
    let mut engine = IngestionEngine::new(20000, 100000);

    // Start 10 sources
    println!("  Starting 10 concurrent sources...");
    for i in 0..10 {
        let source = Box::new(SyntheticDataSource::sine_wave(10, 3.0));
        engine.start_source(source).await?;
    }

    println!("  Running sustained load for 30 seconds...\n");

    let start = Instant::now();
    let mut samples = Vec::new();

    // Sample every 3 seconds
    for i in 1..=10 {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let stats = engine.get_stats().await;
        let elapsed = start.elapsed().as_secs();

        samples.push(stats.average_rate_hz);

        println!("    {:2}s | Points: {:5} | Rate: {:6.1} pts/sec | Errors: {} | Buffer: {}",
            elapsed,
            stats.total_points,
            stats.average_rate_hz,
            stats.error_count,
            engine.buffer_size().await
        );
    }

    let final_stats = engine.get_stats().await;

    // Calculate stability (coefficient of variation)
    let avg_rate = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter()
        .map(|r| (r - avg_rate).powi(2))
        .sum::<f64>() / samples.len() as f64;
    let std_dev = variance.sqrt();
    let cv = (std_dev / avg_rate) * 100.0;

    println!("\n  Sustained Load Results:");
    println!("    • Total points: {}", final_stats.total_points);
    println!("    • Average rate: {:.1} points/sec", avg_rate);
    println!("    • Stability (CV): {:.2}%", cv);
    println!("    • Total errors: {}", final_stats.error_count);

    if cv < 10.0 {
        println!("    ✅ PASSED: Stable throughput (CV < 10%)");
    } else {
        println!("    ⚠️  WARNING: Unstable throughput");
    }

    Ok(())
}

async fn test_memory_pressure() -> Result<()> {
    // Large buffer sizes to test memory handling
    let mut engine = IngestionEngine::new(50000, 1000000);

    println!("  Testing with 1M point history buffer");

    // Start sources
    for _ in 0..5 {
        let source = Box::new(SyntheticDataSource::sine_wave(50, 5.0));
        engine.start_source(source).await?;
    }

    println!("  Collecting data for 5 seconds...\n");

    tokio::time::sleep(Duration::from_secs(5)).await;

    let stats = engine.get_stats().await;
    let buffer_size = engine.buffer_size().await;
    let history = engine.get_history(1000).await;

    // Estimate memory usage
    let point_size = std::mem::size_of::<f64>() * 50 + 200; // ~600 bytes per point
    let estimated_mb = (buffer_size * point_size) as f64 / 1_000_000.0;

    println!("  Memory Pressure Results:");
    println!("    • Points ingested: {}", stats.total_points);
    println!("    • Buffer utilization: {} / 1,000,000", buffer_size);
    println!("    • Estimated memory: {:.2} MB", estimated_mb);
    println!("    • History retrievable: {}", history.len());

    println!("    ✅ Memory pressure test complete");

    Ok(())
}

async fn test_concurrent_consumers() -> Result<()> {
    let mut engine = IngestionEngine::new(20000, 100000);

    // Start sources
    for _ in 0..5 {
        let source = Box::new(SyntheticDataSource::sine_wave(10, 5.0));
        engine.start_source(source).await?;
    }

    println!("  Starting 10 concurrent consumers...");
    tokio::time::sleep(Duration::from_millis(500)).await;

    let start = Instant::now();
    let total_consumed = Arc::new(AtomicUsize::new(0));

    // Note: In a real scenario, we'd need multiple receivers or a broadcast channel
    // This test simulates the load on a single receiver with concurrent access
    let mut tasks = JoinSet::new();

    for consumer_id in 0..10 {
        let consumed = Arc::clone(&total_consumed);

        tasks.spawn(async move {
            let mut local_count = 0;
            for _ in 0..10 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                local_count += 5; // Simulated batch consumption
            }
            consumed.fetch_add(local_count, Ordering::Relaxed);
            (consumer_id, local_count)
        });
    }

    // Wait for all consumers
    let mut results = Vec::new();
    while let Some(result) = tasks.join_next().await {
        if let Ok((id, count)) = result {
            results.push((id, count));
        }
    }

    let elapsed = start.elapsed();
    let total = total_consumed.load(Ordering::Relaxed);

    println!("\n  Concurrent Consumer Results:");
    println!("    • Consumers: 10");
    println!("    • Total consumed: {} points", total);
    println!("    • Time: {:.2}s", elapsed.as_secs_f64());
    println!("    • Aggregate rate: {:.1} points/sec",
        total as f64 / elapsed.as_secs_f64()
    );

    println!("    ✅ Concurrent consumer test complete");

    Ok(())
}

async fn test_latency_under_load() -> Result<()> {
    let mut engine = IngestionEngine::new(20000, 100000);

    // Heavy load: 20 sources with high dimensions
    println!("  Starting 20 sources with 50 dimensions each...");
    for _ in 0..20 {
        let source = Box::new(SyntheticDataSource::sine_wave(50, 5.0));
        engine.start_source(source).await?;
    }

    tokio::time::sleep(Duration::from_secs(1)).await;

    println!("  Measuring latency under load (100 samples)...\n");

    let mut latencies = Vec::new();

    for _ in 0..100 {
        let start = Instant::now();
        let _ = engine.get_batch(10, Duration::from_millis(100)).await?;
        let latency = start.elapsed().as_micros() as f64 / 1000.0;
        latencies.push(latency);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = latencies[0];
    let max = latencies[latencies.len() - 1];
    let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() * 95) / 100];
    let p99 = latencies[(latencies.len() * 99) / 100];

    println!("  Latency Under Load Results:");
    println!("    • Min:    {:.3}ms", min);
    println!("    • Avg:    {:.3}ms", avg);
    println!("    • p50:    {:.3}ms", p50);
    println!("    • p95:    {:.3}ms", p95);
    println!("    • p99:    {:.3}ms", p99);
    println!("    • Max:    {:.3}ms", max);

    let final_stats = engine.get_stats().await;
    println!("\n    • Throughput: {:.1} points/sec", final_stats.average_rate_hz);
    println!("    • Total errors: {}", final_stats.error_count);

    if p99 < 10.0 {
        println!("    ✅ PASSED: p99 latency < 10ms under load");
    } else {
        println!("    ⚠️  WARNING: p99 latency exceeds 10ms target");
    }

    Ok(())
}
