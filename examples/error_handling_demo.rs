//! Error Handling and Recovery Demonstration
//!
//! Demonstrates comprehensive error handling, retry logic, and circuit breakers

use platform_foundation::{
    CircuitBreaker, IngestionEngine, IngestionError, RetryPolicy, SyntheticDataSource,
};
use anyhow::Result;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🛡️  Error Handling and Recovery Demo");
    println!("=====================================\n");

    // Test 1: Retry Policy
    println!("📊 Test 1: Retry Policy Configuration");
    println!("──────────────────────────────────────");

    let default_policy = RetryPolicy::default();
    println!("  Default Policy:");
    println!("    • Max attempts: {}", default_policy.max_attempts);
    println!(
        "    • Initial backoff: {}ms",
        default_policy.initial_backoff_ms
    );
    println!("    • Max backoff: {}ms", default_policy.max_backoff_ms);
    println!("    • Multiplier: {}x", default_policy.backoff_multiplier);

    println!("\n  Backoff delays for each attempt:");
    for attempt in 0..5 {
        println!(
            "    Attempt {}: {}ms",
            attempt + 1,
            default_policy.backoff_delay(attempt)
        );
    }

    let aggressive_policy = RetryPolicy::aggressive();
    println!("\n  Aggressive Policy:");
    println!("    • Max attempts: {}", aggressive_policy.max_attempts);
    println!(
        "    • Initial backoff: {}ms",
        aggressive_policy.initial_backoff_ms
    );
    println!("    • Max backoff: {}ms", aggressive_policy.max_backoff_ms);

    // Test 2: Circuit Breaker
    println!("\n📊 Test 2: Circuit Breaker Behavior");
    println!("────────────────────────────────────");

    let mut cb = CircuitBreaker::new(3, 1000);
    println!("  Created circuit breaker (threshold: 3 errors, timeout: 1000ms)");
    println!("  Initial state: {:?}", cb.state());

    println!("\n  Recording failures:");
    for i in 1..=4 {
        cb.record_failure();
        println!(
            "    Failure {} | State: {:?} | Errors: {}",
            i,
            cb.state(),
            cb.error_count()
        );
    }

    println!("\n  Testing circuit breaker status:");
    println!("    • is_closed(): {}", cb.is_closed());

    println!("\n  Recording success to reset:");
    cb.record_success();
    println!("    • State: {:?}", cb.state());
    println!("    • is_closed(): {}", cb.is_closed());
    println!("    • Errors: {}", cb.error_count());

    // Test 3: Ingestion Engine with Error Handling
    println!("\n📊 Test 3: Ingestion Engine with Recovery");
    println!("──────────────────────────────────────────");

    let custom_policy = RetryPolicy {
        max_attempts: 5,
        initial_backoff_ms: 50,
        max_backoff_ms: 2000,
        backoff_multiplier: 2.0,
    };

    let mut engine = IngestionEngine::with_retry_policy(1000, 10000, custom_policy);
    println!("  ✅ Created engine with custom retry policy");

    // Start reliable sources
    let source1 = Box::new(SyntheticDataSource::sine_wave(5, 1.0));
    engine.start_source(source1).await?;
    println!("  ✅ Started sine wave source");

    let source2 = Box::new(SyntheticDataSource::gaussian(3));
    engine.start_source(source2).await?;
    println!("  ✅ Started Gaussian source");

    // Collect data
    println!("\n  ⏱️  Collecting data for 2 seconds...");
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Get statistics
    let stats = engine.get_stats().await;
    println!("\n  Statistics:");
    println!("    • Total points: {}", stats.total_points);
    println!("    • Active sources: {}", stats.active_sources);
    println!("    • Error count: {}", stats.error_count);
    println!("    • Retry successes: {}", stats.retry_success_count);
    println!("    • Retry failures: {}", stats.retry_failed_count);
    println!("    • Average rate: {:.1} points/sec", stats.average_rate_hz);

    println!("\n  Circuit Breaker States:");
    for (source, state) in &stats.circuit_breaker_states {
        println!("    • {}: {}", source, state);
    }

    // Test 4: IngestionError Types
    println!("\n📊 Test 4: Error Types and Handling");
    println!("────────────────────────────────────");

    let errors = vec![
        IngestionError::ConnectionFailed {
            source: "test-source".to_string(),
            reason: "timeout".to_string(),
            retryable: true,
        },
        IngestionError::ReadFailed {
            source: "test-source".to_string(),
            reason: "network error".to_string(),
            retryable: true,
        },
        IngestionError::ParseError {
            source: "test-source".to_string(),
            reason: "invalid JSON".to_string(),
        },
        IngestionError::CircuitBreakerOpen {
            source: "test-source".to_string(),
            error_count: 5,
            threshold: 3,
        },
        IngestionError::Timeout {
            source: "test-source".to_string(),
            timeout_ms: 5000,
        },
    ];

    println!("  Error Types:");
    for error in errors {
        println!("    • {}", error);
        println!("      Retryable: {}", error.is_retryable());
    }

    // Test 5: Real-world scenario simulation
    println!("\n📊 Test 5: Production Scenario Simulation");
    println!("──────────────────────────────────────────");

    println!("  Simulating high-load scenario with 3 sources...");

    let mut production_engine = IngestionEngine::new(2000, 50000);

    for i in 1..=3 {
        let source = Box::new(SyntheticDataSource::sine_wave(10, 2.0));
        production_engine.start_source(source).await?;
        println!("  ✅ Started production source {}", i);
    }

    println!("\n  Running for 3 seconds...");
    tokio::time::sleep(Duration::from_secs(3)).await;

    let final_stats = production_engine.get_stats().await;
    println!("\n  Production Statistics:");
    println!("    • Total points: {}", final_stats.total_points);
    println!("    • Throughput: {:.1} points/sec", final_stats.average_rate_hz);
    println!("    • Error rate: {:.2}%",
        (final_stats.error_count as f64 / final_stats.total_points.max(1) as f64) * 100.0
    );
    println!("    • Success rate: {:.2}%",
        100.0 - (final_stats.error_count as f64 / final_stats.total_points.max(1) as f64) * 100.0
    );

    // Get circuit breaker status for each source
    for (source, _) in &final_stats.circuit_breaker_states {
        if let Some(status) = production_engine.get_circuit_breaker_status(source).await {
            println!("    • Circuit breaker [{}]: {}", source, status);
        }
    }

    println!("\n✅ Error Handling Demo Complete!");
    println!("═════════════════════════════════\n");

    println!("Summary:");
    println!("  ✅ Retry policies with exponential backoff");
    println!("  ✅ Circuit breakers with automatic recovery");
    println!("  ✅ Automatic reconnection on failures");
    println!("  ✅ Comprehensive error tracking");
    println!("  ✅ Production-ready fault tolerance\n");

    Ok(())
}
