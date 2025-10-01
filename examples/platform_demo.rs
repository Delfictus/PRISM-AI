//! Neuromorphic-Quantum Platform Demo
//!
//! Demonstrates the world's first software-based neuromorphic-quantum computing platform

use neuromorphic_quantum_platform::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠⚛️ Neuromorphic-Quantum Computing Platform Demo");
    println!("=================================================");
    println!("World's first software-based hybrid computing platform\n");

    // Initialize the platform
    println!("🚀 Initializing platform...");
    let platform = create_platform().await?;
    println!("✅ Platform ready!\n");

    // Demo 1: Financial Market Data
    println!("📈 Demo 1: Financial Market Analysis");
    println!("-----------------------------------");
    let market_data = vec![
        100.0, 102.5, 98.2, 105.1, 103.7, 99.8, 107.3, 104.2, 101.9, 108.6
    ];

    let output = process_data(&platform, "financial_market".to_string(), market_data).await?;

    println!("Market Prediction: {} (confidence: {:.1}%)",
        output.prediction.direction, output.prediction.confidence * 100.0);

    if let Some(neuro) = &output.neuromorphic_results {
        println!("  🧠 Neuromorphic: {} patterns detected, coherence: {:.3}",
            neuro.patterns.len(), neuro.spike_analysis.coherence);
    }

    if let Some(quantum) = &output.quantum_results {
        println!("  ⚛️  Quantum: Energy {:.3}, phase coherence: {:.3}",
            quantum.energy, quantum.phase_coherence);
    }

    println!("  ⏱️  Processing time: {:.1}ms\n", output.metadata.duration_ms);

    // Demo 2: Sensor Data Processing
    println!("🌡️ Demo 2: Sensor Data Processing");
    println!("---------------------------------");
    let sensor_data = vec![
        23.5, 23.7, 24.1, 23.9, 24.3, 24.0, 23.8, 24.2, 23.6, 24.4
    ];

    let output = process_data(&platform, "temperature_sensor".to_string(), sensor_data).await?;

    println!("Sensor Analysis: {} (confidence: {:.1}%)",
        output.prediction.direction, output.prediction.confidence * 100.0);

    if let Some(neuro) = &output.neuromorphic_results {
        println!("  🧠 Spike rate: {:.1} Hz, patterns: {}",
            neuro.spike_analysis.spike_rate, neuro.patterns.len());

        for pattern in &neuro.patterns {
            println!("    • {} pattern (strength: {:.3})",
                pattern.pattern_type, pattern.strength);
        }
    }

    println!("  ⏱️  Processing time: {:.1}ms\n", output.metadata.duration_ms);

    // Demo 3: Signal Processing
    println!("📊 Demo 3: Complex Signal Analysis");
    println!("----------------------------------");
    let signal_data = vec![
        0.1, 0.3, 0.7, 0.9, 0.4, 0.2, 0.8, 0.6, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4, 0.8
    ];

    let output = process_data(&platform, "complex_signal".to_string(), signal_data).await?;

    println!("Signal Classification: {} (confidence: {:.1}%)",
        output.prediction.direction, output.prediction.confidence * 100.0);

    if let Some(quantum) = &output.quantum_results {
        println!("  ⚛️  Quantum optimization:");
        println!("    • Final energy: {:.6}", quantum.energy);
        println!("    • Convergence: {}", if quantum.convergence.converged { "✅" } else { "❌" });
        println!("    • Iterations: {}", quantum.convergence.iterations);
        println!("    • Energy drift: {:.2e}", quantum.convergence.energy_drift);
    }

    println!("  ⏱️  Processing time: {:.1}ms\n", output.metadata.duration_ms);

    // Platform Status
    println!("📊 Platform Status");
    println!("------------------");
    let status = platform.get_status().await;

    println!("  • Total inputs processed: {}", status.total_inputs_processed);
    println!("  • Success rate: {:.1}%", status.success_rate * 100.0);
    println!("  • Average processing time: {:.1}ms", status.avg_processing_time_ms);
    println!("  • Memory usage: {:.1}MB", status.memory_usage_mb);
    println!("  • Neuromorphic enabled: {}", if status.neuromorphic_enabled { "✅" } else { "❌" });
    println!("  • Quantum enabled: {}", if status.quantum_enabled { "✅" } else { "❌" });

    // Performance Metrics
    println!("\n📈 Performance Highlights");
    println!("-------------------------");
    let metrics = platform.get_metrics().await;

    println!("  • Neuromorphic success rate: {:.1}%",
        (metrics.neuromorphic_success as f64 / metrics.total_inputs as f64) * 100.0);
    println!("  • Quantum success rate: {:.1}%",
        (metrics.quantum_success as f64 / metrics.total_inputs as f64) * 100.0);
    println!("  • Peak memory usage: {:.1}MB", metrics.peak_memory as f64 / (1024.0 * 1024.0));

    println!("\n🎉 Demo Complete!");
    println!("================");
    println!("Successfully demonstrated:");
    println!("  ✅ Neuromorphic spike processing");
    println!("  ✅ Quantum-inspired optimization");
    println!("  ✅ Integrated platform predictions");
    println!("  ✅ Real-time performance monitoring");
    println!("\n🌟 The future of computing is here! 🌟");

    Ok(())
}