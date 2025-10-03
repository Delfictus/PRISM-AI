//! Fixed test for Phase 3 Task 3.2 with proper dimensions
//!
//! This version uses 900 dimensions (30×30) to match HierarchicalModel requirements

use active_inference_platform::integration::{UnifiedPlatform, PlatformInput};
use ndarray::Array1;
use std::time::Instant;

fn main() {
    println!("Phase 3 Task 3.2 Fixed Validation\n");

    // Use 900 dimensions to match HierarchicalModel requirements (30x30 windows)
    const N_DIMS: usize = 900;

    // Create platform with correct dimensions
    let mut platform = UnifiedPlatform::new(N_DIMS).unwrap();
    platform.initialize();

    println!("✓ Platform initialized with {} dimensions", N_DIMS);

    // Create properly sized input
    let input = PlatformInput::new(
        Array1::from_vec((0..N_DIMS).map(|i| ((i as f64) * 0.01).sin() + 0.5).collect()),
        Array1::from_vec((0..N_DIMS).map(|i| 0.5 + ((i as f64) * 0.02).cos() * 0.3).collect()),
        0.01
    );

    println!("✓ Input created with matching dimensions\n");

    // Test single execution
    let start = Instant::now();
    match platform.process(input.clone()) {
        Ok(output) => {
            println!("✅ Pipeline executed successfully!");
            println!("  Total latency: {:.2} ms", output.metrics.total_latency_ms);
            println!("  Entropy production: {:.6}", output.metrics.entropy_production);
            println!("  Free energy: {:.4}", output.metrics.free_energy);
            println!("  Mutual information: {:.4} bits", output.metrics.mutual_information);
            println!("  Phase coherence: {:.3}", output.metrics.phase_coherence);
            println!();

            // Phase breakdown
            println!("Phase Latencies:");
            let phase_names = [
                "Neuromorphic", "Info Flow", "Coupling", "Thermodynamic",
                "Quantum", "Active Inference", "Control", "Synchronization"
            ];

            for (i, (name, &latency)) in phase_names.iter()
                .zip(output.metrics.phase_latencies.iter())
                .enumerate()
            {
                println!("  {}. {}: {:.3} ms", i+1, name, latency);
            }

            let sum: f64 = output.metrics.phase_latencies.iter().sum();
            println!("  Sum of phases: {:.3} ms", sum);
            println!("  Overhead: {:.3} ms\n", output.metrics.total_latency_ms - sum);

            // Check requirements
            let latency_ok = output.metrics.total_latency_ms < 10.0;
            let thermo_ok = output.metrics.entropy_production >= 0.0;

            println!("Constitution Requirements:");
            println!("  [{}] Latency < 10ms: {:.2} ms",
                if latency_ok { "✓" } else { "✗" },
                output.metrics.total_latency_ms);
            println!("  [{}] Thermodynamic consistency: dS/dt = {:.6} ≥ 0",
                if thermo_ok { "✓" } else { "✗" },
                output.metrics.entropy_production);

            if latency_ok && thermo_ok {
                println!("\n✅ ALL VALIDATION CRITERIA MET");
                println!("Phase 3 Task 3.2: VALIDATED");
            } else {
                println!("\n⚠️ Some criteria not met");
            }
        }
        Err(e) => {
            println!("❌ Pipeline failed: {}", e);
            println!("Error details: {:?}", e);
        }
    }

    // Performance benchmark with correct dimensions
    println!("\n═══════════════════════════════════════");
    println!("Performance Benchmark (10 iterations)");
    println!("═══════════════════════════════════════\n");

    let mut latencies = Vec::new();
    let mut successes = 0;

    for i in 0..10 {
        // Vary input slightly each iteration
        let varied_input = PlatformInput::new(
            Array1::from_vec((0..N_DIMS)
                .map(|j| ((j + i) as f64 * 0.015).cos() + 0.5)
                .collect()),
            Array1::from_vec((0..N_DIMS)
                .map(|j| 0.5 + ((j + i) as f64 * 0.025).sin() * 0.3)
                .collect()),
            0.01
        );

        match platform.process(varied_input) {
            Ok(output) => {
                successes += 1;
                latencies.push(output.metrics.total_latency_ms);
                println!("  Iteration {}: {:.2} ms - {}",
                    i + 1,
                    output.metrics.total_latency_ms,
                    if output.metrics.total_latency_ms < 10.0 { "✓" } else { "✗" });
            }
            Err(e) => {
                println!("  Iteration {}: Failed - {}", i + 1, e);
            }
        }
    }

    if !latencies.is_empty() {
        let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let min = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = latencies.iter().fold(0.0f64, |a, &b| a.max(b));

        println!("\nBenchmark Results:");
        println!("  Success rate: {}/10", successes);
        println!("  Average latency: {:.2} ms", avg);
        println!("  Min latency: {:.2} ms", min);
        println!("  Max latency: {:.2} ms", max);

        if avg < 10.0 && successes == 10 {
            println!("\n✅ PERFORMANCE VALIDATION PASSED");
        }
    }

    println!("\n════════════════════════════════════════");
}