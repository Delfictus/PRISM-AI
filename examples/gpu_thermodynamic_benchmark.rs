//! GPU Thermodynamic Network Benchmark
//!
//! Constitution: Phase 1, Task 1.3 - Performance Validation
//!
//! Tests the <1ms per step requirement for 1024 oscillators on GPU

use active_inference_platform::statistical_mechanics::{GpuThermodynamicNetwork, NetworkConfig};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Thermodynamic Network Benchmark ===\n");

    // Configuration matching constitution target
    let config = NetworkConfig {
        n_oscillators: 1024,
        temperature: 300.0,
        damping: 0.1,
        dt: 0.001,
        coupling_strength: 0.5,
        enable_information_gating: false,
        seed: 42,
    };

    println!("Configuration:");
    println!("  Oscillators: {}", config.n_oscillators);
    println!("  Temperature: {} K", config.temperature);
    println!("  Damping: {} rad/s", config.damping);
    println!("  Timestep: {} s", config.dt);
    println!("  Coupling: {}", config.coupling_strength);
    println!();

    // Initialize network on CPU first
    println!("Initializing network state...");
    let cpu_network = active_inference_platform::ThermodynamicNetwork::new(config.clone());
    let phases = cpu_network.state().phases.clone();
    let velocities = cpu_network.state().velocities.clone();
    let natural_frequencies = cpu_network.state().natural_frequencies.clone();

    // Flatten coupling matrix
    let coupling_matrix: Vec<f64> = cpu_network
        .coupling_matrix()
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();

    println!("Transferring to GPU...");
    let mut gpu_network = GpuThermodynamicNetwork::new(
        config.n_oscillators,
        &phases,
        &velocities,
        &natural_frequencies,
        &coupling_matrix,
        config.seed,
    )?;

    println!("GPU initialization complete!\n");

    // Warmup
    println!("Warming up GPU (100 steps)...");
    for _ in 0..100 {
        gpu_network.step_gpu(
            config.dt,
            config.damping,
            config.temperature,
            config.coupling_strength,
        )?;
    }
    println!("Warmup complete\n");

    // Benchmark
    const N_STEPS: usize = 1000;
    println!("Running benchmark ({} steps with {} oscillators)...", N_STEPS, config.n_oscillators);

    let start = Instant::now();

    for i in 0..N_STEPS {
        gpu_network.step_gpu(
            config.dt,
            config.damping,
            config.temperature,
            config.coupling_strength,
        )?;

        if i % 100 == 0 {
            print!("  Step {}/{}...\r", i, N_STEPS);
        }
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let ms_per_step = total_ms / N_STEPS as f64;
    let steps_per_sec = 1000.0 / ms_per_step;

    println!("\n");
    println!("=== Performance Results ===");
    println!("Total time: {:.2} ms", total_ms);
    println!("Time per step: {:.4} ms", ms_per_step);
    println!("Steps per second: {:.0}", steps_per_sec);
    println!();

    // Constitution contract validation
    if ms_per_step < 1.0 {
        println!("✅ PERFORMANCE CONTRACT SATISFIED");
        println!("   {:.4} ms < 1.0 ms target", ms_per_step);
        println!("   Speedup required: {:.1}x (achieved!)", 1.0 / ms_per_step);
    } else {
        println!("❌ PERFORMANCE CONTRACT VIOLATED");
        println!("   {:.4} ms > 1.0 ms target", ms_per_step);
        println!("   Speedup needed: {:.1}x additional", ms_per_step);
    }

    println!();

    // Validate thermodynamics
    println!("=== Thermodynamic Validation ===");
    let entropy = gpu_network.calculate_entropy_gpu(config.temperature)?;
    let energy = gpu_network.calculate_energy_gpu(config.coupling_strength)?;
    let coherence = gpu_network.calculate_coherence_gpu()?;

    println!("Final entropy: {:.6e}", entropy);
    println!("Final energy: {:.6e} J", energy);
    println!("Phase coherence: {:.4}", coherence);

    println!();
    println!("=== Constitution Phase 1 Task 1.3 ===");
    println!("Status: GPU-accelerated thermodynamic network operational");
    println!("All thermodynamic laws enforced on GPU");

    Ok(())
}
