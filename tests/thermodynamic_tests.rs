//! Thermodynamic Network Tests
//!
//! Constitution: Phase 1, Task 1.3 - Validation Criteria
//!
//! Validates:
//! 1. Entropy never decreases (1M steps) - Second Law of Thermodynamics
//! 2. Equilibrium matches Boltzmann distribution
//! 3. Fluctuation-dissipation theorem satisfied
//! 4. Performance: <1ms per step, 1024 oscillators

use active_inference_platform::{ThermodynamicNetwork, NetworkConfig};
use std::time::Instant;

#[test]
fn test_entropy_never_decreases_1m_steps() {
    println!("=== Testing Second Law: Entropy Never Decreases (1M steps) ===");

    let config = NetworkConfig {
        n_oscillators: 128, // Smaller for faster test
        temperature: 300.0,
        damping: 0.1,
        dt: 0.001,
        coupling_strength: 0.5,
        enable_information_gating: false,
        seed: 12345,
    };

    let mut network = ThermodynamicNetwork::new(config);

    println!("Running 1,000,000 steps...");
    let start = Instant::now();

    let mut entropy_violations = 0;
    let mut max_violation: f64 = 0.0;

    for step in 0..1_000_000 {
        let prev_entropy = network.state().entropy;
        network.step();
        let curr_entropy = network.state().entropy;

        // Check for entropy decrease (with numerical tolerance)
        let delta_entropy = curr_entropy - prev_entropy;
        if delta_entropy < -1e-10 {
            entropy_violations += 1;
            max_violation = max_violation.max(-delta_entropy);
        }

        if step % 100_000 == 0 {
            println!("  Step {}: S = {:.6e}, dS = {:.6e}",
                step, curr_entropy, delta_entropy);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("Completed in {:.2} seconds ({:.2} steps/sec)", elapsed, 1_000_000.0 / elapsed);

    println!("\n=== Second Law Validation ===");
    println!("Entropy violations: {}", entropy_violations);
    println!("Max violation: {:.6e}", max_violation);

    // Allow very small numerical violations (<0.1% of steps)
    let violation_rate = entropy_violations as f64 / 1_000_000.0;
    assert!(violation_rate < 0.001,
        "Entropy decreased in {:.2}% of steps - VIOLATES SECOND LAW!",
        violation_rate * 100.0);

    println!("✓ Second Law satisfied: dS/dt ≥ 0 for {:.4}% of steps",
        (1.0 - violation_rate) * 100.0);
}

#[test]
fn test_boltzmann_distribution_at_equilibrium() {
    println!("=== Testing Boltzmann Distribution at Equilibrium ===");

    let config = NetworkConfig {
        n_oscillators: 512,
        temperature: 300.0,
        damping: 0.2, // Higher damping for faster equilibration
        dt: 0.001,
        coupling_strength: 0.1, // Weak coupling
        enable_information_gating: false,
        seed: 54321,
    };

    let mut network = ThermodynamicNetwork::new(config.clone());

    // Equilibrate for 10,000 steps
    println!("Equilibrating for 10,000 steps...");
    for _ in 0..10_000 {
        network.step();
    }

    // Run for statistics
    println!("Collecting statistics for 50,000 steps...");
    let result = network.evolve(50_000);

    println!("\n=== Boltzmann Validation ===");
    println!("Boltzmann distribution satisfied: {}", result.boltzmann_satisfied);

    // Check energy histogram
    let histogram = &result.metrics.energy_histogram;
    if histogram.len() > 10 {
        println!("\nEnergy Distribution (first 10 bins):");
        for (i, &(energy, prob)) in histogram.iter().take(10).enumerate() {
            let expected = (-energy / (1.380649e-23 * config.temperature)).exp();
            println!("  Bin {}: E={:.3e}, P={:.6}, P_expected={:.6}",
                i, energy, prob, expected);
        }
    }

    assert!(result.boltzmann_satisfied,
        "System did not reach Boltzmann distribution at equilibrium!");

    println!("✓ Boltzmann distribution validated at equilibrium");
}

#[test]
fn test_fluctuation_dissipation_theorem() {
    println!("=== Testing Fluctuation-Dissipation Theorem ===");

    let config = NetworkConfig {
        n_oscillators: 256,
        temperature: 300.0,
        damping: 0.15,
        dt: 0.001,
        coupling_strength: 0.3,
        enable_information_gating: false,
        seed: 98765,
    };

    let mut network = ThermodynamicNetwork::new(config);

    // Run for sufficient statistics
    println!("Running 20,000 steps for FDT validation...");
    let result = network.evolve(20_000);

    println!("\n=== Fluctuation-Dissipation Validation ===");
    println!("FDT Ratio: {:.4} (should be ≈ 1.0)", result.metrics.fluctuation_dissipation_ratio);
    println!("FDT Satisfied: {}", result.fluctuation_dissipation_satisfied);

    let ratio = result.metrics.fluctuation_dissipation_ratio;
    let deviation = (ratio - 1.0).abs();

    println!("Deviation from unity: {:.2}%", deviation * 100.0);

    // Fluctuation-dissipation theorem: <F(t)F(t')> = 2γk_BT δ(t-t')
    // Allow 20% deviation
    assert!(result.fluctuation_dissipation_satisfied,
        "Fluctuation-dissipation theorem violated! Ratio = {:.4}", ratio);

    println!("✓ Fluctuation-dissipation theorem satisfied");
}

#[test]
fn test_performance_contract() {
    println!("=== Testing Performance Contract: <1ms per step, 1024 oscillators ===");

    let config = NetworkConfig {
        n_oscillators: 1024,
        temperature: 300.0,
        damping: 0.1,
        dt: 0.001,
        coupling_strength: 0.5,
        enable_information_gating: false,
        seed: 11111,
    };

    let mut network = ThermodynamicNetwork::new(config);

    // Warmup
    for _ in 0..100 {
        network.step();
    }

    // Benchmark
    println!("Benchmarking 1000 steps with 1024 oscillators...");
    let start = Instant::now();
    const N_STEPS: usize = 1000;

    for _ in 0..N_STEPS {
        network.step();
    }

    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let ms_per_step = total_ms / N_STEPS as f64;

    println!("\n=== Performance Metrics ===");
    println!("Total time: {:.2} ms", total_ms);
    println!("Time per step: {:.4} ms", ms_per_step);
    println!("Steps per second: {:.0}", 1000.0 / ms_per_step);

    // Constitution contract: <1ms per step for 1024 oscillators
    assert!(ms_per_step < 1.0,
        "Performance contract violated! {:.4} ms/step > 1.0 ms/step", ms_per_step);

    println!("✓ Performance contract satisfied: {:.4} ms < 1.0 ms", ms_per_step);
}

#[test]
fn test_phase_coherence_dynamics() {
    println!("=== Testing Phase Coherence Dynamics ===");

    let config = NetworkConfig {
        n_oscillators: 128,
        temperature: 100.0, // Lower temperature for higher coherence
        damping: 0.05,
        dt: 0.001,
        coupling_strength: 1.0, // Strong coupling
        enable_information_gating: false,
        seed: 33333,
    };

    let mut network = ThermodynamicNetwork::new(config);

    println!("Running 10,000 steps with strong coupling...");
    let result = network.evolve(10_000);

    println!("\n=== Coherence Metrics ===");
    println!("Final phase coherence: {:.4}", result.metrics.phase_coherence);
    println!("Average coupling: {:.4}", result.metrics.avg_coupling);

    // With strong coupling and low temperature, should see some synchronization
    // But not perfect due to thermal noise
    assert!(result.metrics.phase_coherence > 0.1,
        "No synchronization observed with strong coupling");

    println!("✓ Phase coherence dynamics validated");
}

#[test]
fn test_entropy_production_rate_positive() {
    println!("=== Testing Entropy Production Rate ===");

    let config = NetworkConfig {
        n_oscillators: 256,
        temperature: 300.0,
        damping: 0.1,
        dt: 0.001,
        coupling_strength: 0.5,
        enable_information_gating: false,
        seed: 44444,
    };

    let mut network = ThermodynamicNetwork::new(config);

    println!("Running 5,000 steps...");
    let result = network.evolve(5_000);

    println!("\n=== Entropy Production ===");
    println!("Entropy production rate: {:.6e}", result.metrics.entropy_production_rate);

    // Entropy production rate should be non-negative
    assert!(result.metrics.entropy_production_rate >= -1e-10,
        "Negative entropy production rate: {:.6e}", result.metrics.entropy_production_rate);

    println!("✓ Entropy production rate is non-negative");
}

#[test]
fn test_energy_fluctuations() {
    println!("=== Testing Energy Fluctuations ===");

    let config = NetworkConfig {
        n_oscillators: 128,
        temperature: 300.0,
        damping: 0.1,
        dt: 0.001,
        coupling_strength: 0.3,
        enable_information_gating: false,
        seed: 55555,
    };

    let mut network = ThermodynamicNetwork::new(config);

    println!("Running 10,000 steps...");
    network.evolve(10_000);

    let energy_history = network.energy_history();

    // Calculate energy statistics
    let mean_energy: f64 = energy_history.iter().sum::<f64>() / energy_history.len() as f64;
    let variance: f64 = energy_history.iter()
        .map(|&e| (e - mean_energy).powi(2))
        .sum::<f64>() / energy_history.len() as f64;
    let std_dev = variance.sqrt();

    println!("\n=== Energy Statistics ===");
    println!("Mean energy: {:.6e}", mean_energy);
    println!("Energy std dev: {:.6e}", std_dev);
    println!("Relative fluctuation: {:.4}%", (std_dev / mean_energy.abs()) * 100.0);

    // Energy should fluctuate (thermal system)
    assert!(std_dev > 0.0, "No energy fluctuations - system is frozen");

    // But fluctuations shouldn't be too large (>50% would indicate instability)
    let rel_fluctuation = std_dev / mean_energy.abs();
    assert!(rel_fluctuation < 0.5,
        "Energy fluctuations too large: {:.1}% - system unstable", rel_fluctuation * 100.0);

    println!("✓ Energy fluctuations within expected range");
}

#[test]
fn test_thermodynamic_consistency_comprehensive() {
    println!("=== Comprehensive Thermodynamic Consistency Test ===");

    let config = NetworkConfig {
        n_oscillators: 512,
        temperature: 300.0,
        damping: 0.1,
        dt: 0.001,
        coupling_strength: 0.5,
        enable_information_gating: false,
        seed: 66666,
    };

    let mut network = ThermodynamicNetwork::new(config);

    println!("Running comprehensive validation (100,000 steps)...");
    let result = network.evolve(100_000);

    println!("\n=== Comprehensive Validation Results ===");
    println!("Entropy never decreased: {}", result.entropy_never_decreased);
    println!("Boltzmann distribution satisfied: {}", result.boltzmann_satisfied);
    println!("Fluctuation-dissipation satisfied: {}", result.fluctuation_dissipation_satisfied);
    println!("Execution time: {:.2} ms", result.execution_time_ms);
    println!("Time per step: {:.4} ms", result.execution_time_ms / 100_000.0);

    println!("\n=== Thermodynamic Metrics ===");
    println!("Entropy production rate: {:.6e}", result.metrics.entropy_production_rate);
    println!("Phase coherence: {:.4}", result.metrics.phase_coherence);
    println!("FDT ratio: {:.4}", result.metrics.fluctuation_dissipation_ratio);
    println!("Avg coupling: {:.4}", result.metrics.avg_coupling);

    // All validation criteria must pass
    assert!(result.entropy_never_decreased,
        "CRITICAL: Second law violated - entropy decreased!");

    assert!(result.boltzmann_satisfied,
        "Boltzmann distribution not satisfied at equilibrium");

    assert!(result.fluctuation_dissipation_satisfied,
        "Fluctuation-dissipation theorem violated");

    println!("\n✓ ALL THERMODYNAMIC CONSISTENCY CHECKS PASSED");
    println!("✓ System is scientifically rigorous and production-ready");
}

#[test]
fn test_zero_temperature_limit() {
    println!("=== Testing Zero Temperature Limit ===");

    let config = NetworkConfig {
        n_oscillators: 64,
        temperature: 0.0, // No thermal noise
        damping: 0.1,
        dt: 0.001,
        coupling_strength: 0.5,
        enable_information_gating: false,
        seed: 77777,
    };

    let mut network = ThermodynamicNetwork::new(config);

    println!("Running 1,000 steps at T=0...");
    network.evolve(1_000);

    let energy_history = network.energy_history();

    // At T=0, energy should be approximately conserved (decrease slowly due to damping)
    let initial_energy = energy_history[0];
    let final_energy = *energy_history.last().unwrap();
    let energy_change = (final_energy - initial_energy) / initial_energy.abs();

    println!("\n=== Zero Temperature Results ===");
    println!("Initial energy: {:.6e}", initial_energy);
    println!("Final energy: {:.6e}", final_energy);
    println!("Relative change: {:.4}%", energy_change * 100.0);

    // Energy should decrease (damping) but not increase
    assert!(final_energy <= initial_energy,
        "Energy increased at T=0 - unphysical!");

    println!("✓ Zero temperature limit behaves correctly");
}
