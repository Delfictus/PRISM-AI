//! STDP Learning Demonstration
//!
//! Demonstrates Spike-Timing-Dependent Plasticity (STDP) in action with various learning profiles.
//! Shows adaptive weight learning, convergence monitoring, and profile comparison.

use neuromorphic_engine::{
    ReservoirComputer, reservoir::ReservoirConfig, STDPProfile, SpikePattern, Spike, PatternMetadata,
};
use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🧠 STDP Learning Demonstration");
    println!("================================\n");

    // Test all STDP profiles
    for profile in STDPProfile::all() {
        if matches!(profile, STDPProfile::Custom) {
            continue; // Skip custom profile
        }

        println!("\n📊 Testing Profile: {:?}", profile);
        println!("─────────────────────────────────────");
        println!("Description: {}\n", profile.description());

        let config = profile.get_config();
        println!("Configuration:");
        println!("  • Learning Rate: {:.4}", config.learning_rate);
        println!("  • Time Constants: τ+={:.1}ms, τ-={:.1}ms",
            config.time_constant_pos, config.time_constant_neg);
        println!("  • Weight Bounds: [{:.2}, {:.2}]",
            config.min_weight, config.max_weight);
        println!("  • Heterosynaptic: {}", config.enable_heterosynaptic);
        println!("  • Homeostasis: {}\n", config.enable_homeostasis);

        test_learning_profile(profile)?;
    }

    println!("\n✅ STDP Learning Demonstration Complete!");
    println!("═══════════════════════════════════════\n");

    Ok(())
}

fn test_learning_profile(profile: STDPProfile) -> Result<()> {
    // Create reservoir with STDP enabled
    let mut config = ReservoirConfig {
        size: 200,  // Smaller for faster demonstration
        input_size: 50,
        spectral_radius: 0.95,
        connection_prob: 0.1,
        leak_rate: 0.3,
        input_scaling: 1.0,
        noise_level: 0.01,
        enable_plasticity: true,
        stdp_profile: profile,
    };

    let mut reservoir = ReservoirComputer::with_config(config)?;

    // Training phase: Present repeating patterns
    println!("📈 Training Phase (50 iterations):");
    let start = Instant::now();

    let training_patterns = create_training_patterns();

    for epoch in 0..10 {
        for pattern in &training_patterns {
            reservoir.process(pattern)?;
        }

        if epoch % 2 == 0 {
            let stats = reservoir.get_learning_stats();
            println!("  Epoch {:2}: Mean Weight = {:.4}, Variance = {:.6}, Saturation = {:.1}%, Entropy = {:.3}",
                epoch, stats.mean_weight, stats.weight_variance, stats.saturation_percentage, stats.weight_entropy);
        }
    }

    let duration = start.elapsed();

    // Final statistics
    println!("\n📊 Final Learning Statistics:");
    let final_stats = reservoir.get_learning_stats();
    println!("  • Total Weight Updates: {}", final_stats.total_updates);
    println!("  • Mean Weight: {:.4}", final_stats.mean_weight);
    println!("  • Weight Variance: {:.6}", final_stats.weight_variance);
    println!("  • Weight Range: [{:.4}, {:.4}]",
        final_stats.min_weight, final_stats.max_weight);
    println!("  • Saturation: {:.1}%", final_stats.saturation_percentage);
    println!("  • Weight Entropy: {:.3}", final_stats.weight_entropy);
    println!("  • Mean Activity: {:.4}", final_stats.mean_activity);
    println!("  • Learning Health Score: {:.2}", final_stats.health_score());
    println!("  • Training Time: {:.2}ms", duration.as_millis());

    // Check convergence
    if reservoir.has_learning_converged(20, 0.0001) {
        println!("  ✅ Learning has converged");
    } else {
        println!("  ⚠️  Learning has not yet converged");
    }

    // Evaluate health
    let health = final_stats.health_score();
    if health > 0.8 {
        println!("  ✅ Healthy learning dynamics");
    } else if health > 0.5 {
        println!("  ⚠️  Moderate learning health");
    } else {
        println!("  ❌ Poor learning health");
    }

    Ok(())
}

/// Create diverse training patterns
fn create_training_patterns() -> Vec<SpikePattern> {
    vec![
        create_pattern_regular(50.0, 5),
        create_pattern_burst(50.0, 10),
        create_pattern_sparse(50.0, 3),
        create_pattern_dense(50.0, 15),
        create_pattern_clustered(50.0, 8),
    ]
}

/// Create regular spike pattern
fn create_pattern_regular(duration_ms: f64, spike_count: usize) -> SpikePattern {
    let interval = duration_ms / spike_count as f64;
    let spikes: Vec<Spike> = (0..spike_count)
        .map(|i| Spike {
            time_ms: i as f64 * interval,
            amplitude: Some(1.0),
            neuron_id: i,
        })
        .collect();

    SpikePattern {
        duration_ms,
        spikes,
        metadata: PatternMetadata::default(),
    }
}

/// Create burst pattern (spikes clustered at beginning)
fn create_pattern_burst(duration_ms: f64, spike_count: usize) -> SpikePattern {
    let burst_window = duration_ms * 0.2;
    let interval = burst_window / spike_count as f64;
    let spikes: Vec<Spike> = (0..spike_count)
        .map(|i| Spike {
            time_ms: i as f64 * interval,
            amplitude: Some(1.5),
            neuron_id: i,
        })
        .collect();

    SpikePattern {
        duration_ms,
        spikes,
        metadata: PatternMetadata::default(),
    }
}

/// Create sparse pattern (few spikes)
fn create_pattern_sparse(duration_ms: f64, spike_count: usize) -> SpikePattern {
    let interval = duration_ms / spike_count as f64;
    let spikes: Vec<Spike> = (0..spike_count)
        .map(|i| Spike {
            time_ms: i as f64 * interval + 5.0,
            amplitude: Some(0.8),
            neuron_id: i,
        })
        .collect();

    SpikePattern {
        duration_ms,
        spikes,
        metadata: PatternMetadata::default(),
    }
}

/// Create dense pattern (many spikes)
fn create_pattern_dense(duration_ms: f64, spike_count: usize) -> SpikePattern {
    let interval = duration_ms / spike_count as f64;
    let spikes: Vec<Spike> = (0..spike_count)
        .map(|i| Spike {
            time_ms: i as f64 * interval,
            amplitude: Some(0.9),
            neuron_id: i % 10,
        })
        .collect();

    SpikePattern {
        duration_ms,
        spikes,
        metadata: PatternMetadata::default(),
    }
}

/// Create clustered pattern (spikes in two clusters)
fn create_pattern_clustered(duration_ms: f64, spike_count: usize) -> SpikePattern {
    let cluster1_count = spike_count / 2;
    let cluster2_count = spike_count - cluster1_count;

    let mut spikes = Vec::new();

    // First cluster
    for i in 0..cluster1_count {
        spikes.push(Spike {
            time_ms: i as f64 * 2.0,
            amplitude: Some(1.2),
            neuron_id: i,
        });
    }

    // Second cluster
    for i in 0..cluster2_count {
        spikes.push(Spike {
            time_ms: duration_ms * 0.7 + i as f64 * 2.0,
            amplitude: Some(1.2),
            neuron_id: i + cluster1_count,
        });
    }

    SpikePattern {
        duration_ms,
        spikes,
        metadata: PatternMetadata::default(),
    }
}
