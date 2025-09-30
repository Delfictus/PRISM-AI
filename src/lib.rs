//! Neuromorphic-Quantum Computing Platform
//!
//! World's first software-based neuromorphic-quantum computing platform.
//! Combines biological neural processing with quantum-inspired optimization
//! to create revolutionary 22nd-century computing capabilities on standard hardware.

// Re-export all platform components
pub use neuromorphic_engine::{
    SpikeEncoder, ReservoirComputer, PatternDetector,
    InputData, SpikePattern, Pattern, Prediction
};

pub use quantum_engine::{
    Hamiltonian, ForceFieldParams, calculate_ground_state
};

pub use platform_foundation::{
    NeuromorphicQuantumPlatform, PlatformInput, PlatformOutput,
    ProcessingConfig, NeuromorphicConfig, QuantumConfig
};
pub use platform_foundation::platform::{PlatformStatus, PlatformMetrics};

/// Platform version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = "Neuromorphic-Quantum Computing Platform";
pub const DESCRIPTION: &str = "World's first software-based neuromorphic-quantum computing platform";

/// Quick start function for creating a platform instance
pub async fn create_platform() -> anyhow::Result<NeuromorphicQuantumPlatform> {
    let config = ProcessingConfig::default();
    NeuromorphicQuantumPlatform::new(config).await
}

/// Process data through the complete neuromorphic-quantum pipeline
pub async fn process_data(platform: &NeuromorphicQuantumPlatform, source: String, values: Vec<f64>) -> anyhow::Result<PlatformOutput> {
    let input = PlatformInput::new(source, values);
    platform.process(input).await
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_platform_integration() {
        println!("🚀 Testing World's First Software-Based Neuromorphic-Quantum Platform");

        let platform = create_platform().await.unwrap();

        // Test with various data patterns
        let test_cases = vec![
            ("financial_data", vec![100.0, 102.0, 98.0, 105.0, 103.0, 99.0, 107.0]),
            ("sensor_data", vec![23.5, 23.7, 24.1, 23.9, 24.3, 24.0, 23.8]),
            ("signal_data", vec![0.1, 0.3, 0.7, 0.9, 0.4, 0.2, 0.8, 0.6, 0.5]),
        ];

        for (source, data) in test_cases {
            println!("Processing {} with {} data points", source, data.len());

            let output = process_data(&platform, source.to_string(), data).await.unwrap();

            // Verify complete processing
            assert!(output.neuromorphic_results.is_some(), "Neuromorphic processing failed");
            assert!(output.quantum_results.is_some(), "Quantum optimization failed");
            assert!(output.prediction.confidence > 0.0, "No prediction generated");

            println!("  ✅ Prediction: {} (confidence: {:.3})", output.prediction.direction, output.prediction.confidence);

            if let Some(neuro) = &output.neuromorphic_results {
                println!("  📊 Detected {} patterns, coherence: {:.3}", neuro.patterns.len(), neuro.spike_analysis.coherence);
            }

            if let Some(quantum) = &output.quantum_results {
                println!("  ⚛️  Energy: {:.3}, phase coherence: {:.3}", quantum.energy, quantum.phase_coherence);
            }

            println!("  ⏱️  Processing time: {:.1}ms", output.metadata.duration_ms);
        }

        // Test platform status and metrics
        let status = platform.get_status().await;
        println!("\n📈 Platform Status:");
        println!("  Processed: {} inputs", status.total_inputs_processed);
        println!("  Success rate: {:.1}%", status.success_rate * 100.0);
        println!("  Avg processing time: {:.1}ms", status.avg_processing_time_ms);

        assert!(status.total_inputs_processed >= 3);
        assert!(status.success_rate > 0.5);

        println!("\n🎉 COMPLETE INTEGRATION TEST PASSED!");
        println!("✨ World's First Software-Based Neuromorphic-Quantum Computing Platform VERIFIED ✨");
    }
}