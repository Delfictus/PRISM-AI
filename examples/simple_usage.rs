//! Simple Usage Example
//!
//! Basic example showing how to use the neuromorphic-quantum platform

use neuromorphic_quantum_platform::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create the platform with default configuration
    let platform = create_platform().await?;

    // Process some sample data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let result = process_data(&platform, "sample_data".to_string(), data).await?;

    // Display results
    println!("Prediction: {}", result.prediction.direction);
    println!("Confidence: {:.1}%", result.prediction.confidence * 100.0);
    println!("Processing time: {:.1}ms", result.metadata.duration_ms);

    if let Some(neuro) = &result.neuromorphic_results {
        println!("Detected {} patterns", neuro.patterns.len());
    }

    Ok(())
}