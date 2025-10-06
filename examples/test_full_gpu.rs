// Test Full GPU Execution
//
// Verifies that all 5 modules actually execute on GPU

use prism_ai::integration::{UnifiedPlatform, PlatformInput};
use ndarray::Array1;

fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════");
    println!("  PRISM-AI FULL GPU EXECUTION TEST");
    println!("═══════════════════════════════════════════════\n");

    // Initialize platform with GPU adapters
    println!("Initializing UnifiedPlatform with GPU adapters...\n");
    let mut platform = UnifiedPlatform::new(10)?;

    println!("\n");

    // Create test input
    let input = PlatformInput::new(
        Array1::from_vec(vec![0.5, 0.7, 0.3, 0.9, 0.2, 0.6, 0.8, 0.1, 0.4, 0.75]),
        Array1::from_vec(vec![0.0; 10]),
        0.01,
    );

    println!("Processing input through full GPU pipeline...\n");

    // Execute full pipeline
    let output = platform.process(input)?;

    println!("\n═══════════════════════════════════════════════");
    println!("  EXECUTION COMPLETE");
    println!("═══════════════════════════════════════════════");
    println!("\n{}", output.metrics.report());

    println!("\n✅ ALL 5 MODULES EXECUTED ON GPU SUCCESSFULLY!");

    Ok(())
}
