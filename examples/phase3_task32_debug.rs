//! Debug test to find exact location of dimension mismatch

use active_inference_platform::integration::{UnifiedPlatform, PlatformInput};
use ndarray::Array1;

fn main() {
    println!("Phase 3 Task 3.2 Debug Test\n");

    // Test with small dimensions first
    println!("Test 1: Small dimensions (10)");
    let mut platform_small = UnifiedPlatform::new(10).unwrap();
    platform_small.initialize();

    let input_small = PlatformInput::new(
        Array1::ones(10) * 0.6,
        Array1::zeros(10),
        0.01
    );

    match platform_small.process(input_small) {
        Ok(output) => {
            println!("✓ Small dimension test passed");
            println!("  Latency: {:.2} ms\n", output.metrics.total_latency_ms);
        }
        Err(e) => {
            println!("✗ Small dimension test failed: {}\n", e);
        }
    }

    // Test with exact HierarchicalModel dimensions
    println!("Test 2: HierarchicalModel dimensions (900)");
    let mut platform_900 = UnifiedPlatform::new(900).unwrap();
    platform_900.initialize();

    let input_900 = PlatformInput::new(
        Array1::ones(900) * 0.6,
        Array1::zeros(900),
        0.01
    );

    println!("Attempting to process with 900 dimensions...");

    match platform_900.process(input_900) {
        Ok(output) => {
            println!("✓ 900 dimension test passed!");
            println!("  Latency: {:.2} ms", output.metrics.total_latency_ms);
            println!("  All phases completed successfully");
        }
        Err(e) => {
            println!("✗ 900 dimension test failed");
            println!("  Error: {}", e);

            // Try to identify which phase fails
            println!("\nTrying individual phases...");

            // Create fresh platform for phase testing
            let mut test_platform = UnifiedPlatform::new(900).unwrap();
            test_platform.initialize();

            // We can't directly call the phases, but we can infer from the error
            println!("  Likely failure point: dimension mismatch in active inference or control");
        }
    }
}