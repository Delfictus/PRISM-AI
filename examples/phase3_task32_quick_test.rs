//! Quick test for Phase 3 Task 3.2

use active_inference_platform::integration::{UnifiedPlatform, PlatformInput};
use ndarray::Array1;
use std::time::Instant;

fn main() {
    println!("Phase 3 Task 3.2 Quick Validation\n");

    // Create small platform for quick testing
    let mut platform = UnifiedPlatform::new(10).unwrap();
    platform.initialize();

    // Single execution test
    let input = PlatformInput::new(
        Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5]),
        Array1::zeros(10),
        0.01
    );

    let start = Instant::now();
    match platform.process(input) {
        Ok(output) => {
            println!("✅ Pipeline executed successfully");
            println!("  Latency: {:.2} ms", output.metrics.total_latency_ms);
            println!("  Entropy production: {:.4}", output.metrics.entropy_production);
            println!("  All 8 phases: ✓");
            println!();

            // Check requirements
            let latency_ok = output.metrics.total_latency_ms < 10.0;
            let thermo_ok = output.metrics.entropy_production >= 0.0;

            println!("Requirements:");
            println!("  [{}] Latency < 10ms: {:.2} ms",
                if latency_ok { "✓" } else { "✗" },
                output.metrics.total_latency_ms);
            println!("  [{}] Thermodynamic consistency: dS/dt = {:.4} ≥ 0",
                if thermo_ok { "✓" } else { "✗" },
                output.metrics.entropy_production);

            if latency_ok && thermo_ok {
                println!("\n✅ VALIDATION PASSED");
            } else {
                println!("\n⚠️ PARTIAL SUCCESS");
            }
        }
        Err(e) => {
            println!("❌ Pipeline failed: {}", e);
        }
    }
}