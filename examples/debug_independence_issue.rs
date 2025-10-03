// Debug: Why do independent series show non-zero TE?
// This is a critical mathematical violation that must be fixed

use active_inference_platform::information_theory::TransferEntropy;
use ndarray::Array1;
use std::collections::HashMap;

fn main() {
    println!("=== DEBUGGING INDEPENDENCE ISSUE ===\n");

    // Create truly independent random series
    let n = 1000;
    let mut rng = rand::thread_rng();
    use rand::Rng;

    // Test 1: Completely random independent series
    println!("Test 1: Random uniform series");
    let x: Array1<f64> = Array1::from_shape_fn(n, |_| rng.gen::<f64>());
    let y: Array1<f64> = Array1::from_shape_fn(n, |_| rng.gen::<f64>());

    let te = TransferEntropy::default();
    let result = te.calculate(&x, &y);

    println!("  TE value: {:.6} bits", result.te_value);
    println!("  Effective TE: {:.6} bits", result.effective_te);
    println!("  P-value: {:.6}", result.p_value);
    println!("  Samples used: {}", result.n_samples);
    println!("  Expected: TE = 0.0 for independent series");
    println!("  Problem: TE = {:.6} != 0\n", result.te_value);

    // Test 2: Check discretization
    println!("Test 2: Checking discretization");
    let x_simple = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
    let y_simple = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]);

    // Manual discretization check
    let te_simple = TransferEntropy::default();
    let result_simple = te_simple.calculate(&x_simple, &y_simple);
    println!("  Simple series TE: {:.6}", result_simple.te_value);

    // Test 3: Finite sample bias
    println!("\nTest 3: Sample size effect");
    for n_samples in [50, 100, 200, 500, 1000, 2000] {
        let x: Array1<f64> = Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>());
        let y: Array1<f64> = Array1::from_shape_fn(n_samples, |_| rng.gen::<f64>());

        let result = te.calculate(&x, &y);
        println!("  n={}: TE={:.6}, Effective={:.6}, p={:.4}",
                 n_samples, result.te_value, result.effective_te, result.p_value);
    }

    // Test 4: Check probability calculation
    println!("\nTest 4: Probability calculation check");

    // Create simple deterministic series for debugging
    let x_det = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let y_det = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

    let mut te_debug = TransferEntropy::new(1, 1, 1);
    te_debug.n_bins = Some(2); // Binary discretization
    let result_det = te_debug.calculate(&x_det, &y_det);

    println!("  Alternating binary series:");
    println!("  X: 0,1,0,1,0,1,0,1,0,1");
    println!("  Y: 1,0,1,0,1,0,1,0,1,0");
    println!("  TE: {:.6}", result_det.te_value);
    println!("  (Y is NOT X shifted, they're independent patterns)");

    // Test 5: Multiple trials to check consistency
    println!("\nTest 5: Multiple independent trials");
    let mut te_values = Vec::new();
    for _ in 0..10 {
        let x: Array1<f64> = Array1::from_shape_fn(500, |_| rng.gen::<f64>());
        let y: Array1<f64> = Array1::from_shape_fn(500, |_| rng.gen::<f64>());
        let result = te.calculate(&x, &y);
        te_values.push(result.te_value);
    }

    let mean_te = te_values.iter().sum::<f64>() / te_values.len() as f64;
    let max_te = te_values.iter().fold(0.0_f64, |a, &b| a.max(b));
    let min_te = te_values.iter().fold(1.0_f64, |a, &b| a.min(b));

    println!("  Mean TE: {:.6}", mean_te);
    println!("  Max TE: {:.6}", max_te);
    println!("  Min TE: {:.6}", min_te);
    println!("  All should be 0 for independent series!");

    // Test 6: Check if it's a bias issue
    println!("\nTest 6: Bias correction analysis");
    let x_test: Array1<f64> = Array1::from_shape_fn(1000, |_| rng.gen::<f64>());
    let y_test: Array1<f64> = Array1::from_shape_fn(1000, |_| rng.gen::<f64>());

    let result_test = te.calculate(&x_test, &y_test);
    let bias = result_test.te_value - result_test.effective_te;

    println!("  Raw TE: {:.6}", result_test.te_value);
    println!("  Effective TE: {:.6}", result_test.effective_te);
    println!("  Bias correction: {:.6}", bias);
    println!("  Is bias correction sufficient? {}",
             if result_test.effective_te < 0.01 { "Maybe" } else { "NO" });

    // Diagnosis
    println!("\n=== DIAGNOSIS ===");
    if mean_te > 0.01 {
        println!("❌ CRITICAL BUG: Independent series showing spurious TE");
        println!("   Likely causes:");
        println!("   1. Finite sample bias not properly corrected");
        println!("   2. Probability estimation errors with small bins");
        println!("   3. Numerical issues in logarithm calculations");
        println!("   4. Key parsing error in joint probability calculation");
    } else {
        println!("✓ TE values are acceptably close to zero");
    }

    // Test 7: Check the formula directly
    println!("\nTest 7: Direct formula verification");

    // For independent X and Y:
    // p(x,y,z) = p(x) * p(y) * p(z)
    // TE = Σ p(x,y,z) log[p(x,y,z)*p(y) / (p(x,y)*p(y,z))]
    //    = Σ p(x)*p(y)*p(z) log[p(x)*p(y)*p(z)*p(y) / (p(x)*p(y)*p(y)*p(z))]
    //    = Σ p(x)*p(y)*p(z) log[1] = 0

    println!("  Mathematical expectation for independent series:");
    println!("  TE = Σ p(x,y,z) log[p(x,y,z)*p(y) / (p(x,y)*p(y,z))]");
    println!("  For independent: p(x,y,z) = p(x)*p(y)*p(z)");
    println!("  Therefore: TE = 0 (exactly)");

    if result.te_value > 0.0 {
        println!("\n  ❌ Implementation violates mathematical definition!");
    }
}