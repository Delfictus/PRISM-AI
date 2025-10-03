// Debug Transfer Entropy Implementation
// Investigating zero values issue

use active_inference_platform::information_theory::TransferEntropy;
use ndarray::Array1;

fn main() {
    println!("=== Debugging Transfer Entropy ===\n");

    // Create a simple deterministic causal system
    let n = 100;
    let mut x = Vec::new();
    let mut y = Vec::new();

    // X is a simple counter
    for i in 0..n {
        x.push(i as f64 / 10.0);
    }

    // Y perfectly follows X with lag 1
    y.push(0.0);
    for i in 1..n {
        y.push(x[i - 1]);
    }

    let x_arr = Array1::from_vec(x.clone());
    let y_arr = Array1::from_vec(y.clone());

    println!("Data created:");
    println!("  X[0..5]: {:?}", &x[0..5]);
    println!("  Y[0..5]: {:?}", &y[0..5]);
    println!("  (Y should equal X shifted by 1)\n");

    // Test with default settings
    let te_default = TransferEntropy::default();
    println!("Testing with default settings:");
    println!("  Embedding dims: source={}, target={}",
             te_default.source_embedding, te_default.target_embedding);
    println!("  Time lag: {}", te_default.time_lag);
    println!("  Bins: {:?}", te_default.n_bins);
    println!("  Use KNN: {}", te_default.use_knn);

    let result_default = te_default.calculate(&x_arr, &y_arr);
    println!("\nResults:");
    println!("  TE value: {:.6}", result_default.te_value);
    println!("  Effective TE: {:.6}", result_default.effective_te);
    println!("  P-value: {:.6}", result_default.p_value);
    println!("  Samples used: {}", result_default.n_samples);

    // Test with explicit parameters
    println!("\n--- Testing with explicit parameters ---");
    let te_custom = TransferEntropy::new(1, 1, 1);
    let result_custom = te_custom.calculate(&x_arr, &y_arr);
    println!("Results with (1,1,1):");
    println!("  TE value: {:.6}", result_custom.te_value);
    println!("  Effective TE: {:.6}", result_custom.effective_te);

    // Test with different bin sizes
    println!("\n--- Testing different bin sizes ---");
    for bins in [2, 5, 10, 20] {
        let mut te_bins = TransferEntropy::new(1, 1, 1);
        te_bins.n_bins = Some(bins);
        let result = te_bins.calculate(&x_arr, &y_arr);
        println!("  Bins={}: TE={:.6}, Effective={:.6}",
                 bins, result.te_value, result.effective_te);
    }

    // Test a simpler binary system
    println!("\n--- Testing binary system ---");
    let mut x_binary = Vec::new();
    let mut y_binary = Vec::new();

    for i in 0..100 {
        x_binary.push(if i % 2 == 0 { 0.0 } else { 1.0 });
    }

    y_binary.push(0.0);
    for i in 1..100 {
        y_binary.push(x_binary[i - 1]);
    }

    let x_bin_arr = Array1::from_vec(x_binary);
    let y_bin_arr = Array1::from_vec(y_binary);

    let mut te_binary = TransferEntropy::new(1, 1, 1);
    te_binary.n_bins = Some(2);
    let result_binary = te_binary.calculate(&x_bin_arr, &y_bin_arr);
    println!("Binary system (perfect copy with lag 1):");
    println!("  TE value: {:.6}", result_binary.te_value);
    println!("  Effective TE: {:.6}", result_binary.effective_te);

    // Check discretization
    println!("\n--- Checking discretization ---");
    let test_series = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    let mut te_disc = TransferEntropy::default();
    // We need to expose discretize function or test it indirectly

    println!("\nDiagnosis:");
    if result_default.te_value == 0.0 {
        println!("❌ Transfer entropy is exactly zero - likely a calculation issue");
        println!("   Possible causes:");
        println!("   - Joint probability calculation error");
        println!("   - Discretization producing constant values");
        println!("   - Embedding creation issue");
        println!("   - Log calculation with zero probabilities");
    } else {
        println!("✓ Transfer entropy is non-zero");
    }
}