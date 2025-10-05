//! GPU Correctness Validation Test
//!
//! Constitutional Compliance:
//! - Article IV, MathValidator: Verifiable proof of mathematical equivalence
//! - Article I, Principle 1: Mathematical rigor through verified proofs
//!
//! This test validates that the GPU-accelerated quantum evolution via CSF-Quantum
//! produces results mathematically equivalent to the CPU implementation.

use prism_ai::quantum::*;
use prism_ai::shared_types::*;
use prism_ai::prct_core::ports::QuantumPort;
use prism_ai::prct_adapters::QuantumAdapter;
use approx::assert_relative_eq;
use num_complex::Complex64;
use ndarray::{Array1, Array2};

/// Tolerance for floating-point comparison (1e-9 as per spec)
const TOLERANCE: f64 = 1e-9;

/// CPU reference implementation of quantum state evolution
fn evolve_state_cpu(
    hamiltonian: &Array2<Complex64>,
    initial_state: &Array1<Complex64>,
    time: f64,
) -> Array1<Complex64> {
    use ndarray_linalg::eigh::Eigh;

    // Diagonalize Hamiltonian H = U†DU
    let (eigenvalues, eigenvectors) = hamiltonian.eigh(ndarray_linalg::UPLO::Lower)
        .expect("Failed to diagonalize Hamiltonian");

    // Evolution operator: exp(-iHt/ħ) = U† exp(-iDt/ħ) U
    let hbar = 1.0; // Natural units
    let i = Complex64::new(0.0, 1.0);

    // Transform to eigenbasis
    let state_eigen = eigenvectors.t().dot(initial_state);

    // Apply time evolution in eigenbasis
    let mut evolved_eigen = Array1::zeros(state_eigen.len());
    for (idx, &coeff) in state_eigen.iter().enumerate() {
        let phase = (-i * eigenvalues[idx] * time / hbar).exp();
        evolved_eigen[idx] = coeff * phase;
    }

    // Transform back to original basis
    eigenvectors.dot(&evolved_eigen)
}

/// Test that GPU and CPU implementations produce equivalent results
#[test]
fn test_gpu_cpu_equivalence() {
    println!("\n=== Quantum GPU Correctness Validation ===");
    println!("Constitutional: Article IV, MathValidator");
    println!("Verifying mathematical equivalence between CPU and GPU implementations\n");

    // Create test graph (small for CPU feasibility)
    let graph = Graph {
        num_vertices: 4,
        edges: vec![
            (0, 1, 1.0),
            (1, 2, 1.5),
            (2, 3, 0.8),
            (3, 0, 1.2),
            (0, 2, 0.5),
        ],
    };

    let evolution_params = EvolutionParams {
        time_step: 0.01,
        total_time: 0.1,
        coupling_strength: 1.0,
        temperature: 1.0,
        convergence_threshold: 1e-6,
        max_iterations: 100,
    };

    // Initialize GPU quantum adapter
    let gpu_adapter = QuantumAdapter::new();

    // Build Hamiltonian using GPU
    let hamiltonian_state = gpu_adapter.build_hamiltonian(&graph, &evolution_params)
        .expect("Failed to build Hamiltonian on GPU");

    // Create initial state (normalized superposition)
    let dimension = hamiltonian_state.dimension;
    let norm_factor = 1.0 / (dimension as f64).sqrt();
    let initial_amplitudes: Vec<(f64, f64)> = (0..dimension)
        .map(|i| {
            if i == 0 {
                (norm_factor, 0.0)
            } else {
                (norm_factor / 2.0, norm_factor / 2.0)
            }
        })
        .collect();

    let initial_state = QuantumState {
        amplitudes: initial_amplitudes.clone(),
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    // Evolve on GPU
    println!("Executing quantum evolution on GPU...");
    let gpu_evolved = gpu_adapter.evolve_state(
        &hamiltonian_state,
        &initial_state,
        evolution_params.time_step,
    ).expect("GPU evolution failed");

    // CPU reference implementation
    println!("Executing quantum evolution on CPU (reference)...");

    // Reconstruct Hamiltonian matrix for CPU
    let mut hamiltonian_cpu = Array2::zeros((dimension, dimension));
    for i in 0..dimension {
        for j in 0..dimension {
            let idx = i * dimension + j;
            let (re, im) = hamiltonian_state.matrix_elements[idx];
            hamiltonian_cpu[[i, j]] = Complex64::new(re, im);
        }
    }

    // Convert initial state to ndarray
    let initial_state_cpu: Array1<Complex64> = initial_amplitudes.iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();

    // Evolve on CPU
    let cpu_evolved = evolve_state_cpu(
        &hamiltonian_cpu,
        &initial_state_cpu,
        evolution_params.time_step,
    );

    // Compare results
    println!("\nValidating results...");
    assert_eq!(gpu_evolved.amplitudes.len(), cpu_evolved.len(),
        "Dimension mismatch between GPU and CPU results");

    let mut max_diff = 0.0;
    for (idx, (&(gpu_re, gpu_im), &cpu_val)) in
        gpu_evolved.amplitudes.iter().zip(cpu_evolved.iter()).enumerate()
    {
        let gpu_val = Complex64::new(gpu_re, gpu_im);
        let diff = (gpu_val - cpu_val).norm();
        max_diff = max_diff.max(diff);

        // Verify within tolerance
        assert_relative_eq!(
            gpu_val.re, cpu_val.re,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
            "Real part mismatch at index {}: GPU={}, CPU={}",
            idx, gpu_val.re, cpu_val.re
        );

        assert_relative_eq!(
            gpu_val.im, cpu_val.im,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
            "Imaginary part mismatch at index {}: GPU={}, CPU={}",
            idx, gpu_val.im, cpu_val.im
        );
    }

    println!("✓ Maximum difference: {:.2e} (tolerance: {:.2e})", max_diff, TOLERANCE);
    println!("✓ GPU and CPU implementations are mathematically equivalent!");
    println!("\n=== Test PASSED ===");
}

/// Test deterministic reproducibility of GPU implementation
#[test]
fn test_gpu_determinism() {
    println!("\n=== GPU Determinism Test ===");
    println!("Verifying bit-for-bit reproducibility of GPU computations\n");

    let graph = Graph {
        num_vertices: 3,
        edges: vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
        ],
    };

    let params = EvolutionParams {
        time_step: 0.1,
        total_time: 1.0,
        coupling_strength: 1.0,
        temperature: 1.0,
        convergence_threshold: 1e-6,
        max_iterations: 100,
    };

    // Run evolution multiple times
    const NUM_RUNS: usize = 5;
    let mut results = Vec::with_capacity(NUM_RUNS);

    for run in 0..NUM_RUNS {
        println!("Run {}/{}", run + 1, NUM_RUNS);

        let adapter = QuantumAdapter::new();
        let hamiltonian = adapter.build_hamiltonian(&graph, &params)
            .expect("Failed to build Hamiltonian");

        let initial_state = QuantumState {
            amplitudes: vec![(1.0, 0.0); hamiltonian.dimension],
            phase_coherence: 1.0,
            energy: 0.0,
            entanglement: 0.0,
            timestamp_ns: 0,
        };

        let evolved = adapter.evolve_state(&hamiltonian, &initial_state, params.time_step)
            .expect("Evolution failed");

        results.push(evolved.amplitudes);
    }

    // Verify all runs produce identical results
    println!("\nVerifying determinism...");
    for run in 1..NUM_RUNS {
        for (idx, (&(ref_re, ref_im), &(run_re, run_im))) in
            results[0].iter().zip(results[run].iter()).enumerate()
        {
            assert_eq!(
                ref_re, run_re,
                "Real part differs at index {} between run 0 and run {}",
                idx, run
            );
            assert_eq!(
                ref_im, run_im,
                "Imaginary part differs at index {} between run 0 and run {}",
                idx, run
            );
        }
    }

    println!("✓ All {} runs produced identical results", NUM_RUNS);
    println!("✓ GPU implementation is deterministic!");
    println!("\n=== Test PASSED ===");
}

/// Test conservation laws in GPU evolution
#[test]
fn test_gpu_conservation_laws() {
    println!("\n=== Conservation Laws Validation ===");
    println!("Verifying unitarity and norm conservation\n");

    let graph = Graph {
        num_vertices: 5,
        edges: vec![
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0),
            (3, 4, 1.0), (4, 0, 1.0),
        ],
    };

    let params = EvolutionParams {
        time_step: 0.01,
        total_time: 1.0,
        coupling_strength: 1.0,
        temperature: 1.0,
        convergence_threshold: 1e-6,
        max_iterations: 100,
    };

    let adapter = QuantumAdapter::new();
    let hamiltonian = adapter.build_hamiltonian(&graph, &params)
        .expect("Failed to build Hamiltonian");

    // Create normalized initial state
    let dimension = hamiltonian.dimension;
    let norm_factor = 1.0 / (dimension as f64).sqrt();
    let initial_state = QuantumState {
        amplitudes: vec![(norm_factor, 0.0); dimension],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    // Calculate initial norm
    let initial_norm: f64 = initial_state.amplitudes.iter()
        .map(|&(re, im)| re * re + im * im)
        .sum::<f64>()
        .sqrt();

    println!("Initial norm: {:.10}", initial_norm);

    // Evolve for multiple steps
    let num_steps = 10;
    let mut current_state = initial_state;

    for step in 0..num_steps {
        current_state = adapter.evolve_state(
            &hamiltonian,
            &current_state,
            params.time_step,
        ).expect("Evolution failed");

        // Check norm conservation
        let norm: f64 = current_state.amplitudes.iter()
            .map(|&(re, im)| re * re + im * im)
            .sum::<f64>()
            .sqrt();

        println!("Step {}: norm = {:.10}, diff = {:.2e}",
            step + 1, norm, (norm - initial_norm).abs());

        assert_relative_eq!(
            norm, initial_norm,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
            "Norm not conserved at step {}: expected {}, got {}",
            step + 1, initial_norm, norm
        );
    }

    println!("\n✓ Unitarity preserved throughout evolution");
    println!("✓ Norm conservation validated");
    println!("\n=== Test PASSED ===");
}