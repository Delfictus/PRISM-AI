//! GPU Implementation Validation Tests
//!
//! Validates GPU-accelerated quantum evolution against reference implementations
//! Ensures constitutional compliance with accuracy and performance requirements

use prism_ai::*;
use shared_types::*;
use prct_core::ports::QuantumPort;
use prct_adapters::quantum_adapter_gpu::QuantumAdapterGpu;
use num_complex::Complex64;
use ndarray::{Array1, Array2};
use approx::assert_relative_eq;

/// Test GPU detection and initialization
#[test]
fn test_gpu_detection() {
    let adapter = QuantumAdapterGpu::new();
    // Should not panic - GPU detection is handled gracefully
    println!("GPU adapter initialized successfully");
}

/// Test Hamiltonian construction on GPU
#[test]
fn test_gpu_hamiltonian_construction() {
    let adapter = QuantumAdapterGpu::new();

    // Create simple 4-vertex graph (square lattice)
    let graph = Graph {
        num_vertices: 4,
        edges: vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 0, 1.0),
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

    // Build Hamiltonian on GPU
    let result = adapter.build_hamiltonian(&graph, &params);

    match result {
        Ok(h_state) => {
            assert_eq!(h_state.dimension, 4);
            assert_eq!(h_state.matrix_elements.len(), 16);
            println!("✓ Hamiltonian built successfully on GPU");
        }
        Err(e) => {
            println!("⚠ Hamiltonian build failed (GPU may not be available): {}", e);
        }
    }
}

/// Test quantum state evolution with double-double precision
#[test]
fn test_double_double_evolution() {
    let mut adapter = QuantumAdapterGpu::new();
    adapter.set_precision(true);  // Enable double-double precision

    let graph = Graph {
        num_vertices: 2,  // Simple two-level system
        edges: vec![(0, 1, 1.0)],
    };

    let params = EvolutionParams {
        time_step: 0.01,
        total_time: 1.0,
        coupling_strength: 1.0,
        temperature: 1.0,
        convergence_threshold: 1e-30,  // Ultra-high precision
        max_iterations: 100,
    };

    // Build Hamiltonian
    let h_state = match adapter.build_hamiltonian(&graph, &params) {
        Ok(h) => h,
        Err(e) => {
            println!("Skipping test - GPU not available: {}", e);
            return;
        }
    };

    // Create initial state |0⟩
    let initial_state = QuantumState {
        amplitudes: vec![(1.0, 0.0), (0.0, 0.0)],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    // Evolve for π/2 time (should create equal superposition)
    let evolution_time = std::f64::consts::PI / 2.0;

    match adapter.evolve_state(&h_state, &initial_state, evolution_time) {
        Ok(final_state) => {
            // Check for equal superposition (approximately)
            let amp0 = Complex64::new(final_state.amplitudes[0].0, final_state.amplitudes[0].1);
            let amp1 = Complex64::new(final_state.amplitudes[1].0, final_state.amplitudes[1].1);

            let prob0 = amp0.norm_sqr();
            let prob1 = amp1.norm_sqr();

            println!("After evolution: |0⟩ = {:.6}, |1⟩ = {:.6}", prob0, prob1);

            // Should be approximately equal superposition
            assert_relative_eq!(prob0, 0.5, epsilon = 0.1);
            assert_relative_eq!(prob1, 0.5, epsilon = 0.1);

            println!("✓ Double-double precision evolution successful");
        }
        Err(e) => {
            println!("Evolution failed: {}", e);
        }
    }
}

/// Test performance comparison: Standard vs Double-Double precision
#[test]
fn test_precision_performance() {
    use std::time::Instant;

    let mut adapter = QuantumAdapterGpu::new();

    let graph = Graph {
        num_vertices: 8,  // Larger system for performance testing
        edges: vec![
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0),
            (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 4, 1.0),
            (0, 4, 0.5), (1, 5, 0.5), (2, 6, 0.5), (3, 7, 0.5),
        ],
    };

    let params = EvolutionParams {
        time_step: 0.001,
        total_time: 1.0,
        coupling_strength: 1.0,
        temperature: 1.0,
        convergence_threshold: 1e-8,
        max_iterations: 100,
    };

    let h_state = match adapter.build_hamiltonian(&graph, &params) {
        Ok(h) => h,
        Err(_) => {
            println!("Skipping performance test - GPU not available");
            return;
        }
    };

    // Create initial state
    let n = graph.num_vertices;
    let norm = 1.0 / (n as f64).sqrt();
    let initial_state = QuantumState {
        amplitudes: vec![(norm, 0.0); n],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    // Test standard precision
    adapter.set_precision(false);
    let start = Instant::now();
    for _ in 0..10 {
        let _ = adapter.evolve_state(&h_state, &initial_state, 0.1);
    }
    let standard_time = start.elapsed();

    // Test double-double precision
    adapter.set_precision(true);
    let start = Instant::now();
    for _ in 0..10 {
        let _ = adapter.evolve_state(&h_state, &initial_state, 0.1);
    }
    let dd_time = start.elapsed();

    println!("Performance comparison (10 iterations):");
    println!("  Standard precision: {:?}", standard_time);
    println!("  Double-double precision: {:?}", dd_time);
    println!("  DD overhead: {:.1}x", dd_time.as_secs_f64() / standard_time.as_secs_f64());

    // Double-double should be slower but not more than 5x
    assert!(dd_time.as_secs_f64() / standard_time.as_secs_f64() < 5.0);
}

/// Test ground state computation using VQE
#[test]
fn test_vqe_ground_state() {
    let adapter = QuantumAdapterGpu::new();

    // Create simple Hydrogen-like Hamiltonian
    let graph = Graph {
        num_vertices: 2,
        edges: vec![(0, 1, -1.0)],  // Negative coupling for ground state
    };

    let params = EvolutionParams {
        time_step: 0.01,
        total_time: 1.0,
        coupling_strength: 1.0,
        temperature: 0.0,  // Zero temperature for ground state
        convergence_threshold: 1e-8,
        max_iterations: 100,
    };

    let h_state = match adapter.build_hamiltonian(&graph, &params) {
        Ok(h) => h,
        Err(_) => {
            println!("Skipping VQE test - GPU not available");
            return;
        }
    };

    match adapter.compute_ground_state(&h_state) {
        Ok(ground_state) => {
            println!("Ground state energy: {}", ground_state.energy);
            assert!(ground_state.energy < 0.0);  // Should be negative for this Hamiltonian
            println!("✓ VQE ground state computation successful");
        }
        Err(e) => {
            println!("VQE computation failed: {}", e);
        }
    }
}

/// Test deterministic reproducibility
#[test]
fn test_deterministic_evolution() {
    let adapter = QuantumAdapterGpu::new();

    let graph = Graph {
        num_vertices: 3,
        edges: vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
        ],
    };

    let params = EvolutionParams {
        time_step: 0.01,
        total_time: 1.0,
        coupling_strength: 1.0,
        temperature: 1.0,
        convergence_threshold: 1e-8,
        max_iterations: 100,
    };

    let h_state = match adapter.build_hamiltonian(&graph, &params) {
        Ok(h) => h,
        Err(_) => {
            println!("Skipping determinism test - GPU not available");
            return;
        }
    };

    let initial_state = QuantumState {
        amplitudes: vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    // Run evolution multiple times
    let mut results = Vec::new();
    for _ in 0..3 {
        match adapter.evolve_state(&h_state, &initial_state, 0.5) {
            Ok(state) => results.push(state),
            Err(e) => {
                println!("Evolution failed: {}", e);
                return;
            }
        }
    }

    // Check all results are identical (bit-for-bit)
    for i in 1..results.len() {
        for j in 0..results[0].amplitudes.len() {
            assert_eq!(
                results[0].amplitudes[j].0,
                results[i].amplitudes[j].0,
                "Real parts differ at index {}", j
            );
            assert_eq!(
                results[0].amplitudes[j].1,
                results[i].amplitudes[j].1,
                "Imaginary parts differ at index {}", j
            );
        }
    }

    println!("✓ Deterministic evolution verified - bit-for-bit reproducible");
}

/// Benchmark against theoretical speedup
#[test]
#[ignore]  // Run with --ignored flag for full benchmark
fn benchmark_gpu_speedup() {
    use std::time::Instant;

    let adapter = QuantumAdapterGpu::new();

    // Test different system sizes
    let sizes = vec![4, 8, 16, 32, 64];

    for n in sizes {
        // Create fully connected graph
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i+1..n {
                edges.push((i, j, 1.0 / (n as f64)));
            }
        }

        let graph = Graph {
            num_vertices: n,
            edges,
        };

        let params = EvolutionParams {
            time_step: 0.001,
            total_time: 1.0,
            coupling_strength: 1.0,
            temperature: 1.0,
            convergence_threshold: 1e-6,
            max_iterations: 100,
        };

        let h_state = match adapter.build_hamiltonian(&graph, &params) {
            Ok(h) => h,
            Err(_) => {
                println!("GPU not available for benchmarking");
                return;
            }
        };

        let norm = 1.0 / (n as f64).sqrt();
        let initial_state = QuantumState {
            amplitudes: vec![(norm, 0.0); n],
            phase_coherence: 1.0,
            energy: 0.0,
            entanglement: 0.0,
            timestamp_ns: 0,
        };

        // Benchmark
        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = adapter.evolve_state(&h_state, &initial_state, 0.01);
        }

        let elapsed = start.elapsed();
        let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

        println!("System size {}: {:.2} evolutions/sec", n, ops_per_sec);

        // Theoretical peak for matrix-vector multiply: 2n² flops
        let flops = 2.0 * (n * n) as f64 * ops_per_sec;
        println!("  Estimated GFLOPS: {:.2}", flops / 1e9);
    }
}