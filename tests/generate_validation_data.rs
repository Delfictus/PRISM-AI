//! Generate validation data for cross-validation with QuTiP
//!
//! Runs quantum evolution on GPU and exports results for Python validation

use prism_ai::*;
use shared_types::*;
use prct_core::ports::QuantumPort;
use prct_adapters::quantum_adapter_gpu::QuantumAdapterGpu;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Serialize, Deserialize)]
struct ValidationData {
    edges: Vec<(usize, usize)>,
    weights: Vec<f64>,
    initial_state: Vec<Vec<f64>>,  // [real, imag] for each amplitude
    evolution_time: f64,
    final_state: Vec<Vec<f64>>,
    execution_time: f64,
    gpu_device: Option<String>,
    precision_mode: String,
}

/// Generate validation data for two-level system
#[test]
fn generate_two_level_validation() {
    let adapter = QuantumAdapterGpu::new();

    let graph = Graph {
        num_vertices: 2,
        edges: vec![(0, 1, 1.0)],
    };

    let params = EvolutionParams {
        time_step: 0.01,
        total_time: 1.0,
        coupling_strength: 1.0,
        temperature: 1.0,
        convergence_threshold: 1e-10,
        max_iterations: 100,
    };

    let h_state = match adapter.build_hamiltonian(&graph, &params) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("GPU not available: {}", e);
            return;
        }
    };

    // Initial state |0⟩
    let initial_state = QuantumState {
        amplitudes: vec![(1.0, 0.0), (0.0, 0.0)],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    let evolution_time = std::f64::consts::PI / 2.0;

    let start = Instant::now();
    let final_state = adapter
        .evolve_state(&h_state, &initial_state, evolution_time)
        .expect("Evolution failed");
    let exec_time = start.elapsed().as_secs_f64();

    // Prepare validation data
    let validation = ValidationData {
        edges: vec![(0, 1)],
        weights: vec![1.0],
        initial_state: initial_state.amplitudes.iter()
            .map(|&(re, im)| vec![re, im])
            .collect(),
        evolution_time,
        final_state: final_state.amplitudes.iter()
            .map(|&(re, im)| vec![re, im])
            .collect(),
        execution_time: exec_time,
        gpu_device: Some("GPU".to_string()),
        precision_mode: "standard".to_string(),
    };

    // Save to JSON
    let json = serde_json::to_string_pretty(&validation).unwrap();
    let mut file = File::create("validation/two_level_gpu.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("✓ Two-level validation data generated");
    println!("  Execution time: {:.3} ms", exec_time * 1000.0);
}

/// Generate validation data for square lattice
#[test]
fn generate_square_lattice_validation() {
    let adapter = QuantumAdapterGpu::new();

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
        convergence_threshold: 1e-10,
        max_iterations: 100,
    };

    let h_state = match adapter.build_hamiltonian(&graph, &params) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("GPU not available: {}", e);
            return;
        }
    };

    // Initial state |0⟩
    let initial_state = QuantumState {
        amplitudes: vec![
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    let evolution_time = 1.0;

    let start = Instant::now();
    let final_state = adapter
        .evolve_state(&h_state, &initial_state, evolution_time)
        .expect("Evolution failed");
    let exec_time = start.elapsed().as_secs_f64();

    // Prepare validation data
    let validation = ValidationData {
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        weights: vec![1.0; 4],
        initial_state: initial_state.amplitudes.iter()
            .map(|&(re, im)| vec![re, im])
            .collect(),
        evolution_time,
        final_state: final_state.amplitudes.iter()
            .map(|&(re, im)| vec![re, im])
            .collect(),
        execution_time: exec_time,
        gpu_device: Some("GPU".to_string()),
        precision_mode: "standard".to_string(),
    };

    // Save to JSON
    let json = serde_json::to_string_pretty(&validation).unwrap();
    let mut file = File::create("validation/square_lattice_gpu.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("✓ Square lattice validation data generated");
    println!("  Execution time: {:.3} ms", exec_time * 1000.0);
}

/// Generate validation data with double-double precision
#[test]
fn generate_double_double_validation() {
    let mut adapter = QuantumAdapterGpu::new();
    adapter.set_precision(true);  // Enable DD precision

    let graph = Graph {
        num_vertices: 3,
        edges: vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
        ],
    };

    let params = EvolutionParams {
        time_step: 0.001,
        total_time: 1.0,
        coupling_strength: 1.0,
        temperature: 1.0,
        convergence_threshold: 1e-30,  // Ultra-high precision
        max_iterations: 1000,
    };

    let h_state = match adapter.build_hamiltonian(&graph, &params) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("GPU not available: {}", e);
            return;
        }
    };

    // Initial uniform superposition
    let norm = 1.0 / (3.0_f64).sqrt();
    let initial_state = QuantumState {
        amplitudes: vec![(norm, 0.0); 3],
        phase_coherence: 1.0,
        energy: 0.0,
        entanglement: 0.0,
        timestamp_ns: 0,
    };

    let evolution_time = 2.0;

    let start = Instant::now();
    let final_state = adapter
        .evolve_state(&h_state, &initial_state, evolution_time)
        .expect("Evolution failed");
    let exec_time = start.elapsed().as_secs_f64();

    // Prepare validation data
    let validation = ValidationData {
        edges: vec![(0, 1), (1, 2), (2, 0)],
        weights: vec![1.0; 3],
        initial_state: initial_state.amplitudes.iter()
            .map(|&(re, im)| vec![re, im])
            .collect(),
        evolution_time,
        final_state: final_state.amplitudes.iter()
            .map(|&(re, im)| vec![re, im])
            .collect(),
        execution_time: exec_time,
        gpu_device: Some("GPU".to_string()),
        precision_mode: "double_double".to_string(),
    };

    // Save to JSON
    let json = serde_json::to_string_pretty(&validation).unwrap();
    let mut file = File::create("validation/triangular_dd_gpu.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("✓ Double-double validation data generated");
    println!("  Execution time: {:.3} ms", exec_time * 1000.0);
    println!("  Precision: 106-bit (double-double)");
}

/// Benchmark scaling for different system sizes
#[test]
fn benchmark_gpu_scaling() {
    let adapter = QuantumAdapterGpu::new();

    let sizes = vec![2, 4, 8, 16, 32];
    let mut results = Vec::new();

    for n in &sizes {
        // Create fully connected graph
        let mut edges = Vec::new();
        for i in 0..*n {
            for j in i+1..*n {
                edges.push((i, j, 1.0 / (*n as f64)));
            }
        }

        let graph = Graph {
            num_vertices: *n,
            edges,
        };

        let params = EvolutionParams {
            time_step: 0.001,
            total_time: 0.1,
            coupling_strength: 1.0,
            temperature: 1.0,
            convergence_threshold: 1e-8,
            max_iterations: 100,
        };

        let h_state = match adapter.build_hamiltonian(&graph, &params) {
            Ok(h) => h,
            Err(_) => continue,
        };

        // Uniform superposition
        let norm = 1.0 / (*n as f64).sqrt();
        let initial_state = QuantumState {
            amplitudes: vec![(norm, 0.0); *n],
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

        let total_time = start.elapsed().as_secs_f64();
        let time_per_evolution = total_time / iterations as f64;

        results.push((*n, time_per_evolution));

        println!("System size {}: {:.3} ms per evolution", n, time_per_evolution * 1000.0);
    }

    // Save benchmark results
    let mut file = File::create("validation/gpu_scaling_benchmark.txt").unwrap();
    writeln!(file, "# GPU Scaling Benchmark").unwrap();
    writeln!(file, "# Size, Time(ms)").unwrap();
    for (size, time) in &results {
        writeln!(file, "{}, {:.6}", size, time * 1000.0).unwrap();
    }

    println!("\n✓ Scaling benchmark complete");

    // Calculate speedup trend
    if results.len() >= 2 {
        let small_time = results[0].1;
        let large_time = results[results.len()-1].1;
        let size_ratio = results[results.len()-1].0 as f64 / results[0].0 as f64;
        let time_ratio = large_time / small_time;

        println!("  Size increased: {:.1}x", size_ratio);
        println!("  Time increased: {:.1}x", time_ratio);
        println!("  Scaling efficiency: {:.1}%", 100.0 * size_ratio / time_ratio);
    }
}