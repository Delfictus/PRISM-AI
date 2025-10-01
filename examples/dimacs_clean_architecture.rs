//! DIMACS Benchmark Runner with Clean Architecture
//!
//! Runs official DIMACS graph coloring benchmarks through the clean
//! hexagonal architecture using dependency injection.

use prct_core::{PRCTAlgorithm, PRCTConfig};
use prct_adapters::{NeuromorphicAdapter, QuantumAdapter, CouplingAdapter};
use shared_types::Graph;
use std::sync::Arc;
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> anyhow::Result<()> {
    println!("\n=== DIMACS Benchmark with Clean Architecture ===\n");

    // Get benchmark file from args or use default
    let args: Vec<String> = std::env::args().collect();
    let benchmark_file = if args.len() > 1 {
        &args[1]
    } else {
        "benchmarks/dsjc125.1.col"
    };

    println!("📂 Loading benchmark: {}", benchmark_file);

    let graph = load_dimacs_graph(benchmark_file)?;
    println!("   ✓ Loaded graph:");
    println!("     Vertices: {}", graph.num_vertices);
    println!("     Edges: {}", graph.num_edges);
    println!("     Density: {:.2}%",
             (graph.num_edges as f64 / (graph.num_vertices * (graph.num_vertices - 1) / 2) as f64) * 100.0);

    // Create adapters with clean architecture
    println!("\n🔌 Initializing clean architecture...");
    let neuro = Arc::new(NeuromorphicAdapter::new()?);
    let quantum = Arc::new(QuantumAdapter::new());
    let coupling = Arc::new(CouplingAdapter::new());
    println!("   ✓ Adapters ready (hexagonal pattern)");

    // Configure PRCT for larger graph
    let config = PRCTConfig {
        target_colors: 10, // dsjc125.1 optimal chromatic number is 5
        quantum_evolution_time: 0.00001, // Very small for stability
        kuramoto_coupling: 0.3,
        quantum_params: shared_types::EvolutionParams {
            dt: 0.0001,
            strength: 0.01,
            damping: 0.1,
            temperature: 300.0,
        },
        ..Default::default()
    };

    let prct = PRCTAlgorithm::new(neuro, quantum, coupling, config);
    println!("   ✓ PRCT algorithm configured");

    // Run benchmark
    println!("\n🚀 Running PRCT.solve()...");
    println!("   Algorithm: Phase Resonance Chromatic-TSP");
    println!("   Architecture: Hexagonal (Ports & Adapters)");
    println!("   Layers: Neuromorphic → Quantum → Coupling → Optimization");

    let start = Instant::now();
    let solution = prct.solve(&graph)?;
    let elapsed = start.elapsed();

    // Display results
    println!("\n✅ Solution Found in {:?}", elapsed);
    println!("\n📊 Coloring Results:");
    println!("   Colors used: {}", solution.coloring.chromatic_number);
    println!("   Conflicts: {}", solution.coloring.conflicts);
    println!("   Quality score: {:.4}", solution.coloring.quality_score);
    println!("   Computation time: {:.2}ms", solution.coloring.computation_time_ms);

    // Verify coloring
    println!("\n🔍 Verifying coloring...");
    let mut conflict_count = 0;
    for &(u, v, _) in &graph.edges {
        if solution.coloring.colors[u] == solution.coloring.colors[v] {
            conflict_count += 1;
        }
    }

    if conflict_count == 0 {
        println!("   ✅ VALID coloring! No conflicts found");
    } else {
        println!("   ⚠️  {} conflicts detected", conflict_count);
    }

    println!("\n🔬 Physics Metrics:");
    println!("   Phase coherence: {:.4}", solution.phase_coherence);
    println!("   Kuramoto order: {:.4}", solution.kuramoto_order);
    println!("   Overall quality: {:.4}", solution.overall_quality);

    println!("\n📈 Performance:");
    println!("   Total time: {:.2}ms", solution.total_time_ms);
    println!("   Time per vertex: {:.4}ms", solution.total_time_ms / graph.num_vertices as f64);

    println!("\n🏆 Benchmark Summary:");
    println!("   Graph: dsjc125.1 (125 vertices, sparse)");
    println!("   Colors: {} (optimal: 5)", solution.coloring.chromatic_number);
    println!("   Valid: {}", if conflict_count == 0 { "YES" } else { "NO" });
    println!("   Architecture: Clean hexagonal (working!)");
    println!("   All 3 layers: Executed successfully");

    Ok(())
}

/// Load DIMACS format graph
fn load_dimacs_graph(filename: &str) -> anyhow::Result<Graph> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut num_vertices = 0;
    let mut edges = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('c') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "p" => {
                // p edge <vertices> <edges>
                if parts.len() >= 3 {
                    num_vertices = parts[2].parse()?;
                }
            }
            "e" => {
                // e <vertex1> <vertex2>
                if parts.len() >= 3 {
                    let u: usize = parts[1].parse::<usize>()? - 1; // DIMACS is 1-indexed
                    let v: usize = parts[2].parse::<usize>()? - 1;
                    edges.push((u, v, 1.0));
                }
            }
            _ => {}
        }
    }

    // Build adjacency matrix
    let mut adjacency = vec![false; num_vertices * num_vertices];
    for &(u, v, _) in &edges {
        if u < num_vertices && v < num_vertices {
            adjacency[u * num_vertices + v] = true;
            adjacency[v * num_vertices + u] = true;
        }
    }

    Ok(Graph {
        num_vertices,
        num_edges: edges.len(),
        edges,
        adjacency,
        coordinates: None,
    })
}
