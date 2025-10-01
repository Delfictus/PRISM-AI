//! Test full PRCT.solve() with tiny graph
//!
//! This tests the complete algorithm pipeline

use prct_core::{PRCTAlgorithm, PRCTConfig};
use prct_adapters::{NeuromorphicAdapter, QuantumAdapter, CouplingAdapter};
use shared_types::Graph;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("\n=== Full PRCT.solve() Test (K4 Graph) ===\n");

    // Create K3 (triangle) - simpler for initial testing
    let edges = vec![
        (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0),
    ];
    let n = 3;
    let mut adjacency = vec![false; n * n];
    for &(u, v, _) in &edges {
        adjacency[u * n + v] = true;
        adjacency[v * n + u] = true;
    }

    let graph = Graph {
        num_vertices: n,
        num_edges: edges.len(),
        edges,
        adjacency,
        coordinates: None,
    };

    println!("📊 Graph: K3 (triangle)");
    println!("   Vertices: {}", graph.num_vertices);
    println!("   Edges: {}", graph.num_edges);
    println!("   Expected colors: 3 (triangle)");

    // Create adapters
    println!("\n🔌 Creating adapters...");
    let neuro = Arc::new(NeuromorphicAdapter::new()?);
    let quantum = Arc::new(QuantumAdapter::new());
    let coupling = Arc::new(CouplingAdapter::new());
    println!("   ✓ All adapters ready");

    // Create PRCT algorithm
    println!("\n⚙️  Creating PRCT algorithm...");
    let config = PRCTConfig {
        target_colors: 3,
        quantum_evolution_time: 0.0001, // Ultra-short for stability
        kuramoto_coupling: 0.5,
        quantum_params: shared_types::EvolutionParams {
            dt: 0.001,  // Very small time step
            strength: 0.1,  // Weak coupling
            damping: 0.1,
            temperature: 300.0,
        },
        ..Default::default()
    };
    let prct = PRCTAlgorithm::new(neuro, quantum, coupling, config);
    println!("   ✓ PRCT configured");

    // Solve
    println!("\n🚀 Running PRCT.solve()...");
    println!("   (This tests all 3 layers + physics coupling)");
    let start = Instant::now();

    let solution = prct.solve(&graph)?;

    let elapsed = start.elapsed();
    println!("   ✓ Solved in {:?}", elapsed);

    // Display results
    println!("\n📈 Solution:");
    println!("   Colors used: {}", solution.coloring.chromatic_number);
    println!("   Coloring: {:?}", solution.coloring.colors);
    println!("   Conflicts: {}", solution.coloring.conflicts);
    println!("   Quality: {:.4}", solution.coloring.quality_score);

    // Verify coloring
    println!("\n🔍 Verifying coloring...");
    let mut is_valid = true;
    let mut conflict_count = 0;
    for &(u, v, _) in &graph.edges {
        if solution.coloring.colors[u] == solution.coloring.colors[v] {
            println!("   ❌ Conflict: vertices {} and {} both have color {}",
                     u, v, solution.coloring.colors[u]);
            is_valid = false;
            conflict_count += 1;
        }
    }

    if is_valid {
        println!("   ✅ Valid coloring! No adjacent vertices share colors");
    } else {
        println!("   ⚠️  Invalid coloring: {} conflicts found", conflict_count);
    }

    println!("\n📊 TSP Tours:");
    for (idx, tour) in solution.color_class_tours.iter().enumerate() {
        println!("   Color {}: tour length = {:.2}", idx, tour.tour_length);
    }

    println!("\n🔬 Physics Metrics:");
    println!("   Phase coherence: {:.4}", solution.phase_coherence);
    println!("   Kuramoto order: {:.4}", solution.kuramoto_order);
    println!("   Overall quality: {:.4}", solution.overall_quality);
    println!("   Total time: {:.2}ms", solution.total_time_ms);

    println!("\n✅ Full PRCT Pipeline Test Complete!");

    Ok(())
}
