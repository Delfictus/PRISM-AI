//! Test encode_graph_as_spikes timing

use prct_adapters::NeuromorphicAdapter;
use prct_core::ports::*;
use shared_types::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("\n=== Spike Encoding Timing Test ===\n");

    // Create tiny graph
    let edges = vec![(0, 1, 1.0)];
    let n = 2;
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

    println!("📊 Graph: {} vertices, {} edges", graph.num_vertices, graph.num_edges);

    // Test encoding
    println!("\n🧠 Creating NeuromorphicAdapter...");
    let start = Instant::now();
    let neuro = NeuromorphicAdapter::new()?;
    println!("   ✓ Adapter created in {:?}", start.elapsed());

    println!("\n⚡ Encoding graph as spikes...");
    let start = Instant::now();
    let params = NeuromorphicEncodingParams::default();
    let spikes = neuro.encode_graph_as_spikes(&graph, &params)?;
    println!("   ✓ Encoded in {:?}", start.elapsed());
    println!("   ✓ Generated {} spikes", spikes.spikes.len());

    println!("\n✅ Encoding completed successfully!");

    Ok(())
}
