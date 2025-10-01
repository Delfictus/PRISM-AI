//! Test reservoir processing timing

use prct_adapters::NeuromorphicAdapter;
use prct_core::ports::*;
use shared_types::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("\n=== Reservoir Processing Timing Test ===\n");

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

    let neuro = NeuromorphicAdapter::new()?;
    println!("✓ Adapter created");

    // Encode
    println!("\n⚡ Encoding...");
    let start = Instant::now();
    let params = NeuromorphicEncodingParams::default();
    let spikes = neuro.encode_graph_as_spikes(&graph, &params)?;
    println!("   ✓ Encoded in {:?} ({} spikes)", start.elapsed(), spikes.spikes.len());

    // Process - this is likely the bottleneck
    println!("\n🧠 Processing with reservoir (THIS MAY BE SLOW)...");
    let start = Instant::now();
    let neuro_state = neuro.process_and_detect_patterns(&spikes)?;
    println!("   ✓ Processed in {:?}", start.elapsed());
    println!("   ✓ Coherence: {:.4}", neuro_state.coherence);

    println!("\n✅ Processing completed!");

    Ok(())
}
