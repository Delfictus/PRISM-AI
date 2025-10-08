use prct_core::parse_mtx_file;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Testing MTX Parser ===\n");
    
    let test_file = "benchmarks/dimacs_official/DSJC500-5.mtx";
    println!("Parsing: {}", test_file);
    
    let start = std::time::Instant::now();
    let graph = parse_mtx_file(test_file)?;
    let load_time = start.elapsed();
    
    println!("\n✅ Successfully parsed!");
    println!("Load time: {:?}", load_time);
    println!("\nGraph Statistics:");
    println!("  Vertices: {}", graph.num_vertices);
    println!("  Edges: {}", graph.num_edges);
    println!("  Density: {:.2}%", 
        (graph.num_edges as f64 / (graph.num_vertices * (graph.num_vertices - 1) / 2) as f64) * 100.0);
    
    println!("\nBest Known Result:");
    println!("  Colors: 47-48 (optimal unknown)");
    println!("\nOur Goal:");
    println!("  Beat 47 colors for world-record potential!");
    
    Ok(())
}
