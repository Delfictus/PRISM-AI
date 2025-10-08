use prct_core::parse_mtx_file;

fn main() -> anyhow::Result<()> {
    println!("Testing MTX parser on DSJC500-5.mtx...");
    
    let graph = parse_mtx_file("benchmarks/dimacs_official/DSJC500-5.mtx")?;
    
    println!("✅ Successfully parsed!");
    println!("Vertices: {}", graph.num_vertices);
    println!("Edges: {}", graph.num_edges);
    println!("Adjacency matrix size: {}", graph.adjacency.len());
    
    Ok(())
}
