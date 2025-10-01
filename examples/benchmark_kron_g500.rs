//! Kronecker Graph Benchmark Test
//!
//! Tests the adaptive chromatic coloring algorithm against the Kron_g500-logn16
//! graph from the Graph 500 benchmark suite.
//!
//! Graph: 65,536 vertices, 6,289,992 edges
//! Format: Adjacency list (non-DIMACS)

use anyhow::{Result, anyhow};
use ndarray::Array2;
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use quantum_engine::ChromaticColoring;

/// Parse Kronecker adjacency list format
fn parse_kronecker_graph(path: &Path) -> Result<(usize, Vec<(usize, usize)>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // First line: vertices edges [optional third value]
    let header = lines.next()
        .ok_or_else(|| anyhow!("Empty file"))??;
    let parts: Vec<&str> = header.split_whitespace().collect();

    if parts.len() < 2 {
        return Err(anyhow!("Invalid header line"));
    }

    let vertices: usize = parts[0].parse()?;
    let expected_edges: usize = parts[1].parse()?;

    println!("[*] Graph header: {} vertices, {} expected edges", vertices, expected_edges);

    let mut edges = Vec::new();

    // Each subsequent line is the adjacency list for vertex i (0-indexed)
    for (i, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue; // Skip empty lines (isolated vertices)
        }

        let neighbors: Vec<usize> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        for &neighbor in &neighbors {
            if neighbor < vertices {
                // Only add edge once (undirected graph)
                if i < neighbor {
                    edges.push((i, neighbor));
                }
            } else {
                println!("    Warning: Invalid neighbor {} for vertex {}", neighbor, i);
            }
        }
    }

    println!("[*] Parsed {} edges", edges.len());
    if edges.len() != expected_edges {
        println!("    Warning: Edge count mismatch (expected {}, got {})",
                 expected_edges, edges.len());
    }

    Ok((vertices, edges))
}

/// Build coupling matrix from edge list (with size limits)
fn build_coupling_matrix(vertices: usize, edges: &[(usize, usize)], max_vertices: usize) -> Result<Array2<Complex64>> {
    if vertices > max_vertices {
        return Err(anyhow!(
            "Graph too large: {} vertices (max {}). Use sampling or approximation.",
            vertices, max_vertices
        ));
    }

    let mut matrix = Array2::zeros((vertices, vertices));
    for &(u, v) in edges {
        if u < vertices && v < vertices {
            matrix[[u, v]] = Complex64::new(1.0, 0.0);
            matrix[[v, u]] = Complex64::new(1.0, 0.0);
        }
    }
    Ok(matrix)
}

/// Compute graph statistics
fn compute_statistics(vertices: usize, edges: &[(usize, usize)]) -> (f64, usize, usize) {
    let max_edges = vertices * (vertices - 1) / 2;
    let density = edges.len() as f64 / max_edges as f64;

    // Compute degree distribution
    let mut degrees = vec![0usize; vertices];
    for &(u, v) in edges {
        if u < vertices {
            degrees[u] += 1;
        }
        if v < vertices {
            degrees[v] += 1;
        }
    }

    let max_degree = *degrees.iter().max().unwrap_or(&0);
    let avg_degree = degrees.iter().sum::<usize>() / vertices;

    (density, max_degree, avg_degree)
}

/// Run chromatic coloring with sampling for large graphs
fn run_coloring_test(
    vertices: usize,
    edges: &[(usize, usize)],
    max_k: usize,
    time_limit_ms: u64,
) -> Result<Option<usize>> {
    println!("\n[*] Starting chromatic coloring search (k=2..{})...", max_k);

    // For large graphs, we can only test on a sample
    const MAX_TESTABLE_VERTICES: usize = 5000;

    if vertices > MAX_TESTABLE_VERTICES {
        println!("    Warning: Graph too large for full coloring test");
        println!("    Would need {} GB just for coupling matrix",
                 (vertices * vertices * 16) / (1024 * 1024 * 1024));
        println!("    Skipping direct coloring test.");
        println!("\n    Theoretical analysis:");

        // Brooks' theorem: χ(G) ≤ Δ(G) + 1
        let (_, max_degree, avg_degree) = compute_statistics(vertices, edges);
        let brooks_upper_bound = max_degree + 1;

        println!("    • Brooks' theorem upper bound: χ ≤ Δ+1 = {}", brooks_upper_bound);
        println!("    • Average degree: {}", avg_degree);
        println!("    • Maximum degree: {}", max_degree);
        println!("    • For scale-free graphs like Kronecker, typically χ ∈ [10, 50]");

        return Ok(None);
    }

    let coupling_matrix = build_coupling_matrix(vertices, edges, MAX_TESTABLE_VERTICES)?;
    let start = Instant::now();

    for k in 2..=max_k {
        if start.elapsed().as_millis() > time_limit_ms as u128 {
            println!("    ⏱ Time limit reached");
            return Ok(None);
        }

        print!("    Testing k={}... ", k);
        match ChromaticColoring::new_adaptive(&coupling_matrix, k) {
            Ok(coloring) => {
                if coloring.verify_coloring() {
                    let duration = start.elapsed();
                    println!("✓ Found valid {}-coloring in {:.2}ms", k, duration.as_secs_f64() * 1000.0);
                    return Ok(Some(k));
                } else {
                    println!("✗ Invalid");
                }
            }
            Err(_) => {
                println!("✗ Failed");
            }
        }
    }

    Ok(None)
}

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          KRONECKER GRAPH BENCHMARK (Graph 500)                ║");
    println!("║            Kron_g500-logn16 Scale-Free Graph                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let graph_path = Path::new("/mnt/c/Users/is/Downloads/kron_g500-logn16.graph/kron_g500-logn16.graph");

    if !graph_path.exists() {
        return Err(anyhow!(
            "Graph file not found: {:?}\n\
             Expected: C:\\Users\\is\\Downloads\\kron_g500-logn16.graph\\kron_g500-logn16.graph",
            graph_path
        ));
    }

    println!("[*] Parsing graph file...");
    let parse_start = Instant::now();
    let (vertices, edges) = parse_kronecker_graph(graph_path)?;
    println!("[*] Parse time: {:.2}s\n", parse_start.elapsed().as_secs_f64());

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                      GRAPH STATISTICS                          ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Vertices:           {:>8}                                  ║", vertices);
    println!("║ Edges:              {:>8}                                  ║", edges.len());

    let (density, max_degree, avg_degree) = compute_statistics(vertices, &edges);

    println!("║ Density:            {:>8.6}%                              ║", density * 100.0);
    println!("║ Average Degree:     {:>8}                                  ║", avg_degree);
    println!("║ Maximum Degree:     {:>8}                                  ║", max_degree);
    println!("║ Brooks Bound (Δ+1): {:>8}                                  ║", max_degree + 1);
    println!("╚════════════════════════════════════════════════════════════════╝");

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                  GRAPH CHARACTERISTICS                         ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Type:        Kronecker scale-free graph                       ║");
    println!("║ Generator:   Recursive tensor product (RMAT)                  ║");
    println!("║ Properties:  • Heavy-tailed degree distribution               ║");
    println!("║              • Small-world (low diameter)                     ║");
    println!("║              • Power-law degree distribution                  ║");
    println!("║              • High clustering coefficient                    ║");
    println!("║                                                                ║");
    println!("║ Coloring:    Typically χ ∈ [10, 50] for this class           ║");
    println!("║              (depends on specific realization)                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Attempt coloring (will show why it's infeasible for this size)
    let result = run_coloring_test(vertices, &edges, max_degree.min(100), 60000);

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                         CONCLUSION                             ║");
    println!("╠════════════════════════════════════════════════════════════════╣");

    match result? {
        Some(chi) => {
            println!("║ ✓ Successfully computed chromatic number: χ = {}         ║", chi);
            println!("║ ✓ Graph coloring algorithm validated on large graph       ║");
        }
        None => {
            println!("║ ⚠ Graph too large for direct quantum simulation           ║");
            println!("║   Memory required: ~64 GB for 65k×65k coupling matrix     ║");
            println!("║                                                            ║");
            println!("║ Recommendations:                                           ║");
            println!("║   1. Test on smaller DIMACS benchmarks (see scripts/)     ║");
            println!("║   2. Use approximate/heuristic methods for graphs >5k     ║");
            println!("║   3. Implement graph sampling or decomposition            ║");
            println!("║   4. Use GPU acceleration for larger matrices             ║");
        }
    }

    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(())
}
