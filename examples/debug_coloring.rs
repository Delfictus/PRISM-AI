//! Debug chromatic coloring to understand threshold behavior

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use quantum_engine::ChromaticColoring;

fn parse_col_file(path: &Path) -> Result<(usize, Vec<(usize, usize)>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut vertices = 0;
    let mut edges = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.get(0) {
            Some(&"p") => {
                if parts.len() >= 3 && parts[1] == "edge" {
                    vertices = parts[2].parse()?;
                }
            }
            Some(&"e") => {
                if parts.len() >= 3 {
                    let u: usize = parts[1].parse()?;
                    let v: usize = parts[2].parse()?;
                    edges.push((u - 1, v - 1));
                }
            }
            _ => {}
        }
    }
    Ok((vertices, edges))
}

fn build_coupling_matrix(vertices: usize, edges: &[(usize, usize)]) -> Array2<Complex64> {
    let mut matrix = Array2::zeros((vertices, vertices));
    for &(u, v) in edges {
        if u < vertices && v < vertices {
            matrix[[u, v]] = Complex64::new(1.0, 0.0);
            matrix[[v, u]] = Complex64::new(1.0, 0.0);
        }
    }
    matrix
}

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           CHROMATIC COLORING DEBUG ANALYSIS                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let test_file = "benchmarks/dsjc125.1.col";
    println!("[*] Testing: {}", test_file);

    let (vertices, edges) = parse_col_file(Path::new(test_file))?;
    println!("[*] Graph: {} vertices, {} edges", vertices, edges.len());
    println!("[*] Expected χ: 5 (known best)\n");

    let coupling_matrix = build_coupling_matrix(vertices, &edges);

    // Test different k values
    for k in 2..=10 {
        println!("Testing k={}", k);

        match ChromaticColoring::new_adaptive(&coupling_matrix, k) {
            Ok(coloring) => {
                let valid = coloring.verify_coloring();
                let conflicts = coloring.get_conflict_count();

                // Get threshold analysis
                match ChromaticColoring::analyze_threshold(&coupling_matrix, k) {
                    Ok(analysis) => {
                        println!("  Threshold: {:.6}", analysis.optimal_threshold);
                        println!("  Edges at threshold: {}", analysis.num_edges_at_threshold);
                        println!("  Graph density: {:.2}%", analysis.graph_density * 100.0);
                        println!("  Max degree: {}", analysis.estimated_chromatic_number - 1);
                        println!("  Valid: {}", valid);
                        println!("  Conflicts: {}", conflicts);

                        if valid {
                            println!("  ✓ Found valid {}-coloring\n", k);
                            break;
                        } else {
                            println!("  ✗ Invalid coloring\n");
                        }
                    }
                    Err(e) => println!("  Error analyzing: {}\n", e),
                }
            }
            Err(e) => {
                println!("  ✗ Failed: {}\n", e);
            }
        }
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    DIAGNOSIS                                   ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ If edges at threshold ≈ 0:                                    ║");
    println!("║   → Threshold is TOO HIGH, excluding all edges                ║");
    println!("║   → Creating trivial empty graph (always 2-colorable)         ║");
    println!("║                                                                ║");
    println!("║ If edges at threshold ≈ actual edges:                         ║");
    println!("║   → Threshold is correct                                      ║");
    println!("║   → Algorithm is working properly                             ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(())
}
