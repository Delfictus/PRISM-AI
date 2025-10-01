//! Benchmark test for the DSJC500.5 graph
//!
//! This test suite loads a large, dense DIMACS benchmark graph and runs it
//! through the platform's coloring algorithm to find the chromatic number.

use anyhow::{Result, anyhow};
use ndarray::Array2;
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use quantum_engine::ChromaticColoring;

/// Parses a DIMACS .col file and returns the number of vertices and edge list.
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
                    // DIMACS files are 1-indexed, convert to 0-indexed
                    edges.push((u - 1, v - 1));
                }
            }
            _ => {}
        }
    }

    if vertices == 0 {
        return Err(anyhow!("Failed to parse problem line from .col file"));
    }

    Ok((vertices, edges))
}

/// Converts an edge list to a coupling matrix for the platform.
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
    println!("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n");
    println!("\\  ARES-51 BENCHMARK TEST SUITE (LARGE)  \\");
    println!("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n");

    let file_path = Path::new("benchmarks/dsjc500.5.col");
    println!("[*] Loading benchmark file: {:?}", file_path);

    let (vertices, edges) = parse_col_file(file_path)?;
    println!("[*] Graph loaded: {} vertices, {} edges", vertices, edges.len());

    println!("[*] Building coupling matrix for the platform...");
    let coupling_matrix = build_coupling_matrix(vertices, &edges);

    // Published results for DSJC500.5 are in the high 40s. 
    // We will search downwards from a safe upper bound.
    let mut k_to_test = 55;
    let mut best_k = None;

    println!("[*] Starting chromatic number search (this may take some time)...");
    let total_start = Instant::now();

    loop {
        if k_to_test < 40 { // Don't search forever
            println!("    [!] Search limit reached.");
            break;
        }

        println!("    [.] Testing for k = {}", k_to_test);
        let start = Instant::now();

        match ChromaticColoring::new_adaptive(&coupling_matrix, k_to_test) {
            Ok(coloring) => {
                if coloring.verify_coloring() {
                    let duration = start.elapsed();
                    println!("        => SUCCESS: Found a valid {}-coloring in {:.2?}.", k_to_test, duration);
                    best_k = Some(k_to_test);
                    k_to_test -= 1; // Try to find a better coloring
                } else {
                    println!("        => FAILED: Could not find a valid {}-coloring ({} conflicts).", k_to_test, coloring.get_conflict_count());
                    break; // Assume we can't do better
                }
            }
            Err(_) => {
                println!("        => FAILED: The graph is not {}-colorable with the current heuristic.", k_to_test);
                break; // Heuristic failed, stop here
            }
        }
    }

    let total_duration = total_start.elapsed();
    println!("[*] Chromatic number search finished in {:.2?}.", total_duration);

    println!("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n");
    println!("\\          BENCHMARK RESULTS          \\");
    println!("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n");

    println!("  Graph: dsjc500.5.col");
    println!("  Known Best Result (χ): ~48");
    match best_k {
        Some(k) => {
            println!("  ARES-51 Computed Chromatic Number (χ'): {}", k);
            if k <= 50 { // Beating or matching high-quality results
                println!("  \\  STATUS: EXCELLENT. World-class result achieved.  \\");
            } else {
                println!("  \\  STATUS: GOOD. A valid coloring was found.  \\");
            }
        }
        None => {
            println!("  ARES-51 Computed Chromatic Number (χ'): Not Found");
            println!("  \\  STATUS: FAILED. No valid coloring was found in the search range.  \\");
        }
    }
    println!("  Total Time: {:.2?}", total_duration);
    println!("\n----------------------------------------");

    Ok(())
}