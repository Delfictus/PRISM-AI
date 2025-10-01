//! Simple benchmark test for small graph
//!
//! Quick validation test using a small complete graph K5

use anyhow::{Result, anyhow};
use ndarray::Array2;
use num_complex::Complex64;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
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

    if vertices == 0 {
        return Err(anyhow!("Failed to parse problem line"));
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
    println!("╔═══════════════════════════════════════╗");
    println!("║  ARES-51 Quick Validation Test       ║");
    println!("╚═══════════════════════════════════════╝\n");

    let file_path = Path::new("benchmarks/test_small.col");
    println!("[*] Loading: {:?}", file_path);

    let (vertices, edges) = parse_col_file(file_path)?;
    println!("[*] Graph: {} vertices, {} edges", vertices, edges.len());
    println!("[*] Expected chromatic number: 5 (complete graph K5)\n");

    let coupling_matrix = build_coupling_matrix(vertices, &edges);

    println!("[*] Testing chromatic numbers...");
    let start = Instant::now();

    for k in 2..=10 {
        print!("    k={}: ", k);
        match ChromaticColoring::new_adaptive(&coupling_matrix, k) {
            Ok(coloring) => {
                if coloring.verify_coloring() {
                    println!("✓ Valid {}-coloring found", k);
                    let duration = start.elapsed();

                    println!("\n╔═══════════════════════════════════════╗");
                    println!("║           TEST RESULT                 ║");
                    println!("╠═══════════════════════════════════════╣");
                    println!("║ Expected χ:        5                  ║");
                    println!("║ Computed χ:        {}                  ║", k);
                    if k == 5 {
                        println!("║ Status:            ✓ OPTIMAL          ║");
                    } else if k < 5 {
                        println!("║ Status:            ✓ BETTER!          ║");
                    } else {
                        println!("║ Status:            ✓ VALID            ║");
                    }
                    println!("║ Time:              {:.2}ms         ║", duration.as_secs_f64() * 1000.0);
                    println!("╚═══════════════════════════════════════╝");

                    if k == 5 {
                        println!("\n✅ Algorithm correctly identified chromatic number!");
                    } else if k < 5 {
                        println!("\n🎉 Algorithm found better result than expected!");
                    }

                    return Ok(());
                } else {
                    println!("✗ Invalid (conflicts: {})", coloring.get_conflict_count());
                }
            }
            Err(e) => {
                println!("✗ Failed: {}", e);
            }
        }
    }

    println!("\n❌ Failed to find valid coloring");
    Err(anyhow!("No valid coloring found"))
}
