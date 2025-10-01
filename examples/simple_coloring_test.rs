//! Simple Graph Coloring Test - ONE benchmark, clear output

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use quantum_engine::GpuChromaticColoring;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SIMPLE GRAPH COLORING TEST - dsjc125.1");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load the benchmark
    let file = File::open("benchmarks/dsjc125.1.col")?;
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
                    edges.push((u - 1, v - 1)); // DIMACS is 1-indexed
                }
            }
            _ => {}
        }
    }

    println!("📊 Loaded: {} vertices, {} edges", vertices, edges.len());
    println!("📊 Known best: χ = 5 (from DIMACS website)");
    println!();

    // Build coupling matrix
    let mut coupling = Array2::zeros((vertices, vertices));
    for &(u, v) in &edges {
        if u < vertices && v < vertices {
            coupling[[u, v]] = Complex64::new(1.0, 0.0);
            coupling[[v, u]] = Complex64::new(1.0, 0.0);
        }
    }

    println!("🔬 Testing GPU Chromatic Coloring...");
    println!();

    // Try to find a valid coloring
    let start = Instant::now();
    let mut colors_found = None;

    for k in 2..=20 {
        print!("   Trying χ = {}... ", k);
        match GpuChromaticColoring::new_adaptive(&coupling, k) {
            Ok(coloring) => {
                if coloring.verify_coloring() {
                    println!("✅ VALID");
                    colors_found = Some(k);
                    break;
                } else {
                    println!("❌ INVALID (conflicts found)");
                }
            }
            Err(e) => {
                println!("❌ ERROR: {}", e);
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");

    if let Some(colors) = colors_found {
        println!("✅ Found valid coloring with χ = {} colors", colors);
        println!("⏱️  Time: {:.2} seconds", elapsed);
        println!("📊 Gap from optimal: +{} colors (optimal is χ=5)", colors - 5);

        if colors <= 10 {
            println!("\n✅ REASONABLE RESULT (within 2× of optimal)");
        } else {
            println!("\n⚠️  HIGH COLOR COUNT (more than 2× optimal)");
        }
    } else {
        println!("❌ Could not find valid coloring with ≤20 colors");
        println!("⏱️  Time: {:.2} seconds", elapsed);
    }

    Ok(())
}
