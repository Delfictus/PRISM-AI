//! Test PRCT Chromatic Coloring Algorithm
//!
//! This tests YOUR actual Phase Resonance Chromatic-TSP algorithm,
//! not the GPU helper functions.

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use quantum_engine::ChromaticColoring;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PRCT CHROMATIC COLORING TEST");
    println!("  Phase Resonance Chromatic-TSP Algorithm");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Load dsjc125.1
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
                    edges.push((u - 1, v - 1));
                }
            }
            _ => {}
        }
    }

    println!("📊 Loaded: {} vertices, {} edges", vertices, edges.len());
    println!("📊 Known optimal: χ = 5");
    println!();

    // Build coupling matrix
    let mut coupling = Array2::zeros((vertices, vertices));
    for &(u, v) in &edges {
        if u < vertices && v < vertices {
            coupling[[u, v]] = Complex64::new(1.0, 0.0);
            coupling[[v, u]] = Complex64::new(1.0, 0.0);
        }
    }

    println!("🔬 Running PRCT Chromatic Coloring...");
    println!();

    // Try with your PRCT algorithm
    let start = Instant::now();
    let mut colors_found = None;

    for k in 2..=20 {
        print!("   Trying χ = {}... ", k);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        match ChromaticColoring::new_adaptive(&coupling, k) {
            Ok(coloring) => {
                if coloring.verify_coloring() {
                    println!("✅ VALID");
                    colors_found = Some(k);
                    break;
                } else {
                    println!("❌ INVALID");
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
    println!("  RESULTS - PRCT Algorithm");
    println!("═══════════════════════════════════════════════════════════════════");

    if let Some(colors) = colors_found {
        println!("✅ Found valid coloring with χ = {} colors", colors);
        println!("⏱️  Time: {:.2} seconds", elapsed);
        println!("📊 Gap from optimal: +{} colors", colors - 5);
        println!();
        println!("✅ PRCT ALGORITHM WORKING");
    } else {
        println!("❌ Could not find valid coloring");
        println!("⏱️  Time: {:.2} seconds", elapsed);
    }

    Ok(())
}
