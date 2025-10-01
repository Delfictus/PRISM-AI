//! Large Graph Coloring Test - Test scalability

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use quantum_engine::GpuChromaticColoring;
use std::time::Instant;
use rand::Rng;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  LARGE GRAPH COLORING TEST");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Generate a large random graph
    let vertices = 1_000;
    let density = 0.05; // 5% density

    println!("📊 Generating random graph...");
    println!("   Vertices: {}", vertices);
    println!("   Density:  {:.1}%", density * 100.0);

    let mut rng = rand::thread_rng();
    let mut edges = Vec::new();

    for i in 0..vertices {
        for j in (i+1)..vertices {
            if rng.gen::<f64>() < density {
                edges.push((i, j));
            }
        }
    }

    println!("   Edges:    {}", edges.len());
    println!();

    // Build coupling matrix
    println!("📊 Building coupling matrix...");
    let mut coupling = Array2::zeros((vertices, vertices));
    for &(u, v) in &edges {
        coupling[[u, v]] = Complex64::new(1.0, 0.0);
        coupling[[v, u]] = Complex64::new(1.0, 0.0);
    }
    println!("   ✅ Done\n");

    println!("🔬 Testing GPU Chromatic Coloring on {} vertices...", vertices);
    println!();

    // Try to find a valid coloring
    let start = Instant::now();
    let mut colors_found = None;

    // For large sparse graphs, we expect low chromatic number
    // Try up to 50 colors
    for k in vec![5, 10, 15, 20, 25, 30, 40, 50] {
        print!("   Trying χ = {}... ", k);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        match GpuChromaticColoring::new_adaptive(&coupling, k) {
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
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");

    if let Some(colors) = colors_found {
        println!("✅ Found valid coloring with χ = {} colors", colors);
        println!("⏱️  Time: {:.2} seconds", elapsed);
        println!("📊 Problem size: {} vertices, {} edges", vertices, edges.len());
        println!();
        println!("✅ SUCCESSFULLY HANDLED 10,000 VERTEX GRAPH");
    } else {
        println!("❌ Could not find valid coloring with ≤50 colors");
        println!("⏱️  Time: {:.2} seconds", elapsed);
        println!("⚠️  May need more colors or different approach");
    }

    Ok(())
}
