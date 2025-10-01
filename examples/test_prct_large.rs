//! Large-Scale PRCT Chromatic Coloring Test
//!
//! Tests YOUR Phase Resonance Chromatic-TSP algorithm on large graphs

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use quantum_engine::ChromaticColoring;
use std::time::Instant;
use rand::Rng;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  LARGE-SCALE PRCT CHROMATIC COLORING TEST");
    println!("  Phase Resonance Chromatic-TSP Algorithm");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Test progressively larger graphs
    let test_sizes = vec![
        (1_000, 0.05, "1K vertices, 5% density"),
        (2_000, 0.03, "2K vertices, 3% density"),
        (5_000, 0.01, "5K vertices, 1% density"),
    ];

    for (vertices, density, description) in test_sizes {
        println!("───────────────────────────────────────────────────────────────────");
        println!("📊 Test: {}", description);
        println!("───────────────────────────────────────────────────────────────────");

        // Generate random graph
        println!("   Generating graph...");
        let gen_start = Instant::now();

        let mut rng = rand::thread_rng();
        let mut edges = Vec::new();

        for i in 0..vertices {
            for j in (i+1)..vertices {
                if rng.gen::<f64>() < density {
                    edges.push((i, j));
                }
            }
        }

        println!("   Generated {} edges in {:.2}s", edges.len(), gen_start.elapsed().as_secs_f64());

        // Build coupling matrix
        println!("   Building coupling matrix...");
        let mut coupling = Array2::zeros((vertices, vertices));
        for &(u, v) in &edges {
            coupling[[u, v]] = Complex64::new(1.0, 0.0);
            coupling[[v, u]] = Complex64::new(1.0, 0.0);
        }
        println!("   ✅ Done\n");

        // Run PRCT algorithm
        println!("   Running PRCT Chromatic Coloring...");
        let start = Instant::now();
        let mut colors_found = None;

        // Try reasonable color counts for sparse graphs
        let max_colors = (vertices as f64 * density * 2.0).ceil() as usize + 20;
        let try_colors = vec![5, 10, 15, 20, 25, 30, 40, 50, max_colors];

        for k in try_colors {
            if k > max_colors { break; }

            print!("      χ = {}... ", k);
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
        if let Some(colors) = colors_found {
            println!("   ✅ SUCCESS");
            println!("      Colors found: χ = {}", colors);
            println!("      Time:         {:.2} seconds", elapsed);
            println!("      Vertices:     {}", vertices);
            println!("      Edges:        {}", edges.len());
        } else {
            println!("   ❌ FAILED - Could not find valid coloring");
            println!("      Time:         {:.2} seconds", elapsed);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PRCT SCALABILITY TEST COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}
