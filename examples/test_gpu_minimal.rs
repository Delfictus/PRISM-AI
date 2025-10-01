//! Minimal GPU test to debug failures

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use quantum_engine::GpuChromaticColoring;

fn main() -> Result<()> {
    println!("Testing GPU chromatic coloring on minimal graph...\n");

    // Test 1: Empty graph (no edges)
    println!("Test 1: Empty graph K_3 (3 vertices, 0 edges)");
    let empty_graph = Array2::zeros((3, 3));

    match GpuChromaticColoring::new_adaptive(&empty_graph, 2) {
        Ok(coloring) => {
            println!("  ✓ Created coloring");
            println!("  Colors: {:?}", coloring.get_coloring());
            println!("  Valid: {}", coloring.verify_coloring());
        }
        Err(e) => {
            println!("  ✗ Failed: {:?}", e);
        }
    }

    // Test 2: Complete graph K_3 (triangle)
    println!("\nTest 2: Complete graph K_3 (3 vertices, 3 edges)");
    let mut triangle = Array2::zeros((3, 3));
    triangle[[0, 1]] = Complex64::new(1.0, 0.0);
    triangle[[1, 0]] = Complex64::new(1.0, 0.0);
    triangle[[0, 2]] = Complex64::new(1.0, 0.0);
    triangle[[2, 0]] = Complex64::new(1.0, 0.0);
    triangle[[1, 2]] = Complex64::new(1.0, 0.0);
    triangle[[2, 1]] = Complex64::new(1.0, 0.0);

    match GpuChromaticColoring::new_adaptive(&triangle, 3) {
        Ok(coloring) => {
            println!("  ✓ Created coloring");
            println!("  Colors: {:?}", coloring.get_coloring());
            println!("  Valid: {}", coloring.verify_coloring());
        }
        Err(e) => {
            println!("  ✗ Failed: {:?}", e);
        }
    }

    // Test 3: Simple path P_4 (4 vertices, 3 edges in a line)
    println!("\nTest 3: Path graph P_4 (4 vertices, 3 edges)");
    let mut path = Array2::zeros((4, 4));
    path[[0, 1]] = Complex64::new(1.0, 0.0);
    path[[1, 0]] = Complex64::new(1.0, 0.0);
    path[[1, 2]] = Complex64::new(1.0, 0.0);
    path[[2, 1]] = Complex64::new(1.0, 0.0);
    path[[2, 3]] = Complex64::new(1.0, 0.0);
    path[[3, 2]] = Complex64::new(1.0, 0.0);

    match GpuChromaticColoring::new_adaptive(&path, 2) {
        Ok(coloring) => {
            println!("  ✓ Created coloring");
            println!("  Colors: {:?}", coloring.get_coloring());
            println!("  Valid: {}", coloring.verify_coloring());
        }
        Err(e) => {
            println!("  ✗ Failed: {:?}", e);
        }
    }

    Ok(())
}
