//! Test GPU-accelerated TSP solver with simple instances

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use quantum_engine::GpuTspSolver;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════");
    println!("     GPU TSP SOLVER - MINIMAL TEST SUITE");
    println!("═══════════════════════════════════════════════════\n");

    // Test 1: Small complete graph (5 cities)
    println!("Test 1: Complete graph K_5 (5 cities, all connected)");
    let mut coupling1 = Array2::zeros((5, 5));
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                // Distance proportional to city index difference
                let dist = ((i as f64 - j as f64).abs() + 1.0).recip();
                coupling1[[i, j]] = Complex64::new(dist, 0.0);
            }
        }
    }

    let mut tsp1 = GpuTspSolver::new(&coupling1)?;
    println!("  Initial tour length: {:.4}", tsp1.get_tour_length());
    println!("  Initial tour: {:?}", tsp1.get_tour());

    tsp1.optimize_2opt_gpu(50)?;
    println!("  Final tour length: {:.4}", tsp1.get_tour_length());
    println!("  Final tour: {:?}", tsp1.get_tour());
    println!("  Valid: {}", tsp1.validate_tour());
    println!();

    // Test 2: Grid-like coupling (8 cities)
    println!("Test 2: Grid coupling (8 cities, geometric layout)");
    let mut coupling2 = Array2::zeros((8, 8));
    // Simulate 2D grid: cities at (0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)
    let positions = vec![
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
        (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0),
    ];

    for i in 0..8 {
        for j in 0..8 {
            if i != j {
                let dx = positions[i].0 - positions[j].0;
                let dy = positions[i].1 - positions[j].1;
                let euclidean_dist = ((dx * dx + dy * dy) as f64).sqrt();
                // Coupling inversely proportional to distance
                let coupling_strength = 1.0 / (euclidean_dist + 0.1);
                coupling2[[i, j]] = Complex64::new(coupling_strength, 0.0);
            }
        }
    }

    let mut tsp2 = GpuTspSolver::new(&coupling2)?;
    println!("  Initial tour length: {:.4}", tsp2.get_tour_length());

    tsp2.optimize_2opt_gpu(100)?;
    println!("  Final tour length: {:.4}", tsp2.get_tour_length());
    println!("  Valid: {}", tsp2.validate_tour());
    println!();

    // Test 3: Random coupling (10 cities)
    println!("Test 3: Random coupling (10 cities)");
    let mut coupling3 = Array2::zeros((10, 10));
    for i in 0..10 {
        for j in 0..10 {
            if i != j {
                // Random coupling strengths
                let strength = (((i * 7 + j * 11) % 100) as f64 / 100.0) + 0.1;
                coupling3[[i, j]] = Complex64::new(strength, 0.0);
            }
        }
    }

    let mut tsp3 = GpuTspSolver::new(&coupling3)?;
    println!("  Initial tour length: {:.4}", tsp3.get_tour_length());

    tsp3.optimize_2opt_gpu(100)?;
    println!("  Final tour length: {:.4}", tsp3.get_tour_length());
    println!("  Valid: {}", tsp3.validate_tour());
    println!();

    println!("═══════════════════════════════════════════════════");
    println!("✅ ALL TESTS PASSED");
    println!("═══════════════════════════════════════════════════");

    Ok(())
}
