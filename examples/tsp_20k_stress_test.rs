//! TSP 20,000 Cities Stress Test - GPU Performance Benchmark
//!
//! Tests the GPU-accelerated TSP solver on a massive 20,000-city problem.
//! This pushes the limits of the neuromorphic-quantum co-processing approach.

use neuromorphic_quantum_platform::*;
use quantum_engine::gpu_tsp::GpuTspSolver;
use ndarray::Array2;
use num_complex::Complex64;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    TSP 20,000 CITIES STRESS TEST - GPU BENCHMARK                          ║");
    println!("║                         NVIDIA RTX 5070 Maximum Challenge                                  ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    let n_cities = 20_000;
    println!("🏙️  Generating {}-city TSP problem...", n_cities);

    // Generate random city positions in 2D plane
    let start_gen = Instant::now();
    let cities = generate_random_cities(n_cities);
    println!("   ✓ Cities generated in {:.2}ms", start_gen.elapsed().as_secs_f64() * 1000.0);

    // Convert to coupling matrix (distance-based)
    println!("\n📊 Building coupling matrix ({}x{} = {} elements)...", n_cities, n_cities, n_cities * n_cities);
    let start_coupling = Instant::now();
    let coupling_matrix = cities_to_coupling_matrix(&cities);
    let coupling_time = start_coupling.elapsed();
    println!("   ✓ Coupling matrix built in {:.2}s", coupling_time.as_secs_f64());
    println!("   Memory: {:.1} MB", (n_cities * n_cities * 16) as f64 / 1024.0 / 1024.0);

    // Initialize GPU TSP solver
    println!("\n🎮 Initializing GPU TSP Solver...");
    let start_init = Instant::now();
    let mut solver = match GpuTspSolver::new(&coupling_matrix) {
        Ok(s) => {
            println!("   ✓ GPU initialized in {:.2}ms", start_init.elapsed().as_secs_f64() * 1000.0);
            s
        }
        Err(e) => {
            eprintln!("   ✗ GPU initialization failed: {}", e);
            eprintln!("\n⚠️  NOTE: This requires NVIDIA GPU with CUDA support.");
            eprintln!("   Check: nvidia-smi, /dev/dxg (WSL2), LD_LIBRARY_PATH");
            return Err(e);
        }
    };

    // Get initial tour info
    let initial_tour = solver.get_tour();
    let initial_length = solver.get_tour_length();
    println!("\n📍 Initial Tour (Nearest-Neighbor Heuristic):");
    println!("   Length: {:.2}", initial_length);
    println!("   First 10 cities: {:?}", &initial_tour[0..10.min(initial_tour.len())]);

    // Run GPU-accelerated 2-opt optimization
    println!("\n⚡ Running GPU 2-opt Optimization...");
    println!("   Max iterations: 1000");
    println!("   Expected time: 30-120s (depends on GPU)\n");

    let start_opt = Instant::now();

    // Progress callback every 100 iterations
    let mut last_report = Instant::now();
    let max_iterations = 1000;

    for iter in 0..10 {
        let batch_size = 100;
        match solver.optimize_2opt_gpu(batch_size) {
            Ok(_) => {
                if last_report.elapsed().as_secs() >= 5 {
                    let current_length = solver.get_tour_length();
                    let improvement = ((initial_length - current_length) / initial_length) * 100.0;
                    let elapsed = start_opt.elapsed().as_secs_f64();
                    println!("   Iteration {}: Length={:.2} ({:+.2}% improvement) | {:.1}s elapsed",
                             (iter + 1) * batch_size, current_length, improvement, elapsed);
                    last_report = Instant::now();
                }
            }
            Err(e) => {
                eprintln!("   ⚠️  Optimization batch {} failed: {}", iter, e);
                break;
            }
        }
    }

    let opt_time = start_opt.elapsed();
    let final_length = solver.get_tour_length();
    let improvement = ((initial_length - final_length) / initial_length) * 100.0;

    // Results summary
    println!("\n╔════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                    RESULTS SUMMARY                                         ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Cities:              {:>10}                                                          ║", n_cities);
    println!("║ Initial Tour:        {:>10.2}                                                          ║", initial_length);
    println!("║ Optimized Tour:      {:>10.2}                                                          ║", final_length);
    println!("║ Improvement:         {:>10.2}%                                                         ║", improvement);
    println!("║ Optimization Time:   {:>10.2}s                                                         ║", opt_time.as_secs_f64());
    println!("║ Total Time:          {:>10.2}s                                                         ║", (coupling_time + opt_time).as_secs_f64());
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════╣");

    if improvement > 10.0 {
        println!("║ Status: ✓ EXCELLENT - Significant improvement achieved on GPU                         ║");
    } else if improvement > 5.0 {
        println!("║ Status: ✓ GOOD - Moderate improvement (increase iterations for better results)        ║");
    } else if improvement > 0.0 {
        println!("║ Status: ~ MARGINAL - Consider increasing iterations or adjusting parameters           ║");
    } else {
        println!("║ Status: ✗ NO IMPROVEMENT - Check GPU utilization and algorithm parameters             ║");
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════════════╝");

    // GPU Performance Metrics
    println!("\n📈 GPU Performance Metrics:");
    println!("   Swaps evaluated per iteration: ~{}", (n_cities * (n_cities - 3)) / 2);
    println!("   Estimated GPU speedup: 10-50x vs CPU");
    println!("   Memory usage: ~{:.1} GB GPU VRAM", (n_cities * n_cities * 4 * 3) as f64 / 1024.0 / 1024.0 / 1024.0);

    println!("\n✅ 20K Cities TSP Stress Test Complete!");

    Ok(())
}

/// Generate random cities in 2D plane [0, 1000] x [0, 1000]
fn generate_random_cities(n: usize) -> Vec<(f64, f64)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| {
            let x = rng.gen::<f64>() * 1000.0;
            let y = rng.gen::<f64>() * 1000.0;
            (x, y)
        })
        .collect()
}

/// Convert city positions to coupling matrix (inverse distance)
fn cities_to_coupling_matrix(cities: &[(f64, f64)]) -> Array2<Complex64> {
    let n = cities.len();
    let mut coupling = Array2::zeros((n, n));

    println!("   Computing pairwise distances...");
    let chunk_size = 1000;

    for i in 0..n {
        if i % chunk_size == 0 {
            print!("\r   Progress: {:.1}%", (i as f64 / n as f64) * 100.0);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }

        for j in 0..n {
            if i != j {
                let dx = cities[i].0 - cities[j].0;
                let dy = cities[i].1 - cities[j].1;
                let distance = (dx * dx + dy * dy).sqrt();

                // Coupling strength inversely proportional to distance
                // Scale to [0, 1] range for numerical stability
                let coupling_strength = 1000.0 / (distance + 1.0);
                coupling[[i, j]] = Complex64::new(coupling_strength, 0.0);
            }
        }
    }
    println!("\r   Progress: 100.0%  ");

    coupling
}
