//! TSP Benchmark Runner for pla85900 (85,900 cities)
//!
//! GPU-accelerated solving with H100 optimization
//! Real TSPLIB benchmark data - no simulation

use quantum::gpu_tsp::GpuTspSolver;
use ndarray::Array2;
use num_complex::Complex64;
use anyhow::{Result, Context};
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug)]
struct TspInstance {
    name: String,
    dimension: usize,
    coords: Vec<(f64, f64)>,
}

impl TspInstance {
    fn parse_tsplib(path: &str) -> Result<Self> {
        println!("📂 Parsing TSPLIB file: {}", path);
        let file = File::open(path)
            .context(format!("Failed to open {}", path))?;
        let reader = BufReader::new(file);

        let mut name = String::new();
        let mut dimension = 0;
        let mut coords = Vec::new();
        let mut in_coords = false;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.starts_with("NAME") {
                name = line.split(':').nth(1).unwrap_or("").trim().to_string();
            } else if line.starts_with("DIMENSION") {
                dimension = line.split(':').nth(1).unwrap_or("0").trim().parse()?;
            } else if line == "NODE_COORD_SECTION" {
                in_coords = true;
                continue;
            } else if line == "EOF" {
                break;
            } else if in_coords && !line.is_empty() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let x: f64 = parts[1].parse()?;
                    let y: f64 = parts[2].parse()?;
                    coords.push((x, y));
                }
            }
        }

        println!("  ✓ Parsed {} cities", coords.len());
        Ok(TspInstance { name, dimension, coords })
    }

    /// Compute coupling matrix from coordinates using distance-based coupling
    fn to_coupling_matrix(&self, subset_size: Option<usize>) -> Result<Array2<Complex64>> {
        let n = subset_size.unwrap_or(self.coords.len()).min(self.coords.len());

        println!("📐 Computing coupling matrix for {} cities...", n);

        // Normalize coordinates
        let x_min = self.coords.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max = self.coords.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.coords.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = self.coords.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        let mut coupling = Array2::zeros((n, n));

        // Compute distance-based coupling
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    coupling[[i, j]] = Complex64::new(0.0, 0.0);
                } else {
                    let (x1, y1) = self.coords[i];
                    let (x2, y2) = self.coords[j];

                    // Normalize
                    let nx1 = (x1 - x_min) / x_range;
                    let ny1 = (y1 - y_min) / y_range;
                    let nx2 = (x2 - x_min) / x_range;
                    let ny2 = (y2 - y_min) / y_range;

                    // Euclidean distance
                    let dist = ((nx1 - nx2).powi(2) + (ny1 - ny2).powi(2)).sqrt();

                    // Coupling strength inversely proportional to distance
                    // This creates a phase field that guides TSP optimization
                    let strength = 1.0 / (1.0 + dist);
                    let phase = dist * 2.0 * std::f64::consts::PI;

                    coupling[[i, j]] = Complex64::new(
                        strength * phase.cos(),
                        strength * phase.sin()
                    );
                }
            }
        }

        println!("  ✓ Coupling matrix computed");
        Ok(coupling)
    }
}

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  PRISM-AI TSP BENCHMARK: pla85900                           ║");
    println!("║  85,900 Cities - TSPLIB Official Benchmark                  ║");
    println!("║  GPU-Accelerated with H100 Optimization                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Check for GPU
    println!("🔍 Checking GPU availability...");
    match std::process::Command::new("nvidia-smi").output() {
        Ok(output) => {
            let gpu_info = String::from_utf8_lossy(&output.stdout);
            if gpu_info.contains("H100") {
                println!("  ✅ NVIDIA H100 detected!");
            } else if gpu_info.contains("GPU") {
                println!("  ⚠️  GPU detected (not H100)");
            } else {
                println!("  ❌ No GPU detected");
            }
        }
        Err(_) => println!("  ⚠️  nvidia-smi not available"),
    }
    println!();

    // Load TSP instance
    let tsp_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmarks/tsp/pla85900.tsp".to_string());

    let tsp = TspInstance::parse_tsplib(&tsp_file)
        .context("Failed to parse TSP file")?;

    println!("\n📊 Instance Information:");
    println!("  Name:        {}", tsp.name);
    println!("  Dimension:   {} cities", tsp.dimension);
    println!("  Type:        Real TSPLIB benchmark");
    println!("  Known best:  142,382,641 (as of 2024)\n");

    // Determine problem size
    let problem_sizes = [
        (100, "tiny test"),
        (500, "small"),
        (1000, "medium"),
        (2000, "large"),
        (5000, "very large"),
        (10000, "huge"),
        (20000, "massive"),
        (50000, "extreme"),
        (85900, "FULL INSTANCE")
    ];

    let problem_size = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);  // Default to 1000 cities for testing

    let size_label = problem_sizes.iter()
        .find(|(size, _)| *size == problem_size)
        .map(|(_, label)| *label)
        .unwrap_or("custom");

    println!("🎯 Running on {} cities ({})\n", problem_size, size_label);

    // Generate coupling matrix
    let coupling_start = Instant::now();
    let coupling_matrix = tsp.to_coupling_matrix(Some(problem_size))?;
    let coupling_time = coupling_start.elapsed();

    println!("  ⏱️  Coupling computation: {:.2}s\n", coupling_time.as_secs_f64());

    // Initialize GPU TSP solver
    println!("🚀 Initializing GPU TSP Solver...");
    let init_start = Instant::now();
    let mut solver = GpuTspSolver::new(&coupling_matrix)
        .context("Failed to initialize GPU TSP solver")?;
    let init_time = init_start.elapsed();

    println!("  ✓ Solver initialized in {:.2}s", init_time.as_secs_f64());
    println!("  ✓ Initial tour (nearest-neighbor): {:.2}", solver.get_tour_length());
    println!();

    // Run GPU 2-opt optimization
    let max_iterations = std::env::args()
        .nth(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);  // Default to 1000 iterations

    println!("═══════════════════════════════════════════════════════════════");
    println!("  GPU 2-OPT OPTIMIZATION");
    println!("═══════════════════════════════════════════════════════════════\n");

    let opt_start = Instant::now();
    solver.optimize_2opt_gpu(max_iterations)?;
    let opt_time = opt_start.elapsed();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Instance:            {}", tsp.name);
    println!("Cities processed:    {} / {}", problem_size, tsp.dimension);
    println!("Known best (full):   142,382,641\n");

    println!("PERFORMANCE:");
    println!("  Coupling setup:    {:.2}s", coupling_time.as_secs_f64());
    println!("  Initialization:    {:.2}s", init_time.as_secs_f64());
    println!("  Optimization:      {:.2}s", opt_time.as_secs_f64());
    println!("  Total time:        {:.2}s\n",
        (coupling_time + init_time + opt_time).as_secs_f64());

    println!("SOLUTION:");
    println!("  Final tour length: {:.2}", solver.get_tour_length());
    println!("  Tour valid:        {}",
        if solver.validate_tour() { "✓ YES" } else { "✗ NO" });
    println!();

    // Save tour if requested
    if let Some(output_file) = std::env::args().nth(4) {
        let tour = solver.get_tour();
        let tour_str: Vec<String> = tour.iter().map(|&v| v.to_string()).collect();
        std::fs::write(&output_file, tour_str.join("\n"))?;
        println!("💾 Tour saved to: {}\n", output_file);
    }

    println!("═══════════════════════════════════════════════════════════════\n");

    // GPU utilization summary
    println!("🎮 GPU UTILIZATION:");
    println!("  • Distance matrix computation: GPU");
    println!("  • 2-opt swap evaluation: GPU (parallel)");
    println!("  • Tour construction: CPU");
    println!("  • Swap application: CPU\n");

    println!("📌 USAGE:");
    println!("  ./run_tsp_pla85900 [tsp_file] [num_cities] [max_iter] [output_file]");
    println!("  Example: ./run_tsp_pla85900 benchmarks/tsp/pla85900.tsp 10000 5000 tour.txt\n");

    Ok(())
}
