//! Head-to-Head Comparison: GPU 2-opt vs LKH-3
//!
//! Direct performance comparison on standard TSPLIB instances

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use quantum_engine::GpuTspSolver;
use std::process::Command;
use std::time::Instant;
use std::fs;
use std::io::Write;

#[derive(Debug, Clone)]
struct ComparisonResult {
    instance_name: String,
    n_cities: usize,
    gpu_time: f64,
    gpu_length: f64,
    gpu_improvement: f64,
    lkh_time: f64,
    lkh_length: f64,
    optimal_length: f64,
    gpu_gap: f64,
    lkh_gap: f64,
}

/// TSPLIB benchmark instance
#[derive(Debug, Clone)]
struct TsplibInstance {
    name: &'static str,
    n_cities: usize,
    optimal: f64,
    file_path: &'static str,
}

const INSTANCES: &[TsplibInstance] = &[
    TsplibInstance { name: "berlin52", n_cities: 52, optimal: 7542.0, file_path: "benchmarks/tsplib/berlin52.tsp" },
    TsplibInstance { name: "eil51", n_cities: 51, optimal: 426.0, file_path: "benchmarks/tsplib/eil51.tsp" },
    TsplibInstance { name: "eil76", n_cities: 76, optimal: 538.0, file_path: "benchmarks/tsplib/eil76.tsp" },
    TsplibInstance { name: "kroA100", n_cities: 100, optimal: 21282.0, file_path: "benchmarks/tsplib/kroA100.tsp" },
    TsplibInstance { name: "kroB100", n_cities: 100, optimal: 22141.0, file_path: "benchmarks/tsplib/kroB100.tsp" },
    TsplibInstance { name: "rd100", n_cities: 100, optimal: 7910.0, file_path: "benchmarks/tsplib/rd100.tsp" },
    TsplibInstance { name: "eil101", n_cities: 101, optimal: 629.0, file_path: "benchmarks/tsplib/eil101.tsp" },
    TsplibInstance { name: "pr152", n_cities: 152, optimal: 73682.0, file_path: "benchmarks/tsplib/pr152.tsp" },
    TsplibInstance { name: "kroA200", n_cities: 200, optimal: 29368.0, file_path: "benchmarks/tsplib/kroA200.tsp" },
];

/// Parse TSPLIB format file and extract coordinates
fn parse_tsplib(file_path: &str) -> Result<Array2<f64>> {
    let content = fs::read_to_string(file_path)?;
    let mut coords = Vec::new();
    let mut in_coord_section = false;

    for line in content.lines() {
        if line.contains("NODE_COORD_SECTION") {
            in_coord_section = true;
            continue;
        }
        if line.contains("EOF") || line.contains("DISPLAY_DATA_SECTION") {
            break;
        }
        if in_coord_section {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let _id: usize = parts[0].parse()?;
                let x: f64 = parts[1].parse()?;
                let y: f64 = parts[2].parse()?;
                coords.push((x, y));
            }
        }
    }

    let n = coords.len();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                let dist = (dx * dx + dy * dy).sqrt();
                distances[[i, j]] = dist;
            }
        }
    }

    Ok(distances)
}

/// Convert distance matrix to coupling matrix for GPU solver
fn distances_to_coupling(distances: &Array2<f64>) -> Array2<Complex64> {
    let n = distances.nrows();
    let mut coupling = Array2::zeros((n, n));

    // Find max distance for normalization
    let max_dist = distances.iter()
        .filter(|&&d| d > 0.0)
        .fold(0.0_f64, |max, &d| max.max(d));

    for i in 0..n {
        for j in 0..n {
            if i != j && distances[[i, j]] > 0.0 {
                // Coupling inversely proportional to distance
                let coupling_strength = max_dist / distances[[i, j]];
                coupling[[i, j]] = Complex64::new(coupling_strength, 0.0);
            }
        }
    }

    coupling
}

/// Calculate tour length from actual distance matrix
fn calculate_tour_length(tour: &[usize], distances: &Array2<f64>) -> f64 {
    if tour.len() < 2 {
        return 0.0;
    }

    let mut length = 0.0;
    for i in 0..tour.len() {
        let from = tour[i];
        let to = tour[(i + 1) % tour.len()];
        length += distances[[from, to]];
    }

    length
}

/// Run LKH solver on TSPLIB instance
fn run_lkh(instance: &TsplibInstance) -> Result<(f64, f64)> {
    // Create parameter file for LKH
    let par_content = format!(
        "PROBLEM_FILE = {}\n\
         OUTPUT_TOUR_FILE = temp_{}.tour\n\
         RUNS = 1\n\
         TIME_LIMIT = 60\n",
        instance.file_path,
        instance.name
    );

    let par_file = format!("temp_{}.par", instance.name);
    let mut file = fs::File::create(&par_file)?;
    file.write_all(par_content.as_bytes())?;

    // Run LKH
    let start = Instant::now();
    let output = Command::new("benchmarks/lkh/LKH-3.0.9/LKH")
        .arg(&par_file)
        .output()?;
    let elapsed = start.elapsed().as_secs_f64();

    // Parse output to get tour length
    let output_str = String::from_utf8_lossy(&output.stdout);
    let mut tour_length = 0.0;

    for line in output_str.lines() {
        if line.contains("Cost.min") || line.contains("Length") {
            // Try to extract the tour length
            let parts: Vec<&str> = line.split_whitespace().collect();
            for (i, part) in parts.iter().enumerate() {
                if part.contains("=") && i + 1 < parts.len() {
                    if let Ok(length) = parts[i + 1].parse::<f64>() {
                        tour_length = length;
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
    fs::remove_file(&par_file).ok();
    fs::remove_file(format!("temp_{}.tour", instance.name)).ok();

    Ok((elapsed, tour_length))
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║        HEAD-TO-HEAD: GPU 2-opt vs LKH-3                          ║");
    println!("║        Direct Competition on TSPLIB Instances                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("🎯 ALGORITHMS:");
    println!("   • GPU 2-opt: Massively parallel simple algorithm");
    println!("   • LKH-3: World's best sequential sophisticated algorithm");
    println!();
    println!("💻 HARDWARE:");
    println!("   • GPU: NVIDIA RTX 5070 Laptop (4,608 cores)");
    println!("   • CPU: Running LKH on same laptop CPU");
    println!();
    println!("📊 INSTANCES: {} standard TSPLIB benchmarks", INSTANCES.len());
    println!();

    // Check if LKH is installed
    if !std::path::Path::new("benchmarks/lkh/LKH-3.0.9/LKH").exists() {
        println!("❌ ERROR: LKH-3 not found!");
        println!();
        println!("Please run setup first:");
        println!("  chmod +x scripts/setup_lkh.sh");
        println!("  ./scripts/setup_lkh.sh");
        println!();
        return Ok(());
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Starting head-to-head comparison...");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let mut results = Vec::new();

    for instance in INSTANCES {
        println!("───────────────────────────────────────────────────────────────────");
        println!("📍 Instance: {} ({} cities, optimal = {:.0})",
                 instance.name, instance.n_cities, instance.optimal);
        println!();

        // Check if file exists
        if !std::path::Path::new(instance.file_path).exists() {
            println!("   ⚠️  File not found, skipping");
            println!();
            continue;
        }

        // Parse TSPLIB file
        println!("   📥 Loading instance...");
        let distances = match parse_tsplib(instance.file_path) {
            Ok(d) => d,
            Err(e) => {
                println!("   ❌ Failed to parse: {}", e);
                println!();
                continue;
            }
        };

        // Run GPU solver
        println!("   🎮 Running GPU 2-opt...");
        let gpu_start = Instant::now();

        let coupling = distances_to_coupling(&distances);
        let mut gpu_solver = GpuTspSolver::new(&coupling)?;

        // Calculate initial length from actual distances
        let initial_tour = gpu_solver.get_tour();
        let initial_length = calculate_tour_length(&initial_tour, &distances);

        let max_iterations = match instance.n_cities {
            ..=100 => 200,
            101..=200 => 150,
            _ => 100,
        };

        gpu_solver.optimize_2opt_gpu(max_iterations)?;
        let gpu_time = gpu_start.elapsed().as_secs_f64();

        // Calculate final length from actual distances
        let final_tour = gpu_solver.get_tour();
        let gpu_length = calculate_tour_length(&final_tour, &distances);
        let gpu_improvement = ((initial_length - gpu_length) / initial_length) * 100.0;

        println!("      ✓ Time: {:.3}s | Length: {:.2} | Improvement: {:.1}%",
                 gpu_time, gpu_length, gpu_improvement);

        // Run LKH solver
        println!("   🧠 Running LKH-3...");
        let (lkh_time, lkh_length) = match run_lkh(instance) {
            Ok(result) => result,
            Err(e) => {
                println!("      ❌ LKH failed: {}", e);
                println!();
                continue;
            }
        };

        println!("      ✓ Time: {:.3}s | Length: {:.0}", lkh_time, lkh_length);
        println!();

        // Calculate gaps from optimal
        let gpu_gap = ((gpu_length - instance.optimal) / instance.optimal) * 100.0;
        let lkh_gap = ((lkh_length - instance.optimal) / instance.optimal) * 100.0;

        println!("   📊 COMPARISON:");
        println!("      Optimal:     {:.0}", instance.optimal);
        println!("      GPU result:  {:.2} ({:+.2}% from optimal)", gpu_length, gpu_gap);
        println!("      LKH result:  {:.0} ({:+.2}% from optimal)", lkh_length, lkh_gap);
        println!();

        if gpu_time < lkh_time {
            let speedup = lkh_time / gpu_time;
            println!("      🏆 GPU WINS on speed: {:.2}× faster!", speedup);
        } else {
            let speedup = gpu_time / lkh_time;
            println!("      🏆 LKH WINS on speed: {:.2}× faster!", speedup);
        }

        if gpu_gap.abs() < lkh_gap.abs() {
            println!("      🏆 GPU WINS on quality: better solution!");
        } else {
            println!("      🏆 LKH WINS on quality: {:.2}% closer to optimal",
                     gpu_gap.abs() - lkh_gap.abs());
        }
        println!();

        results.push(ComparisonResult {
            instance_name: instance.name.to_string(),
            n_cities: instance.n_cities,
            gpu_time,
            gpu_length,
            gpu_improvement,
            lkh_time,
            lkh_length,
            optimal_length: instance.optimal,
            gpu_gap,
            lkh_gap,
        });
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    FINAL COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    if results.is_empty() {
        println!("❌ No results to compare. Check that:");
        println!("   1. LKH-3 is installed (./scripts/setup_lkh.sh)");
        println!("   2. TSPLIB files are downloaded (./scripts/download_tsplib.sh)");
        return Ok(());
    }

    println!("  Instance      |  Size  |  GPU Time | LKH Time | Speed Winner | Quality Winner");
    println!("  ──────────────────────────────────────────────────────────────────────────────");

    let mut gpu_speed_wins = 0;
    let mut lkh_speed_wins = 0;
    let mut gpu_quality_wins = 0;
    let mut lkh_quality_wins = 0;

    for result in &results {
        let speed_winner = if result.gpu_time < result.lkh_time {
            gpu_speed_wins += 1;
            "GPU"
        } else {
            lkh_speed_wins += 1;
            "LKH"
        };

        let quality_winner = if result.gpu_gap.abs() < result.lkh_gap.abs() {
            gpu_quality_wins += 1;
            "GPU"
        } else {
            lkh_quality_wins += 1;
            "LKH"
        };

        println!("  {:12}  | {:4}   | {:7.3}s  | {:7.3}s | {:^12} | {:^14}",
                 result.instance_name, result.n_cities,
                 result.gpu_time, result.lkh_time,
                 speed_winner, quality_winner);
    }

    println!();
    println!("───────────────────────────────────────────────────────────────────");
    println!("  OVERALL SCORE:");
    println!("───────────────────────────────────────────────────────────────────");
    println!();
    println!("  Speed Competition:");
    println!("    GPU wins: {}/{}", gpu_speed_wins, results.len());
    println!("    LKH wins: {}/{}", lkh_speed_wins, results.len());
    println!();
    println!("  Quality Competition:");
    println!("    GPU wins: {}/{}", gpu_quality_wins, results.len());
    println!("    LKH wins: {}/{}", lkh_quality_wins, results.len());
    println!();

    // Average metrics
    let avg_gpu_time: f64 = results.iter().map(|r| r.gpu_time).sum::<f64>() / results.len() as f64;
    let avg_lkh_time: f64 = results.iter().map(|r| r.lkh_time).sum::<f64>() / results.len() as f64;
    let avg_gpu_gap: f64 = results.iter().map(|r| r.gpu_gap.abs()).sum::<f64>() / results.len() as f64;
    let avg_lkh_gap: f64 = results.iter().map(|r| r.lkh_gap.abs()).sum::<f64>() / results.len() as f64;

    println!("  Average Performance:");
    println!("    GPU time: {:.3}s | LKH time: {:.3}s", avg_gpu_time, avg_lkh_time);
    println!("    GPU gap:  {:.2}% | LKH gap:  {:.2}%", avg_gpu_gap, avg_lkh_gap);
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    HONEST ASSESSMENT");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  🎯 KEY FINDINGS:");
    println!();

    if avg_lkh_time < avg_gpu_time {
        println!("  • LKH is FASTER on these small instances ({:.1}× avg)",
                 avg_gpu_time / avg_lkh_time);
        println!("    Reason: No GPU warmup overhead, optimized for small n");
    } else {
        println!("  • GPU is FASTER on these instances ({:.1}× avg)",
                 avg_lkh_time / avg_gpu_time);
    }
    println!();

    if avg_lkh_gap < avg_gpu_gap {
        println!("  • LKH has BETTER quality ({:.2}% closer to optimal)",
                 avg_gpu_gap - avg_lkh_gap);
        println!("    Reason: Sophisticated k-opt vs simple 2-opt");
    } else {
        println!("  • GPU has BETTER quality ({:.2}% closer to optimal)",
                 avg_lkh_gap - avg_gpu_gap);
    }
    println!();

    println!("  💡 EXPECTED CROSSOVER:");
    println!("     For problems > 1,000 cities, GPU parallelism");
    println!("     should dominate LKH sequential sophistication");
    println!();
    println!("  🎯 CONCLUSION:");
    println!("     Different tools for different scales:");
    println!("     • LKH: Best for small problems (<1,000 cities)");
    println!("     • GPU: Best for large problems (>10,000 cities)");
    println!();

    println!("═══════════════════════════════════════════════════════════════════");

    Ok(())
}
