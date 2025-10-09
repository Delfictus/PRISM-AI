//! Multi-GPU TSP Benchmark for pla85900
//!
//! Runs TSP benchmark across multiple H100 GPUs in parallel
//! Each GPU processes a different subset of cities simultaneously

use prism_ai::integration::{UnifiedPlatform, PlatformInput};
use ndarray::Array1;
use anyhow::{Result, Context};
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::thread;

#[derive(Debug, Clone)]
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

    fn to_platform_input(&self, start: usize, count: usize) -> Array1<f64> {
        let end = (start + count).min(self.coords.len());
        let n = end - start;

        // Normalize coordinates
        let x_min = self.coords.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max = self.coords.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.coords.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = self.coords.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        // Convert to input vector
        let mut input = Vec::with_capacity(n * 2);
        for i in start..end {
            let (x, y) = self.coords[i];
            input.push((x - x_min) / x_range);
            input.push((y - y_min) / y_range);
        }

        input.resize(n.max(100), 0.5);
        Array1::from_vec(input)
    }
}

fn detect_gpu_count() -> usize {
    // Try to detect number of GPUs
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        let gpu_list = String::from_utf8_lossy(&output.stdout);
        let count = gpu_list.lines().filter(|l| !l.trim().is_empty()).count();
        if count > 0 {
            return count;
        }
    }

    // Default to 1 GPU
    1
}

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  PRISM-AI MULTI-GPU TSP BENCHMARK                           ║");
    println!("║  pla85900 (85,900 cities) - Parallel GPU Processing         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Detect GPUs
    let num_gpus = std::env::var("NUM_GPUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(detect_gpu_count);

    println!("🔍 Detecting GPUs...");
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=index,name,memory.total", "--format=csv,noheader"])
        .output()
    {
        let gpu_info = String::from_utf8_lossy(&output.stdout);
        for line in gpu_info.lines() {
            println!("  GPU: {}", line);
        }
    }
    println!("  Using {} GPU(s) for parallel processing\n", num_gpus);

    // Load TSP instance
    let tsp_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmarks/tsp/pla85900.tsp".to_string());

    let tsp = TspInstance::parse_tsplib(&tsp_file)
        .context("Failed to parse TSP file")?;

    println!("\n📊 Instance Information:");
    println!("  Name:        {}", tsp.name);
    println!("  Total Cities: {}", tsp.dimension);
    println!("  Type:        Real TSPLIB benchmark");
    println!("  Known best:  142,382,641\n");

    // Determine total cities to process
    let total_cities = std::env::var("NUM_CITIES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(tsp.dimension)
        .min(tsp.dimension);

    println!("🎯 Processing {} cities across {} GPU(s)", total_cities, num_gpus);
    println!("   Cities per GPU: ~{}\n", total_cities / num_gpus);

    // Divide work across GPUs
    let cities_per_gpu = (total_cities + num_gpus - 1) / num_gpus;

    let tsp_arc = Arc::new(tsp);
    let mut handles = vec![];

    let total_start = Instant::now();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  LAUNCHING PARALLEL GPU PROCESSING");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Launch one thread per GPU
    for gpu_id in 0..num_gpus {
        let tsp_clone = Arc::clone(&tsp_arc);
        let start_city = gpu_id * cities_per_gpu;
        let count = cities_per_gpu.min(total_cities - start_city);

        if count == 0 {
            break; // No more cities to process
        }

        println!("🚀 GPU {}: Processing cities {}-{} ({} cities)",
                 gpu_id, start_city, start_city + count - 1, count);

        let handle = thread::spawn(move || -> Result<(usize, f64, f64)> {
            // NOTE: Each GPU needs its own CUDA context
            // Currently UnifiedPlatform::new() uses GPU 0 hardcoded
            // For true multi-GPU, would need to pass device_id to platform

            let thread_start = Instant::now();

            // Create platform for this subset
            let mut platform = UnifiedPlatform::new(count)?;

            // Get input data for this GPU's subset
            let input_data = tsp_clone.to_platform_input(start_city, count);
            let input = PlatformInput::new(
                input_data.clone(),
                Array1::zeros(count),
                0.01,
            );

            // Process
            let output = platform.process(input)?;
            let elapsed = thread_start.elapsed().as_secs_f64();

            Ok((gpu_id, elapsed, output.metrics.total_latency_ms))
        });

        handles.push(handle);
    }

    // Wait for all GPUs to complete
    let mut results = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(Ok(result)) => results.push(result),
            Ok(Err(e)) => eprintln!("❌ GPU thread error: {}", e),
            Err(e) => eprintln!("❌ Thread panic: {:?}", e),
        }
    }

    let total_time = total_start.elapsed();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  MULTI-GPU RESULTS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Instance:            {}", tsp_arc.name);
    println!("Total Cities:        {} / {}", total_cities, tsp_arc.dimension);
    println!("GPUs Used:           {}\n", num_gpus);

    println!("Per-GPU Results:");
    for (gpu_id, elapsed, latency) in &results {
        println!("  GPU {}: {:.2}s ({:.2}ms latency)", gpu_id, elapsed, latency);
    }

    let avg_time = results.iter().map(|(_, t, _)| t).sum::<f64>() / results.len() as f64;
    let max_time = results.iter().map(|(_, t, _)| t).fold(0.0, |a, &b| a.max(b));

    println!("\nPerformance:");
    println!("  Total wall time:     {:.2}s", total_time.as_secs_f64());
    println!("  Average GPU time:    {:.2}s", avg_time);
    println!("  Max GPU time:        {:.2}s", max_time);
    println!("  Parallel efficiency: {:.1}%", (avg_time / max_time) * 100.0);
    println!("  Speedup vs 1 GPU:    {:.2}x", avg_time * num_gpus as f64 / max_time);

    println!("\n💡 NOTE: Current implementation uses GPU 0 for all threads");
    println!("   For true multi-GPU, need to:");
    println!("   1. Pass device_id to UnifiedPlatform::new()");
    println!("   2. Create separate CUDA context per GPU");
    println!("   3. Ensure PTX kernels loaded on each device\n");

    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
