//! 8-GPU Parallel TSP Benchmark
//!
//! Uses all 8 GPUs simultaneously in a single process
//! Designed for RunPod 8× H200 SXM instance

use prism_ai::integration::{UnifiedPlatform, PlatformInput};
use ndarray::Array1;
use anyhow::{Result, Context};
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone)]
struct TspInstance {
    name: String,
    dimension: usize,
    coords: Vec<(f64, f64)>,
}

impl TspInstance {
    fn parse_tsplib(path: &str) -> Result<Self> {
        let file = File::open(path)?;
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

        Ok(TspInstance { name, dimension, coords })
    }

    fn to_platform_input(&self, start: usize, count: usize) -> Array1<f64> {
        let end = (start + count).min(self.coords.len());
        let n = end - start;

        let x_min = self.coords.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max = self.coords.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.coords.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = self.coords.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

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

#[derive(Debug, Clone)]
struct GpuResult {
    gpu_id: usize,
    cities_processed: usize,
    execution_time_ms: f64,
    latency_ms: f64,
    free_energy: f64,
    entropy_production: f64,
}

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  PRISM-AI 8-GPU PARALLEL TSP BENCHMARK                      ║");
    println!("║  Single Process - All GPUs Simultaneously                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Detect GPUs
    println!("🔍 Detecting GPUs...");
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=index,name,memory.total", "--format=csv,noheader"])
        .output()
    {
        let gpu_info = String::from_utf8_lossy(&output.stdout);
        let gpu_count = gpu_info.lines().filter(|l| !l.trim().is_empty()).count();
        println!("  ✅ Found {} GPU(s):", gpu_count);
        for line in gpu_info.lines() {
            println!("     {}", line);
        }
    }
    println!();

    // Get number of GPUs to use
    let num_gpus = std::env::var("NUM_GPUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);

    println!("🎯 Configuration:");
    println!("   GPUs to use: {}", num_gpus);

    // Load TSP
    let tsp_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/prism-ai/benchmarks/tsp/pla85900.tsp".to_string());

    println!("   Loading: {}\n", tsp_file);

    let tsp = TspInstance::parse_tsplib(&tsp_file)
        .context("Failed to parse TSP file")?;

    // Get total cities to process
    let total_cities = std::env::var("NUM_CITIES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(tsp.dimension)
        .min(tsp.dimension);

    println!("📊 Benchmark Configuration:");
    println!("   Instance:     {}", tsp.name);
    println!("   Total cities: {}", tsp.dimension);
    println!("   Processing:   {} cities", total_cities);
    println!("   GPUs:         {}", num_gpus);
    println!("   Per GPU:      ~{} cities\n", total_cities / num_gpus);

    // Divide work
    let cities_per_gpu = (total_cities + num_gpus - 1) / num_gpus;

    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    let total_start = Instant::now();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  LAUNCHING {} GPU THREADS", num_gpus);
    println!("═══════════════════════════════════════════════════════════════\n");

    // Launch one thread per GPU
    for gpu_id in 0..num_gpus {
        let start_city = gpu_id * cities_per_gpu;
        let count = cities_per_gpu.min(total_cities - start_city);

        if count == 0 {
            break;
        }

        println!("🚀 GPU {}: Cities {}-{} ({} cities)",
                 gpu_id, start_city, start_city + count - 1, count);

        let tsp_clone = tsp.clone();
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || -> Result<()> {
            let gpu_start = Instant::now();

            // Create platform on this specific GPU
            let mut platform = UnifiedPlatform::new_with_device(count, gpu_id)?;

            // Get input data
            let input_data = tsp_clone.to_platform_input(start_city, count);
            let input = PlatformInput::new(
                input_data,
                Array1::zeros(count),
                0.01,
            );

            // Process
            let output = platform.process(input)?;
            let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0;

            // Store results
            let result = GpuResult {
                gpu_id,
                cities_processed: count,
                execution_time_ms: gpu_time,
                latency_ms: output.metrics.total_latency_ms,
                free_energy: output.metrics.free_energy,
                entropy_production: output.metrics.entropy_production,
            };

            results_clone.lock().unwrap().push(result);
            Ok(())
        });

        handles.push(handle);

        // Small delay to stagger GPU initialization
        thread::sleep(std::time::Duration::from_millis(100));
    }

    println!("\n⏳ All GPU threads launched, waiting for completion...\n");

    // Wait for all threads
    let mut errors = Vec::new();
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => println!("✅ GPU {} completed", i),
            Ok(Err(e)) => {
                eprintln!("❌ GPU {} error: {}", i, e);
                errors.push(format!("GPU {}: {}", i, e));
            }
            Err(e) => {
                eprintln!("❌ GPU {} panic: {:?}", i, e);
                errors.push(format!("GPU {} panicked", i));
            }
        }
    }

    let total_time = total_start.elapsed();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  MULTI-GPU RESULTS");
    println!("═══════════════════════════════════════════════════════════════\n");

    let results_vec = results.lock().unwrap();

    if results_vec.is_empty() {
        eprintln!("❌ No results collected!");
        for error in &errors {
            eprintln!("   {}", error);
        }
        std::process::exit(1);
    }

    println!("Instance:          {}", tsp.name);
    println!("Total Cities:      {} / {}", total_cities, tsp.dimension);
    println!("GPUs Used:         {}\n", num_gpus);

    println!("Per-GPU Performance:");
    println!("┌──────┬────────────┬────────────┬───────────┬──────────────┬─────────────┐");
    println!("│ GPU  │   Cities   │    Time    │  Latency  │ Free Energy  │   Entropy   │");
    println!("├──────┼────────────┼────────────┼───────────┼──────────────┼─────────────┤");

    let mut total_cities_processed = 0;
    for r in results_vec.iter() {
        println!("│  {}   │  {:>8}  │ {:>7.2}s │ {:>7.2}ms │ {:>11.2} │ {:>10.6} │",
                 r.gpu_id, r.cities_processed, r.execution_time_ms / 1000.0,
                 r.latency_ms, r.free_energy, r.entropy_production);
        total_cities_processed += r.cities_processed;
    }
    println!("└──────┴────────────┴────────────┴───────────┴──────────────┴─────────────┘");

    let avg_time = results_vec.iter().map(|r| r.execution_time_ms).sum::<f64>() / results_vec.len() as f64 / 1000.0;
    let max_time = results_vec.iter().map(|r| r.execution_time_ms).fold(0.0f64, |a, b| a.max(b)) / 1000.0;
    let min_time = results_vec.iter().map(|r| r.execution_time_ms).fold(f64::INFINITY, |a, b| a.min(b)) / 1000.0;

    println!("\nAggregate Statistics:");
    println!("  Total cities processed: {}", total_cities_processed);
    println!("  Total wall time:        {:.2}s", total_time.as_secs_f64());
    println!("  Average GPU time:       {:.2}s", avg_time);
    println!("  Fastest GPU:            {:.2}s", min_time);
    println!("  Slowest GPU:            {:.2}s", max_time);
    println!("  Parallel efficiency:    {:.1}%", (avg_time / max_time) * 100.0);

    if num_gpus > 1 {
        let speedup = (avg_time * num_gpus as f64) / max_time;
        println!("  Speedup vs 1 GPU:       {:.2}x", speedup);
        println!("  Scaling efficiency:     {:.1}%", (speedup / num_gpus as f64) * 100.0);
    }

    println!("\n🎉 All {} GPUs completed successfully!", results_vec.len());

    if !errors.is_empty() {
        println!("\n⚠️  Errors encountered:");
        for error in &errors {
            println!("   - {}", error);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════\n");

    Ok(())
}
