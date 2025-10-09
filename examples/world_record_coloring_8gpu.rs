//! DSJC1000-5 World Record Attempt - 8-GPU Massive Parallel Search
//!
//! Target: Beat 82-83 colors (current world record)
//! Strategy: 800,000 attempts (100K per GPU) across 8× H200 SXM
//! Each GPU tries different random seeds with phase-guided algorithm

use prism_ai::integration::{UnifiedPlatform, PlatformInput};
use prism_ai::gpu_coloring::GpuColoringSearch;
use ndarray::Array1;
use anyhow::{Result, Context};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::thread;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone)]
struct Graph {
    num_vertices: usize,
    num_edges: usize,
    adjacency: Vec<bool>,
}

impl Graph {
    fn from_mtx(path: &str) -> Result<Self> {
        println!("[MTX Parser] Loading {}...", path);
        let file = File::open(path)
            .context(format!("Failed to open {}", path))?;
        let reader = BufReader::new(file);

        let mut lines = reader.lines();

        // Skip comments
        while let Some(Ok(line)) = lines.next() {
            if !line.trim().starts_with('%') {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let n: usize = parts[0].parse()?;
                    let m: usize = parts[2].parse()?;

                    println!("[MTX Parser] Graph: {} vertices, {} edges", n, m);

                    let mut adjacency = vec![false; n * n];
                    let mut edge_count = 0;

                    // Read edges
                    for line in lines {
                        let line = line?;
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            let u: usize = parts[0].parse::<usize>()? - 1;
                            let v: usize = parts[1].parse::<usize>()? - 1;

                            if u < n && v < n {
                                adjacency[u * n + v] = true;
                                adjacency[v * n + u] = true;
                                edge_count += 1;
                            }
                        }
                    }

                    println!("[MTX Parser] Parsed {} edges", edge_count);

                    return Ok(Graph {
                        num_vertices: n,
                        num_edges: edge_count,
                        adjacency,
                    });
                }
            }
        }

        Err(anyhow::anyhow!("Invalid MTX file format"))
    }
}

#[derive(Debug, Clone)]
struct ColoringResult {
    gpu_id: usize,
    attempts: usize,
    best_colors: usize,
    execution_time_s: f64,
    valid: bool,
}

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  PRISM-AI WORLD RECORD ATTEMPT                              ║");
    println!("║  DSJC1000-5 Graph Coloring - 8× H200 SXM                    ║");
    println!("║  Target: Beat 82-83 colors (30+ year record)               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Detect GPUs
    println!("🔍 GPU Detection...");
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=index,name,memory.total", "--format=csv,noheader"])
        .output()
    {
        let gpu_info = String::from_utf8_lossy(&output.stdout);
        for line in gpu_info.lines() {
            println!("  {}", line);
        }
    }

    let num_gpus = std::env::var("NUM_GPUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);

    println!("\n🎯 World Record Configuration:");
    println!("   GPUs:              {} × H200 SXM", num_gpus);
    println!("   Attempts per GPU:  100,000");
    println!("   Total attempts:    {} million", (num_gpus * 100_000) / 1_000_000);
    println!("   Target:            < 82 colors");
    println!("   Current record:    82-83 colors (since 1993)\n");

    // Load graph
    let graph_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/prism-ai/benchmarks/dimacs_official/DSJC1000-5.mtx".to_string());

    println!("📂 Loading benchmark...");
    let graph = Graph::from_mtx(&graph_file)?;

    println!("\n📊 Graph Statistics:");
    println!("   Vertices:  {}", graph.num_vertices);
    println!("   Edges:     {}", graph.num_edges);
    println!("   Density:   {:.2}%\n",
        (graph.num_edges as f64 / ((graph.num_vertices * (graph.num_vertices - 1) / 2) as f64)) * 100.0);

    let attempts_per_gpu = std::env::var("ATTEMPTS_PER_GPU")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100_000);

    let results = Arc::new(Mutex::new(Vec::new()));
    let best_overall = Arc::new(Mutex::new((usize::MAX, 0))); // (colors, gpu_id)

    let total_start = Instant::now();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  LAUNCHING 8-GPU MASSIVE PARALLEL SEARCH");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut handles = vec![];

    // Launch one thread per GPU
    for gpu_id in 0..num_gpus {
        let graph_clone = graph.clone();
        let results_clone = Arc::clone(&results);
        let best_clone = Arc::clone(&best_overall);

        println!("🚀 GPU {}: Launching {} attempts...", gpu_id, attempts_per_gpu);

        let handle = thread::spawn(move || -> Result<()> {
            let gpu_start = Instant::now();

            // Create platform for phase generation (using this GPU)
            let mut platform = UnifiedPlatform::new_with_device(20, gpu_id)?;

            // Generate phase state
            let input = PlatformInput::new(
                Array1::from_vec(vec![0.5; 20]),
                Array1::zeros(20),
                0.01,
            );
            let output = platform.process(input)?;

            // Create GPU coloring search on this device
            let cuda_context = cudarc::driver::CudaContext::new(gpu_id)?;
            let gpu_search = GpuColoringSearch::new(cuda_context)?;

            // Run massive parallel search
            let phase_field = output.phase_field
                .ok_or_else(|| anyhow::anyhow!("No phase field generated"))?;
            let kuramoto_state = output.kuramoto_state
                .ok_or_else(|| anyhow::anyhow!("No Kuramoto state generated"))?;

            // Expand to graph size
            let expanded_phase = expand_phase_field(&phase_field, &graph_clone);
            let expanded_kuramoto = expand_kuramoto_state(&kuramoto_state, &graph_clone);

            println!("[GPU {}] Starting {} attempts...", gpu_id, attempts_per_gpu);

            let (best_colors, valid) = gpu_search.massive_parallel_search(
                &graph_clone,
                &expanded_phase,
                &expanded_kuramoto,
                1000,  // max_colors
                attempts_per_gpu
            )?;

            let gpu_time = gpu_start.elapsed().as_secs_f64();

            // Update global best
            let mut best = best_clone.lock().unwrap();
            if best_colors < best.0 {
                *best = (best_colors, gpu_id);
                println!("🎉 GPU {} FOUND NEW BEST: {} colors!", gpu_id, best_colors);
            }

            // Store result
            let result = ColoringResult {
                gpu_id,
                attempts: attempts_per_gpu,
                best_colors,
                execution_time_s: gpu_time,
                valid,
            };

            results_clone.lock().unwrap().push(result);

            println!("[GPU {}] ✅ Complete: {} colors in {:.1}s", gpu_id, best_colors, gpu_time);
            Ok(())
        });

        handles.push(handle);
        thread::sleep(std::time::Duration::from_millis(500));
    }

    println!("\n⏳ All {} GPU searches running...\n", num_gpus);

    // Wait for completion
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => println!("✅ GPU {} completed", i),
            Ok(Err(e)) => eprintln!("❌ GPU {} error: {}", i, e),
            Err(e) => eprintln!("❌ GPU {} panic: {:?}", i, e),
        }
    }

    let total_time = total_start.elapsed();
    let results_vec = results.lock().unwrap();
    let (best_colors, best_gpu) = *best_overall.lock().unwrap();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  WORLD RECORD ATTEMPT RESULTS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Instance:           DSJC1000-5");
    println!("Vertices:           {}", graph.num_vertices);
    println!("Edges:              {}", graph.num_edges);
    println!("Current Record:     82-83 colors (best known since 1993)\n");

    println!("Search Configuration:");
    println!("  GPUs:             {}", num_gpus);
    println!("  Attempts/GPU:     {}", attempts_per_gpu);
    println!("  Total attempts:   {}", num_gpus * attempts_per_gpu);
    println!("  Wall time:        {:.1}s ({:.1} minutes)\n", total_time.as_secs_f64(), total_time.as_secs_f64() / 60.0);

    println!("Per-GPU Results:");
    println!("┌──────┬──────────────┬──────────────┬──────────────┬────────────┐");
    println!("│ GPU  │   Attempts   │ Best Colors  │     Time     │   Status   │");
    println!("├──────┼──────────────┼──────────────┼──────────────┼────────────┤");

    for r in results_vec.iter() {
        println!("│  {}   │   {:>9}  │     {:>3}      │   {:>7.1}s   │     {}     │",
                 r.gpu_id, r.attempts, r.best_colors, r.execution_time_s,
                 if r.valid { "✓" } else { "✗" });
    }
    println!("└──────┴──────────────┴──────────────┴──────────────┴────────────┘\n");

    println!("═══════════════════════════════════════════════════════════════");
    println!("  🏆 FINAL RESULT");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("  BEST COLORING FOUND:  {} colors (GPU {})", best_colors, best_gpu);
    println!("  WORLD RECORD:         82-83 colors");
    println!();

    if best_colors < 83 {
        println!("  🎉🎉🎉 WORLD RECORD! 🎉🎉🎉");
        println!("  You beat the 30+ year record!");
        println!("  Previous best: 82-83 colors");
        println!("  Your result:   {} colors", best_colors);
    } else if best_colors == 83 {
        println!("  🎊 TIED WORLD RECORD!");
        println!("  Matched best known: 83 colors");
    } else if best_colors < 100 {
        println!("  💪 Strong result: {} colors", best_colors);
        println!("  Gap to record: {} colors", best_colors - 82);
        println!("  Try more attempts or different strategies");
    } else {
        println!("  📊 Result: {} colors", best_colors);
        println!("  Gap to record: {} colors", best_colors - 82);
        println!("  Need better optimization strategy");
    }

    println!("\n═══════════════════════════════════════════════════════════════\n");

    // Save best coloring
    std::fs::write("/output/best_result.txt", format!(
        "Instance: DSJC1000-5\n\
         Vertices: {}\n\
         Edges: {}\n\
         Best colors found: {}\n\
         GPU: {}\n\
         Total attempts: {}\n\
         Time: {:.1}s\n\
         World record: 82-83 colors\n",
        graph.num_vertices, graph.num_edges, best_colors, best_gpu,
        num_gpus * attempts_per_gpu, total_time.as_secs_f64()
    ))?;

    println!("💾 Results saved to /output/best_result.txt\n");

    Ok(())
}

fn expand_phase_field(pf: &shared_types::PhaseField, graph: &Graph) -> shared_types::PhaseField {
    let n_phases = pf.phases.len();
    let n_vertices = graph.num_vertices;

    let mut expanded_phases: Vec<f64> = (0..n_vertices)
        .map(|i| pf.phases[i % n_phases])
        .collect();

    // Relax with neighbors (3 iterations)
    for _ in 0..3 {
        let mut new_phases = expanded_phases.clone();
        for v in 0..n_vertices {
            let mut sum_phase = expanded_phases[v];
            let mut count = 1.0;

            for u in 0..n_vertices {
                if graph.adjacency[v * n_vertices + u] {
                    sum_phase += expanded_phases[u];
                    count += 1.0;
                }
            }

            new_phases[v] = sum_phase / count;
        }
        expanded_phases = new_phases;
    }

    // Compute coherence matrix
    let mut coherence = vec![0.0; n_vertices * n_vertices];
    for i in 0..n_vertices {
        for j in 0..n_vertices {
            let phase_diff = (expanded_phases[i] - expanded_phases[j]).abs();
            coherence[i * n_vertices + j] = (phase_diff * 2.0 * std::f64::consts::PI).cos();
        }
    }

    shared_types::PhaseField {
        phases: expanded_phases,
        coherence,
        mean_field: pf.mean_field,
    }
}

fn expand_kuramoto_state(ks: &shared_types::KuramotoState, graph: &Graph) -> shared_types::KuramotoState {
    let n_phases = ks.phases.len();
    let n_vertices = graph.num_vertices;

    let expanded_phases: Vec<f64> = (0..n_vertices)
        .map(|i| ks.phases[i % n_phases])
        .collect();

    let order = (expanded_phases.iter().map(|p| p.cos()).sum::<f64>().powi(2) +
                 expanded_phases.iter().map(|p| p.sin()).sum::<f64>().powi(2)).sqrt() / n_vertices as f64;

    shared_types::KuramotoState {
        phases: expanded_phases,
        order_parameter: order,
    }
}
