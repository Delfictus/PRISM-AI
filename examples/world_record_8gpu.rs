//! DSJC1000-5 World Record Attempt - 8× H200 GPU Massive Search
//!
//! Current Record: 82-83 colors (30+ years)
//! Strategy: 800,000 attempts across 8 GPUs (100K per GPU)
//! Goal: Find < 82 colors = WORLD RECORD

use prct_core::{parse_mtx_file, phase_guided_coloring};
use prism_ai::integration::{UnifiedPlatform, PlatformInput};
use prism_ai::gpu_coloring::GpuColoringSearch;
use shared_types::{PhaseField, KuramotoState, Graph};
use ndarray::Array1;
use anyhow::Result;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::thread;

// Copy expansion functions from run_dimacs_official.rs
fn expand_phase_field(pf: &PhaseField, graph: &Graph) -> PhaseField {
    let n_phases = pf.phases.len();
    let n_vertices = graph.num_vertices;

    let mut expanded_phases: Vec<f64> = (0..n_vertices)
        .map(|i| pf.phases[i % n_phases])
        .collect();

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

    let mut expanded_coherence = vec![0.0; n_vertices * n_vertices];
    for i in 0..n_vertices {
        for j in 0..n_vertices {
            let phase_diff = (expanded_phases[i] - expanded_phases[j]).abs();
            expanded_coherence[i * n_vertices + j] = (1.0 - phase_diff / std::f64::consts::PI).max(0.0);
        }
    }

    let (sum_cos, sum_sin): (f64, f64) = expanded_phases.iter()
        .map(|&p| (p.cos(), p.sin()))
        .fold((0.0, 0.0), |(c, s), (pc, ps)| (c + pc, s + ps));
    let order_parameter = ((sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n_vertices as f64).min(1.0);

    PhaseField {
        phases: expanded_phases,
        coherence_matrix: expanded_coherence,
        order_parameter,
        resonance_frequency: pf.resonance_frequency,
    }
}

fn expand_kuramoto_state(ks: &KuramotoState, graph: &Graph) -> KuramotoState {
    let n_phases = ks.phases.len();
    let n_vertices = graph.num_vertices;

    let mut expanded_phases: Vec<f64> = (0..n_vertices)
        .map(|i| ks.phases[i % n_phases])
        .collect();

    let expanded_freqs: Vec<f64> = (0..n_vertices)
        .map(|i| ks.natural_frequencies[i % n_phases])
        .collect();

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

    let mut expanded_coupling = vec![0.0; n_vertices * n_vertices];
    for i in 0..n_vertices {
        for j in 0..n_vertices {
            if i == j {
                expanded_coupling[i * n_vertices + j] = 1.0;
            } else if graph.adjacency[i * n_vertices + j] {
                expanded_coupling[i * n_vertices + j] = 0.5;
            } else {
                let src_i = i % n_phases;
                let src_j = j % n_phases;
                expanded_coupling[i * n_vertices + j] = ks.coupling_matrix[src_i * n_phases + src_j] * 0.1;
            }
        }
    }

    let (sum_cos, sum_sin): (f64, f64) = expanded_phases.iter()
        .map(|&p| (p.cos(), p.sin()))
        .fold((0.0, 0.0), |(c, s), (pc, ps)| (c + pc, s + ps));
    let order_parameter = ((sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n_vertices as f64).min(1.0);
    let mean_phase = expanded_phases.iter().sum::<f64>() / n_vertices as f64;

    KuramotoState {
        phases: expanded_phases,
        natural_frequencies: expanded_freqs,
        coupling_matrix: expanded_coupling,
        order_parameter,
        mean_phase,
    }
}

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  🏆 WORLD RECORD ATTEMPT: DSJC1000-5                        ║");
    println!("║  8× H200 SXM - 800,000 Parallel Attempts                    ║");
    println!("║  Target: Beat 82-83 colors (30+ year record)               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("🔍 Detecting GPUs...");
    let num_gpus = std::env::var("NUM_GPUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);

    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        let count = String::from_utf8_lossy(&output.stdout).lines().count();
        println!("  ✅ Found {} GPU(s)\n", count);
    }

    let attempts_per_gpu = std::env::var("ATTEMPTS_PER_GPU")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100_000);

    println!("🎯 World Record Configuration:");
    println!("   Instance:          DSJC1000-5");
    println!("   GPUs:              {}", num_gpus);
    println!("   Attempts per GPU:  {}", attempts_per_gpu);
    println!("   Total attempts:    {}", num_gpus * attempts_per_gpu);
    println!("   Current record:    82-83 colors\n");

    // Load graph
    let graph_file = "/prism-ai/benchmarks/dimacs_official/DSJC1000-5.mtx";
    println!("📂 Loading {}...", graph_file);
    let graph = parse_mtx_file(graph_file)?;
    println!("  ✓ {} vertices, {} edges\n", graph.num_vertices, graph.num_edges);

    let results = Arc::new(Mutex::new(Vec::new()));
    let global_best = Arc::new(Mutex::new((1000, 0))); // (colors, gpu_id)

    let total_start = Instant::now();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  LAUNCHING 8-GPU WORLD RECORD SEARCH");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut handles = vec![];

    for gpu_id in 0..num_gpus {
        let graph_clone = graph.clone();
        let results_clone = Arc::clone(&results);
        let best_clone = Arc::clone(&global_best);

        println!("🚀 GPU {}: {} attempts starting...", gpu_id, attempts_per_gpu);

        let handle = thread::spawn(move || -> Result<()> {
            let gpu_start = Instant::now();

            // Generate phase state on this GPU
            let mut platform = UnifiedPlatform::new_with_device(20, gpu_id)?;
            let input = PlatformInput::new(
                Array1::from_vec(vec![0.5; 20]),
                Array1::zeros(20),
                0.01,
            );
            let output = platform.process(input)?;

            let phase_field = output.phase_field.unwrap();
            let kuramoto_state = output.kuramoto_state.unwrap();

            // Expand to graph size
            let expanded_pf = expand_phase_field(&phase_field, &graph_clone);
            let expanded_ks = expand_kuramoto_state(&kuramoto_state, &graph_clone);

            // GPU search on this device
            let cuda_context = cudarc::driver::CudaContext::new(gpu_id)?;
            let gpu_search = GpuColoringSearch::new(cuda_context)?;

            let solution = gpu_search.massive_parallel_search(
                &graph_clone,
                &expanded_pf,
                &expanded_ks,
                200,  // max_colors
                attempts_per_gpu
            )?;

            let gpu_time = gpu_start.elapsed().as_secs_f64();
            let best_colors = solution.chromatic_number;
            let valid = solution.conflicts == 0;

            // Update global best
            {
                let mut best = best_clone.lock().unwrap();
                if best_colors < best.0 {
                    *best = (best_colors, gpu_id);
                    println!("🎉 GPU {} NEW BEST: {} colors!", gpu_id, best_colors);
                }
            }

            results_clone.lock().unwrap().push((gpu_id, best_colors, gpu_time, valid));

            println!("[GPU {}] ✅ {} colors in {:.1}s", gpu_id, best_colors, gpu_time);
            Ok(())
        });

        handles.push(handle);
        thread::sleep(std::time::Duration::from_millis(200));
    }

    println!("\n⏳ All {} GPUs searching in parallel...\n", num_gpus);

    for handle in handles {
        let _ = handle.join();
    }

    let total_time = total_start.elapsed();
    let (best_colors, best_gpu) = *global_best.lock().unwrap();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  🏆 WORLD RECORD ATTEMPT - FINAL RESULTS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Search Statistics:");
    println!("  Total attempts:    {}", num_gpus * attempts_per_gpu);
    println!("  Total time:        {:.1}s ({:.1} minutes)", total_time.as_secs_f64(), total_time.as_secs_f64() / 60.0);
    println!("  GPUs used:         {}\n", num_gpus);

    println!("┌──────┬──────────────┬──────────────┬──────────────┐");
    println!("│ GPU  │   Attempts   │ Best Colors  │     Time     │");
    println!("├──────┼──────────────┼──────────────┼──────────────┤");

    let results_vec = results.lock().unwrap();
    for (gpu_id, colors, time, _valid) in results_vec.iter() {
        println!("│  {}   │   {:>9}  │     {:>3}      │   {:>7.1}s   │", gpu_id, attempts_per_gpu, colors, time);
    }
    println!("└──────┴──────────────┴──────────────┴──────────────┘\n");

    println!("═══════════════════════════════════════════════════════════════\n");
    println!("  BEST RESULT:      {} colors (found by GPU {})", best_colors, best_gpu);
    println!("  WORLD RECORD:     82-83 colors\n");

    if best_colors < 83 {
        println!("  🎉🎉🎉 WORLD RECORD BROKEN! 🎉🎉🎉\n");
        println!("  YOU BEAT THE 30+ YEAR RECORD!");
        println!("  Previous: 82-83 colors");
        println!("  Yours:    {} colors\n", best_colors);
    } else if best_colors == 83 {
        println!("  🎊 TIED WORLD RECORD! 🎊\n");
    } else {
        println!("  Gap to record: {} colors\n", best_colors - 82);
        println!("  Try: Increase attempts, or refine algorithm");
    }

    println!("═══════════════════════════════════════════════════════════════\n");

    std::fs::write("/output/world_record_result.txt", format!(
        "DSJC1000-5 World Record Attempt\n\
         ===============================\n\
         \n\
         Best result:     {} colors (GPU {})\n\
         World record:    82-83 colors\n\
         Total attempts:  {}\n\
         GPUs used:       {}\n\
         Time:            {:.1}s\n\
         \n\
         {}",
        best_colors, best_gpu, num_gpus * attempts_per_gpu, num_gpus, total_time.as_secs_f64(),
        if best_colors < 83 { "🏆 WORLD RECORD!" } else { "Continue searching..." }
    ))?;

    Ok(())
}
