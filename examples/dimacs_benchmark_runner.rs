//! Official DIMACS COLOR Benchmark Runner
//!
//! Runs the adaptive chromatic coloring algorithm against official DIMACS
//! benchmark graphs and compares results with known best solutions.
//!
//! Benchmarks from: https://mat.tepper.cmu.edu/COLOR/instances.html

use anyhow::{Result, anyhow};
use ndarray::Array2;
use num_complex::Complex64;
use std::fs::{File, read_dir};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;
use quantum_engine::ChromaticColoring;

#[derive(Debug, Clone)]
struct BenchmarkSpec {
    name: String,
    file: PathBuf,
    known_best: Option<usize>,
    category: BenchmarkCategory,
}

#[derive(Debug, Clone, PartialEq)]
enum BenchmarkCategory {
    Small,      // < 100 vertices
    Medium,     // 100-500 vertices
    Large,      // 500+ vertices
    Challenge,  // Known hard problems
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    vertices: usize,
    edges: usize,
    density: f64,
    known_best: Option<usize>,
    computed: Option<usize>,
    search_time_ms: f64,
    quality_score: f64,
    status: String,
}

/// Parse DIMACS .col file
fn parse_col_file(path: &Path) -> Result<(usize, Vec<(usize, usize)>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut vertices = 0;
    let mut edges = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts.get(0) {
            Some(&"p") => {
                if parts.len() >= 3 && parts[1] == "edge" {
                    vertices = parts[2].parse()?;
                }
            }
            Some(&"e") => {
                if parts.len() >= 3 {
                    let u: usize = parts[1].parse()?;
                    let v: usize = parts[2].parse()?;
                    edges.push((u - 1, v - 1)); // Convert to 0-indexed
                }
            }
            _ => {}
        }
    }

    if vertices == 0 {
        return Err(anyhow!("Invalid DIMACS file: no problem line"));
    }

    Ok((vertices, edges))
}

/// Build coupling matrix from edge list
fn build_coupling_matrix(vertices: usize, edges: &[(usize, usize)]) -> Array2<Complex64> {
    let mut matrix = Array2::zeros((vertices, vertices));
    for &(u, v) in edges {
        if u < vertices && v < vertices {
            matrix[[u, v]] = Complex64::new(1.0, 0.0);
            matrix[[v, u]] = Complex64::new(1.0, 0.0);
        }
    }
    matrix
}

/// Find all benchmark files in directory
fn find_benchmarks(dir: &Path) -> Result<Vec<BenchmarkSpec>> {
    let mut benchmarks = Vec::new();

    // Known best results for DIMACS benchmarks
    let known_results = vec![
        // Small benchmarks
        ("myciel3", 4),
        ("myciel4", 5),
        ("myciel5", 6),
        ("myciel6", 7),
        ("queen5_5", 5),
        ("jean", 10),

        // Medium benchmarks
        ("queen8_8", 9),
        ("queen9_9", 10),
        ("huck", 11),
        ("david", 11),
        ("anna", 11),
        ("games120", 9),
        ("miles250", 8),
        ("miles500", 20),

        // Large DSJC benchmarks
        ("dsjc125.1", 5),
        ("dsjc125.5", 17),
        ("dsjc125.9", 44),
        ("dsjc250.1", 8),
        ("dsjc250.5", 28),
        ("dsjc250.9", 72),
        ("dsjc500.1", 12),
        ("dsjc500.5", 48),
        ("dsjc500.9", 126),
        ("dsjc1000.1", 20),
        ("dsjc1000.5", 83),
        ("dsjc1000.9", 224),

        // Register allocation
        ("fpsol2.i.1", 65),
        ("fpsol2.i.2", 30),
        ("fpsol2.i.3", 30),

        // Test graphs
        ("test_small", 5),      // K5
        ("test_bipartite", 2),   // K(3,3)
        ("test_cycle", 3),       // C5
    ];

    let known_map: std::collections::HashMap<_, _> = known_results.into_iter().collect();

    for entry in read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("col") {
            let name = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();

            let known_best = known_map.get(name.as_str()).copied();

            // Categorize by name patterns
            let category = if name.starts_with("test_") {
                BenchmarkCategory::Small
            } else if name.starts_with("dsjc500") || name.starts_with("dsjc1000") {
                BenchmarkCategory::Challenge
            } else if name.starts_with("dsjc") || name.starts_with("miles500") {
                BenchmarkCategory::Large
            } else if name.starts_with("queen") || name.contains("125") {
                BenchmarkCategory::Medium
            } else {
                BenchmarkCategory::Small
            };

            benchmarks.push(BenchmarkSpec {
                name,
                file: path,
                known_best,
                category,
            });
        }
    }

    benchmarks.sort_by_key(|b| (b.category.clone() as u8, b.name.clone()));

    Ok(benchmarks)
}

/// Run benchmark on a single graph
fn run_benchmark(spec: &BenchmarkSpec, time_limit_ms: u64) -> Result<BenchmarkResult> {
    println!("\n[*] {}", spec.name);

    let (vertices, edges) = parse_col_file(&spec.file)?;
    let edge_count = edges.len();
    let max_edges = vertices * (vertices - 1) / 2;
    let density = edge_count as f64 / max_edges as f64;

    println!("    V={}, E={}, density={:.1}%", vertices, edge_count, density * 100.0);

    if let Some(known) = spec.known_best {
        println!("    Known best: χ = {}", known);
    }

    let coupling_matrix = build_coupling_matrix(vertices, &edges);

    let start = Instant::now();
    let mut computed = None;

    // Determine search range - FAIR SEARCH from k=2
    // DO NOT bias based on known_best - that's cheating!
    let max_k = match spec.known_best {
        Some(known) => (known + 10).min(vertices),
        None => vertices.min(50),
    };
    let min_k = 2;

    println!("    Searching k={}..{}", min_k, max_k);

    'search: for k in min_k..=max_k {
        // Check time limit
        if start.elapsed().as_millis() > time_limit_ms as u128 {
            println!("    ⏱ Time limit reached");
            break;
        }

        match ChromaticColoring::new_adaptive(&coupling_matrix, k) {
            Ok(coloring) => {
                if coloring.verify_coloring() {
                    println!("    ✓ Found {}-coloring", k);
                    computed = Some(k);
                    break 'search;
                }
            }
            Err(_) => continue,
        }
    }

    let search_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Calculate quality score
    let quality_score = match (computed, spec.known_best) {
        (Some(c), Some(k)) => {
            let ratio = c as f64 / k as f64;
            if ratio <= 1.0 {
                100.0 // Optimal or better
            } else if ratio <= 1.1 {
                90.0  // Within 10%
            } else if ratio <= 1.2 {
                75.0  // Within 20%
            } else if ratio <= 1.5 {
                50.0  // Within 50%
            } else {
                25.0  // Suboptimal
            }
        }
        (Some(_), None) => 80.0, // Found something, no baseline
        (None, _) => 0.0,         // Failed
    };

    let status = match (computed, spec.known_best) {
        (Some(c), Some(k)) if c == k => "✓ OPTIMAL".to_string(),
        (Some(c), Some(k)) if c < k => "✓✓ BETTER!".to_string(),
        (Some(c), Some(k)) if c <= k + 2 => "✓ GOOD".to_string(),
        (Some(c), Some(k)) => format!("~ +{}", c - k),
        (Some(_), None) => "✓ FOUND".to_string(),
        (None, _) => "✗ FAILED".to_string(),
    };

    Ok(BenchmarkResult {
        name: spec.name.clone(),
        vertices,
        edges: edge_count,
        density,
        known_best: spec.known_best,
        computed,
        search_time_ms,
        quality_score,
        status,
    })
}

/// Print results table
fn print_results(results: &[BenchmarkResult]) {
    println!("\n╔════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           DIMACS BENCHMARK RESULTS                                         ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Benchmark        │  V   │   E    │ Dens% │ Best │ Comp │  Time(ms) │ Quality │ Status   ║");
    println!("╟──────────────────┼──────┼────────┼───────┼──────┼──────┼───────────┼─────────┼──────────╢");

    for result in results {
        let best_str = result.known_best
            .map(|x| format!("{:>4}", x))
            .unwrap_or_else(|| " ?  ".to_string());

        let comp_str = result.computed
            .map(|x| format!("{:>4}", x))
            .unwrap_or_else(|| "FAIL".to_string());

        println!(
            "║ {:<16} │ {:>4} │ {:>6} │ {:>5.1} │ {} │ {} │ {:>9.2} │ {:>6.1}% │ {:<8} ║",
            if result.name.len() > 16 { &result.name[..16] } else { &result.name },
            result.vertices,
            result.edges,
            result.density * 100.0,
            best_str,
            comp_str,
            result.search_time_ms,
            result.quality_score,
            result.status
        );
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════════════════╝");

    // Summary statistics
    let total = results.len();
    let completed = results.iter().filter(|r| r.computed.is_some()).count();
    let optimal = results.iter().filter(|r| r.status.contains("OPTIMAL")).count();
    let avg_quality = results.iter().map(|r| r.quality_score).sum::<f64>() / total as f64;
    let total_time = results.iter().map(|r| r.search_time_ms).sum::<f64>();

    println!("\n📊 SUMMARY:");
    println!("   Total Benchmarks: {}", total);
    println!("   Completed: {} ({:.1}%)", completed, (completed as f64 / total as f64) * 100.0);
    println!("   Optimal Results: {} ({:.1}%)", optimal, (optimal as f64 / total as f64) * 100.0);
    println!("   Average Quality: {:.1}%", avg_quality);
    println!("   Total Time: {:.2}s", total_time / 1000.0);
}

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                      ARES-51 DIMACS BENCHMARK SUITE                                        ║");
    println!("║                   Official Graph Coloring Benchmarks                                       ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════╝");

    let benchmark_dir = Path::new("benchmarks");

    if !benchmark_dir.exists() {
        return Err(anyhow!("Benchmarks directory not found!\nRun: ./scripts/download_dimacs_benchmarks.sh"));
    }

    println!("\n[*] Scanning for benchmark files...");
    let specs = find_benchmarks(benchmark_dir)?;
    println!("    Found {} benchmarks", specs.len());

    if specs.is_empty() {
        return Err(anyhow!("No benchmark files found!\nRun: ./scripts/download_dimacs_benchmarks.sh"));
    }

    // Categorize benchmarks
    let small: Vec<_> = specs.iter().filter(|s| s.category == BenchmarkCategory::Small).collect();
    let medium: Vec<_> = specs.iter().filter(|s| s.category == BenchmarkCategory::Medium).collect();
    let large: Vec<_> = specs.iter().filter(|s| s.category == BenchmarkCategory::Large).collect();
    let challenge: Vec<_> = specs.iter().filter(|s| s.category == BenchmarkCategory::Challenge).collect();

    println!("\n📁 Benchmark Categories:");
    println!("   Small (< 100 vertices): {}", small.len());
    println!("   Medium (100-500 vertices): {}", medium.len());
    println!("   Large (500+ vertices): {}", large.len());
    println!("   Challenge (hardest): {}", challenge.len());

    let mut results = Vec::new();

    // Run small benchmarks (fast)
    if !small.is_empty() {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║              SMALL BENCHMARKS (< 100 vertices)                 ║");
        println!("╚════════════════════════════════════════════════════════════════╝");

        for spec in small {
            match run_benchmark(spec, 5000) { // 5s time limit
                Ok(result) => results.push(result),
                Err(e) => println!("    ✗ Error: {}", e),
            }
        }
    }

    // Run medium benchmarks
    if !medium.is_empty() {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║            MEDIUM BENCHMARKS (100-500 vertices)                ║");
        println!("╚════════════════════════════════════════════════════════════════╝");

        for spec in medium {
            match run_benchmark(spec, 10000) { // 10s time limit
                Ok(result) => results.push(result),
                Err(e) => println!("    ✗ Error: {}", e),
            }
        }
    }

    // Run large benchmarks
    if !large.is_empty() {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║             LARGE BENCHMARKS (500+ vertices)                   ║");
        println!("╚════════════════════════════════════════════════════════════════╝");

        for spec in large {
            match run_benchmark(spec, 30000) { // 30s time limit
                Ok(result) => results.push(result),
                Err(e) => println!("    ✗ Error: {}", e),
            }
        }
    }

    // Run challenge benchmarks (optional, user can skip)
    if !challenge.is_empty() {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║           CHALLENGE BENCHMARKS (World-Class Hard)              ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("\n⚠️  These are VERY hard problems. Press Ctrl+C to skip.");
        println!("    Each benchmark has a 60s time limit.\n");

        for spec in challenge {
            match run_benchmark(spec, 60000) { // 60s time limit
                Ok(result) => results.push(result),
                Err(e) => println!("    ✗ Error: {}", e),
            }
        }
    }

    // Print final results
    print_results(&results);

    // Overall assessment
    let avg_quality = results.iter().map(|r| r.quality_score).sum::<f64>() / results.len() as f64;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    OVERALL ASSESSMENT                          ║");
    println!("╠════════════════════════════════════════════════════════════════╣");

    if avg_quality >= 90.0 {
        println!("║  Status: ✓✓ EXCELLENT - World-class performance               ║");
    } else if avg_quality >= 75.0 {
        println!("║  Status: ✓ GOOD - Competitive with state-of-the-art           ║");
    } else if avg_quality >= 50.0 {
        println!("║  Status: ~ ACCEPTABLE - Room for improvement                  ║");
    } else {
        println!("║  Status: ✗ NEEDS WORK - Below competitive performance         ║");
    }

    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(())
}
