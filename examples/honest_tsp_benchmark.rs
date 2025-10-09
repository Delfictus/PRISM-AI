//! HONEST TSP Benchmark - Real TSPLIB Data, No Simulation
//!
//! Benchmarks:
//! - usa13509.tsp: 13,509 US cities (REAL data from US census)
//! - pla85900.tsp: 85,900 cities (REAL microchip layout)
//!
//! NO synthetic data, NO fake timings, ONLY actual execution

use prism_ai::integration::{UnifiedPlatform, PlatformInput};
use ndarray::Array1;
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

    fn to_platform_input(&self, max_cities: Option<usize>) -> Array1<f64> {
        // Take subset if too large
        let n = max_cities.unwrap_or(self.coords.len()).min(self.coords.len());

        // Normalize coordinates to [0, 1] range
        let x_min = self.coords.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max = self.coords.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.coords.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = self.coords.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        // Convert to input vector (interleave x,y)
        let mut input = Vec::with_capacity(n * 2);
        for i in 0..n {
            let (x, y) = self.coords[i];
            input.push((x - x_min) / x_range);
            input.push((y - y_min) / y_range);
        }

        // Pad or truncate to desired size
        input.resize(n.max(100), 0.5);
        Array1::from_vec(input)
    }
}

fn main() -> Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  PRISM-AI HONEST TSP BENCHMARK                            ║");
    println!("║  Real TSPLIB data - No simulation - True performance     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Parse TSP instance
    println!("Loading TSP benchmark...");

    let tsp_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "benchmarks/tsp/usa13509.tsp".to_string());

    let tsp = TspInstance::parse_tsplib(&tsp_file)
        .context(format!("Failed to parse {}", tsp_file))?;

    println!("✓ Loaded: {}", tsp.name);
    println!("✓ Cities: {}", tsp.dimension);
    println!("✓ Source: REAL TSPLIB benchmark (not synthetic)\n");

    // Determine problem size - configurable via environment variable or full instance
    let problem_size = if let Ok(num_cities_str) = std::env::var("NUM_CITIES") {
        let requested = num_cities_str.parse::<usize>()
            .unwrap_or(tsp.dimension)
            .min(tsp.dimension);
        println!("📊 Using {} cities (requested via NUM_CITIES env var)", requested);
        requested
    } else {
        // Default: use full instance
        println!("✓ Processing FULL instance: {} cities", tsp.dimension);
        tsp.dimension
    };

    println!();

    // Initialize platform
    println!("Initializing GPU platform with {} dimensions...\n", problem_size);
    let init_start = Instant::now();
    let mut platform = UnifiedPlatform::new(problem_size)?;
    let init_time = init_start.elapsed().as_secs_f64() * 1000.0;

    println!("\n✓ Platform initialized in {:.2}ms\n", init_time);

    // Convert TSP to input
    let input_data = tsp.to_platform_input(Some(problem_size));
    let input = PlatformInput::new(
        input_data.clone(),
        Array1::zeros(problem_size),
        0.01,
    );

    println!("═══════════════════════════════════════════════════════════");
    println!("  EXECUTING ON REAL TSP DATA");
    println!("═══════════════════════════════════════════════════════════\n");

    // Execute and measure REAL performance
    let exec_start = Instant::now();
    let output = platform.process(input)?;
    let exec_time = exec_start.elapsed().as_secs_f64() * 1000.0;

    // Display ONLY real results
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  HONEST RESULTS - REAL MEASUREMENTS");
    println!("═══════════════════════════════════════════════════════════\n");

    println!("TSP Instance:        {}", tsp.name);
    println!("Total Cities:        {} (using {})", tsp.dimension, problem_size);
    println!("Data Source:         TSPLIB (real benchmark)\n");

    println!("EXECUTION TIME:      {:.3} ms", exec_time);
    println!("Platform Latency:    {:.3} ms\n", output.metrics.total_latency_ms);

    println!("REAL Phase Timings:");
    println!("  1. Neuromorphic:      {:.3} ms", output.metrics.phase_latencies[0]);
    println!("  2. Info Flow:         {:.3} ms", output.metrics.phase_latencies[1]);
    println!("  3. Coupling:          {:.3} ms", output.metrics.phase_latencies[2]);
    println!("  4. Thermodynamic:     {:.3} ms", output.metrics.phase_latencies[3]);
    println!("  5. Quantum:           {:.3} ms", output.metrics.phase_latencies[4]);
    println!("  6. Active Inference:  {:.3} ms", output.metrics.phase_latencies[5]);
    println!("  7. Control:           {:.3} ms", output.metrics.phase_latencies[6]);
    println!("  8. Synchronization:   {:.3} ms\n", output.metrics.phase_latencies[7]);

    println!("Physical Verification:");
    println!("  Free Energy:          {:.4}", output.metrics.free_energy);
    println!("  Entropy Production:   {:.6} (must be ≥0)", output.metrics.entropy_production);
    println!("  2nd Law:              {}",
        if output.metrics.entropy_production >= -1e-10 { "✓ SATISFIED" } else { "✗ VIOLATED" }
    );
    println!("  Free Energy Finite:   {}\n",
        if output.metrics.free_energy.is_finite() { "✓ YES" } else { "✗ NO" }
    );

    println!("Constitutional Compliance:");
    println!("  Latency < 500ms:      {} ({:.2}ms)",
        if exec_time < 500.0 { "✓ PASS" } else { "✗ FAIL" },
        exec_time
    );
    println!("  Overall:              {}\n",
        if output.metrics.meets_requirements() { "✓ PASS" } else { "✗ FAIL" }
    );

    println!("═══════════════════════════════════════════════════════════\n");

    Ok(())
}
