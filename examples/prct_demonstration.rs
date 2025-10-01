//! PRCT (Phase Resonance Chromatic-TSP) Algorithm Demonstration
//!
//! Demonstrates the complete PRCT patent algorithm including:
//! - Chromatic graph coloring optimization
//! - TSP path optimization
//! - Phase coherence maximization
//! - Performance validation (target: >60% coherence)

use quantum_engine::{PhaseResonanceField, ChromaticColoring, TSPPathOptimizer};
use ndarray::Array2;
use num_complex::Complex64;
use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🌊 PRCT Algorithm Demonstration");
    println!("=================================\n");
    println!("Phase Resonance Chromatic-TSP Patent Algorithm");
    println!("Targets: >60% phase coherence, optimized graph coloring & TSP\n");

    // Test with various system sizes
    let test_sizes = vec![10, 20, 30, 50];

    for size in test_sizes {
        println!("\n📊 Testing PRCT with {} vertices", size);
        println!("─────────────────────────────────────");
        test_prct_system(size)?;
    }

    println!("\n✅ PRCT Algorithm Demonstration Complete!");
    println!("═══════════════════════════════════════\n");

    Ok(())
}

fn test_prct_system(n_vertices: usize) -> Result<()> {
    // Generate realistic coupling matrix (protein-like structure)
    let coupling_matrix = generate_coupling_matrix(n_vertices);

    println!("\n🔧 Building Optimized PRCT Field...");
    let start = Instant::now();

    // Build optimized PRCT field with full algorithm
    let num_colors = 4; // Protein secondary structure colors
    let tsp_iterations = 200;

    let prct_field = PhaseResonanceField::build_optimized(
        coupling_matrix.clone(),
        num_colors,
        tsp_iterations,
    )?;

    let build_time = start.elapsed();

    // Get diagnostic information
    let diagnostics = prct_field.get_prct_diagnostics();

    println!("\n📈 PRCT Performance Metrics:");
    println!("  • Build Time: {:.2}ms", build_time.as_millis());
    println!("  • Vertices: {}", diagnostics.num_vertices);
    println!("  • Colors Used: {}", diagnostics.num_colors);
    println!("  • TSP Path Length: {}", diagnostics.tsp_path_length);
    println!("  • Phase Coherence: {:.2}% (target: >60%)", diagnostics.phase_coherence * 100.0);
    println!("  • Mean Coupling: {:.4}", diagnostics.mean_coupling_strength);
    println!("  • TSP Quality: {:.4}", diagnostics.tsp_quality);
    println!("  • Coloring Balance: {:.4}", diagnostics.coloring_balance);

    // Validate phase coherence
    if diagnostics.phase_coherence > 0.6 {
        println!("  ✅ Phase coherence TARGET MET (>{:.0}%)", 60.0);
    } else {
        println!("  ⚠️  Phase coherence below target ({:.1}% < 60%)",
            diagnostics.phase_coherence * 100.0);
    }

    // Test chromatic coloring separately
    println!("\n🎨 Chromatic Coloring Analysis:");
    test_chromatic_coloring(&coupling_matrix, num_colors)?;

    // Test TSP optimization separately
    println!("\n🗺️  TSP Path Optimization Analysis:");
    test_tsp_optimization(&coupling_matrix)?;

    // Test phase coherence evolution over time
    println!("\n⏱️  Phase Coherence Evolution:");
    test_phase_coherence_evolution(&prct_field)?;

    Ok(())
}

fn generate_coupling_matrix(n: usize) -> Array2<Complex64> {
    let mut coupling = Array2::zeros((n, n));

    // Generate realistic protein-like coupling (exponential decay with distance)
    // Very sparse coupling to ensure 4-colorability
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            // Distance-based coupling (very fast exponential decay)
            // Only nearest neighbors have strong coupling above threshold
            let distance = (i as f64 - j as f64).abs();
            let coupling_strength = 0.8 * (-distance / 1.5).exp(); // Very fast decay

            // Add phase based on sequence position
            let phase = std::f64::consts::PI * (i + j) as f64 / n as f64;

            coupling[[i, j]] = Complex64::from_polar(coupling_strength, phase);
        }
    }

    coupling
}

fn test_chromatic_coloring(coupling_matrix: &Array2<Complex64>, num_colors: usize) -> Result<()> {
    let threshold = 0.3;

    // Test basic coloring
    let start = Instant::now();
    let mut coloring = ChromaticColoring::new(coupling_matrix, num_colors, threshold)?;
    let basic_time = start.elapsed();

    println!("  • Basic coloring: {:.2}ms", basic_time.as_micros() as f64 / 1000.0);
    println!("  • Initial conflicts: {}", coloring.get_conflict_count());

    // Test optimization
    let start = Instant::now();
    coloring.optimize(100, 5.0)?;
    let opt_time = start.elapsed();

    println!("  • Optimization: {:.2}ms", opt_time.as_micros() as f64 / 1000.0);
    println!("  • Final conflicts: {}", coloring.get_conflict_count());
    println!("  • Color balance: {:.4}", coloring.color_balance());

    if coloring.verify_coloring() {
        println!("  ✅ Valid coloring (no conflicts)");
    } else {
        println!("  ⚠️  Coloring has conflicts");
    }

    // Show color distribution
    let distribution = coloring.get_color_distribution();
    println!("  • Color distribution:");
    for color in 0..num_colors {
        let count = distribution.get(&color).unwrap_or(&0);
        println!("    Color {}: {} vertices", color, count);
    }

    Ok(())
}

fn test_tsp_optimization(coupling_matrix: &Array2<Complex64>) -> Result<()> {
    // Test TSP with 2-opt
    let start = Instant::now();
    let mut tsp = TSPPathOptimizer::new(coupling_matrix);
    let initial_length = tsp.get_tour_length();
    let init_time = start.elapsed();

    println!("  • Initial tour: {:.2}ms, length={:.4}",
        init_time.as_micros() as f64 / 1000.0, initial_length);

    // 2-opt optimization
    let start = Instant::now();
    tsp.optimize(100)?;
    let opt_time = start.elapsed();
    let opt_length = tsp.get_tour_length();

    println!("  • 2-opt optimization: {:.2}ms, length={:.4}",
        opt_time.as_micros() as f64 / 1000.0, opt_length);
    println!("  • Improvement: {:.2}%",
        (initial_length - opt_length) / initial_length * 100.0);

    // Simulated annealing
    let start = Instant::now();
    tsp.optimize_annealing(200, 10.0)?;
    let anneal_time = start.elapsed();
    let final_length = tsp.get_tour_length();

    println!("  • Annealing: {:.2}ms, length={:.4}",
        anneal_time.as_micros() as f64 / 1000.0, final_length);
    println!("  • Total improvement: {:.2}%",
        (initial_length - final_length) / initial_length * 100.0);

    println!("  • Path quality: {:.4}", tsp.get_path_quality());
    println!("  • Circularity: {:.4}", tsp.get_circularity());

    if tsp.validate_tour() {
        println!("  ✅ Valid TSP tour");
    } else {
        println!("  ❌ Invalid TSP tour");
    }

    Ok(())
}

fn test_phase_coherence_evolution(prct_field: &PhaseResonanceField) -> Result<()> {
    let time_points = vec![0.0, 1.0, 5.0, 10.0, 20.0];

    println!("  Time (fs)  |  Phase Coherence");
    println!("  -----------|------------------");

    for t in time_points {
        let coherence = prct_field.phase_coherence(t);
        println!("  {:8.1}   |  {:.4} ({:.1}%)",
            t, coherence, coherence * 100.0);
    }

    Ok(())
}
