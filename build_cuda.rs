// Build script for CUDA kernel compilation

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/graph_coloring.cu");
    println!("cargo:rerun-if-changed=cuda/tsp_solver.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Check if nvcc is available
    let nvcc_check = Command::new("nvcc").arg("--version").output();

    if nvcc_check.is_err() {
        println!("cargo:warning=nvcc not found, skipping CUDA compilation");
        println!("cargo:warning=GPU features will not be available");

        // Create empty PTX files as placeholders
        let graph_coloring_ptx = out_dir.join("graph_coloring.ptx");
        let tsp_solver_ptx = out_dir.join("tsp_solver.ptx");
        std::fs::write(&graph_coloring_ptx, "// CUDA not available\n").unwrap();
        std::fs::write(&tsp_solver_ptx, "// CUDA not available\n").unwrap();
        return;
    }

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    // Compile graph coloring kernel
    let graph_coloring_file = "cuda/graph_coloring.cu";
    let graph_coloring_ptx = out_dir.join("graph_coloring.ptx");

    let status = Command::new("nvcc")
        .args(&[
            "--ptx",
            "--gpu-architecture=sm_89", // RTX 5070 (Blackwell)
            "-o",
            graph_coloring_ptx.to_str().unwrap(),
            graph_coloring_file,
            "--use_fast_math",
            "--generate-line-info",
        ])
        .status()
        .expect("Failed to execute nvcc for graph_coloring.cu");

    if !status.success() {
        panic!("nvcc compilation failed for graph_coloring.cu");
    }

    // Compile TSP solver kernel
    let tsp_solver_file = "cuda/tsp_solver.cu";
    let tsp_solver_ptx = out_dir.join("tsp_solver.ptx");

    let status = Command::new("nvcc")
        .args(&[
            "--ptx",
            "--gpu-architecture=sm_89", // RTX 5070 (Blackwell)
            "-o",
            tsp_solver_ptx.to_str().unwrap(),
            tsp_solver_file,
            "--use_fast_math",
            "--generate-line-info",
        ])
        .status()
        .expect("Failed to execute nvcc for tsp_solver.cu");

    if !status.success() {
        panic!("nvcc compilation failed for tsp_solver.cu");
    }

    // Copy PTX files to a known location for runtime access
    let project_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let runtime_ptx_dir = project_root.join("target").join("ptx");
    std::fs::create_dir_all(&runtime_ptx_dir).unwrap();

    std::fs::copy(&graph_coloring_ptx, runtime_ptx_dir.join("graph_coloring.ptx")).unwrap();
    std::fs::copy(&tsp_solver_ptx, runtime_ptx_dir.join("tsp_solver.ptx")).unwrap();

    println!("cargo:warning=CUDA kernels compiled successfully (graph_coloring.cu + tsp_solver.cu)");
    println!("cargo:warning=PTX files copied to target/ptx/ for runtime access");
}
