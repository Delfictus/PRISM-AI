//! Debug GPU coloring logic

use anyhow::Result;
use ndarray::Array2;
use num_complex::Complex64;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn test_adjacency_download() -> Result<()> {
    println!("Testing adjacency matrix download...\n");

    // Create simple path graph P_4: 0-1-2-3
    let mut coupling = Array2::zeros((4, 4));
    coupling[[0, 1]] = Complex64::new(1.0, 0.0);
    coupling[[1, 0]] = Complex64::new(1.0, 0.0);
    coupling[[1, 2]] = Complex64::new(1.0, 0.0);
    coupling[[2, 1]] = Complex64::new(1.0, 0.0);
    coupling[[2, 3]] = Complex64::new(1.0, 0.0);
    coupling[[3, 2]] = Complex64::new(1.0, 0.0);

    println!("Coupling matrix:");
    println!("{:?}\n", coupling);

    let device = Arc::new(CudaDevice::new(0)?);
    let n = 4;

    // Flatten coupling matrix
    let coupling_flat: Vec<f32> = coupling
        .iter()
        .map(|c| c.norm() as f32)
        .collect();

    // Upload to GPU
    let gpu_coupling = device.htod_sync_copy(&coupling_flat)?;

    // Allocate adjacency matrix (packed bits)
    let adjacency_bytes = (n * n + 7) / 8;
    let gpu_adjacency = device.alloc_zeros::<u8>(adjacency_bytes)?;

    // Load PTX kernel
    let out_dir = std::env::var("OUT_DIR")?;
    let ptx_path = std::path::Path::new(&out_dir).join("graph_coloring.ptx");
    let ptx = std::fs::read_to_string(&ptx_path)?;

    device.load_ptx(ptx.into(), "graph_coloring", &["build_adjacency"])?;

    // Launch kernel
    use cudarc::driver::LaunchConfig;
    let build_adjacency = device.get_func("graph_coloring", "build_adjacency")
        .ok_or_else(|| anyhow::anyhow!("Failed to get build_adjacency kernel"))?;

    let cfg = LaunchConfig::for_num_elems((n * n) as u32);
    let threshold = 0.5f32;

    unsafe {
        build_adjacency.launch(
            cfg,
            (&gpu_coupling, threshold, &gpu_adjacency, n as u32),
        )?;
    }

    device.synchronize()?;

    // Download adjacency
    let packed = device.dtoh_sync_copy(&gpu_adjacency)?;
    println!("Packed adjacency bytes: {:?}\n", packed);

    let mut adjacency = Array2::from_elem((n, n), false);
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let byte_idx = idx / 8;
            let bit_idx = idx % 8;
            if byte_idx < packed.len() {
                adjacency[[i, j]] = (packed[byte_idx] & (1 << bit_idx)) != 0;
            }
        }
    }

    println!("Adjacency matrix:");
    for i in 0..n {
        print!("{}: ", i);
        for j in 0..n {
            print!("{} ", if adjacency[[i, j]] { "1" } else { "0" });
        }
        println!();
    }
    println!();

    // Test coloring manually
    println!("Expected edges:");
    println!("  0-1, 1-2, 2-3");
    println!("\nActual edges from GPU:");
    for i in 0..n {
        for j in (i+1)..n {
            if adjacency[[i, j]] {
                println!("  {}-{}", i, j);
            }
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let _ = std::env::var("LD_LIBRARY_PATH");

    test_adjacency_download()?;

    Ok(())
}
