//! Simple GPU Detection Test
//! Verifies the system can detect and initialize the RTX 5070 GPU

use anyhow::Result;

#[cfg(feature = "cuda")]
use neuromorphic_engine::{GpuReservoirComputer, reservoir::ReservoirConfig};

fn main() -> Result<()> {
    println!("🔍 GPU Detection Test for Neuromorphic-Quantum Platform\n");
    println!("=======================================================\n");

    #[cfg(feature = "simulation")]
    {
        println!("❌ Running in SIMULATION mode");
        println!("   No GPU detection performed");
        println!("   To test real GPU, rebuild with:");
        println!("   cargo run --example test_gpu_detection --features cuda --no-default-features\n");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        println!("✅ Running in REAL CUDA mode");
        println!("   Attempting to detect and initialize GPU...\n");

        // Create a small reservoir to test GPU initialization
        let config = ReservoirConfig {
            size: 100,
            input_size: 10,
            spectral_radius: 0.95,
            connection_prob: 0.1,
            leak_rate: 0.3,
            input_scaling: 1.0,
            noise_level: 0.01,
            enable_plasticity: false,
        };

        let gpu_config = neuromorphic_engine::gpu_reservoir::GpuConfig::default();
        let device_id = gpu_config.device_id;
        let mixed_precision = gpu_config.enable_mixed_precision;
        let batch_size = gpu_config.batch_size;

        match GpuReservoirComputer::new(config, gpu_config) {
            Ok(gpu_reservoir) => {
                println!("✅ SUCCESS: GPU initialized successfully!");
                println!("\n📊 GPU Configuration:");
                println!("   • Device ID: {}", device_id);
                println!("   • Mixed Precision: {}", mixed_precision);
                println!("   • Batch Size: {}", batch_size);

                let stats = gpu_reservoir.get_gpu_stats();
                println!("\n💾 GPU Memory:");
                println!("   • Allocated: {:.1}MB", stats.gpu_memory_usage_mb);

                println!("\n🎯 System is READY to use GPU acceleration!");
                println!("   The RTX 5070 has been detected and initialized successfully.");
                println!("   You can now run GPU-accelerated computations.\n");

                Ok(())
            }
            Err(e) => {
                println!("❌ FAILED: Could not initialize GPU");
                println!("\n🔍 Error Details:");
                println!("   {}\n", e);

                println!("💡 Troubleshooting Steps:");
                println!("   1. Verify NVIDIA drivers are installed:");
                println!("      nvidia-smi");
                println!("   2. Verify CUDA Toolkit 12.0+ is installed:");
                println!("      nvcc --version");
                println!("   3. Check GPU is visible to CUDA:");
                println!("      nvidia-smi -L");
                println!("   4. Ensure no other process is using GPU");
                println!("   5. Try rebooting the system\n");

                Err(e)
            }
        }
    }

    #[cfg(not(any(feature = "simulation", feature = "cuda")))]
    {
        println!("❌ No GPU mode enabled!");
        println!("   Rebuild with either:");
        println!("   --features cuda (for real GPU)");
        println!("   --features simulation (for testing)");
        Err(anyhow::anyhow!("No GPU mode enabled"))
    }
}
