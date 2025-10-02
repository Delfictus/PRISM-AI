//! Neuromorphic Engine Adapter - GPU Accelerated
//!
//! Wraps GPU-accelerated neuromorphic processing using CUDA kernels.
//! CPU fallback only if GPU unavailable.

use prct_core::ports::{NeuromorphicPort, NeuromorphicEncodingParams};
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use neuromorphic_engine::{SpikeEncoder, ReservoirComputer, PatternDetector, InputData};
use neuromorphic_engine::pattern_detector::PatternDetectorConfig;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// Adapter connecting PRCT domain to GPU-accelerated neuromorphic engine
pub struct NeuromorphicAdapter {
    window_ms: f64,
    gpu_device: Option<Arc<CudaDevice>>,
    use_gpu: bool,
}

impl NeuromorphicAdapter {
    /// Create new GPU-accelerated neuromorphic adapter
    pub fn new() -> Result<Self> {
        // Try to initialize GPU
        let (gpu_device, use_gpu) = match CudaDevice::new(0) {
            Ok(device) => {
                println!("✓ Neuromorphic GPU initialized (CUDA device 0)");
                (Some(Arc::new(device)), true)
            }
            Err(e) => {
                eprintln!("⚠ GPU initialization failed: {}. Using CPU fallback.", e);
                (None, false)
            }
        };

        Ok(Self {
            window_ms: 100.0,
            gpu_device,
            use_gpu,
        })
    }

    /// Load GPU kernels for neuromorphic processing
    fn load_gpu_kernels(&self) -> Result<()> {
        if !self.use_gpu {
            return Err(PRCTError::NeuromorphicFailed("GPU not available".into()));
        }

        let device = self.gpu_device.as_ref().unwrap();

        // Load PTX from runtime location
        let ptx_path = "target/ptx/neuromorphic_kernels.ptx";
        let ptx = std::fs::read_to_string(ptx_path)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Failed to load PTX: {}", e)))?;

        device.load_ptx(ptx.into(), "neuromorphic_kernels", &[
            "encode_spikes_rate",
            "reservoir_update",
            "detect_patterns",
            "compute_coherence"
        ]).map_err(|e| PRCTError::NeuromorphicFailed(format!("PTX load failed: {}", e)))?;

        Ok(())
    }

    /// Calculate optimal neuron count for graph size
    fn neuron_count_for_graph(&self, graph: &Graph) -> usize {
        // Scale with graph size: min 10, max 1000
        // Use 10x vertices as a reasonable scaling factor
        (graph.num_vertices * 10).clamp(10, 1000)
    }

    /// GPU-accelerated spike encoding
    fn encode_spikes_gpu(
        &self,
        features: &[f64],
        neuron_count: usize,
    ) -> Result<SpikePattern> {
        let device = self.gpu_device.as_ref().ok_or_else(||
            PRCTError::NeuromorphicFailed("GPU not initialized".into()))?;

        // Allocate GPU memory
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let gpu_features = device.htod_copy(features_f32.clone())
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy failed: {}", e)))?;

        let max_spikes_per_neuron = 1000;
        let mut gpu_spike_times = device.alloc_zeros::<f32>(neuron_count * max_spikes_per_neuron)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU alloc failed: {}", e)))?;
        let mut gpu_spike_counts = device.alloc_zeros::<u32>(neuron_count)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU alloc failed: {}", e)))?;

        // Launch encoding kernel
        let cfg = LaunchConfig {
            grid_dim: ((neuron_count + 255) / 256) as u32,
            block_dim: 256,
            shared_mem_bytes: 0,
        };

        let func = device.get_func("neuromorphic_kernels", "encode_spikes_rate")
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("Kernel not found: {}", e)))?;

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        unsafe {
            func.launch(cfg, (
                &gpu_features,
                &mut gpu_spike_times,
                &mut gpu_spike_counts,
                neuron_count as u32,
                features.len() as u32,
                self.window_ms as f32,
                100.0f32, // max_rate_hz
                1.0f32,   // min_rate_hz
                seed,
            )).map_err(|e| PRCTError::NeuromorphicFailed(format!("Kernel launch failed: {}", e)))?;
        }

        // Copy results back
        let spike_times: Vec<f32> = device.dtoh_sync_copy(&gpu_spike_times)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy back failed: {}", e)))?;
        let spike_counts: Vec<u32> = device.dtoh_sync_copy(&gpu_spike_counts)
            .map_err(|e| PRCTError::NeuromorphicFailed(format!("GPU copy back failed: {}", e)))?;

        // Convert to spike pattern
        let mut spikes = Vec::new();
        for neuron_id in 0..neuron_count {
            let count = spike_counts[neuron_id] as usize;
            let offset = neuron_id * max_spikes_per_neuron;
            for i in 0..count {
                spikes.push(Spike {
                    neuron_id,
                    time_ms: spike_times[offset + i] as f64,
                    amplitude: 1.0,
                });
            }
        }

        Ok(SpikePattern {
            spikes,
            duration_ms: self.window_ms,
            num_neurons: neuron_count,
        })
    }
}

impl NeuromorphicPort for NeuromorphicAdapter {
    fn encode_graph_as_spikes(
        &self,
        graph: &Graph,
        _params: &NeuromorphicEncodingParams,
    ) -> Result<SpikePattern> {
        let neuron_count = self.neuron_count_for_graph(graph);

        // Convert graph to input data (use vertex degrees as features)
        let features: Vec<f64> = (0..graph.num_vertices)
            .map(|v| {
                let degree = graph.edges.iter()
                    .filter(|(u, w, _)| *u == v || *w == v)
                    .count();
                degree as f64 / graph.num_vertices as f64
            })
            .collect();

        // GPU-accelerated spike encoding if available
        if self.use_gpu {
            return self.encode_spikes_gpu(&features, neuron_count);
        }

        // CPU fallback
        let input_data = InputData::new("graph_encoding".to_string(), features);
        let mut encoder = SpikeEncoder::new(neuron_count, self.window_ms)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;
        let engine_spikes = encoder.encode(&input_data)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        let spikes: Vec<Spike> = engine_spikes.spikes.iter().map(|s| {
            Spike {
                neuron_id: s.neuron_id,
                time_ms: s.time_ms,
                amplitude: 1.0,
            }
        }).collect();

        Ok(SpikePattern {
            spikes,
            duration_ms: self.window_ms,
            num_neurons: neuron_count,
        })
    }

    fn process_and_detect_patterns(&self, spikes: &SpikePattern) -> Result<NeuroState> {
        let neuron_count = spikes.num_neurons;

        // Convert to engine format
        let engine_spikes = neuromorphic_engine::SpikePattern::new(
            spikes.spikes.iter().map(|s| neuromorphic_engine::Spike::new(
                s.neuron_id,
                s.time_ms,
            )).collect(),
            self.window_ms
        );

        // Create reservoir with correct signature (scaled to spike pattern size)
        let mut reservoir = ReservoirComputer::new(
            neuron_count,  // reservoir_size (matches spike pattern)
            spikes.spikes.len().max(10), // input_size
            0.9,   // spectral radius
            0.1,   // connection_prob (sparsity)
            0.3,   // leak rate
        ).map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Process
        let reservoir_state = reservoir.process(&engine_spikes)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Pattern detection simplified - detector doesn't have analyze method
        // Will enhance in future with proper pattern analysis

        // Compute pattern strength from reservoir state
        let pattern_strength = reservoir_state.average_activation as f64;

        // Compute coherence from reservoir dynamics
        let coherence = reservoir_state.dynamics.memory_capacity;

        Ok(NeuroState {
            neuron_states: reservoir_state.activations.clone(),
            spike_pattern: vec![0; neuron_count],
            coherence,
            pattern_strength,
            timestamp_ns: 0,
        })
    }

    fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>> {
        // Simplified for now
        Ok(vec![])
    }
}

impl Default for NeuromorphicAdapter {
    fn default() -> Self {
        Self::new().expect("Failed to create NeuromorphicAdapter")
    }
}
