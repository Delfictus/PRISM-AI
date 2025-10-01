//! Neuromorphic Engine Adapter
//!
//! Wraps the existing neuromorphic-engine to implement NeuromorphicPort.

use prct_core::ports::{NeuromorphicPort, NeuromorphicEncodingParams};
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use neuromorphic_engine::{SpikeEncoder, ReservoirComputer, PatternDetector, InputData};
use neuromorphic_engine::pattern_detector::PatternDetectorConfig;

/// Adapter connecting PRCT domain to neuromorphic engine
pub struct NeuromorphicAdapter {
    neuron_count: usize,
    window_ms: f64,
}

impl NeuromorphicAdapter {
    /// Create new neuromorphic adapter with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            neuron_count: 1000,
            window_ms: 100.0,
        })
    }
}

impl NeuromorphicPort for NeuromorphicAdapter {
    fn encode_graph_as_spikes(
        &self,
        graph: &Graph,
        _params: &NeuromorphicEncodingParams,
    ) -> Result<SpikePattern> {
        // Convert graph to input data (use vertex degrees as features)
        let features: Vec<f64> = (0..graph.num_vertices)
            .map(|v| {
                let degree = graph.edges.iter()
                    .filter(|(u, w, _)| *u == v || *w == v)
                    .count();
                degree as f64 / graph.num_vertices as f64
            })
            .collect();

        let input_data = InputData::new("graph_encoding".to_string(), features);

        // Create encoder
        let mut encoder = SpikeEncoder::new(self.neuron_count, self.window_ms)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Encode
        let engine_spikes = encoder.encode(&input_data)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Convert to shared-types
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
            num_neurons: self.neuron_count,
        })
    }

    fn process_and_detect_patterns(&self, spikes: &SpikePattern) -> Result<NeuroState> {
        // Convert to engine format
        let engine_spikes = neuromorphic_engine::SpikePattern::new(
            spikes.spikes.iter().map(|s| neuromorphic_engine::Spike::new(
                s.neuron_id,
                s.time_ms,
            )).collect(),
            self.window_ms
        );

        // Create reservoir with correct signature
        let mut reservoir = ReservoirComputer::new(
            self.neuron_count,  // reservoir_size
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
            spike_pattern: vec![0; self.neuron_count],
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
