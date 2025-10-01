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

        let input_data = InputData::new(&features);

        // Create encoder
        let mut encoder = SpikeEncoder::new(self.neuron_count, self.window_ms)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Encode
        let engine_spikes = encoder.encode(input_data)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Convert to shared-types
        let spikes: Vec<Spike> = engine_spikes.spikes.iter().map(|s| {
            Spike {
                neuron_id: s.neuron_id,
                time_ms: s.timestamp,
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
        let engine_spikes = neuromorphic_engine::SpikePattern {
            spikes: spikes.spikes.iter().map(|s| neuromorphic_engine::Spike {
                neuron_id: s.neuron_id,
                timestamp: s.time_ms,
            }).collect(),
        };

        // Create reservoir
        let mut reservoir = ReservoirComputer::new(
            self.neuron_count,
            0.9,   // spectral radius
            0.3,   // leak rate
            0.1,   // sparsity
        ).map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Process
        let reservoir_state = reservoir.process(&engine_spikes)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Create pattern detector
        let mut detector = PatternDetector::new(PatternDetectorConfig {
            max_patterns: 100,
            min_support: 3,
            min_confidence: 0.7,
            time_window_ms: self.window_ms,
        });

        // Detect patterns
        let detected = detector.detect_patterns(&reservoir_state);

        // Compute pattern strength
        let pattern_strength = if detected.is_empty() {
            0.0
        } else {
            detected.iter().take(5).map(|p| p.confidence).sum::<f64>() / detected.len().min(5) as f64
        };

        Ok(NeuroState {
            neuron_states: reservoir_state.states.clone(),
            spike_pattern: vec![0; self.neuron_count],
            coherence: reservoir_state.synchrony,
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
