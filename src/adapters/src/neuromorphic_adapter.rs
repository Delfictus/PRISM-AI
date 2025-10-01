//! Neuromorphic Engine Adapter
//!
//! Wraps the existing neuromorphic-engine to implement NeuromorphicPort.

use prct_core::ports::{NeuromorphicPort, NeuromorphicEncodingParams};
use prct_core::errors::{PRCTError, Result};
use shared_types::*;
use neuromorphic_engine::{SpikeEncoder, ReservoirComputer, PatternDetector, InputData, EncodingMethod};
use std::sync::Arc;
use parking_lot::Mutex;

/// Adapter connecting PRCT domain to neuromorphic engine
pub struct NeuromorphicAdapter {
    spike_encoder: Arc<Mutex<SpikeEncoder>>,
    reservoir: Arc<Mutex<ReservoirComputer>>,
    pattern_detector: Arc<Mutex<PatternDetector>>,
}

impl NeuromorphicAdapter {
    /// Create new neuromorphic adapter with default configuration
    pub fn new() -> Result<Self> {
        let spike_encoder = SpikeEncoder::new(
            EncodingMethod::RateEncoding,
            neuromorphic_engine::EncodingParameters {
                time_window_ms: 100.0,
                max_rate_hz: 100.0,
                num_neurons: 1000,
                spike_threshold: 0.5,
            },
        );

        let reservoir = ReservoirComputer::new(
            1000,  // neurons
            0.1,   // spectral radius
            0.3,   // leak rate
            0.05,  // sparsity
        ).map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        let pattern_detector = PatternDetector::new(neuromorphic_engine::pattern_detector::PatternDetectorConfig {
            max_patterns: 100,
            min_support: 3,
            min_confidence: 0.7,
            time_window_ms: 100.0,
        });

        Ok(Self {
            spike_encoder: Arc::new(Mutex::new(spike_encoder)),
            reservoir: Arc::new(Mutex::new(reservoir)),
            pattern_detector: Arc::new(Mutex::new(pattern_detector)),
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
                // Count neighbors
                let degree = graph.edges.iter()
                    .filter(|(u, w, _)| *u == v || *w == v)
                    .count();
                degree as f64 / graph.num_vertices as f64
            })
            .collect();

        let input_data = InputData::new(&features);

        // Encode as spikes
        let mut encoder = self.spike_encoder.lock();
        let engine_spikes = encoder.encode(input_data)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Convert to shared-types SpikePattern
        let spikes: Vec<Spike> = engine_spikes.spikes.iter().map(|s| {
            Spike {
                neuron_id: s.neuron_id,
                time_ms: s.timestamp,
                amplitude: 1.0,
            }
        }).collect();

        Ok(SpikePattern {
            spikes,
            duration_ms: 100.0,
            num_neurons: 1000,
        })
    }

    fn process_and_detect_patterns(&self, spikes: &SpikePattern) -> Result<NeuroState> {
        // Convert back to engine spike format
        let engine_spikes = neuromorphic_engine::SpikePattern {
            spikes: spikes.spikes.iter().map(|s| neuromorphic_engine::Spike {
                neuron_id: s.neuron_id,
                timestamp: s.time_ms,
            }).collect(),
        };

        // Process through reservoir
        let mut reservoir = self.reservoir.lock();
        let reservoir_state = reservoir.process(&engine_spikes)
            .map_err(|e| PRCTError::NeuromorphicFailed(e.to_string()))?;

        // Detect patterns
        let mut detector = self.pattern_detector.lock();
        let detected = detector.detect_patterns(&reservoir_state);

        // Compute pattern strength (average confidence of top patterns)
        let pattern_strength = if detected.is_empty() {
            0.0
        } else {
            detected.iter().take(5).map(|p| p.confidence).sum::<f64>() / 5.0
        };

        Ok(NeuroState {
            neuron_states: reservoir_state.states.clone(),
            spike_pattern: spikes.spikes.iter().map(|s| s.neuron_id as u8).collect(),
            coherence: reservoir_state.synchrony,
            pattern_strength,
            timestamp_ns: 0,
        })
    }

    fn get_detected_patterns(&self) -> Result<Vec<DetectedPattern>> {
        // Return cached patterns (simplified for now)
        Ok(vec![])
    }
}

impl Default for NeuromorphicAdapter {
    fn default() -> Self {
        Self::new().expect("Failed to create default NeuromorphicAdapter")
    }
}
