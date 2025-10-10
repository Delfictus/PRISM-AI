# HFT Demo Phase 2: Neuromorphic Trading Strategy - Detailed Plan

**Phase:** 2 of 7
**Goal:** Implement neuromorphic spiking neural network for HFT trading signals
**Estimated Time:** 10-12 hours
**Dependencies:** Phase 1 (Data Engine) ✅ Complete

---

## 📋 Overview

Phase 2 integrates PRISM-AI's neuromorphic computing capabilities to create a biologically-inspired trading strategy that processes market features as spike trains, similar to how biological neurons process information.

### Key Concepts

**Neuromorphic Computing:**
- Uses spiking neural networks (SNNs) instead of traditional ANNs
- Information encoded as spike timing and rates
- Event-driven, energy-efficient processing
- Temporal dynamics enable prediction of time-series data

**Trading Application:**
- Market features → Spike encoding
- SNN processes spike patterns
- Output spikes → Trading signals (buy/hold/sell)
- GPU acceleration for real-time performance

---

## 🎯 Phase 2 Tasks Breakdown

### Task 2.1: Spike Encoding (3 hours)

**Goal:** Convert market features to spike trains

#### Subtasks

##### 2.1.1: Spike Encoder Structure (45 min)
**Deliverable:** Core spike encoding data structures

```rust
/// Spike train representation
pub struct SpikeTrain {
    /// Neuron spike times (ms)
    pub spike_times: Vec<Vec<f64>>,

    /// Encoding window duration (ms)
    pub duration_ms: f64,

    /// Number of neurons
    pub n_neurons: usize,

    /// Feature metadata
    pub feature_names: Vec<String>,
}

/// Spike encoder configuration
pub struct SpikeEncoderConfig {
    /// Encoding method
    pub encoding_type: EncodingType,

    /// Time window for encoding (ms)
    pub window_ms: f64,

    /// Number of neurons per feature
    pub neurons_per_feature: usize,

    /// Spike rate scaling factor
    pub rate_scale: f64,
}

pub enum EncodingType {
    /// Rate coding: feature value → spike rate
    RateCoding,

    /// Temporal coding: feature value → spike timing
    TemporalCoding,

    /// Population coding: distributed representation
    PopulationCoding,
}

pub struct SpikeEncoder {
    config: SpikeEncoderConfig,
}
```

**Tests:**
- Test spike train creation
- Verify neuron count matches config
- Check duration is set correctly

**ARES Compliance:**
- No hardcoded spike times
- Spike rates computed from feature values
- Different features → different spike patterns

##### 2.1.2: Rate Coding Implementation (1 hour)
**Deliverable:** Rate-based spike encoding

**Algorithm:**
```
Rate Coding:
  For each normalized feature f in [-1, 1]:
    1. Map to positive rate: rate = rate_scale * (f + 1) / 2
       - Example: f=-1 → rate=0 Hz, f=0 → rate=50 Hz, f=1 → rate=100 Hz

    2. Generate Poisson spike train:
       - Inter-spike intervals follow exponential distribution
       - ISI ~ Exp(rate)

    3. Assign to neuron population:
       - Each feature gets neurons_per_feature neurons
       - All neurons encode same feature with slight noise
```

**Implementation:**
```rust
impl SpikeEncoder {
    pub fn encode_rate(&self, features: &MarketFeatures) -> Result<SpikeTrain> {
        let feature_vector = features.to_normalized_vector();
        let n_features = feature_vector.len();
        let n_neurons = n_features * self.config.neurons_per_feature;

        let mut spike_times = vec![Vec::new(); n_neurons];

        for (feat_idx, &feature_val) in feature_vector.iter().enumerate() {
            // Map [-1, 1] to [0, rate_scale]
            let rate_hz = self.config.rate_scale * (feature_val + 1.0) / 2.0;

            // Generate spikes for each neuron in population
            for neuron_offset in 0..self.config.neurons_per_feature {
                let neuron_idx = feat_idx * self.config.neurons_per_feature + neuron_offset;

                // Poisson process spike generation
                let mut t = 0.0;
                while t < self.config.window_ms {
                    let isi = rng.sample(Exp::new(rate_hz / 1000.0)?);
                    t += isi;
                    if t < self.config.window_ms {
                        spike_times[neuron_idx].push(t);
                    }
                }
            }
        }

        Ok(SpikeTrain {
            spike_times,
            duration_ms: self.config.window_ms,
            n_neurons,
            feature_names: Self::feature_names(),
        })
    }
}
```

**Tests:**
- Test high feature value → high spike rate
- Test low feature value → low spike rate
- Test rate scaling matches expected Hz
- **Anti-drift:** Different feature values → different spike counts

**ARES Requirements:**
- Spike rates MUST vary with input features
- No fixed spike patterns
- Stochastic but reproducible with seed

##### 2.1.3: Temporal Coding Implementation (1 hour)
**Deliverable:** Time-to-first-spike encoding

**Algorithm:**
```
Temporal Coding:
  For each normalized feature f in [-1, 1]:
    1. Map to latency: latency_ms = (1 - f) * max_latency / 2
       - High values → early spikes (low latency)
       - Low values → late spikes (high latency)

    2. Generate single spike per neuron at computed time

    3. Add jitter for population coding:
       - jitter ~ Normal(0, sigma_jitter)
```

**Tests:**
- Test high features → early spikes
- Test low features → late spikes
- Test latency range matches config
- **Anti-drift:** Feature changes → latency changes

##### 2.1.4: Integration with Feature Extractor (15 min)
**Deliverable:** End-to-end feature → spike pipeline

```rust
pub fn encode_features(&self, features: &MarketFeatures) -> Result<SpikeTrain> {
    match self.config.encoding_type {
        EncodingType::RateCoding => self.encode_rate(features),
        EncodingType::TemporalCoding => self.encode_temporal(features),
        EncodingType::PopulationCoding => self.encode_population(features),
    }
}
```

**Tests:**
- Integration test: MarketFeatures → SpikeTrain
- Test all encoding types
- Verify spike train properties

---

### Task 2.2: Neuromorphic Network Architecture (3 hours)

**Goal:** Build spiking neural network for trading signal generation

#### Subtasks

##### 2.2.1: Network Configuration (30 min)
**Deliverable:** SNN architecture definition

```rust
pub struct TradingNetworkConfig {
    /// Input layer size (matches encoded features)
    pub n_input: usize,

    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,

    /// Output layer size (3 neurons: buy, hold, sell)
    pub n_output: usize,

    /// Neuron model parameters
    pub neuron_params: NeuronParams,

    /// Synaptic weight initialization
    pub weight_init: WeightInit,

    /// Learning rate (for future training)
    pub learning_rate: f64,
}

pub struct NeuronParams {
    /// Membrane time constant (ms)
    pub tau_mem: f64,

    /// Synaptic time constant (ms)
    pub tau_syn: f64,

    /// Resting potential (mV)
    pub v_rest: f64,

    /// Threshold potential (mV)
    pub v_thresh: f64,

    /// Reset potential (mV)
    pub v_reset: f64,

    /// Refractory period (ms)
    pub t_refrac: f64,
}

pub enum WeightInit {
    /// Xavier/Glorot initialization
    Xavier,

    /// He initialization
    He,

    /// Uniform random
    Uniform { min: f64, max: f64 },

    /// Load from file
    Pretrained { path: String },
}
```

##### 2.2.2: PRISM Integration Setup (1 hour)
**Deliverable:** Re-enable PRISM-AI dependency and verify compilation

**Steps:**
1. Uncomment PRISM-AI dependency in `hft-demo/Cargo.toml`
2. Verify CUDA toolkit is available
3. Test compilation with GPU features
4. Create fallback for CPU-only mode
5. Document CUDA requirements

**Cargo.toml:**
```toml
[dependencies]
prism-ai = { path = "..", features = ["neuromorphic", "gpu"] }
# Fallback: features = ["neuromorphic"] for CPU-only
```

**Tests:**
- Test PRISM imports work
- Verify GPU detection
- Test CPU fallback mode

**Potential Issues:**
- CUDA not installed → Use CPU mode
- CUDA version mismatch → Document required version
- Compilation errors → Debug and document

##### 2.2.3: SNN Layer Implementation (1.5 hours)
**Deliverable:** Leaky Integrate-and-Fire (LIF) neuron layer

**Algorithm:**
```
Leaky Integrate-and-Fire (LIF):
  For each timestep dt:
    1. Update membrane potential:
       dv/dt = (v_rest - v + I_syn) / tau_mem

    2. Update synaptic current:
       dI/dt = -I / tau_syn + Σ(w_ij * s_j)
       where s_j are input spikes

    3. Check threshold:
       if v >= v_thresh:
         - Emit spike
         - v = v_reset
         - Enter refractory period
```

```rust
pub struct TradingNetwork {
    config: TradingNetworkConfig,

    /// Layer neuron states
    layers: Vec<LayerState>,

    /// Inter-layer synaptic weights
    weights: Vec<Array2<f64>>,

    /// Spike history for each layer
    spike_history: Vec<Vec<Vec<f64>>>,
}

pub struct LayerState {
    /// Membrane potentials (mV)
    pub v_mem: Array1<f64>,

    /// Synaptic currents (pA)
    pub i_syn: Array1<f64>,

    /// Refractory countdown (ms)
    pub refrac_count: Array1<f64>,
}

impl TradingNetwork {
    pub fn new(config: TradingNetworkConfig) -> Result<Self> {
        // Initialize layers
        // Initialize weights
        // Setup PRISM neuromorphic engine
    }

    pub fn process_spikes(
        &mut self,
        input_spikes: &SpikeTrain,
        duration_ms: f64,
    ) -> Result<TradingSignal> {
        // Run SNN simulation
        // Collect output spikes
        // Decode to trading signal
    }
}
```

**Tests:**
- Test layer initialization
- Test spike propagation through network
- Test output spike generation
- **Anti-drift:** Different inputs → different outputs

---

### Task 2.3: Signal Generation (2 hours)

**Goal:** Convert output spikes to trading signals

#### Subtasks

##### 2.3.1: Output Spike Decoding (1 hour)
**Deliverable:** Spike → trading signal decoder

```rust
pub struct TradingSignal {
    /// Signal type
    pub signal: SignalType,

    /// Confidence [0, 1]
    pub confidence: f64,

    /// Signal timestamp
    pub timestamp_ns: u64,

    /// Contributing spike pattern
    pub spike_pattern: SpikePattern,
}

pub enum SignalType {
    Buy,
    Hold,
    Sell,
}

pub struct SignalDecoder {
    /// Decoding window (ms)
    pub window_ms: f64,

    /// Confidence threshold
    pub confidence_threshold: f64,
}

impl SignalDecoder {
    pub fn decode(&self, output_spikes: &SpikeTrain) -> Result<TradingSignal> {
        // Count spikes for each output neuron (buy=0, hold=1, sell=2)
        let buy_count = output_spikes.spike_times[0].len();
        let hold_count = output_spikes.spike_times[1].len();
        let sell_count = output_spikes.spike_times[2].len();

        let total_spikes = buy_count + hold_count + sell_count;

        if total_spikes == 0 {
            return Ok(TradingSignal {
                signal: SignalType::Hold,
                confidence: 0.0,
                timestamp_ns: current_time,
                spike_pattern: self.analyze_pattern(output_spikes),
            });
        }

        // Winner-takes-all with confidence
        let (signal, max_count) = if buy_count > hold_count && buy_count > sell_count {
            (SignalType::Buy, buy_count)
        } else if sell_count > buy_count && sell_count > hold_count {
            (SignalType::Sell, sell_count)
        } else {
            (SignalType::Hold, hold_count)
        };

        let confidence = max_count as f64 / total_spikes as f64;

        Ok(TradingSignal { signal, confidence, timestamp_ns, spike_pattern })
    }
}
```

**ARES Requirements:**
- Confidence COMPUTED from spike counts
- No hardcoded probabilities
- Signal varies with output spike pattern

**Tests:**
- Test high buy spikes → Buy signal
- Test high sell spikes → Sell signal
- Test balanced spikes → Hold signal
- Test confidence calculation accuracy

##### 2.3.2: Signal Filtering (30 min)
**Deliverable:** Confidence-based signal filtering

```rust
impl SignalDecoder {
    pub fn should_execute(&self, signal: &TradingSignal) -> bool {
        signal.confidence >= self.confidence_threshold
    }

    pub fn apply_momentum_filter(
        &self,
        signal: &TradingSignal,
        prev_signals: &[TradingSignal],
    ) -> TradingSignal {
        // Require N consecutive signals for reversals
        // Allow immediate continuation of current direction
    }
}
```

**Tests:**
- Test confidence threshold filtering
- Test momentum filter for reversals
- Test signal consistency

##### 2.3.3: Integration Tests (30 min)
**Deliverable:** End-to-end pipeline tests

```rust
#[test]
fn test_full_neuromorphic_pipeline() {
    // Create sample features
    let features = create_test_features();

    // Encode to spikes
    let encoder = SpikeEncoder::new(config);
    let spike_train = encoder.encode_features(&features).unwrap();

    // Process with SNN
    let mut network = TradingNetwork::new(network_config).unwrap();
    let output_spikes = network.process_spikes(&spike_train, 100.0).unwrap();

    // Decode to signal
    let decoder = SignalDecoder::new(decoder_config);
    let signal = decoder.decode(&output_spikes).unwrap();

    // Verify signal is valid
    assert!(matches!(signal.signal, SignalType::Buy | SignalType::Hold | SignalType::Sell));
    assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
}
```

---

### Task 2.4: GPU Acceleration Setup (2 hours)

**Goal:** Enable GPU acceleration for SNN processing

#### Subtasks

##### 2.4.1: CUDA Environment Validation (30 min)
**Deliverable:** CUDA setup verification

**Checklist:**
- [ ] CUDA Toolkit installed (version 11.0+)
- [ ] cuDNN library available
- [ ] GPU detected by system
- [ ] PRISM-AI GPU features compile
- [ ] Basic GPU operations work

**Script:**
```bash
# Check CUDA installation
nvcc --version

# Check GPU availability
nvidia-smi

# Test PRISM GPU compilation
cargo build -p hft-demo --features gpu

# Run GPU test
cargo test -p hft-demo test_gpu_spike_processing
```

##### 2.4.2: GPU Spike Processing (1 hour)
**Deliverable:** GPU-accelerated spike propagation

```rust
pub struct GpuSpikeProcessor {
    /// CUDA context
    context: CudaContext,

    /// Device memory for network state
    device_state: DeviceBuffer<f32>,

    /// Device memory for weights
    device_weights: DeviceBuffer<f32>,
}

impl GpuSpikeProcessor {
    pub fn process_layer_gpu(
        &mut self,
        input_spikes: &[Vec<f64>],
        weights: &Array2<f64>,
        duration_ms: f64,
    ) -> Result<Vec<Vec<f64>>> {
        // Transfer data to GPU
        // Launch CUDA kernel for LIF simulation
        // Transfer results back to CPU
    }
}
```

**CUDA Kernel (simplified):**
```cuda
__global__ void lif_step(
    float* v_mem,
    float* i_syn,
    const float* weights,
    const int* input_spikes,
    float dt,
    int n_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;

    // LIF update equations
    v_mem[idx] += dt * ((V_REST - v_mem[idx] + i_syn[idx]) / TAU_MEM);
    i_syn[idx] += dt * (-i_syn[idx] / TAU_SYN);

    // Check threshold
    if (v_mem[idx] >= V_THRESH) {
        // Emit spike
        v_mem[idx] = V_RESET;
    }
}
```

**Tests:**
- Test GPU vs CPU results match
- Benchmark GPU speedup
- Test with different network sizes

##### 2.4.3: CPU Fallback Implementation (30 min)
**Deliverable:** Graceful degradation for systems without GPU

```rust
pub enum ProcessingBackend {
    Gpu(GpuSpikeProcessor),
    Cpu(CpuSpikeProcessor),
}

impl TradingNetwork {
    pub fn new_auto(config: TradingNetworkConfig) -> Result<Self> {
        let backend = if cuda_available() {
            ProcessingBackend::Gpu(GpuSpikeProcessor::new()?)
        } else {
            log::warn!("GPU not available, using CPU backend");
            ProcessingBackend::Cpu(CpuSpikeProcessor::new())
        };

        Ok(Self { config, backend, /* ... */ })
    }
}
```

**Tests:**
- Test auto-detection
- Test CPU backend correctness
- Test backend switching

---

### Task 2.5: Strategy Integration (2 hours)

**Goal:** Integrate neuromorphic strategy with market data pipeline

#### Subtasks

##### 2.5.1: Strategy Runner (1 hour)
**Deliverable:** Main strategy execution loop

```rust
pub struct NeuromorphicStrategy {
    /// Feature extractor
    feature_extractor: FeatureExtractor,

    /// Spike encoder
    spike_encoder: SpikeEncoder,

    /// SNN trading network
    network: TradingNetwork,

    /// Signal decoder
    signal_decoder: SignalDecoder,

    /// Strategy state
    state: StrategyState,
}

pub struct StrategyState {
    /// Current position
    pub position: Position,

    /// Recent signals
    pub signal_history: VecDeque<TradingSignal>,

    /// Performance metrics
    pub metrics: StrategyMetrics,
}

pub enum Position {
    Long { size: f64, entry_price: f64 },
    Short { size: f64, entry_price: f64 },
    Flat,
}

impl NeuromorphicStrategy {
    pub async fn process_tick(&mut self, tick: MarketTick) -> Result<Option<TradeAction>> {
        // 1. Add tick to feature extractor
        self.feature_extractor.add_tick(tick);

        // 2. Extract features
        let features = match self.feature_extractor.extract_features()? {
            Some(f) => f,
            None => return Ok(None), // Not enough data yet
        };

        // 3. Encode to spikes
        let spike_train = self.spike_encoder.encode_features(&features)?;

        // 4. Process with SNN
        let output_spikes = self.network.process_spikes(&spike_train, 100.0)?;

        // 5. Decode to signal
        let signal = self.signal_decoder.decode(&output_spikes)?;

        // 6. Update signal history
        self.signal_history.push_back(signal.clone());
        if self.signal_history.len() > 10 {
            self.signal_history.pop_front();
        }

        // 7. Generate trade action
        let action = self.generate_action(&signal)?;

        // 8. Update state
        if let Some(ref trade) = action {
            self.update_position(trade, tick.price)?;
        }

        Ok(action)
    }

    fn generate_action(&self, signal: &TradingSignal) -> Result<Option<TradeAction>> {
        // Risk management
        // Position sizing
        // Entry/exit logic
    }
}
```

##### 2.5.2: Risk Management (45 min)
**Deliverable:** Position sizing and risk controls

```rust
pub struct RiskConfig {
    /// Max position size (% of capital)
    pub max_position_pct: f64,

    /// Stop loss (% from entry)
    pub stop_loss_pct: f64,

    /// Take profit (% from entry)
    pub take_profit_pct: f64,

    /// Max daily loss (% of capital)
    pub max_daily_loss_pct: f64,
}

impl NeuromorphicStrategy {
    fn apply_risk_management(
        &self,
        signal: &TradingSignal,
        current_price: f64,
    ) -> Option<TradeAction> {
        // Check position limits
        // Check stop loss / take profit
        // Check daily loss limit
        // Apply position sizing
    }
}
```

##### 2.5.3: Performance Tracking (15 min)
**Deliverable:** Strategy metrics collection

```rust
pub struct StrategyMetrics {
    /// Total trades executed
    pub total_trades: u64,

    /// Win rate
    pub win_rate: f64,

    /// Average profit per trade
    pub avg_profit: f64,

    /// Current equity
    pub equity: f64,

    /// Max drawdown
    pub max_drawdown: f64,
}
```

---

## 🧪 Testing Strategy

### Unit Tests (per task)
- Each subtask includes specific unit tests
- Focus on ARES compliance (no hardcoded values)
- Test edge cases and error conditions

### Integration Tests
```rust
#[test]
fn test_end_to_end_strategy() {
    // Load sample data
    let ticks = load_sample_ticks();

    // Create strategy
    let mut strategy = NeuromorphicStrategy::new(config);

    // Process all ticks
    let mut actions = Vec::new();
    for tick in ticks {
        if let Some(action) = strategy.process_tick(tick).await? {
            actions.push(action);
        }
    }

    // Verify strategy executed trades
    assert!(actions.len() > 0);

    // Verify metrics are computed
    let metrics = strategy.state.metrics;
    assert!(metrics.total_trades > 0);
}
```

### Performance Tests
- Benchmark spike encoding speed
- Benchmark SNN processing latency
- Benchmark GPU vs CPU performance
- Target: <10ms per tick processing

---

## 📊 Success Criteria

### Functional Requirements
- ✅ Market features encode to spike trains
- ✅ SNN processes spikes and generates output
- ✅ Output spikes decode to trading signals
- ✅ Strategy executes trades based on signals
- ✅ GPU acceleration working (or CPU fallback)

### Performance Requirements
- ✅ <10ms per tick processing (with GPU)
- ✅ >100 ticks/sec throughput
- ✅ <500MB memory usage

### ARES Compliance
- ✅ All spike rates computed from features
- ✅ All trading signals computed from spikes
- ✅ No hardcoded probabilities or decisions
- ✅ Different market conditions → different signals

### Test Coverage
- ✅ All public methods tested
- ✅ Integration tests pass
- ✅ GPU tests pass (or skip if no GPU)
- ✅ Anti-drift tests verify variability

---

## 🚨 Risk Mitigation

### Risk 1: CUDA Compilation Issues
**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Document CUDA requirements upfront
- Implement CPU fallback early
- Test on system without GPU
- Provide Docker image with CUDA pre-installed

### Risk 2: SNN Convergence/Stability
**Likelihood:** Medium
**Impact:** Medium
**Mitigation:**
- Start with simple 2-layer network
- Use tested neuron parameters from literature
- Add debugging/visualization tools
- Test with synthetic data first

### Risk 3: Real-time Performance
**Likelihood:** Low
**Impact:** Medium
**Mitigation:**
- Profile early and often
- Optimize hot paths
- Use GPU for large networks
- Cache intermediate results

### Risk 4: Signal Quality
**Likelihood:** Medium
**Impact:** High
**Mitigation:**
- Test with multiple encoding schemes
- Tune confidence thresholds empirically
- Add signal filtering logic
- Compare against baseline strategy

---

## 📁 File Structure

```
hft-demo/src/
├── neuromorphic/
│   ├── mod.rs                  # Module exports
│   ├── spike_encoder.rs        # Task 2.1
│   ├── network.rs              # Task 2.2
│   ├── signal_decoder.rs       # Task 2.3
│   ├── gpu_processor.rs        # Task 2.4 (optional)
│   └── strategy.rs             # Task 2.5
└── lib.rs                      # Add neuromorphic module

hft-demo/tests/
└── neuromorphic_tests.rs       # Integration tests

docs/obsidian-vault/04-Development/
└── HFT Demo Phase 2 Progress.md  # Progress tracking
```

---

## 🔄 Dependencies

### Internal
- `market_data::MarketFeatures` (Phase 1)
- `market_data::MarketTick` (Phase 1)
- `market_data::FeatureExtractor` (Phase 1)

### External Crates
```toml
[dependencies]
prism-ai = { path = "..", features = ["neuromorphic", "gpu"] }
rand = "0.8"
rand_distr = "0.4"
ndarray = "0.15"
anyhow = "1.0"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
log = "0.4"
```

### System Requirements
- CUDA Toolkit 11.0+ (optional, for GPU)
- cuDNN 8.0+ (optional, for GPU)
- 4GB+ RAM
- GPU with 2GB+ VRAM (optional)

---

## 📈 Timeline

| Task | Estimated | Cumulative | Priority |
|------|-----------|------------|----------|
| 2.1: Spike Encoding | 3h | 3h | Critical |
| 2.2: Network Architecture | 3h | 6h | Critical |
| 2.3: Signal Generation | 2h | 8h | Critical |
| 2.4: GPU Acceleration | 2h | 10h | Medium |
| 2.5: Strategy Integration | 2h | 12h | Critical |

**Buffer Time:** +2h for debugging/refinement
**Total Estimate:** 10-14 hours

---

## 🎯 Phase 2 Milestones

### Milestone 1: Spike Encoding Works (3h)
- ✅ Features convert to spike trains
- ✅ Rate coding implemented
- ✅ Tests passing

### Milestone 2: SNN Processes Spikes (6h)
- ✅ PRISM integration working
- ✅ Network architecture defined
- ✅ Spikes propagate through layers

### Milestone 3: Signals Generated (8h)
- ✅ Output spikes decode to Buy/Hold/Sell
- ✅ Confidence calculated from spike counts
- ✅ End-to-end pipeline working

### Milestone 4: GPU Acceleration (10h)
- ✅ GPU processing implemented OR
- ✅ CPU fallback working
- ✅ Performance benchmarks collected

### Milestone 5: Strategy Complete (12h)
- ✅ Strategy executes trades
- ✅ Risk management applied
- ✅ All tests passing
- ✅ Ready for Phase 3 (backtesting)

---

## 📝 ARES Anti-Drift Checklist

For each component, verify:

### Spike Encoding
- [ ] Spike rates computed from feature values
- [ ] No hardcoded spike patterns
- [ ] Different features → different spike trains
- [ ] Test verifies variability

### SNN Processing
- [ ] Neuron states computed from spike inputs
- [ ] No hardcoded membrane potentials
- [ ] Different spike inputs → different outputs
- [ ] Test verifies network is not constant

### Signal Generation
- [ ] Confidence computed from spike counts
- [ ] No hardcoded probabilities
- [ ] Different spike patterns → different signals
- [ ] Test verifies signal variability

### Strategy
- [ ] Trade decisions based on computed signals
- [ ] Position sizing based on risk calculations
- [ ] No hardcoded trade frequencies
- [ ] Test verifies strategy adapts to market

---

## 📚 References

### Neuromorphic Computing
- Izhikevich, E. M. (2003). "Simple model of spiking neurons"
- Maass, W. (1997). "Networks of spiking neurons"
- Pfeiffer, M. & Pfeil, T. (2018). "Deep learning with spiking neurons"

### Spike Encoding
- Bohte, S. M. (2004). "The evidence for neural information encoding"
- Gütig, R. (2014). "To spike, or when to spike?"

### Trading Applications
- Tsai, C. H. & Wang, S. P. (2021). "Stock price prediction using SNN"
- Lin, Y. et al. (2022). "Neuromorphic trading systems"

---

*Plan Date: 2025-10-10*
*Status: Ready to begin*
*Prerequisites: Phase 1 complete ✅*
