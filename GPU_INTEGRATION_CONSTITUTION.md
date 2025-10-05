# GPU Integration Constitution
## Strict Implementation Guidelines for World-Class Development

**Version:** 1.0
**Authority:** Inviolable
**Scope:** All GPU-accelerated computations in PRISM-AI

---

## Article I: Hexagonal Architecture Compliance

### I.1 Port-Adapter-Engine Separation

**Constitutional Requirement:**
```
Domain (prct-core) ─> Ports (traits) ─> Adapters ─> Engines (GPU implementation)
```

**Implementation Standard:**

1. **Ports MUST be traits** defining domain interfaces
   ```rust
   // src/prct-core/src/ports.rs
   pub trait NeuromorphicPort {
       fn encode_spikes(&self, input: &Array1<f64>) -> Result<SpikePattern>;
       fn process_reservoir(&self, spikes: &SpikePattern) -> Result<ReservoirState>;
   }
   ```

2. **Adapters MUST implement ports** without domain logic
   ```rust
   // src/adapters/src/neuromorphic_adapter.rs
   pub struct NeuromorphicAdapter {
       engine: Arc<GpuReservoir>,  // Infrastructure dependency
   }

   impl NeuromorphicPort for NeuromorphicAdapter {
       fn encode_spikes(&self, input: &Array1<f64>) -> Result<SpikePattern> {
           self.engine.encode_on_gpu(input)  // Delegates to engine
       }
   }
   ```

3. **Engines contain GPU implementation** with no domain knowledge
   ```rust
   // src/neuromorphic/src/gpu_reservoir.rs
   pub struct GpuReservoir {
       device: Arc<CudaContext>,
       kernels: CompiledKernels,
   }

   impl GpuReservoir {
       pub fn encode_on_gpu(&self, input: &[f64]) -> Result<SpikePattern> {
           // Pure GPU implementation
       }
   }
   ```

**VIOLATION:** Unified platform directly implements logic instead of using adapters

**REMEDY:** Platform MUST use adapters, adapters MUST use GPU engines

---

## Article II: GPU Acceleration Mandate

### II.1 GPU-First Principle

**Constitutional Requirement:**
```
IF GPU implementation exists AND GPU is available
THEN GPU MUST be used
ELSE graceful CPU fallback
```

**Implementation Standard:**

```rust
pub struct UnifiedPlatform {
    // Adapters (which use GPU engines internally)
    neuromorphic: Arc<dyn NeuromorphicPort>,
    quantum: Arc<dyn QuantumPort>,
    physics: Arc<dyn PhysicsCouplingPort>,

    // NOT simplified implementations!
}

impl UnifiedPlatform {
    pub fn new(n_dimensions: usize) -> Result<Self> {
        // Try GPU first
        let neuromorphic: Arc<dyn NeuromorphicPort> =
            if let Ok(adapter) = NeuromorphicAdapter::with_gpu() {
                Arc::new(adapter)  // GPU version
            } else {
                Arc::new(NeuromorphicAdapter::cpu_fallback())  // CPU version
            };

        // Repeat for all modules...
    }

    fn neuromorphic_encoding(&mut self, input: &Array1<f64>) -> Result<Output> {
        // Delegate to adapter (NO direct implementation)
        self.neuromorphic.encode_spikes(input)
    }
}
```

**VIOLATION:** Platform uses `input.mapv(|x| x > threshold)` instead of GPU reservoir

**REMEDY:** Must call `self.neuromorphic.encode_spikes()` which uses GPU

---

## Article III: GPU Kernel Loading Standards

### III.1 PTX Runtime Loading Protocol

**Constitutional Requirement:**
```
ALL GPU kernels MUST be loaded via PTX at runtime
NO static linking of .o files
NO FFI to compiled objects
```

**Implementation Standard:**

```rust
pub struct GpuEngine {
    context: Arc<CudaContext>,
    module: CudaModule,
    kernels: HashMap<&'static str, CudaFunction>,
}

impl GpuEngine {
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        // 1. Locate PTX file (multiple paths for robustness)
        let ptx_path = Self::find_ptx_file()?;

        // 2. Load PTX module
        let ptx = cudarc::nvrtc::Ptx::from_file(ptx_path);
        let module = context.load_module(ptx)?;

        // 3. Load all kernel functions
        let mut kernels = HashMap::new();
        for name in Self::KERNEL_NAMES {
            let func = module.load_function(name)?;
            kernels.insert(name, func);
        }

        Ok(Self { context, module, kernels })
    }

    fn find_ptx_file() -> Result<&'static str> {
        // Try multiple locations
        for path in &["target/ptx/module.ptx", "../target/ptx/module.ptx"] {
            if Path::new(path).exists() {
                return Ok(path);
            }
        }
        Err(anyhow!("PTX not found"))
    }
}
```

**COMPLIANCE:** quantum_mlir follows this ✅
**VIOLATION:** Other modules don't use PTX loading
**REMEDY:** Neuromorphic, thermodynamic must follow same pattern

---

## Article IV: Kernel Launch Standards

### IV.1 cudarc Launch Builder Pattern

**Constitutional Requirement:**
```
ALL kernel launches MUST use cudarc's LaunchBuilder
NO raw CUDA API calls
NO driver API shortcuts
```

**Implementation Standard:**

```rust
pub fn execute_kernel(&self, data: &mut CudaSlice<T>) -> Result<()> {
    // 1. Get kernel function
    let func = self.kernels.get("kernel_name")
        .ok_or_else(|| anyhow!("Kernel not loaded"))?;

    // 2. Calculate grid/block dimensions
    let num_blocks = (data.len() + 255) / 256;
    let num_threads = 256;

    let config = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (num_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // 3. Build launch arguments (parameters must outlive launch)
    let stream = self.context.default_stream();
    let param1_i32 = param1 as i32;  // Lifetime extended

    let mut launch_args = stream.launch_builder(func);
    launch_args.arg(data);
    launch_args.arg(&param1_i32);

    // 4. Launch kernel
    unsafe {
        launch_args.launch(config)?;
    }

    // 5. Synchronize if needed
    stream.synchronize()?;

    Ok(())
}
```

**COMPLIANCE:** quantum_mlir follows this ✅
**VIOLATION:** Other modules may use old patterns
**REMEDY:** All must follow this exact pattern

---

## Article V: Memory Management Standards

### V.1 Unified Memory Manager Pattern

**Constitutional Requirement:**
```
Each GPU module MUST have a memory manager
Memory manager MUST accept Arc<CudaContext>
NO internal context creation per module
```

**Implementation Standard:**

```rust
pub struct GpuMemoryManager {
    pub context: Arc<CudaContext>,  // Shared context
    stream: Arc<CudaStream>,
}

impl GpuMemoryManager {
    /// Accept context - don't create it
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        let stream = context.new_stream()?;
        Ok(Self { context, stream })
    }

    pub fn allocate<T: DeviceRepr>(&self, size: usize) -> Result<CudaSlice<T>> {
        let stream = self.context.default_stream();
        stream.alloc_zeros(size)
    }

    pub fn upload<T: DeviceRepr>(&self, data: &[T]) -> Result<CudaSlice<T>> {
        let stream = self.context.default_stream();
        stream.memcpy_stod(data)
    }

    pub fn download<T: DeviceRepr>(&self, gpu_data: &CudaSlice<T>) -> Result<Vec<T>> {
        let stream = self.context.default_stream();
        stream.memcpy_dtov(gpu_data)
    }
}
```

**COMPLIANCE:** quantum_mlir follows this ✅
**VIOLATION:** Neuromorphic may have separate context
**REMEDY:** All modules must share single context

---

## Article VI: Data Flow Verification

### VI.1 No CPU-GPU Ping-Pong

**Constitutional Requirement:**
```
Data flow: CPU -> GPU -> (process on GPU) -> CPU
NO unnecessary transfers
NO processing on CPU that could be GPU
```

**Anti-Pattern (FORBIDDEN):**
```rust
// ❌ BAD: CPU processing
let result = input.mapv(|x| expensive_operation(x));

// ❌ BAD: Download/process/upload
let cpu_data = gpu_data.download()?;
let processed = cpu_process(cpu_data);  // Should be on GPU!
gpu_data.upload(processed)?;
```

**Correct Pattern (REQUIRED):**
```rust
// ✅ GOOD: GPU processing
let gpu_input = memory.upload(input)?;
let gpu_output = kernel.process(gpu_input)?;
let result = memory.download(gpu_output)?;
```

**CURRENT VIOLATION:** Platform does CPU threshold encoding instead of GPU reservoir

---

## Article VII: Kernel Compilation Standards

### VII.1 CUDA Kernel Requirements

**Constitutional Requirement:**
```
ALL CUDA kernels MUST:
1. Use extern "C" (prevent name mangling)
2. Be __global__ (GPU entry point)
3. Use cuDoubleComplex for complex numbers (NO tuples)
4. Handle edge cases (idx >= N checks)
5. Be properly documented
```

**Implementation Standard:**

```cuda
// ✅ CORRECT kernel structure
extern "C" __global__ void process_kernel(
    double* output,           // GPU output buffer
    const double* input,      // GPU input buffer
    int size,                 // Problem size
    double parameter          // Runtime parameter
) {
    // 1. Thread index calculation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Bounds check
    if (idx >= size) return;

    // 3. Actual computation
    double value = input[idx];
    output[idx] = expensive_gpu_operation(value, parameter);

    // 4. Synchronization if needed
    __syncthreads();
}
```

**COMPLIANCE:** quantum_mlir.cu follows this ✅
**CHECK NEEDED:** Other CUDA files must be verified

---

## Article VIII: Integration Layer Requirements

### VIII.1 UnifiedPlatform Integration Standard

**Constitutional Requirement:**
```
UnifiedPlatform MUST:
1. Use dependency injection (adapters passed in)
2. Delegate to ports (NO direct implementation)
3. Coordinate, not compute
4. Be GPU-agnostic (adapters handle GPU)
```

**World-Class Implementation:**

```rust
pub struct UnifiedPlatform {
    // Ports (trait objects)
    neuromorphic: Arc<dyn NeuromorphicPort>,
    quantum: Arc<dyn QuantumPort>,
    coupling: Arc<dyn PhysicsCouplingPort>,

    // Coordination state (minimal)
    n_dimensions: usize,
    spike_history: Vec<SpikePattern>,
}

impl UnifiedPlatform {
    /// Dependency injection - adapters created externally
    pub fn new_with_adapters(
        neuromorphic: Arc<dyn NeuromorphicPort>,
        quantum: Arc<dyn QuantumPort>,
        coupling: Arc<dyn PhysicsCouplingPort>,
        n_dimensions: usize,
    ) -> Self {
        Self {
            neuromorphic,
            quantum,
            coupling,
            n_dimensions,
            spike_history: Vec::new(),
        }
    }

    /// Convenience constructor with GPU auto-detection
    pub fn new(n_dimensions: usize) -> Result<Self> {
        // Create shared CUDA context once
        let cuda_context = Arc::new(CudaContext::new(0)?);

        // Create GPU-enabled adapters
        let neuromorphic = Arc::new(NeuromorphicAdapter::new_gpu(
            cuda_context.clone()
        )?);

        let quantum = Arc::new(QuantumAdapter::new_gpu(
            cuda_context.clone()
        )?);

        let coupling = Arc::new(CouplingAdapter::new(n_dimensions));

        Ok(Self::new_with_adapters(
            neuromorphic,
            quantum,
            coupling,
            n_dimensions
        ))
    }

    /// Phase 1: Neuromorphic encoding
    fn neuromorphic_encoding(&mut self, input: &Array1<f64>) -> Result<Output> {
        // DELEGATE - don't implement
        let spike_pattern = self.neuromorphic.encode_spikes(input)?;
        let reservoir_state = self.neuromorphic.process_reservoir(&spike_pattern)?;
        Ok((reservoir_state, timing))
    }
}
```

**CURRENT VIOLATION:** Platform directly does `input.mapv(|x| x > threshold)`
**MUST FIX:** Platform must call `self.neuromorphic.encode_spikes(input)`

---

## Article IX: Adapter Implementation Standards

### IX.1 NeuromorphicAdapter Specification

**Constitutional Requirement:**
```
Adapter MUST:
1. Accept Arc<CudaContext> in constructor
2. Create GPU engine with that context
3. Delegate ALL computation to engine
4. Handle GPU/CPU fallback transparently
```

**World-Class Implementation:**

```rust
// src/adapters/src/neuromorphic_adapter.rs

use prct_core::ports::NeuromorphicPort;
use neuromorphic_engine::GpuReservoir;

pub struct NeuromorphicAdapter {
    gpu_engine: Option<Arc<GpuReservoir>>,
    n_neurons: usize,
}

impl NeuromorphicAdapter {
    /// Create with GPU acceleration
    pub fn new_gpu(context: Arc<CudaContext>) -> Result<Self> {
        let gpu_engine = GpuReservoir::new(
            context,
            GpuReservoirConfig {
                n_neurons: 1000,
                spectral_radius: 0.95,
                leak_rate: 0.1,
                sparsity: 0.1,
            }
        )?;

        Ok(Self {
            gpu_engine: Some(Arc::new(gpu_engine)),
            n_neurons: 1000,
        })
    }

    /// CPU fallback
    pub fn cpu_fallback() -> Self {
        Self {
            gpu_engine: None,
            n_neurons: 1000,
        }
    }
}

impl NeuromorphicPort for NeuromorphicAdapter {
    fn encode_spikes(&self, input: &Array1<f64>) -> Result<SpikePattern> {
        if let Some(ref engine) = self.gpu_engine {
            // GPU path
            engine.encode_spikes_gpu(input.as_slice().unwrap())
        } else {
            // CPU fallback
            Self::encode_spikes_cpu(input)
        }
    }

    fn process_reservoir(&self, spikes: &SpikePattern) -> Result<ReservoirState> {
        if let Some(ref engine) = self.gpu_engine {
            // GPU path
            engine.propagate_spikes_gpu(spikes)
        } else {
            // CPU fallback
            Self::process_cpu(spikes)
        }
    }
}
```

**MUST EXIST:** Real implementation in adapters/src/neuromorphic_adapter.rs
**MUST USE:** GPU engine from neuromorphic-engine crate

---

## Article X: GPU Engine Implementation Standards

### X.1 GpuReservoir Specification

**Constitutional Requirement:**
```
GPU Engine MUST:
1. Load kernels via PTX at initialization
2. Maintain GPU state (NO CPU copies during processing)
3. Expose clean API to adapter
4. Handle errors with anyhow::Result
```

**World-Class Implementation:**

```rust
// src/neuromorphic/src/gpu_reservoir.rs

pub struct GpuReservoir {
    context: Arc<CudaContext>,
    kernels: ReservoirKernels,

    // GPU state (lives on GPU)
    reservoir_state: Mutex<CudaSlice<f32>>,
    weights: CudaSlice<f32>,
    config: GpuReservoirConfig,
}

struct ReservoirKernels {
    spike_encoding: Arc<CudaFunction>,
    leaky_integration: Arc<CudaFunction>,
    recurrent_dynamics: Arc<CudaFunction>,
}

impl GpuReservoir {
    pub fn new(context: Arc<CudaContext>, config: GpuReservoirConfig) -> Result<Self> {
        // Load PTX
        let ptx = Ptx::from_file("target/ptx/neuromorphic.ptx");
        let module = context.load_module(ptx)?;

        // Load kernel functions
        let kernels = ReservoirKernels {
            spike_encoding: Arc::new(module.load_function("spike_encoding_kernel")?),
            leaky_integration: Arc::new(module.load_function("leaky_integration_kernel")?),
            recurrent_dynamics: Arc::new(module.load_function("recurrent_dynamics_kernel")?),
        };

        // Allocate GPU state
        let stream = context.default_stream();
        let reservoir_state = stream.alloc_zeros(config.n_neurons)?;
        let weights = Self::initialize_weights_gpu(&context, &config)?;

        Ok(Self {
            context,
            kernels,
            reservoir_state: Mutex::new(reservoir_state),
            weights,
            config,
        })
    }

    pub fn encode_spikes_gpu(&self, input: &[f64]) -> Result<SpikePattern> {
        let stream = self.context.default_stream();

        // Upload input to GPU
        let gpu_input = stream.memcpy_stod(input)?;

        // Allocate output
        let mut gpu_spikes = stream.alloc_zeros::<bool>(self.config.n_neurons)?;

        // Launch spike encoding kernel
        let threshold = self.config.spike_threshold;
        let mut launch = stream.launch_builder(&self.kernels.spike_encoding);
        launch.arg(&gpu_input);
        launch.arg(&mut gpu_spikes);
        launch.arg(&threshold);
        launch.arg(&(self.config.n_neurons as i32));

        let config = LaunchConfig {
            grid_dim: ((self.config.n_neurons + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { launch.launch(config)?; }

        // Download result
        let spikes = stream.memcpy_dtov(&gpu_spikes)?;

        Ok(SpikePattern { spikes, timestamp: now() })
    }
}
```

**MUST EXIST:** This full implementation in gpu_reservoir.rs
**MUST BE USED:** By NeuromorphicAdapter
**CURRENTLY:** Not being used!

---

## Article XI: Thermodynamic GPU Standards

### XI.1 GPU Thermodynamic Network

**Constitutional Requirement:**
```
Thermodynamic evolution MUST use GPU for:
1. Free energy computation
2. Entropy tracking
3. Phase transition detection
4. Network state updates
```

**World-Class Implementation:**

```rust
// src/statistical_mechanics/thermodynamic_network.rs

pub struct ThermodynamicNetwork {
    gpu_engine: Option<Arc<GpuThermodynamics>>,
    cpu_fallback: CpuThermodynamics,
}

impl ThermodynamicNetwork {
    pub fn new_with_gpu(context: Arc<CudaContext>) -> Result<Self> {
        let gpu_engine = GpuThermodynamics::new(context)?;
        Ok(Self {
            gpu_engine: Some(Arc::new(gpu_engine)),
            cpu_fallback: CpuThermodynamics::new(),
        })
    }

    pub fn evolve(&mut self, coupling: &CouplingMatrix, dt: f64) -> Result<State> {
        if let Some(ref engine) = self.gpu_engine {
            // GPU path
            engine.evolve_gpu(coupling, dt)
        } else {
            // CPU fallback
            self.cpu_fallback.evolve(coupling, dt)
        }
    }
}

// src/statistical_mechanics/gpu_thermodynamics.rs

pub struct GpuThermodynamics {
    context: Arc<CudaContext>,
    kernels: ThermoKernels,
    state: Mutex<CudaSlice<f64>>,
}

struct ThermoKernels {
    free_energy: Arc<CudaFunction>,
    entropy: Arc<CudaFunction>,
    evolution_step: Arc<CudaFunction>,
}

impl GpuThermodynamics {
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        // Load thermodynamic PTX kernels
        let ptx = Ptx::from_file("target/ptx/thermodynamics.ptx");
        let module = context.load_module(ptx)?;

        let kernels = ThermoKernels {
            free_energy: Arc::new(module.load_function("compute_free_energy")?),
            entropy: Arc::new(module.load_function("compute_entropy")?),
            evolution_step: Arc::new(module.load_function("evolution_step")?),
        };

        // Allocate GPU state
        let state = context.default_stream().alloc_zeros(1000)?;

        Ok(Self { context, kernels, state: Mutex::new(state) })
    }

    pub fn evolve_gpu(&self, coupling: &CouplingMatrix, dt: f64) -> Result<State> {
        // Upload coupling if needed
        // Launch evolution kernel on GPU
        // Compute entropy on GPU
        // Return result
    }
}
```

**MUST IMPLEMENT:** GPU thermodynamics with PTX kernels
**MUST INTEGRATE:** Into thermodynamic_network.rs
**CURRENTLY:** Using CPU only!

---

## Article XII: Transfer Entropy GPU Standards

### XII.1 Information Flow GPU Computation

**Constitutional Requirement:**
```
Transfer entropy computation MUST use GPU for:
1. Probability distribution estimation
2. Conditional entropy calculation
3. Mutual information
```

**World-Class Implementation:**

```rust
// src/cma/transfer_entropy_gpu.rs (already exists!)

pub struct TransferEntropyGpu {
    context: Arc<CudaContext>,
    kernels: TeKernels,
}

impl TransferEntropyGpu {
    pub fn compute_te_gpu(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        // Upload data to GPU
        let gpu_x = self.context.default_stream().memcpy_stod(x)?;
        let gpu_y = self.context.default_stream().memcpy_stod(y)?;

        // Compute on GPU
        let te = self.kernels.transfer_entropy.launch(&gpu_x, &gpu_y)?;

        // Download result
        let result = self.context.default_stream().memcpy_dtov(&te)?;
        Ok(result[0])
    }
}
```

**FILE EXISTS:** src/cma/transfer_entropy_gpu.rs
**MUST USE:** In Phase 2 information flow
**CURRENTLY:** Using CPU TransferEntropy::new()!

---

## Article XIII: Active Inference GPU Standards

### XIII.1 Variational Inference GPU Requirements

**Constitutional Requirement:**
```
Variational inference MUST use GPU for:
1. Belief updates (gradient computations)
2. KL divergence calculations
3. Expected free energy
4. Action selection
```

**World-Class Implementation:**

```rust
// src/active_inference/gpu_inference.rs (already exists!)

pub struct GpuVariationalInference {
    context: Arc<CudaContext>,
    kernels: InferenceKernels,
}

impl GpuVariationalInference {
    pub fn update_beliefs_gpu(
        &self,
        model: &mut HierarchicalModel,
        observations: &Array1<f64>
    ) -> Result<()> {
        // Upload observations to GPU
        // Compute gradients on GPU
        // Update beliefs on GPU
        // Download updated beliefs
    }

    pub fn compute_free_energy_gpu(
        &self,
        beliefs: &CudaSlice<f64>,
        observations: &CudaSlice<f64>
    ) -> Result<f64> {
        // Compute prediction error on GPU
        // Compute KL divergence on GPU
        // Sum on GPU (reduction kernel)
        // Return scalar result
    }
}
```

**FILE EXISTS:** src/active_inference/gpu_inference.rs
**MUST USE:** In Phase 6
**CURRENTLY:** Using CPU VariationalInference!

---

## Article XIV: Implementation Roadmap

### XIV.1 Required Changes to Achieve Full GPU

**Step 1: Context Sharing**
```rust
// unified_platform.rs
pub fn new(n_dimensions: usize) -> Result<Self> {
    // Create single shared CUDA context
    let cuda_context = Arc::new(CudaContext::new(0)?);

    // All GPU modules share this context
}
```

**Step 2: Adapter Integration**
```rust
// Replace direct implementations with adapter calls
- self.neuromorphic.encode_spikes(input)  // Not input.mapv()
- self.quantum.evolve_state()  // Not simplified calc
- self.physics.compute_coupling()  // Not manual computation
```

**Step 3: GPU Engine Activation**
```rust
// Each adapter creates GPU engine
NeuromorphicAdapter::new_gpu(context) -> uses GpuReservoir
QuantumAdapter::new_gpu(context) -> uses QuantumGpuRuntime (✅ already done)
CouplingAdapter::new_gpu(context) -> uses GPU coupling kernels
```

**Step 4: Module-by-Module Conversion**

**Phase 1 - Neuromorphic:**
```
CURRENT: input.mapv(|x| x > threshold)  // CPU
REQUIRED: neuromorphic_adapter.encode_spikes() -> gpu_reservoir.encode() -> spike_encoding_kernel<<<>>>
```

**Phase 2 - Information Flow:**
```
CURRENT: TransferEntropy::new().compute()  // CPU
REQUIRED: transfer_entropy_gpu.compute_te_gpu() -> transfer_entropy_kernel<<<>>>
```

**Phase 4 - Thermodynamic:**
```
CURRENT: thermo_network.evolve()  // CPU
REQUIRED: thermo_network.evolve_gpu() -> evolution_kernel<<<>>>
```

**Phase 6 - Active Inference:**
```
CURRENT: inference_engine.update_beliefs()  // CPU
REQUIRED: gpu_inference.update_beliefs_gpu() -> belief_update_kernel<<<>>>
```

---

## Article XV: Verification Requirements

### XV.1 GPU Execution Proof

**Constitutional Requirement:**
```
MUST verify GPU execution via:
1. nvidia-smi shows GPU utilization
2. CUDA profiler shows kernel launches
3. Timing shows GPU speedup
4. Log messages confirm GPU code path
```

**Verification Code:**

```rust
impl GpuEngine {
    fn verify_gpu_execution(&self) -> Result<()> {
        // Launch test kernel
        let test_data = vec![1.0; 1000];
        let start = Instant::now();
        self.process_on_gpu(&test_data)?;
        let gpu_time = start.elapsed();

        // Compare with CPU
        let start = Instant::now();
        Self::process_on_cpu(&test_data);
        let cpu_time = start.elapsed();

        // GPU should be faster
        if gpu_time >= cpu_time {
            eprintln!("⚠ WARNING: GPU not faster than CPU");
            eprintln!("  GPU: {:?}, CPU: {:?}", gpu_time, cpu_time);
        }

        println!("[GPU Verification] Speedup: {:.1}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64());

        Ok(())
    }
}
```

**MUST IMPLEMENT:** Verification in each GPU module
**MUST RUN:** At initialization to prove GPU active

---

## Article XVI: No Shortcuts Allowed

### XVI.1 Forbidden Practices

**ABSOLUTELY FORBIDDEN:**

❌ **Fake GPU execution:**
```rust
// FORBIDDEN
fn process_on_gpu(&self, data: &[f64]) -> Result<Vec<f64>> {
    println!("[GPU] Processing...");  // Lying
    Ok(process_on_cpu(data))  // Actually CPU!
}
```

❌ **Hardcoded results:**
```rust
// FORBIDDEN
fn compute_metric(&self) -> f64 {
    0.95  // Fake good result
}
```

❌ **Simplified implementations in production:**
```rust
// FORBIDDEN in production
fn expensive_operation(&self, data: &[f64]) -> Result<f64> {
    // TODO: Real implementation
    Ok(0.0)  // Placeholder returning fake value
}
```

❌ **Masking errors:**
```rust
// FORBIDDEN
let result = gpu_compute();
if result.is_err() {
    return Ok(fake_value);  // Hiding failure
}
```

**REQUIRED INSTEAD:**

✅ **Honest GPU execution:**
```rust
fn process_on_gpu(&self, data: &[f64]) -> Result<Vec<f64>> {
    let gpu_data = self.upload(data)?;
    let gpu_result = self.kernel.launch(gpu_data)?;
    self.download(gpu_result)  // Actually did GPU work
}
```

✅ **Real computation:**
```rust
fn compute_metric(&self, state: &State) -> f64 {
    // Actual formula with real inputs
    state.values.iter().map(|x| x.powi(2)).sum() / state.len() as f64
}
```

✅ **Complete implementations:**
```rust
fn expensive_operation(&self, data: &[f64]) -> Result<f64> {
    // Full implementation, not TODO
    let gpu_data = self.upload(data)?;
    let result = self.compute_kernel.launch(gpu_data)?;
    Ok(self.download_scalar(result)?)
}
```

✅ **Transparent errors:**
```rust
let result = gpu_compute()?;  // Propagate errors
// Don't mask, don't fake
```

---

## Article XVII: Testing Requirements

### XVII.1 GPU Verification Tests

**Constitutional Requirement:**
```
MUST have tests that:
1. Verify GPU is actually used (not CPU)
2. Check results match reference implementation
3. Measure actual GPU speedup
4. Confirm memory stays on GPU
```

**Required Test Pattern:**

```rust
#[test]
fn test_gpu_actually_executes() {
    let context = Arc::new(CudaContext::new(0).unwrap());
    let engine = GpuEngine::new(context).unwrap();

    // Process data
    let input = vec![1.0; 1000];
    let result = engine.process_gpu(&input).unwrap();

    // Verify not just returning zeros/defaults
    assert!(result.iter().any(|&x| x != 0.0), "GPU returned all zeros - fake?");

    // Verify matches CPU reference
    let cpu_result = CpuEngine::process(&input);
    for (gpu, cpu) in result.iter().zip(cpu_result.iter()) {
        assert!((gpu - cpu).abs() < 1e-6, "GPU != CPU - incorrect computation");
    }

    // Verify speedup exists
    let gpu_time = benchmark_gpu();
    let cpu_time = benchmark_cpu();
    assert!(gpu_time < cpu_time, "GPU not faster - not actually using GPU?");
}
```

**MUST IMPLEMENT:** For each GPU module
**MUST PASS:** Before claiming GPU acceleration

---

## IMPLEMENTATION CHECKLIST

### Current Status:

- [x] Quantum GPU (PTX loading, real kernels) ✅
- [ ] Neuromorphic GPU (exists but not used) ❌
- [ ] Thermodynamic GPU (exists but not used) ❌
- [ ] Information Flow GPU (exists but not used) ❌
- [ ] Active Inference GPU (exists but not used) ❌
- [x] Hexagonal architecture (ports exist) ✅
- [ ] Adapters properly use GPU engines ❌
- [ ] Platform delegates to adapters ❌

### Required Work:

1. ✅ quantum_mlir: Already compliant
2. ❌ neuromorphic_adapter: Must use GpuReservoir
3. ❌ unified_platform Phase 1: Must call adapter
4. ❌ unified_platform Phase 2: Must use GPU transfer entropy
5. ❌ unified_platform Phase 4: Must use GPU thermodynamics
6. ❌ unified_platform Phase 6: Must use GPU inference
7. ❌ Integration: Wire all adapters with shared context
8. ❌ Verification: Add GPU execution tests

**Estimated effort:** 4-6 hours of rigorous implementation

---

## CONSTITUTIONAL STANDARD FOR SUCCESS

**System passes constitution when:**

✅ ALL computational phases use GPU when available
✅ Platform delegates to adapters (no direct implementation)
✅ Adapters delegate to GPU engines
✅ GPU engines use PTX runtime loading
✅ Single shared CUDA context across all modules
✅ Memory stays on GPU during processing
✅ Tests verify actual GPU execution
✅ Speedup measurements confirm GPU benefit
✅ No hardcoded values
✅ No masked errors
✅ No fake execution

**Current compliance: 2/10** (only quantum + architecture)

**Required for "full GPU" claim: 10/10**

---

## RECOMMENDATION

**Honest disclosure for current state:**

> "PRISM-AI implements GPU-accelerated quantum processing with native
> cuDoubleComplex via PTX runtime loading, achieving 0.03ms execution
> on DIMACS graph coloring benchmarks. Additional modules (neuromorphic,
> thermodynamic, active inference) have GPU implementations available
> but are currently using CPU paths in the integration layer."

**After full implementation:**

> "PRISM-AI implements full GPU-accelerated quantum-neuromorphic fusion
> across all 8 processing phases via PTX runtime loading, achieving
> sub-10ms end-to-end latency on DIMACS benchmarks."

**DO NOT CLAIM the second until ALL checklist items complete.**

---

**This constitution is your implementation guide.**
**Follow it exactly. No shortcuts. No cheating.**
**Publication-worthy means constitution-compliant.**
