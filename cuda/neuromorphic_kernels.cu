// CUDA kernels for GPU-accelerated neuromorphic processing
// Spike encoding, reservoir computing, and pattern detection

extern "C" {

/// Rate-based spike encoding kernel
/// Converts input features to spike trains using Poisson process
__global__ void encode_spikes_rate(
    const float* features,      // Input: feature vector
    float* spike_times,         // Output: spike times (ms)
    unsigned int* spike_counts, // Output: spikes per neuron
    unsigned int neuron_count,
    unsigned int feature_dim,
    float window_ms,
    float max_rate_hz,
    float min_rate_hz,
    unsigned long seed
) {
    unsigned int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= neuron_count) return;

    // Map neuron to feature
    unsigned int feature_idx = neuron_idx % feature_dim;
    float feature_value = features[feature_idx];

    // Normalize feature to [0, 1]
    feature_value = fminf(fmaxf(feature_value, 0.0f), 1.0f);

    // Calculate spike rate for this neuron
    float rate_hz = min_rate_hz + feature_value * (max_rate_hz - min_rate_hz);
    float mean_interval_ms = 1000.0f / rate_hz;

    // Generate Poisson spike train
    // Use thread-specific RNG
    unsigned long thread_seed = seed + neuron_idx;
    float t = 0.0f;
    unsigned int spike_count = 0;
    unsigned int spike_offset = neuron_idx * 1000; // Max 1000 spikes per neuron

    while (t < window_ms && spike_count < 1000) {
        // Generate exponential inter-spike interval
        float u = (float)(thread_seed % 1000000) / 1000000.0f;
        thread_seed = thread_seed * 1103515245 + 12345; // LCG
        float interval = -mean_interval_ms * logf(1.0f - u);

        t += interval;
        if (t < window_ms) {
            spike_times[spike_offset + spike_count] = t;
            spike_count++;
        }
    }

    spike_counts[neuron_idx] = spike_count;
}

/// Reservoir state update kernel (ESN/LSM dynamics)
/// Implements: x(t+1) = (1-α)x(t) + α·tanh(W_in·u(t) + W·x(t) + W_fb·y(t))
__global__ void reservoir_update(
    const float* input_spikes,    // Input: spike counts per input neuron
    const float* reservoir_state,  // Input: current reservoir state
    const float* w_in,             // Input: input weight matrix (reservoir_size × input_size)
    const float* w_reservoir,      // Input: recurrent weight matrix (reservoir_size × reservoir_size)
    float* new_state,              // Output: updated reservoir state
    unsigned int reservoir_size,
    unsigned int input_size,
    float leak_rate,               // α (leak rate)
    float spectral_radius
) {
    unsigned int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= reservoir_size) return;

    // Compute input contribution: W_in · u(t)
    float input_contrib = 0.0f;
    for (unsigned int i = 0; i < input_size; i++) {
        input_contrib += w_in[neuron_idx * input_size + i] * input_spikes[i];
    }

    // Compute recurrent contribution: W · x(t)
    float recurrent_contrib = 0.0f;
    for (unsigned int i = 0; i < reservoir_size; i++) {
        recurrent_contrib += w_reservoir[neuron_idx * reservoir_size + i] * reservoir_state[i];
    }

    // Apply non-linearity and leak rate
    float activation = tanhf(input_contrib + recurrent_contrib * spectral_radius);
    new_state[neuron_idx] = (1.0f - leak_rate) * reservoir_state[neuron_idx]
                           + leak_rate * activation;
}

/// Pattern detection kernel using reservoir readout
/// Computes pattern strength from reservoir state
__global__ void detect_patterns(
    const float* reservoir_state,  // Input: reservoir state
    const float* readout_weights,  // Input: readout weight matrix
    float* pattern_strengths,      // Output: detected pattern strengths
    unsigned int reservoir_size,
    unsigned int num_patterns
) {
    unsigned int pattern_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pattern_idx >= num_patterns) return;

    // Linear readout: y = W_out · x(t)
    float strength = 0.0f;
    for (unsigned int i = 0; i < reservoir_size; i++) {
        strength += readout_weights[pattern_idx * reservoir_size + i] * reservoir_state[i];
    }

    // Apply sigmoid for bounded output [0, 1]
    pattern_strengths[pattern_idx] = 1.0f / (1.0f + expf(-strength));
}

/// Compute coherence measure from reservoir state
/// Measures synchronization across reservoir neurons
__global__ void compute_coherence(
    const float* reservoir_state,
    float* coherence,
    unsigned int reservoir_size
) {
    // Use parallel reduction for mean and variance
    __shared__ float shared_state[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load state into shared memory
    shared_state[tid] = (idx < reservoir_size) ? reservoir_state[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction for sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < reservoir_size) {
            shared_state[tid] += shared_state[tid + s];
        }
        __syncthreads();
    }

    // Compute mean
    if (tid == 0) {
        float mean = shared_state[0] / reservoir_size;

        // Compute variance (second pass - simplified for GPU)
        float variance = 0.0f;
        for (unsigned int i = 0; i < reservoir_size; i++) {
            float diff = reservoir_state[i] - mean;
            variance += diff * diff;
        }
        variance /= reservoir_size;

        // Coherence = 1 - normalized_variance
        *coherence = 1.0f / (1.0f + variance);
    }
}

} // extern "C"
