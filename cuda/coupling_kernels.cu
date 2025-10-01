// CUDA kernels for GPU-accelerated physics coupling
// Kuramoto synchronization and transfer entropy

extern "C" {

/// Kuramoto phase update kernel
/// dθ_i/dt = ω_i + (K/N) Σ_j A_ij sin(θ_j - θ_i)
__global__ void kuramoto_step(
    const float* phases,         // Input: current phases
    const float* frequencies,    // Input: natural frequencies
    const float* coupling,       // Input: coupling matrix (n×n)
    float* new_phases,           // Output: updated phases
    unsigned int n,
    float coupling_strength,
    float dt
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float phase_i = phases[i];
    float omega_i = frequencies[i];

    // Compute coupling term: Σ_j A_ij sin(θ_j - θ_i)
    float coupling_sum = 0.0f;
    for (unsigned int j = 0; j < n; j++) {
        if (i != j) {
            float coupling_ij = coupling[i * n + j];
            float phase_diff = phases[j] - phase_i;
            coupling_sum += coupling_ij * sinf(phase_diff);
        }
    }

    // Update phase: θ_i(t+dt) = θ_i(t) + dt·[ω_i + (K/N)·coupling_sum]
    float dphase = omega_i + (coupling_strength / n) * coupling_sum;
    new_phases[i] = phase_i + dt * dphase;

    // Wrap phase to [-π, π]
    while (new_phases[i] > M_PI) new_phases[i] -= 2.0f * M_PI;
    while (new_phases[i] < -M_PI) new_phases[i] += 2.0f * M_PI;
}

/// Compute Kuramoto order parameter
/// r = |⟨e^(iθ)⟩| = |Σ_k e^(iθ_k)| / n
__global__ void kuramoto_order_parameter(
    const float* phases,    // Input: phases
    float* order_param,     // Output: order parameter
    float* mean_phase,      // Output: mean phase
    unsigned int n
) {
    __shared__ float shared_cos[256];
    __shared__ float shared_sin[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute sin/cos
    if (idx < n) {
        shared_cos[tid] = cosf(phases[idx]);
        shared_sin[tid] = sinf(phases[idx]);
    } else {
        shared_cos[tid] = 0.0f;
        shared_sin[tid] = 0.0f;
    }
    __syncthreads();

    // Parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
            shared_cos[tid] += shared_cos[tid + s];
            shared_sin[tid] += shared_sin[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum_cos = shared_cos[0] / n;
        float sum_sin = shared_sin[0] / n;

        // Order parameter: r = sqrt(⟨cos⟩² + ⟨sin⟩²)
        *order_param = sqrtf(sum_cos * sum_cos + sum_sin * sum_sin);

        // Mean phase: Ψ = atan2(⟨sin⟩, ⟨cos⟩)
        *mean_phase = atan2f(sum_sin, sum_cos);
    }
}

/// Transfer entropy calculation kernel
/// TE(X→Y) measures information flow from X to Y
/// Simplified GPU implementation using time-lagged mutual information
__global__ void transfer_entropy(
    const float* source,       // Input: source time series
    const float* target,       // Input: target time series
    float* te_value,           // Output: transfer entropy (bits)
    unsigned int n,            // Length of time series
    unsigned int lag           // Time lag for causality
) {
    __shared__ float hist_joint[256];    // Joint histogram
    __shared__ float hist_source[256];   // Source marginal
    __shared__ float hist_target[256];   // Target marginal

    unsigned int tid = threadIdx.x;

    // Initialize histograms
    if (tid < 256) {
        hist_joint[tid] = 0.0f;
        hist_source[tid] = 0.0f;
        hist_target[tid] = 0.0f;
    }
    __syncthreads();

    // Build histograms (simplified 8-bit quantization)
    for (unsigned int i = tid; i < n - lag; i += blockDim.x) {
        // Normalize to [0, 255]
        unsigned int s_idx = (unsigned int)(fminf(fmaxf(source[i] * 128.0f + 128.0f, 0.0f), 255.0f));
        unsigned int t_idx = (unsigned int)(fminf(fmaxf(target[i + lag] * 128.0f + 128.0f, 0.0f), 255.0f));

        atomicAdd(&hist_source[s_idx], 1.0f);
        atomicAdd(&hist_target[t_idx], 1.0f);
        atomicAdd(&hist_joint[s_idx], 1.0f); // Simplified joint histogram
    }
    __syncthreads();

    // Compute transfer entropy using Shannon entropy
    // TE = H(target) + H(source) - H(source, target)
    if (tid == 0) {
        float h_source = 0.0f;
        float h_target = 0.0f;
        float h_joint = 0.0f;
        float total = (float)(n - lag);

        for (int i = 0; i < 256; i++) {
            float p_s = hist_source[i] / total;
            float p_t = hist_target[i] / total;
            float p_st = hist_joint[i] / total;

            if (p_s > 1e-10f) h_source -= p_s * log2f(p_s);
            if (p_t > 1e-10f) h_target -= p_t * log2f(p_t);
            if (p_st > 1e-10f) h_joint -= p_st * log2f(p_st);
        }

        *te_value = h_target + h_source - h_joint;
    }
}

/// Compute coupling strength from phase coherence
__global__ void compute_coupling_strength(
    const float* neuro_phases,   // Input: neuromorphic phases
    const float* quantum_phases,  // Input: quantum phases
    float* coupling_strength,     // Output: coupling strength
    unsigned int n_neuro,
    unsigned int n_quantum
) {
    __shared__ float shared_coherence[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float coherence = 0.0f;

    // Compute cross-phase coherence
    if (i < n_neuro) {
        for (unsigned int j = 0; j < n_quantum; j++) {
            float phase_diff = neuro_phases[i] - quantum_phases[j];
            coherence += cosf(phase_diff);
        }
        coherence /= n_quantum;
    }

    shared_coherence[tid] = (i < n_neuro) ? coherence : 0.0f;
    __syncthreads();

    // Parallel reduction for average coherence
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_coherence[tid] += shared_coherence[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(coupling_strength, shared_coherence[0] / n_neuro);
    }
}

/// Build bidirectional coupling matrix
/// A_ij = coherence(i,j) * strength
__global__ void build_coupling_matrix(
    const float* neuro_phases,    // Input: neuromorphic phases
    const float* quantum_phases,   // Input: quantum phases
    float* coupling_matrix,        // Output: coupling matrix
    unsigned int n_total,          // Total neurons (neuro + quantum)
    unsigned int n_neuro,
    float base_strength
) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_total || j >= n_total) return;

    float phase_i = (i < n_neuro) ? neuro_phases[i] : quantum_phases[i - n_neuro];
    float phase_j = (j < n_neuro) ? neuro_phases[j] : quantum_phases[j - n_neuro];

    // Coupling strength proportional to phase coherence
    float phase_diff = phase_i - phase_j;
    float coherence = cosf(phase_diff);

    // Stronger coupling between different subsystems
    float cross_system_factor = ((i < n_neuro) != (j < n_neuro)) ? 1.5f : 1.0f;

    coupling_matrix[i * n_total + j] = base_strength * coherence * cross_system_factor;
}

} // extern "C"
