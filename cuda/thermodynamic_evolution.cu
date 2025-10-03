/**
 * Thermodynamic Oscillator Network - CUDA Kernel
 *
 * Constitution: Phase 1, Task 1.3
 *
 * GPU-accelerated evolution of thermodynamically consistent oscillator network.
 * Implements Langevin dynamics with:
 * - Kuramoto-style coupling
 * - Thermodynamic damping
 * - Thermal noise (fluctuation-dissipation theorem)
 * - Entropy tracking
 *
 * Performance Contract: <1ms per step for 1024 oscillators
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

// Physical constants
#define KB 1.380649e-23  // Boltzmann constant (J/K)
#define TWO_PI 6.28318530718

/**
 * CUDA kernel for one step of Langevin dynamics
 *
 * Implements: dθ_i/dt = ω_i + Σ_j C_ij sin(θ_j - θ_i) - γ v_i + √(2γk_BT) η(t)
 *
 * @param phases Current phase angles [n_oscillators]
 * @param velocities Current angular velocities [n_oscillators]
 * @param natural_frequencies Natural frequencies ω_i [n_oscillators]
 * @param coupling_matrix Coupling matrix C_ij [n_oscillators x n_oscillators]
 * @param new_phases Output phase angles [n_oscillators]
 * @param new_velocities Output velocities [n_oscillators]
 * @param forces Output forces (for FDT validation) [n_oscillators]
 * @param n_oscillators Number of oscillators
 * @param dt Integration timestep (s)
 * @param damping Damping coefficient γ (rad/s)
 * @param temperature Temperature T (K)
 * @param coupling_strength Overall coupling scale
 * @param rng_states Random number generator states [n_oscillators]
 */
__global__ void langevin_step_kernel(
    const double* __restrict__ phases,
    const double* __restrict__ velocities,
    const double* __restrict__ natural_frequencies,
    const double* __restrict__ coupling_matrix,
    double* __restrict__ new_phases,
    double* __restrict__ new_velocities,
    double* __restrict__ forces,
    int n_oscillators,
    double dt,
    double damping,
    double temperature,
    double coupling_strength,
    curandState* rng_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_oscillators) return;

    // Initialize RNG for this thread
    curandState local_state = rng_states[i];

    // Start with natural frequency
    double force = natural_frequencies[i];

    // Coupling term: Σ_j C_ij sin(θ_j - θ_i)
    double coupling_sum = 0.0;
    for (int j = 0; j < n_oscillators; j++) {
        if (i != j) {
            double phase_diff = phases[j] - phases[i];
            double coupling_coeff = coupling_matrix[i * n_oscillators + j];
            coupling_sum += coupling_coeff * sin(phase_diff);
        }
    }
    force += coupling_strength * coupling_sum;

    // Damping term: -γ v_i
    force -= damping * velocities[i];

    // Thermal noise: √(2γk_BT) η(t)
    // Generate Gaussian noise using Box-Muller
    double u1 = curand_uniform_double(&local_state);
    double u2 = curand_uniform_double(&local_state);
    double gaussian = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);

    double noise_scale = sqrt(2.0 * damping * KB * temperature / dt);
    double thermal_force = noise_scale * gaussian;
    force += thermal_force;

    // Store force for FDT validation
    forces[i] = force;

    // Euler-Maruyama integration for SDE
    new_velocities[i] = velocities[i] + force * dt;
    new_phases[i] = phases[i] + new_velocities[i] * dt;

    // Keep phases in [0, 2π)
    if (new_phases[i] < 0.0) {
        new_phases[i] += TWO_PI * ceil(-new_phases[i] / TWO_PI);
    } else if (new_phases[i] >= TWO_PI) {
        new_phases[i] -= TWO_PI * floor(new_phases[i] / TWO_PI);
    }

    // Update RNG state
    rng_states[i] = local_state;
}

/**
 * Initialize RNG states for each oscillator
 */
__global__ void init_rng_kernel(curandState* states, unsigned long long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curand_init(seed, i, 0, &states[i]);
    }
}

/**
 * Calculate system entropy from phase space distribution
 *
 * Uses Gibbs entropy: S = -k_B Σ p_i ln(p_i)
 *
 * @param velocities Velocity array [n_oscillators]
 * @param n_oscillators Number of oscillators
 * @param temperature System temperature (K)
 * @param entropy Output: system entropy (dimensionless)
 */
__global__ void calculate_entropy_kernel(
    const double* __restrict__ velocities,
    int n_oscillators,
    double temperature,
    double* __restrict__ entropy_out
) {
    // Shared memory for reduction
    extern __shared__ double s_entropy[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_entropy = 0.0;

    if (i < n_oscillators) {
        // Boltzmann probability for this velocity
        double v2 = velocities[i] * velocities[i];
        double p = exp(-v2 / (2.0 * KB * temperature));

        if (p > 1e-10) {
            local_entropy = -KB * p * log(p);
        }
    }

    s_entropy[tid] = local_entropy;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_entropy[tid] += s_entropy[tid + stride];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicAdd(entropy_out, s_entropy[0]);
    }
}

/**
 * Calculate system energy
 *
 * E = Σ_i (1/2 v_i^2) - Σ_{i,j} C_ij cos(θ_j - θ_i)
 *
 * @param phases Phase array [n_oscillators]
 * @param velocities Velocity array [n_oscillators]
 * @param coupling_matrix Coupling matrix [n x n]
 * @param n_oscillators Number of oscillators
 * @param coupling_strength Overall coupling scale
 * @param energy_out Output: system energy (J)
 */
__global__ void calculate_energy_kernel(
    const double* __restrict__ phases,
    const double* __restrict__ velocities,
    const double* __restrict__ coupling_matrix,
    int n_oscillators,
    double coupling_strength,
    double* __restrict__ energy_out
) {
    extern __shared__ double s_energy[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_energy = 0.0;

    if (i < n_oscillators) {
        // Kinetic energy
        local_energy = 0.5 * velocities[i] * velocities[i];

        // Interaction energy (half to avoid double counting)
        double interaction = 0.0;
        for (int j = 0; j < n_oscillators; j++) {
            if (i != j) {
                double phase_diff = phases[j] - phases[i];
                double coupling = coupling_matrix[i * n_oscillators + j];
                interaction -= 0.5 * coupling_strength * coupling * cos(phase_diff);
            }
        }
        local_energy += interaction;
    }

    s_energy[tid] = local_energy;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_energy[tid] += s_energy[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(energy_out, s_energy[0]);
    }
}

/**
 * Calculate phase coherence (Kuramoto order parameter)
 *
 * r = |Σ_i e^(i θ_i)| / N
 *
 * @param phases Phase array [n_oscillators]
 * @param n_oscillators Number of oscillators
 * @param coherence_out Output: coherence value [0, 1]
 */
__global__ void calculate_coherence_kernel(
    const double* __restrict__ phases,
    int n_oscillators,
    double* __restrict__ sum_real,
    double* __restrict__ sum_imag
) {
    extern __shared__ double s_data[];
    double* s_real = s_data;
    double* s_imag = &s_data[blockDim.x];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_real = 0.0;
    double local_imag = 0.0;

    if (i < n_oscillators) {
        local_real = cos(phases[i]);
        local_imag = sin(phases[i]);
    }

    s_real[tid] = local_real;
    s_imag[tid] = local_imag;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_real[tid] += s_real[tid + stride];
            s_imag[tid] += s_imag[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_real, s_real[0]);
        atomicAdd(sum_imag, s_imag[0]);
    }
}

// Host API functions (C linkage for Rust FFI)
extern "C" {

/**
 * Initialize CUDA context for thermodynamic network
 */
cudaError_t thermodynamic_network_init(
    curandState** rng_states_out,
    int n_oscillators,
    unsigned long long seed
) {
    curandState* rng_states;
    cudaError_t status = cudaMalloc(&rng_states, n_oscillators * sizeof(curandState));
    if (status != cudaSuccess) return status;

    int threads = 256;
    int blocks = (n_oscillators + threads - 1) / threads;

    init_rng_kernel<<<blocks, threads>>>(rng_states, seed, n_oscillators);

    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cudaFree(rng_states);
        return status;
    }

    *rng_states_out = rng_states;
    return cudaSuccess;
}

/**
 * Execute one Langevin dynamics step on GPU
 */
cudaError_t thermodynamic_network_step(
    const double* d_phases,
    const double* d_velocities,
    const double* d_natural_frequencies,
    const double* d_coupling_matrix,
    double* d_new_phases,
    double* d_new_velocities,
    double* d_forces,
    int n_oscillators,
    double dt,
    double damping,
    double temperature,
    double coupling_strength,
    curandState* d_rng_states
) {
    int threads = 256;
    int blocks = (n_oscillators + threads - 1) / threads;

    langevin_step_kernel<<<blocks, threads>>>(
        d_phases,
        d_velocities,
        d_natural_frequencies,
        d_coupling_matrix,
        d_new_phases,
        d_new_velocities,
        d_forces,
        n_oscillators,
        dt,
        damping,
        temperature,
        coupling_strength,
        d_rng_states
    );

    return cudaGetLastError();
}

/**
 * Calculate entropy on GPU
 */
cudaError_t thermodynamic_network_entropy(
    const double* d_velocities,
    int n_oscillators,
    double temperature,
    double* d_entropy_out
) {
    // Zero output
    cudaMemset(d_entropy_out, 0, sizeof(double));

    int threads = 256;
    int blocks = (n_oscillators + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(double);

    calculate_entropy_kernel<<<blocks, threads, shared_mem>>>(
        d_velocities,
        n_oscillators,
        temperature,
        d_entropy_out
    );

    return cudaGetLastError();
}

/**
 * Calculate energy on GPU
 */
cudaError_t thermodynamic_network_energy(
    const double* d_phases,
    const double* d_velocities,
    const double* d_coupling_matrix,
    int n_oscillators,
    double coupling_strength,
    double* d_energy_out
) {
    cudaMemset(d_energy_out, 0, sizeof(double));

    int threads = 256;
    int blocks = (n_oscillators + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(double);

    calculate_energy_kernel<<<blocks, threads, shared_mem>>>(
        d_phases,
        d_velocities,
        d_coupling_matrix,
        n_oscillators,
        coupling_strength,
        d_energy_out
    );

    return cudaGetLastError();
}

/**
 * Calculate phase coherence on GPU
 */
cudaError_t thermodynamic_network_coherence(
    const double* d_phases,
    int n_oscillators,
    double* coherence_out
) {
    double *d_sum_real, *d_sum_imag;
    cudaMalloc(&d_sum_real, sizeof(double));
    cudaMalloc(&d_sum_imag, sizeof(double));
    cudaMemset(d_sum_real, 0, sizeof(double));
    cudaMemset(d_sum_imag, 0, sizeof(double));

    int threads = 256;
    int blocks = (n_oscillators + threads - 1) / threads;
    size_t shared_mem = 2 * threads * sizeof(double);

    calculate_coherence_kernel<<<blocks, threads, shared_mem>>>(
        d_phases,
        n_oscillators,
        d_sum_real,
        d_sum_imag
    );

    double sum_real, sum_imag;
    cudaMemcpy(&sum_real, d_sum_real, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_imag, d_sum_imag, sizeof(double), cudaMemcpyDeviceToHost);

    double magnitude = sqrt(sum_real * sum_real + sum_imag * sum_imag);
    *coherence_out = magnitude / n_oscillators;

    cudaFree(d_sum_real);
    cudaFree(d_sum_imag);

    return cudaGetLastError();
}

/**
 * Cleanup CUDA resources
 */
cudaError_t thermodynamic_network_cleanup(curandState* d_rng_states) {
    return cudaFree(d_rng_states);
}

} // extern "C"
