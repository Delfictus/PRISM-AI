// CUDA kernels for GPU-accelerated Traveling Salesman Problem (TSP)

extern "C" {

/// Compute distance matrix from coupling strengths (parallel)
/// Each thread computes one distance entry
__global__ void compute_distance_matrix(
    const float* coupling_real,      // Input: n×n coupling matrix real part
    const float* coupling_imag,      // Input: n×n coupling matrix imag part
    float* distances,                 // Output: n×n distance matrix
    unsigned int n                    // Input: number of cities
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_entries = n * n;

    if (idx >= total_entries) return;

    unsigned int i = idx / n;
    unsigned int j = idx % n;

    if (i == j) {
        distances[idx] = 0.0f;
    } else {
        // Distance = 1 / coupling_strength (normalized)
        float real = coupling_real[idx];
        float imag = coupling_imag[idx];
        float coupling_norm = sqrtf(real * real + imag * imag);

        // Avoid division by zero
        distances[idx] = 1.0f / fmaxf(coupling_norm, 1e-10f);
    }
}

/// Normalize distance matrix (find max, divide all by max)
__global__ void normalize_distances(
    float* distances,                 // Input/Output: distance matrix
    float max_distance,               // Input: maximum distance found
    unsigned int n                    // Input: number of cities
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_entries = n * n;

    if (idx >= total_entries) return;

    if (max_distance > 1e-10f) {
        distances[idx] /= max_distance;
    }
}

/// Find maximum value in array (parallel reduction)
__global__ void find_max_distance(
    const float* distances,           // Input: distance matrix
    float* partial_maxs,              // Output: partial maximums per block
    unsigned int n                    // Input: number of cities
) {
    extern __shared__ float shared_max[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_entries = n * n;

    // Load data into shared memory
    float local_max = 0.0f;
    if (idx < total_entries) {
        local_max = distances[idx];
    }
    shared_max[tid] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_maxs[blockIdx.x] = shared_max[0];
    }
}

/// Calculate tour length given a tour permutation
__global__ void calculate_tour_length(
    const float* distances,           // Input: n×n distance matrix
    const unsigned int* tour,         // Input: tour permutation
    float* tour_length,               // Output: total tour length
    unsigned int n                    // Input: number of cities
) {
    extern __shared__ float shared_length[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one edge length
    float local_length = 0.0f;
    if (idx < n) {
        unsigned int from = tour[idx];
        unsigned int to = tour[(idx + 1) % n];
        local_length = distances[from * n + to];
    }
    shared_length[tid] = local_length;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < n) {
            shared_length[tid] += shared_length[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicAdd(tour_length, shared_length[0]);
    }
}

/// Evaluate all 2-opt swaps in parallel (find best improvement)
/// For each position pair (i, j), calculate delta = new_length - old_length
__global__ void evaluate_2opt_swaps(
    const float* distances,           // Input: n×n distance matrix
    const unsigned int* tour,         // Input: current tour
    float* deltas,                    // Output: delta for each swap
    unsigned int* swap_pairs,         // Output: (i, j) pairs
    unsigned int n                    // Input: number of cities
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of valid 2-opt swaps: n * (n - 3) / 2
    // For each i from 0 to n-2, j from i+2 to n-1
    unsigned int total_swaps = n * (n - 3) / 2;

    if (idx >= total_swaps) return;

    // Decode idx into (i, j) pair
    // This is a bit tricky - we need to map linear index to triangular matrix
    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int remaining = idx;

    for (unsigned int ii = 0; ii < n - 1; ii++) {
        unsigned int swaps_for_i = n - ii - 2;
        if (remaining < swaps_for_i) {
            i = ii;
            j = ii + 2 + remaining;
            break;
        }
        remaining -= swaps_for_i;
    }

    // Calculate 2-opt delta
    // Current edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1])
    // New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1])
    unsigned int v1 = tour[i];
    unsigned int v2 = tour[(i + 1) % n];
    unsigned int v3 = tour[j];
    unsigned int v4 = tour[(j + 1) % n];

    float current_length = distances[v1 * n + v2] + distances[v3 * n + v4];
    float new_length = distances[v1 * n + v3] + distances[v2 * n + v4];

    deltas[idx] = new_length - current_length;
    swap_pairs[idx * 2] = i;
    swap_pairs[idx * 2 + 1] = j;
}

/// Find minimum delta (best improvement) using parallel reduction
__global__ void find_min_delta(
    const float* deltas,              // Input: all deltas
    float* partial_mins,              // Output: partial minimums per block
    unsigned int* partial_indices,    // Output: indices of partial minimums
    unsigned int total_swaps          // Input: number of swaps
) {
    extern __shared__ float shared_data[];
    float* shared_min = shared_data;
    unsigned int* shared_idx = (unsigned int*)&shared_data[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    float local_min = 1e10f;  // Large positive number
    unsigned int local_idx = 0;
    if (idx < total_swaps) {
        local_min = deltas[idx];
        local_idx = idx;
    }
    shared_min[tid] = local_min;
    shared_idx[tid] = local_idx;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_min[tid + s] < shared_min[tid]) {
                shared_min[tid] = shared_min[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_mins[blockIdx.x] = shared_min[0];
        partial_indices[blockIdx.x] = shared_idx[0];
    }
}

/// Apply 2-opt swap to tour (reverse segment between i+1 and j)
__global__ void apply_2opt_swap(
    unsigned int* tour,               // Input/Output: tour to modify
    unsigned int i,                   // Input: first swap position
    unsigned int j,                   // Input: second swap position
    unsigned int n                    // Input: number of cities
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Number of elements to swap
    unsigned int swap_count = (j - i) / 2;

    if (idx >= swap_count) return;

    // Swap elements symmetrically
    unsigned int left = i + 1 + idx;
    unsigned int right = j - idx;

    unsigned int temp = tour[left];
    tour[left] = tour[right];
    tour[right] = temp;
}

} // extern "C"
