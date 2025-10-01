// CUDA kernels for GPU-accelerated graph coloring

extern "C" {

/// Build adjacency matrix from coupling strengths (parallel)
/// PRODUCTION-GRADE: Uses byte-level atomic operations for correctness
__global__ void build_adjacency(
    const float* coupling,      // Input: n×n coupling matrix (flattened)
    float threshold,             // Input: coupling threshold
    unsigned char* adjacency,    // Output: packed adjacency matrix (bits)
    unsigned int n               // Input: number of vertices
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_edges = n * n;

    if (idx >= total_edges) return;

    unsigned int i = idx / n;
    unsigned int j = idx % n;

    // CRITICAL: Only add edges between different vertices (no self-loops)
    // and only if coupling strength meets threshold
    if (i != j && coupling[idx] >= threshold) {
        // Set bit in packed adjacency matrix
        // Use 32-bit word-aligned atomic operations for performance
        unsigned int bit_position = idx;
        unsigned int word_idx = bit_position / 32;
        unsigned int bit_in_word = bit_position % 32;

        // Cast to 32-bit word pointer and use atomic OR
        unsigned int* adjacency_words = (unsigned int*)adjacency;
        atomicOr(&adjacency_words[word_idx], (1U << bit_in_word));
    }
}

/// Count color conflicts in graph (parallel)
__global__ void count_conflicts(
    const unsigned char* adjacency,  // Input: packed adjacency matrix
    const unsigned int* coloring,     // Input: color assignment
    unsigned int* conflicts,          // Output: conflict count
    unsigned int n                    // Input: number of vertices
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_edges = n * n;

    if (idx >= total_edges) return;

    unsigned int i = idx / n;
    unsigned int j = idx % n;

    // Only count each edge once (i < j)
    if (i >= j) return;

    // Check if edge exists
    unsigned int byte_idx = idx / 8;
    unsigned int bit_idx = idx % 8;
    bool edge_exists = (adjacency[byte_idx] & (1 << bit_idx)) != 0;

    // If edge exists and both vertices have same color, it's a conflict
    if (edge_exists && coloring[i] == coloring[j]) {
        atomicAdd(conflicts, 1);
    }
}

/// Compute saturation degree for each vertex (parallel)
__global__ void compute_saturation(
    const unsigned char* adjacency,  // Input: packed adjacency matrix
    const unsigned int* coloring,     // Input: current color assignment
    unsigned int* saturation,         // Output: saturation degree per vertex
    unsigned int n,                   // Input: number of vertices
    unsigned int num_colors           // Input: number of colors
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= n) return;

    // Count distinct colors in neighborhood
    // Use shared memory for color set (bit vector)
    extern __shared__ unsigned int color_bits[];

    // Initialize
    if (threadIdx.x == 0) {
        for (unsigned int c = 0; c < (num_colors + 31) / 32; c++) {
            color_bits[c] = 0;
        }
    }
    __syncthreads();

    // Check all neighbors
    for (unsigned int u = 0; u < n; u++) {
        if (u == v) continue;

        // Check if edge (v, u) exists
        unsigned int idx = v * n + u;
        unsigned int byte_idx = idx / 8;
        unsigned int bit_idx = idx % 8;
        bool edge_exists = (adjacency[byte_idx] & (1 << bit_idx)) != 0;

        // If neighbor is colored, mark its color
        if (edge_exists && coloring[u] != 0xFFFFFFFF) {
            unsigned int color = coloring[u];
            unsigned int word = color / 32;
            unsigned int bit = color % 32;
            atomicOr(&color_bits[word], (1 << bit));
        }
    }

    __syncthreads();

    // Count number of distinct colors (popcount)
    unsigned int sat = 0;
    for (unsigned int c = 0; c < (num_colors + 31) / 32; c++) {
        sat += __popc(color_bits[c]);
    }

    saturation[v] = sat;
}

/// Compute vertex degree (number of neighbors) - parallel
__global__ void compute_degree(
    const unsigned char* adjacency,  // Input: packed adjacency matrix
    unsigned int* degree,             // Output: degree per vertex
    unsigned int n                    // Input: number of vertices
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= n) return;

    unsigned int deg = 0;

    // Count neighbors
    for (unsigned int u = 0; u < n; u++) {
        if (u == v) continue;

        unsigned int idx = v * n + u;
        unsigned int byte_idx = idx / 8;
        unsigned int bit_idx = idx % 8;

        if ((adjacency[byte_idx] & (1 << bit_idx)) != 0) {
            deg++;
        }
    }

    degree[v] = deg;
}

/// Find maximum degree vertex (reduction)
__global__ void find_max_degree(
    const unsigned int* degree,    // Input: degree per vertex
    unsigned int* max_degree,      // Output: maximum degree
    unsigned int* max_vertex,      // Output: vertex with max degree
    unsigned int n                 // Input: number of vertices
) {
    extern __shared__ unsigned int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    if (i < n) {
        sdata[tid * 2] = degree[i];
        sdata[tid * 2 + 1] = i;
    } else {
        sdata[tid * 2] = 0;
        sdata[tid * 2 + 1] = 0;
    }
    __syncthreads();

    // Reduction to find maximum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[(tid + s) * 2] > sdata[tid * 2]) {
                sdata[tid * 2] = sdata[(tid + s) * 2];
                sdata[tid * 2 + 1] = sdata[(tid + s) * 2 + 1];
            }
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicMax(max_degree, sdata[0]);
        if (sdata[0] == *max_degree) {
            *max_vertex = sdata[1];
        }
    }
}

} // extern "C"
