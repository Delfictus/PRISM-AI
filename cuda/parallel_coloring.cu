// CUDA kernel for parallel graph coloring (Jones-Plassmann algorithm)

extern "C" {

/// Initialize random priorities for each vertex
__global__ void init_priorities(
    float* priorities,          // Output: random priority per vertex
    unsigned int* colors,       // Output: color assignment (init to UNCOLORED)
    unsigned int n,             // Input: number of vertices
    unsigned long long seed     // Input: random seed
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    // Simple LCG random number generator
    unsigned long long state = seed + v * 1103515245ULL;
    state = (state * 1103515245ULL + 12345ULL) & 0x7FFFFFFF;

    priorities[v] = (float)state / (float)0x7FFFFFFF;
    colors[v] = 0xFFFFFFFF; // UNCOLORED
}

/// Check if vertex can be colored (has highest priority among uncolored neighbors)
__global__ void find_independent_set(
    const unsigned char* adjacency,  // Input: packed adjacency matrix
    const float* priorities,         // Input: vertex priorities
    const unsigned int* colors,      // Input: current coloring
    unsigned int* can_color,         // Output: 1 if vertex can be colored
    unsigned int n                   // Input: number of vertices
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    // Already colored? Skip
    if (colors[v] != 0xFFFFFFFF) {
        can_color[v] = 0;
        return;
    }

    float my_priority = priorities[v];
    bool has_highest = true;

    // Check all neighbors
    for (unsigned int u = 0; u < n; u++) {
        if (u == v) continue;

        // Check if edge exists
        unsigned int idx = v * n + u;
        unsigned int byte_idx = idx / 8;
        unsigned int bit_idx = idx % 8;
        bool is_neighbor = (adjacency[byte_idx] & (1 << bit_idx)) != 0;

        if (is_neighbor && colors[u] == 0xFFFFFFFF) {
            // Neighbor is uncolored - check priority
            if (priorities[u] > my_priority ||
                (priorities[u] == my_priority && u > v)) {
                has_highest = false;
                break;
            }
        }
    }

    can_color[v] = has_highest ? 1 : 0;
}

/// Color vertices in independent set with smallest available color
__global__ void color_independent_set(
    const unsigned char* adjacency,  // Input: packed adjacency matrix
    const unsigned int* can_color,   // Input: which vertices to color
    unsigned int* colors,            // Output: color assignment
    unsigned int n,                  // Input: number of vertices
    unsigned int max_colors          // Input: maximum colors to try
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    // Only color if we're in the independent set
    if (can_color[v] == 0) return;

    // Find smallest available color
    // Use shared memory for used colors (bit vector)
    extern __shared__ unsigned int used_colors[];

    // Initialize shared memory
    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < (max_colors + 31) / 32; i++) {
            used_colors[i] = 0;
        }
    }
    __syncthreads();

    // Mark colors used by neighbors
    for (unsigned int u = 0; u < n; u++) {
        if (u == v) continue;

        // Check if edge exists
        unsigned int idx = v * n + u;
        unsigned int byte_idx = idx / 8;
        unsigned int bit_idx = idx % 8;
        bool is_neighbor = (adjacency[byte_idx] & (1 << bit_idx)) != 0;

        if (is_neighbor && colors[u] != 0xFFFFFFFF) {
            unsigned int neighbor_color = colors[u];
            if (neighbor_color < max_colors) {
                unsigned int word = neighbor_color / 32;
                unsigned int bit = neighbor_color % 32;
                atomicOr(&used_colors[word], (1U << bit));
            }
        }
    }

    __syncthreads();

    // Find first available color
    for (unsigned int c = 0; c < max_colors; c++) {
        unsigned int word = c / 32;
        unsigned int bit = c % 32;
        if ((used_colors[word] & (1U << bit)) == 0) {
            colors[v] = c;
            return;
        }
    }

    // If we get here, couldn't find a color (shouldn't happen if max_colors is sufficient)
    colors[v] = max_colors; // Use overflow color
}

/// Count how many vertices are still uncolored
__global__ void count_uncolored(
    const unsigned int* colors,      // Input: color assignment
    unsigned int* uncolored_count,   // Output: number of uncolored vertices
    unsigned int n                   // Input: number of vertices
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    if (colors[v] == 0xFFFFFFFF) {
        atomicAdd(uncolored_count, 1);
    }
}

/// Verify coloring has no conflicts
__global__ void verify_coloring(
    const unsigned char* adjacency,  // Input: packed adjacency matrix
    const unsigned int* colors,      // Input: color assignment
    unsigned int* conflicts,         // Output: number of conflicts
    unsigned int n                   // Input: number of vertices
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_edges = n * n;
    if (idx >= total_edges) return;

    unsigned int i = idx / n;
    unsigned int j = idx % n;

    // Only check each edge once (i < j)
    if (i >= j) return;

    // Check if edge exists
    unsigned int byte_idx = idx / 8;
    unsigned int bit_idx = idx % 8;
    bool edge_exists = (adjacency[byte_idx] & (1 << bit_idx)) != 0;

    // If edge exists and both vertices have same color, it's a conflict
    if (edge_exists && colors[i] != 0xFFFFFFFF && colors[j] != 0xFFFFFFFF &&
        colors[i] == colors[j]) {
        atomicAdd(conflicts, 1);
    }
}

} // extern "C"
