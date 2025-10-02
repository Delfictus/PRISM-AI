// CUDA kernels for GPU-accelerated quantum Hamiltonian evolution
// Matrix operations, state evolution, phase extraction

#include <cuComplex.h>

extern "C" {

/// Build kinetic energy operator using finite difference stencil
/// T = -∇²/2m with 9-point stencil (O(h⁸) accuracy)
__global__ void build_kinetic_operator(
    cuDoubleComplex* kinetic_matrix,  // Output: n×n kinetic matrix
    const double* masses,              // Input: atomic masses
    unsigned int n_atoms,
    double grid_spacing,
    double hartree_to_kcalmol
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dim = n_atoms * 3;

    if (row >= dim || col >= dim) return;

    // 9-point stencil coefficients
    const double STENCIL[5] = {
        -205.0/72.0,   // Central
        8.0/5.0,       // ±1
        -1.0/5.0,      // ±2
        8.0/315.0,     // ±3
        -1.0/560.0     // ±4
    };

    unsigned int atom_idx = row / 3;
    double mass = masses[atom_idx];
    double prefactor = 1.0 / (2.0 * mass * grid_spacing * grid_spacing);

    cuDoubleComplex value = make_cuDoubleComplex(0.0, 0.0);

    // Diagonal element
    if (row == col) {
        value.x = -prefactor * STENCIL[0] * hartree_to_kcalmol;
    }
    // Off-diagonal elements (stencil points)
    else {
        int offset = abs((int)col - (int)row);
        if (offset >= 1 && offset <= 4) {
            value.x = prefactor * STENCIL[offset] * hartree_to_kcalmol;
        }
    }

    kinetic_matrix[row * dim + col] = value;
}

/// Build potential energy operator (Lennard-Jones + Coulomb)
__global__ void build_potential_operator(
    cuDoubleComplex* potential_matrix,  // Output: n×n potential matrix
    const double* positions,             // Input: atomic positions (n_atoms × 3)
    unsigned int n_atoms,
    double lj_epsilon,
    double lj_sigma,
    double coulomb_cutoff,
    double kcalmol_to_hartree
) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_atoms || j >= n_atoms || i == j) return;

    // Calculate distance
    double dx = positions[i * 3 + 0] - positions[j * 3 + 0];
    double dy = positions[i * 3 + 1] - positions[j * 3 + 1];
    double dz = positions[i * 3 + 2] - positions[j * 3 + 2];
    double r = sqrt(dx*dx + dy*dy + dz*dz);

    if (r > coulomb_cutoff) return;

    // Lennard-Jones potential: 4ε[(σ/r)¹² - (σ/r)⁶]
    double sigma_over_r = lj_sigma / r;
    double sr6 = sigma_over_r * sigma_over_r * sigma_over_r;
    sr6 = sr6 * sr6;
    double sr12 = sr6 * sr6;
    double lj_energy = 4.0 * lj_epsilon * (sr12 - sr6);

    // Convert to Hartree
    double energy_hartree = lj_energy * kcalmol_to_hartree;

    // Add to diagonal blocks (3×3 for each atom pair)
    for (int dim = 0; dim < 3; dim++) {
        unsigned int idx = (i * 3 + dim) * (n_atoms * 3) + (j * 3 + dim);
        potential_matrix[idx].x += energy_hartree;
    }
}

/// Matrix-vector multiplication for Hamiltonian application: y = H·x
__global__ void hamiltonian_matvec(
    const cuDoubleComplex* hamiltonian,  // Input: H matrix (n×n)
    const cuDoubleComplex* state_vec,    // Input: state vector (n)
    cuDoubleComplex* result_vec,         // Output: H·x (n)
    unsigned int n
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    for (unsigned int col = 0; col < n; col++) {
        cuDoubleComplex h_elem = hamiltonian[row * n + col];
        cuDoubleComplex x_elem = state_vec[col];

        // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        sum.x += h_elem.x * x_elem.x - h_elem.y * x_elem.y;
        sum.y += h_elem.x * x_elem.y + h_elem.y * x_elem.x;
    }

    result_vec[row] = sum;
}

/// Runge-Kutta 4th order integrator for quantum state evolution
/// Solves: iℏ dψ/dt = H·ψ
__global__ void rk4_step(
    const cuDoubleComplex* hamiltonian,  // Input: H matrix
    const cuDoubleComplex* state,        // Input: current state
    cuDoubleComplex* new_state,          // Output: evolved state
    cuDoubleComplex* k1,                 // Workspace
    cuDoubleComplex* k2,                 // Workspace
    cuDoubleComplex* k3,                 // Workspace
    cuDoubleComplex* k4,                 // Workspace
    unsigned int n,
    double dt,
    double hbar
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // RK4 coefficients are computed via multiple kernel launches
    // This kernel performs final combination: ψ(t+dt) = ψ(t) + dt/6·(k1 + 2k2 + 2k3 + k4)

    double factor = dt / (6.0 * hbar);

    // Combine RK4 stages
    cuDoubleComplex increment;
    increment.x = factor * (k1[idx].x + 2.0*k2[idx].x + 2.0*k3[idx].x + k4[idx].x);
    increment.y = factor * (k1[idx].y + 2.0*k2[idx].y + 2.0*k3[idx].y + k4[idx].y);

    // Note: Schrödinger equation has -i factor
    new_state[idx].x = state[idx].x + increment.y;  // Real part gets imaginary increment
    new_state[idx].y = state[idx].y - increment.x;  // Imaginary part gets real increment (with sign flip)
}

/// Extract phases from quantum state
/// Phase(k) = atan2(Im(ψ_k), Re(ψ_k))
__global__ void extract_phases(
    const cuDoubleComplex* state,  // Input: quantum state
    double* phases,                 // Output: phase angles
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    phases[idx] = atan2(state[idx].y, state[idx].x);
}

/// Compute phase coherence matrix
/// C_ij = |⟨ψ_i|ψ_j⟩|²
__global__ void compute_phase_coherence(
    const cuDoubleComplex* state,  // Input: quantum state
    double* coherence_matrix,       // Output: coherence matrix (n×n)
    unsigned int n
) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    cuDoubleComplex psi_i = state[i];
    cuDoubleComplex psi_j = state[j];

    // Inner product: ⟨ψ_i|ψ_j⟩ = ψ_i* · ψ_j
    double real_part = psi_i.x * psi_j.x + psi_i.y * psi_j.y;
    double imag_part = psi_i.x * psi_j.y - psi_i.y * psi_j.x;

    // |⟨ψ_i|ψ_j⟩|²
    coherence_matrix[i * n + j] = real_part * real_part + imag_part * imag_part;
}

/// Compute quantum order parameter (phase coherence)
/// r = |⟨e^(iθ)⟩| = |Σ_k e^(iθ_k)| / n
__global__ void compute_order_parameter(
    const double* phases,   // Input: phase angles
    double* order_param,    // Output: order parameter
    unsigned int n
) {
    __shared__ double shared_cos[256];
    __shared__ double shared_sin[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load phases and compute sin/cos
    if (idx < n) {
        shared_cos[tid] = cos(phases[idx]);
        shared_sin[tid] = sin(phases[idx]);
    } else {
        shared_cos[tid] = 0.0;
        shared_sin[tid] = 0.0;
    }
    __syncthreads();

    // Parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_cos[tid] += shared_cos[tid + s];
            shared_sin[tid] += shared_sin[tid + s];
        }
        __syncthreads();
    }

    // Compute magnitude
    if (tid == 0) {
        double sum_cos = shared_cos[0] / n;
        double sum_sin = shared_sin[0] / n;
        atomicAdd(order_param, sqrt(sum_cos*sum_cos + sum_sin*sum_sin));
    }
}

} // extern "C"
