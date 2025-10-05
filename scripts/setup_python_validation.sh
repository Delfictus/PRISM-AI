#!/bin/bash
# Python Validation Environment Setup for PRISM-AI
# Installs QuTiP, NumPy, SciPy, and other validation libraries

set -e

echo "=== PRISM-AI Python Validation Environment Setup ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Python version
check_python() {
    echo -e "${YELLOW}Checking Python installation...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 not found!${NC}"
        echo "Installing Python 3..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}Python $PYTHON_VERSION found${NC}"

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        echo "Installing pip..."
        sudo apt-get install -y python3-pip
    fi
}

# Create virtual environment
create_venv() {
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"

    VENV_DIR="$HOME/.prism-ai-venv"

    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment already exists at $VENV_DIR"
        read -p "Recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            return
        fi
    fi

    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}Virtual environment created at $VENV_DIR${NC}"

    # Activate and upgrade pip
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
}

# Install validation libraries
install_libraries() {
    echo -e "${YELLOW}Installing validation libraries...${NC}"

    source "$HOME/.prism-ai-venv/bin/activate"

    # Core scientific computing
    echo "Installing core scientific libraries..."
    pip install numpy==1.24.3 scipy==1.11.4 matplotlib==3.8.2

    # Quantum computing libraries
    echo "Installing quantum computing libraries..."
    pip install qutip==4.7.3
    pip install qiskit==0.45.1 qiskit-aer==0.13.1
    pip install pennylane==0.33.1

    # GPU acceleration
    echo "Installing GPU acceleration libraries..."
    pip install cupy-cuda12x  # CuPy for CUDA 12.x

    # Machine learning
    echo "Installing machine learning libraries..."
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    pip install jax[cuda12_pip]==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # Testing and validation
    echo "Installing testing tools..."
    pip install pytest==7.4.3 pytest-benchmark==4.0.0
    pip install hypothesis==6.92.1
    pip install memory_profiler==0.61.0
    pip install line_profiler==4.1.2

    # Jupyter for interactive validation
    echo "Installing Jupyter..."
    pip install jupyter==1.0.0 jupyterlab==4.0.9 ipywidgets==8.1.1

    # Performance profiling
    echo "Installing profiling tools..."
    pip install py-spy==0.3.14
    pip install scalene==1.5.38

    # Data handling
    echo "Installing data handling libraries..."
    pip install pandas==2.1.4 h5py==3.10.0

    echo -e "${GREEN}All libraries installed successfully${NC}"
}

# Create validation scripts
create_validation_scripts() {
    echo -e "${YELLOW}Creating validation scripts...${NC}"

    mkdir -p validation

    # QuTiP validation script
    cat << 'EOF' > validation/qutip_validator.py
#!/usr/bin/env python3
"""
QuTiP Validation Module for PRISM-AI
Validates quantum evolution against QuTiP reference implementation
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm
import ctypes
import os
from typing import Tuple, List
import time

class QuantumValidator:
    """Validates PRISM-AI quantum computations against QuTiP."""

    def __init__(self, rust_lib_path: str = None):
        """Initialize validator with Rust library."""
        if rust_lib_path is None:
            rust_lib_path = os.path.join(
                os.path.dirname(__file__),
                "../target/release/libprism_ai.so"
            )

        if os.path.exists(rust_lib_path):
            self.rust_lib = ctypes.CDLL(rust_lib_path)
            self.setup_rust_bindings()
        else:
            print(f"Warning: Rust library not found at {rust_lib_path}")
            self.rust_lib = None

    def setup_rust_bindings(self):
        """Setup ctypes bindings for Rust functions."""
        if self.rust_lib:
            # Define function signatures
            # evolve_quantum_state(H_ptr, psi_ptr, time, dim) -> psi_out_ptr
            self.rust_lib.evolve_quantum_state.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # H real
                ctypes.POINTER(ctypes.c_double),  # H imag
                ctypes.POINTER(ctypes.c_double),  # psi real
                ctypes.POINTER(ctypes.c_double),  # psi imag
                ctypes.c_double,                   # time
                ctypes.c_int                       # dimension
            ]
            self.rust_lib.evolve_quantum_state.restype = ctypes.c_int

    def create_test_hamiltonian(self, n: int, model: str = "tight-binding") -> np.ndarray:
        """Create test Hamiltonian for validation."""
        if model == "tight-binding":
            # Tight-binding model on a chain
            H = np.zeros((n, n), dtype=complex)
            for i in range(n - 1):
                H[i, i + 1] = -1.0
                H[i + 1, i] = -1.0
            # Periodic boundary
            H[0, n - 1] = -1.0
            H[n - 1, 0] = -1.0
        elif model == "ising":
            # Transverse field Ising model
            H = np.zeros((n, n), dtype=complex)
            # Add implementation
        else:
            raise ValueError(f"Unknown model: {model}")

        return H

    def validate_evolution(self, H: np.ndarray, psi0: np.ndarray,
                          time: float, tolerance: float = 1e-12) -> dict:
        """Validate quantum evolution against QuTiP."""
        # QuTiP reference calculation
        H_qt = qt.Qobj(H)
        psi0_qt = qt.Qobj(psi0)

        # Time evolution
        t_list = [0, time]
        result = qt.mesolve(H_qt, psi0_qt, t_list, [])
        psi_qt_final = result.states[-1].full().flatten()

        # Direct exponentiation (for small systems)
        if H.shape[0] <= 1024:
            U = expm(-1j * H * time)
            psi_expm_final = U @ psi0

        # Rust GPU result (if available)
        psi_rust_final = None
        if self.rust_lib:
            psi_rust_final = self.call_rust_evolution(H, psi0, time)

        # Calculate fidelities
        results = {
            "qutip_state": psi_qt_final,
            "expm_state": psi_expm_final if H.shape[0] <= 1024 else None,
            "rust_state": psi_rust_final,
        }

        if psi_rust_final is not None:
            fidelity = np.abs(np.vdot(psi_qt_final, psi_rust_final)) ** 2
            results["fidelity"] = fidelity
            results["passed"] = fidelity > (1 - tolerance)
            results["error"] = np.linalg.norm(psi_qt_final - psi_rust_final)

        return results

    def call_rust_evolution(self, H: np.ndarray, psi0: np.ndarray,
                           time: float) -> np.ndarray:
        """Call Rust GPU evolution function."""
        n = H.shape[0]

        # Separate real and imaginary parts
        H_real = np.ascontiguousarray(H.real.flatten(), dtype=np.float64)
        H_imag = np.ascontiguousarray(H.imag.flatten(), dtype=np.float64)
        psi_real = np.ascontiguousarray(psi0.real, dtype=np.float64)
        psi_imag = np.ascontiguousarray(psi0.imag, dtype=np.float64)

        # Output buffers
        psi_out_real = np.zeros(n, dtype=np.float64)
        psi_out_imag = np.zeros(n, dtype=np.float64)

        # Call Rust function
        ret = self.rust_lib.evolve_quantum_state(
            H_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            H_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            psi_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            psi_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(time),
            ctypes.c_int(n)
        )

        if ret == 0:
            return psi_out_real + 1j * psi_out_imag
        else:
            raise RuntimeError(f"Rust evolution failed with code {ret}")

    def benchmark_evolution(self, sizes: List[int]) -> dict:
        """Benchmark evolution for different system sizes."""
        results = {}

        for n in sizes:
            print(f"Benchmarking size {n}...")
            H = self.create_test_hamiltonian(n)
            psi0 = np.zeros(n, dtype=complex)
            psi0[0] = 1.0

            # QuTiP timing
            start = time.time()
            H_qt = qt.Qobj(H)
            psi0_qt = qt.Qobj(psi0)
            result = qt.mesolve(H_qt, psi0_qt, [0, 1.0], [])
            qutip_time = time.time() - start

            # Rust timing (if available)
            rust_time = None
            if self.rust_lib:
                start = time.time()
                self.call_rust_evolution(H, psi0, 1.0)
                rust_time = time.time() - start

            results[n] = {
                "qutip_time": qutip_time,
                "rust_time": rust_time,
                "speedup": qutip_time / rust_time if rust_time else None
            }

        return results


def run_validation_suite():
    """Run complete validation suite."""
    print("=== PRISM-AI Quantum Validation Suite ===\n")

    validator = QuantumValidator()

    # Test 1: Small system exact validation
    print("Test 1: Small system validation (n=10)")
    H = validator.create_test_hamiltonian(10)
    psi0 = np.zeros(10, dtype=complex)
    psi0[0] = 1.0

    result = validator.validate_evolution(H, psi0, 1.0)
    if "passed" in result:
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"  Fidelity: {result['fidelity']:.12f} - {status}")
        print(f"  Error: {result['error']:.2e}")

    # Test 2: Benchmark
    print("\nTest 2: Performance benchmark")
    sizes = [10, 50, 100, 500, 1000]
    benchmark_results = validator.benchmark_evolution(sizes)

    print("\nBenchmark Results:")
    print("-" * 50)
    print(f"{'Size':<10} {'QuTiP (s)':<15} {'Rust (s)':<15} {'Speedup':<10}")
    print("-" * 50)

    for size, times in benchmark_results.items():
        qutip_str = f"{times['qutip_time']:.4f}" if times['qutip_time'] else "N/A"
        rust_str = f"{times['rust_time']:.4f}" if times['rust_time'] else "N/A"
        speedup_str = f"{times['speedup']:.1f}x" if times['speedup'] else "N/A"
        print(f"{size:<10} {qutip_str:<15} {rust_str:<15} {speedup_str:<10}")

    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    run_validation_suite()
EOF

    chmod +x validation/qutip_validator.py

    echo -e "${GREEN}Validation scripts created${NC}"
}

# Setup Jupyter kernel
setup_jupyter() {
    echo -e "${YELLOW}Setting up Jupyter kernel...${NC}"

    source "$HOME/.prism-ai-venv/bin/activate"

    # Install kernel
    python -m ipykernel install --user --name prism-ai --display-name "PRISM-AI Validation"

    echo -e "${GREEN}Jupyter kernel 'prism-ai' installed${NC}"
}

# Create environment file for reproducibility
create_environment_file() {
    echo -e "${YELLOW}Creating environment file...${NC}"

    source "$HOME/.prism-ai-venv/bin/activate"

    pip freeze > requirements.txt

    # Create conda environment file
    cat << 'EOF' > environment.yml
name: prism-ai-validation
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.11
  - numpy=1.24.3
  - scipy=1.11.4
  - matplotlib=3.8.2
  - jupyter=1.0.0
  - pip
  - pip:
    - qutip==4.7.3
    - qiskit==0.45.1
    - cupy-cuda12x
    - torch==2.1.2
    - jax[cuda12_pip]==0.4.23
    - pytest==7.4.3
    - memory_profiler==0.61.0
EOF

    echo -e "${GREEN}Environment files created${NC}"
}

# Verify installation
verify_installation() {
    echo -e "${YELLOW}Verifying Python environment...${NC}"

    source "$HOME/.prism-ai-venv/bin/activate"

    # Test imports
    python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import scipy
    print(f"✓ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy: {e}")

try:
    import qutip
    print(f"✓ QuTiP {qutip.__version__}")
except ImportError as e:
    print(f"✗ QuTiP: {e}")

try:
    import qiskit
    print(f"✓ Qiskit {qiskit.__version__}")
except ImportError as e:
    print(f"✗ Qiskit: {e}")

try:
    import cupy
    print(f"✓ CuPy {cupy.__version__}")
    # Test GPU
    a = cupy.array([1, 2, 3])
    print(f"  GPU compute: {a.sum()}")
except Exception as e:
    print(f"✗ CuPy: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
EOF

    echo -e "${GREEN}Verification complete${NC}"
}

# Main execution
main() {
    echo "Starting Python validation environment setup..."
    echo

    check_python
    create_venv
    install_libraries
    create_validation_scripts
    setup_jupyter
    create_environment_file
    verify_installation

    echo
    echo -e "${GREEN}=== Python Validation Environment Ready ===${NC}"
    echo
    echo "To activate the environment:"
    echo "  source ~/.prism-ai-venv/bin/activate"
    echo
    echo "To run validation:"
    echo "  python validation/qutip_validator.py"
    echo
    echo "To start Jupyter:"
    echo "  jupyter lab"
    echo
    echo "Next steps:"
    echo "1. Build Rust project: cargo build --release --features validation"
    echo "2. Run full validation: ./scripts/validate_all.sh"
}

# Run if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi