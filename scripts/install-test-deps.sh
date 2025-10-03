#!/bin/bash
# Install Testing Dependencies for Active Inference Platform
# Constitution: Phase 2, Task 2.1 - Testing Requirements
#
# This script installs BLAS/LAPACK libraries required for running tests

set -e

echo "================================"
echo "Installing Testing Dependencies"
echo "================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script requires sudo access."
    echo "You will be prompted for your password."
    echo ""
fi

# Update package lists
echo "[1/4] Updating package lists..."
sudo apt-get update -qq

# Install BLAS development libraries
echo "[2/4] Installing BLAS (Basic Linear Algebra Subprograms)..."
sudo apt-get install -y libblas-dev

# Install LAPACK development libraries
echo "[3/4] Installing LAPACK (Linear Algebra PACKage)..."
sudo apt-get install -y liblapack-dev

# Install Fortran compiler (required by some LAPACK operations)
echo "[4/4] Installing GNU Fortran compiler..."
sudo apt-get install -y gfortran

echo ""
echo "================================"
echo "Installation Complete!"
echo "================================"
echo ""

# Verify installation
echo "Verifying installation..."
if dpkg -l | grep -q "libblas-dev"; then
    echo "✅ libblas-dev: Installed"
else
    echo "❌ libblas-dev: NOT installed"
fi

if dpkg -l | grep -q "liblapack-dev"; then
    echo "✅ liblapack-dev: Installed"
else
    echo "❌ liblapack-dev: NOT installed"
fi

if dpkg -l | grep -q "gfortran"; then
    echo "✅ gfortran: Installed"
else
    echo "❌ gfortran: NOT installed"
fi

echo ""
echo "Library paths:"
ldconfig -p | grep -E "blas|lapack" | head -5

echo ""
echo "================================"
echo "Next Steps"
echo "================================"
echo ""
echo "Run tests with:"
echo "  cargo test --lib active_inference"
echo ""
echo "Or run all tests:"
echo "  cargo test"
echo ""
echo "See TESTING_REQUIREMENTS.md for more details."
echo ""
