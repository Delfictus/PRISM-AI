//! Quantum gate definitions

use num_complex::Complex64;
use ndarray::Array2;

/// Standard quantum gates
pub struct Gates;

impl Gates {
    /// Pauli-X (NOT) gate
    pub fn x() -> Array2<Complex64> {
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap()
    }

    /// Pauli-Y gate
    pub fn y() -> Array2<Complex64> {
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap()
    }

    /// Pauli-Z gate
    pub fn z() -> Array2<Complex64> {
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
            ],
        )
        .unwrap()
    }

    /// Hadamard gate
    pub fn h() -> Array2<Complex64> {
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(sqrt2_inv, 0.0), Complex64::new(sqrt2_inv, 0.0),
                Complex64::new(sqrt2_inv, 0.0), Complex64::new(-sqrt2_inv, 0.0),
            ],
        )
        .unwrap()
    }

    /// CNOT gate
    pub fn cnot() -> Array2<Complex64> {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap()
    }
}