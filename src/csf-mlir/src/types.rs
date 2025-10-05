//! Type system for MLIR quantum dialect

use std::fmt;

/// Complex number type representation
#[derive(Clone, Debug, PartialEq)]
pub enum ComplexType {
    /// Standard complex<f64>
    Complex64,
    /// Double-double precision complex
    ComplexDD,
}

impl fmt::Display for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ComplexType::Complex64 => write!(f, "complex<f64>"),
            ComplexType::ComplexDD => write!(f, "complex<dd>"),
        }
    }
}

/// Tensor type representation
#[derive(Clone, Debug)]
pub struct TensorType {
    pub element_type: ComplexType,
    pub shape: Vec<usize>,
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "tensor<")?;
        for (i, dim) in self.shape.iter().enumerate() {
            if i > 0 {
                write!(f, "x")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "x{}>", self.element_type)
    }
}