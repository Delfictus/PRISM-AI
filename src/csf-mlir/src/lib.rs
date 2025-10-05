//! CSF-MLIR: Multi-Level Intermediate Representation Compiler
//!
//! This module provides JIT compilation and execution of quantum and neural
//! operations through MLIR, abstracting hardware-specific details.

pub mod runtime;
pub mod dialects;
pub mod passes;
pub mod types;

pub use runtime::{MlirJit, ExecutionResult};
pub use types::{ComplexType, TensorType};