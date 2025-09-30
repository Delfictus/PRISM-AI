//! Platform Foundation
//!
//! Unified API for the world's first software-based neuromorphic-quantum computing platform

pub mod platform;
pub mod types;

// Re-export main components
pub use platform::NeuromorphicQuantumPlatform;
pub use types::*;