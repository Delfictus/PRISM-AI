//! Integration Module
//!
//! Constitution: Phase 3 - Integration Architecture
//!
//! Couples neuromorphic and quantum domains through information-theoretic
//! principles without physical simulation. Implements:
//!
//! 1. **Mutual Information Maximization**: I(X;Y) maximization for coupling
//! 2. **Information Bottleneck**: Compress while preserving task-relevant info
//! 3. **Causal Consistency**: Maintain cause-effect relationships
//! 4. **Phase Synchronization**: Coordinate oscillatory dynamics
//!
//! Mathematical Foundation:
//! ```text
//! L = I(X;Y) - β·I(X;Z)  // Information bottleneck
//! I(X;Y) = H(X) + H(Y) - H(X,Y)  // Mutual information
//! ρ = |⟨e^{iθ}⟩|  // Kuramoto order parameter (phase coherence)
//! ```

pub mod cross_domain_bridge;
pub mod information_channel;
pub mod synchronization;

pub use cross_domain_bridge::{
    CrossDomainBridge,
    DomainState,
    CouplingStrength,
    BridgeMetrics,
};

pub use information_channel::{
    InformationChannel,
    ChannelState,
    TransferResult,
};

pub use synchronization::{
    PhaseSynchronizer,
    SynchronizationMetrics,
    CoherenceLevel,
};
