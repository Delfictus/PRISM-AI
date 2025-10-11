/// Event Sourcing Architecture for PRISM-AI Web Platform
///
/// Provides complete audit trail and replay capability through event-based state management
/// Week 3 Enhancement: Event Sourcing Architecture

pub mod types;
pub mod store;
pub mod aggregate;
pub mod projections;

pub use types::*;
pub use store::EventStore;
pub use aggregate::{Aggregate, AggregateRoot};
pub use projections::*;
