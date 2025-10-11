/// Web Platform Backend
///
/// Provides WebSocket server and API endpoints for the PRISM-AI
/// interactive web dashboards (4 dashboards total)
///
/// Week 3 Enhancements:
/// - Plugin Architecture: Hot-reloadable data source plugins
/// - Event Sourcing: Complete audit trail and replay capability
/// - SGP4 Orbital Mechanics: Physics-based satellite tracking
/// - Transfer Entropy: Shannon entropy-based causality detection
/// - NVML GPU Metrics: Real GPU monitoring via NVIDIA Management Library
/// - MessagePack: Binary WebSocket protocol for efficient data transmission

pub mod metrics_api;
pub mod websocket;
pub mod server;
pub mod types;
pub mod validation;

// WebSocket actors for each dashboard
pub mod pwsa_websocket;
pub mod telecom_websocket;
pub mod hft_websocket;

// Week 3 Enhancement Modules
pub mod plugin;              // Actor-based plugin architecture
pub mod event_sourcing;      // Event sourcing with CQRS
pub mod orbital;             // SGP4 orbital mechanics
pub mod transfer_entropy;    // Shannon entropy calculator
pub mod gpu_metrics;         // NVML GPU monitoring
pub mod messagepack;         // MessagePack compression
pub mod prism_bridge;        // PRISM-AI core integration bridge (Task 3.1.1)

// Tests module
#[cfg(test)]
mod tests;

pub use metrics_api::MetricsApi;
pub use websocket::MetricsWebSocket;
pub use server::start_web_server;
pub use types::*;
pub use validation::*;

// Re-export WebSocket actors
pub use pwsa_websocket::PwsaWebSocket;
pub use telecom_websocket::TelecomWebSocket;
pub use hft_websocket::HftWebSocket;

// Re-export enhancement modules
pub use plugin::{PluginManager, PrismPlugin};
pub use event_sourcing::{EventStore, DomainEvent};
pub use orbital::{SGP4Propagator, SatelliteTracker, TLE};
pub use transfer_entropy::{TransferEntropyCalculator, TimeSeries};
pub use gpu_metrics::{GpuMetricsCollector, GpuMetrics};
pub use messagepack::{MessagePackCodec, MessagePackWebSocket};
pub use prism_bridge::{PrismBridge, DefaultPrismBridge, SystemState, ColoringResult};
