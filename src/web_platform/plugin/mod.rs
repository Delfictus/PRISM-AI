/// Plugin Architecture for PRISM-AI Web Platform
///
/// Provides hot-reloadable, actor-based plugin system for data sources
/// Week 3 Enhancement: Actor-Based Plugin Architecture

pub mod types;
pub mod manager;
pub mod event_bus;
pub mod health;

// Built-in plugins
pub mod pwsa_plugin;
pub mod telecom_plugin;
pub mod hft_plugin;
pub mod metrics_plugin;

pub use types::*;
pub use manager::PluginManager;
pub use event_bus::EventBus;
pub use health::{HealthStatus, HealthCheck};

// Re-export built-in plugins
pub use pwsa_plugin::PwsaPlugin;
pub use telecom_plugin::TelecomPlugin;
pub use hft_plugin::HftPlugin;
pub use metrics_plugin::MetricsPlugin;
