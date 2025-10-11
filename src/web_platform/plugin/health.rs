/// Health monitoring for plugins
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthLevel {
    /// Plugin is functioning normally
    Healthy,
    /// Plugin has minor issues but is operational
    Degraded,
    /// Plugin has critical issues
    Unhealthy,
    /// Plugin status unknown
    Unknown,
}

/// Detailed health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub level: HealthLevel,
    pub message: String,
    pub last_check: u64,
    pub metrics: HealthMetrics,
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// Total data generations
    pub total_generations: u64,
    /// Failed data generations
    pub failed_generations: u64,
    /// Average generation time (microseconds)
    pub avg_generation_time_us: u64,
    /// Last successful generation timestamp
    pub last_success_ts: u64,
    /// Consecutive failures
    pub consecutive_failures: u32,
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self {
            total_generations: 0,
            failed_generations: 0,
            avg_generation_time_us: 0,
            last_success_ts: 0,
            consecutive_failures: 0,
        }
    }
}

impl HealthStatus {
    /// Create a healthy status
    pub fn healthy(message: impl Into<String>) -> Self {
        Self {
            level: HealthLevel::Healthy,
            message: message.into(),
            last_check: current_timestamp(),
            metrics: HealthMetrics::default(),
        }
    }

    /// Create a degraded status
    pub fn degraded(message: impl Into<String>) -> Self {
        Self {
            level: HealthLevel::Degraded,
            message: message.into(),
            last_check: current_timestamp(),
            metrics: HealthMetrics::default(),
        }
    }

    /// Create an unhealthy status
    pub fn unhealthy(message: impl Into<String>) -> Self {
        Self {
            level: HealthLevel::Unhealthy,
            message: message.into(),
            last_check: current_timestamp(),
            metrics: HealthMetrics::default(),
        }
    }

    /// Create an unknown status
    pub fn unknown(message: impl Into<String>) -> Self {
        Self {
            level: HealthLevel::Unknown,
            message: message.into(),
            last_check: current_timestamp(),
            metrics: HealthMetrics::default(),
        }
    }

    /// Update health status with metrics
    pub fn with_metrics(mut self, metrics: HealthMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Check if status is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.level, HealthLevel::Healthy | HealthLevel::Degraded)
    }

    /// Get failure rate (0.0 to 1.0)
    pub fn failure_rate(&self) -> f64 {
        if self.metrics.total_generations == 0 {
            return 0.0;
        }
        self.metrics.failed_generations as f64 / self.metrics.total_generations as f64
    }
}

/// Health check trait for components that can be monitored
pub trait HealthCheck {
    /// Perform health check
    fn health_check(&self) -> HealthStatus;
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_creation() {
        let status = HealthStatus::healthy("All systems operational");
        assert_eq!(status.level, HealthLevel::Healthy);
        assert!(status.is_healthy());
    }

    #[test]
    fn test_failure_rate_calculation() {
        let mut status = HealthStatus::healthy("Test");
        status.metrics.total_generations = 100;
        status.metrics.failed_generations = 10;
        assert_eq!(status.failure_rate(), 0.1);
    }

    #[test]
    fn test_unhealthy_status() {
        let status = HealthStatus::unhealthy("System failure");
        assert_eq!(status.level, HealthLevel::Unhealthy);
        assert!(!status.is_healthy());
    }
}
