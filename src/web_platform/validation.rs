/// JSON Schema Validation for Web Platform Types
///
/// Ensures type safety and correct serialization/deserialization
/// across the Rust ↔ TypeScript boundary
///
/// Task 2.1.3: JSON schema validation

use serde_json::{json, Value};
use crate::web_platform::types::*;

/// Validation result with detailed error reporting
#[derive(Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn success() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            valid: false,
            errors: vec![message],
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, message: String) {
        self.valid = false;
        self.errors.push(message);
    }

    pub fn add_warning(&mut self, message: String) {
        self.warnings.push(message);
    }
}

/// Validate PwsaTelemetry structure
pub fn validate_pwsa_telemetry(telemetry: &PwsaTelemetry) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Validate timestamp
    if telemetry.timestamp <= 0 {
        result.add_error("Timestamp must be positive".to_string());
    }

    // Validate transport layer
    if telemetry.transport_layer.satellites.is_empty() {
        result.add_error("Transport layer must have at least one satellite".to_string());
    }

    for sat in &telemetry.transport_layer.satellites {
        if sat.lat < -90.0 || sat.lat > 90.0 {
            result.add_error(format!("Satellite {} latitude out of range: {}", sat.id, sat.lat));
        }
        if sat.lon < -180.0 || sat.lon > 180.0 {
            result.add_error(format!("Satellite {} longitude out of range: {}", sat.id, sat.lon));
        }
        if sat.altitude <= 0.0 {
            result.add_error(format!("Satellite {} altitude must be positive", sat.id));
        }
        if !["transport", "tracking"].contains(&sat.layer.as_str()) {
            result.add_error(format!("Satellite {} invalid layer: {}", sat.id, sat.layer));
        }
        if !["healthy", "degraded", "failed"].contains(&sat.status.as_str()) {
            result.add_error(format!("Satellite {} invalid status: {}", sat.id, sat.status));
        }
    }

    // Validate link quality
    if telemetry.transport_layer.link_quality < 0.0 || telemetry.transport_layer.link_quality > 1.0 {
        result.add_error("Link quality must be between 0 and 1".to_string());
    }

    // Validate constellation health
    if telemetry.transport_layer.constellation_health < 0.0 || telemetry.transport_layer.constellation_health > 1.0 {
        result.add_error("Constellation health must be between 0 and 1".to_string());
    }

    // Validate tracking layer
    for threat in &telemetry.tracking_layer.threats {
        if threat.probability < 0.0 || threat.probability > 1.0 {
            result.add_error(format!("Threat {} probability out of range: {}", threat.id, threat.probability));
        }
        if threat.confidence < 0.0 || threat.confidence > 1.0 {
            result.add_error(format!("Threat {} confidence out of range: {}", threat.id, threat.confidence));
        }
    }

    // Validate mission awareness
    if telemetry.mission_awareness.transport_health < 0.0 || telemetry.mission_awareness.transport_health > 1.0 {
        result.add_error("Transport health must be between 0 and 1".to_string());
    }

    if telemetry.mission_awareness.tracking_effectiveness < 0.0 || telemetry.mission_awareness.tracking_effectiveness > 1.0 {
        result.add_error("Tracking effectiveness must be between 0 and 1".to_string());
    }

    result
}

/// Validate TelecomUpdate structure
pub fn validate_telecom_update(update: &TelecomUpdate) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Validate timestamp
    if update.timestamp <= 0 {
        result.add_error("Timestamp must be positive".to_string());
    }

    // Validate network topology
    if update.network_topology.nodes.is_empty() {
        result.add_error("Network must have at least one node".to_string());
    }

    for node in &update.network_topology.nodes {
        if node.load < 0.0 || node.load > 1.0 {
            result.add_error(format!("Node {} load out of range: {}", node.id, node.load));
        }
        if node.capacity <= 0.0 {
            result.add_warning(format!("Node {} has zero or negative capacity", node.id));
        }
        if !["router", "switch", "endpoint", "hub"].contains(&node.node_type.as_str()) {
            result.add_error(format!("Node {} invalid type: {}", node.id, node.node_type));
        }
    }

    for edge in &update.network_topology.edges {
        if edge.utilization < 0.0 || edge.utilization > 1.0 {
            result.add_error(format!("Edge {} utilization out of range: {}", edge.id, edge.utilization));
        }
        if edge.packet_loss < 0.0 || edge.packet_loss > 1.0 {
            result.add_error(format!("Edge {} packet loss out of range: {}", edge.id, edge.packet_loss));
        }
        if edge.bandwidth <= 0.0 {
            result.add_warning(format!("Edge {} has zero or negative bandwidth", edge.id));
        }
    }

    // Validate optimization state
    if update.optimization_state.current_coloring == 0 {
        result.add_error("Current coloring cannot be zero".to_string());
    }

    if update.optimization_state.best_coloring > update.optimization_state.current_coloring {
        result.add_error("Best coloring cannot exceed current coloring".to_string());
    }

    if update.optimization_state.convergence < 0.0 || update.optimization_state.convergence > 1.0 {
        result.add_error("Convergence must be between 0 and 1".to_string());
    }

    // Validate network performance
    if update.performance.packet_loss_rate < 0.0 || update.performance.packet_loss_rate > 1.0 {
        result.add_error("Packet loss rate must be between 0 and 1".to_string());
    }

    if update.performance.uptime < 0.0 || update.performance.uptime > 1.0 {
        result.add_error("Uptime must be between 0 and 1".to_string());
    }

    if update.performance.quality_of_service < 0.0 || update.performance.quality_of_service > 1.0 {
        result.add_error("QoS must be between 0 and 1".to_string());
    }

    result
}

/// Validate MarketUpdate structure
pub fn validate_market_update(update: &MarketUpdate) -> ValidationResult {
    let mut result = ValidationResult::success();

    // Validate timestamp
    if update.timestamp <= 0 {
        result.add_error("Timestamp must be positive".to_string());
    }

    // Validate prices
    if update.prices.is_empty() {
        result.add_error("Must have at least one price".to_string());
    }

    for price in &update.prices {
        if price.price <= 0.0 {
            result.add_error(format!("Symbol {} has non-positive price: {}", price.symbol, price.price));
        }
        if price.bid <= 0.0 || price.ask <= 0.0 {
            result.add_error(format!("Symbol {} has invalid bid/ask", price.symbol));
        }
        if price.bid >= price.ask {
            result.add_warning(format!("Symbol {} bid >= ask (crossed market)", price.symbol));
        }
        if price.volume < 0.0 {
            result.add_error(format!("Symbol {} has negative volume", price.symbol));
        }
    }

    // Validate order book
    if update.order_book.bids.is_empty() || update.order_book.asks.is_empty() {
        result.add_error("Order book must have bids and asks".to_string());
    }

    // Check bid/ask ordering
    for i in 1..update.order_book.bids.len() {
        if update.order_book.bids[i].price > update.order_book.bids[i - 1].price {
            result.add_error("Bids must be in descending price order".to_string());
            break;
        }
    }

    for i in 1..update.order_book.asks.len() {
        if update.order_book.asks[i].price < update.order_book.asks[i - 1].price {
            result.add_error("Asks must be in ascending price order".to_string());
            break;
        }
    }

    // Validate trading signals
    if update.signals.confidence < 0.0 || update.signals.confidence > 1.0 {
        result.add_error("Signal confidence must be between 0 and 1".to_string());
    }

    if update.signals.signal_strength < 0.0 || update.signals.signal_strength > 1.0 {
        result.add_error("Signal strength must be between 0 and 1".to_string());
    }

    if update.signals.risk_score < 0.0 || update.signals.risk_score > 1.0 {
        result.add_error("Risk score must be between 0 and 1".to_string());
    }

    if !["up", "down", "neutral"].contains(&update.signals.predicted_direction.as_str()) {
        result.add_error(format!("Invalid predicted direction: {}", update.signals.predicted_direction));
    }

    if !["buy", "sell", "hold"].contains(&update.signals.recommended_action.as_str()) {
        result.add_error(format!("Invalid recommended action: {}", update.signals.recommended_action));
    }

    // Validate transfer entropy signal
    if update.signals.transfer_entropy.significance < 0.0 || update.signals.transfer_entropy.significance > 1.0 {
        result.add_error("TE significance (p-value) must be between 0 and 1".to_string());
    }

    if update.signals.transfer_entropy.causal_strength < 0.0 || update.signals.transfer_entropy.causal_strength > 1.0 {
        result.add_error("TE causal strength must be between 0 and 1".to_string());
    }

    // Validate execution metrics
    if update.execution.fill_rate < 0.0 || update.execution.fill_rate > 1.0 {
        result.add_error("Fill rate must be between 0 and 1".to_string());
    }

    if update.execution.reject_rate < 0.0 || update.execution.reject_rate > 1.0 {
        result.add_error("Reject rate must be between 0 and 1".to_string());
    }

    // Validate portfolio
    if update.portfolio.total_value <= 0.0 {
        result.add_warning("Portfolio has zero or negative total value".to_string());
    }

    if update.portfolio.win_rate < 0.0 || update.portfolio.win_rate > 1.0 {
        result.add_error("Win rate must be between 0 and 1".to_string());
    }

    result
}

/// Test serialization roundtrip (Rust → JSON → Rust)
pub fn test_serialization_roundtrip<T>(data: &T) -> Result<(), String>
where
    T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug,
{
    // Serialize to JSON
    let json_str = serde_json::to_string(data)
        .map_err(|e| format!("Serialization failed: {}", e))?;

    // Deserialize back
    let _deserialized: T = serde_json::from_str(&json_str)
        .map_err(|e| format!("Deserialization failed: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pwsa_validation_success() {
        let telemetry = PwsaTelemetry {
            timestamp: 1234567890,
            transport_layer: TransportLayer {
                satellites: vec![SatelliteState {
                    id: 1,
                    lat: 45.0,
                    lon: -120.0,
                    altitude: 550.0,
                    layer: "transport".to_string(),
                    status: "healthy".to_string(),
                    velocity: 7.5,
                    heading: 90.0,
                }],
                link_quality: 0.95,
                active_links: 10,
                constellation_health: 0.98,
            },
            tracking_layer: TrackingLayer {
                threats: vec![],
                sensor_coverage: vec![],
                tracking_quality: 0.90,
            },
            ground_layer: GroundLayer {
                stations: vec![],
                communication_links: vec![],
            },
            mission_awareness: MissionAwareness {
                transport_health: 0.95,
                tracking_effectiveness: 0.92,
                threat_status: vec![],
                coupling_matrix: vec![],
                recommended_actions: vec![],
                overall_mission_status: "nominal".to_string(),
            },
        };

        let result = validate_pwsa_telemetry(&telemetry);
        assert!(result.valid, "Validation should succeed");
        assert!(result.errors.is_empty(), "Should have no errors");
    }

    #[test]
    fn test_pwsa_validation_invalid_latitude() {
        let mut telemetry = PwsaTelemetry {
            timestamp: 1234567890,
            transport_layer: TransportLayer {
                satellites: vec![SatelliteState {
                    id: 1,
                    lat: 95.0, // Invalid: > 90
                    lon: -120.0,
                    altitude: 550.0,
                    layer: "transport".to_string(),
                    status: "healthy".to_string(),
                    velocity: 7.5,
                    heading: 90.0,
                }],
                link_quality: 0.95,
                active_links: 10,
                constellation_health: 0.98,
            },
            tracking_layer: TrackingLayer {
                threats: vec![],
                sensor_coverage: vec![],
                tracking_quality: 0.90,
            },
            ground_layer: GroundLayer {
                stations: vec![],
                communication_links: vec![],
            },
            mission_awareness: MissionAwareness {
                transport_health: 0.95,
                tracking_effectiveness: 0.92,
                threat_status: vec![],
                coupling_matrix: vec![],
                recommended_actions: vec![],
                overall_mission_status: "nominal".to_string(),
            },
        };

        let result = validate_pwsa_telemetry(&telemetry);
        assert!(!result.valid, "Validation should fail");
        assert!(!result.errors.is_empty(), "Should have errors");
        assert!(result.errors[0].contains("latitude"), "Error should mention latitude");
    }

    #[test]
    fn test_serialization_roundtrip_pwsa() {
        let telemetry = PwsaTelemetry {
            timestamp: 1234567890,
            transport_layer: TransportLayer {
                satellites: vec![],
                link_quality: 0.95,
                active_links: 10,
                constellation_health: 0.98,
            },
            tracking_layer: TrackingLayer {
                threats: vec![],
                sensor_coverage: vec![],
                tracking_quality: 0.90,
            },
            ground_layer: GroundLayer {
                stations: vec![],
                communication_links: vec![],
            },
            mission_awareness: MissionAwareness {
                transport_health: 0.95,
                tracking_effectiveness: 0.92,
                threat_status: vec![],
                coupling_matrix: vec![],
                recommended_actions: vec![],
                overall_mission_status: "nominal".to_string(),
            },
        };

        let result = test_serialization_roundtrip(&telemetry);
        assert!(result.is_ok(), "Serialization roundtrip should succeed");
    }
}
