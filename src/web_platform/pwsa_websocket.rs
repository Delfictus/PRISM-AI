/// PWSA (Proliferated Warfighter Space Architecture) WebSocket Actor
///
/// Streams real-time telemetry from Space Force data fusion system
/// Dashboard #1: 3D globe visualization with satellites, threats, and ground stations
/// Implements Actor pattern using actix-web-actors

use actix::{Actor, AsyncContext, StreamHandler};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::web_platform::types::*;

/// WebSocket actor that streams PWSA telemetry to connected clients
pub struct PwsaWebSocket {
    /// Heartbeat interval for keepalive
    hb_interval: Duration,
    /// Update interval for telemetry streaming
    update_interval: Duration,
}

impl Default for PwsaWebSocket {
    fn default() -> Self {
        Self {
            hb_interval: Duration::from_secs(5),
            update_interval: Duration::from_millis(500), // 2 Hz updates
        }
    }
}

impl PwsaWebSocket {
    pub fn new() -> Self {
        Self::default()
    }

    /// Send heartbeat to client
    fn hb(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(self.hb_interval, |_act, ctx| {
            ctx.ping(b"");
        });
    }

    /// Generate mock PWSA telemetry data
    /// In production, this would fetch real data from the PWSA system
    fn generate_telemetry() -> Result<String, Box<dyn std::error::Error>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let timestamp = chrono::Utc::now().timestamp();

        // Generate satellites (Transport + Tracking layers)
        let mut satellites = Vec::new();

        // Transport layer satellites (LEO constellation)
        for i in 0..12 {
            let angle = (i as f64) * 30.0; // 30° apart
            satellites.push(SatelliteState {
                id: i,
                lat: 45.0 + rng.gen_range(-20.0..20.0),
                lon: angle + rng.gen_range(-5.0..5.0),
                altitude: 550.0 + rng.gen_range(-50.0..50.0),
                layer: "transport".to_string(),
                status: if rng.gen_bool(0.95) {
                    "healthy".to_string()
                } else {
                    "degraded".to_string()
                },
                velocity: 7.5 + rng.gen_range(-0.2..0.2),
                heading: angle,
            });
        }

        // Tracking layer satellites (MEO)
        for i in 12..18 {
            let angle = (i as f64) * 60.0;
            satellites.push(SatelliteState {
                id: i,
                lat: 30.0 + rng.gen_range(-30.0..30.0),
                lon: angle + rng.gen_range(-10.0..10.0),
                altitude: 1200.0 + rng.gen_range(-100.0..100.0),
                layer: "tracking".to_string(),
                status: "healthy".to_string(),
                velocity: 5.8 + rng.gen_range(-0.1..0.1),
                heading: angle,
            });
        }

        // Generate threats
        let mut threats = Vec::new();
        let threat_classes = ["aircraft", "cruise", "ballistic", "hypersonic"];

        for i in 0..5 {
            if rng.gen_bool(0.7) {
                // 70% chance of threat
                threats.push(ThreatDetection {
                    id: i,
                    class: threat_classes[rng.gen_range(0..threat_classes.len())].to_string(),
                    probability: rng.gen_range(0.3..0.95),
                    location: (
                        rng.gen_range(-60.0..60.0),
                        rng.gen_range(-180.0..180.0),
                    ),
                    velocity: rng.gen_range(0.2..5.0),
                    heading: rng.gen_range(0.0..360.0),
                    timestamp,
                    confidence: rng.gen_range(0.6..0.99),
                });
            }
        }

        // Generate ground stations
        let ground_stations = vec![
            GroundStation {
                id: "gs-colorado".to_string(),
                name: "Schriever Space Force Base".to_string(),
                lat: 38.8,
                lon: -104.5,
                status: "active".to_string(),
                uplink_capacity: 100.0,
                downlink_capacity: 200.0,
                connected_satellites: vec![0, 1, 2, 12, 13],
            },
            GroundStation {
                id: "gs-alaska".to_string(),
                name: "Clear Space Force Station".to_string(),
                lat: 64.3,
                lon: -149.2,
                status: "active".to_string(),
                uplink_capacity: 80.0,
                downlink_capacity: 150.0,
                connected_satellites: vec![3, 4, 14],
            },
            GroundStation {
                id: "gs-hawaii".to_string(),
                name: "Hawaii Ground Station".to_string(),
                lat: 22.1,
                lon: -159.8,
                status: "standby".to_string(),
                uplink_capacity: 90.0,
                downlink_capacity: 180.0,
                connected_satellites: vec![5, 6],
            },
        ];

        // Generate communication links
        let mut communication_links = Vec::new();
        for gs in &ground_stations {
            for &sat_id in &gs.connected_satellites {
                communication_links.push(CommunicationLink {
                    source: gs.id.clone(),
                    target: format!("sat-{}", sat_id),
                    bandwidth: rng.gen_range(50.0..120.0),
                    latency: rng.gen_range(5.0..50.0),
                    packet_loss: rng.gen_range(0.0..0.05),
                    status: if rng.gen_bool(0.9) {
                        "active".to_string()
                    } else {
                        "degraded".to_string()
                    },
                });
            }
        }

        // Calculate mission awareness metrics
        let transport_health = satellites
            .iter()
            .filter(|s| s.layer == "transport")
            .filter(|s| s.status == "healthy")
            .count() as f64
            / satellites.iter().filter(|s| s.layer == "transport").count() as f64;

        let tracking_effectiveness = if threats.is_empty() {
            1.0
        } else {
            threats.iter().map(|t| t.confidence).sum::<f64>() / threats.len() as f64
        };

        // Generate coupling matrix (3x3: transport, tracking, ground)
        let coupling_matrix = vec![
            vec![1.0, 0.75, 0.5],  // Transport coupling
            vec![0.75, 1.0, 0.6],  // Tracking coupling
            vec![0.5, 0.6, 1.0],   // Ground coupling
        ];

        // Generate recommended actions
        let mut recommended_actions = Vec::new();
        if transport_health < 0.8 {
            recommended_actions.push(RecommendedAction {
                priority: "high".to_string(),
                category: "maintenance".to_string(),
                description: "Transport layer satellites require attention".to_string(),
                estimated_impact: 0.85,
            });
        }
        if !threats.is_empty() {
            let max_threat = threats.iter().max_by(|a, b| {
                a.probability.partial_cmp(&b.probability).unwrap()
            }).unwrap();

            if max_threat.probability > 0.7 {
                recommended_actions.push(RecommendedAction {
                    priority: "critical".to_string(),
                    category: "threat_response".to_string(),
                    description: format!(
                        "High-probability {} threat detected",
                        max_threat.class
                    ),
                    estimated_impact: 0.95,
                });
            }
        }

        let overall_mission_status = if transport_health < 0.6 || threats.iter().any(|t| t.probability > 0.85) {
            "critical"
        } else if transport_health < 0.8 || threats.iter().any(|t| t.probability > 0.6) {
            "degraded"
        } else {
            "nominal"
        };

        let telemetry = PwsaTelemetry {
            timestamp,
            transport_layer: TransportLayer {
                satellites: satellites.clone(),
                link_quality: rng.gen_range(0.85..0.99),
                active_links: communication_links.len() as u32,
                constellation_health: transport_health,
            },
            tracking_layer: TrackingLayer {
                threats,
                sensor_coverage: vec![],
                tracking_quality: tracking_effectiveness,
            },
            ground_layer: GroundLayer {
                stations: ground_stations,
                communication_links,
            },
            mission_awareness: MissionAwareness {
                transport_health,
                tracking_effectiveness,
                threat_status: vec![0.1, 0.2, 0.15, 0.05], // probabilities for each class
                coupling_matrix,
                recommended_actions,
                overall_mission_status: overall_mission_status.to_string(),
            },
        };

        Ok(serde_json::to_string(&telemetry)?)
    }
}

impl Actor for PwsaWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Start heartbeat
        self.hb(ctx);

        // Start periodic telemetry push
        let update_interval = self.update_interval;
        ctx.run_interval(update_interval, |_act, ctx| {
            match Self::generate_telemetry() {
                Ok(telemetry_json) => ctx.text(telemetry_json),
                Err(e) => {
                    eprintln!("[PwsaWebSocket] Failed to generate telemetry: {}", e);
                }
            }
        });
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for PwsaWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Pong(_)) => {}
            Ok(ws::Message::Text(text)) => {
                // Handle client messages (e.g., refresh request)
                if text.trim() == "refresh" {
                    match Self::generate_telemetry() {
                        Ok(telemetry_json) => ctx.text(telemetry_json),
                        Err(e) => {
                            eprintln!("[PwsaWebSocket] Failed to generate telemetry: {}", e);
                        }
                    }
                }
            }
            Ok(ws::Message::Binary(_)) => {}
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => ctx.stop(),
        }
    }
}

/// WebSocket route handler
pub async fn websocket_route(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, Error> {
    ws::start(PwsaWebSocket::new(), &req, stream)
}

/// Configure WebSocket routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/ws/pwsa").route(web::get().to(websocket_route)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pwsa_telemetry_generation() {
        let result = PwsaWebSocket::generate_telemetry();
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("transport_layer"));
        assert!(json.contains("tracking_layer"));
        assert!(json.contains("mission_awareness"));
    }
}
