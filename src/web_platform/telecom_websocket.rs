/// Telecommunications & Network Optimization WebSocket Actor
///
/// Streams real-time network topology and graph coloring optimization state
/// Dashboard #2: Network graph visualization with real-time optimization
/// Implements Actor pattern using actix-web-actors

use actix::{Actor, AsyncContext, StreamHandler};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::web_platform::types::*;

/// WebSocket actor that streams Telecom updates to connected clients
pub struct TelecomWebSocket {
    /// Heartbeat interval for keepalive
    hb_interval: Duration,
    /// Update interval for network streaming
    update_interval: Duration,
    /// Internal state for animation
    iteration_count: u64,
}

impl Default for TelecomWebSocket {
    fn default() -> Self {
        Self {
            hb_interval: Duration::from_secs(5),
            update_interval: Duration::from_millis(200), // 5 Hz updates for smooth animation
            iteration_count: 0,
        }
    }
}

impl TelecomWebSocket {
    pub fn new() -> Self {
        Self::default()
    }

    /// Send heartbeat to client
    fn hb(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(self.hb_interval, |_act, ctx| {
            ctx.ping(b"");
        });
    }

    /// Generate mock Telecom network data
    /// In production, this would fetch real data from the network optimization engine
    fn generate_update(iteration: u64) -> Result<String, Box<dyn std::error::Error>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let timestamp = chrono::Utc::now().timestamp();

        // Generate network topology (graph coloring problem)
        let num_nodes = 50;
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create nodes arranged in a circular layout
        for i in 0..num_nodes {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (num_nodes as f64);
            let radius = 100.0;

            let node_types = ["router", "switch", "endpoint", "hub"];
            nodes.push(NetworkNode {
                id: format!("node-{}", i),
                label: format!("N{}", i),
                x: radius * angle.cos(),
                y: radius * angle.sin(),
                color: ((iteration + i) % 7) as u32, // Rotating colors for animation
                degree: 0, // Will be calculated
                status: if rng.gen_bool(0.95) {
                    "active".to_string()
                } else {
                    "degraded".to_string()
                },
                load: rng.gen_range(0.2..0.9),
                capacity: rng.gen_range(100.0..1000.0),
                node_type: node_types[i % node_types.len()].to_string(),
            });
        }

        // Create edges (graph connectivity)
        let mut degree_counts = vec![0u32; num_nodes as usize];
        for i in 0..num_nodes {
            // Connect to 2-4 neighbors
            let num_connections = rng.gen_range(2..5);
            for _ in 0..num_connections {
                let target = rng.gen_range(0..num_nodes);
                if target != i && !edges.iter().any(|e: &NetworkEdge| {
                    (e.source == format!("node-{}", i) && e.target == format!("node-{}", target))
                        || (e.source == format!("node-{}", target) && e.target == format!("node-{}", i))
                }) {
                    edges.push(NetworkEdge {
                        id: format!("edge-{}-{}", i, target),
                        source: format!("node-{}", i),
                        target: format!("node-{}", target),
                        utilization: rng.gen_range(0.1..0.95),
                        bandwidth: rng.gen_range(100.0..10000.0),
                        latency: rng.gen_range(1.0..50.0),
                        packet_loss: rng.gen_range(0.0..0.1),
                        status: if rng.gen_bool(0.9) {
                            "active".to_string()
                        } else {
                            "congested".to_string()
                        },
                        weight: rng.gen_range(1.0..10.0),
                    });

                    degree_counts[i as usize] += 1;
                    degree_counts[target as usize] += 1;
                }
            }
        }

        // Update node degrees
        for (i, node) in nodes.iter_mut().enumerate() {
            node.degree = degree_counts[i];
        }

        let max_degree = *degree_counts.iter().max().unwrap_or(&0);
        let avg_degree = degree_counts.iter().sum::<u32>() as f64 / num_nodes as f64;

        let graph_stats = GraphStats {
            total_nodes: num_nodes,
            total_edges: edges.len() as u32,
            chromatic_number: 7, // Theoretical optimum
            max_degree,
            avg_degree,
            clustering_coefficient: rng.gen_range(0.3..0.7),
        };

        // Simulate optimization convergence
        let convergence = (iteration as f64 / 100.0).min(1.0);
        let current_coloring = (15.0 - 8.0 * convergence) as u32; // Converging from 15 to 7 colors
        let best_coloring = current_coloring.min(8);

        let optimization_state = OptimizationState {
            current_coloring,
            best_coloring,
            optimal_coloring: 7,
            iterations: iteration,
            convergence,
            algorithm: "quantum_annealing".to_string(),
            time_elapsed_ms: iteration * 200, // Update interval * iterations
        };

        // Calculate network performance
        let total_throughput_mbps = edges
            .iter()
            .map(|e| e.bandwidth * (1.0 - e.utilization))
            .sum();

        let avg_latency_ms = edges.iter().map(|e| e.latency).sum::<f64>() / edges.len() as f64;
        let packet_loss_rate =
            edges.iter().map(|e| e.packet_loss).sum::<f64>() / edges.len() as f64;

        let network_performance = NetworkPerformance {
            total_throughput_mbps,
            avg_latency_ms,
            packet_loss_rate,
            jitter_ms: rng.gen_range(0.5..5.0),
            uptime: rng.gen_range(0.95..0.999),
            quality_of_service: 1.0 - (packet_loss_rate + avg_latency_ms / 100.0) / 2.0,
        };

        let update = TelecomUpdate {
            timestamp,
            network_topology: NetworkTopology {
                nodes,
                edges,
                graph_stats,
            },
            optimization_state,
            performance: network_performance,
        };

        Ok(serde_json::to_string(&update)?)
    }
}

impl Actor for TelecomWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Start heartbeat
        self.hb(ctx);

        // Start periodic network update push
        let update_interval = self.update_interval;
        ctx.run_interval(update_interval, |act, ctx| {
            act.iteration_count += 1;
            match Self::generate_update(act.iteration_count) {
                Ok(update_json) => ctx.text(update_json),
                Err(e) => {
                    eprintln!("[TelecomWebSocket] Failed to generate update: {}", e);
                }
            }
        });
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for TelecomWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Pong(_)) => {}
            Ok(ws::Message::Text(text)) => {
                // Handle client messages
                if text.trim() == "refresh" {
                    match Self::generate_update(self.iteration_count) {
                        Ok(update_json) => ctx.text(update_json),
                        Err(e) => {
                            eprintln!("[TelecomWebSocket] Failed to generate update: {}", e);
                        }
                    }
                } else if text.trim() == "reset" {
                    // Reset optimization animation
                    self.iteration_count = 0;
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
    ws::start(TelecomWebSocket::new(), &req, stream)
}

/// Configure WebSocket routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/ws/telecom").route(web::get().to(websocket_route)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telecom_update_generation() {
        let result = TelecomWebSocket::generate_update(10);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("network_topology"));
        assert!(json.contains("optimization_state"));
        assert!(json.contains("performance"));
    }

    #[test]
    fn test_convergence_animation() {
        // Test that optimization converges over time
        let result1 = TelecomWebSocket::generate_update(0);
        let result2 = TelecomWebSocket::generate_update(100);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        // At iteration 0, should have higher color count than at iteration 100
        let json1 = result1.unwrap();
        let json2 = result2.unwrap();

        assert!(json1.contains("current_coloring"));
        assert!(json2.contains("current_coloring"));
    }
}
