/// WebSocket server for real-time metrics streaming
///
/// Streams Prometheus metrics to React frontend via WebSocket connection
/// Implements Actor pattern using actix-web-actors

use actix::{Actor, StreamHandler, AsyncContext, ActorContext};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use serde::{Serialize, Deserialize};
use std::time::Duration;
use crate::metrics::*;

/// WebSocket actor that streams metrics to connected clients
pub struct MetricsWebSocket {
    /// Heartbeat interval for keepalive
    hb_interval: Duration,
}

impl Default for MetricsWebSocket {
    fn default() -> Self {
        Self {
            hb_interval: Duration::from_secs(5),
        }
    }
}

impl MetricsWebSocket {
    pub fn new() -> Self {
        Self::default()
    }

    /// Send heartbeat to client
    fn hb(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(self.hb_interval, |_act, ctx| {
            ctx.ping(b"");
        });
    }

    /// Collect and serialize current metrics
    fn collect_metrics() -> Result<String, Box<dyn std::error::Error>> {
        let metrics = MetricsSnapshot {
            optimization_iterations: OPTIMIZATION_ITERATIONS.get() as u64,
            solution_quality: SOLUTION_QUALITY.get(),
            gpu_utilization: GPU_UTILIZATION.get(),
            gpu_memory_used: GPU_MEMORY_USED.get(),
            cpu_memory_used: CPU_MEMORY_USED.get(),
            processing_time_count: 0, // Would get from histogram
            active_problems: ACTIVE_PROBLEMS.get() as u64,
            quantum_annealing_steps: QUANTUM_ANNEALING_STEPS.get() as u64,
            neuromorphic_spikes: NEUROMORPHIC_SPIKES.get() as u64,
            transfer_entropy_calculations: TRANSFER_ENTROPY_CALCULATIONS.get() as u64,
            tsp_tour_length: TSP_TOUR_LENGTH.get(),
            tsp_cities: TSP_CITIES.get(),
            graph_colors_used: GRAPH_COLORS_USED.get(),
            graph_vertices: GRAPH_VERTICES.get(),
            graph_edges: GRAPH_EDGES.get(),
            errors_total: ERRORS_TOTAL.get() as u64,
            timestamp: chrono::Utc::now().timestamp(),
        };

        Ok(serde_json::to_string(&metrics)?)
    }
}

/// Metrics snapshot for JSON serialization
#[derive(Serialize, Deserialize, Clone)]
pub struct MetricsSnapshot {
    pub optimization_iterations: u64,
    pub solution_quality: f64,
    pub gpu_utilization: f64,
    pub gpu_memory_used: f64,
    pub cpu_memory_used: f64,
    pub processing_time_count: u64,
    pub active_problems: u64,
    pub quantum_annealing_steps: u64,
    pub neuromorphic_spikes: u64,
    pub transfer_entropy_calculations: u64,
    pub tsp_tour_length: f64,
    pub tsp_cities: f64,
    pub graph_colors_used: f64,
    pub graph_vertices: f64,
    pub graph_edges: f64,
    pub errors_total: u64,
    pub timestamp: i64,
}

impl Actor for MetricsWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Start heartbeat
        self.hb(ctx);

        // Start periodic metrics push (every 1 second)
        ctx.run_interval(Duration::from_secs(1), |_act, ctx| {
            match Self::collect_metrics() {
                Ok(metrics_json) => ctx.text(metrics_json),
                Err(e) => {
                    eprintln!("Failed to collect metrics: {}", e);
                    ERRORS_TOTAL.inc();
                }
            }
        });
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MetricsWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Pong(_)) => {},
            Ok(ws::Message::Text(text)) => {
                // Handle client messages (e.g., metric queries)
                if text.trim() == "refresh" {
                    match Self::collect_metrics() {
                        Ok(metrics_json) => ctx.text(metrics_json),
                        Err(e) => {
                            eprintln!("Failed to collect metrics: {}", e);
                            ERRORS_TOTAL.inc();
                        }
                    }
                }
            },
            Ok(ws::Message::Binary(_)) => {},
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            },
            _ => ctx.stop(),
        }
    }
}

/// WebSocket route handler
pub async fn websocket_route(
    req: HttpRequest,
    stream: web::Payload,
) -> Result<HttpResponse, Error> {
    ws::start(MetricsWebSocket::new(), &req, stream)
}

/// Configure WebSocket routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/ws/metrics")
            .route(web::get().to(websocket_route))
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_snapshot_serialization() {
        let snapshot = MetricsSnapshot {
            optimization_iterations: 100,
            solution_quality: 95.5,
            gpu_utilization: 78.3,
            gpu_memory_used: 4_000_000_000.0,
            cpu_memory_used: 2_000_000_000.0,
            processing_time_count: 50,
            active_problems: 2,
            quantum_annealing_steps: 1000,
            neuromorphic_spikes: 5000,
            transfer_entropy_calculations: 25,
            tsp_tour_length: 12345.6,
            tsp_cities: 100.0,
            graph_colors_used: 20.0,
            graph_vertices: 1000.0,
            graph_edges: 5000.0,
            errors_total: 3,
            timestamp: 1696973600,
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(json.contains("optimization_iterations"));
        assert!(json.contains("95.5"));
    }
}
