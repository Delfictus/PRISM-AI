/// Metrics API endpoint for Web Platform
///
/// Provides HTTP endpoint that exports Prometheus metrics in text format
/// for consumption by the React dashboard via WebSocket bridge

use actix_web::{web, HttpResponse, Result};
use prometheus::TextEncoder;

/// HTTP endpoint that returns all Prometheus metrics in text format
pub async fn metrics_endpoint() -> Result<HttpResponse> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();

    match encoder.encode(&metric_families, &mut buffer) {
        Ok(_) => Ok(HttpResponse::Ok()
            .content_type("text/plain; version=0.0.4")
            .body(buffer)),
        Err(e) => Ok(HttpResponse::InternalServerError()
            .body(format!("Failed to encode metrics: {}", e))),
    }
}

/// Configure metrics API routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/api/metrics")
            .route(web::get().to(metrics_endpoint))
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    #[actix_web::test]
    async fn test_metrics_endpoint() {
        let app = test::init_service(
            App::new().configure(configure)
        ).await;

        let req = test::TestRequest::get()
            .uri("/api/metrics")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }
}

/// Public API for metrics retrieval
pub struct MetricsApi;

impl MetricsApi {
    /// Get all current metrics as JSON
    pub fn get_metrics_json() -> Result<String, Box<dyn std::error::Error>> {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;

        // Convert Prometheus text format to JSON for WebSocket
        // This is a simplified version - full implementation would parse
        // the Prometheus format and convert to structured JSON
        Ok(String::from_utf8(buffer)?)
    }
}
