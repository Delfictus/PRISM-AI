/// Main web server for PRISM-AI Web Platform
///
/// Starts Actix-web server with WebSocket and API routes
/// Task 2.3.1: WebSocket server integration (all 4 dashboards)

use actix_web::{web, App, HttpServer, middleware};
use actix_cors::Cors;
use std::net::TcpListener;

use super::{metrics_api, websocket, pwsa_websocket, telecom_websocket, hft_websocket};

/// Start the web server on the specified address
pub async fn start_web_server(bind_addr: &str) -> std::io::Result<()> {
    println!("🚀 Starting PRISM-AI Web Platform Server");
    println!("📡 WebSocket Endpoints:");
    println!("   Dashboard #1 (PWSA):    ws://{}/ws/pwsa", bind_addr);
    println!("   Dashboard #2 (Telecom): ws://{}/ws/telecom", bind_addr);
    println!("   Dashboard #3 (HFT):     ws://{}/ws/hft", bind_addr);
    println!("   Dashboard #4 (Metrics): ws://{}/ws/metrics", bind_addr);
    println!("🌐 HTTP Endpoints:");
    println!("   Metrics API: http://{}/api/metrics", bind_addr);
    println!("   Health:      http://{}/health", bind_addr);

    HttpServer::new(|| {
        // Configure CORS for development
        let cors = Cors::permissive(); // TODO: Restrict in production

        App::new()
            .wrap(cors)
            .wrap(middleware::Logger::default())
            .wrap(middleware::Compress::default())
            // WebSocket routes (all 4 dashboards)
            .configure(pwsa_websocket::configure)     // Dashboard #1
            .configure(telecom_websocket::configure)  // Dashboard #2
            .configure(hft_websocket::configure)      // Dashboard #3
            .configure(websocket::configure)          // Dashboard #4
            // API routes
            .configure(metrics_api::configure)
            // Health check
            .route("/health", web::get().to(health_check))
    })
    .bind(bind_addr)?
    .run()
    .await
}

/// Health check endpoint
async fn health_check() -> actix_web::Result<String> {
    Ok("OK".to_string())
}

/// Start server with automatic port selection if default is unavailable
pub async fn start_with_fallback() -> std::io::Result<()> {
    let bind_addrs = vec![
        "127.0.0.1:8080",
        "127.0.0.1:8081",
        "127.0.0.1:8082",
    ];

    for addr in bind_addrs {
        // Try to bind to check if port is available
        if TcpListener::bind(addr).is_ok() {
            println!("✅ Successfully bound to {}", addr);
            return start_web_server(addr).await;
        } else {
            println!("⚠️  Port {} in use, trying next...", addr);
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AddrInUse,
        "No available ports found"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    #[actix_web::test]
    async fn test_health_check() {
        let app = test::init_service(
            App::new().route("/health", web::get().to(health_check))
        ).await;

        let req = test::TestRequest::get().uri("/health").to_request();
        let resp: String = test::call_and_read_body(&app, req).await
            .iter().map(|&b| b as char).collect();

        assert_eq!(resp, "OK");
    }
}
