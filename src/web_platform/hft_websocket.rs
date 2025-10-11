/// High-Frequency Trading (HFT) WebSocket Actor
///
/// Streams real-time market data with transfer entropy signals
/// Dashboard #3: Trading terminal with orderbook, signals, and execution metrics
/// Implements Actor pattern using actix-web-actors

use actix::{Actor, AsyncContext, StreamHandler};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::web_platform::types::*;

/// WebSocket actor that streams HFT market updates to connected clients
pub struct HftWebSocket {
    /// Heartbeat interval for keepalive
    hb_interval: Duration,
    /// Update interval for market streaming
    update_interval: Duration,
    /// Internal state for realistic market simulation
    tick_count: u64,
    /// Simulated prices for symbols
    prices: std::collections::HashMap<String, f64>,
}

impl Default for HftWebSocket {
    fn default() -> Self {
        let mut prices = std::collections::HashMap::new();
        prices.insert("AAPL".to_string(), 175.50);
        prices.insert("GOOGL".to_string(), 142.30);
        prices.insert("MSFT".to_string(), 378.90);
        prices.insert("AMZN".to_string(), 155.20);
        prices.insert("TSLA".to_string(), 248.50);

        Self {
            hb_interval: Duration::from_secs(5),
            update_interval: Duration::from_millis(100), // 10 Hz updates for HFT
            tick_count: 0,
            prices,
        }
    }
}

impl HftWebSocket {
    pub fn new() -> Self {
        Self::default()
    }

    /// Send heartbeat to client
    fn hb(&self, ctx: &mut ws::WebsocketContext<Self>) {
        ctx.run_interval(self.hb_interval, |_act, ctx| {
            ctx.ping(b"");
        });
    }

    /// Generate mock HFT market data
    /// In production, this would connect to real market data feeds
    fn generate_market_update(
        tick: u64,
        prices: &mut std::collections::HashMap<String, f64>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let timestamp = chrono::Utc::now().timestamp();

        // Symbols to track
        let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];

        // Update prices with realistic random walk
        let mut price_data = Vec::new();
        for symbol in &symbols {
            let current_price = prices.get(*symbol).cloned().unwrap_or(100.0);

            // Random walk with mean reversion
            let change_pct = rng.gen_range(-0.002..0.002); // ±0.2% per tick
            let new_price = current_price * (1.0 + change_pct);

            prices.insert(symbol.to_string(), new_price);

            let open = current_price;
            let close = new_price;
            let high = current_price.max(new_price) * (1.0 + rng.gen_range(0.0..0.001));
            let low = current_price.min(new_price) * (1.0 - rng.gen_range(0.0..0.001));

            price_data.push(PriceData {
                symbol: symbol.to_string(),
                price: new_price,
                volume: rng.gen_range(100000.0..10000000.0),
                bid: new_price * 0.9995,
                ask: new_price * 1.0005,
                high,
                low,
                open,
                close,
                change: change_pct * 100.0,
                change_value: new_price - current_price,
            });
        }

        // Generate order book for primary symbol (AAPL)
        let aapl_price = prices.get("AAPL").cloned().unwrap_or(175.50);
        let mut bids = Vec::new();
        let mut asks = Vec::new();

        for i in 0..10 {
            bids.push(OrderLevel {
                price: aapl_price - (i as f64) * 0.01,
                volume: rng.gen_range(100.0..5000.0),
                order_count: rng.gen_range(5..50),
            });

            asks.push(OrderLevel {
                price: aapl_price + (i as f64) * 0.01,
                volume: rng.gen_range(100.0..5000.0),
                order_count: rng.gen_range(5..50),
            });
        }

        let total_bid_volume: f64 = bids.iter().map(|b| b.volume).sum();
        let total_ask_volume: f64 = asks.iter().map(|a| a.volume).sum();

        let order_book = OrderBook {
            symbol: "AAPL".to_string(),
            bids,
            asks,
            spread: aapl_price * 0.001,
            depth: (total_bid_volume + total_ask_volume) / 2.0,
            imbalance: total_bid_volume - total_ask_volume,
        };

        // Generate transfer entropy signal (causality detection)
        let te_signal = TransferEntropySignal {
            source: "GOOGL".to_string(),
            target: "AAPL".to_string(),
            te_value: rng.gen_range(0.1..0.9),
            lag: rng.gen_range(50..500), // 50-500 ms lag
            significance: rng.gen_range(0.01..0.05), // p-value
            causal_strength: rng.gen_range(0.5..0.95),
        };

        // Generate trading signals
        let predicted_direction = if te_signal.causal_strength > 0.75 {
            if rng.gen_bool(0.6) { "up" } else { "down" }
        } else {
            "neutral"
        };

        let confidence = te_signal.causal_strength * rng.gen_range(0.8..0.95);
        let signal_strength = te_signal.te_value * confidence;

        let recommended_action = if predicted_direction == "up" && confidence > 0.7 {
            "buy"
        } else if predicted_direction == "down" && confidence > 0.7 {
            "sell"
        } else {
            "hold"
        };

        let trading_signals = TradingSignals {
            transfer_entropy: te_signal,
            predicted_direction: predicted_direction.to_string(),
            confidence,
            signal_strength,
            risk_score: 1.0 - confidence,
            recommended_action: recommended_action.to_string(),
            position_size: if recommended_action != "hold" {
                confidence * 1000.0 // Shares
            } else {
                0.0
            },
        };

        // Generate execution metrics
        let execution_metrics = ExecutionMetrics {
            latency_us: rng.gen_range(50..500), // 50-500 microseconds
            slippage_bps: rng.gen_range(0.1..5.0), // 0.1-5 basis points
            fill_rate: rng.gen_range(0.85..0.99),
            reject_rate: rng.gen_range(0.001..0.05),
            orders_per_second: rng.gen_range(100.0..1000.0),
            pnl: rng.gen_range(-5000.0..15000.0),
            sharpe_ratio: rng.gen_range(0.5..3.0),
        };

        // Generate portfolio state
        let positions = vec![
            Position {
                symbol: "AAPL".to_string(),
                quantity: 1000.0,
                avg_price: 173.20,
                current_price: aapl_price,
                unrealized_pnl: (aapl_price - 173.20) * 1000.0,
                unrealized_pnl_pct: ((aapl_price - 173.20) / 173.20) * 100.0,
                market_value: aapl_price * 1000.0,
            },
            Position {
                symbol: "GOOGL".to_string(),
                quantity: 500.0,
                avg_price: 140.50,
                current_price: prices.get("GOOGL").cloned().unwrap_or(142.30),
                unrealized_pnl: (prices.get("GOOGL").cloned().unwrap_or(142.30) - 140.50) * 500.0,
                unrealized_pnl_pct: ((prices.get("GOOGL").cloned().unwrap_or(142.30) - 140.50) / 140.50) * 100.0,
                market_value: prices.get("GOOGL").cloned().unwrap_or(142.30) * 500.0,
            },
        ];

        let total_value = positions.iter().map(|p| p.market_value).sum::<f64>() + 50000.0; // Cash
        let daily_pnl = execution_metrics.pnl;
        let daily_pnl_pct = (daily_pnl / total_value) * 100.0;

        let portfolio = PortfolioState {
            cash: 50000.0,
            positions,
            total_value,
            daily_pnl,
            daily_pnl_pct,
            max_drawdown: rng.gen_range(0.02..0.15),
            win_rate: rng.gen_range(0.55..0.75),
        };

        let market_update = MarketUpdate {
            timestamp,
            prices: price_data,
            order_book,
            signals: trading_signals,
            execution: execution_metrics,
            portfolio,
        };

        Ok(serde_json::to_string(&market_update)?)
    }
}

impl Actor for HftWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Start heartbeat
        self.hb(ctx);

        // Start periodic market update push (10 Hz for HFT)
        let update_interval = self.update_interval;
        ctx.run_interval(update_interval, |act, ctx| {
            act.tick_count += 1;
            match Self::generate_market_update(act.tick_count, &mut act.prices) {
                Ok(update_json) => ctx.text(update_json),
                Err(e) => {
                    eprintln!("[HftWebSocket] Failed to generate market update: {}", e);
                }
            }
        });
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for HftWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Pong(_)) => {}
            Ok(ws::Message::Text(text)) => {
                // Handle client messages
                if text.trim() == "refresh" {
                    match Self::generate_market_update(self.tick_count, &mut self.prices) {
                        Ok(update_json) => ctx.text(update_json),
                        Err(e) => {
                            eprintln!("[HftWebSocket] Failed to generate market update: {}", e);
                        }
                    }
                } else if text.trim() == "reset" {
                    // Reset market simulation
                    self.tick_count = 0;
                    self.prices.insert("AAPL".to_string(), 175.50);
                    self.prices.insert("GOOGL".to_string(), 142.30);
                    self.prices.insert("MSFT".to_string(), 378.90);
                    self.prices.insert("AMZN".to_string(), 155.20);
                    self.prices.insert("TSLA".to_string(), 248.50);
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
    ws::start(HftWebSocket::new(), &req, stream)
}

/// Configure WebSocket routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/ws/hft").route(web::get().to(websocket_route)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_update_generation() {
        let mut prices = std::collections::HashMap::new();
        prices.insert("AAPL".to_string(), 175.50);
        prices.insert("GOOGL".to_string(), 142.30);

        let result = HftWebSocket::generate_market_update(0, &mut prices);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("prices"));
        assert!(json.contains("order_book"));
        assert!(json.contains("signals"));
        assert!(json.contains("execution"));
        assert!(json.contains("portfolio"));
    }

    #[test]
    fn test_price_evolution() {
        let mut prices = std::collections::HashMap::new();
        prices.insert("AAPL".to_string(), 175.50);

        let initial_price = prices.get("AAPL").cloned().unwrap();

        // Generate multiple updates
        for _ in 0..10 {
            let _ = HftWebSocket::generate_market_update(0, &mut prices);
        }

        let final_price = prices.get("AAPL").cloned().unwrap();

        // Price should have changed (random walk)
        assert!(initial_price != final_price);
    }
}
