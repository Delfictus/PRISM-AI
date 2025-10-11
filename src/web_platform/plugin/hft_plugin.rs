/// HFT (High-Frequency Trading) Plugin
/// Wraps existing HFT WebSocket functionality into plugin architecture
use async_trait::async_trait;
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use rand::Rng;

use super::types::*;
use super::health::{HealthStatus, HealthMetrics};
use crate::web_platform::types::*;

/// HFT Plugin - High-Frequency Trading Dashboard
pub struct HftPlugin {
    config: PluginConfig,
    metrics: HealthMetrics,
    running: bool,
    tick_count: u64,
    prices: HashMap<String, f64>,
}

impl HftPlugin {
    /// Create new HFT plugin with default configuration
    pub fn new() -> Self {
        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 150.00);
        prices.insert("GOOGL".to_string(), 2800.00);
        prices.insert("MSFT".to_string(), 380.00);
        prices.insert("AMZN".to_string(), 3200.00);
        prices.insert("TSLA".to_string(), 720.00);

        Self {
            config: PluginConfig {
                enabled: true,
                update_interval_ms: 100,  // 10 Hz market data
                auto_restart: true,
                max_retries: 3,
                custom_params: Default::default(),
            },
            metrics: HealthMetrics::default(),
            running: false,
            tick_count: 0,
            prices,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PluginConfig) -> Self {
        let mut plugin = Self::new();
        plugin.config = config;
        plugin
    }

    /// Generate HFT market update
    fn generate_market_update(
        tick: u64,
        prices: &mut HashMap<String, f64>,
    ) -> Result<HftUpdate, PluginError> {
        let mut rng = rand::thread_rng();

        // Update prices with random walk (±0.2% per tick)
        let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];
        let mut market_data = Vec::new();

        for symbol in &symbols {
            let price = prices.get_mut(*symbol).unwrap();
            let change_pct = rng.gen_range(-0.002..0.002);
            *price *= 1.0 + change_pct;

            // Generate order book (10 levels each side)
            let mut bids = Vec::new();
            let mut asks = Vec::new();
            for i in 0..10 {
                let bid_price = *price * (1.0 - 0.0001 * (i as f64 + 1.0));
                let ask_price = *price * (1.0 + 0.0001 * (i as f64 + 1.0));
                bids.push(HftOrderLevel {
                    price: bid_price,
                    size: rng.gen_range(100..1000),
                });
                asks.push(HftOrderLevel {
                    price: ask_price,
                    size: rng.gen_range(100..1000),
                });
            }

            market_data.push(HftMarketData {
                symbol: symbol.to_string(),
                last_price: *price,
                bid: bids[0].price,
                ask: asks[0].price,
                volume: rng.gen_range(100000..1000000),
                order_book: HftOrderBook { bids, asks },
            });
        }

        // Generate transfer entropy signals
        let mut te_signals = Vec::new();
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let te_value = rng.gen_range(0.0..0.5);
                if te_value > 0.15 {
                    te_signals.push(HftTransferEntropySignal {
                        from_symbol: symbols[i].to_string(),
                        to_symbol: symbols[j].to_string(),
                        te_value,
                        significant: te_value > 0.25,
                    });
                }
            }
        }

        // Execution metrics
        let execution = HftExecutionMetrics {
            orders_per_second: rng.gen_range(1000..5000),
            avg_latency_us: rng.gen_range(50.0..500.0),
            fill_rate: rng.gen_range(0.85..0.98),
        };

        Ok(HftUpdate {
            tick,
            market_data,
            transfer_entropy: te_signals,
            execution,
        })
    }
}

impl Default for HftPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PrismPlugin for HftPlugin {
    fn id(&self) -> &str {
        "hft"
    }

    fn name(&self) -> &str {
        "High-Frequency Trading (HFT)"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "Generates real-time market data, order book depth, transfer entropy signals, \
         and execution metrics for high-frequency trading simulation"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![
            Capability::TelemetryGenerator,
            Capability::RealtimeStreaming,
            Capability::TransferEntropy,
            Capability::Custom("market_data".to_string()),
            Capability::Custom("order_book".to_string()),
        ]
    }

    async fn initialize(&mut self) -> Result<(), PluginError> {
        println!("💹 Initializing HFT Plugin...");
        println!("   - Symbols: AAPL, GOOGL, MSFT, AMZN, TSLA");
        println!("   - Order Book: 10 levels");
        println!("   - Transfer Entropy: Pairwise analysis");
        println!("   - Update Rate: 10 Hz");
        Ok(())
    }

    async fn start(&mut self) -> Result<(), PluginError> {
        if self.running {
            return Err(PluginError::AlreadyRunning(self.id().to_string()));
        }
        self.running = true;
        self.tick_count = 0;
        self.metrics = HealthMetrics::default();
        println!("▶️  HFT Plugin started ({}ms interval)", self.config.update_interval_ms);
        Ok(())
    }

    async fn stop(&mut self) -> Result<(), PluginError> {
        if !self.running {
            return Err(PluginError::NotRunning(self.id().to_string()));
        }
        self.running = false;
        println!("⏸️  HFT Plugin stopped at tick {}", self.tick_count);
        Ok(())
    }

    async fn generate_data(&self) -> Result<PluginData, PluginError> {
        if !self.running {
            return Err(PluginError::NotRunning(self.id().to_string()));
        }

        // Note: In real implementation, use Arc<RwLock<HashMap>> for prices
        // For now, clone prices for mutation
        let mut prices_copy = self.prices.clone();
        let update = Self::generate_market_update(self.tick_count, &mut prices_copy)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| PluginError::DataGenerationFailed(e.to_string()))?
            .as_secs();

        // Convert to JSON
        let payload = serde_json::to_value(&update)
            .map_err(|e| PluginError::DataGenerationFailed(e.to_string()))?;

        Ok(PluginData {
            plugin_id: self.id().to_string(),
            timestamp: now,
            data_type: "hft_update".to_string(),
            payload,
        })
    }

    fn health_check(&self) -> HealthStatus {
        let level = if self.metrics.consecutive_failures > 5 {
            super::health::HealthLevel::Unhealthy
        } else if self.metrics.consecutive_failures > 2 {
            super::health::HealthLevel::Degraded
        } else {
            super::health::HealthLevel::Healthy
        };

        let message = if self.running {
            format!("Market data at tick {} ({}ms intervals)",
                    self.tick_count, self.config.update_interval_ms)
        } else {
            "Plugin stopped".to_string()
        };

        HealthStatus {
            level,
            message,
            last_check: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metrics: self.metrics.clone(),
        }
    }

    async fn reconfigure(&mut self, config: PluginConfig) -> Result<(), PluginError> {
        // Validate configuration
        if config.update_interval_ms < 50 {
            return Err(PluginError::ConfigurationError(
                "Update interval must be at least 50ms for market data".to_string()
            ));
        }

        if config.update_interval_ms > 5000 {
            return Err(PluginError::ConfigurationError(
                "Update interval must be at most 5000ms".to_string()
            ));
        }

        self.config = config;
        println!("🔧 HFT Plugin reconfigured: {}ms interval", self.config.update_interval_ms);
        Ok(())
    }

    fn get_config(&self) -> &PluginConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hft_plugin_initialization() {
        let mut plugin = HftPlugin::new();
        let result = plugin.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_hft_plugin_lifecycle() {
        let mut plugin = HftPlugin::new();
        plugin.initialize().await.unwrap();

        // Start
        let result = plugin.start().await;
        assert!(result.is_ok());

        // Generate data
        let data = plugin.generate_data().await.unwrap();
        assert_eq!(data.plugin_id, "hft");
        assert_eq!(data.data_type, "hft_update");

        // Stop
        let result = plugin.stop().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_market_data_generation() {
        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 150.00);
        prices.insert("GOOGL".to_string(), 2800.00);
        prices.insert("MSFT".to_string(), 380.00);
        prices.insert("AMZN".to_string(), 3200.00);
        prices.insert("TSLA".to_string(), 720.00);

        let update = HftPlugin::generate_market_update(1, &mut prices).unwrap();
        assert_eq!(update.market_data.len(), 5);
        assert_eq!(update.market_data[0].order_book.bids.len(), 10);
        assert_eq!(update.market_data[0].order_book.asks.len(), 10);
    }
}
