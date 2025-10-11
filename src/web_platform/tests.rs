/// Comprehensive Serialization/Deserialization Tests
///
/// Task 2.1.4: Test serialization/deserialization for all dashboard types
/// Ensures type safety across the Rust ↔ TypeScript boundary

#[cfg(test)]
mod tests {
    use crate::web_platform::types::*;
    use crate::web_platform::validation::*;
    use serde_json;

    /// Helper to test full roundtrip: Rust → JSON → Rust
    fn test_roundtrip<T>(data: &T) -> Result<String, String>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug + PartialEq,
    {
        // Step 1: Serialize to JSON
        let json_str = serde_json::to_string_pretty(data)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        // Step 2: Deserialize back
        let deserialized: T = serde_json::from_str(&json_str)
            .map_err(|e| format!("Deserialization failed: {}", e))?;

        // Step 3: Verify data integrity (if PartialEq is implemented)
        // For now, just return JSON for inspection

        Ok(json_str)
    }

    // =======================================================================
    // Dashboard #1: PWSA Telemetry Tests
    // =======================================================================

    #[test]
    fn test_pwsa_satellite_state_serialization() {
        let sat = SatelliteState {
            id: 1,
            lat: 45.5,
            lon: -122.6,
            altitude: 550.0,
            layer: "transport".to_string(),
            status: "healthy".to_string(),
            velocity: 7.5,
            heading: 90.0,
        };

        let json = serde_json::to_string(&sat).expect("Serialization failed");
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"lat\":45.5"));
        assert!(json.contains("\"layer\":\"transport\""));

        let deserialized: SatelliteState = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.id, 1);
        assert_eq!(deserialized.layer, "transport");
    }

    #[test]
    fn test_pwsa_threat_detection_serialization() {
        let threat = ThreatDetection {
            id: 42,
            class: "hypersonic".to_string(),
            probability: 0.85,
            location: (35.0, -110.0),
            velocity: 8.5,
            heading: 270.0,
            timestamp: 1234567890,
            confidence: 0.92,
        };

        let json = serde_json::to_string(&threat).expect("Serialization failed");
        assert!(json.contains("\"class\":\"hypersonic\""));
        assert!(json.contains("\"probability\":0.85"));

        let deserialized: ThreatDetection = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.id, 42);
        assert_eq!(deserialized.class, "hypersonic");
        assert_eq!(deserialized.probability, 0.85);
    }

    #[test]
    fn test_pwsa_ground_station_serialization() {
        let station = GroundStation {
            id: "gs-colorado".to_string(),
            name: "Schriever SFB".to_string(),
            lat: 38.8,
            lon: -104.5,
            status: "active".to_string(),
            uplink_capacity: 100.0,
            downlink_capacity: 200.0,
            connected_satellites: vec![1, 2, 3],
        };

        let json = serde_json::to_string(&station).expect("Serialization failed");
        assert!(json.contains("\"id\":\"gs-colorado\""));
        assert!(json.contains("\"connected_satellites\":[1,2,3]"));

        let deserialized: GroundStation = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.id, "gs-colorado");
        assert_eq!(deserialized.connected_satellites.len(), 3);
    }

    #[test]
    fn test_pwsa_full_telemetry_serialization() {
        let telemetry = PwsaTelemetry {
            timestamp: 1234567890,
            transport_layer: TransportLayer {
                satellites: vec![
                    SatelliteState {
                        id: 1,
                        lat: 45.0,
                        lon: -120.0,
                        altitude: 550.0,
                        layer: "transport".to_string(),
                        status: "healthy".to_string(),
                        velocity: 7.5,
                        heading: 90.0,
                    },
                ],
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
                threat_status: vec![0.1, 0.2, 0.15, 0.05],
                coupling_matrix: vec![
                    vec![1.0, 0.75, 0.5],
                    vec![0.75, 1.0, 0.6],
                    vec![0.5, 0.6, 1.0],
                ],
                recommended_actions: vec![],
                overall_mission_status: "nominal".to_string(),
            },
        };

        let json = serde_json::to_string_pretty(&telemetry).expect("Serialization failed");
        assert!(json.contains("\"transport_layer\""));
        assert!(json.contains("\"mission_awareness\""));

        let deserialized: PwsaTelemetry = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.timestamp, 1234567890);
        assert_eq!(deserialized.transport_layer.satellites.len(), 1);
        assert_eq!(deserialized.mission_awareness.overall_mission_status, "nominal");
    }

    // =======================================================================
    // Dashboard #2: Telecom Network Tests
    // =======================================================================

    #[test]
    fn test_telecom_network_node_serialization() {
        let node = NetworkNode {
            id: "node-1".to_string(),
            label: "Router A".to_string(),
            x: 100.0,
            y: 200.0,
            color: 3,
            degree: 5,
            status: "active".to_string(),
            load: 0.75,
            capacity: 1000.0,
            node_type: "router".to_string(),
        };

        let json = serde_json::to_string(&node).expect("Serialization failed");
        assert!(json.contains("\"id\":\"node-1\""));
        assert!(json.contains("\"color\":3"));
        assert!(json.contains("\"node_type\":\"router\""));

        let deserialized: NetworkNode = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.id, "node-1");
        assert_eq!(deserialized.color, 3);
    }

    #[test]
    fn test_telecom_network_edge_serialization() {
        let edge = NetworkEdge {
            id: "edge-1-2".to_string(),
            source: "node-1".to_string(),
            target: "node-2".to_string(),
            utilization: 0.65,
            bandwidth: 1000.0,
            latency: 10.5,
            packet_loss: 0.01,
            status: "active".to_string(),
            weight: 1.5,
        };

        let json = serde_json::to_string(&edge).expect("Serialization failed");
        assert!(json.contains("\"source\":\"node-1\""));
        assert!(json.contains("\"target\":\"node-2\""));

        let deserialized: NetworkEdge = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.source, "node-1");
        assert_eq!(deserialized.target, "node-2");
    }

    #[test]
    fn test_telecom_optimization_state_serialization() {
        let opt_state = OptimizationState {
            current_coloring: 12,
            best_coloring: 8,
            optimal_coloring: 7,
            iterations: 1500,
            convergence: 0.75,
            algorithm: "quantum_annealing".to_string(),
            time_elapsed_ms: 3000,
        };

        let json = serde_json::to_string(&opt_state).expect("Serialization failed");
        assert!(json.contains("\"algorithm\":\"quantum_annealing\""));
        assert!(json.contains("\"convergence\":0.75"));

        let deserialized: OptimizationState = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.current_coloring, 12);
        assert_eq!(deserialized.best_coloring, 8);
    }

    #[test]
    fn test_telecom_full_update_serialization() {
        let update = TelecomUpdate {
            timestamp: 1234567890,
            network_topology: NetworkTopology {
                nodes: vec![],
                edges: vec![],
                graph_stats: GraphStats {
                    total_nodes: 50,
                    total_edges: 120,
                    chromatic_number: 7,
                    max_degree: 12,
                    avg_degree: 4.8,
                    clustering_coefficient: 0.65,
                },
            },
            optimization_state: OptimizationState {
                current_coloring: 10,
                best_coloring: 8,
                optimal_coloring: 7,
                iterations: 1000,
                convergence: 0.8,
                algorithm: "quantum_annealing".to_string(),
                time_elapsed_ms: 2000,
            },
            performance: NetworkPerformance {
                total_throughput_mbps: 5000.0,
                avg_latency_ms: 15.5,
                packet_loss_rate: 0.01,
                jitter_ms: 2.5,
                uptime: 0.999,
                quality_of_service: 0.95,
            },
        };

        let json = serde_json::to_string_pretty(&update).expect("Serialization failed");
        assert!(json.contains("\"network_topology\""));
        assert!(json.contains("\"optimization_state\""));
        assert!(json.contains("\"performance\""));

        let deserialized: TelecomUpdate = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.timestamp, 1234567890);
        assert_eq!(deserialized.network_topology.graph_stats.total_nodes, 50);
    }

    // =======================================================================
    // Dashboard #3: HFT Market Data Tests
    // =======================================================================

    #[test]
    fn test_hft_price_data_serialization() {
        let price = PriceData {
            symbol: "AAPL".to_string(),
            price: 175.50,
            volume: 1_000_000.0,
            bid: 175.48,
            ask: 175.52,
            high: 176.20,
            low: 174.80,
            open: 175.00,
            close: 175.50,
            change: 0.5,
            change_value: 0.50,
        };

        let json = serde_json::to_string(&price).expect("Serialization failed");
        assert!(json.contains("\"symbol\":\"AAPL\""));
        assert!(json.contains("\"price\":175.5"));

        let deserialized: PriceData = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.symbol, "AAPL");
        assert_eq!(deserialized.price, 175.50);
    }

    #[test]
    fn test_hft_order_book_serialization() {
        let order_book = OrderBook {
            symbol: "AAPL".to_string(),
            bids: vec![
                OrderLevel {
                    price: 175.48,
                    volume: 1000.0,
                    order_count: 10,
                },
                OrderLevel {
                    price: 175.47,
                    volume: 500.0,
                    order_count: 5,
                },
            ],
            asks: vec![
                OrderLevel {
                    price: 175.52,
                    volume: 800.0,
                    order_count: 8,
                },
                OrderLevel {
                    price: 175.53,
                    volume: 600.0,
                    order_count: 6,
                },
            ],
            spread: 0.04,
            depth: 2900.0,
            imbalance: 200.0,
        };

        let json = serde_json::to_string(&order_book).expect("Serialization failed");
        assert!(json.contains("\"symbol\":\"AAPL\""));
        assert!(json.contains("\"bids\""));
        assert!(json.contains("\"asks\""));

        let deserialized: OrderBook = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.symbol, "AAPL");
        assert_eq!(deserialized.bids.len(), 2);
        assert_eq!(deserialized.asks.len(), 2);
    }

    #[test]
    fn test_hft_transfer_entropy_signal_serialization() {
        let te_signal = TransferEntropySignal {
            source: "GOOGL".to_string(),
            target: "AAPL".to_string(),
            te_value: 0.75,
            lag: 250,
            significance: 0.01,
            causal_strength: 0.85,
        };

        let json = serde_json::to_string(&te_signal).expect("Serialization failed");
        assert!(json.contains("\"source\":\"GOOGL\""));
        assert!(json.contains("\"target\":\"AAPL\""));
        assert!(json.contains("\"causal_strength\":0.85"));

        let deserialized: TransferEntropySignal = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.source, "GOOGL");
        assert_eq!(deserialized.target, "AAPL");
        assert_eq!(deserialized.causal_strength, 0.85);
    }

    #[test]
    fn test_hft_full_market_update_serialization() {
        let market_update = MarketUpdate {
            timestamp: 1234567890,
            prices: vec![],
            order_book: OrderBook {
                symbol: "AAPL".to_string(),
                bids: vec![],
                asks: vec![],
                spread: 0.04,
                depth: 1000.0,
                imbalance: 0.0,
            },
            signals: TradingSignals {
                transfer_entropy: TransferEntropySignal {
                    source: "GOOGL".to_string(),
                    target: "AAPL".to_string(),
                    te_value: 0.75,
                    lag: 250,
                    significance: 0.01,
                    causal_strength: 0.85,
                },
                predicted_direction: "up".to_string(),
                confidence: 0.82,
                signal_strength: 0.75,
                risk_score: 0.18,
                recommended_action: "buy".to_string(),
                position_size: 1000.0,
            },
            execution: ExecutionMetrics {
                latency_us: 150,
                slippage_bps: 1.5,
                fill_rate: 0.95,
                reject_rate: 0.02,
                orders_per_second: 500.0,
                pnl: 5000.0,
                sharpe_ratio: 2.5,
            },
            portfolio: PortfolioState {
                cash: 50000.0,
                positions: vec![],
                total_value: 250000.0,
                daily_pnl: 2500.0,
                daily_pnl_pct: 1.0,
                max_drawdown: 0.05,
                win_rate: 0.65,
            },
        };

        let json = serde_json::to_string_pretty(&market_update).expect("Serialization failed");
        assert!(json.contains("\"order_book\""));
        assert!(json.contains("\"signals\""));
        assert!(json.contains("\"execution\""));
        assert!(json.contains("\"portfolio\""));

        let deserialized: MarketUpdate = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(deserialized.timestamp, 1234567890);
        assert_eq!(deserialized.signals.recommended_action, "buy");
        assert_eq!(deserialized.portfolio.cash, 50000.0);
    }

    // =======================================================================
    // Cross-Dashboard Validation Tests
    // =======================================================================

    #[test]
    fn test_validation_integration_pwsa() {
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

        // Validate before serialization
        let validation_result = validate_pwsa_telemetry(&telemetry);
        assert!(validation_result.valid, "Validation should pass");

        // Test serialization roundtrip
        let json = serde_json::to_string(&telemetry).expect("Serialization failed");
        let _deserialized: PwsaTelemetry = serde_json::from_str(&json).expect("Deserialization failed");

        // Validate after deserialization
        let validation_result2 = validate_pwsa_telemetry(&_deserialized);
        assert!(validation_result2.valid, "Validation should still pass after roundtrip");
    }

    #[test]
    fn test_json_format_compatibility() {
        // Test that JSON format matches TypeScript expectations
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

        let json: serde_json::Value = serde_json::to_value(&telemetry).expect("Serialization to Value failed");

        // Check JSON structure matches TypeScript interface
        assert!(json.get("timestamp").is_some());
        assert!(json.get("transport_layer").is_some());
        assert!(json.get("tracking_layer").is_some());
        assert!(json.get("ground_layer").is_some());
        assert!(json.get("mission_awareness").is_some());

        // Check nested structure
        let transport = json.get("transport_layer").unwrap();
        assert!(transport.get("satellites").is_some());
        assert!(transport.get("link_quality").is_some());
        assert!(transport.get("active_links").is_some());
        assert!(transport.get("constellation_health").is_some());
    }
}
