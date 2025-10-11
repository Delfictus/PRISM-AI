/**
 * Rust Type Definitions for All 4 Dashboards
 * Mirrors TypeScript interfaces in prism-web-platform/src/types/dashboards.ts
 * Uses serde for JSON serialization/deserialization
 */

use serde::{Deserialize, Serialize};

// =============================================================================
// Dashboard #1: Space Force Data Fusion (PWSA)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PwsaTelemetry {
    pub timestamp: i64,
    pub transport_layer: TransportLayer,
    pub tracking_layer: TrackingLayer,
    pub ground_layer: GroundLayer,
    pub mission_awareness: MissionAwareness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportLayer {
    pub satellites: Vec<SatelliteState>,
    pub link_quality: f64,
    pub active_links: u32,
    pub constellation_health: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatelliteState {
    pub id: u32,
    pub lat: f64,
    pub lon: f64,
    pub altitude: f64, // km
    pub layer: String, // "transport" or "tracking"
    pub status: String, // "healthy", "degraded", or "failed"
    pub velocity: f64, // km/s
    pub heading: f64, // degrees
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingLayer {
    pub threats: Vec<ThreatDetection>,
    pub sensor_coverage: Vec<GeoPolygon>,
    pub tracking_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetection {
    pub id: u32,
    pub class: String, // "none", "aircraft", "cruise", "ballistic", "hypersonic"
    pub probability: f64, // 0-1
    pub location: (f64, f64), // (lat, lon)
    pub velocity: f64, // km/s
    pub heading: f64, // degrees
    pub timestamp: i64,
    pub confidence: f64, // 0-1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoPolygon {
    pub id: String,
    pub coordinates: Vec<(f64, f64)>, // [(lat, lon), ...]
    #[serde(rename = "type")]
    pub poly_type: String, // "coverage", "threat_zone", "no_fly_zone"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundLayer {
    pub stations: Vec<GroundStation>,
    pub communication_links: Vec<CommunicationLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundStation {
    pub id: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub status: String, // "active", "standby", "offline"
    pub uplink_capacity: f64, // Mbps
    pub downlink_capacity: f64, // Mbps
    pub connected_satellites: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationLink {
    pub source: String, // satellite or station ID
    pub target: String,
    pub bandwidth: f64, // Mbps
    pub latency: f64, // ms
    pub packet_loss: f64, // 0-1
    pub status: String, // "active", "degraded", "failed"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionAwareness {
    pub transport_health: f64, // 0-1
    pub tracking_effectiveness: f64, // 0-1
    pub threat_status: Vec<f64>, // probabilities for each threat class
    pub coupling_matrix: Vec<Vec<f64>>, // transfer entropy between layers
    pub recommended_actions: Vec<RecommendedAction>,
    pub overall_mission_status: String, // "nominal", "degraded", "critical"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedAction {
    pub priority: String, // "low", "medium", "high", "critical"
    pub category: String, // "routing", "sensor_tasking", "threat_response", "maintenance"
    pub description: String,
    pub estimated_impact: f64, // 0-1
}

// =============================================================================
// Dashboard #2: Telecommunications & Logistics
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelecomUpdate {
    pub timestamp: i64,
    pub network_topology: NetworkTopology,
    pub optimization_state: OptimizationState,
    pub performance: NetworkPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    pub nodes: Vec<NetworkNode>,
    pub edges: Vec<NetworkEdge>,
    pub graph_stats: GraphStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkNode {
    pub id: String,
    pub label: String,
    pub x: f64,
    pub y: f64,
    pub color: u32, // assigned color in graph coloring
    pub degree: u32, // number of connections
    pub status: String, // "active", "degraded", "failed"
    pub load: f64, // 0-1
    pub capacity: f64, // Mbps
    pub node_type: String, // "router", "switch", "endpoint", "hub"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub utilization: f64, // 0-1
    pub bandwidth: f64, // Mbps
    pub latency: f64, // ms
    pub packet_loss: f64, // 0-1
    pub status: String, // "active", "congested", "failed"
    pub weight: f64, // for routing algorithms
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_nodes: u32,
    pub total_edges: u32,
    pub chromatic_number: u32, // minimum colors needed
    pub max_degree: u32,
    pub avg_degree: f64,
    pub clustering_coefficient: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationState {
    pub current_coloring: u32,
    pub best_coloring: u32,
    pub optimal_coloring: u32, // theoretical optimum
    pub iterations: u64,
    pub convergence: f64, // 0-1
    pub algorithm: String, // "dsatur", "greedy", "backtracking", "quantum_annealing"
    pub time_elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformance {
    pub total_throughput_mbps: f64,
    pub avg_latency_ms: f64,
    pub packet_loss_rate: f64, // 0-1
    pub jitter_ms: f64,
    pub uptime: f64, // 0-1
    pub quality_of_service: f64, // 0-1
}

// =============================================================================
// Dashboard #3: High-Frequency Trading
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketUpdate {
    pub timestamp: i64,
    pub prices: Vec<PriceData>,
    pub order_book: OrderBook,
    pub signals: TradingSignals,
    pub execution: ExecutionMetrics,
    pub portfolio: PortfolioState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceData {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub high: f64,
    pub low: f64,
    pub open: f64,
    pub close: f64,
    pub change: f64, // percentage
    pub change_value: f64, // absolute
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderLevel>,
    pub asks: Vec<OrderLevel>,
    pub spread: f64,
    pub depth: f64,
    pub imbalance: f64, // bid volume - ask volume
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLevel {
    pub price: f64,
    pub volume: f64,
    pub order_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignals {
    pub transfer_entropy: TransferEntropySignal,
    pub predicted_direction: String, // "up", "down", "neutral"
    pub confidence: f64, // 0-1
    pub signal_strength: f64, // 0-1
    pub risk_score: f64, // 0-1
    pub recommended_action: String, // "buy", "sell", "hold"
    pub position_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEntropySignal {
    pub source: String, // symbol
    pub target: String, // symbol
    pub te_value: f64,
    pub lag: u64, // time lag in ms
    pub significance: f64, // p-value
    pub causal_strength: f64, // 0-1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub latency_us: u64, // microseconds
    pub slippage_bps: f64, // basis points
    pub fill_rate: f64, // 0-1
    pub reject_rate: f64, // 0-1
    pub orders_per_second: f64,
    pub pnl: f64, // profit/loss
    pub sharpe_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    pub cash: f64,
    pub positions: Vec<Position>,
    pub total_value: f64,
    pub daily_pnl: f64,
    pub daily_pnl_pct: f64,
    pub max_drawdown: f64,
    pub win_rate: f64, // 0-1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub unrealized_pnl_pct: f64,
    pub market_value: f64,
}
