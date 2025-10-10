# HFT Backtesting Demo - PRISM-AI

High-Frequency Trading backtesting demonstration using neuromorphic-quantum prediction strategies with GPU acceleration.

## Overview

This demo showcases:
- **Realistic market simulation** with tick-level data
- **Neuromorphic spike encoding** for price prediction
- **GPU-accelerated backtesting** processing years of data in minutes
- **Real-time visualization** of trading strategy performance
- **Comprehensive metrics** including Sharpe ratio, drawdown, win rate

## Quick Start

```bash
# Build the project
cd hft-demo
cargo build --release

# Generate sample data
cargo run --release --bin generate-sample-data

# Run backtest (coming in Phase 3)
cargo run --release --bin hft-demo-server
```

## Project Structure

```
hft-demo/
├── src/
│   ├── lib.rs                   # Library root
│   ├── market_data/             # Phase 1: Market Data Engine
│   │   ├── mod.rs
│   │   ├── loader.rs            # Data loading (Task 1.1)
│   │   ├── simulator.rs         # Market simulation (Task 1.2)
│   │   ├── features.rs          # Feature extraction (Task 1.3)
│   │   └── validation.rs        # Data validation (Task 1.4)
│   ├── strategy/                # Phase 2: Trading Strategy
│   ├── backtest/                # Phase 3: Backtesting Engine
│   └── bin/
│       ├── server.rs            # Web server
│       └── generate_sample_data.rs
├── frontend/                    # Phase 4: Web UI
├── data/                        # Sample market data
└── tests/                       # Integration tests
```

## Implementation Status

### Phase 1: Market Data Engine (In Progress)
- [x] Project setup (Task 1.1.1) ✅
- [ ] Data structures (Task 1.1.2)
- [ ] CSV loader (Task 1.1.3)
- [ ] Alpaca API (Task 1.1.4)
- [ ] Data caching (Task 1.1.5)
- [ ] Sample data generation (Task 1.1.6)
- [ ] Unit tests (Task 1.1.7)

### Phase 2: Neuromorphic Trading Strategy (Pending)
- [ ] Spike encoding
- [ ] Neural network
- [ ] Trade execution

### Phase 3: Backtesting Engine (Pending)
- [ ] Backtest runner
- [ ] Performance metrics
- [ ] Strategy comparison

### Phase 4: Web Interface (Pending)
- [ ] Backend API
- [ ] Real-time dashboard
- [ ] Interactive charts

## ARES Anti-Drift Compliance

This implementation strictly follows ARES standards to ensure all metrics are computed from actual data rather than hardcoded values.

### Forbidden Patterns
```rust
// ❌ NEVER do this
pub fn sharpe_ratio(&self) -> f64 { 2.4 }  // Hardcoded!
```

### Required Patterns
```rust
// ✅ Always do this
pub fn sharpe_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64 {
    let excess_returns: Vec<f64> = returns.iter()
        .map(|r| r - risk_free_rate / 252.0)
        .collect();
    let mean = excess_returns.mean();
    let std = excess_returns.std_dev();
    mean / std * (252.0_f64).sqrt()  // COMPUTED from actual data
}
```

## Testing

```bash
# Run all tests
cargo test

# Run with anti-drift validation enabled
HFT_DEMO_STRICT_VALIDATION=1 cargo test

# Run benchmarks
cargo bench
```

## Documentation

See detailed documentation in `docs/obsidian-vault/PRISM-AI/04-Development/`:
- **HFT Backtesting Demo Plan.md** - Overall plan
- **HFT Demo Phase 1 Detailed Tasks.md** - Current phase breakdown

## License

MIT License - See LICENSE file for details.

## Contact

PRISM-AI Research Team
