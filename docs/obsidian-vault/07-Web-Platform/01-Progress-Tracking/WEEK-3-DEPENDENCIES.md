# Week 3 Enhancement Dependencies

## Required Cargo.toml Dependencies

Add these dependencies to your `Cargo.toml` file:

```toml
[dependencies]
# Core async runtime
tokio = { version = "1.35", features = ["full"] }
actix-web = "4.4"
actix-web-actors = "4.2"
actix = "0.13"
actix-cors = "0.7"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# MessagePack compression (Week 3 Enhancement #6)
rmp-serde = "1.1"
flate2 = "1.0"

# Async utilities
async-trait = "0.1"

# Random number generation (for simulated data)
rand = "0.8"

# UUID generation (for event sourcing)
uuid = { version = "1.6", features = ["v4", "serde"] }

# Optional: Real NVML support (Week 3 Enhancement #5)
# Uncomment if you have NVIDIA GPUs and want real GPU metrics
# nvml-wrapper = { version = "0.9", optional = true }

[features]
# Enable this feature to use real NVML instead of simulated GPU metrics
nvml = ["nvml-wrapper"]
```

## Module-Specific Dependencies

### 1. Plugin Architecture
- `tokio` (async runtime)
- `async-trait` (async trait support)
- `serde` (serialization)

### 2. Event Sourcing
- `tokio` (RwLock for concurrent access)
- `serde` (event serialization)
- `uuid` (event IDs)

### 3. SGP4 Orbital Mechanics
- `serde` (coordinate serialization)
- **No external dependencies** - pure Rust implementation

### 4. Transfer Entropy
- `rand` (permutation testing)
- `serde` (results serialization)
- **No external dependencies** - pure Shannon entropy implementation

### 5. NVML GPU Metrics
- `tokio` (async collection)
- `rand` (simulated data)
- `nvml-wrapper` (optional, for real NVIDIA GPU support)

### 6. MessagePack Compression
- `rmp-serde` (MessagePack codec)
- `flate2` (gzip/zlib compression)
- `actix-web-actors` (WebSocket integration)

## Installation

```bash
# Add dependencies to Cargo.toml, then run:
cargo build

# To enable real NVML support (requires NVIDIA drivers):
cargo build --features nvml
```

## Platform-Specific Notes

### Linux
- NVML support requires NVIDIA drivers installed
- No additional setup needed for other features

### macOS
- NVML not available (no NVIDIA GPUs on modern Macs)
- Will use simulated GPU metrics automatically

### Windows
- NVML support requires NVIDIA drivers
- May need Visual Studio Build Tools for compilation

## Testing

```bash
# Run all tests
cargo test

# Run tests for specific module
cargo test plugin::
cargo test event_sourcing::
cargo test orbital::
cargo test transfer_entropy::
cargo test gpu_metrics::
cargo test messagepack::
```

## Documentation

Generate documentation for all modules:

```bash
cargo doc --open --no-deps
```

This will open comprehensive API documentation in your browser.
