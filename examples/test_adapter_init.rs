//! Test just adapter initialization to find bottleneck

use prct_adapters::{NeuromorphicAdapter, QuantumAdapter, CouplingAdapter};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("\n=== Adapter Initialization Timing Test ===\n");

    // Test NeuromorphicAdapter
    println!("🧠 Creating NeuromorphicAdapter...");
    let start = Instant::now();
    let _neuro = NeuromorphicAdapter::new()?;
    println!("   ✓ Took: {:?}", start.elapsed());

    // Test QuantumAdapter
    println!("\n⚛️  Creating QuantumAdapter...");
    let start = Instant::now();
    let _quantum = QuantumAdapter::new();
    println!("   ✓ Took: {:?}", start.elapsed());

    // Test CouplingAdapter
    println!("\n🔗 Creating CouplingAdapter...");
    let start = Instant::now();
    let _coupling = CouplingAdapter::new();
    println!("   ✓ Took: {:?}", start.elapsed());

    println!("\n✅ All adapters initialized successfully!");

    Ok(())
}
