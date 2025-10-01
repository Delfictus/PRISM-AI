//! Test Hamiltonian construction with tiny graph

use quantum_engine::{Hamiltonian, ForceFieldParams};
use ndarray::{Array1, Array2};

fn main() -> anyhow::Result<()> {
    println!("\n=== Tiny Hamiltonian Test ===\n");

    let n = 3;
    println!("Creating Hamiltonian for {} atoms...", n);

    // Simple positions
    let positions = Array2::from_shape_fn((n, 3), |(i, dim)| {
        if dim == 0 { i as f64 } else { 0.0 }
    });

    let masses = Array1::from_elem(n, 1.0);
    let force_field = ForceFieldParams::new();

    println!("Calling Hamiltonian::new...");
    let hamiltonian = Hamiltonian::new(positions, masses, force_field)?;

    println!("✓ Success! n_atoms={}", hamiltonian.n_atoms());

    Ok(())
}
