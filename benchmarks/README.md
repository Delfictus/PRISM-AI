# DIMACS Graph Coloring Benchmarks

This directory contains graph coloring benchmark files in DIMACS `.col` format for testing the ARES-51 chromatic coloring algorithm.

## File Format

DIMACS `.col` files use the following format:

```
c <comment line>
p edge <num_vertices> <num_edges>
e <vertex1> <vertex2>
e <vertex1> <vertex2>
...
```

- Lines starting with `c` are comments
- The `p` line defines the problem: `p edge V E` where V = vertices, E = edges
- `e` lines define edges (1-indexed in DIMACS format)

## Benchmark Files

### Small Test Graphs

**`test_small.col`** - 5 vertices, complete graph K5
- Expected chromatic number: χ = 5
- Purpose: Quick validation test

**`test_medium.col`** - 50 vertices, random graph
- Expected chromatic number: χ ≈ 10-15
- Purpose: Medium-scale testing

### Standard DIMACS Benchmarks

**`dsjc500.5.col`** - Johnson graph DSJC500.5
- 500 vertices, ~62,624 edges (density ~50%)
- Known best result: χ ≈ 48
- Challenge: Large, dense graph

**`dsjc250.5.col`** - Johnson graph DSJC250.5
- 250 vertices, ~15,668 edges (density ~50%)
- Known best result: χ ≈ 28
- Challenge: Medium-large graph

## Downloading Benchmark Files

Standard DIMACS benchmarks can be downloaded from:
- https://mat.tepper.cmu.edu/COLOR/instances.html
- https://iridia.ulb.ac.be/~fmascia/maximum_clique/DIMACS-benchmark

## Running Benchmarks

```bash
# Run small test
cargo run --release --example benchmark_small

# Run DSJC500.5 benchmark
cargo run --release --example benchmark_dsjc500_5

# Run full benchmark suite
cargo run --release --example benchmark_suite
```

## Creating Custom Benchmarks

You can create custom `.col` files:

```bash
# Example: 4-vertex cycle graph (chromatic number = 2)
cat > benchmarks/cycle4.col << EOF
c 4-vertex cycle graph
p edge 4 4
e 1 2
e 2 3
e 3 4
e 4 1
EOF
```
