//! Debug script to show exactly what's wrong with the baselines

use anyhow::Result;
use ndarray::Array2;
use std::collections::HashSet;

// Copy of DSATUR implementation
struct DSaturSolver {
    pub num_vertices: usize,
    pub adjacency: Array2<bool>,
}

impl DSaturSolver {
    pub fn new(adjacency: Array2<bool>) -> Self {
        let num_vertices = adjacency.nrows();
        Self { num_vertices, adjacency }
    }

    pub fn solve(&self, max_colors: usize) -> Result<(Vec<usize>, usize)> {
        let n = self.num_vertices;
        if n == 0 {
            return Ok((Vec::new(), 0));
        }

        let mut coloring = vec![usize::MAX; n];
        let mut uncolored: HashSet<usize> = (0..n).collect();

        // Color first vertex with highest degree
        let first = self.find_max_degree_vertex(&uncolored);
        coloring[first] = 0;
        uncolored.remove(&first);

        while !uncolored.is_empty() {
            let v = self.find_max_saturation_vertex(&uncolored, &coloring);

            let used_colors: HashSet<usize> = (0..n)
                .filter(|&u| u != v && coloring[u] != usize::MAX && self.adjacency[[v, u]])
                .map(|u| coloring[u])
                .collect();

            let color = (0..max_colors)
                .find(|c| !used_colors.contains(c))
                .ok_or_else(|| anyhow::anyhow!("Not enough colors"))?;

            coloring[v] = color;
            uncolored.remove(&v);
        }

        let num_colors = coloring.iter().max().map(|&c| c + 1).unwrap_or(0);
        Ok((coloring, num_colors))
    }

    fn find_max_degree_vertex(&self, uncolored: &HashSet<usize>) -> usize {
        let mut max_degree = 0;
        let mut max_vertex = 0;

        for &v in uncolored {
            let degree = (0..self.num_vertices)
                .filter(|&u| u != v && self.adjacency[[v, u]])
                .count();

            if degree > max_degree {
                max_degree = degree;
                max_vertex = v;
            }
        }
        max_vertex
    }

    fn find_max_saturation_vertex(&self, uncolored: &HashSet<usize>, coloring: &[usize]) -> usize {
        let n = self.num_vertices;
        let mut max_saturation = 0;
        let mut max_degree = 0;
        let mut best_vertex = *uncolored.iter().next().unwrap();

        for &v in uncolored {
            let neighbor_colors: HashSet<usize> = (0..n)
                .filter(|&u| u != v && coloring[u] != usize::MAX && self.adjacency[[v, u]])
                .map(|u| coloring[u])
                .collect();

            let saturation = neighbor_colors.len();
            let degree = (0..n).filter(|&u| u != v && self.adjacency[[v, u]]).count();

            if saturation > max_saturation || (saturation == max_saturation && degree > max_degree) {
                max_saturation = saturation;
                max_degree = degree;
                best_vertex = v;
            }
        }
        best_vertex
    }

    pub fn validate(&self, coloring: &[usize]) -> Result<bool> {
        for i in 0..self.num_vertices {
            for j in (i+1)..self.num_vertices {
                if self.adjacency[[i, j]] && coloring[i] == coloring[j] {
                    println!("❌ CONFLICT: Vertices {} and {} both have color {}", i, j, coloring[i]);
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  BASELINE DEBUGGING - Finding the Issues");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // TEST 1: Triangle (K3) - MUST need 3 colors
    println!("TEST 1: Triangle (K3)");
    println!("───────────────────────────────────────────────────────────────────");
    let mut triangle = Array2::from_elem((3, 3), false);
    triangle[[0, 1]] = true; triangle[[1, 0]] = true;
    triangle[[1, 2]] = true; triangle[[2, 1]] = true;
    triangle[[2, 0]] = true; triangle[[0, 2]] = true;

    println!("Graph: Complete triangle (every vertex connected to every other)");
    println!("Expected: χ = 3 (it's a clique, needs one color per vertex)");

    let solver = DSaturSolver::new(triangle);
    let (coloring, num_colors) = solver.solve(10)?;

    println!("DSATUR found: χ = {} with coloring {:?}", num_colors, coloring);
    let valid = solver.validate(&coloring)?;
    println!("Valid coloring? {}", if valid { "✅ YES" } else { "❌ NO" });

    if num_colors == 3 {
        println!("✅ CORRECT\n");
    } else {
        println!("❌ WRONG - Should be 3!\n");
    }

    // TEST 2: Square (C4) - MUST need 2 colors
    println!("TEST 2: Square Cycle (C4)");
    println!("───────────────────────────────────────────────────────────────────");
    let mut square = Array2::from_elem((4, 4), false);
    square[[0, 1]] = true; square[[1, 0]] = true;
    square[[1, 2]] = true; square[[2, 1]] = true;
    square[[2, 3]] = true; square[[3, 2]] = true;
    square[[3, 0]] = true; square[[0, 3]] = true;

    println!("Graph: 0-1-2-3-0 (cycle)");
    println!("Expected: χ = 2 (bipartite: color 0,2 one color, 1,3 another)");

    let solver = DSaturSolver::new(square);
    let (coloring, num_colors) = solver.solve(10)?;

    println!("DSATUR found: χ = {} with coloring {:?}", num_colors, coloring);
    let valid = solver.validate(&coloring)?;
    println!("Valid coloring? {}", if valid { "✅ YES" } else { "❌ NO" });

    if num_colors == 2 {
        println!("✅ CORRECT\n");
    } else {
        println!("❌ WRONG - Should be 2!\n");
    }

    // TEST 3: Small DIMACS graph - load dsjc125.1
    println!("TEST 3: dsjc125.1 (Real DIMACS benchmark)");
    println!("───────────────────────────────────────────────────────────────────");

    if let Ok((vertices, edges)) = parse_col_file("benchmarks/dsjc125.1.col") {
        println!("Loaded: {} vertices, {} edges", vertices, edges.len());
        println!("Known best: χ = 5 (from DIMACS website)");

        let adjacency = edges_to_adjacency(vertices, &edges);
        let solver = DSaturSolver::new(adjacency);
        let (coloring, num_colors) = solver.solve(vertices)?;

        println!("DSATUR found: χ = {}", num_colors);
        let valid = solver.validate(&coloring)?;
        println!("Valid coloring? {}", if valid { "✅ YES" } else { "❌ NO" });

        if num_colors <= 7 {
            println!("✅ REASONABLE (optimal is 5, DSATUR typically finds 5-7)\n");
        } else {
            println!("❌ SUSPICIOUS - Should be ~5-7, found {}\n", num_colors);
        }
    } else {
        println!("⚠️  Could not load benchmarks/dsjc125.1.col\n");
    }

    // TEST 4: Show the TSP distance scale problem
    println!("TEST 4: TSP Distance Scale Mismatch");
    println!("───────────────────────────────────────────────────────────────════");
    println!("Problem: Classical uses REAL distances, GPU uses COUPLING matrix\n");

    println!("Example: 3 cities in a line");
    println!("  City 0 at (0, 0)");
    println!("  City 1 at (100, 0)");
    println!("  City 2 at (200, 0)");
    println!();

    let distances = vec![
        vec![0.0, 100.0, 200.0],
        vec![100.0, 0.0, 100.0],
        vec![200.0, 100.0, 0.0],
    ];

    println!("Classical distance matrix:");
    for row in &distances {
        println!("  {:?}", row);
    }
    println!();

    println!("Optimal tour: 0 → 1 → 2 → 0");
    println!("  Classical length: 100 + 100 + 200 = 400.0");
    println!();

    println!("GPU coupling matrix (1/(dist+0.1)):");
    for i in 0..3 {
        print!("  [");
        for j in 0..3 {
            if i == j {
                print!("  0.000, ");
            } else {
                let coupling = 1.0 / (distances[i][j] + 0.1);
                print!("{:6.3}, ", coupling);
            }
        }
        println!("]");
    }
    println!();

    println!("Same tour in coupling space: 0 → 1 → 2 → 0");
    let coupling_length = 1.0/(100.0+0.1) + 1.0/(100.0+0.1) + 1.0/(200.0+0.1);
    println!("  Coupling length: {:.6}", coupling_length);
    println!();
    println!("❌ Cannot compare 400.0 to 0.015 directly!");
    println!("✅ Fixed by converting classical tour to coupling space\n");

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Run this to see if DSATUR implementation is correct on simple graphs.");
    println!("If tests 1 & 2 pass, DSATUR is correct.");
    println!("If test 3 shows χ >> 7, something is wrong with edge parsing.");

    Ok(())
}

fn parse_col_file(path: &str) -> Result<(usize, Vec<(usize, usize)>)> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut vertices = 0;
    let mut edges = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts.get(0) {
            Some(&"p") => {
                if parts.len() >= 3 && parts[1] == "edge" {
                    vertices = parts[2].parse()?;
                }
            }
            Some(&"e") => {
                if parts.len() >= 3 {
                    let u: usize = parts[1].parse()?;
                    let v: usize = parts[2].parse()?;
                    edges.push((u - 1, v - 1));
                }
            }
            _ => {}
        }
    }

    Ok((vertices, edges))
}

fn edges_to_adjacency(vertices: usize, edges: &[(usize, usize)]) -> Array2<bool> {
    let mut adj = Array2::from_elem((vertices, vertices), false);
    for &(u, v) in edges {
        if u < vertices && v < vertices {
            adj[[u, v]] = true;
            adj[[v, u]] = true;
        }
    }
    adj
}
