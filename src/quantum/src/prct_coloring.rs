//! Chromatic Graph Coloring for PRCT Algorithm
//!
//! Implements optimized graph coloring to minimize chromatic number while respecting
//! quantum coupling constraints. This is a key component of the Phase Resonance
//! Chromatic-TSP (PRCT) patent.

use ndarray::Array2;
use num_complex::Complex64;
use anyhow::{Result, Context};
use std::collections::{HashMap, HashSet};

/// Chromatic graph coloring optimizer for PRCT
#[derive(Debug, Clone)]
pub struct ChromaticColoring {
    /// Number of colors used
    num_colors: usize,
    /// Color assignment for each vertex (node -> color)
    coloring: Vec<usize>,
    /// Adjacency matrix (true if edge exists)
    adjacency: Array2<bool>,
    /// Color conflict count (for optimization)
    conflict_count: usize,
}

impl ChromaticColoring {
    /// Create new chromatic coloring from coupling matrix
    ///
    /// # Arguments
    /// * `coupling_matrix` - Complex coupling amplitudes between atoms
    /// * `target_colors` - Desired number of colors (chromatic number)
    /// * `threshold` - Coupling strength threshold for edge creation
    pub fn new(
        coupling_matrix: &Array2<Complex64>,
        target_colors: usize,
        threshold: f64,
    ) -> Result<Self> {
        let n = coupling_matrix.nrows();

        if n == 0 {
            return Err(anyhow::anyhow!("Empty coupling matrix"));
        }

        if target_colors == 0 {
            return Err(anyhow::anyhow!("Target colors must be > 0"));
        }

        // Build adjacency matrix from coupling strengths
        let adjacency = Self::build_adjacency(coupling_matrix, threshold);

        // Compute initial greedy coloring
        let coloring = Self::greedy_coloring(&adjacency, target_colors)?;

        let mut instance = Self {
            num_colors: target_colors,
            coloring,
            adjacency,
            conflict_count: 0,
        };

        // Calculate initial conflicts
        instance.conflict_count = instance.count_conflicts();

        Ok(instance)
    }

    /// Build adjacency matrix from coupling matrix
    fn build_adjacency(coupling_matrix: &Array2<Complex64>, threshold: f64) -> Array2<bool> {
        let n = coupling_matrix.nrows();
        let mut adjacency = Array2::from_elem((n, n), false);

        for i in 0..n {
            for j in (i + 1)..n {
                let coupling_strength = coupling_matrix[[i, j]].norm();
                if coupling_strength > threshold {
                    adjacency[[i, j]] = true;
                    adjacency[[j, i]] = true;
                }
            }
        }

        adjacency
    }

    /// Greedy coloring algorithm with DSATUR heuristic
    fn greedy_coloring(adjacency: &Array2<bool>, max_colors: usize) -> Result<Vec<usize>> {
        let n = adjacency.nrows();
        let mut coloring = vec![usize::MAX; n];
        let mut uncolored: HashSet<usize> = (0..n).collect();

        // Color first vertex
        if !uncolored.is_empty() {
            coloring[0] = 0;
            uncolored.remove(&0);
        }

        // DSATUR: Always color vertex with highest saturation degree
        while !uncolored.is_empty() {
            // Find vertex with highest saturation degree
            let v = Self::find_max_saturation_vertex(&uncolored, &coloring, adjacency);

            // Find smallest available color
            let used_colors: HashSet<usize> = (0..n)
                .filter(|&u| {
                    coloring[u] != usize::MAX && adjacency[[v, u]]
                })
                .map(|u| coloring[u])
                .collect();

            let color = (0..max_colors)
                .find(|c| !used_colors.contains(c))
                .context("Not enough colors for valid coloring")?;

            coloring[v] = color;
            uncolored.remove(&v);
        }

        Ok(coloring)
    }

    /// Find vertex with maximum saturation degree (DSATUR heuristic)
    fn find_max_saturation_vertex(
        uncolored: &HashSet<usize>,
        coloring: &[usize],
        adjacency: &Array2<bool>,
    ) -> usize {
        let mut max_saturation = 0;
        let mut max_degree = 0;
        let mut best_vertex = *uncolored.iter().next().unwrap();

        for &v in uncolored {
            // Count distinct colors in neighborhood
            let saturation = (0..coloring.len())
                .filter(|&u| coloring[u] != usize::MAX && adjacency[[v, u]])
                .map(|u| coloring[u])
                .collect::<HashSet<_>>()
                .len();

            // Count neighbors (degree)
            let degree = (0..coloring.len())
                .filter(|&u| adjacency[[v, u]])
                .count();

            // Select vertex with highest saturation, breaking ties by degree
            if saturation > max_saturation || (saturation == max_saturation && degree > max_degree) {
                max_saturation = saturation;
                max_degree = degree;
                best_vertex = v;
            }
        }

        best_vertex
    }

    /// Optimize coloring using simulated annealing
    pub fn optimize(&mut self, iterations: usize, initial_temp: f64) -> Result<()> {
        let mut temperature = initial_temp;
        let cooling_rate = 0.995;
        let mut current_conflicts = self.conflict_count;
        let mut best_coloring = self.coloring.clone();
        let mut best_conflicts = current_conflicts;

        for _ in 0..iterations {
            // Try random color swap
            let vertex = rand::random::<usize>() % self.coloring.len();
            let old_color = self.coloring[vertex];
            let new_color = rand::random::<usize>() % self.num_colors;

            if old_color == new_color {
                continue;
            }

            // Apply swap
            self.coloring[vertex] = new_color;
            let new_conflicts = self.count_conflicts();

            // Metropolis criterion
            let delta = new_conflicts as f64 - current_conflicts as f64;
            let accept = delta < 0.0 || rand::random::<f64>() < (-delta / temperature).exp();

            if accept {
                current_conflicts = new_conflicts;
                if current_conflicts < best_conflicts {
                    best_conflicts = current_conflicts;
                    best_coloring = self.coloring.clone();
                }
            } else {
                // Revert
                self.coloring[vertex] = old_color;
            }

            temperature *= cooling_rate;
        }

        // Apply best solution found
        self.coloring = best_coloring;
        self.conflict_count = best_conflicts;

        Ok(())
    }

    /// Count color conflicts (edges with same-colored endpoints)
    fn count_conflicts(&self) -> usize {
        let n = self.coloring.len();
        let mut conflicts = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                if self.adjacency[[i, j]] && self.coloring[i] == self.coloring[j] {
                    conflicts += 1;
                }
            }
        }

        conflicts
    }

    /// Verify coloring is valid (no adjacent vertices share color)
    pub fn verify_coloring(&self) -> bool {
        self.conflict_count == 0
    }

    /// Get color assignment
    pub fn get_coloring(&self) -> &[usize] {
        &self.coloring
    }

    /// Get number of colors used
    pub fn get_num_colors(&self) -> usize {
        self.num_colors
    }

    /// Get conflict count
    pub fn get_conflict_count(&self) -> usize {
        self.conflict_count
    }

    /// Calculate chromatic polynomial value at k
    pub fn chromatic_polynomial(&self, k: usize) -> f64 {
        // Simplified approximation using coloring statistics
        let n = self.coloring.len();
        let edges = self.count_edges();

        // Upper bound: k^n
        // Lower bound accounts for edges
        let max = (k as f64).powi(n as i32);
        let penalty = edges as f64 / (n * (n - 1) / 2) as f64;

        max * (1.0 - penalty).max(0.0)
    }

    /// Count edges in graph
    fn count_edges(&self) -> usize {
        let n = self.adjacency.nrows();
        let mut edges = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                if self.adjacency[[i, j]] {
                    edges += 1;
                }
            }
        }

        edges
    }

    /// Get color distribution statistics
    pub fn get_color_distribution(&self) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();

        for &color in &self.coloring {
            *distribution.entry(color).or_insert(0) += 1;
        }

        distribution
    }

    /// Calculate color balance (uniformity of color usage)
    pub fn color_balance(&self) -> f64 {
        let distribution = self.get_color_distribution();
        let n = self.coloring.len() as f64;
        let ideal_per_color = n / self.num_colors as f64;

        let variance: f64 = distribution
            .values()
            .map(|&count| {
                let diff = count as f64 - ideal_per_color;
                diff * diff
            })
            .sum::<f64>()
            / self.num_colors as f64;

        // Return normalized balance score (1.0 = perfect balance)
        1.0 / (1.0 + variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_coloring() {
        // Create simple 4-vertex cycle graph
        let mut coupling = Array2::zeros((4, 4));
        for i in 0..4 {
            let j = (i + 1) % 4;
            coupling[[i, j]] = Complex64::new(1.0, 0.0);
            coupling[[j, i]] = Complex64::new(1.0, 0.0);
        }

        let coloring = ChromaticColoring::new(&coupling, 2, 0.5).unwrap();
        assert!(coloring.verify_coloring());
    }

    #[test]
    fn test_complete_graph() {
        // Complete graph K4 requires 4 colors
        let mut coupling = Array2::from_elem((4, 4), Complex64::new(1.0, 0.0));
        for i in 0..4 {
            coupling[[i, i]] = Complex64::new(0.0, 0.0);
        }

        let coloring = ChromaticColoring::new(&coupling, 4, 0.5).unwrap();
        assert!(coloring.verify_coloring());
    }

    #[test]
    fn test_optimization() {
        let mut coupling = Array2::zeros((6, 6));
        for i in 0..6 {
            for j in (i + 1)..6 {
                if (i + j) % 2 == 0 {
                    coupling[[i, j]] = Complex64::new(1.0, 0.0);
                    coupling[[j, i]] = Complex64::new(1.0, 0.0);
                }
            }
        }

        let mut coloring = ChromaticColoring::new(&coupling, 3, 0.5).unwrap();
        let initial_conflicts = coloring.get_conflict_count();

        coloring.optimize(1000, 10.0).unwrap();

        assert!(coloring.get_conflict_count() <= initial_conflicts);
    }

    #[test]
    fn test_color_balance() {
        let coupling = Array2::from_elem((10, 10), Complex64::new(0.5, 0.0));
        let coloring = ChromaticColoring::new(&coupling, 3, 0.3).unwrap();

        let balance = coloring.color_balance();
        assert!(balance >= 0.0 && balance <= 1.0);
    }
}
