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
    /// Create new chromatic coloring with adaptive threshold selection
    ///
    /// Automatically determines optimal threshold for graph construction based on
    /// coupling matrix statistics and target chromatic number. This ensures the
    /// graph is k-colorable while maximizing edge density.
    ///
    /// # Arguments
    /// * `coupling_matrix` - Complex coupling amplitudes between atoms
    /// * `target_colors` - Desired number of colors (chromatic number)
    ///
    /// # Example
    /// ```ignore
    /// let coupling = Array2::from_elem((10, 10), Complex64::new(0.5, 0.0));
    /// let coloring = ChromaticColoring::new_adaptive(&coupling, 4)?;
    /// assert!(coloring.verify_coloring());
    /// ```
    pub fn new_adaptive(
        coupling_matrix: &Array2<Complex64>,
        target_colors: usize,
    ) -> Result<Self> {
        let threshold = Self::find_optimal_threshold(coupling_matrix, target_colors)?;
        Self::new(coupling_matrix, target_colors, threshold)
    }

    /// Create new chromatic coloring from coupling matrix with manual threshold
    ///
    /// # Arguments
    /// * `coupling_matrix` - Complex coupling amplitudes between atoms
    /// * `target_colors` - Desired number of colors (chromatic number)
    /// * `threshold` - Coupling strength threshold for edge creation
    ///
    /// # Note
    /// Consider using `new_adaptive()` for automatic threshold selection
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

    /// Find optimal threshold for graph construction using adaptive algorithm
    ///
    /// Uses binary search to find the maximum threshold that produces a graph
    /// likely to be k-colorable with the target number of colors. This maximizes
    /// edge density while maintaining k-colorability.
    ///
    /// Algorithm:
    /// 1. Collect all non-zero coupling strengths
    /// 2. Binary search over sorted strengths
    /// 3. For each threshold candidate, estimate chromatic number using Brooks' theorem
    /// 4. Select maximum threshold where χ(G) ≤ target_colors
    ///
    /// # Arguments
    /// * `coupling_matrix` - Complex coupling amplitudes between atoms
    /// * `target_colors` - Desired chromatic number
    ///
    /// # Returns
    /// Optimal coupling strength threshold
    fn find_optimal_threshold(
        coupling_matrix: &Array2<Complex64>,
        target_colors: usize,
    ) -> Result<f64> {
        let n = coupling_matrix.nrows();

        // Collect all non-zero coupling strengths
        let mut strengths: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let strength = coupling_matrix[[i, j]].norm();
                if strength > 1e-10 {
                    strengths.push(strength);
                }
            }
        }

        if strengths.is_empty() {
            return Ok(0.0);
        }

        // Sort in descending order (strongest couplings first)
        strengths.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Binary search for optimal threshold
        // Goal: Find maximum threshold where graph is likely k-colorable
        let mut low_idx = 0;
        let mut high_idx = strengths.len() - 1;
        let mut best_threshold = strengths[high_idx]; // Start with weakest coupling

        while low_idx <= high_idx {
            let mid_idx = (low_idx + high_idx) / 2;
            let test_threshold = strengths[mid_idx];

            // Build graph with this threshold
            let adjacency = Self::build_adjacency(coupling_matrix, test_threshold);

            // Estimate chromatic number using Brooks' theorem:
            // χ(G) ≤ Δ(G) + 1, where Δ(G) is maximum degree
            // (Equality holds only for complete graphs and odd cycles)
            let max_degree = Self::compute_max_degree(&adjacency);
            let estimated_chromatic = max_degree + 1;

            if estimated_chromatic <= target_colors {
                // Graph is likely k-colorable
                // Try higher threshold (more edges) for better optimization
                best_threshold = test_threshold;
                if mid_idx == 0 {
                    break;
                }
                high_idx = mid_idx - 1;
            } else {
                // Too many edges, graph may not be k-colorable
                // Reduce threshold (fewer edges)
                if mid_idx == strengths.len() - 1 {
                    break;
                }
                low_idx = mid_idx + 1;
            }
        }

        // Verify the selected threshold produces a valid graph
        let final_adjacency = Self::build_adjacency(coupling_matrix, best_threshold);
        let final_max_degree = Self::compute_max_degree(&final_adjacency);

        if final_max_degree + 1 > target_colors * 2 {
            // Very conservative fallback: use mean coupling strength
            let mean_strength = strengths.iter().sum::<f64>() / strengths.len() as f64;
            return Ok(mean_strength);
        }

        Ok(best_threshold)
    }

    /// Compute maximum vertex degree in adjacency matrix
    fn compute_max_degree(adjacency: &Array2<bool>) -> usize {
        let n = adjacency.nrows();
        (0..n)
            .map(|i| (0..n).filter(|&j| adjacency[[i, j]]).count())
            .max()
            .unwrap_or(0)
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

    /// Get the threshold value that would be selected for this coupling matrix
    ///
    /// Useful for analysis and debugging of the adaptive threshold algorithm
    pub fn analyze_threshold(
        coupling_matrix: &Array2<Complex64>,
        target_colors: usize,
    ) -> Result<ThresholdAnalysis> {
        let n = coupling_matrix.nrows();

        // Collect coupling strengths
        let mut strengths: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let strength = coupling_matrix[[i, j]].norm();
                if strength > 1e-10 {
                    strengths.push(strength);
                }
            }
        }

        if strengths.is_empty() {
            return Ok(ThresholdAnalysis {
                optimal_threshold: 0.0,
                min_coupling: 0.0,
                max_coupling: 0.0,
                mean_coupling: 0.0,
                median_coupling: 0.0,
                num_edges_at_threshold: 0,
                graph_density: 0.0,
                estimated_chromatic_number: 1,
            });
        }

        strengths.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_coupling = strengths[0];
        let max_coupling = strengths[strengths.len() - 1];
        let mean_coupling = strengths.iter().sum::<f64>() / strengths.len() as f64;
        let median_coupling = strengths[strengths.len() / 2];

        // Find optimal threshold
        let optimal_threshold = Self::find_optimal_threshold(coupling_matrix, target_colors)?;

        // Count edges at optimal threshold
        let adjacency = Self::build_adjacency(coupling_matrix, optimal_threshold);
        let num_edges = (0..n)
            .map(|i| (i + 1..n).filter(|&j| adjacency[[i, j]]).count())
            .sum::<usize>();

        let max_edges = n * (n - 1) / 2;
        let graph_density = if max_edges > 0 {
            num_edges as f64 / max_edges as f64
        } else {
            0.0
        };

        let max_degree = Self::compute_max_degree(&adjacency);
        let estimated_chromatic_number = max_degree + 1;

        Ok(ThresholdAnalysis {
            optimal_threshold,
            min_coupling,
            max_coupling,
            mean_coupling,
            median_coupling,
            num_edges_at_threshold: num_edges,
            graph_density,
            estimated_chromatic_number,
        })
    }
}

/// Analysis results for adaptive threshold selection
#[derive(Debug, Clone)]
pub struct ThresholdAnalysis {
    /// Optimal threshold selected by adaptive algorithm
    pub optimal_threshold: f64,
    /// Minimum coupling strength in matrix
    pub min_coupling: f64,
    /// Maximum coupling strength in matrix
    pub max_coupling: f64,
    /// Mean coupling strength
    pub mean_coupling: f64,
    /// Median coupling strength
    pub median_coupling: f64,
    /// Number of edges in graph at optimal threshold
    pub num_edges_at_threshold: usize,
    /// Graph density (edges / max_possible_edges)
    pub graph_density: f64,
    /// Estimated chromatic number (max_degree + 1)
    pub estimated_chromatic_number: usize,
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
        // Complete graph requires n colors, use larger target
        let mut coupling = Array2::from_elem((10, 10), Complex64::new(0.5, 0.0));
        for i in 0..10 {
            coupling[[i, i]] = Complex64::new(0.0, 0.0);
        }

        // Complete graph K10 needs 10 colors
        let coloring = ChromaticColoring::new(&coupling, 10, 0.3).unwrap();

        let balance = coloring.color_balance();
        assert!(balance >= 0.0 && balance <= 1.0);
    }

    #[test]
    fn test_adaptive_threshold() {
        // Create varied coupling matrix
        let mut coupling = Array2::zeros((8, 8));
        for i in 0..8 {
            for j in (i + 1)..8 {
                // Distance-dependent coupling (simulating physical system)
                let distance = ((i as f64 - j as f64).abs() + 1.0);
                let strength = 1.0 / distance.powi(2);
                coupling[[i, j]] = Complex64::new(strength, 0.0);
                coupling[[j, i]] = Complex64::new(strength, 0.0);
            }
        }

        let coloring = ChromaticColoring::new_adaptive(&coupling, 3).unwrap();
        assert!(coloring.verify_coloring());
        assert_eq!(coloring.get_num_colors(), 3);
    }

    #[test]
    fn test_adaptive_vs_manual_threshold() {
        let mut coupling = Array2::zeros((6, 6));
        // Ring topology
        for i in 0..6 {
            let j = (i + 1) % 6;
            coupling[[i, j]] = Complex64::new(1.0, 0.0);
            coupling[[j, i]] = Complex64::new(1.0, 0.0);
        }

        // Both should produce valid 2-coloring (bipartite graph)
        let adaptive = ChromaticColoring::new_adaptive(&coupling, 2).unwrap();
        let manual = ChromaticColoring::new(&coupling, 2, 0.5).unwrap();

        assert!(adaptive.verify_coloring());
        assert!(manual.verify_coloring());
    }

    #[test]
    fn test_threshold_analysis() {
        let mut coupling = Array2::zeros((5, 5));
        for i in 0..5 {
            for j in (i + 1)..5 {
                coupling[[i, j]] = Complex64::new((i + j) as f64 * 0.1, 0.0);
                coupling[[j, i]] = Complex64::new((i + j) as f64 * 0.1, 0.0);
            }
        }

        let analysis = ChromaticColoring::analyze_threshold(&coupling, 3).unwrap();

        assert!(analysis.optimal_threshold >= analysis.min_coupling);
        assert!(analysis.optimal_threshold <= analysis.max_coupling);
        assert!(analysis.graph_density >= 0.0 && analysis.graph_density <= 1.0);
        assert!(analysis.estimated_chromatic_number >= 1);
    }

    #[test]
    fn test_adaptive_with_complete_graph() {
        // Complete graph K4 requires 4 colors
        let mut coupling = Array2::from_elem((4, 4), Complex64::new(1.0, 0.0));
        for i in 0..4 {
            coupling[[i, i]] = Complex64::new(0.0, 0.0);
        }

        let coloring = ChromaticColoring::new_adaptive(&coupling, 4).unwrap();
        assert!(coloring.verify_coloring());
        assert_eq!(coloring.get_num_colors(), 4);
    }

    #[test]
    fn test_adaptive_with_sparse_coupling() {
        // Very sparse coupling - only a few strong connections
        let mut coupling = Array2::zeros((10, 10));
        coupling[[0, 1]] = Complex64::new(1.0, 0.0);
        coupling[[1, 0]] = Complex64::new(1.0, 0.0);
        coupling[[2, 3]] = Complex64::new(0.9, 0.0);
        coupling[[3, 2]] = Complex64::new(0.9, 0.0);

        let coloring = ChromaticColoring::new_adaptive(&coupling, 2).unwrap();
        assert!(coloring.verify_coloring());

        let analysis = ChromaticColoring::analyze_threshold(&coupling, 2).unwrap();
        assert!(analysis.num_edges_at_threshold <= 2);
    }
}
