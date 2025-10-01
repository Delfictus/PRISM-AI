//! Classical Algorithm Baselines for Competitive Comparison
//!
//! Implements state-of-the-art classical algorithms:
//! - DSATUR (graph coloring)
//! - Greedy nearest neighbor (TSP)
//! - 2-opt local search (TSP)

use anyhow::{Result, Context};
use ndarray::Array2;
use std::collections::HashSet;

// ============================================================================
// GRAPH COLORING: DSATUR
// ============================================================================

/// Classical DSATUR algorithm for graph coloring
/// Reference: Brélaz, D. (1979). "New methods to color the vertices of a graph"
pub struct DSaturSolver {
    pub num_vertices: usize,
    pub adjacency: Array2<bool>,
}

impl DSaturSolver {
    pub fn new(adjacency: Array2<bool>) -> Self {
        let num_vertices = adjacency.nrows();
        Self { num_vertices, adjacency }
    }

    /// Run DSATUR algorithm
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

        // DSATUR main loop
        while !uncolored.is_empty() {
            // Find vertex with highest saturation degree
            let v = self.find_max_saturation_vertex(&uncolored, &coloring);

            // Find smallest available color
            let used_colors: HashSet<usize> = (0..n)
                .filter(|&u| {
                    u != v && coloring[u] != usize::MAX && self.adjacency[[v, u]]
                })
                .map(|u| coloring[u])
                .collect();

            let color = (0..max_colors)
                .find(|c| !used_colors.contains(c))
                .context("Not enough colors for valid coloring")?;

            coloring[v] = color;
            uncolored.remove(&v);
        }

        // Count colors used
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

    fn find_max_saturation_vertex(
        &self,
        uncolored: &HashSet<usize>,
        coloring: &[usize],
    ) -> usize {
        let n = self.num_vertices;
        let mut max_saturation = 0;
        let mut max_degree = 0;
        let mut best_vertex = *uncolored.iter().next().unwrap();

        for &v in uncolored {
            // Calculate saturation degree (number of distinct colors in neighborhood)
            let neighbor_colors: HashSet<usize> = (0..n)
                .filter(|&u| {
                    u != v && coloring[u] != usize::MAX && self.adjacency[[v, u]]
                })
                .map(|u| coloring[u])
                .collect();

            let saturation = neighbor_colors.len();

            // Calculate degree (tie-breaker)
            let degree = (0..n)
                .filter(|&u| u != v && self.adjacency[[v, u]])
                .count();

            // Select vertex with highest saturation, then highest degree
            if saturation > max_saturation ||
               (saturation == max_saturation && degree > max_degree) {
                max_saturation = saturation;
                max_degree = degree;
                best_vertex = v;
            }
        }

        best_vertex
    }

    /// Validate coloring is correct
    pub fn validate(&self, coloring: &[usize]) -> bool {
        for i in 0..self.num_vertices {
            for j in (i+1)..self.num_vertices {
                if self.adjacency[[i, j]] && coloring[i] == coloring[j] {
                    return false; // Adjacent vertices have same color
                }
            }
        }
        true
    }
}

// ============================================================================
// TSP: NEAREST NEIGHBOR + 2-OPT
// ============================================================================

/// Classical TSP solver: Nearest neighbor + 2-opt
pub struct ClassicalTspSolver {
    pub num_cities: usize,
    pub distances: Array2<f64>,
}

impl ClassicalTspSolver {
    pub fn new(distances: Array2<f64>) -> Self {
        let num_cities = distances.nrows();
        Self { num_cities, distances }
    }

    /// Nearest neighbor heuristic
    pub fn nearest_neighbor(&self, start: usize) -> (Vec<usize>, f64) {
        let mut tour = vec![start];
        let mut unvisited: HashSet<usize> = (0..self.num_cities).collect();
        unvisited.remove(&start);

        let mut current = start;
        let mut length = 0.0;

        while !unvisited.is_empty() {
            // Find nearest unvisited city
            let mut nearest = *unvisited.iter().next().unwrap();
            let mut min_dist = f64::INFINITY;

            for &city in &unvisited {
                let dist = self.distances[[current, city]];
                if dist < min_dist {
                    min_dist = dist;
                    nearest = city;
                }
            }

            tour.push(nearest);
            length += min_dist;
            current = nearest;
            unvisited.remove(&nearest);
        }

        // Close tour
        length += self.distances[[current, start]];

        (tour, length)
    }

    /// 2-opt local search improvement
    pub fn two_opt(&self, tour: &[usize], max_iterations: usize) -> (Vec<usize>, f64) {
        let mut current_tour = tour.to_vec();
        let mut current_length = self.calculate_tour_length(&current_tour);
        let mut improved = true;
        let mut iteration = 0;

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            for i in 1..(self.num_cities - 1) {
                for j in (i + 1)..self.num_cities {
                    // Try reversing segment [i..j]
                    let new_tour = self.two_opt_swap(&current_tour, i, j);
                    let new_length = self.calculate_tour_length(&new_tour);

                    if new_length < current_length {
                        current_tour = new_tour;
                        current_length = new_length;
                        improved = true;
                        break;
                    }
                }
                if improved {
                    break;
                }
            }
        }

        (current_tour, current_length)
    }

    fn two_opt_swap(&self, tour: &[usize], i: usize, j: usize) -> Vec<usize> {
        let mut new_tour = Vec::with_capacity(tour.len());

        // Add [0..i] unchanged
        new_tour.extend_from_slice(&tour[0..i]);

        // Add [i..j] reversed
        for k in (i..=j).rev() {
            new_tour.push(tour[k]);
        }

        // Add [j+1..] unchanged
        if j + 1 < tour.len() {
            new_tour.extend_from_slice(&tour[j+1..]);
        }

        new_tour
    }

    fn calculate_tour_length(&self, tour: &[usize]) -> f64 {
        let mut length = 0.0;
        for i in 0..(tour.len() - 1) {
            length += self.distances[[tour[i], tour[i + 1]]];
        }
        // Close the tour
        if !tour.is_empty() {
            length += self.distances[[tour[tour.len() - 1], tour[0]]];
        }
        length
    }

    /// Solve TSP with nearest neighbor + 2-opt
    pub fn solve(&self, iterations_2opt: usize) -> (Vec<usize>, f64) {
        // Try nearest neighbor from multiple starting points
        let mut best_tour = Vec::new();
        let mut best_length = f64::INFINITY;

        // Try from first few cities as starting points
        let start_points = (self.num_cities / 10).max(1).min(10);

        for start in 0..start_points {
            let (tour, length) = self.nearest_neighbor(start);
            if length < best_length {
                best_tour = tour;
                best_length = length;
            }
        }

        // Improve with 2-opt
        self.two_opt(&best_tour, iterations_2opt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsatur_simple() {
        // Triangle graph: needs 3 colors
        let mut adj = Array2::from_elem((3, 3), false);
        adj[[0, 1]] = true; adj[[1, 0]] = true;
        adj[[1, 2]] = true; adj[[2, 1]] = true;
        adj[[2, 0]] = true; adj[[0, 2]] = true;

        let solver = DSaturSolver::new(adj);
        let (coloring, num_colors) = solver.solve(3).unwrap();

        assert_eq!(num_colors, 3);
        assert!(solver.validate(&coloring));
    }

    #[test]
    fn test_tsp_simple() {
        // 4 cities in a square
        let mut dist = Array2::zeros((4, 4));
        dist[[0, 1]] = 1.0; dist[[1, 0]] = 1.0;
        dist[[1, 2]] = 1.0; dist[[2, 1]] = 1.0;
        dist[[2, 3]] = 1.0; dist[[3, 2]] = 1.0;
        dist[[3, 0]] = 1.0; dist[[0, 3]] = 1.0;
        dist[[0, 2]] = 1.4; dist[[2, 0]] = 1.4;
        dist[[1, 3]] = 1.4; dist[[3, 1]] = 1.4;

        let solver = ClassicalTspSolver::new(dist);
        let (_tour, length) = solver.solve(100);

        // Optimal tour length is 4.0
        assert!((length - 4.0).abs() < 0.1);
    }
}
