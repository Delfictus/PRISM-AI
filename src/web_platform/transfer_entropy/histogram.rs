/// Histogram-based probability estimation for entropy calculations
use std::collections::HashMap;

/// Histogram for discrete probability estimation
pub struct Histogram {
    bins: Vec<f64>,
    counts: Vec<usize>,
    total_count: usize,
}

impl Histogram {
    /// Create histogram from data with specified number of bins
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return Self {
                bins: Vec::new(),
                counts: Vec::new(),
                total_count: 0,
            };
        }

        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Create bin edges
        let bin_width = (max - min) / num_bins as f64;
        let bins: Vec<f64> = (0..=num_bins)
            .map(|i| min + i as f64 * bin_width)
            .collect();

        // Count data points in each bin
        let mut counts = vec![0; num_bins];
        for &value in data {
            let bin_idx = ((value - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(num_bins - 1); // Handle edge case where value == max
            counts[bin_idx] += 1;
        }

        Self {
            bins,
            counts,
            total_count: data.len(),
        }
    }

    /// Get probability for a bin index
    pub fn probability(&self, bin_idx: usize) -> f64 {
        if bin_idx >= self.counts.len() || self.total_count == 0 {
            return 0.0;
        }
        self.counts[bin_idx] as f64 / self.total_count as f64
    }

    /// Get number of bins
    pub fn num_bins(&self) -> usize {
        self.counts.len()
    }

    /// Get total count
    pub fn total_count(&self) -> usize {
        self.total_count
    }
}

/// Multidimensional histogram for joint probability estimation
pub struct JointHistogram {
    /// Bin indices to counts
    bin_map: HashMap<Vec<usize>, usize>,
    /// Number of bins per dimension
    bins_per_dim: Vec<usize>,
    /// Bin edges per dimension
    bin_edges: Vec<Vec<f64>>,
    /// Total count
    total_count: usize,
}

impl JointHistogram {
    /// Create joint histogram from multi-dimensional data
    pub fn from_data(data: &[Vec<f64>], bins_per_dim: Vec<usize>) -> Self {
        if data.is_empty() || bins_per_dim.is_empty() {
            return Self {
                bin_map: HashMap::new(),
                bins_per_dim,
                bin_edges: Vec::new(),
                total_count: 0,
            };
        }

        let num_dims = data[0].len();
        if num_dims != bins_per_dim.len() {
            panic!("Dimension mismatch: data has {} dims, bins_per_dim has {} dims",
                   num_dims, bins_per_dim.len());
        }

        // Calculate bin edges for each dimension
        let mut bin_edges = Vec::new();
        for dim in 0..num_dims {
            let dim_data: Vec<f64> = data.iter().map(|row| row[dim]).collect();
            let min = dim_data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = dim_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let bin_width = (max - min) / bins_per_dim[dim] as f64;
            let edges: Vec<f64> = (0..=bins_per_dim[dim])
                .map(|i| min + i as f64 * bin_width)
                .collect();
            bin_edges.push(edges);
        }

        // Count data points in bins
        let mut bin_map = HashMap::new();
        for row in data {
            let mut bin_indices = Vec::new();
            for (dim, &value) in row.iter().enumerate() {
                let min = bin_edges[dim][0];
                let max = bin_edges[dim][bin_edges[dim].len() - 1];
                let bin_width = (max - min) / bins_per_dim[dim] as f64;

                let bin_idx = ((value - min) / bin_width).floor() as usize;
                let bin_idx = bin_idx.min(bins_per_dim[dim] - 1);
                bin_indices.push(bin_idx);
            }

            *bin_map.entry(bin_indices).or_insert(0) += 1;
        }

        Self {
            bin_map,
            bins_per_dim,
            bin_edges,
            total_count: data.len(),
        }
    }

    /// Get joint probability for bin indices
    pub fn probability(&self, bin_indices: &[usize]) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let count = self.bin_map.get(bin_indices).copied().unwrap_or(0);
        count as f64 / self.total_count as f64
    }

    /// Get all non-zero bin indices and their probabilities
    pub fn non_zero_bins(&self) -> Vec<(Vec<usize>, f64)> {
        self.bin_map.iter()
            .map(|(indices, &count)| {
                (indices.clone(), count as f64 / self.total_count as f64)
            })
            .collect()
    }

    /// Get total count
    pub fn total_count(&self) -> usize {
        self.total_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let hist = Histogram::from_data(&data, 5);

        assert_eq!(hist.num_bins(), 5);
        assert_eq!(hist.total_count(), 10);

        // Each bin should have 2 values, so probability = 0.2
        for i in 0..5 {
            assert!((hist.probability(i) - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn test_joint_histogram() {
        let data = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
        ];

        let hist = JointHistogram::from_data(&data, vec![2, 2]);

        assert_eq!(hist.total_count(), 4);

        let non_zero = hist.non_zero_bins();
        assert_eq!(non_zero.len(), 4); // Each point in different bin

        // Each bin has 1 out of 4 points
        for (_, prob) in non_zero {
            assert!((prob - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_joint_histogram_same_bin() {
        let data = vec![
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![9.0, 9.0],
            vec![9.1, 9.1],
        ];

        let hist = JointHistogram::from_data(&data, vec![2, 2]);

        // Should have 2 bins with 2 points each
        let non_zero = hist.non_zero_bins();
        assert_eq!(non_zero.len(), 2);

        for (_, prob) in non_zero {
            assert!((prob - 0.5).abs() < 1e-6);
        }
    }
}
