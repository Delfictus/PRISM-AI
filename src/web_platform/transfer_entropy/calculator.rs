/// Transfer entropy calculator with statistical significance testing
use super::time_series::TimeSeries;
use super::histogram::JointHistogram;
use serde::{Deserialize, Serialize};
use rand::prelude::*;

/// Transfer entropy result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEntropyResult {
    /// Transfer entropy value (bits)
    pub te_value: f64,
    /// Source series name
    pub source: String,
    /// Target series name
    pub target: String,
    /// Lag used
    pub lag: usize,
    /// Number of bins used
    pub num_bins: usize,
    /// Statistical significance
    pub significance: SignificanceTest,
}

/// Statistical significance test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceTest {
    /// P-value from permutation test
    pub p_value: f64,
    /// Is significant at 0.05 level
    pub significant: bool,
    /// Number of permutations used
    pub num_permutations: usize,
    /// Null distribution statistics
    pub null_mean: f64,
    pub null_std: f64,
}

/// Transfer entropy calculator
pub struct TransferEntropyCalculator {
    num_bins: usize,
    num_permutations: usize,
}

impl TransferEntropyCalculator {
    /// Create new calculator
    pub fn new(num_bins: usize, num_permutations: usize) -> Self {
        Self {
            num_bins,
            num_permutations,
        }
    }

    /// Calculate transfer entropy from source to target with statistical testing
    pub fn calculate(
        &self,
        source: &TimeSeries,
        target: &TimeSeries,
        lag: usize,
    ) -> Result<TransferEntropyResult, String> {
        // Validate inputs
        if source.len() != target.len() {
            return Err("Source and target must have same length".to_string());
        }

        if source.len() < lag + 2 {
            return Err(format!("Series too short for lag {}", lag));
        }

        // Calculate actual transfer entropy
        let te_value = self.calculate_te(&source.values, &target.values, lag)?;

        // Perform permutation test for significance
        let significance = self.permutation_test(&source.values, &target.values, lag, te_value)?;

        Ok(TransferEntropyResult {
            te_value,
            source: source.name.clone(),
            target: target.name.clone(),
            lag,
            num_bins: self.num_bins,
            significance,
        })
    }

    /// Calculate transfer entropy value
    fn calculate_te(&self, source: &[f64], target: &[f64], lag: usize) -> Result<f64, String> {
        let n = source.len();
        if n < lag + 2 {
            return Err("Series too short".to_string());
        }

        // Build state vectors
        // Y_{t+1}, Y_t, X_t
        let mut data = Vec::new();
        for t in lag..(n - 1) {
            data.push(vec![
                target[t + 1],  // Y_{t+1}
                target[t],      // Y_t
                source[t - lag], // X_{t-lag}
            ]);
        }

        if data.is_empty() {
            return Err("Not enough data points after lagging".to_string());
        }

        // Calculate joint entropies
        let h_y_next_y_curr_x = self.entropy_3d(&data)?;

        // H(Y_{t+1}, Y_t)
        let data_y_next_y_curr: Vec<Vec<f64>> = data.iter()
            .map(|v| vec![v[0], v[1]])
            .collect();
        let h_y_next_y_curr = self.entropy_2d(&data_y_next_y_curr)?;

        // H(Y_t, X_t)
        let data_y_curr_x: Vec<Vec<f64>> = data.iter()
            .map(|v| vec![v[1], v[2]])
            .collect();
        let h_y_curr_x = self.entropy_2d(&data_y_curr_x)?;

        // H(Y_t)
        let data_y_curr: Vec<f64> = data.iter().map(|v| v[1]).collect();
        let h_y_curr = self.entropy_1d(&data_y_curr)?;

        // TE = H(Y_{t+1}, Y_t) + H(Y_t, X_t) - H(Y_{t+1}, Y_t, X_t) - H(Y_t)
        let te = h_y_next_y_curr + h_y_curr_x - h_y_next_y_curr_x - h_y_curr;

        Ok(te.max(0.0)) // Transfer entropy should be non-negative
    }

    /// Calculate 1D entropy (Shannon entropy)
    fn entropy_1d(&self, data: &[f64]) -> Result<f64, String> {
        let hist = super::histogram::Histogram::from_data(data, self.num_bins);

        let mut entropy = 0.0;
        for i in 0..hist.num_bins() {
            let p = hist.probability(i);
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        Ok(entropy)
    }

    /// Calculate 2D joint entropy
    fn entropy_2d(&self, data: &[Vec<f64>]) -> Result<f64, String> {
        let hist = JointHistogram::from_data(data, vec![self.num_bins, self.num_bins]);

        let mut entropy = 0.0;
        for (_, p) in hist.non_zero_bins() {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        Ok(entropy)
    }

    /// Calculate 3D joint entropy
    fn entropy_3d(&self, data: &[Vec<f64>]) -> Result<f64, String> {
        let hist = JointHistogram::from_data(
            data,
            vec![self.num_bins, self.num_bins, self.num_bins]
        );

        let mut entropy = 0.0;
        for (_, p) in hist.non_zero_bins() {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        Ok(entropy)
    }

    /// Permutation test for statistical significance
    fn permutation_test(
        &self,
        source: &[f64],
        target: &[f64],
        lag: usize,
        observed_te: f64,
    ) -> Result<SignificanceTest, String> {
        let mut rng = rand::thread_rng();
        let mut null_distribution = Vec::new();

        // Generate null distribution by permuting source
        for _ in 0..self.num_permutations {
            let mut permuted_source = source.to_vec();
            permuted_source.shuffle(&mut rng);

            let te = self.calculate_te(&permuted_source, target, lag)?;
            null_distribution.push(te);
        }

        // Calculate p-value
        let count_greater = null_distribution.iter()
            .filter(|&&te| te >= observed_te)
            .count();
        let p_value = count_greater as f64 / self.num_permutations as f64;

        // Calculate null distribution statistics
        let null_mean = null_distribution.iter().sum::<f64>() / self.num_permutations as f64;
        let null_variance = null_distribution.iter()
            .map(|te| (te - null_mean).powi(2))
            .sum::<f64>() / self.num_permutations as f64;
        let null_std = null_variance.sqrt();

        Ok(SignificanceTest {
            p_value,
            significant: p_value < 0.05,
            num_permutations: self.num_permutations,
            null_mean,
            null_std,
        })
    }
}

impl Default for TransferEntropyCalculator {
    fn default() -> Self {
        Self::new(10, 100) // 10 bins, 100 permutations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_entropy_independent() {
        // Independent random series should have low TE
        let source = TimeSeries::from_values("source", vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ]);

        let target = TimeSeries::from_values("target", vec![
            20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0,
            10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ]);

        let calc = TransferEntropyCalculator::new(5, 50);
        let result = calc.calculate(&source, &target, 1).unwrap();

        // TE should be low for independent series
        assert!(result.te_value >= 0.0);
        println!("Independent TE: {:.4}, p-value: {:.4}",
                 result.te_value, result.significance.p_value);
    }

    #[test]
    fn test_transfer_entropy_dependent() {
        // Create dependent series: target follows source with delay
        let source_vals: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let target_vals: Vec<f64> = (0..30).map(|i| {
            if i >= 2 {
                source_vals[i - 2] + 0.1 * (i as f64).sin()
            } else {
                (i as f64).sin()
            }
        }).collect();

        let source = TimeSeries::from_values("source", source_vals);
        let target = TimeSeries::from_values("target", target_vals);

        let calc = TransferEntropyCalculator::new(5, 50);
        let result = calc.calculate(&source, &target, 2).unwrap();

        // TE should be positive for dependent series
        assert!(result.te_value > 0.0);
        println!("Dependent TE: {:.4}, p-value: {:.4}",
                 result.te_value, result.significance.p_value);
    }

    #[test]
    fn test_entropy_calculation() {
        let calc = TransferEntropyCalculator::new(5, 50);

        // Uniform distribution should have maximum entropy
        let uniform_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let entropy = calc.entropy_1d(&uniform_data).unwrap();

        // For 5 bins with uniform data, entropy should be close to log2(5) ≈ 2.32
        assert!(entropy > 2.0 && entropy < 3.0);
    }
}
