/// Time series data structures and preprocessing
use serde::{Deserialize, Serialize};

/// Time series data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub name: String,
    pub values: Vec<f64>,
    pub timestamps: Vec<u64>,
}

impl TimeSeries {
    /// Create new time series
    pub fn new(name: impl Into<String>, values: Vec<f64>, timestamps: Vec<u64>) -> Self {
        Self {
            name: name.into(),
            values,
            timestamps,
        }
    }

    /// Create from values only (generates sequential timestamps)
    pub fn from_values(name: impl Into<String>, values: Vec<f64>) -> Self {
        let timestamps: Vec<u64> = (0..values.len() as u64).collect();
        Self::new(name, values, timestamps)
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get returns (price changes)
    pub fn returns(&self) -> Vec<f64> {
        if self.values.len() < 2 {
            return Vec::new();
        }

        self.values.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Get log returns
    pub fn log_returns(&self) -> Vec<f64> {
        if self.values.len() < 2 {
            return Vec::new();
        }

        self.values.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect()
    }

    /// Normalize to [0, 1]
    pub fn normalize(&self) -> Vec<f64> {
        if self.values.is_empty() {
            return Vec::new();
        }

        let min = self.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            return vec![0.5; self.values.len()];
        }

        self.values.iter()
            .map(|v| (v - min) / (max - min))
            .collect()
    }

    /// Z-score normalization (mean=0, std=1)
    pub fn standardize(&self) -> Vec<f64> {
        if self.values.is_empty() {
            return Vec::new();
        }

        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        let variance = self.values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / self.values.len() as f64;
        let std = variance.sqrt();

        if std < 1e-10 {
            return vec![0.0; self.values.len()];
        }

        self.values.iter()
            .map(|v| (v - mean) / std)
            .collect()
    }

    /// Create lagged version of time series
    pub fn lag(&self, lag: usize) -> Vec<f64> {
        if lag >= self.values.len() {
            return Vec::new();
        }

        self.values[..self.values.len() - lag].to_vec()
    }

    /// Get statistics
    pub fn statistics(&self) -> TimeSeriesStats {
        if self.values.is_empty() {
            return TimeSeriesStats::default();
        }

        let n = self.values.len() as f64;
        let mean = self.values.iter().sum::<f64>() / n;

        let variance = self.values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / n;
        let std = variance.sqrt();

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        TimeSeriesStats {
            count: self.values.len(),
            mean,
            std,
            min,
            max,
            median,
        }
    }
}

/// Time series statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

impl Default for TimeSeriesStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_returns() {
        let ts = TimeSeries::from_values("test", vec![100.0, 105.0, 103.0, 108.0]);
        let returns = ts.returns();

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.05).abs() < 1e-6); // (105-100)/100 = 0.05
        assert!((returns[1] - (-0.019047619)).abs() < 1e-6); // (103-105)/105
    }

    #[test]
    fn test_normalize() {
        let ts = TimeSeries::from_values("test", vec![10.0, 20.0, 30.0]);
        let normalized = ts.normalize();

        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.5).abs() < 1e-6);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_standardize() {
        let ts = TimeSeries::from_values("test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let standardized = ts.standardize();

        let mean = standardized.iter().sum::<f64>() / standardized.len() as f64;
        assert!(mean.abs() < 1e-6); // Mean should be ~0
    }

    #[test]
    fn test_lag() {
        let ts = TimeSeries::from_values("test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let lagged = ts.lag(1);

        assert_eq!(lagged, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_statistics() {
        let ts = TimeSeries::from_values("test", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = ts.statistics();

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.median - 3.0).abs() < 1e-6);
    }
}
