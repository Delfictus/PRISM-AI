/// Statistical utilities for transfer entropy analysis
use super::calculator::{TransferEntropyResult, SignificanceTest};
use super::time_series::TimeSeries;
use serde::{Deserialize, Serialize};

/// Pairwise transfer entropy analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseAnalysis {
    pub pairs: Vec<TransferEntropyPair>,
    pub significant_count: usize,
    pub total_count: usize,
}

/// Transfer entropy for a pair (bidirectional)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEntropyPair {
    pub symbol_a: String,
    pub symbol_b: String,
    pub te_a_to_b: TransferEntropyResult,
    pub te_b_to_a: TransferEntropyResult,
    pub net_flow: f64, // te_a_to_b - te_b_to_a
    pub dominant_direction: String,
}

impl PairwiseAnalysis {
    /// Create new pairwise analysis
    pub fn new(pairs: Vec<TransferEntropyPair>) -> Self {
        let significant_count = pairs.iter()
            .filter(|p| p.te_a_to_b.significance.significant || p.te_b_to_a.significance.significant)
            .count();

        let total_count = pairs.len();

        Self {
            pairs,
            significant_count,
            total_count,
        }
    }

    /// Get significant pairs only
    pub fn significant_pairs(&self) -> Vec<&TransferEntropyPair> {
        self.pairs.iter()
            .filter(|p| p.te_a_to_b.significance.significant || p.te_b_to_a.significance.significant)
            .collect()
    }

    /// Get pairs where A dominates B
    pub fn a_dominates(&self) -> Vec<&TransferEntropyPair> {
        self.pairs.iter()
            .filter(|p| p.net_flow > 0.0 && p.te_a_to_b.significance.significant)
            .collect()
    }

    /// Get pairs where B dominates A
    pub fn b_dominates(&self) -> Vec<&TransferEntropyPair> {
        self.pairs.iter()
            .filter(|p| p.net_flow < 0.0 && p.te_b_to_a.significance.significant)
            .collect()
    }

    /// Get top N pairs by absolute net flow
    pub fn top_pairs(&self, n: usize) -> Vec<&TransferEntropyPair> {
        let mut sorted = self.pairs.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.net_flow.abs().partial_cmp(&a.net_flow.abs()).unwrap());
        sorted.into_iter().take(n).collect()
    }
}

impl TransferEntropyPair {
    /// Create from bidirectional TE results
    pub fn new(
        symbol_a: String,
        symbol_b: String,
        te_a_to_b: TransferEntropyResult,
        te_b_to_a: TransferEntropyResult,
    ) -> Self {
        let net_flow = te_a_to_b.te_value - te_b_to_a.te_value;

        let dominant_direction = if net_flow.abs() < 0.01 {
            "balanced".to_string()
        } else if net_flow > 0.0 {
            format!("{} → {}", symbol_a, symbol_b)
        } else {
            format!("{} → {}", symbol_b, symbol_a)
        };

        Self {
            symbol_a,
            symbol_b,
            te_a_to_b,
            te_b_to_a,
            net_flow,
            dominant_direction,
        }
    }

    /// Is this pair statistically significant in either direction?
    pub fn is_significant(&self) -> bool {
        self.te_a_to_b.significance.significant || self.te_b_to_a.significance.significant
    }

    /// Get strength of information flow (max TE in either direction)
    pub fn flow_strength(&self) -> f64 {
        self.te_a_to_b.te_value.max(self.te_b_to_a.te_value)
    }
}

/// Multi-lag analysis - find optimal lag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLagAnalysis {
    pub source: String,
    pub target: String,
    pub lag_results: Vec<(usize, f64, f64)>, // (lag, te_value, p_value)
    pub optimal_lag: usize,
    pub optimal_te: f64,
}

impl MultiLagAnalysis {
    /// Create from lag results
    pub fn new(
        source: String,
        target: String,
        lag_results: Vec<(usize, f64, f64)>,
    ) -> Self {
        // Find lag with maximum TE
        let (optimal_lag, optimal_te, _) = lag_results.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
            .unwrap_or((0, 0.0, 1.0));

        Self {
            source,
            target,
            lag_results,
            optimal_lag,
            optimal_te,
        }
    }

    /// Get lags with significant TE
    pub fn significant_lags(&self) -> Vec<(usize, f64)> {
        self.lag_results.iter()
            .filter(|(_, _, p_value)| *p_value < 0.05)
            .map(|(lag, te, _)| (*lag, *te))
            .collect()
    }
}

/// Effect size categories (Cohen's d interpretation)
pub fn effect_size_category(te_value: f64) -> &'static str {
    if te_value < 0.1 {
        "negligible"
    } else if te_value < 0.3 {
        "small"
    } else if te_value < 0.5 {
        "medium"
    } else {
        "large"
    }
}

/// FDR correction (Benjamini-Hochberg procedure)
pub fn fdr_correction(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let n = p_values.len();
    if n == 0 {
        return Vec::new();
    }

    // Create (index, p_value) pairs and sort by p_value
    let mut indexed: Vec<(usize, f64)> = p_values.iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Apply BH procedure
    let mut rejected = vec![false; n];
    for (rank, (original_idx, p_value)) in indexed.iter().enumerate() {
        let threshold = alpha * (rank + 1) as f64 / n as f64;
        if *p_value <= threshold {
            rejected[*original_idx] = true;
        } else {
            break; // BH procedure stops at first non-rejection
        }
    }

    rejected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web_platform::transfer_entropy::calculator::SignificanceTest;

    fn create_dummy_te_result(value: f64, p_value: f64, source: &str, target: &str) -> TransferEntropyResult {
        TransferEntropyResult {
            te_value: value,
            source: source.to_string(),
            target: target.to_string(),
            lag: 1,
            num_bins: 10,
            significance: SignificanceTest {
                p_value,
                significant: p_value < 0.05,
                num_permutations: 100,
                null_mean: 0.0,
                null_std: 0.1,
            },
        }
    }

    #[test]
    fn test_transfer_entropy_pair() {
        let te_a_to_b = create_dummy_te_result(0.5, 0.01, "A", "B");
        let te_b_to_a = create_dummy_te_result(0.2, 0.10, "B", "A");

        let pair = TransferEntropyPair::new(
            "A".to_string(),
            "B".to_string(),
            te_a_to_b,
            te_b_to_a,
        );

        assert!(pair.is_significant());
        assert_eq!(pair.net_flow, 0.3);
        assert!(pair.dominant_direction.contains("A → B"));
        assert_eq!(pair.flow_strength(), 0.5);
    }

    #[test]
    fn test_pairwise_analysis() {
        let pair1 = TransferEntropyPair::new(
            "A".to_string(),
            "B".to_string(),
            create_dummy_te_result(0.5, 0.01, "A", "B"),
            create_dummy_te_result(0.2, 0.10, "B", "A"),
        );

        let pair2 = TransferEntropyPair::new(
            "C".to_string(),
            "D".to_string(),
            create_dummy_te_result(0.1, 0.50, "C", "D"),
            create_dummy_te_result(0.1, 0.50, "D", "C"),
        );

        let analysis = PairwiseAnalysis::new(vec![pair1, pair2]);

        assert_eq!(analysis.total_count, 2);
        assert_eq!(analysis.significant_count, 1);
        assert_eq!(analysis.significant_pairs().len(), 1);
    }

    #[test]
    fn test_effect_size_category() {
        assert_eq!(effect_size_category(0.05), "negligible");
        assert_eq!(effect_size_category(0.2), "small");
        assert_eq!(effect_size_category(0.4), "medium");
        assert_eq!(effect_size_category(0.6), "large");
    }

    #[test]
    fn test_fdr_correction() {
        let p_values = vec![0.001, 0.02, 0.04, 0.08, 0.15];
        let rejected = fdr_correction(&p_values, 0.05);

        // With alpha=0.05, first few should be rejected
        assert!(rejected[0]); // 0.001 < 0.05 * 1/5 = 0.01
        assert!(rejected[1]); // 0.02 < 0.05 * 2/5 = 0.02
        assert!(rejected[2]); // 0.04 < 0.05 * 3/5 = 0.03 (false, so stops)
    }

    #[test]
    fn test_multi_lag_analysis() {
        let lag_results = vec![
            (1, 0.1, 0.50),
            (2, 0.5, 0.01),
            (3, 0.3, 0.05),
            (4, 0.2, 0.10),
        ];

        let analysis = MultiLagAnalysis::new(
            "source".to_string(),
            "target".to_string(),
            lag_results,
        );

        assert_eq!(analysis.optimal_lag, 2);
        assert_eq!(analysis.optimal_te, 0.5);

        let sig_lags = analysis.significant_lags();
        assert_eq!(sig_lags.len(), 2); // lag 2 and 3
    }
}
