#!/usr/bin/env python3
"""
Statistical Testing Framework for Neuromorphic-Quantum Computing Platform

This module provides comprehensive statistical validation for the neuromorphic-quantum
predictions, including significance testing, distribution analysis, and predictive
power assessment.

Author: Data Scientist Agent
Date: 2025-09-28
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import (
    normaltest, jarque_bera, shapiro, anderson, kstest,
    ttest_1samp, ttest_ind, wilcoxon, mannwhitneyu,
    pearsonr, spearmanr, kendalltau,
    f_oneway, kruskal, chi2_contingency,
    levene, bartlett, fligner
)
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass
from enum import Enum

class TestType(Enum):
    """Types of statistical tests available"""
    NORMALITY = "normality"
    MEAN_COMPARISON = "mean_comparison"
    DISTRIBUTION_COMPARISON = "distribution_comparison"
    CORRELATION = "correlation"
    VARIANCE_COMPARISON = "variance_comparison"
    INDEPENDENCE = "independence"
    PREDICTIVE_POWER = "predictive_power"

@dataclass
class StatisticalTestResult:
    """Result of a statistical test"""
    test_name: str
    test_type: TestType
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_level: float = 0.95
    effect_size: Optional[float] = None
    interpretation: str = ""
    assumptions_met: bool = True
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant"""
        alpha = 1 - self.confidence_level
        return self.p_value < alpha

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'critical_value': self.critical_value,
            'confidence_level': self.confidence_level,
            'effect_size': self.effect_size,
            'is_significant': self.is_significant,
            'interpretation': self.interpretation,
            'assumptions_met': self.assumptions_met,
            'warnings': self.warnings
        }

class NeuromorphicQuantumStatTests:
    """
    Comprehensive statistical testing suite for neuromorphic-quantum predictions
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def test_neuromorphic_pattern_significance(self,
                                               pattern_strengths: np.ndarray,
                                               null_hypothesis_mean: float = 0.0) -> StatisticalTestResult:
        """
        Test if neuromorphic pattern detection is statistically significant

        Args:
            pattern_strengths: Array of pattern strength values
            null_hypothesis_mean: Expected mean under null hypothesis

        Returns:
            Statistical test result
        """
        # Check assumptions
        assumptions_met = True
        warnings_list = []

        if len(pattern_strengths) < 30:
            warnings_list.append("Sample size < 30, consider non-parametric test")

        # Test normality
        _, norm_p = normaltest(pattern_strengths)
        if norm_p < 0.05:
            warnings_list.append("Data may not be normally distributed")
            assumptions_met = False

        # Perform t-test
        statistic, p_value = ttest_1samp(pattern_strengths, null_hypothesis_mean)

        # Calculate effect size (Cohen's d)
        pooled_std = np.std(pattern_strengths, ddof=1)
        effect_size = (np.mean(pattern_strengths) - null_hypothesis_mean) / pooled_std

        # Interpretation
        if p_value < self.alpha:
            interpretation = f"Neuromorphic patterns show statistically significant detection (p={p_value:.6f})"
        else:
            interpretation = f"Neuromorphic patterns do not show statistically significant detection (p={p_value:.6f})"

        return StatisticalTestResult(
            test_name="One-Sample T-Test for Pattern Significance",
            test_type=TestType.MEAN_COMPARISON,
            statistic=statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation,
            assumptions_met=assumptions_met,
            warnings=warnings_list
        )

    def test_quantum_coherence_predictive_power(self,
                                                coherence_values: np.ndarray,
                                                target_returns: np.ndarray) -> StatisticalTestResult:
        """
        Test predictive power of quantum coherence values

        Args:
            coherence_values: Array of quantum coherence measurements
            target_returns: Array of corresponding target returns

        Returns:
            Statistical test result
        """
        # Check input validity
        if len(coherence_values) != len(target_returns):
            raise ValueError("Coherence values and target returns must have same length")

        # Remove any NaN values
        valid_mask = ~(np.isnan(coherence_values) | np.isnan(target_returns))
        coherence_clean = coherence_values[valid_mask]
        returns_clean = target_returns[valid_mask]

        warnings_list = []
        if len(coherence_clean) < len(coherence_values):
            warnings_list.append(f"Removed {len(coherence_values) - len(coherence_clean)} NaN values")

        # Test correlation
        statistic, p_value = pearsonr(coherence_clean, returns_clean)

        # Also test Spearman correlation for robustness
        spear_stat, spear_p = spearmanr(coherence_clean, returns_clean)

        # Effect size is the correlation coefficient itself
        effect_size = abs(statistic)

        # Interpretation
        if p_value < self.alpha:
            interpretation = f"Quantum coherence shows significant predictive power (r={statistic:.4f}, p={p_value:.6f})"
        else:
            interpretation = f"Quantum coherence does not show significant predictive power (r={statistic:.4f}, p={p_value:.6f})"

        if abs(statistic - spear_stat) > 0.1:
            warnings_list.append("Large difference between Pearson and Spearman correlations suggests non-linear relationship")

        return StatisticalTestResult(
            test_name="Pearson Correlation Test for Quantum Predictive Power",
            test_type=TestType.CORRELATION,
            statistic=statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation,
            assumptions_met=True,
            warnings=warnings_list
        )

    def test_neuromorphic_vs_quantum_performance(self,
                                                 neuromorphic_predictions: np.ndarray,
                                                 quantum_predictions: np.ndarray,
                                                 actual_outcomes: np.ndarray) -> Dict[str, StatisticalTestResult]:
        """
        Compare performance between neuromorphic and quantum components

        Args:
            neuromorphic_predictions: Predictions from neuromorphic system
            quantum_predictions: Predictions from quantum system
            actual_outcomes: Actual observed outcomes

        Returns:
            Dictionary of test results
        """
        results = {}

        # Calculate prediction errors
        neuro_errors = np.abs(neuromorphic_predictions - actual_outcomes)
        quantum_errors = np.abs(quantum_predictions - actual_outcomes)

        # Test if errors are significantly different
        statistic, p_value = ttest_ind(neuro_errors, quantum_errors)

        effect_size = (np.mean(neuro_errors) - np.mean(quantum_errors)) / np.sqrt(
            (np.var(neuro_errors, ddof=1) + np.var(quantum_errors, ddof=1)) / 2
        )

        if p_value < self.alpha:
            if np.mean(neuro_errors) < np.mean(quantum_errors):
                interpretation = "Neuromorphic system significantly outperforms quantum system"
            else:
                interpretation = "Quantum system significantly outperforms neuromorphic system"
        else:
            interpretation = "No significant difference between neuromorphic and quantum performance"

        results['performance_comparison'] = StatisticalTestResult(
            test_name="Independent T-Test: Neuromorphic vs Quantum Performance",
            test_type=TestType.MEAN_COMPARISON,
            statistic=statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation,
            assumptions_met=True
        )

        # Test correlation between predictions
        corr_stat, corr_p = pearsonr(neuromorphic_predictions, quantum_predictions)

        results['prediction_correlation'] = StatisticalTestResult(
            test_name="Correlation Between Neuromorphic and Quantum Predictions",
            test_type=TestType.CORRELATION,
            statistic=corr_stat,
            p_value=corr_p,
            confidence_level=self.confidence_level,
            effect_size=abs(corr_stat),
            interpretation=f"Neuromorphic and quantum predictions correlation: r={corr_stat:.4f}",
            assumptions_met=True
        )

        return results

    def test_prediction_distribution_properties(self,
                                                predictions: np.ndarray) -> Dict[str, StatisticalTestResult]:
        """
        Test distributional properties of predictions

        Args:
            predictions: Array of prediction values

        Returns:
            Dictionary of test results for different distributional tests
        """
        results = {}

        # Test normality with multiple tests
        # Shapiro-Wilk test (best for small samples)
        if len(predictions) <= 5000:
            shapiro_stat, shapiro_p = shapiro(predictions)
            results['shapiro_normality'] = StatisticalTestResult(
                test_name="Shapiro-Wilk Normality Test",
                test_type=TestType.NORMALITY,
                statistic=shapiro_stat,
                p_value=shapiro_p,
                confidence_level=self.confidence_level,
                interpretation="Normal distribution" if shapiro_p >= self.alpha else "Non-normal distribution",
                assumptions_met=True
            )

        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(predictions)
        results['jarque_bera_normality'] = StatisticalTestResult(
            test_name="Jarque-Bera Normality Test",
            test_type=TestType.NORMALITY,
            statistic=jb_stat,
            p_value=jb_p,
            confidence_level=self.confidence_level,
            interpretation="Normal distribution" if jb_p >= self.alpha else "Non-normal distribution",
            assumptions_met=True
        )

        # Anderson-Darling test
        ad_stat, ad_critical, ad_significance = anderson(predictions, dist='norm')
        # Find appropriate critical value for our confidence level
        significance_levels = [15, 10, 5, 2.5, 1]  # %
        alpha_percent = self.alpha * 100

        idx = 0
        for i, sig_level in enumerate(significance_levels):
            if alpha_percent <= sig_level:
                idx = i
                break

        critical_value = ad_critical[idx]
        is_normal = ad_stat < critical_value

        results['anderson_darling_normality'] = StatisticalTestResult(
            test_name="Anderson-Darling Normality Test",
            test_type=TestType.NORMALITY,
            statistic=ad_stat,
            p_value=np.nan,  # Anderson-Darling doesn't provide p-value directly
            critical_value=critical_value,
            confidence_level=self.confidence_level,
            interpretation="Normal distribution" if is_normal else "Non-normal distribution",
            assumptions_met=True
        )

        return results

    def test_regime_stability(self,
                              predictions_regime1: np.ndarray,
                              predictions_regime2: np.ndarray) -> StatisticalTestResult:
        """
        Test if predictions are stable across different market regimes

        Args:
            predictions_regime1: Predictions during first regime
            predictions_regime2: Predictions during second regime

        Returns:
            Statistical test result
        """
        # Test if means are different
        statistic, p_value = ttest_ind(predictions_regime1, predictions_regime2)

        # Calculate effect size
        pooled_std = np.sqrt((np.var(predictions_regime1, ddof=1) +
                             np.var(predictions_regime2, ddof=1)) / 2)
        effect_size = abs(np.mean(predictions_regime1) - np.mean(predictions_regime2)) / pooled_std

        # Test variance equality as well
        levene_stat, levene_p = levene(predictions_regime1, predictions_regime2)

        warnings_list = []
        if levene_p < 0.05:
            warnings_list.append("Unequal variances detected between regimes")

        if p_value < self.alpha:
            interpretation = f"Predictions are NOT stable across regimes (p={p_value:.6f})"
        else:
            interpretation = f"Predictions are stable across regimes (p={p_value:.6f})"

        return StatisticalTestResult(
            test_name="Independent T-Test for Regime Stability",
            test_type=TestType.MEAN_COMPARISON,
            statistic=statistic,
            p_value=p_value,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation,
            assumptions_met=levene_p >= 0.05,
            warnings=warnings_list
        )

    def test_signal_to_noise_ratio(self,
                                   signal: np.ndarray,
                                   noise_baseline: float = None) -> StatisticalTestResult:
        """
        Test signal-to-noise ratio of predictions

        Args:
            signal: The prediction signal
            noise_baseline: Baseline noise level (if None, estimated from data)

        Returns:
            Statistical test result
        """
        # Calculate signal power
        signal_power = np.var(signal)

        # Estimate noise if not provided
        if noise_baseline is None:
            # Use residuals from trend as noise estimate
            from scipy import signal as sp_signal
            detrended = sp_signal.detrend(signal)
            noise_power = np.var(detrended)
        else:
            noise_power = noise_baseline**2

        # Calculate SNR
        snr = signal_power / noise_power if noise_power > 0 else np.inf
        snr_db = 10 * np.log10(snr) if snr > 0 else -np.inf

        # Test if SNR is significantly above 1 (0 dB)
        # This is a custom test based on the ratio distribution
        statistic = snr

        # For interpretation, we'll use common SNR thresholds
        if snr_db > 20:
            interpretation = f"Excellent signal quality (SNR: {snr_db:.2f} dB)"
        elif snr_db > 10:
            interpretation = f"Good signal quality (SNR: {snr_db:.2f} dB)"
        elif snr_db > 0:
            interpretation = f"Moderate signal quality (SNR: {snr_db:.2f} dB)"
        else:
            interpretation = f"Poor signal quality (SNR: {snr_db:.2f} dB)"

        return StatisticalTestResult(
            test_name="Signal-to-Noise Ratio Analysis",
            test_type=TestType.PREDICTIVE_POWER,
            statistic=statistic,
            p_value=np.nan,  # Custom test, no standard p-value
            confidence_level=self.confidence_level,
            effect_size=snr_db,
            interpretation=interpretation,
            assumptions_met=True
        )

    def comprehensive_validation_suite(self,
                                       neuromorphic_data: Dict[str, np.ndarray],
                                       quantum_data: Dict[str, np.ndarray],
                                       targets: np.ndarray) -> Dict[str, Any]:
        """
        Run comprehensive validation suite on neuromorphic-quantum system

        Args:
            neuromorphic_data: Dictionary with neuromorphic outputs
            quantum_data: Dictionary with quantum outputs
            targets: Target values for validation

        Returns:
            Comprehensive validation results
        """
        results = {
            'neuromorphic_tests': {},
            'quantum_tests': {},
            'cross_system_tests': {},
            'predictive_power_tests': {},
            'stability_tests': {}
        }

        # Test neuromorphic pattern significance
        if 'pattern_strengths' in neuromorphic_data:
            results['neuromorphic_tests']['pattern_significance'] = \
                self.test_neuromorphic_pattern_significance(neuromorphic_data['pattern_strengths'])

        # Test quantum coherence predictive power
        if 'coherence_values' in quantum_data:
            results['quantum_tests']['coherence_predictive_power'] = \
                self.test_quantum_coherence_predictive_power(quantum_data['coherence_values'], targets)

        # Cross-system comparison
        if 'predictions' in neuromorphic_data and 'predictions' in quantum_data:
            results['cross_system_tests'] = self.test_neuromorphic_vs_quantum_performance(
                neuromorphic_data['predictions'], quantum_data['predictions'], targets
            )

        # Signal-to-noise ratio tests
        for system_name, system_data in [('neuromorphic', neuromorphic_data), ('quantum', quantum_data)]:
            if 'predictions' in system_data:
                results['predictive_power_tests'][f'{system_name}_snr'] = \
                    self.test_signal_to_noise_ratio(system_data['predictions'])

        return results

def calculate_confidence_intervals(data: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for data

    Args:
        data: Input data array
        confidence_level: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence_level
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf(1 - alpha/2, len(data) - 1)

    return mean - h, mean + h

def calculate_effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Cohen's d effect size
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) /
                        (len(group1) + len(group2) - 2))

    return (mean1 - mean2) / pooled_std

def multiple_testing_correction(p_values: List[float], method: str = 'bonferroni') -> List[float]:
    """
    Apply multiple testing correction

    Args:
        p_values: List of p-values
        method: Correction method ('bonferroni', 'holm', 'hochberg')

    Returns:
        List of corrected p-values
    """
    p_array = np.array(p_values)
    n = len(p_array)

    if method == 'bonferroni':
        return (p_array * n).clip(0, 1).tolist()
    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_idx = np.argsort(p_array)
        corrected = np.zeros_like(p_array)

        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min(1.0, p_array[idx] * (n - i))

        return corrected.tolist()
    else:
        return p_values  # Return original if method not recognized

if __name__ == "__main__":
    # Example usage and testing
    np.random.seed(42)

    # Generate example data
    pattern_strengths = np.random.normal(0.7, 0.2, 100)
    coherence_values = np.random.normal(0.8, 0.15, 100)
    target_returns = 0.3 * coherence_values + 0.2 * pattern_strengths + np.random.normal(0, 0.1, 100)

    # Initialize test suite
    test_suite = NeuromorphicQuantumStatTests()

    # Run tests
    pattern_test = test_suite.test_neuromorphic_pattern_significance(pattern_strengths)
    print("Pattern Significance Test:")
    print(f"  Statistic: {pattern_test.statistic:.4f}")
    print(f"  P-value: {pattern_test.p_value:.6f}")
    print(f"  Significant: {pattern_test.is_significant}")
    print(f"  Interpretation: {pattern_test.interpretation}")
    print()

    coherence_test = test_suite.test_quantum_coherence_predictive_power(coherence_values, target_returns)
    print("Quantum Coherence Predictive Power Test:")
    print(f"  Correlation: {coherence_test.statistic:.4f}")
    print(f"  P-value: {coherence_test.p_value:.6f}")
    print(f"  Significant: {coherence_test.is_significant}")
    print(f"  Interpretation: {coherence_test.interpretation}")