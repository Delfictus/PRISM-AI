#!/usr/bin/env python3
"""
Model Validation and Cross-Validation Framework
for Neuromorphic-Quantum Computing Platform

This module provides comprehensive model validation, cross-validation,
bootstrap analysis, and out-of-sample testing capabilities.

Author: Data Scientist Agent
Date: 2025-09-28
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    TimeSeriesSplit, KFold, StratifiedKFold, LeaveOneOut,
    cross_val_score, validation_curve, learning_curve
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from scipy import stats
from scipy.stats import bootstrap
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class ValidationMethod(Enum):
    """Available validation methods"""
    TIME_SERIES_SPLIT = "time_series_split"
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    LEAVE_ONE_OUT = "leave_one_out"
    BOOTSTRAP = "bootstrap"
    WALK_FORWARD = "walk_forward"

class MetricType(Enum):
    """Types of evaluation metrics"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    RANKING = "ranking"

@dataclass
class ValidationResult:
    """Result of model validation"""
    validation_method: str
    n_splits: int
    scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    metric_name: str
    training_scores: Optional[List[float]] = None
    validation_times: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None

@dataclass
class CrossValidationSummary:
    """Summary of cross-validation results"""
    validation_results: Dict[str, ValidationResult]
    best_method: str
    best_score: float
    overfitting_detected: bool
    stability_score: float  # How consistent scores are across methods
    recommendation: str

@dataclass
class BootstrapResult:
    """Bootstrap analysis result"""
    original_statistic: float
    bootstrap_statistics: np.ndarray
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None
    bias: float = 0.0
    standard_error: float = 0.0

class NeuromorphicQuantumValidator:
    """
    Comprehensive validation framework for neuromorphic-quantum predictions
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize validator

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def time_series_cross_validation(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     model: Any,
                                     n_splits: int = 5,
                                     metric: str = 'mse',
                                     gap: int = 0) -> ValidationResult:
        """
        Time series cross-validation with forward chaining

        Args:
            X: Feature matrix
            y: Target values
            model: Model to validate
            n_splits: Number of splits
            metric: Evaluation metric
            gap: Gap between train and test sets

        Returns:
            Validation result
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        scores = []
        training_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Get predictions
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            # Calculate scores
            test_score = self._calculate_metric(y_test, y_pred_test, metric)
            train_score = self._calculate_metric(y_train, y_pred_train, metric)

            scores.append(test_score)
            training_scores.append(train_score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_lower, ci_upper = self._calculate_confidence_interval(scores)

        return ValidationResult(
            validation_method="Time Series Cross-Validation",
            n_splits=n_splits,
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=(ci_lower, ci_upper),
            metric_name=metric,
            training_scores=training_scores
        )

    def walk_forward_validation(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                model: Any,
                                initial_window: int,
                                step_size: int = 1,
                                metric: str = 'mse') -> ValidationResult:
        """
        Walk-forward validation for time series

        Args:
            X: Feature matrix
            y: Target values
            model: Model to validate
            initial_window: Initial training window size
            step_size: Step size for moving window
            metric: Evaluation metric

        Returns:
            Validation result
        """
        if initial_window >= len(X):
            raise ValueError("Initial window size must be less than data length")

        scores = []
        training_scores = []
        n_tests = (len(X) - initial_window) // step_size

        for i in range(n_tests):
            train_end = initial_window + i * step_size
            test_start = train_end
            test_end = min(test_start + step_size, len(X))

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            if len(X_test) == 0:
                break

            # Train model
            model.fit(X_train, y_train)

            # Get predictions
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train[-len(X_test):])  # Last part of training set

            # Calculate scores
            test_score = self._calculate_metric(y_test, y_pred_test, metric)
            train_score = self._calculate_metric(y_train[-len(X_test):], y_pred_train, metric)

            scores.append(test_score)
            training_scores.append(train_score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_lower, ci_upper = self._calculate_confidence_interval(scores)

        return ValidationResult(
            validation_method="Walk-Forward Validation",
            n_splits=len(scores),
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=(ci_lower, ci_upper),
            metric_name=metric,
            training_scores=training_scores
        )

    def bootstrap_validation(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             model: Any,
                             n_bootstrap: int = 1000,
                             metric: str = 'mse',
                             confidence_level: float = 0.95) -> BootstrapResult:
        """
        Bootstrap validation for model performance estimation

        Args:
            X: Feature matrix
            y: Target values
            model: Model to validate
            n_bootstrap: Number of bootstrap samples
            metric: Evaluation metric
            confidence_level: Confidence level for intervals

        Returns:
            Bootstrap result
        """
        def bootstrap_score(data):
            """Calculate score for bootstrap sample"""
            n_samples = len(data)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)

            if len(oob_indices) == 0:  # Fallback if no OOB samples
                oob_indices = indices[:len(indices)//4]

            X_boot, y_boot = data[0][indices], data[1][indices]
            X_oob, y_oob = data[0][oob_indices], data[1][oob_indices]

            # Train on bootstrap sample
            model.fit(X_boot, y_boot)
            y_pred = model.predict(X_oob)

            return self._calculate_metric(y_oob, y_pred, metric)

        # Original model performance
        model.fit(X, y)
        y_pred_orig = model.predict(X)
        original_score = self._calculate_metric(y, y_pred_orig, metric)

        # Bootstrap resampling
        data = (X, y)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            try:
                score = bootstrap_score(data)
                bootstrap_scores.append(score)
            except Exception:
                continue  # Skip failed bootstrap samples

        bootstrap_scores = np.array(bootstrap_scores)

        # Calculate statistics
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

        bias = np.mean(bootstrap_scores) - original_score
        standard_error = np.std(bootstrap_scores)

        return BootstrapResult(
            original_statistic=original_score,
            bootstrap_statistics=bootstrap_scores,
            confidence_interval=(ci_lower, ci_upper),
            bias=bias,
            standard_error=standard_error
        )

    def learning_curve_analysis(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                model: Any,
                                train_sizes: Optional[np.ndarray] = None,
                                metric: str = 'mse') -> Dict[str, np.ndarray]:
        """
        Analyze learning curves to detect overfitting/underfitting

        Args:
            X: Feature matrix
            y: Target values
            model: Model to analyze
            train_sizes: Training set sizes to evaluate
            metric: Evaluation metric

        Returns:
            Dictionary with learning curve data
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Use sklearn's learning_curve but adapt scoring
        scoring = self._get_sklearn_scoring(metric)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring=scoring,
            random_state=self.random_state
        )

        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }

    def validation_curve_analysis(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  model: Any,
                                  param_name: str,
                                  param_range: np.ndarray,
                                  metric: str = 'mse') -> Dict[str, np.ndarray]:
        """
        Analyze validation curves for hyperparameter tuning

        Args:
            X: Feature matrix
            y: Target values
            model: Model to analyze
            param_name: Parameter name to vary
            param_range: Range of parameter values
            metric: Evaluation metric

        Returns:
            Dictionary with validation curve data
        """
        scoring = self._get_sklearn_scoring(metric)

        train_scores, val_scores = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=5,
            scoring=scoring
        )

        return {
            'param_range': param_range,
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1)
        }

    def comprehensive_cross_validation(self,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       model: Any,
                                       metrics: List[str] = ['mse', 'mae', 'r2'],
                                       is_time_series: bool = True) -> CrossValidationSummary:
        """
        Comprehensive cross-validation using multiple methods

        Args:
            X: Feature matrix
            y: Target values
            model: Model to validate
            metrics: List of evaluation metrics
            is_time_series: Whether data is time series

        Returns:
            Cross-validation summary
        """
        results = {}

        # Time series cross-validation
        if is_time_series:
            for metric in metrics:
                try:
                    result = self.time_series_cross_validation(X, y, model, metric=metric)
                    results[f'time_series_{metric}'] = result
                except Exception as e:
                    warnings.warn(f"Time series CV failed for {metric}: {e}")

            # Walk-forward validation
            try:
                initial_window = min(len(X) // 3, 100)
                wf_result = self.walk_forward_validation(X, y, model, initial_window, metric=metrics[0])
                results['walk_forward'] = wf_result
            except Exception as e:
                warnings.warn(f"Walk-forward validation failed: {e}")

        else:
            # Regular k-fold for non-time series data
            for metric in metrics:
                try:
                    kf_result = self._k_fold_validation(X, y, model, metric=metric)
                    results[f'k_fold_{metric}'] = kf_result
                except Exception as e:
                    warnings.warn(f"K-fold CV failed for {metric}: {e}")

        # Bootstrap validation
        try:
            bootstrap_result = self.bootstrap_validation(X, y, model, metric=metrics[0])
            results['bootstrap'] = ValidationResult(
                validation_method="Bootstrap",
                n_splits=len(bootstrap_result.bootstrap_statistics),
                scores=bootstrap_result.bootstrap_statistics.tolist(),
                mean_score=np.mean(bootstrap_result.bootstrap_statistics),
                std_score=bootstrap_result.standard_error,
                confidence_interval=bootstrap_result.confidence_interval,
                metric_name=metrics[0]
            )
        except Exception as e:
            warnings.warn(f"Bootstrap validation failed: {e}")

        # Determine best method and detect overfitting
        best_method, best_score = self._find_best_method(results)
        overfitting_detected = self._detect_overfitting(results)
        stability_score = self._calculate_stability_score(results)
        recommendation = self._generate_recommendation(results, overfitting_detected, stability_score)

        return CrossValidationSummary(
            validation_results=results,
            best_method=best_method,
            best_score=best_score,
            overfitting_detected=overfitting_detected,
            stability_score=stability_score,
            recommendation=recommendation
        )

    def neuromorphic_quantum_specific_validation(self,
                                                 neuromorphic_predictions: np.ndarray,
                                                 quantum_predictions: np.ndarray,
                                                 combined_predictions: np.ndarray,
                                                 actual_targets: np.ndarray) -> Dict[str, Any]:
        """
        Validation specific to neuromorphic-quantum system

        Args:
            neuromorphic_predictions: Predictions from neuromorphic component
            quantum_predictions: Predictions from quantum component
            combined_predictions: Combined system predictions
            actual_targets: Actual target values

        Returns:
            Dictionary with validation results
        """
        results = {}

        # Individual component validation
        results['neuromorphic'] = self._component_validation(neuromorphic_predictions, actual_targets, 'Neuromorphic')
        results['quantum'] = self._component_validation(quantum_predictions, actual_targets, 'Quantum')
        results['combined'] = self._component_validation(combined_predictions, actual_targets, 'Combined')

        # Complementarity analysis
        results['complementarity'] = self._analyze_complementarity(
            neuromorphic_predictions, quantum_predictions, actual_targets
        )

        # Information content analysis
        results['information_content'] = self._analyze_information_content(
            neuromorphic_predictions, quantum_predictions, combined_predictions, actual_targets
        )

        # Ensemble effectiveness
        results['ensemble_effectiveness'] = self._analyze_ensemble_effectiveness(
            neuromorphic_predictions, quantum_predictions, combined_predictions, actual_targets
        )

        return results

    def _component_validation(self, predictions: np.ndarray, targets: np.ndarray, component_name: str) -> Dict[str, float]:
        """Validate individual component"""
        return {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            'correlation': np.corrcoef(targets, predictions)[0, 1] if len(predictions) > 1 else 0.0,
            'rmse': np.sqrt(mean_squared_error(targets, predictions))
        }

    def _analyze_complementarity(self, neuro_pred: np.ndarray, quantum_pred: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Analyze complementarity between neuromorphic and quantum predictions"""
        # Calculate residuals
        neuro_residuals = targets - neuro_pred
        quantum_residuals = targets - quantum_pred

        # Complementarity metrics
        residual_correlation = np.corrcoef(neuro_residuals, quantum_residuals)[0, 1]

        # Diversification benefit
        individual_mse = (mean_squared_error(targets, neuro_pred) + mean_squared_error(targets, quantum_pred)) / 2
        combined_pred = (neuro_pred + quantum_pred) / 2
        combined_mse = mean_squared_error(targets, combined_pred)
        diversification_benefit = (individual_mse - combined_mse) / individual_mse

        return {
            'residual_correlation': residual_correlation,
            'diversification_benefit': diversification_benefit,
            'complementarity_score': 1 - abs(residual_correlation)  # Higher score = more complementary
        }

    def _analyze_information_content(self, neuro_pred: np.ndarray, quantum_pred: np.ndarray,
                                   combined_pred: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Analyze information content of different components"""
        # Calculate predictive R²
        neuro_r2 = r2_score(targets, neuro_pred)
        quantum_r2 = r2_score(targets, quantum_pred)
        combined_r2 = r2_score(targets, combined_pred)

        # Information gain from combination
        information_gain = combined_r2 - max(neuro_r2, quantum_r2)

        # Unique information content
        neuro_unique = self._calculate_unique_information(neuro_pred, quantum_pred, targets)
        quantum_unique = self._calculate_unique_information(quantum_pred, neuro_pred, targets)

        return {
            'neuromorphic_r2': neuro_r2,
            'quantum_r2': quantum_r2,
            'combined_r2': combined_r2,
            'information_gain': information_gain,
            'neuromorphic_unique_info': neuro_unique,
            'quantum_unique_info': quantum_unique
        }

    def _calculate_unique_information(self, pred1: np.ndarray, pred2: np.ndarray, targets: np.ndarray) -> float:
        """Calculate unique information content of pred1 given pred2"""
        # Partial correlation: correlation of pred1 with targets after removing effect of pred2
        try:
            from sklearn.linear_model import LinearRegression

            # Remove effect of pred2 from targets
            reg = LinearRegression()
            reg.fit(pred2.reshape(-1, 1), targets)
            targets_residual = targets - reg.predict(pred2.reshape(-1, 1))

            # Remove effect of pred2 from pred1
            reg.fit(pred2.reshape(-1, 1), pred1)
            pred1_residual = pred1 - reg.predict(pred2.reshape(-1, 1))

            # Correlation of residuals
            partial_corr = np.corrcoef(targets_residual, pred1_residual)[0, 1]
            return partial_corr**2  # Squared partial correlation
        except Exception:
            return 0.0

    def _analyze_ensemble_effectiveness(self, neuro_pred: np.ndarray, quantum_pred: np.ndarray,
                                      combined_pred: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Analyze effectiveness of ensemble combination"""
        # Simple average ensemble
        simple_avg = (neuro_pred + quantum_pred) / 2
        simple_avg_mse = mean_squared_error(targets, simple_avg)

        # Actual combined MSE
        combined_mse = mean_squared_error(targets, combined_pred)

        # Individual MSEs
        neuro_mse = mean_squared_error(targets, neuro_pred)
        quantum_mse = mean_squared_error(targets, quantum_pred)

        # Ensemble effectiveness metrics
        ensemble_improvement = (min(neuro_mse, quantum_mse) - combined_mse) / min(neuro_mse, quantum_mse)
        optimal_vs_actual = (simple_avg_mse - combined_mse) / simple_avg_mse

        return {
            'ensemble_improvement': ensemble_improvement,
            'optimal_vs_actual': optimal_vs_actual,
            'is_better_than_best_individual': combined_mse < min(neuro_mse, quantum_mse),
            'is_better_than_simple_average': combined_mse < simple_avg_mse
        }

    def _k_fold_validation(self, X: np.ndarray, y: np.ndarray, model: Any,
                          n_splits: int = 5, metric: str = 'mse') -> ValidationResult:
        """Standard k-fold cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        training_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            test_score = self._calculate_metric(y_test, y_pred_test, metric)
            train_score = self._calculate_metric(y_train, y_pred_train, metric)

            scores.append(test_score)
            training_scores.append(train_score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_lower, ci_upper = self._calculate_confidence_interval(scores)

        return ValidationResult(
            validation_method=f"{n_splits}-Fold Cross-Validation",
            n_splits=n_splits,
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=(ci_lower, ci_upper),
            metric_name=metric,
            training_scores=training_scores
        )

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Calculate evaluation metric"""
        if metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'correlation':
            return np.corrcoef(y_true, y_pred)[0, 1] if len(y_pred) > 1 else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _get_sklearn_scoring(self, metric: str) -> str:
        """Convert metric name to sklearn scoring format"""
        mapping = {
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'rmse': 'neg_root_mean_squared_error'
        }
        return mapping.get(metric, 'neg_mean_squared_error')

    def _calculate_confidence_interval(self, scores: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for scores"""
        alpha = 1 - confidence_level
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        sem = stats.sem(scores_array)
        h = sem * stats.t.ppf(1 - alpha/2, len(scores) - 1)
        return mean_score - h, mean_score + h

    def _find_best_method(self, results: Dict[str, ValidationResult]) -> Tuple[str, float]:
        """Find best validation method based on scores"""
        if not results:
            return "", 0.0

        # For regression metrics, lower is better (except R²)
        best_method = ""
        best_score = float('inf')

        for method_name, result in results.items():
            if result.metric_name == 'r2':
                # Higher R² is better
                if result.mean_score > -best_score:
                    best_method = method_name
                    best_score = -result.mean_score
            else:
                # Lower error is better
                if result.mean_score < best_score:
                    best_method = method_name
                    best_score = result.mean_score

        return best_method, best_score if best_score != float('inf') else 0.0

    def _detect_overfitting(self, results: Dict[str, ValidationResult]) -> bool:
        """Detect overfitting from validation results"""
        for result in results.values():
            if result.training_scores is not None:
                # Check if training scores are much better than validation scores
                mean_train = np.mean(result.training_scores)
                mean_val = result.mean_score

                # For error metrics, lower is better
                if result.metric_name != 'r2':
                    if mean_train < mean_val * 0.7:  # Training error much smaller
                        return True
                else:
                    if mean_train > mean_val * 1.3:  # Training R² much higher
                        return True

        return False

    def _calculate_stability_score(self, results: Dict[str, ValidationResult]) -> float:
        """Calculate stability score across validation methods"""
        if not results:
            return 0.0

        scores = [result.mean_score for result in results.values()]
        if len(scores) <= 1:
            return 1.0

        # Coefficient of variation as stability measure
        cv = np.std(scores) / np.mean(np.abs(scores)) if np.mean(np.abs(scores)) > 0 else 0
        return max(0, 1 - cv)  # Higher stability = lower coefficient of variation

    def _generate_recommendation(self, results: Dict[str, ValidationResult],
                               overfitting_detected: bool, stability_score: float) -> str:
        """Generate recommendation based on validation results"""
        recommendations = []

        if overfitting_detected:
            recommendations.append("Model shows signs of overfitting - consider regularization or feature selection")

        if stability_score < 0.7:
            recommendations.append("Low stability across validation methods - results may not be reliable")

        if not results:
            recommendations.append("No successful validation results - check data and model configuration")

        if len(results) < 2:
            recommendations.append("Limited validation methods completed - consider additional validation approaches")

        best_score = max([r.mean_score for r in results.values()]) if results else 0
        if results and list(results.values())[0].metric_name != 'r2' and best_score > 1.0:
            recommendations.append("High error values detected - consider model improvement")

        if not recommendations:
            recommendations.append("Validation results look good - model appears to be performing well")

        return "; ".join(recommendations)


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

    # Generate example data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # Create model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Initialize validator
    validator = NeuromorphicQuantumValidator()

    # Run comprehensive validation
    cv_summary = validator.comprehensive_cross_validation(X, y, model, is_time_series=False)

    print("Cross-Validation Summary:")
    print(f"Best method: {cv_summary.best_method}")
    print(f"Best score: {cv_summary.best_score:.4f}")
    print(f"Overfitting detected: {cv_summary.overfitting_detected}")
    print(f"Stability score: {cv_summary.stability_score:.3f}")
    print(f"Recommendation: {cv_summary.recommendation}")

    # Example neuromorphic-quantum specific validation
    n_samples = 500
    np.random.seed(42)
    neuro_pred = np.random.normal(0, 1, n_samples)
    quantum_pred = neuro_pred + np.random.normal(0, 0.5, n_samples)
    combined_pred = 0.6 * neuro_pred + 0.4 * quantum_pred
    targets = neuro_pred + 0.5 * quantum_pred + np.random.normal(0, 0.2, n_samples)

    nq_validation = validator.neuromorphic_quantum_specific_validation(
        neuro_pred, quantum_pred, combined_pred, targets
    )

    print("\nNeuromorphic-Quantum Specific Validation:")
    print(f"Combined R²: {nq_validation['combined']['r2']:.4f}")
    print(f"Information gain: {nq_validation['information_content']['information_gain']:.4f}")
    print(f"Complementarity score: {nq_validation['complementarity']['complementarity_score']:.4f}")