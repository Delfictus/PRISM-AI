#!/usr/bin/env python3
"""
Data Preprocessing and Feature Engineering Pipeline
for Neuromorphic-Quantum Computing Platform

This module provides comprehensive data preprocessing, cleaning, validation,
and feature engineering capabilities for the neuromorphic-quantum platform.

Author: Data Scientist Agent
Date: 2025-09-28
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import hilbert, butter, sosfilt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import json

class ScalingMethod(Enum):
    """Available scaling methods"""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"

class FeatureType(Enum):
    """Types of features that can be extracted"""
    TEMPORAL = "temporal"
    SPECTRAL = "spectral"
    STATISTICAL = "statistical"
    NEUROMORPHIC = "neuromorphic"
    QUANTUM = "quantum"

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    missing_data_percentage: float
    outlier_percentage: float
    duplicate_percentage: float
    data_completeness_score: float
    temporal_consistency_score: float
    feature_correlation_max: float
    signal_to_noise_ratio: float
    quality_score: float  # Overall quality score 0-1

@dataclass
class FeatureImportanceResult:
    """Feature importance analysis result"""
    feature_names: List[str]
    importance_scores: List[float]
    selection_method: str
    selected_features: List[str]
    explained_variance_ratio: Optional[float] = None

class NeuromorphicQuantumDataProcessor:
    """
    Comprehensive data preprocessing and feature engineering pipeline
    for neuromorphic-quantum systems
    """

    def __init__(self,
                 scaling_method: ScalingMethod = ScalingMethod.ROBUST,
                 handle_missing: str = 'interpolate',
                 outlier_method: str = 'iqr',
                 feature_selection_k: int = 50):
        """
        Initialize data processor

        Args:
            scaling_method: Method for feature scaling
            handle_missing: How to handle missing data ('interpolate', 'forward_fill', 'drop')
            outlier_method: Method for outlier detection ('iqr', 'z_score', 'isolation_forest')
            feature_selection_k: Number of features to select
        """
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        self.outlier_method = outlier_method
        self.feature_selection_k = feature_selection_k

        # Initialize scalers
        self.scaler = self._get_scaler()
        self.feature_selector = None
        self.pca_transformer = None

        # Processing statistics
        self.processing_stats = {}

    def _get_scaler(self):
        """Get appropriate scaler based on method"""
        if self.scaling_method == ScalingMethod.STANDARD:
            return StandardScaler()
        elif self.scaling_method == ScalingMethod.ROBUST:
            return RobustScaler()
        elif self.scaling_method == ScalingMethod.MINMAX:
            return MinMaxScaler()
        else:
            return None

    def assess_data_quality(self, data: pd.DataFrame) -> DataQualityMetrics:
        """
        Comprehensive data quality assessment

        Args:
            data: Input dataframe

        Returns:
            Data quality metrics
        """
        # Missing data analysis
        missing_percentage = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100

        # Outlier detection (using IQR method)
        outlier_count = 0
        total_values = 0
        for col in data.select_dtypes(include=[np.number]).columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_count += outliers
            total_values += len(data[col].dropna())

        outlier_percentage = (outlier_count / total_values) * 100 if total_values > 0 else 0

        # Duplicate analysis
        duplicate_percentage = (data.duplicated().sum() / len(data)) * 100

        # Data completeness
        completeness_score = 1.0 - (missing_percentage / 100.0)

        # Temporal consistency (if timestamp column exists)
        temporal_consistency = 1.0
        if 'timestamp' in data.columns:
            time_diffs = pd.to_datetime(data['timestamp']).diff().dropna()
            if len(time_diffs) > 1:
                cv_time = time_diffs.std() / time_diffs.mean()
                temporal_consistency = max(0, 1.0 - cv_time.total_seconds() / 3600)  # Normalize by hour

        # Feature correlation analysis
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr().abs()
            # Get maximum correlation excluding diagonal
            np.fill_diagonal(corr_matrix.values, 0)
            max_correlation = corr_matrix.max().max()
        else:
            max_correlation = 0

        # Signal-to-noise ratio estimation
        snr = self._estimate_signal_to_noise(numeric_data)

        # Overall quality score
        quality_score = (
            0.3 * completeness_score +
            0.2 * max(0, 1.0 - outlier_percentage / 20) +  # Penalize >20% outliers
            0.2 * max(0, 1.0 - duplicate_percentage / 10) +  # Penalize >10% duplicates
            0.15 * temporal_consistency +
            0.15 * min(1.0, snr / 10)  # Normalize SNR to 0-1 scale
        )

        return DataQualityMetrics(
            missing_data_percentage=missing_percentage,
            outlier_percentage=outlier_percentage,
            duplicate_percentage=duplicate_percentage,
            data_completeness_score=completeness_score,
            temporal_consistency_score=temporal_consistency,
            feature_correlation_max=max_correlation,
            signal_to_noise_ratio=snr,
            quality_score=quality_score
        )

    def _estimate_signal_to_noise(self, data: pd.DataFrame) -> float:
        """Estimate signal-to-noise ratio of data"""
        if data.empty:
            return 0.0

        # Use first column as example
        col = data.columns[0]
        signal_data = data[col].dropna()

        if len(signal_data) < 10:
            return 0.0

        # Signal power (variance of data)
        signal_power = signal_data.var()

        # Noise estimation using high-frequency components
        if len(signal_data) > 20:
            # Use residuals from polynomial fit as noise estimate
            x = np.arange(len(signal_data))
            poly_coeffs = np.polyfit(x, signal_data, min(3, len(signal_data) // 10))
            polynomial = np.polyval(poly_coeffs, x)
            residuals = signal_data - polynomial
            noise_power = residuals.var()
        else:
            # Use simple noise estimate
            noise_power = signal_power * 0.1

        if noise_power == 0:
            return float('inf')

        snr = signal_power / noise_power
        return 10 * np.log10(snr) if snr > 0 else 0

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values, outliers, and duplicates

        Args:
            data: Input dataframe

        Returns:
            Cleaned dataframe
        """
        cleaned_data = data.copy()
        cleaning_stats = {}

        # Handle duplicates
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        duplicates_removed = initial_rows - len(cleaned_data)
        cleaning_stats['duplicates_removed'] = duplicates_removed

        # Handle missing values
        missing_before = cleaned_data.isnull().sum().sum()

        if self.handle_missing == 'interpolate':
            # Use linear interpolation for numeric columns
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numeric_cols] = cleaned_data[numeric_cols].interpolate(method='linear')
        elif self.handle_missing == 'forward_fill':
            cleaned_data = cleaned_data.fillna(method='ffill')
        elif self.handle_missing == 'drop':
            cleaned_data = cleaned_data.dropna()

        missing_after = cleaned_data.isnull().sum().sum()
        cleaning_stats['missing_values_handled'] = missing_before - missing_after

        # Handle outliers
        outliers_removed = 0
        if self.outlier_method == 'iqr':
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                q1 = cleaned_data[col].quantile(0.25)
                q3 = cleaned_data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_mask = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound))
                outliers_removed += outlier_mask.sum()

                # Replace outliers with median
                cleaned_data.loc[outlier_mask, col] = cleaned_data[col].median()

        elif self.outlier_method == 'z_score':
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                z_scores = np.abs(stats.zscore(cleaned_data[col].dropna()))
                outlier_mask = z_scores > 3
                outliers_removed += outlier_mask.sum()

                # Replace outliers with median
                cleaned_data.loc[cleaned_data.index[outlier_mask], col] = cleaned_data[col].median()

        cleaning_stats['outliers_handled'] = outliers_removed

        self.processing_stats['cleaning'] = cleaning_stats
        return cleaned_data

    def extract_temporal_features(self, data: pd.DataFrame, value_col: str = 'values') -> pd.DataFrame:
        """
        Extract temporal/time-series features

        Args:
            data: Input dataframe with time series data
            value_col: Name of the value column

        Returns:
            Dataframe with temporal features
        """
        if value_col not in data.columns:
            raise ValueError(f"Column {value_col} not found in data")

        values = data[value_col].values
        features = {}

        # Basic statistical features
        features['mean'] = np.mean(values)
        features['std'] = np.std(values)
        features['skewness'] = stats.skew(values)
        features['kurtosis'] = stats.kurtosis(values)
        features['min'] = np.min(values)
        features['max'] = np.max(values)
        features['median'] = np.median(values)
        features['q25'] = np.percentile(values, 25)
        features['q75'] = np.percentile(values, 75)

        # Trend features
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        features['trend_slope'] = slope
        features['trend_r2'] = r_value**2
        features['trend_p_value'] = p_value

        # Autocorrelation features
        if len(values) > 10:
            autocorr_lags = min(10, len(values) // 4)
            autocorrs = [np.corrcoef(values[:-i], values[i:])[0, 1]
                        for i in range(1, autocorr_lags + 1)]
            for i, ac in enumerate(autocorrs):
                if not np.isnan(ac):
                    features[f'autocorr_lag_{i+1}'] = ac

        # Volatility clustering features
        if len(values) > 5:
            returns = np.diff(values) / values[:-1]
            returns = returns[~np.isnan(returns)]
            if len(returns) > 1:
                features['volatility'] = np.std(returns)
                features['return_mean'] = np.mean(returns)

                # ARCH effects (volatility clustering)
                if len(returns) > 10:
                    squared_returns = returns**2
                    arch_corr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                    if not np.isnan(arch_corr):
                        features['arch_effect'] = arch_corr

        # Seasonal decomposition features (if enough data)
        if len(values) > 24:  # Need enough points for seasonal decomposition
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomp = seasonal_decompose(values, period=min(12, len(values)//2), model='additive')
                features['seasonal_strength'] = np.std(decomp.seasonal) / np.std(values)
                features['trend_strength'] = np.std(decomp.trend[~np.isnan(decomp.trend)]) / np.std(values)
            except Exception:
                pass  # Skip if seasonal decomposition fails

        return pd.DataFrame([features])

    def extract_spectral_features(self, data: pd.DataFrame, value_col: str = 'values',
                                  sampling_rate: float = 1.0) -> pd.DataFrame:
        """
        Extract frequency domain features

        Args:
            data: Input dataframe
            value_col: Name of the value column
            sampling_rate: Sampling rate of the signal

        Returns:
            Dataframe with spectral features
        """
        if value_col not in data.columns:
            raise ValueError(f"Column {value_col} not found in data")

        values = data[value_col].values
        features = {}

        # Ensure we have enough data points
        if len(values) < 8:
            return pd.DataFrame([features])

        # Remove trend for spectral analysis
        detrended = signal.detrend(values)

        # Power spectral density
        frequencies, psd = signal.welch(detrended, fs=sampling_rate, nperseg=min(len(values)//2, 256))

        # Spectral features
        features['spectral_centroid'] = np.sum(frequencies * psd) / np.sum(psd)
        features['spectral_bandwidth'] = np.sqrt(np.sum(((frequencies - features['spectral_centroid'])**2) * psd) / np.sum(psd))
        features['spectral_rolloff'] = frequencies[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]

        # Peak frequency
        peak_freq_idx = np.argmax(psd)
        features['peak_frequency'] = frequencies[peak_freq_idx]
        features['peak_power'] = psd[peak_freq_idx]

        # Spectral entropy
        psd_normalized = psd / np.sum(psd)
        features['spectral_entropy'] = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

        # Band power features (if sampling rate allows)
        if sampling_rate >= 2:  # Nyquist frequency >= 1 Hz
            low_freq = frequencies <= sampling_rate / 4
            mid_freq = (frequencies > sampling_rate / 4) & (frequencies <= sampling_rate / 2)

            features['low_freq_power'] = np.sum(psd[low_freq])
            features['mid_freq_power'] = np.sum(psd[mid_freq])

            total_power = np.sum(psd)
            if total_power > 0:
                features['low_freq_ratio'] = features['low_freq_power'] / total_power
                features['mid_freq_ratio'] = features['mid_freq_power'] / total_power

        return pd.DataFrame([features])

    def extract_neuromorphic_features(self, neuromorphic_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract features from neuromorphic processing results

        Args:
            neuromorphic_data: Dictionary containing neuromorphic processing results

        Returns:
            Dataframe with neuromorphic features
        """
        features = {}

        # Pattern-based features
        if 'patterns' in neuromorphic_data:
            patterns = neuromorphic_data['patterns']
            features['pattern_count'] = len(patterns)

            if patterns:
                # Pattern strength statistics
                strengths = [p.get('strength', 0) for p in patterns]
                features['pattern_strength_mean'] = np.mean(strengths)
                features['pattern_strength_max'] = np.max(strengths)
                features['pattern_strength_std'] = np.std(strengths)

                # Pattern type diversity
                pattern_types = [p.get('pattern_type', '') for p in patterns]
                unique_types = set(pattern_types)
                features['pattern_type_diversity'] = len(unique_types)

                # Spatial and temporal feature aggregation
                spatial_features = []
                temporal_features = []

                for pattern in patterns:
                    if 'spatial_features' in pattern:
                        spatial_features.extend(pattern['spatial_features'])
                    if 'temporal_features' in pattern:
                        temporal_features.extend(pattern['temporal_features'])

                if spatial_features:
                    features['spatial_feature_mean'] = np.mean(spatial_features)
                    features['spatial_feature_std'] = np.std(spatial_features)

                if temporal_features:
                    features['temporal_feature_mean'] = np.mean(temporal_features)
                    features['temporal_feature_std'] = np.std(temporal_features)

        # Spike analysis features
        if 'spike_analysis' in neuromorphic_data:
            spike_data = neuromorphic_data['spike_analysis']
            features['spike_count'] = spike_data.get('spike_count', 0)
            features['spike_rate'] = spike_data.get('spike_rate', 0)
            features['spike_coherence'] = spike_data.get('coherence', 0)

            if 'dynamics' in spike_data:
                dynamics = spike_data['dynamics']
                features['spike_dynamics_mean'] = np.mean(dynamics)
                features['spike_dynamics_std'] = np.std(dynamics)
                features['spike_dynamics_entropy'] = stats.entropy(np.abs(dynamics) + 1e-12)

        # Reservoir state features
        if 'reservoir_state' in neuromorphic_data:
            reservoir = neuromorphic_data['reservoir_state']
            features['reservoir_avg_activation'] = reservoir.get('avg_activation', 0)
            features['reservoir_memory_capacity'] = reservoir.get('memory_capacity', 0)
            features['reservoir_separation'] = reservoir.get('separation', 0)

            if 'activations' in reservoir:
                activations = reservoir['activations']
                features['reservoir_activation_std'] = np.std(activations)
                features['reservoir_activation_skew'] = stats.skew(activations)
                features['reservoir_activation_entropy'] = stats.entropy(np.abs(activations) + 1e-12)

        return pd.DataFrame([features])

    def extract_quantum_features(self, quantum_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract features from quantum processing results

        Args:
            quantum_data: Dictionary containing quantum processing results

        Returns:
            Dataframe with quantum features
        """
        features = {}

        # Basic quantum state features
        features['quantum_energy'] = quantum_data.get('energy', 0)
        features['quantum_phase_coherence'] = quantum_data.get('phase_coherence', 0)

        # Convergence features
        if 'convergence' in quantum_data:
            convergence = quantum_data['convergence']
            features['quantum_converged'] = float(convergence.get('converged', False))
            features['quantum_iterations'] = convergence.get('iterations', 0)
            features['quantum_final_error'] = convergence.get('final_error', 0)
            features['quantum_energy_drift'] = convergence.get('energy_drift', 0)

        # Quantum state features
        if 'state_features' in quantum_data:
            state_features = quantum_data['state_features']
            features['quantum_state_mean'] = np.mean(state_features)
            features['quantum_state_std'] = np.std(state_features)
            features['quantum_state_entropy'] = stats.entropy(np.abs(state_features) + 1e-12)

            # Quantum correlation features
            if len(state_features) > 1:
                # Autocorrelation of quantum state
                autocorr = np.corrcoef(state_features[:-1], state_features[1:])[0, 1]
                if not np.isnan(autocorr):
                    features['quantum_state_autocorr'] = autocorr

        return pd.DataFrame([features])

    def feature_selection(self, X: pd.DataFrame, y: np.ndarray,
                         method: str = 'mutual_info') -> FeatureImportanceResult:
        """
        Perform feature selection

        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('mutual_info', 'f_score', 'pca')

        Returns:
            Feature importance result
        """
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(self.feature_selection_k, X.shape[1]))
        elif method == 'f_score':
            selector = SelectKBest(score_func=f_regression, k=min(self.feature_selection_k, X.shape[1]))
        else:
            raise ValueError(f"Unknown selection method: {method}")

        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support()
        selected_features = X.columns[selected_indices].tolist()
        importance_scores = selector.scores_.tolist()

        self.feature_selector = selector

        return FeatureImportanceResult(
            feature_names=X.columns.tolist(),
            importance_scores=importance_scores,
            selection_method=method,
            selected_features=selected_features
        )

    def create_feature_pipeline(self, raw_data: pd.DataFrame,
                               neuromorphic_data: Optional[Dict[str, Any]] = None,
                               quantum_data: Optional[Dict[str, Any]] = None,
                               value_col: str = 'values') -> pd.DataFrame:
        """
        Complete feature engineering pipeline

        Args:
            raw_data: Raw input data
            neuromorphic_data: Neuromorphic processing results
            quantum_data: Quantum processing results
            value_col: Name of value column in raw data

        Returns:
            Engineered feature dataframe
        """
        # Start with temporal features from raw data
        features_list = []

        # Extract temporal features
        try:
            temporal_features = self.extract_temporal_features(raw_data, value_col)
            features_list.append(temporal_features)
        except Exception as e:
            warnings.warn(f"Could not extract temporal features: {e}")

        # Extract spectral features
        try:
            spectral_features = self.extract_spectral_features(raw_data, value_col)
            features_list.append(spectral_features)
        except Exception as e:
            warnings.warn(f"Could not extract spectral features: {e}")

        # Extract neuromorphic features
        if neuromorphic_data:
            try:
                neuro_features = self.extract_neuromorphic_features(neuromorphic_data)
                features_list.append(neuro_features)
            except Exception as e:
                warnings.warn(f"Could not extract neuromorphic features: {e}")

        # Extract quantum features
        if quantum_data:
            try:
                quantum_features = self.extract_quantum_features(quantum_data)
                features_list.append(quantum_features)
            except Exception as e:
                warnings.warn(f"Could not extract quantum features: {e}")

        # Combine all features
        if features_list:
            combined_features = pd.concat(features_list, axis=1)
        else:
            combined_features = pd.DataFrame()

        # Apply scaling if specified
        if self.scaler is not None and not combined_features.empty:
            scaled_features = self.scaler.fit_transform(combined_features)
            combined_features = pd.DataFrame(scaled_features, columns=combined_features.columns)

        return combined_features

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics and metadata"""
        return self.processing_stats.copy()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate example data
    n_samples = 1000
    time_index = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

    # Create sample data with trend and seasonality
    trend = np.linspace(100, 120, n_samples)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily seasonality
    noise = np.random.normal(0, 2, n_samples)
    values = trend + seasonal + noise

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': time_index,
        'values': values,
        'source': 'test_data'
    })

    # Initialize processor
    processor = NeuromorphicQuantumDataProcessor()

    # Assess data quality
    quality_metrics = processor.assess_data_quality(data)
    print("Data Quality Assessment:")
    print(f"  Overall Quality Score: {quality_metrics.quality_score:.3f}")
    print(f"  Missing Data: {quality_metrics.missing_data_percentage:.1f}%")
    print(f"  Outliers: {quality_metrics.outlier_percentage:.1f}%")
    print(f"  Signal-to-Noise Ratio: {quality_metrics.signal_to_noise_ratio:.2f} dB")
    print()

    # Clean data
    cleaned_data = processor.clean_data(data)
    print("Data cleaned successfully")

    # Extract features
    features = processor.create_feature_pipeline(cleaned_data)
    print(f"Extracted {features.shape[1]} features from {features.shape[0]} samples")
    print("Feature types:", features.columns.tolist()[:10], "...")  # Show first 10 features