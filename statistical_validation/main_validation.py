#!/usr/bin/env python3
"""
Main Integration Script for Neuromorphic-Quantum Statistical Validation

This script provides a complete example of how to use the statistical validation
framework with the neuromorphic-quantum computing platform.

Author: Data Scientist Agent
Date: 2025-09-28
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the validation framework to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reports.statistical_report_generator import NeuromorphicQuantumStatisticalReportGenerator

def create_sample_neuromorphic_results(n_samples: int) -> dict:
    """
    Create sample neuromorphic processing results

    Args:
        n_samples: Number of samples

    Returns:
        Dictionary with neuromorphic results
    """
    # Generate realistic neuromorphic patterns
    patterns = []
    n_patterns = np.random.poisson(5)  # Average 5 patterns

    pattern_types = ['synchronous', 'traveling', 'standing', 'emergent', 'rhythmic', 'sparse', 'burst', 'chaotic']

    for i in range(n_patterns):
        pattern = {
            'pattern_type': np.random.choice(pattern_types),
            'strength': np.random.beta(2, 1),  # Biased toward higher strengths
            'spatial_features': np.random.normal(0, 1, 8).tolist(),
            'temporal_features': np.random.normal(0, 1, 6).tolist()
        }
        patterns.append(pattern)

    # Generate spike analysis
    base_rate = 50 + np.random.normal(0, 10)
    spike_analysis = {
        'spike_count': int(base_rate * n_samples / 100),
        'spike_rate': max(0, base_rate),
        'coherence': np.random.beta(3, 1),  # High coherence expected
        'dynamics': np.random.normal(0.5, 0.2, 12).tolist()
    }

    # Generate reservoir state
    reservoir_state = {
        'activations': np.random.normal(0.4, 0.3, 500).tolist(),
        'avg_activation': np.random.normal(0.4, 0.1),
        'memory_capacity': np.random.beta(4, 2),  # Good memory capacity
        'separation': np.random.beta(3, 2)  # Good separation property
    }

    # Generate predictions with some skill
    trend_component = np.linspace(0, 0.02, n_samples)
    noise_component = np.random.normal(0, 0.01, n_samples)
    predictions = trend_component + noise_component

    return {
        'patterns': patterns,
        'spike_analysis': spike_analysis,
        'reservoir_state': reservoir_state,
        'predictions': predictions
    }

def create_sample_quantum_results(n_samples: int) -> dict:
    """
    Create sample quantum processing results

    Args:
        n_samples: Number of samples

    Returns:
        Dictionary with quantum results
    """
    # Generate quantum state features
    state_features = np.random.normal(0, 1, 25)

    # Generate convergence info
    convergence = {
        'converged': np.random.choice([True, False], p=[0.9, 0.1]),  # Usually converges
        'iterations': np.random.poisson(30) + 10,  # 10-50 iterations typically
        'final_error': np.random.exponential(1e-8),  # Small final error
        'energy_drift': np.random.exponential(1e-12)  # Very small drift
    }

    # Generate main quantum metrics
    energy = -10 - np.random.exponential(5)  # Negative energy (bound state)
    phase_coherence = np.random.beta(4, 1)  # High coherence expected

    # Generate predictions complementary to neuromorphic
    base_signal = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.01
    quantum_noise = np.random.normal(0, 0.008, n_samples)
    predictions = base_signal + quantum_noise

    return {
        'energy': energy,
        'phase_coherence': phase_coherence,
        'convergence': convergence,
        'state_features': state_features.tolist(),
        'predictions': predictions
    }

def create_sample_market_data(n_samples: int) -> pd.DataFrame:
    """
    Create realistic market data with different regimes

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with market data
    """
    # Create timestamp series
    start_date = datetime.now() - timedelta(days=n_samples // 24)
    timestamps = pd.date_range(start_date, periods=n_samples, freq='1H')

    # Generate returns with regime changes
    returns = np.zeros(n_samples)
    volatilities = np.zeros(n_samples)

    # Define regime periods
    regime_length = n_samples // 4

    for i, t in enumerate(timestamps):
        regime = i // regime_length

        if regime == 0:  # Low volatility period
            vol = 0.01
            drift = 0.0002
        elif regime == 1:  # High volatility period
            vol = 0.035
            drift = -0.0005
        elif regime == 2:  # Bull market
            vol = 0.02
            drift = 0.003
        else:  # Normal period
            vol = 0.025
            drift = 0.0001

        volatilities[i] = vol
        returns[i] = np.random.normal(drift, vol)

    # Generate price series
    initial_price = 100.0
    prices = initial_price * np.exp(np.cumsum(returns))

    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'values': prices,
        'returns': returns,
        'volatility': volatilities,
        'source': 'synthetic_market_data'
    })

    return data

def main():
    """
    Main function demonstrating complete statistical validation workflow
    """
    print("🧠⚛️📊 Neuromorphic-Quantum Statistical Validation Framework")
    print("=" * 70)
    print("Comprehensive Validation Example")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Configuration
    n_samples = 2000
    print(f"📊 Generating {n_samples} samples of synthetic data...")

    # 1. Create sample data
    market_data = create_sample_market_data(n_samples)
    neuromorphic_results = create_sample_neuromorphic_results(n_samples)
    quantum_results = create_sample_quantum_results(n_samples)

    # 2. Create combined predictions (weighted ensemble)
    neuro_weight = 0.6
    quantum_weight = 0.4

    combined_predictions = (neuro_weight * neuromorphic_results['predictions'] +
                          quantum_weight * quantum_results['predictions'])

    # 3. Create target values (next period returns scaled to percentage)
    targets = market_data['returns'].shift(-1).dropna() * 100  # Next period returns as %

    # Align lengths
    market_data = market_data[:-1]  # Remove last row
    combined_predictions = combined_predictions[:-1]
    neuromorphic_results['predictions'] = neuromorphic_results['predictions'][:-1]
    quantum_results['predictions'] = quantum_results['predictions'][:-1]

    print(f"✅ Data generation complete:")
    print(f"   - Market data: {len(market_data)} samples")
    print(f"   - Neuromorphic patterns: {len(neuromorphic_results['patterns'])}")
    print(f"   - Quantum coherence: {quantum_results['phase_coherence']:.3f}")
    print(f"   - Combined predictions range: [{combined_predictions.min():.4f}, {combined_predictions.max():.4f}]")
    print()

    # 4. Initialize report generator
    print("🚀 Initializing Statistical Validation Framework...")
    report_generator = NeuromorphicQuantumStatisticalReportGenerator(
        output_directory="./validation_reports"
    )

    # 5. Generate comprehensive report
    print("⚡ Running Comprehensive Statistical Validation...")
    print("   This may take a few minutes for thorough analysis...")
    print()

    try:
        report = report_generator.generate_comprehensive_report(
            data=market_data,
            neuromorphic_results=neuromorphic_results,
            quantum_results=quantum_results,
            combined_predictions=combined_predictions,
            targets=targets.values,
            report_title="Comprehensive Neuromorphic-Quantum Platform Validation"
        )

        # 6. Display key results
        print()
        print("🎉 VALIDATION COMPLETE - KEY RESULTS:")
        print("=" * 50)

        exec_summary = report['executive_summary']
        print(f"📋 FINAL VERDICT:")
        print(f"   {exec_summary['overall_conclusion']}")
        print()

        print(f"📊 CONFIDENCE METRICS:")
        print(f"   Overall Confidence: {exec_summary['confidence_level']:.1f}%")
        print(f"   Statistical Significance: {'YES' if exec_summary['statistical_significance'] else 'NO'}")
        print()

        print(f"🎯 RECOMMENDATION:")
        print(f"   {exec_summary['recommendation']}")
        print()

        print(f"⚠️  RISK ASSESSMENT:")
        print(f"   {exec_summary['risk_assessment']}")
        print()

        print(f"🔍 KEY FINDINGS:")
        for i, finding in enumerate(exec_summary['key_findings'][:5], 1):
            print(f"   {i}. {finding}")

        # Display robustness summary
        robustness = report['robustness_analysis']
        print()
        print(f"🛡️  ROBUSTNESS SUMMARY:")
        print(f"   Overall Robustness: {robustness['overall_robustness_score']:.2f}/1.00")
        print(f"   Cross-Regime Stability: {robustness['cross_regime_stability']:.2f}/1.00")
        print(f"   Regime Performance Tests: {len(robustness['regime_performances'])}")
        print(f"   Stress Tests Completed: {len(robustness['stress_test_results'])}")

        if robustness['critical_failure_modes']:
            print(f"   ⚠️  Critical Failures: {len(robustness['critical_failure_modes'])}")
        else:
            print(f"   ✅ No Critical Failures Detected")

        # Display data quality
        quality = report['data_quality']
        print()
        print(f"📈 DATA QUALITY SUMMARY:")
        print(f"   Overall Quality Score: {quality['quality_score']:.2f}/1.00")
        print(f"   Missing Data: {quality['missing_data_percentage']:.1f}%")
        print(f"   Outliers: {quality['outlier_percentage']:.1f}%")
        print(f"   Signal-to-Noise Ratio: {quality['signal_to_noise_ratio']:.1f} dB")

        # Display attribution summary
        if 'attribution_analysis' in report and 'summary' in report['attribution_analysis']:
            attribution = report['attribution_analysis']['summary']
            if 'consensus_ranking' in attribution:
                ranking = attribution['consensus_ranking']
                print()
                print(f"🎯 PERFORMANCE ATTRIBUTION:")
                print(f"   Primary Driver: {ranking['primary_driver'].upper()}")
                print(f"   Attribution Confidence: {ranking['confidence']:.2f}")

                if 'average_contributions' in attribution:
                    avg_contrib = attribution['average_contributions']
                    print(f"   Neuromorphic Contribution: {avg_contrib['neuromorphic']:.1%}")
                    print(f"   Quantum Contribution: {avg_contrib['quantum']:.1%}")

        print()
        print("📄 REPORT FILES GENERATED:")
        print(f"   - Full JSON Report: ./validation_reports/{report['metadata']['report_id']}.json")
        print(f"   - Executive Summary: ./validation_reports/{report['metadata']['report_id']}_executive_summary.txt")
        print(f"   - Summary Plots: ./validation_reports/{report['metadata']['report_id']}_summary_plots.png")

        print()
        print("✨ STATISTICAL VALIDATION FRAMEWORK DEMONSTRATION COMPLETE!")
        print()
        print("🔬 SCIENTIFIC CONCLUSION:")

        # Scientific assessment
        confidence = exec_summary['confidence_level']
        significance = exec_summary['statistical_significance']
        robustness_score = robustness['overall_robustness_score']

        if confidence >= 80 and significance and robustness_score >= 0.7:
            print("   🟢 STRONG EVIDENCE for neuromorphic-quantum predictive capability")
            print("   📊 Results meet highest standards of statistical rigor")
            print("   🚀 RECOMMENDED for production deployment")
        elif confidence >= 60 and robustness_score >= 0.5:
            print("   🟡 MODERATE EVIDENCE for neuromorphic-quantum predictive capability")
            print("   📊 Results show promise but with limitations")
            print("   ⚠️  CONDITIONAL recommendation with monitoring")
        else:
            print("   🔴 INSUFFICIENT EVIDENCE for reliable neuromorphic-quantum predictions")
            print("   📊 Results do not meet minimum statistical standards")
            print("   ❌ NOT RECOMMENDED for production deployment")

        print()
        print("🎓 FRAMEWORK CAPABILITIES DEMONSTRATED:")
        print("   ✅ Comprehensive statistical testing suite")
        print("   ✅ Advanced model validation and cross-validation")
        print("   ✅ Sophisticated performance attribution analysis")
        print("   ✅ Rigorous robustness testing across market regimes")
        print("   ✅ Professional-grade statistical reporting")
        print("   ✅ Definitive conclusions with confidence quantification")

        return report

    except Exception as e:
        print(f"❌ Error during validation: {str(e)}")
        print("🔧 Please check your data and configuration")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()