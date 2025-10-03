# CRITICAL BUG: Transfer Entropy Shows Non-Zero Values for Independent Series

## Problem Statement
Independent random time series are showing TE values of 0.5-1.5 bits when they should be exactly 0.0. This violates the fundamental mathematical property that TE(X→Y) = 0 when X and Y are independent.

## Root Cause Analysis

### Issue 1: Finite Sample Bias with Discretization
When continuous random variables are discretized into bins, finite sample effects create spurious correlations:

```
Expected for independent uniform [0,1] with 10 bins:
- Each bin should have probability 0.1
- Joint bins should have probability 0.01
- But with 1000 samples, we only get ~10 samples per joint bin
- Sampling variability creates spurious information
```

### Issue 2: Bias Correction Inadequacy
Current bias correction uses Miller-Madow:
```
bias = (n_states - 1) / (2 * n_samples * ln(2))
```

But for TE, we need to consider:
- n_states = bins^(source_dim + target_dim + 1)
- With bins=10, dim=1: n_states = 10^3 = 1000
- With 1000 samples: only 1 sample per state on average!
- Severe undersampling leads to massive bias

### Issue 3: Mathematical Formula
The formula is correct but sensitive to estimation errors:
```
TE = Σ p(x,y,z) * log[p(x,y,z)*p(y) / (p(x,y)*p(y,z))]
```

For independent series:
- True: p(x,y,z) = p(x)*p(y)*p(z)
- Observed: p̂(x,y,z) ≠ p̂(x)*p̂(y)*p̂(z) due to finite samples

## Solutions

### Solution 1: Adaptive Binning
Reduce bins when sample size is small:
```rust
let optimal_bins = (n_samples as f64).powf(1.0/3.0).floor() as usize;
```

### Solution 2: Shuffled Baseline Subtraction
Calculate TE on shuffled data and subtract:
```rust
let te_shuffled = calculate_te_shuffled(x, y);
let te_corrected = (te_raw - te_shuffled).max(0.0);
```

### Solution 3: Use Continuous Estimators
Switch to KL estimator for continuous data to avoid discretization entirely.

### Solution 4: Statistical Thresholding
Only report TE if significantly different from shuffled baseline:
```rust
if p_value < 0.05 && te > threshold {
    return te;
} else {
    return 0.0;  // Not significant
}
```

## Immediate Fix Required

The current implementation is mathematically incorrect for practical use. We must either:

1. **Fix the bias correction** to properly handle finite sample effects
2. **Use adaptive binning** based on sample size
3. **Switch to continuous estimators** by default
4. **Apply significance testing** to filter spurious TE

## Test Cases Failing

- Independent uniform random: Shows TE = 0.5-1.0 (should be 0)
- Independent normal random: Shows TE = 0.3-0.7 (should be 0)
- Small sample sizes: TE increases as samples decrease (opposite of expected)

## Impact

This bug makes the implementation unusable for:
- Causal network inference (false positives everywhere)
- Scientific analysis (violates theoretical guarantees)
- DARPA demonstration (incorrect results)

## Priority: CRITICAL - Must fix before any production use