# PRISM-AI Mathematical Constitution

## Inviolable Principles

This system is bound by physical and mathematical laws that **CANNOT** be violated or masked.

### Article I: Thermodynamic Integrity

**The Second Law of Thermodynamics is ABSOLUTE:**

```
dS/dt ≥ 0  (Entropy production must be non-negative)
```

**Requirements:**
1. Entropy production MUST be computed honestly from thermodynamic evolution
2. If negative entropy is computed, this is a BUG that must be fixed
3. NEVER force, clamp, or mask negative entropy values
4. The system MUST halt if the 2nd law is violated

**Code Enforcement:**
```rust
// ✅ ALLOWED:
let entropy_prod = (S_final - S_initial) / dt;
assert!(entropy_prod >= -1e-15, "2nd law violated - this is a bug");

// ❌ FORBIDDEN:
let entropy_prod = computed_value.max(0.0);  // Masking violations
let entropy_prod = 1e-10;  // Hardcoding
```

---

### Article II: Information Theory Integrity

**Shannon Entropy and Mutual Information are Fundamental:**

```
H(X) ≥ 0  (Entropy is non-negative by definition)
I(X;Y) ≥ 0  (Mutual information is non-negative)
I(X;Y) ≤ min(H(X), H(Y))  (Bounded by marginal entropies)
```

**Requirements:**
1. Mutual information must be computed from actual probability distributions
2. Zero mutual information means no statistical dependency - this is valid
3. NEVER hardcode or force non-zero values
4. If computation fails, this is a BUG that must be fixed

---

### Article III: Variational Free Energy Integrity

**Free Energy Principle (Karl Friston):**

```
F = E_q[log q(s) - log p(o,s)]
F must be finite and well-defined
```

**Requirements:**
1. Free energy MUST be computed from actual belief distributions
2. Infinity means numerical instability - this is a BUG
3. NEVER replace infinity with arbitrary values
4. Fix the belief update that causes infinity

**Code Enforcement:**
```rust
// ✅ ALLOWED:
let F = model.compute_free_energy(observations);
if !F.is_finite() {
    return Err("Free energy computation failed - numerical instability");
}

// ❌ FORBIDDEN:
if !F.is_finite() { F = -1.0; }  // Masking bugs
```

---

### Article IV: Numerical Precision Integrity

**All computations must be numerically sound:**

**Requirements:**
1. Check for NaN, infinity after critical computations
2. If found, this is a BUG in the algorithm
3. Use appropriate numerical precision (double-double when needed)
4. NEVER mask numerical errors with defaults

---

### Article V: Performance Honesty

**Performance metrics must reflect actual execution:**

**Requirements:**
1. Timing must be from std::time::Instant (real wall-clock time)
2. Targets should be realistic but not artificially inflated
3. If target not met, report honestly - don't hide it
4. Allow graceful completion but report failures clearly

**Acceptable:**
```rust
// Realistic targets based on hardware capabilities
const LATENCY_TARGET_MS: f64 = 500.0;  // Documented reasoning

if latency > LATENCY_TARGET_MS {
    eprintln!("⚠ Latency target not met: {:.2}ms > {:.2}ms", latency, target);
    // Continue execution, but report honestly
}
```

---

### Article VI: Error Handling Integrity

**Errors must be meaningful and actionable:**

**Requirements:**
1. Physical law violations MUST error (2nd law, causality, etc.)
2. Numerical failures MUST error (NaN, infinity in critical paths)
3. Performance targets MAY warn (not always critical)
4. System MUST NOT mask bugs with arbitrary defaults

---

## Enforcement

### What is FORBIDDEN:

❌ `.max(0.0)` to force positive entropy
❌ `if infinity { value = default }` to hide numerical errors
❌ Hardcoded success values
❌ Masking computation failures
❌ Arbitrary defaults that hide bugs

### What is REQUIRED:

✅ Honest computation from actual physics/math
✅ Error on physical law violations
✅ Fix root causes, not symptoms
✅ Transparent reporting of all metrics
✅ Documented, justified defaults (only where truly appropriate)

---

## Validation Checklist

Before any result is reported as "successful":

- [ ] Entropy production computed from actual thermodynamic evolution
- [ ] No forced positive values
- [ ] Free energy computed from actual belief distributions
- [ ] No infinity masking
- [ ] All NaN/infinity cases investigated and fixed
- [ ] Performance metrics are actual measured values
- [ ] No hardcoded success values
- [ ] Physical laws verified, not assumed

---

## Publication Standards

For this system to be publication-worthy:

1. **Reproducibility:** All results from actual computation
2. **Transparency:** No hidden defaults or masks
3. **Integrity:** Physical laws honored absolutely
4. **Honesty:** Failures reported, not hidden
5. **Scientific Rigor:** Results must be defendable under peer review

---

**This constitution is INVIOLABLE.**

Any violation makes results unpublishable and scientifically invalid.

All code must honor these principles or be rejected.
