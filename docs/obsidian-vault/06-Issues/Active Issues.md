# Active Issues

**Last Updated:** 2025-10-04

---

## 🔴 Critical (Must Fix) - 0 issues

None! 🎉

---

## 🟡 High Priority - 3 issues

### Issue #1: Example Files Have Broken Imports
**Status:** Open
**Priority:** High
**Category:** Documentation/Examples
**Effort:** 1-2 hours

**Problem:**
All 10 example files reference old crate names from before the rebrand.

**Affected Files:**
1. `examples/platform_demo.rs`
2. `examples/transfer_entropy_demo.rs`
3. `examples/phase6_cma_demo.rs`
4. `examples/gpu_performance_demo.rs`
5. `examples/rtx5070_validation_demo.rs`
6. `examples/stress_test_demo.rs`
7. `examples/error_handling_demo.rs`
8. `examples/comprehensive_benchmark.rs`
9. `examples/large_scale_tsp_demo.rs`
10. `examples/gpu_thermodynamic_benchmark.rs`

**Current Errors:**
```rust
// Wrong:
use active_inference_platform::*;
use neuromorphic_quantum_platform::*;

// Should be:
use prism_ai::*;
```

**Impact:**
- Cannot run any demos
- No working examples for users
- Blocks onboarding

**Fix:**
Global search-replace in all example files:
- `active_inference_platform` → `prism_ai`
- `neuromorphic_quantum_platform` → `prism_ai`

---

### Issue #2: Incomplete GPU Features (4 TODOs)
**Status:** Open
**Priority:** High
**Category:** GPU/Implementation
**Effort:** 8-12 hours

**Problem:**
4 GPU code paths have TODO comments for complex number handling.

**Locations:**
1. `src/adapters/src/quantum_adapter.rs:93`
2. `src/adapters/src/quantum_adapter.rs:197`
3. `src/adapters/src/quantum_adapter.rs:324`
4. `src/adapters/src/quantum_adapter.rs:360`

**TODO Text:**
```rust
/// TODO: Implement with proper complex number handling (separate real/imag buffers)
```

**Impact:**
- GPU quantum operations may not work correctly
- Falls back to CPU
- Performance degradation

**Fix Approach:**
- Implement proper complex number GPU buffers
- Separate real/imaginary components
- Update CUDA kernels accordingly

---

### Issue #3: 109 Compiler Warnings
**Status:** Open
**Priority:** High
**Category:** Code Quality
**Effort:** 4-8 hours

**Problem:**
109 warnings from unused code, affecting code cleanliness.

**Breakdown:**
- 45 unused variables
- 35 unused struct fields
- 13 unused imports
- 10 unused methods
- 6 code quality issues

**Top Warnings:**
1. `hamiltonian` variable unused (3 occurrences)
2. `target`, `rng` variables unused (2 each)
3. `solution_dim` field unused (2 occurrences)
4. GPU-related fields unused in structs

**Impact:**
- Code clutter
- Harder to maintain
- Potential dead code

**Fix Approach:**
1. Run `cargo fix --lib --allow-dirty` (fixes 15 automatically)
2. Manually review and fix remaining 94
3. Remove truly dead code
4. Prefix intentionally unused with `_`

---

## 🟠 Medium Priority - 4 issues

### Issue #4: Missing Cargo.toml Metadata
**Status:** Open
**Priority:** Medium
**Category:** Publishing
**Effort:** 15 minutes

**Problem:**
Cannot publish to crates.io without metadata.

**Missing Fields:**
```toml
repository = "https://github.com/Delfictus/PRISM-AI"
homepage = "https://github.com/Delfictus/PRISM-AI"
documentation = "https://docs.rs/prism-ai"
```

**Impact:**
- Cannot publish to crates.io
- Warning when building
- Poor discoverability

**Fix:**
Add 3 lines to `Cargo.toml` `[package]` section.

---

### Issue #5: Documentation Gaps
**Status:** Open
**Priority:** Medium
**Category:** Documentation
**Effort:** 4-6 hours

**Problems:**
1. Some modules lack doc comments
2. Math symbol formatting warnings (16)
3. No top-level usage guide
4. GPU requirements not clearly documented

**Affected Modules:**
- Several CMA sub-modules
- Some internal types

**Math Symbol Warnings:**
```
warning: unresolved link to `σ`
warning: unresolved link to `ψ`
```

**Fix Approach:**
1. Add module-level doc comments
2. Escape math symbols properly: `\[` and `\]`
3. Create usage guide in README
4. Document GPU requirements clearly

---

### Issue #6: Type Visibility Issues
**Status:** Open
**Priority:** Medium
**Category:** API Design
**Effort:** 30 minutes

**Problem:**
2 types are more private than their public methods.

**Instances:**
1. `ReservoirStatistics` in `reservoir.rs:224`
   - Method is `pub` but type is `pub(self)`

2. `GpuEmbeddings` in KSG estimator
   - Similar visibility mismatch

**Impact:**
- API inconsistency
- Compiler warnings
- Confusing for users

**Fix:**
Make types fully public or make methods private.

---

### Issue #7: Test Flakiness
**Status:** Open
**Priority:** Medium
**Category:** Testing
**Effort:** 2-4 hours

**Problem:**
1 test occasionally fails.

**Flaky Test:**
- `active_inference::gpu_inference::tests::test_gpu_jacobian_transpose`

**Behavior:**
- Usually passes
- Occasionally fails (timing/GPU state dependent)
- Non-deterministic

**Impact:**
- CI may fail randomly
- Confidence in test suite reduced

**Fix Approach:**
1. Investigate race conditions
2. Add synchronization if needed
3. Increase tolerances if numerical
4. Make deterministic

---

## 🟢 Low Priority - 2 issues

### Issue #8: Other GPU TODOs
**Status:** Open
**Priority:** Low
**Category:** Implementation
**Effort:** 2-4 hours

**Locations:**
1. `src/prct-core/src/drpp_algorithm.rs:205`
   - "TODO: Full integration requires cross-crate coordination"

2. `src/cma/gpu_integration.rs:107`
   - "TODO: Implement proper pooling with size-based caching"

**Impact:** Minor - features work without these

---

### Issue #9: Unused Methods (10 occurrences)
**Status:** Open
**Priority:** Low
**Category:** Code Quality
**Effort:** 2-3 hours

**Examples:**
- `generate_chromatic_coloring`
- `optimize_tsp_ordering`
- `estimate_kl_divergence`
- `calculate_coupling_strength`

**Fix:**
Remove if truly unused, or use them if needed.

---

## 📊 Issue Statistics

### By Priority
| Priority | Count |
|----------|-------|
| Critical | 0 |
| High | 3 |
| Medium | 4 |
| Low | 2 |
| **Total** | **9** |

### By Category
| Category | Count |
|----------|-------|
| Code Quality | 3 |
| GPU/Implementation | 2 |
| Documentation | 2 |
| Publishing | 1 |
| Testing | 1 |

### Estimated Total Effort
- High Priority: 13-22 hours
- Medium Priority: 7-11 hours
- Low Priority: 4-7 hours
- **Total: 24-40 hours**

---

## 🎯 Recommended Fix Order

1. **Issue #4:** Add Cargo metadata (15 min) ✅ Quick win
2. **Issue #1:** Fix example imports (1-2 hrs) 📈 Enables demos
3. **Issue #7:** Fix flaky test (2-4 hrs) 🧪 Improves reliability
4. **Issue #3:** Clean top 20 warnings (2 hrs) 🧹 Partial fix
5. **Issue #5:** Add basic docs (2 hrs) 📚 Improves usability
6. **Issue #6:** Fix visibility (30 min) ✅ Quick win
7. **Issue #2:** GPU complex numbers (8-12 hrs) ⚡ Full GPU support
8. **Issue #3:** Clean remaining warnings (2-6 hrs) 🧹 Complete
9. **Issue #8-9:** Low priority TODOs (4-7 hrs) 🎁 Nice to have

---

## 🔗 Related Documents

- [[Current Status]] - Overall status
- [[Recent Changes]] - Change history
- [[Development Workflow]] - How to fix issues
- [[Testing Guide]] - Testing procedures

---

*Issues tracked: 9 active, 0 closed*
