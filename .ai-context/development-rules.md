# Development Rules for AI-Assisted Sessions

## Mandatory Rules for Every Code Change

### 1. Every Function Must Include:

```rust
/// Mathematical description of what this computes
///
/// # Mathematical Foundation
/// ```math
/// f(x) = ...  // LaTeX notation
/// ```
///
/// # Complexity
/// - Time: O(...)
/// - Space: O(...)
///
/// # Error Conditions
/// - Returns Err if ...
///
/// # Examples
/// ```
/// let result = function(input)?;
/// assert_eq!(result, expected);
/// ```
pub fn function(input: T) -> Result<R, E> {
    // Implementation
}
```

### 2. Every GPU Kernel Must Include:

```cuda
/**
 * Kernel: kernel_name
 *
 * Mathematical Operation: ...
 *
 * Thread Organization:
 * - Grid: (blocks_x, blocks_y, blocks_z)
 * - Block: (threads_x, threads_y, threads_z)
 * - Shared memory: X bytes
 *
 * Memory Access Pattern:
 * - Coalesced: Yes/No
 * - Bank conflicts: None/Details
 *
 * Synchronization Points:
 * - __syncthreads() at lines: ...
 *
 * Occupancy:
 * - Theoretical: X%
 * - Achieved: Y%
 */
__global__ void kernel_name(...) {
    // Implementation
}
```

### 3. Every Module Must Include:

```rust
//! Module: module_name
//!
//! # Purpose
//! What this module does in one sentence.
//!
//! # Mathematical Basis
//! Theoretical foundation and key equations.
//!
//! # Integration Points
//! How this connects to other modules.
//!
//! # Validation
//! How correctness is verified.
```

## Forbidden Patterns

### ❌ Never Write:
```rust
// Bad: Arbitrary magic numbers
let threshold = 0.42;  // WHY 0.42?

// Bad: Unclear variable names
let x = compute();

// Bad: Missing error handling
let result = risky_operation().unwrap();

// Bad: No documentation
fn mystery_function(a: f64, b: f64) -> f64 {
    a * b + 3.14
}
```

### ✅ Always Write:
```rust
// Good: Named constants with justification
/// Synchronization threshold from Kuramoto 1975
const SYNC_THRESHOLD: f64 = 0.42;

// Good: Descriptive names
let phase_coherence = compute_kuramoto_order_parameter();

// Good: Proper error handling
let result = risky_operation()
    .context("Failed to perform X because Y")?;

// Good: Full documentation
/// Compute the product with pi offset
///
/// # Mathematical Foundation
/// ```math
/// f(a,b) = ab + π
/// ```
fn compute_product_with_pi_offset(a: f64, b: f64) -> f64 {
    a * b + std::f64::consts::PI
}
```

## Code Quality Standards

### Testing Requirements:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test happy path
    }

    #[test]
    fn test_edge_cases() {
        // Test boundary conditions
    }

    #[test]
    #[should_panic(expected = "specific error")]
    fn test_error_conditions() {
        // Test failure modes
    }

    #[test]
    fn test_mathematical_properties() {
        // Verify mathematical invariants
    }
}
```

### Performance Requirements:
```rust
#[bench]
fn bench_function(b: &mut Bencher) {
    b.iter(|| {
        // Code to benchmark
    });

    // Assert performance contract
    assert!(b.elapsed_time() < REQUIRED_TIME);
}
```

## Validation Before Commit

```bash
# Must pass ALL of these:
cargo fmt --check           # Code formatting
cargo clippy -- -D warnings # Linting
cargo test --all            # All tests
cargo bench                 # Performance tests
cargo doc --no-deps         # Documentation builds
./scripts/check_constitution_compliance.sh  # Constitution check
```

## Git Commit Standards

### Commit Message Template:
```
<type>(<phase>.<task>): <short description>

Constitution: Phase X Task Y.Z Section N
Validation: PASSED ✅
Coverage: 97%
Performance: All contracts met

Detailed explanation of changes:
- What was changed
- Why it was changed
- How it was validated

Refs: #issue-number
```

### Commit Types:
- `feat`: New feature (constitution task)
- `fix`: Bug fix
- `refactor`: Code restructure (no behavior change)
- `perf`: Performance improvement
- `test`: Add/modify tests
- `docs`: Documentation only
- `chore`: Maintenance tasks

## AI Assistant Guidelines

### When Starting a Task:
1. Read constitution section first
2. Identify mathematical requirements
3. Plan validation strategy
4. Write tests BEFORE implementation
5. Implement with full documentation
6. Validate against constitution
7. Run all checks before presenting

### When Reviewing Code:
1. Check constitution compliance
2. Verify mathematical correctness
3. Ensure proper error handling
4. Validate test coverage
5. Check performance contracts
6. Review documentation quality

### When Blocked:
1. Document the blocker clearly
2. Reference constitution section
3. Propose solutions maintaining compliance
4. Don't proceed without approval
5. Update PROJECT_STATUS.md

## Emergency Procedures

### If Constitution Violation Detected:
```bash
# Immediate actions:
git stash  # Save current work
./scripts/verify_constitution_integrity.sh
cat DISASTER_RECOVERY.md
# Follow recovery procedures
```

### If Tests Fail:
```bash
# Do NOT proceed with:
git commit
git push

# Instead:
cargo test --verbose  # Get details
cargo test -- --nocapture  # See output
# Fix issues
# Re-run all validations
```

### If Performance Degrades:
```bash
cargo bench --baseline
# Compare to previous
# If regression > 5%, STOP
# Profile and fix before proceeding
```

## Documentation Standards

### Function Documentation:
```rust
/// Short one-line description
///
/// Longer description explaining the purpose and behavior.
///
/// # Mathematical Foundation
/// ```math
/// f(x) = ...
/// ```
///
/// # Arguments
/// * `param1` - Description
/// * `param2` - Description
///
/// # Returns
/// Description of return value
///
/// # Errors
/// * `ErrorType1` - When ...
/// * `ErrorType2` - When ...
///
/// # Examples
/// ```
/// # use crate::module::function;
/// let result = function(arg1, arg2)?;
/// assert_eq!(result, expected);
/// ```
///
/// # Panics
/// This function panics if ...
///
/// # Safety
/// (If unsafe) This function is unsafe because ...
///
/// # Performance
/// O(n) time, O(1) space
///
/// # Constitution Reference
/// Phase X, Task Y.Z
pub fn function() {}
```

## Quality Checklist

Before marking any task complete:

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Coverage > 95%
- [ ] Performance contracts met
- [ ] Documentation complete
- [ ] Constitution section referenced
- [ ] Validation gates passed
- [ ] Peer review completed (if applicable)
- [ ] No forbidden terms used
- [ ] Error handling comprehensive
- [ ] Mathematical correctness verified

---

**Remember**: The constitution is supreme. When these rules conflict with the constitution, the constitution wins. When in doubt, STOP and ask.
