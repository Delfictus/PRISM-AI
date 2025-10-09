# Next Session: DIMACS Graph Coloring Integration

**Session Goal:** Wire existing `phase_guided_coloring()` to DIMACS benchmarks
**Estimated Time:** 2-4 hours
**Target:** Test DSJC500-5 and DSJC1000-5, measure chromatic numbers

---

## Key Discovery

**The coloring algorithm already exists!**
- File: `src/prct-core/src/coloring.rs` (225 lines)
- Function: `phase_guided_coloring(graph, phase_field, kuramoto_state, target_colors)`
- Returns: `ColoringSolution` with `chromatic_number` and `conflicts`

**This is 90% done - just needs integration!**

---

## What Needs to be Done

### Step 1: Expose Phase State from Pipeline (30 min)

**Problem:** `PlatformOutput` doesn't expose `PhaseField` or `KuramotoState`

**Fix:** Modify `src/integration/unified_platform.rs`:
```rust
pub struct PlatformOutput {
    pub metrics: PerformanceMetrics,
    pub timing: TimingMetrics,
    // ADD THESE:
    pub phase_field: Option<PhaseField>,
    pub kuramoto_state: Option<KuramotoState>,
}
```

### Step 2: Wire to run_dimacs_official.rs (1 hour)

```rust
use prct_core::{parse_mtx_file, phase_guided_coloring};

// Load graph
let graph = parse_mtx_file("benchmarks/dimacs_official/DSJC500-5.mtx")?;

// Run pipeline (already working)
let mut platform = UnifiedPlatform::new(graph.num_vertices.min(20))?;
let output = platform.process(input)?;

// Extract coloring using existing algorithm
let solution = phase_guided_coloring(
    &graph,
    &output.phase_field.unwrap(),
    &output.kuramoto_state.unwrap(),
    100  // target colors (will optimize)
)?;

println!("Colors found: {}", solution.chromatic_number);
println!("Conflicts: {}", solution.conflicts);
println!("Best known: 47-48");
```

### Step 3: Test and Compare (30 min)

Run on:
- DSJC500-5 (best: 47-48)
- DSJC1000-5 (best: 82-83)

Compare chromatic numbers.

---

## Files to Modify

1. `src/integration/unified_platform.rs` - Add fields to PlatformOutput
2. `src/prct-core/src/lib.rs` - Export coloring module ✅ (done)
3. `examples/run_dimacs_official.rs` - Wire it together

**Total changes:** ~50 lines of code

---

## Expected First Results

**Realistic:**
- DSJC500-5: 60-80 colors (first attempt, no optimization)
- DSJC1000-5: 100-150 colors (first attempt)

**If we're lucky:**
- DSJC500-5: <60 colors (promising!)
- DSJC1000-5: <100 colors (very promising!)

**Even 100-150 colors is progress - shows algorithm works, then optimize**

---

## Quick Commands

```bash
# Edit files above
# Then test:
cargo run --release --features cuda --example run_dimacs_official

# Look for:
# - Chromatic number for each instance
# - Conflicts (should be 0)
# - Comparison to best known
```

---

**Status:** Infrastructure exists, just needs wiring
**Next:** Expose phase state, wire coloring function, test
**Time:** 2-4 hours to first results
**Goal:** Beat 47 colors (DSJC500-5) or 82 colors (DSJC1000-5)
