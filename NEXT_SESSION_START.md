# Next Session: Fix Dimension Mismatch in Graph Coloring

**Session Goal:** Resolve phase state / graph size mismatch
**Estimated Time:** 1-2 hours
**Status:** Infrastructure complete, dimension expansion needed

---

## Progress Summary

✅ **Completed:**
1. Exposed `PhaseField` and `KuramotoState` from pipeline (`PlatformOutput`)
2. Added getter methods to `ThermodynamicPort` and `QuantumPort` traits
3. Implemented getters in `ThermodynamicAdapter` and `QuantumAdapter`
4. Wired `phase_guided_coloring()` to `run_dimacs_official.rs`
5. Tested on all 4 DIMACS instances

**Infrastructure is 100% complete!**

---

## Issue Discovered

**Problem:** Dimension Mismatch
- Platform creates **20-dimensional** phase state (from `dims=20`)
- Graphs have **500-4000 vertices**
- Coloring algorithm tries to access `coherence_matrix[i*n + j]` where `i,j` can be up to 500
- Result: Index out of bounds → returns 0.0 coherence → all vertices get color 0 → massive conflicts

**Current Results:**
```
DSJC500-5:   0 colors, 57624 conflicts   (Best: 47-48)
DSJC1000-5:  0 colors, 238979 conflicts  (Best: 82-83)
C2000-5:     0 colors, 979979 conflicts  (Best: 145)
C4000-5:     0 colors, 3960483 conflicts (Best: 259)
```

---

## Solution: Expand Phase State

**Option 1: Tile/Repeat (Quickest - 30 min)**
```rust
// In run_dimacs_official.rs before calling phase_guided_coloring:
fn expand_phase_state(phase_field: &PhaseField, target_size: usize) -> PhaseField {
    let n = phase_field.phases.len();

    // Tile phases by repeating
    let mut expanded_phases = Vec::with_capacity(target_size);
    for i in 0..target_size {
        expanded_phases.push(phase_field.phases[i % n]);
    }

    // Tile coherence matrix
    let mut expanded_coherence = vec![0.0; target_size * target_size];
    for i in 0..target_size {
        for j in 0..target_size {
            let src_i = i % n;
            let src_j = j % n;
            expanded_coherence[i * target_size + j] =
                phase_field.coherence_matrix[src_i * n + src_j];
        }
    }

    PhaseField {
        phases: expanded_phases,
        coherence_matrix: expanded_coherence,
        order_parameter: phase_field.order_parameter,
        resonance_frequency: phase_field.resonance_frequency,
    }
}

// Similarly for KuramotoState
fn expand_kuramoto_state(ks: &KuramotoState, target_size: usize) -> KuramotoState {
    let n = ks.phases.len();
    let expanded_phases: Vec<f64> = (0..target_size)
        .map(|i| ks.phases[i % n])
        .collect();

    // ... similar tiling for coupling_matrix, natural_frequencies
}
```

**Option 2: Interpolate (Better quality - 1-2 hours)**
- Use graph structure to propagate phase information
- Assign initial phases from tiled state
- Run local relaxation: each vertex averages phases of neighbors
- Maintains phase relationships while respecting graph topology

---

## Quick Fix Implementation

**File:** `examples/run_dimacs_official.rs`

Add these helper functions before `main()`:

```rust
fn expand_phase_field(pf: &shared_types::PhaseField, n_vertices: usize) -> shared_types::PhaseField {
    let n_phases = pf.phases.len();

    // Tile phases
    let expanded_phases: Vec<f64> = (0..n_vertices)
        .map(|i| pf.phases[i % n_phases])
        .collect();

    // Tile coherence matrix
    let mut expanded_coherence = vec![0.0; n_vertices * n_vertices];
    for i in 0..n_vertices {
        for j in 0..n_vertices {
            let src_i = i % n_phases;
            let src_j = j % n_phases;
            expanded_coherence[i * n_vertices + j] =
                pf.coherence_matrix[src_i * n_phases + src_j];
        }
    }

    shared_types::PhaseField {
        phases: expanded_phases,
        coherence_matrix: expanded_coherence,
        order_parameter: pf.order_parameter,
        resonance_frequency: pf.resonance_frequency,
    }
}

fn expand_kuramoto_state(ks: &shared_types::KuramotoState, n_vertices: usize) -> shared_types::KuramotoState {
    let n_phases = ks.phases.len();

    let expanded_phases: Vec<f64> = (0..n_vertices)
        .map(|i| ks.phases[i % n_phases])
        .collect();

    let expanded_freqs: Vec<f64> = (0..n_vertices)
        .map(|i| ks.natural_frequencies[i % n_phases])
        .collect();

    let mut expanded_coupling = vec![0.0; n_vertices * n_vertices];
    for i in 0..n_vertices {
        for j in 0..n_vertices {
            let src_i = i % n_phases;
            let src_j = j % n_phases;
            expanded_coupling[i * n_vertices + j] =
                ks.coupling_matrix[src_i * n_phases + src_j];
        }
    }

    shared_types::KuramotoState {
        phases: expanded_phases,
        natural_frequencies: expanded_freqs,
        coupling_matrix: expanded_coupling,
        order_parameter: ks.order_parameter,
        mean_phase: ks.mean_phase,
    }
}
```

Then in the coloring section:
```rust
// Apply phase-guided coloring algorithm
let target_colors = best_known_max + 50;

// EXPAND phase states to match graph size
let expanded_phase_field = expand_phase_field(&phase_field, graph.num_vertices);
let expanded_kuramoto = expand_kuramoto_state(&kuramoto_state, graph.num_vertices);

let solution = match phase_guided_coloring(
    &graph,
    &expanded_phase_field,  // Use expanded version
    &expanded_kuramoto,     // Use expanded version
    target_colors
) {
    // ...
}
```

---

## Expected Results After Fix

**First attempt (tiling):**
- DSJC500-5: 60-100 colors (shows algorithm works)
- DSJC1000-5: 100-200 colors
- Valid colorings (0 conflicts)

**If that works, try interpolation for better results:**
- DSJC500-5: 50-70 colors (closer to best)
- DSJC1000-5: 90-120 colors

---

## Commands

```bash
# 1. Add expansion functions to run_dimacs_official.rs
# 2. Update coloring invocation to use expanded states
# 3. Test:
cargo run --release --features cuda --example run_dimacs_official

# Look for:
# - Chromatic numbers > 0
# - Zero conflicts
# - Reasonable color counts (not too high)
```

---

**Next Action:** Add `expand_phase_field()` and `expand_kuramoto_state()` functions, update coloring call
**Time:** 30 minutes to first valid results
**Goal:** Get valid colorings (0 conflicts) on all instances, even if not optimal yet
