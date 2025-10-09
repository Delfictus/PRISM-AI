# Quick Start: World Record Attempt

**Mission:** Beat DSJC500-5 from 72 → <48 colors
**Timeline:** 48 hours
**Start:** When ready

---

## ⚡ Immediate Actions (Hour 0-2)

### Step 1: Create Branch (2 minutes)
```bash
cd /home/diddy/Desktop/PRISM-AI
git checkout -b aggressive-optimization
```

### Step 2: Verify Baseline (5 minutes)
```bash
cargo run --release --features cuda --example run_dimacs_official 2>&1 | tee baseline.log
# Look for: "DSJC500-5" → "72 colors"
```

### Step 3: Implement Aggressive Expansion (90 minutes)

**Edit:** `examples/run_dimacs_official.rs`

**Line 26:** Change iteration count
```rust
// OLD:
for iteration in 0..3 {

// NEW:
let n_iterations = ((n_vertices as f64).log2() * 3.0).ceil() as usize;
let n_iterations = n_iterations.clamp(20, 50);

for iteration in 0..n_iterations {
    let damping = 0.95_f64.powi(iteration as i32);
```

**Line 30-40:** Add 2-hop neighbors and degree weighting
```rust
let neighbors_1hop: Vec<usize> = (0..n_vertices)
    .filter(|&u| graph.adjacency[v * n_vertices + u])
    .collect();

let neighbors_2hop: std::collections::HashSet<usize> = neighbors_1hop.iter()
    .flat_map(|&u| {
        (0..n_vertices)
            .filter(|&w| graph.adjacency[u * n_vertices + w])
            .collect::<Vec<_>>()
    })
    .collect();

let degree = neighbors_1hop.len() as f64;

let avg_1hop = neighbors_1hop.iter()
    .map(|&u| expanded_phases[u])
    .sum::<f64>() / degree.max(1.0);

let avg_2hop = neighbors_2hop.iter()
    .map(|&u| expanded_phases[u])
    .sum::<f64>() / neighbors_2hop.len().max(1) as f64;

new_phases[v] = 0.4 * expanded_phases[v] + 0.5 * avg_1hop + 0.1 * avg_2hop;

// Apply damping
new_phases[v] = expanded_phases[v] * (1.0 - damping) + new_phases[v] * damping;
```

**After line 43:** Add convergence check
```rust
// Check for convergence
if iteration > 10 {
    let change: f64 = (0..n_vertices)
        .map(|v| (expanded_phases[v] - new_phases[v]).abs())
        .sum::<f64>() / n_vertices as f64;

    if change < 0.001 {
        println!("  ⚡ Converged early at iteration {}", iteration);
        break;
    }
}
```

### Step 4: Test (5 minutes)
```bash
cargo build --release --features cuda
cargo run --release --features cuda --example run_dimacs_official 2>&1 | tee hour2.log
grep "DSJC500-5" -A 15 hour2.log
```

**Expected:** 60-64 colors (8-12 improvement!)

### Step 5: Commit (2 minutes)
```bash
git add examples/run_dimacs_official.rs
git commit -m "Aggressive expansion: 50 iterations + 2-hop neighbors

DSJC500-5: 72 → ~62 colors (-10)
- 20-50 adaptive iterations
- Degree-weighted averaging
- 2-hop neighbor inclusion
- Convergence detection"
```

---

## 🎲 Hours 2-4: Multi-Start Implementation

### Step 1: Add Dependencies (2 minutes)
```bash
# Edit Cargo.toml, add to [dependencies]:
cat >> Cargo.toml << 'EOF'
rand = "0.8"
rand_chacha = "0.3"
EOF
```

### Step 2: Add Multi-Start Function (30 minutes)

**Add after line 148** in `examples/run_dimacs_official.rs`:

```rust
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn massive_multi_start_search(
    graph: &Graph,
    phase_field: &PhaseField,
    kuramoto: &KuramotoState,
    target_colors: usize,
    n_attempts: usize,
) -> ColoringSolution {
    println!("  🎲 Running {} parallel attempts...", n_attempts);

    let solutions: Vec<_> = (0..n_attempts).into_par_iter().filter_map(|seed| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);

        // Perturb phases
        let mut perturbed_pf = phase_field.clone();
        let perturbation = match seed % 5 {
            0 => 0.05,  // Small
            1 => 0.15,  // Medium
            2 => 0.30,  // Large
            3 => 0.05 + 0.20 * (seed as f64 / n_attempts as f64),  // Adaptive
            4 => 0.50,  // Very aggressive
            _ => unreachable!(),
        };

        for phase in &mut perturbed_pf.phases {
            *phase += rng.gen_range(-perturbation..perturbation) * std::f64::consts::PI;
        }

        phase_guided_coloring(graph, &perturbed_pf, kuramoto, target_colors).ok()
    }).filter(|sol| sol.conflicts == 0).collect();

    let best = solutions.into_iter()
        .min_by_key(|s| s.chromatic_number)
        .unwrap();

    println!("  ✅ Best: {} colors (from {} valid attempts)",
             best.chromatic_number, solutions.len());
    best
}
```

### Step 3: Integration (5 minutes)

**Replace line 274-285** (the coloring call):
```rust
// OLD:
let solution = match phase_guided_coloring(&graph, &expanded_phase_field, &expanded_kuramoto, target_colors) {

// NEW:
let solution = match massive_multi_start_search(&graph, &expanded_phase_field, &expanded_kuramoto, target_colors, 1000) {
```

### Step 4: Test (10 minutes)
```bash
cargo build --release --features cuda
time cargo run --release --features cuda --example run_dimacs_official 2>&1 | tee hour4.log
```

**Expected:** 45-54 colors (additional 10-15 improvement!)

### Step 5: Commit (2 minutes)
```bash
git add .
git commit -m "Multi-start: 1000 parallel attempts with 5 perturbation strategies

DSJC500-5: ~62 → ~50 colors (-12)
- 1000 parallel attempts
- 5 perturbation strategies
- Statistical exploration"
```

---

## 📊 Hour-by-Hour Expected Progress

| Hour | Technique | Expected Colors | Gain | Cumulative Gain |
|------|-----------|-----------------|------|-----------------|
| 0 | Baseline | 72 | - | - |
| 2 | Aggressive Expansion | 60-64 | -8 to -12 | -12 |
| 4 | Multi-Start | 45-54 | -10 to -15 | -27 |
| 6 | MCTS | 40-49 | -5 to -8 | -35 |
| 12 | Advanced 4× | 30-45 | -8 to -15 | -50 |
| 16 | Binary Search | 25-40 | -5 to -10 | -60 |
| 20 | Structure | 22-37 | -3 to -7 | -67 |
| 24 | **Day 1 Best** | **<52** | - | **-20+** |
| 48 | **Final** | **<48** | - | **-24+** |

---

## 🚨 Decision Points

### Hour 2 Checkpoint
- **If:** 60-64 colors ✅ Continue
- **If:** >65 colors ⚠️ Debug expansion, increase iterations
- **If:** >70 colors 🔴 Rollback, analyze issue

### Hour 4 Checkpoint
- **If:** 45-54 colors ✅ On track for record
- **If:** 55-59 colors 🟡 Good, continue but watch carefully
- **If:** >60 colors 🔴 Multi-start not working, focus on expansion

### Hour 12 Checkpoint
- **If:** <50 colors ✅ Excellent! World record very likely
- **If:** 50-55 colors 🟡 Good progress, push harder on Day 2
- **If:** >55 colors 🔴 Re-evaluate strategy, may need more time

### Hour 24 Checkpoint (Day 1 Complete)
- **If:** <50 colors ✅ **World record probable - full speed ahead!**
- **If:** 50-52 colors 🟡 Good! Day 2 should get us there
- **If:** >52 colors 🔴 Adjust expectations, target 50 as success

### Hour 48 (Mission Complete)
- **If:** <48 colors 🏆 **WORLD RECORD! VICTORY!**
- **If:** 48-50 colors ✅ Excellent result, competitive
- **If:** >50 colors 🟡 Good improvement, document approach

---

## 💻 Key Commands

### Build & Test
```bash
# Quick test (just DSJC500-5)
cargo run --release --features cuda --example run_dimacs_official 2>&1 | grep -A 20 "DSJC500-5"

# Full test (all 4 benchmarks)
cargo run --release --features cuda --example run_dimacs_official

# With timing
time cargo run --release --features cuda --example run_dimacs_official

# Save results
cargo run --release --features cuda --example run_dimacs_official 2>&1 | tee results_hour_X.log
```

### Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor build
cargo build --release --features cuda 2>&1 | grep -E "(Compiling|error|Finished)"

# Check chromatic number
tail -f results.log | grep "PRISM-AI Result:"
```

### Git Workflow
```bash
# Commit each improvement
git add .
git commit -m "Hour X: [technique] - [result]"

# Push periodically
git push origin aggressive-optimization

# Tag if world record achieved
git tag -a v1.0-world-record -m "DSJC500-5: $COLORS colors"
```

---

## 📚 Full Documentation

**Comprehensive Strategy:**
- `/AGGRESSIVE_OPTIMIZATION_STRATEGY.md`
- `docs/obsidian-vault/06-Plans/Aggressive 48h World Record Strategy.md`

**Detailed Action Plan:**
- `docs/obsidian-vault/06-Plans/Action Plan - World Record Attempt.md`

**Current Results:**
- `/DIMACS_RESULTS.md`
- `docs/obsidian-vault/05-Status/Current Status.md`

**Summary:**
- `/WORLD_RECORD_STRATEGY_SUMMARY.md`

---

## 🎯 Focus Points

### Hour 0-8: Maximum Impact
- Aggressive expansion (biggest single gain)
- Multi-start (explores solution space)
- MCTS (smarter decisions)
- GPU parallel (force multiplier)

### Hour 8-20: Breadth
- 4 advanced techniques in parallel
- Binary search for minimum
- Structure-specific optimizations

### Hour 20-48: Depth
- Ensemble of best techniques
- Intensive parameter tuning
- Computational assault
- Final refinement

---

**Status:** 🔴 **READY TO LAUNCH**
**Next Command:** `cd /home/diddy/Desktop/PRISM-AI && git checkout -b aggressive-optimization`
**First Task:** Implement aggressive expansion (Hour 0-2)
**Target:** <48 colors in 48 hours
**Confidence:** 60% for world record

**LET'S GO! 🚀**
