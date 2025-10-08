# Priority DIMACS Benchmarks - 2025 Challenge Problems

**Updated:** 2025-10-08
**Focus:** Most recent, challenging, and publication-worthy instances
**Goal:** Beat current best-known results for world-record validation

---

## Tier 1: Recently Active (2020-2024) - HIGHEST PRIORITY

### 1. r1000.1c ⭐⭐⭐ TOP PRIORITY

**Status:** Chromatic number **recently determined in 2024**
**Challenge:** 1000 vertices, random graph
**Significance:** Active research, recent progress
**Why Important:** Any improvement on recent work = immediate publication

**Download:**
```bash
wget -O r1000-1c.zip https://nrvis.com/download/data/dimacs/r1000-1c.zip
unzip r1000-1c.zip
```

**Publication Value:** VERY HIGH (recent activity = active community interest)

---

### 2. flat1000_76_0 ⭐⭐⭐ PROVE OPTIMALITY

**Known Optimal:** 76 colors (proven)
**Best Found:** 81-82 colors
**Gap to Close:** 5-6 colors
**Challenge:** Find optimal or beat 81

**Why Critical:**
- Known optimal exists (can prove correctness)
- Gap shows room for improvement
- Closing gap = provable contribution

**Download:**
```bash
wget -O flat1000-76-0.zip https://nrvis.com/download/data/dimacs/flat1000-76-0.zip
unzip flat1000-76-0.zip
```

**Publication Value:** VERY HIGH (can prove optimality or prove gap)

---

## Tier 2: Large-Scale Challenges (1000+ vertices)

### 3. DSJC1000.5 ⭐⭐ STANDARD LARGE BENCHMARK

**Already Downloaded:** ✅
**Challenge:** 1000 vertices, 500K edges
**Best Known:** 82-83 colors
**Status:** Standard benchmark, widely cited

**Why Important:**
- Most cited 1000-vertex benchmark
- Beating 82 = publishable
- Standard comparison point

**Publication Value:** HIGH (standard benchmark, widely recognized)

---

### 4. DSJC1000.9 ⭐⭐ DENSE GRAPH CHALLENGE

**Challenge:** 1000 vertices, 900K edges (90% density)
**Best Known:** 222-223 colors
**Difficulty:** Very dense graph

**Download:**
```bash
wget -O DSJC1000-9.zip https://nrvis.com/download/data/dimacs/DSJC1000-9.zip
unzip DSJC1000-9.zip
```

**Publication Value:** MEDIUM-HIGH (tests dense graph performance)

---

## Tier 3: Extreme Scale (2000-4000 vertices)

### 5. C2000.5 ⭐ SCALABILITY TEST

**Already Downloaded:** ✅
**Challenge:** 2000 vertices, 1M edges
**Best Known:** 145 colors
**Purpose:** Show scalability beyond 1000

---

### 6. C4000.5 ⭐ ULTIMATE CHALLENGE

**Already Downloaded:** ✅
**Challenge:** 4000 vertices, 4M edges
**Best Known:** 259 colors
**Warning:** May exceed memory/time limits

---

## Recommended Test Strategy

### Phase 1: Prove Capability (2-4 hours)

**Test in Order:**
```markdown
1. flat1000_76_0 (FIRST - can prove optimality)
   - Goal: Find 76 colors (optimal) or beat 81
   - Impact: Prove you can match/beat known results

2. r1000.1c (SECOND - recent activity)
   - Goal: Beat current best
   - Impact: Immediate publication relevance

3. DSJC500.5 (THIRD - validation)
   - Goal: Beat 47 colors
   - Impact: Validate on medium scale

4. DSJC1000.5 (FOURTH - standard benchmark)
   - Goal: Beat 82 colors
   - Impact: Standard large-scale benchmark
```

### Phase 2: Scalability (if Phase 1 successful)

```markdown
5. DSJC1000.9 (dense graph)
6. C2000.5 (2000 vertices)
7. C4000.5 (ultimate - if feasible)
```

---

## Download Script for Priority Instances

```bash
#!/bin/bash
# Download highest priority 2025 challenge instances

cd /home/diddy/Desktop/PRISM-AI/benchmarks/dimacs_official

echo "Downloading Tier 1 priority instances..."

# Priority 1: Recently solved
wget -O r1000-1c.zip https://nrvis.com/download/data/dimacs/r1000-1c.zip
unzip -o r1000-1c.zip

# Priority 2: Known optimal with gap
wget -O flat1000-76-0.zip https://nrvis.com/download/data/dimacs/flat1000-76-0.zip
unzip -o flat1000-76-0.zip

# Priority 3: Dense challenge
wget -O DSJC1000-9.zip https://nrvis.com/download/data/dimacs/DSJC1000-9.zip
unzip -o DSJC1000-9.zip

# Additional validation instances
wget -O DSJC250-5.zip https://nrvis.com/download/data/dimacs/DSJC250-5.zip
unzip -o DSJC250-5.zip

wget -O DSJC125-5.zip https://nrvis.com/download/data/dimacs/DSJC125-5.zip
unzip -o DSJC125-5.zip

echo "Download complete!"
ls -lh *.mtx
```

---

## Expected Impact by Instance

**flat1000_76_0:**
- Best case: Find optimal (76 colors) = **Major achievement**
- Good case: Beat 81 colors = **Publishable**
- Worst case: Match 81-82 = Shows competitiveness

**r1000.1c:**
- Any improvement = **Immediate publication** (active research area)
- Shows system can tackle recent challenges

**DSJC1000.5:**
- Beat 82 = **Standard benchmark victory**
- Match 82-83 = Competitive
- Most cited 1000-vertex graph

**Recommendation for Publication:**
- Focus on flat1000_76_0 (known optimal, can prove gap closure)
- Then r1000.1c (recent activity)
- Then DSJC1000.5 (standard comparison)

**These 3 instances are your best bet for publication-quality results.**

---

**Status:** You have DSJC500-5, DSJC1000-5, C2000-5, C4000-5
**Need:** r1000.1c, flat1000_76_0, DSJC1000-9 for complete validation
**Next:** Download these 3 priority instances and test
