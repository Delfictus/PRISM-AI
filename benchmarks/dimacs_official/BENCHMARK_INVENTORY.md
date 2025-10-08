# DIMACS Official Benchmark Inventory

**Downloaded:** 2025-10-08
**Source:** Network Repository (nrvis.com)
**Format:** Matrix Market (.mtx)
**Purpose:** Official world-record validation

---

## Downloaded Benchmarks

### ⭐ PRIORITY #1: DSJC1000.5

**File:** `DSJC1000-5.mtx` (1.9MB)
**Challenge:**
- Vertices: 1000
- Edges: ~500,000 (50% density)
- Type: Random graph (Johnson DIMACS)

**Best Known Results:**
- Colors: 82-83 (optimal unknown)
- Published: DIMACS Challenge results
- Gap: Unknown if optimal

**Why Important:**
- Large-scale challenge graph
- Standard benchmark for comparisons
- Unsolved optimal coloring
- Beating 82 colors = publishable result

**Status:** ✅ Downloaded

---

### ⭐ PRIORITY #2: DSJC500.5

**File:** `DSJC500-5.mtx` (463KB)
**Challenge:**
- Vertices: 500
- Edges: ~125,000 (50% density)
- Type: Random graph

**Best Known Results:**
- Colors: 47-48 (optimal unknown)
- Easier than DSJC1000.5
- Good for testing scalability

**Why Important:**
- Medium-scale validation
- Faster than 1000-vertex graphs
- Still challenging
- Beating 47 colors = good result

**Status:** ✅ Downloaded

---

### ⭐ PRIORITY #4: C2000.5

**File:** `C2000-5.mtx` (8.5MB)
**Challenge:**
- Vertices: 2000
- Edges: ~1,000,000 (50% density)
- Type: Random graph

**Best Known Results:**
- Colors: 145 (optimal unknown)
- Large-scale challenge
- Tests scalability beyond 1000

**Why Important:**
- Largest graph we can likely handle
- Shows scalability
- Beating 145 colors = significant

**Status:** ✅ Downloaded

---

### ⭐ PRIORITY #5: C4000.5 (THE BIG ONE)

**File:** `C4000-5.mtx` (37MB!)
**Challenge:**
- Vertices: 4000
- Edges: ~4,000,000 (50% density)
- Type: Random graph

**Best Known Results:**
- Colors: 259 (optimal unknown)
- Extremely large for our system
- Ultimate scalability test

**Why Important:**
- Tests absolute limits
- Shows if approach scales
- Solving this = legendary status

**Status:** ✅ Downloaded

**Warning:** May be too large for our current system (4K vertices, 4M edges)

---

## File Format

**Matrix Market (.mtx) Format:**
```
%%MatrixMarket matrix coordinate pattern symmetric
[vertices] [vertices] [edges]
[vertex1] [vertex2]
...
```

**Example (C2000-5.mtx):**
```
%%MatrixMarket matrix coordinate pattern symmetric
2000 2000 999836
3 2
4 3
5 1
...
```

**Note:** This is NOT .col format (DIMACS standard). We'll need to:
1. Parse MTX format, OR
2. Convert to .col format, OR
3. Find .col versions

---

## Missing Priority Benchmarks

**Still Need to Download:**

**Priority #3: DSJC1000.9**
- 1000 vertices, 900K edges (very dense)
- Best: 222-223 colors
- Status: Download timed out (need retry)

**Priority #6: flat1000_76_0**
- Known optimal: 76 colors
- Best found: 81-82 colors
- Gap to close: 5-6 colors
- Status: Not downloaded yet

**Validation Benchmarks:**
- DSJC250.5 - Medium scale
- DSJC125.5 - Small scale
- R1000.5 - Known optimal (234 colors)
- r1000.1c - Recently solved (2024)

**Action:** Continue downloads when ready

---

## Best Known Results Reference

| Instance | Vertices | Edges | Best Known | Optimal? | Priority |
|----------|----------|-------|------------|----------|----------|
| DSJC1000.5 | 1000 | 500K | 82-83 | Unknown | ⭐⭐⭐ #1 |
| DSJC500.5 | 500 | 125K | 47-48 | Unknown | ⭐⭐ #2 |
| DSJC1000.9 | 1000 | 900K | 222-223 | Unknown | ⭐ #3 |
| C2000.5 | 2000 | 1M | 145 | Unknown | ⭐ #4 |
| C4000.5 | 4000 | 4M | 259 | Unknown | ⭐ #5 |
| flat1000_76_0 | 1000 | ? | 81-82 | 76 (known) | 🎯 #6 |
| R1000.5 | 1000 | ? | 234 | 234 (optimal) | ✅ Val |
| DSJC250.5 | 250 | ? | ? | Unknown | ✅ Val |
| DSJC125.5 | 125 | ? | ? | Unknown | ✅ Val |

---

## Next Steps

### Task 1.1.2: Parse MTX Format (2 hours)

**Options:**
1. Write MTX parser for our system
2. Convert MTX to .col format
3. Use existing parser library

**Preferred:** Write converter MTX → DIMACS .col format

### Task 1.1.3: Run Small Instance (30 min)

**First Test:** DSJC500.5 (easiest of the priorities)
- Verify system can load and process
- Check solution validity
- Measure performance

### Task 1.1.4: Document Results

**For Each Instance:**
- Time to solve
- Number of colors found
- Comparison to best known
- Solution verification

---

## Storage

**Location:** `/home/diddy/Desktop/PRISM-AI/benchmarks/dimacs_official/`

**Total Downloaded:** 4 instances, ~48MB
**Format:** Matrix Market (.mtx)
**Quality:** Official DIMACS instances ✅

**Next:** Parse format and run first benchmark

---

**Status:** ✅ Task 1.1.1 COMPLETE (partial - 4/11 priority benchmarks)
**Time Spent:** ~10 minutes (downloads)
**Next Task:** 1.1.2 - Implement MTX parser or converter
