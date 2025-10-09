# TSP Datasets - Most Current Challenges

**Updated:** 2025-10-08
**Source:** TSPLIB + Recent TSP Challenges
**Goal:** Test on standard and cutting-edge TSP instances

---

## TSPLIB - Standard Benchmark Suite

**Official Source:** http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

**Status:** Gold standard for TSP benchmarking, but from 1995
**Coverage:** 110+ instances with known optimal solutions
**Format:** Various (.tsp files)

### Key TSPLIB Instances by Size

**Small (Good for Testing):**
- att48 (48 cities, optimal: 10,628)
- eil51 (51 cities, optimal: 426)
- berlin52 (52 cities, optimal: 7,542)
- st70 (70 cities, optimal: 675)
- pr76 (76 cities, optimal: 108,159)

**Medium (Standard Benchmarks):**
- kroA100 (100 cities, optimal: 21,282)
- kroB100 (100 cities, optimal: 22,141)
- lin105 (105 cities, optimal: 14,379)
- pr144 (144 cities, optimal: 58,537)
- ch150 (150 cities, optimal: 6,528)

**Large (Challenge):**
- pr299 (299 cities, optimal: 48,191)
- lin318 (318 cities, optimal: 42,029)
- pcb442 (442 cities, optimal: 50,778)
- rat575 (575 cities, optimal: 6,773)
- pr1002 (1002 cities, optimal: 259,045)

**Huge (Ultimate):**
- pr2392 (2392 cities, optimal: 378,032)
- pcb3038 (3038 cities, optimal: 137,694)
- fl3795 (3795 cities, optimal: 28,772)

---

## Recent TSP Challenges (2020-2025)

### 1. TSP Challenge - Ongoing

**Source:** http://www.math.uwaterloo.ca/tsp/

**Recent Solved Instances (2020-2024):**
- **pla85900** - 85,900 cities (solved 2024)
- **usa115475** - 115,475 cities (solved 2023)

**Unsolved Mega-Challenges:**
- **World TSP** - 1,904,711 cities (largest)
- **Various national tours** - 10,000-100,000 cities

---

## Download Commands

### TSPLIB Standard Suite

```bash
cd /home/diddy/Desktop/PRISM-AI/benchmarks

# Create tsplib directory
mkdir -p tsplib

cd tsplib

# Small instances (testing)
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/att48.tsp.gz
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/berlin52.tsp.gz
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/pr76.tsp.gz

# Medium instances (standard)
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/kroA100.tsp.gz
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/lin105.tsp.gz
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ch150.tsp.gz

# Large instances (challenge)
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/pr299.tsp.gz
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/lin318.tsp.gz
wget http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/pcb442.tsp.gz

# Extract all
gunzip *.gz
```

---

## Most Relevant for Your System

**Given your 4.07ms baseline and GPU acceleration:**

**Priority 1: Medium Scale (100-300 cities)**
- kroA100, kroB100, lin105 (optimal known, widely cited)
- Can solve in <10ms with your speed
- Direct comparison to LKH solver possible

**Priority 2: Large Scale (300-1000 cities)**
- pr299, lin318, pcb442
- Tests scalability
- Still feasible with 4ms overhead

**Priority 3: Very Large (1000+ cities)**
- pr1002, pr2392
- Shows extreme scalability
- May exceed time/memory limits

---

## Comparison to Graph Coloring

**DIMACS Graph Coloring:**
- DSJC1000-5: 1000 vertices, best: 82-83 colors
- 30+ years unsolved
- **Beating it = major achievement**

**TSPLIB:**
- Most instances have **known optimal solutions**
- Goal is to **match optimal** and **beat on time**
- Or find optimal on unsolved instances

**For Publication:**
- Graph coloring: Beat best-known = world record
- TSP: Match optimal faster = competitive, beat LKH = significant

---

## Recommendation

**For Journal Publication, prioritize:**

1. **DSJC1000-5 graph coloring** (beat 82 = major)
2. **kroA100, kroB100 TSP** (match optimal in <10ms = competitive)
3. **pr299, lin318 TSP** (larger scale validation)

**DSJC1000-5 has higher impact potential** because:
- Optimal unknown (room for discovery)
- 30+ years unsolved
- Beating it = definitive achievement

**TSP is still valuable** but most goals are known, so you're optimizing time not quality.

---

## Current Status

**TSP Datasets:** ❌ Not downloaded yet
**TSP Parser:** ❌ Not implemented (need .tsp format parser)
**TSP Solver:** ❌ Not integrated

**Effort to add TSP:**
- Download: 30 minutes
- Parser: 2-4 hours
- Integration: 4-8 hours
- **Total: ~8-12 hours**

**Recommendation:** Focus on DSJC1000-5 graph coloring first (higher impact), then add TSP if time allows.

---

**Answer:** Beating 83 colors = **YES, definitely journal-worthy** (top-tier publication). Matching 82 = **conference-worthy**. Either way, testing DSJC1000-5 is the highest-value next step.