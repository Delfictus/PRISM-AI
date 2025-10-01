#!/usr/bin/env python3
"""
Calculate maximum TSP problem size for RTX 5070 GPU
"""

# RTX 5070 Laptop GPU specs
GPU_MEMORY_GB = 8.0
GPU_MEMORY_BYTES = GPU_MEMORY_GB * 1024**3

# Leave headroom for OS/driver overhead
USABLE_MEMORY_FRACTION = 0.80
USABLE_MEMORY_BYTES = GPU_MEMORY_BYTES * USABLE_MEMORY_FRACTION

print("=" * 70)
print("GPU TSP Memory Calculator - RTX 5070 Laptop (8GB)")
print("=" * 70)
print()

print(f"Total GPU Memory: {GPU_MEMORY_GB:.1f} GB")
print(f"Usable Memory (80%): {USABLE_MEMORY_BYTES / 1024**3:.2f} GB")
print()

print("Memory Requirements for TSP Solver:")
print("-" * 70)

def calculate_memory(n_cities):
    """Calculate GPU memory needed for n cities"""

    # Distance matrix: n x n floats (f32 = 4 bytes)
    distance_matrix = n_cities * n_cities * 4

    # Tour array: n integers (u32 = 4 bytes)
    tour = n_cities * 4

    # Coupling matrices (real + imaginary): 2 * n x n floats
    coupling_matrices = 2 * n_cities * n_cities * 4

    # 2-opt evaluation arrays
    total_swaps = n_cities * (n_cities - 3) // 2
    deltas = total_swaps * 4  # f32 array
    swap_pairs = total_swaps * 2 * 4  # 2 u32s per swap

    # Reduction buffers (for finding min/max)
    blocks = (total_swaps + 255) // 256
    partial_results = blocks * 8  # f32 + u32 per block

    total = (distance_matrix + tour + coupling_matrices +
             deltas + swap_pairs + partial_results)

    return {
        'distance_matrix': distance_matrix,
        'tour': tour,
        'coupling': coupling_matrices,
        'deltas': deltas,
        'swap_pairs': swap_pairs,
        'reduction': partial_results,
        'total': total
    }

# Test various problem sizes
test_sizes = [200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 29000, 30000]

print(f"{'Cities':<10} {'Total Memory':<15} {'% of GPU':<12} {'Status':<10}")
print("-" * 70)

max_feasible = 0
for n in test_sizes:
    mem = calculate_memory(n)
    total_mb = mem['total'] / 1024**2
    percent = (mem['total'] / USABLE_MEMORY_BYTES) * 100

    if mem['total'] <= USABLE_MEMORY_BYTES:
        status = "✅ Feasible"
        max_feasible = n
    else:
        status = "❌ Too large"

    print(f"{n:<10} {total_mb:>8.2f} MB    {percent:>6.1f}%      {status}")

print()
print("=" * 70)
print(f"MAXIMUM FEASIBLE SIZE: ~{max_feasible:,} cities")
print("=" * 70)
print()

# Detailed breakdown for max size
print(f"Memory Breakdown for {max_feasible:,} cities:")
print("-" * 70)
mem = calculate_memory(max_feasible)
for key, value in mem.items():
    if key != 'total':
        mb = value / 1024**2
        percent = (value / mem['total']) * 100
        print(f"  {key:<20} {mb:>8.2f} MB  ({percent:>5.1f}%)")
print(f"  {'TOTAL':<20} {mem['total'] / 1024**2:>8.2f} MB")
print()

print("Notes:")
print("  • Assumes f32 (4 bytes) for floats, u32 (4 bytes) for integers")
print("  • Does not include CUDA runtime overhead (~100-200 MB)")
print("  • Does not include kernel code/stack memory")
print("  • Conservative estimate - actual limit may be slightly lower")
print()

# Calculate for the famous 29k city problem
print("=" * 70)
print("FAMOUS TSP INSTANCES:")
print("=" * 70)
famous = [
    ("usa13509", 13509, "USA Road Network"),
    ("d15112", 15112, "Germany Road Network"),
    ("d18512", 18512, "Germany Road Network"),
    ("pla33810", 33810, "Programmed Logic Array"),
    ("pla85900", 85900, "Programmed Logic Array"),
]

for name, n, description in famous:
    mem = calculate_memory(n)
    total_gb = mem['total'] / 1024**3
    percent = (mem['total'] / USABLE_MEMORY_BYTES) * 100

    if mem['total'] <= USABLE_MEMORY_BYTES:
        status = "✅ FITS"
    else:
        status = "❌ TOO LARGE"

    print(f"\n{name} ({n:,} cities) - {description}")
    print(f"  Memory: {total_gb:.2f} GB ({percent:.1f}% of GPU)")
    print(f"  Status: {status}")
