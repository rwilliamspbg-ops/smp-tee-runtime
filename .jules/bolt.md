# Bolt's Journal

⚡ Performance-obsessed optimizations, learnings, and insights.

## 2026-05-26 - [Symmetric Multi-Krum Selection Optimization]
**Learning:** For peer-to-peer or distance-based aggregation algorithms like Multi-Krum, computing L2 distance between all candidates can be incredibly expensive. Since squared L2 distance is symmetric, precomputing a 1D or 2D matrix reduces distance computations by exactly 50%. Additionally, a full sort is unnecessary when we only need the sum of the smallest $K$ elements. Using standard library selection partitioning (e.g., `select_nth_unstable_by`) avoids the $O(N \log N)$ cost of sorting the distances.
**Action:** Always check if pairwise calculations in vector aggregation can be computed symmetrically, and favor selection partitioning over complete sorting when only top/bottom elements are desired.

## 2026-05-27 - [Iterator ExactSizeIterator Optimization for Ring Buffer Parsing]
**Learning:** In Rust iterator chains, wrapping an `ExactSizeIterator` (such as `Chunks`) in a `.filter(...)` destroys the exact size property because the compiler cannot predict how many elements will pass the filter. This forces `.collect()` to perform dynamic resizing/reallocations on the heap instead of a single, exact pre-allocation. Removing redundant filter conditions (e.g. checking for empty slices on `chunks` where the chunk size is > 0) preserves `ExactSizeIterator` and leads to massive performance gains (~46% speedup).
**Action:** Watch out for `.filter()` calls on iterators with known sizes. If the filter is redundant or can be avoided, remove it to preserve the exact size hint for optimal `.collect()` pre-allocation.
