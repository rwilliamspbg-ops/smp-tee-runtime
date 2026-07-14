# Bolt's Journal

⚡ Performance-obsessed optimizations, learnings, and insights.

## 2026-05-26 - [Symmetric Multi-Krum Selection Optimization]
**Learning:** For peer-to-peer or distance-based aggregation algorithms like Multi-Krum, computing L2 distance between all candidates can be incredibly expensive. Since squared L2 distance is symmetric, precomputing a 1D or 2D matrix reduces distance computations by exactly 50%. Additionally, a full sort is unnecessary when we only need the sum of the smallest $K$ elements. Using standard library selection partitioning (e.g., `select_nth_unstable_by`) avoids the $O(N \log N)$ cost of sorting the distances.
**Action:** Always check if pairwise calculations in vector aggregation can be computed symmetrically, and favor selection partitioning over complete sorting when only top/bottom elements are desired.
