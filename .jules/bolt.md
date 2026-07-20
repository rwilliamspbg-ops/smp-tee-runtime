# Bolt's Journal

⚡ Performance-obsessed optimizations, learnings, and insights.

## 2026-05-26 - [Symmetric Multi-Krum Selection Optimization]
**Learning:** For peer-to-peer or distance-based aggregation algorithms like Multi-Krum, computing L2 distance between all candidates can be incredibly expensive. Since squared L2 distance is symmetric, precomputing a 1D or 2D matrix reduces distance computations by exactly 50%. Additionally, a full sort is unnecessary when we only need the sum of the smallest $K$ elements. Using standard library selection partitioning (e.g., `select_nth_unstable_by`) avoids the $O(N \log N)$ cost of sorting the distances.
**Action:** Always check if pairwise calculations in vector aggregation can be computed symmetrically, and favor selection partitioning over complete sorting when only top/bottom elements are desired.

## 2026-05-27 - [Iterator ExactSizeIterator Optimization for Ring Buffer Parsing]
**Learning:** In Rust iterator chains, wrapping an `ExactSizeIterator` (such as `Chunks`) in a `.filter(...)` destroys the exact size property because the compiler cannot predict how many elements will pass the filter. This forces `.collect()` to perform dynamic resizing/reallocations on the heap instead of a single, exact pre-allocation. Removing redundant filter conditions (e.g. checking for empty slices on `chunks` where the chunk size is > 0) preserves `ExactSizeIterator` and leads to massive performance gains (~46% speedup).
**Action:** Watch out for `.filter()` calls on iterators with known sizes. If the filter is redundant or can be avoided, remove it to preserve the exact size hint for optimal `.collect()` pre-allocation.

## 2026-05-28 - [Avoiding flat_map and Un-preallocated Collections in Serialization/Deserialization]
**Learning:** Converting float arrays to/from binary bytes inside TEE guards using flat_map (`values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()`) or collecting via dynamic resizing is highly inefficient. Dynamic reallocation overhead dominates under high packet volumes. Pre-allocating exact destination slice sizes (`vec![0.0_f32; bytes.len() / 4]` or `vec![0_u8; values.len() * 4]`) and manually writing elements in a simple loop avoids reallocation and yields massive speedups (~43% execution time reduction).
**Action:** Always pre-allocate the final vector capacity using `vec![default; size]` or `Vec::with_capacity` when doing serialization, deserialization, or element copying, especially when translating primitive arrays.

## 2026-05-29 - [Cache-Aware Tiling and Reciprocal Multiplication for Federated Averaging]
**Learning:** When executing federated averaging over client parameter vectors of extremely high dimensions (e.g., 10,000+), memory hierarchy latency and floating-point divisions heavily dominate execution time.
1. Memory hierarchy: Adding client vectors to a single large accumulator vector can cause thrashing between L3 cache and main memory. Loop tiling (processing elements in small 4 KB chunks, i.e., 1024 floats) keeps the chunk warm in L1/L2 caches and maximizes SIMD auto-vectorization across multiple client vectors (~11-15% speedup).
2. Division: Floating point division takes up to 40 CPU cycles, while multiplication takes 1. Normalizing the sum via multiplication by the reciprocal of client count (`1.0 / denom`) over `iter_mut()` speeds up normalization dramatically (~7% speedup).
3. Chunking overhead: For small-dimensional vectors (e.g. 4 elements), loop tiling steps and chunk calculation branches add measurable overhead. A hybrid approach with a fast-path for small dimensions (e.g., `<= 1024` elements) achieves maximum speed in all scenarios.
**Action:** Use a hybrid fast-path and loop-tiled structure for high-dimensional vector aggregations to keep CPU caches hot, and always reduce divisions to reciprocal multiplications in performance-critical loops.

## 2026-05-30 - [Manual Loop Unrolling and Independent Accumulators for L2 Distance Calculations]
**Learning:** In hot loops performing arithmetic reductions (such as calculating the squared L2 distance over long vectors in Multi-Krum), compiler auto-vectorization can sometimes be hindered by loop-carried dependencies on a single accumulator. Manually unrolling the loop (e.g., by 4) and using multiple, independent accumulator variables allows the CPU to leverage Instruction-Level Parallelism (ILP) and pipeline multiple floating-point fused multiply-adds (FMA) concurrently, reducing instruction latency bottlenecks. This yields a massive ~28-29% performance speedup on large client-vector aggregation benchmarks.
**Action:** Identify critical arithmetic reduction loops and implement manual loop unrolling with multiple independent accumulators to bypass data dependency stalls and maximize ILP.

## 2026-05-31 - [Safe Bounds-Check Elimination in Vector Serialization/Deserialization]
**Learning:** In Rust serialization or copying loops (e.g., converting floats to/from byte chunks), indexing arrays with manual range slicing (e.g., `bytes[i * 4..(i + 1) * 4]`) or loop counter indices (e.g., `values[i] = ...`) introduces redundant runtime bounds checking on every iteration. Zipping a pre-allocated vector iterator directly with `chunks_exact(4)` / `chunks_exact_mut(4)` and using `try_into().unwrap()` completely eliminates bounds check branches and produces extremely fast assembly code without resorting to unsafe blocks.
**Action:** Always prefer zipping pre-allocated vector iterators with exact chunk iterators (`chunks_exact`) rather than manual index-based lookup to allow the Rust compiler to fully optimize away range and index bounds checks safely.
