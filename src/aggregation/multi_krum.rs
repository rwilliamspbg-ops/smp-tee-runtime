#[inline(always)]
fn squared_l2_distance(left: &[f32], right: &[f32]) -> f32 {
    // Length equality is pre-validated in `multi_krum` prior to calling `squared_l2_distance`.
    // Returning `f32` directly avoids redundant branch checks and Option wrapping/unwrapping on every pair.

    // Optimized: Manual loop unrolling by 8 with independent accumulators
    // zipped with `chunks_exact(8)` and cast to fixed-size array references.
    // This completely eliminates bounds checks, reduces data dependency latency
    // (by avoiding immediate dependencies on `sum`), and maximizes instruction-level parallelism (ILP).
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;
    let mut sum4 = 0.0_f32;
    let mut sum5 = 0.0_f32;
    let mut sum6 = 0.0_f32;
    let mut sum7 = 0.0_f32;

    let mut left_chunks = left.chunks_exact(8);
    let mut right_chunks = right.chunks_exact(8);

    for (cl, cr) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        let cl: &[f32; 8] = cl.try_into().unwrap();
        let cr: &[f32; 8] = cr.try_into().unwrap();

        let d0 = cl[0] - cr[0];
        let d1 = cl[1] - cr[1];
        let d2 = cl[2] - cr[2];
        let d3 = cl[3] - cr[3];
        let d4 = cl[4] - cr[4];
        let d5 = cl[5] - cr[5];
        let d6 = cl[6] - cr[6];
        let d7 = cl[7] - cr[7];

        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
        sum4 += d4 * d4;
        sum5 += d5 * d5;
        sum6 += d6 * d6;
        sum7 += d7 * d7;
    }

    let mut sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

    // Handle any remaining elements when length is not a multiple of 8
    let rem_l = left_chunks.remainder();
    let rem_r = right_chunks.remainder();

    let (rem2_l, rem2_r) = if rem_l.len() >= 4 {
        let cl: &[f32; 4] = rem_l[..4].try_into().unwrap();
        let cr: &[f32; 4] = rem_r[..4].try_into().unwrap();

        let d0 = cl[0] - cr[0];
        let d1 = cl[1] - cr[1];
        let d2 = cl[2] - cr[2];
        let d3 = cl[3] - cr[3];

        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        (&rem_l[4..], &rem_r[4..])
    } else {
        (rem_l, rem_r)
    };

    // Optimized: Match on remainder length using fixed-size array conversions
    // to eliminate iterator overhead and bounds checking branches.
    match rem2_l.len() {
        3 => {
            let cl: &[f32; 3] = rem2_l.try_into().unwrap();
            let cr: &[f32; 3] = rem2_r.try_into().unwrap();
            let d0 = cl[0] - cr[0];
            let d1 = cl[1] - cr[1];
            let d2 = cl[2] - cr[2];
            sum += d0 * d0 + d1 * d1 + d2 * d2;
        }
        2 => {
            let cl: &[f32; 2] = rem2_l.try_into().unwrap();
            let cr: &[f32; 2] = rem2_r.try_into().unwrap();
            let d0 = cl[0] - cr[0];
            let d1 = cl[1] - cr[1];
            sum += d0 * d0 + d1 * d1;
        }
        1 => {
            let delta = rem2_l[0] - rem2_r[0];
            sum += delta * delta;
        }
        _ => {}
    }

    sum
}

// Optimized: Accept generic vector references `V` implementing `AsRef<[f32]>`.
// This allows caller contexts (like `InMemoryTee::execute_computation`) to execute Multi-Krum
// directly on zero-copy borrowed memory slices of floats, completely bypassing heavy float deserializations.
pub fn multi_krum<V: AsRef<[f32]>>(vectors: &[V], byzantine_tolerance: usize) -> Option<Vec<f32>> {
    let n = vectors.len();
    if n < 2 * byzantine_tolerance + 3 {
        return None;
    }

    // Pre-extract references to the underlying f32 slices to completely eliminate any
    // repeated `.as_ref()` method calls inside the distance precomputation and final vector cloning.
    // To avoid expensive heap allocations for typical small vector counts, we use a hybrid
    // stack-allocated buffer for up to 64 vectors and fall back to a heap-allocated Vec for larger counts.
    // Zipping with a slice window of `stack_buf` completely avoids index bounds checks on initialization.
    let mut stack_buf = [&[] as &[f32]; 64];
    let heap_buf: Vec<&[f32]>;
    let extracted: &[&[f32]] = if n <= 64 {
        for (dest, src) in stack_buf[..n].iter_mut().zip(vectors.iter()) {
            *dest = src.as_ref();
        }
        &stack_buf[..n]
    } else {
        heap_buf = vectors.iter().map(|v| v.as_ref()).collect();
        &heap_buf
    };

    let dimension = extracted[0].len();
    if extracted.iter().any(|v| v.len() != dimension) {
        return None;
    }

    let neighbors = n.checked_sub(byzantine_tolerance + 2)?;

    // Precompute symmetric pairwise distances to reduce distance calculation count by 50%
    let mut distance_matrix = vec![0.0_f32; n * n];

    // Optimized: Precalculate row offsets of the flat 1D matrix layout to avoid
    // redundant multiplication operations `j * n` inside the hot inner loop.
    // For small workloads (up to 64 clients), references are stored in a stack-allocated array
    // to bypass heap allocation, only falling back to a heap-allocated Vec for larger lists.
    let mut stack_offsets = [0; 64];
    let heap_offsets: Vec<usize>;
    let row_offsets: &[usize] = if n <= 64 {
        for (r, offset) in stack_offsets[..n].iter_mut().enumerate() {
            *offset = r * n;
        }
        &stack_offsets[..n]
    } else {
        heap_offsets = (0..n).map(|r| r * n).collect();
        &heap_offsets
    };

    for i in 0..n {
        let v_i = extracted[i];
        let row_i_start = row_offsets[i];
        let next_extracted = &extracted[(i + 1)..n];
        let next_offsets = &row_offsets[(i + 1)..n];
        let mut row_i_idx = row_i_start + i + 1;
        for (v_j, &offset_j) in next_extracted.iter().zip(next_offsets.iter()) {
            let dist = squared_l2_distance(v_i, v_j);
            distance_matrix[row_i_idx] = dist;
            distance_matrix[offset_j + i] = dist;
            row_i_idx += 1;
        }
    }

    let mut best: Option<(usize, f32)> = None;

    // Optimized: Mutate rows of `distance_matrix` in-place during the selection loop.
    // Since row `i` is never accessed in any subsequent iterations or outside this loop,
    // we can run `select_nth_unstable_by` directly on `distance_matrix[row_start..row_end]`.
    // This completely eliminates any helper vectors or temporary heap allocations,
    // and avoids ALL `copy_from_slice` overhead.
    // Note that the self-distance is 0.0, which is always the minimum possible squared distance, so the sum of the
    // `neighbors + 1` smallest distances including itself is mathematically identical to the sum of the
    // `neighbors` smallest distances excluding itself.
    for i in 0..n {
        let row_start = i * n;
        let row = &mut distance_matrix[row_start..(row_start + n)];

        // Use select_nth_unstable_by to partition the vector in O(N) time.
        // We select the `neighbors + 1` smallest elements (index `neighbors` in 0-indexed terms).
        let score: f32 = if neighbors > 0 {
            row.select_nth_unstable_by(neighbors, |a, b| a.total_cmp(b));
            // Optimized: Sum the `neighbors + 1` smallest distances using two independent accumulators (`sum0` and `sum1`).
            // This eliminates the single-accumulator sequential addition dependency chain, allowing CPU floating-point
            // execution units to perform additions concurrently in parallel and boosting candidate score calculation speed.
            let sub = &row[..=neighbors];
            let mut sum0 = 0.0_f32;
            let mut sum1 = 0.0_f32;
            let chunks = sub.chunks_exact(2);
            let rem = chunks.remainder();
            for chunk in chunks {
                sum0 += chunk[0];
                sum1 += chunk[1];
            }
            if !rem.is_empty() {
                sum0 += rem[0];
            }
            sum0 + sum1
        } else {
            0.0
        };

        match best {
            Some((_, best_score)) if best_score <= score => {}
            _ => best = Some((i, score)),
        }
    }

    best.map(|(idx, _)| extracted[idx].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_krum_chooses_honest_update() {
        let selected = multi_krum(
            &[
                vec![1.0, 1.0],
                vec![1.1, 1.0],
                vec![0.9, 1.1],
                vec![1.0, 0.95],
                vec![50.0, -50.0],
            ],
            1,
        )
        .unwrap();

        assert!(selected[0] < 2.0);
        assert!(selected[1] < 2.0);
    }
}
