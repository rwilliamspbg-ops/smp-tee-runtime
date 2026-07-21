#[inline(always)]
fn squared_l2_distance(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }

    // Optimized: Manual loop unrolling by 4 with independent accumulators
    // zipped with `chunks_exact(4)` and cast to fixed-size array references.
    // This completely eliminates bounds checks, reduces data dependency latency
    // (by avoiding immediate dependencies on `sum`), and maximizes instruction-level parallelism (ILP).
    let mut sum0 = 0.0_f32;
    let mut sum1 = 0.0_f32;
    let mut sum2 = 0.0_f32;
    let mut sum3 = 0.0_f32;

    let mut left_chunks = left.chunks_exact(4);
    let mut right_chunks = right.chunks_exact(4);

    for (cl, cr) in left_chunks.by_ref().zip(right_chunks.by_ref()) {
        let cl: &[f32; 4] = cl.try_into().unwrap();
        let cr: &[f32; 4] = cr.try_into().unwrap();

        let d0 = cl[0] - cr[0];
        let d1 = cl[1] - cr[1];
        let d2 = cl[2] - cr[2];
        let d3 = cl[3] - cr[3];

        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }

    let mut sum = sum0 + sum1 + sum2 + sum3;

    // Handle any remaining elements when length is not a multiple of 4
    let rem_l = left_chunks.remainder();
    let rem_r = right_chunks.remainder();
    for (l, r) in rem_l.iter().zip(rem_r.iter()) {
        let delta = l - r;
        sum += delta * delta;
    }

    Some(sum)
}

pub fn multi_krum(vectors: &[Vec<f32>], byzantine_tolerance: usize) -> Option<Vec<f32>> {
    let n = vectors.len();
    if n < 2 * byzantine_tolerance + 3 {
        return None;
    }
    let dimension = vectors.first()?.len();
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return None;
    }

    let neighbors = n.checked_sub(byzantine_tolerance + 2)?;

    // Precompute symmetric pairwise distances to reduce distance calculation count by 50%
    let mut distance_matrix = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = squared_l2_distance(&vectors[i], &vectors[j])?;
            distance_matrix[i * n + j] = dist;
            distance_matrix[j * n + i] = dist;
        }
    }

    let mut best: Option<(usize, f32)> = None;

    // Reuse a single pre-allocated distances buffer to avoid repeated vector allocations inside the loop
    let mut distances = vec![0.0_f32; n - 1];

    for i in 0..n {
        let row_start = i * n;
        distances[..i].copy_from_slice(&distance_matrix[row_start..(row_start + i)]);
        distances[i..].copy_from_slice(&distance_matrix[(row_start + i + 1)..(row_start + n)]);

        // Use select_nth_unstable_by to partition the vector in O(N) time
        // so the `neighbors` smallest distances are at indices 0..neighbors.
        let score: f32 = if neighbors > 0 {
            distances.select_nth_unstable_by(neighbors - 1, |a, b| a.total_cmp(b));
            distances[..neighbors].iter().sum()
        } else {
            0.0
        };

        match best {
            Some((_, best_score)) if best_score <= score => {}
            _ => best = Some((i, score)),
        }
    }

    best.map(|(idx, _)| vectors[idx].clone())
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
