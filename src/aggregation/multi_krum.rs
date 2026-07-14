fn squared_l2_distance(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }

    Some(
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| {
                let delta = l - r;
                delta * delta
            })
            .sum(),
    )
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

    for i in 0..n {
        let mut distances = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j {
                distances.push(distance_matrix[i * n + j]);
            }
        }

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
