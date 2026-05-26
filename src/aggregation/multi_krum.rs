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

pub fn federated_averaging(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dimension = vectors.first()?.len();
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return None;
    }

    let mut acc = vec![0.0_f32; dimension];
    for vector in vectors {
        for (sum, value) in acc.iter_mut().zip(vector.iter()) {
            *sum += value;
        }
    }

    let denom = vectors.len() as f32;
    acc.iter_mut().for_each(|sum| *sum /= denom);
    Some(acc)
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
    let mut best: Option<(usize, f32)> = None;

    for (i, candidate) in vectors.iter().enumerate() {
        let mut distances = vectors
            .iter()
            .enumerate()
            .filter_map(|(j, other)| {
                if i == j {
                    return None;
                }
                squared_l2_distance(candidate, other)
            })
            .collect::<Vec<_>>();

        distances.sort_by(|a, b| a.total_cmp(b));
        let score: f32 = distances.iter().take(neighbors).sum();

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
    fn federated_averaging_returns_mean() {
        let avg = federated_averaging(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        assert_eq!(avg, vec![3.0, 4.0]);
    }

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
