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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn federated_averaging_returns_mean() {
        let avg = federated_averaging(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]).unwrap();
        assert_eq!(avg, vec![3.0, 4.0]);
    }
}
