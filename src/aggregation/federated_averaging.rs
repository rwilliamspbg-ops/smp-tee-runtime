pub fn federated_averaging(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dimension = vectors.first()?.len();
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return None;
    }

    let mut acc = vec![0.0_f32; dimension];
    let len = dimension;
    let acc_slice = &mut acc[..len];

    if len <= 1024 {
        // Fast path for small dimensions: direct iteration to avoid chunking and branch overhead
        for vector in vectors {
            let vector_slice = &vector[..len];
            for i in 0..len {
                acc_slice[i] += vector_slice[i];
            }
        }
    } else {
        // Loop tiling/blocking for large dimensions: accumulate in small chunks (e.g., 1024 elements)
        // to keep data in L1/L2 cache and maximize SIMD auto-vectorization across multiple client vectors.
        const CHUNK_SIZE: usize = 1024;
        for chunk_start in (0..len).step_by(CHUNK_SIZE) {
            let chunk_end = if chunk_start + CHUNK_SIZE > len {
                len
            } else {
                chunk_start + CHUNK_SIZE
            };
            let chunk_len = chunk_end - chunk_start;
            let acc_chunk = &mut acc_slice[chunk_start..chunk_end];
            for vector in vectors {
                let vector_chunk = &vector[chunk_start..chunk_end];
                for i in 0..chunk_len {
                    acc_chunk[i] += vector_chunk[i];
                }
            }
        }
    }

    let denom = vectors.len() as f32;
    let inv_denom = 1.0_f32 / denom;
    for val in acc_slice.iter_mut() {
        *val *= inv_denom;
    }
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
