pub fn federated_averaging(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dimension = vectors.first()?.len();
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return None;
    }

    // Optimized: If there is only one client vector to average, we can return a cloned copy of it immediately.
    // This completely bypasses any addition loops, bounds-check/assertion logic, and division/normalization multiplication.
    if vectors.len() == 1 {
        return Some(vectors[0].clone());
    }

    // Optimized: Initialize `acc` directly with a cloned copy of the first vector
    // rather than allocating a zero-filled vector and performing redundant addition in the first iteration.
    let mut acc = vectors[0].clone();
    let len = dimension;
    let acc_slice = &mut acc[..len];

    let remaining_vectors = &vectors[1..];

    if len <= 1024 {
        // Fast path for small dimensions: direct iteration to avoid chunking and branch overhead
        // Assert slice lengths to completely eliminate runtime bounds checks and enable full auto-vectorization.
        assert_eq!(acc_slice.len(), len);
        if remaining_vectors.len() >= 4 {
            // Optimized: Process remaining vectors in chunks of 4.
            // This significantly reduces memory bandwidth/traffic on `acc_slice` (reading and writing to it 75% fewer times)
            // and enables parallel floating-point operations in registers, maximizing ILP and compiler vectorization.
            let mut chunks = remaining_vectors.chunks_exact(4);
            for chunk in chunks.by_ref() {
                let v0 = &chunk[0][..len];
                let v1 = &chunk[1][..len];
                let v2 = &chunk[2][..len];
                let v3 = &chunk[3][..len];
                assert_eq!(v0.len(), len);
                assert_eq!(v1.len(), len);
                assert_eq!(v2.len(), len);
                assert_eq!(v3.len(), len);
                for i in 0..len {
                    acc_slice[i] += v0[i] + v1[i] + v2[i] + v3[i];
                }
            }
            for vector in chunks.remainder() {
                let vector_slice = &vector[..len];
                assert_eq!(vector_slice.len(), len);
                for i in 0..len {
                    acc_slice[i] += vector_slice[i];
                }
            }
        } else {
            for vector in remaining_vectors {
                let vector_slice = &vector[..len];
                assert_eq!(vector_slice.len(), len);
                for i in 0..len {
                    acc_slice[i] += vector_slice[i];
                }
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
            assert_eq!(acc_chunk.len(), chunk_len);

            // Optimized: Process remaining vectors in chunks of 4.
            // This reduces writeback overhead and cache traffic on `acc_chunk` by up to 75% inside the hot loop,
            // while maintaining strict alignment constraints and unleashing SIMD vectorization.
            let mut chunks = remaining_vectors.chunks_exact(4);
            for chunk in chunks.by_ref() {
                let v0 = &chunk[0][chunk_start..chunk_end];
                let v1 = &chunk[1][chunk_start..chunk_end];
                let v2 = &chunk[2][chunk_start..chunk_end];
                let v3 = &chunk[3][chunk_start..chunk_end];
                assert_eq!(v0.len(), chunk_len);
                assert_eq!(v1.len(), chunk_len);
                assert_eq!(v2.len(), chunk_len);
                assert_eq!(v3.len(), chunk_len);
                for i in 0..chunk_len {
                    acc_chunk[i] += v0[i] + v1[i] + v2[i] + v3[i];
                }
            }
            for vector in chunks.remainder() {
                let vector_chunk = &vector[chunk_start..chunk_end];
                assert_eq!(vector_chunk.len(), chunk_len);
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

    #[test]
    fn federated_averaging_chunks_exact_4_small_dimensions() {
        // Having 6 clients means vectors[0] is the accumulator, and remaining_vectors has 5 elements.
        // This exercises both the chunk of 4 loop and the remainder loop in small dimensions path.
        let inputs = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
            vec![5.0, 6.0],
            vec![6.0, 7.0],
        ];
        let avg = federated_averaging(&inputs).unwrap();
        // Sums: [21.0, 27.0]. Counts: 6. Averages: [3.5, 4.5]
        assert_eq!(avg, vec![3.5, 4.5]);
    }

    #[test]
    fn federated_averaging_large_dimensions_and_many_clients() {
        // Length 2000 (forces loop tiling) and 6 clients (forces both chunk_exact and remainder).
        let mut inputs = Vec::new();
        for i in 1..=6 {
            inputs.push(vec![i as f32; 2000]);
        }
        let avg = federated_averaging(&inputs).unwrap();
        // Expected value: mean of 1..=6 is 3.5.
        assert_eq!(avg.len(), 2000);
        for val in avg {
            assert_eq!(val, 3.5);
        }
    }
}
