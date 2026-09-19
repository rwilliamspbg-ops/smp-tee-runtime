// Optimized: Accept generic vector references `V` implementing `AsRef<[f32]>`.
// This allows caller contexts (like `InMemoryTee::execute_computation`) to execute averaging
// directly on zero-copy borrowed memory slices of floats, completely bypassing heavy float deserializations.
pub fn federated_averaging<V: AsRef<[f32]>>(vectors: &[V]) -> Option<Vec<f32>> {
    if vectors.is_empty() {
        return None;
    }

    // Pre-extract references to the underlying f32 slices to completely eliminate any
    // repeated `.as_ref()` method calls (which could involve virtual/generic dispatch, Match branches,
    // and layout checks) inside the hot loops.
    // To avoid expensive heap allocations for small vector counts, we use a stack-allocated buffer
    // for up to 64 vectors and fall back to a heap-allocated Vec for larger counts.
    // Using `MaybeUninit` avoids zero/dummy initialization of all 64 elements on every call.
    let mut stack_buf = [std::mem::MaybeUninit::<&[f32]>::uninit(); 64];
    let heap_buf: Vec<&[f32]>;
    let extracted: &[&[f32]] = if vectors.len() <= 64 {
        let len = vectors.len();
        for (dest, src) in stack_buf[..len].iter_mut().zip(vectors.iter()) {
            dest.write(src.as_ref());
        }
        unsafe { std::slice::from_raw_parts(stack_buf.as_ptr() as *const &[f32], len) }
    } else {
        heap_buf = vectors.iter().map(|v| v.as_ref()).collect();
        &heap_buf
    };

    let dimension = extracted[0].len();
    if extracted[1..].iter().any(|v| v.len() != dimension) {
        return None;
    }

    // Optimized: If there is only one client vector to average, we can return a cloned copy of it immediately.
    // This completely bypasses any addition loops, bounds-check/assertion logic, and division/normalization multiplication.
    if extracted.len() == 1 {
        return Some(extracted[0].to_vec());
    }

    // Optimized: Initialize `acc` directly with a cloned copy of the first vector
    // rather than allocating a zero-filled vector and performing redundant addition in the first iteration.
    let mut acc = extracted[0].to_vec();
    let len = dimension;
    let acc_slice = &mut acc[..len];

    let remaining_vectors = &extracted[1..];

    if len <= 1024 {
        // Fast path for small dimensions: direct iteration to avoid chunking and branch overhead.
        if remaining_vectors.len() >= 4 {
            // Optimized: Process remaining vectors in chunks of 4.
            // Avoid `.by_ref()` iterator indirection and tautological assertions to maximize SIMD instruction scheduling.
            let chunks = remaining_vectors.chunks_exact(4);
            let remainder = chunks.remainder();
            for chunk in chunks {
                let chunk: &[&[f32]; 4] = chunk.try_into().unwrap();
                let v0 = &chunk[0][..len];
                let v1 = &chunk[1][..len];
                let v2 = &chunk[2][..len];
                let v3 = &chunk[3][..len];
                for i in 0..len {
                    // Optimized: Group additions as `(v0[i] + v1[i]) + (v2[i] + v3[i])` to reduce
                    // instruction dependency chain latency from 4 sequential additions to 3.
                    // This allows the CPU's execution units to compute `v0 + v1` and `v2 + v3` in parallel,
                    // maximizing Instruction-Level Parallelism (ILP).
                    acc_slice[i] += (v0[i] + v1[i]) + (v2[i] + v3[i]);
                }
            }
            match remainder.len() {
                3 => {
                    // Convert remainder slice to fixed-size array reference `&[&[f32]; 3]` to statically
                    // elide runtime bounds checks on `rem[0]`, `rem[1]`, and `rem[2]`.
                    let rem: &[&[f32]; 3] = remainder.try_into().unwrap();
                    let v0 = &rem[0][..len];
                    let v1 = &rem[1][..len];
                    let v2 = &rem[2][..len];
                    for i in 0..len {
                        acc_slice[i] += (v0[i] + v1[i]) + v2[i];
                    }
                }
                2 => {
                    // Convert remainder slice to fixed-size array reference `&[&[f32]; 2]` to statically
                    // elide runtime bounds checks on `rem[0]` and `rem[1]`.
                    let rem: &[&[f32]; 2] = remainder.try_into().unwrap();
                    let v0 = &rem[0][..len];
                    let v1 = &rem[1][..len];
                    for i in 0..len {
                        acc_slice[i] += v0[i] + v1[i];
                    }
                }
                1 => {
                    let v0 = &remainder[0][..len];
                    for i in 0..len {
                        acc_slice[i] += v0[i];
                    }
                }
                _ => {}
            }
        } else {
            // Optimized: Fuse addition and normalization into a single pass when remaining_vectors.len() < 4.
            // This avoids a second pass over memory (`acc_slice`) for normalization.
            let denom = extracted.len() as f32;
            let inv_denom = 1.0_f32 / denom;
            match remaining_vectors.len() {
                3 => {
                    let rem: &[&[f32]; 3] = remaining_vectors.try_into().unwrap();
                    let v0 = &rem[0][..len];
                    let v1 = &rem[1][..len];
                    let v2 = &rem[2][..len];
                    for i in 0..len {
                        acc_slice[i] = (acc_slice[i] + (v0[i] + v1[i]) + v2[i]) * inv_denom;
                    }
                }
                2 => {
                    let rem: &[&[f32]; 2] = remaining_vectors.try_into().unwrap();
                    let v0 = &rem[0][..len];
                    let v1 = &rem[1][..len];
                    for i in 0..len {
                        acc_slice[i] = (acc_slice[i] + v0[i] + v1[i]) * inv_denom;
                    }
                }
                1 => {
                    let v0 = &remaining_vectors[0][..len];
                    for i in 0..len {
                        acc_slice[i] = (acc_slice[i] + v0[i]) * inv_denom;
                    }
                }
                _ => {
                    for val in acc_slice.iter_mut() {
                        *val *= inv_denom;
                    }
                }
            }
            return Some(acc);
        }
    } else {
        // Loop tiling/blocking for large dimensions: accumulate in small chunks (e.g., 1024 elements)
        // to keep data in L1/L2 cache and maximize SIMD auto-vectorization across multiple client vectors.
        const CHUNK_SIZE: usize = 1024;

        // Optimized: Hoist remaining_vectors chunking and array conversions outside of the outer
        // dimension tiling loop. This avoids re-creating the chunk iterator and performing repeated `try_into()`
        // array conversions on every dimension tile step.
        let chunks_exact = remaining_vectors.chunks_exact(4);
        let remainder = chunks_exact.remainder();
        let num_chunks = chunks_exact.len();

        let mut stack_chunks = [std::mem::MaybeUninit::<&[&[f32]; 4]>::uninit(); 16];
        let heap_chunks: Vec<&[&[f32]; 4]>;
        let vector_chunks: &[&[&[f32]; 4]] = if num_chunks <= 16 {
            for (dest, src) in stack_chunks[..num_chunks].iter_mut().zip(chunks_exact) {
                dest.write(src.try_into().unwrap());
            }
            unsafe {
                std::slice::from_raw_parts(stack_chunks.as_ptr() as *const &[&[f32]; 4], num_chunks)
            }
        } else {
            heap_chunks = chunks_exact.map(|c| c.try_into().unwrap()).collect();
            &heap_chunks
        };

        for chunk_start in (0..len).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(len);
            let chunk_len = chunk_end - chunk_start;
            let acc_chunk = &mut acc_slice[chunk_start..chunk_end];

            for &chunk in vector_chunks {
                let v0 = &chunk[0][chunk_start..chunk_end];
                let v1 = &chunk[1][chunk_start..chunk_end];
                let v2 = &chunk[2][chunk_start..chunk_end];
                let v3 = &chunk[3][chunk_start..chunk_end];
                for i in 0..chunk_len {
                    // Group additions as `(v0[i] + v1[i]) + (v2[i] + v3[i])` to minimize instruction latency.
                    acc_chunk[i] += (v0[i] + v1[i]) + (v2[i] + v3[i]);
                }
            }

            match remainder.len() {
                3 => {
                    // Convert remainder slice to fixed-size array reference `&[&[f32]; 3]` to statically
                    // elide runtime bounds checks on `rem[0]`, `rem[1]`, and `rem[2]`.
                    let rem: &[&[f32]; 3] = remainder.try_into().unwrap();
                    let v0 = &rem[0][chunk_start..chunk_end];
                    let v1 = &rem[1][chunk_start..chunk_end];
                    let v2 = &rem[2][chunk_start..chunk_end];
                    for i in 0..chunk_len {
                        acc_chunk[i] += (v0[i] + v1[i]) + v2[i];
                    }
                }
                2 => {
                    // Convert remainder slice to fixed-size array reference `&[&[f32]; 2]` to statically
                    // elide runtime bounds checks on `rem[0]` and `rem[1]`.
                    let rem: &[&[f32]; 2] = remainder.try_into().unwrap();
                    let v0 = &rem[0][chunk_start..chunk_end];
                    let v1 = &rem[1][chunk_start..chunk_end];
                    for i in 0..chunk_len {
                        acc_chunk[i] += v0[i] + v1[i];
                    }
                }
                1 => {
                    let v0 = &remainder[0][chunk_start..chunk_end];
                    for i in 0..chunk_len {
                        acc_chunk[i] += v0[i];
                    }
                }
                _ => {}
            }
        }
    }

    let denom = extracted.len() as f32;
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
    fn federated_averaging_4_and_5_clients() {
        let inputs4 = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let avg4 = federated_averaging(&inputs4).unwrap();
        assert_eq!(avg4, vec![4.0, 5.0]);

        let inputs5 = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
            vec![5.0, 6.0],
        ];
        let avg5 = federated_averaging(&inputs5).unwrap();
        assert_eq!(avg5, vec![3.0, 4.0]);
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
