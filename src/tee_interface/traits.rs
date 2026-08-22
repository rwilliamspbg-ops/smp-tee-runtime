use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use crate::aggregation::{federated_averaging, multi_krum};

/// A high-performance, non-cryptographic hasher optimized specifically for `usize` pointer keys.
/// It completely bypasses SipHash overhead (which is designed for HashDoS resistance) to speed up
/// TEE memory allocation lookup on performance-critical paths.
#[derive(Default, Clone, Copy)]
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    #[inline(always)]
    fn add_to_hash(&mut self, i: u64) {
        const K: u64 = 0x517cc1b727220a95;
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(K);
    }
}

impl Hasher for FastHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.add_to_hash(byte as u64);
        }
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.add_to_hash(i as u64);
    }

    #[inline(always)]
    fn write_u16(&mut self, i: u16) {
        self.add_to_hash(i as u64);
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.add_to_hash(i as u64);
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.add_to_hash(i);
    }

    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.add_to_hash(i as u64);
    }
}

/// A builder for `FastHasher`.
#[derive(Default, Clone, Copy)]
pub struct BuildFastHasher;

impl BuildHasher for BuildFastHasher {
    type Hasher = FastHasher;

    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher {
        FastHasher { hash: 0 }
    }
}

/// A clone-on-write styled slice containing either borrowed or owned floating-point data.
/// This enables zero-copy passing of contiguous floating-point data extracted directly from TEE allocations.
#[derive(Debug, Clone)]
pub enum CowSlice<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl<'a> AsRef<[f32]> for CowSlice<'a> {
    #[inline(always)]
    fn as_ref(&self) -> &[f32] {
        match self {
            CowSlice::Borrowed(slice) => slice,
            CowSlice::Owned(vec) => vec,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregationAlgorithm {
    FederatedAveraging,
    MultiKrum { byzantine_tolerance: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputationParams {
    pub algorithm: AggregationAlgorithm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeeError {
    NotInitialized,
    InvalidAllocationSize,
    InvalidPointer,
    InvalidInput(&'static str),
}

pub trait TeeGuard {
    fn initialize(&mut self) -> Result<(), TeeError>;
    fn allocate_memory(&mut self, size: usize) -> Result<*mut u8, TeeError>;
    fn write_data(&mut self, ptr: *mut u8, data: &[u8]) -> Result<(), TeeError>;
    fn execute_computation(
        &self,
        input_ptrs: &[*const u8],
        params: &ComputationParams,
    ) -> Result<Vec<u8>, TeeError>;
}

#[derive(Debug, Default)]
pub struct InMemoryTee {
    initialized: bool,
    allocations: HashMap<usize, Vec<u8>, BuildFastHasher>,
}

impl InMemoryTee {
    /// Extracts a highly-optimized zero-copy reference to the underlying allocation if aligned and on little-endian,
    /// or decodes/fallback converts to an owned vector.
    fn get_input_slice<'a>(&'a self, ptr: *const u8) -> Result<CowSlice<'a>, TeeError> {
        let bytes = self
            .allocations
            .get(&(ptr as usize))
            .ok_or(TeeError::InvalidPointer)?;
        if bytes.len() % 4 != 0 {
            return Err(TeeError::InvalidInput(
                "payload length must be a multiple of 4",
            ));
        }

        #[cfg(target_endian = "little")]
        {
            let raw_ptr = bytes.as_ptr();
            // Check if the memory alignment matches f32 (4 bytes).
            // This is safe since heap-allocated vectors are always at least 8-byte aligned on modern architectures.
            if (raw_ptr as usize).is_multiple_of(std::mem::align_of::<f32>()) {
                let slice =
                    unsafe { std::slice::from_raw_parts(raw_ptr as *const f32, bytes.len() / 4) };
                return Ok(CowSlice::Borrowed(slice));
            }
        }

        // Fallback for non-little-endian architectures or unaligned buffers
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(CowSlice::Owned(values))
    }

    fn encode_vector(values: &[f32]) -> Vec<u8> {
        #[cfg(target_endian = "little")]
        {
            // Optimized: On little-endian architectures, copy the float slice directly
            // into a pre-allocated but uninitialized byte vector via `copy_nonoverlapping`.
            // This avoids any loop/element-wise overhead and completely bypasses bounds checks.
            let byte_len = values.len() * 4;
            let mut bytes = Vec::with_capacity(byte_len);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr() as *const u8,
                    bytes.as_mut_ptr(),
                    byte_len,
                );
                bytes.set_len(byte_len);
            }
            bytes
        }

        #[cfg(not(target_endian = "little"))]
        {
            let mut bytes = vec![0_u8; values.len() * 4];
            for (value, chunk) in values.iter().zip(bytes.chunks_exact_mut(4)) {
                chunk.copy_from_slice(&value.to_le_bytes());
            }
            bytes
        }
    }
}

impl TeeGuard for InMemoryTee {
    fn initialize(&mut self) -> Result<(), TeeError> {
        self.initialized = true;
        Ok(())
    }

    fn allocate_memory(&mut self, size: usize) -> Result<*mut u8, TeeError> {
        if !self.initialized {
            return Err(TeeError::NotInitialized);
        }
        if size == 0 {
            return Err(TeeError::InvalidAllocationSize);
        }

        let mut allocation = vec![0_u8; size];
        let ptr = allocation.as_mut_ptr();
        self.allocations.insert(ptr as usize, allocation);
        Ok(ptr)
    }

    fn write_data(&mut self, ptr: *mut u8, data: &[u8]) -> Result<(), TeeError> {
        if !self.initialized {
            return Err(TeeError::NotInitialized);
        }

        let buffer = self
            .allocations
            .get_mut(&(ptr as usize))
            .ok_or(TeeError::InvalidPointer)?;

        if data.len() > buffer.len() {
            return Err(TeeError::InvalidAllocationSize);
        }

        buffer[..data.len()].copy_from_slice(data);
        Ok(())
    }

    fn execute_computation(
        &self,
        input_ptrs: &[*const u8],
        params: &ComputationParams,
    ) -> Result<Vec<u8>, TeeError> {
        if !self.initialized {
            return Err(TeeError::NotInitialized);
        }

        if input_ptrs.is_empty() {
            return Err(TeeError::InvalidInput("empty inputs"));
        }

        // Optimized: Introduce a single-pointer fast path for Federated Averaging.
        // Instead of converting bytes to f32 vectors, running averaging, cloning, and converting
        // back to bytes, we directly return a cloned copy of the source vector bytes.
        // This reduces heap allocations from 3 to 1 and completely elides floating-point deserialization/serialization.
        if input_ptrs.len() == 1 && params.algorithm == AggregationAlgorithm::FederatedAveraging {
            let ptr = input_ptrs[0];
            let bytes = self
                .allocations
                .get(&(ptr as usize))
                .ok_or(TeeError::InvalidPointer)?;
            if bytes.len() % 4 != 0 {
                return Err(TeeError::InvalidInput(
                    "payload length must be a multiple of 4",
                ));
            }
            return Ok(bytes.clone());
        }

        // Highly Optimized: Extract zero-copy `&[f32]` references directly from TEE allocated memory.
        // Storing `Copy` slice references `&[f32]` in the stack buffer completely eliminates drop flags and
        // destructor calls. For borrowed slices, lifetime is tied to `&self`, bypassing intermediate allocations.
        let mut stack_vectors = [&[] as &[f32]; 64];
        let mut heap_owned: Vec<Vec<f32>> = Vec::new();
        let heap_cows: Vec<CowSlice<'_>>;
        let heap_vectors: Vec<&[f32]>;
        let vectors: &[&[f32]] = if input_ptrs.len() <= 64 {
            let len = input_ptrs.len();
            for (dest, &ptr) in stack_vectors[..len].iter_mut().zip(input_ptrs.iter()) {
                match self.get_input_slice(ptr)? {
                    CowSlice::Borrowed(slice) => *dest = slice,
                    CowSlice::Owned(vec) => {
                        heap_owned.push(vec);
                        let last = heap_owned.last().unwrap();
                        *dest = unsafe { std::slice::from_raw_parts(last.as_ptr(), last.len()) };
                    }
                }
            }
            &stack_vectors[..len]
        } else {
            let mut cows = Vec::with_capacity(input_ptrs.len());
            for &ptr in input_ptrs {
                cows.push(self.get_input_slice(ptr)?);
            }
            heap_cows = cows;
            heap_vectors = heap_cows
                .iter()
                .map(|c| match c {
                    CowSlice::Borrowed(s) => *s,
                    CowSlice::Owned(v) => v.as_slice(),
                })
                .collect();
            &heap_vectors
        };

        let result = match params.algorithm {
            AggregationAlgorithm::FederatedAveraging => federated_averaging(vectors)
                .ok_or(TeeError::InvalidInput("invalid federated averaging input"))?,
            AggregationAlgorithm::MultiKrum {
                byzantine_tolerance,
            } => multi_krum(vectors, byzantine_tolerance)
                .ok_or(TeeError::InvalidInput("invalid multi-krum input"))?,
        };

        Ok(Self::encode_vector(&result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn tee_executes_federated_averaging() {
        let mut tee = InMemoryTee::default();
        tee.initialize().unwrap();

        let p1 = tee.allocate_memory(8).unwrap();
        let p2 = tee.allocate_memory(8).unwrap();

        tee.write_data(p1, &to_bytes(&[1.0, 3.0])).unwrap();
        tee.write_data(p2, &to_bytes(&[3.0, 5.0])).unwrap();

        let out = tee
            .execute_computation(
                &[p1.cast_const(), p2.cast_const()],
                &ComputationParams {
                    algorithm: AggregationAlgorithm::FederatedAveraging,
                },
            )
            .unwrap();

        assert_eq!(to_f32(&out), vec![2.0, 4.0]);
    }
}
