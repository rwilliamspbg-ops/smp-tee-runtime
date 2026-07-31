use std::collections::HashMap;

use crate::aggregation::{federated_averaging, multi_krum};

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
    allocations: HashMap<usize, Vec<u8>>,
}

impl InMemoryTee {
    fn read_vector(&self, ptr: *const u8) -> Result<Vec<f32>, TeeError> {
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
            // Optimized: On little-endian architectures, copy the byte slice directly
            // into a pre-allocated but uninitialized float vector via `copy_nonoverlapping`.
            // This avoids any loop/element-wise overhead and completely bypasses bounds checks.
            let float_len = bytes.len() / 4;
            let mut values = Vec::with_capacity(float_len);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    values.as_mut_ptr() as *mut u8,
                    bytes.len(),
                );
                values.set_len(float_len);
            }
            Ok(values)
        }

        #[cfg(not(target_endian = "little"))]
        {
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Ok(values)
        }
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

        let vectors = input_ptrs
            .iter()
            .map(|ptr| self.read_vector(*ptr))
            .collect::<Result<Vec<_>, _>>()?;

        let result = match params.algorithm {
            AggregationAlgorithm::FederatedAveraging => federated_averaging(&vectors)
                .ok_or(TeeError::InvalidInput("invalid federated averaging input"))?,
            AggregationAlgorithm::MultiKrum {
                byzantine_tolerance,
            } => multi_krum(&vectors, byzantine_tolerance)
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
