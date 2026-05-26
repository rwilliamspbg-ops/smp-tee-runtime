use smp_tee_runtime::{AggregationAlgorithm, ComputationParams, InMemoryTee, TeeGuard};

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn main() {
    let mut tee = InMemoryTee::default();
    tee.initialize().expect("TEE init failed");

    let p1 = tee.allocate_memory(8).expect("allocation failed");
    let p2 = tee.allocate_memory(8).expect("allocation failed");

    tee.write_data(p1, &to_bytes(&[1.0, 3.0]))
        .expect("write failed");
    tee.write_data(p2, &to_bytes(&[3.0, 5.0]))
        .expect("write failed");

    let result = tee
        .execute_computation(
            &[p1.cast_const(), p2.cast_const()],
            &ComputationParams {
                algorithm: AggregationAlgorithm::FederatedAveraging,
            },
        )
        .expect("computation failed");

    println!("federated average = {:?}", to_f32(&result));
}
