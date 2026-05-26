use smp_tee_runtime::{
    AggregationAlgorithm, ComputationParams, InMemoryTee, PacketView, TeeError, TeeGuard,
    XdpIngress,
};

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.flat_map(|value| value.to_le_bytes()).collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn direct_tee_call_round_trips_federated_averaging() {
    let mut tee = InMemoryTee::default();
    tee.initialize().expect("TEE init failed");

    let left = tee.allocate_memory(8).expect("left allocation failed");
    let right = tee.allocate_memory(8).expect("right allocation failed");

    tee.write_data(left, &encode_f32(&[1.0, 3.0]))
        .expect("left write failed");
    tee.write_data(right, &encode_f32(&[3.0, 5.0]))
        .expect("right write failed");

    let output = tee
        .execute_computation(
            &[left.cast_const(), right.cast_const()],
            &ComputationParams {
                algorithm: AggregationAlgorithm::FederatedAveraging,
            },
        )
        .expect("aggregation failed");

    assert_eq!(decode_f32(&output), vec![2.0, 4.0]);
}

#[test]
fn xdp_ingress_pipeline_matches_expected_average() {
    let mut tee = InMemoryTee::default();
    tee.initialize().expect("TEE init failed");

    let ingress = XdpIngress;
    let packet_a = encode_f32(&[1.0, 2.0]);
    let packet_b = encode_f32(&[3.0, 4.0]);

    let ptr_a = ingress
        .write_packet_into_tee(&mut tee, PacketView { data: &packet_a })
        .expect("failed to write packet A");
    let ptr_b = ingress
        .write_packet_into_tee(&mut tee, PacketView { data: &packet_b })
        .expect("failed to write packet B");

    let output = tee
        .execute_computation(
            &[ptr_a.cast_const(), ptr_b.cast_const()],
            &ComputationParams {
                algorithm: AggregationAlgorithm::FederatedAveraging,
            },
        )
        .expect("aggregation failed");

    assert_eq!(decode_f32(&output), vec![2.0, 3.0]);
}

#[test]
fn tee_rejects_misaligned_payloads() {
    let mut tee = InMemoryTee::default();
    tee.initialize().expect("TEE init failed");

    let ptr = tee.allocate_memory(3).expect("allocation failed");
    tee.write_data(ptr, &[1, 2, 3]).expect("write failed");

    let err = tee
        .execute_computation(
            &[ptr.cast_const()],
            &ComputationParams {
                algorithm: AggregationAlgorithm::FederatedAveraging,
            },
        )
        .expect_err("misaligned payload should fail");

    assert_eq!(
        err,
        TeeError::InvalidInput("payload length must be a multiple of 4")
    );
}

#[test]
fn tee_rejects_too_small_multi_krum_batches() {
    let mut tee = InMemoryTee::default();
    tee.initialize().expect("TEE init failed");

    let left = tee.allocate_memory(8).expect("left allocation failed");
    let right = tee.allocate_memory(8).expect("right allocation failed");

    tee.write_data(left, &encode_f32(&[1.0, 1.0]))
        .expect("left write failed");
    tee.write_data(right, &encode_f32(&[2.0, 2.0]))
        .expect("right write failed");

    let err = tee
        .execute_computation(
            &[left.cast_const(), right.cast_const()],
            &ComputationParams {
                algorithm: AggregationAlgorithm::MultiKrum {
                    byzantine_tolerance: 0,
                },
            },
        )
        .expect_err("multi-krum should reject undersized batches");

    assert_eq!(err, TeeError::InvalidInput("invalid multi-krum input"));
}
