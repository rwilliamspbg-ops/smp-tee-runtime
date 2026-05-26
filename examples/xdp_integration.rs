use smp_tee_runtime::{
    AggregationAlgorithm, ComputationParams, InMemoryTee, PacketView, TeeGuard, XdpIngress,
};

fn encode(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn main() {
    let mut tee = InMemoryTee::default();
    tee.initialize().expect("TEE init failed");

    let ingress = XdpIngress;

    let packet_a = PacketView {
        data: &encode(&[1.0, 2.0]),
    };
    let packet_b = PacketView {
        data: &encode(&[3.0, 4.0]),
    };

    let ptr_a = ingress
        .write_packet_into_tee(&mut tee, packet_a)
        .expect("failed to push packet A into TEE memory");
    let ptr_b = ingress
        .write_packet_into_tee(&mut tee, packet_b)
        .expect("failed to push packet B into TEE memory");

    let output = tee
        .execute_computation(
            &[ptr_a.cast_const(), ptr_b.cast_const()],
            &ComputationParams {
                algorithm: AggregationAlgorithm::FederatedAveraging,
            },
        )
        .expect("aggregation failed");

    println!("end-to-end output = {:?}", decode(&output));
}
