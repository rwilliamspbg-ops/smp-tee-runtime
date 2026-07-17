use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smp_tee_runtime::{federated_averaging, multi_krum, XdpIngress};

fn benchmark_ring_buffer_parsing(c: &mut Criterion) {
    let size = 65536; // 64KB
    let ring_buffer = vec![0_u8; size];
    let frame_size = 64;

    c.bench_function("parse_ring_buffer_64kb", |b| {
        b.iter(|| XdpIngress::parse_ring_buffer(black_box(&ring_buffer), black_box(frame_size)))
    });
}

fn benchmark_fedavg(c: &mut Criterion) {
    let input = vec![
        vec![1.0_f32, 2.0, 3.0, 4.0],
        vec![1.1_f32, 2.1, 3.1, 4.1],
        vec![0.9_f32, 1.9, 2.9, 3.9],
    ];

    c.bench_function("federated_averaging", |b| {
        b.iter(|| federated_averaging(black_box(&input)))
    });
}

fn benchmark_fedavg_large(c: &mut Criterion) {
    let mut large_input = Vec::new();
    for i in 0..50 {
        large_input.push(vec![i as f32; 10000]);
    }

    c.bench_function("federated_averaging_large_50_clients", |b| {
        b.iter(|| federated_averaging(black_box(&large_input)))
    });
}

fn benchmark_tee_encode_decode(c: &mut Criterion) {
    use smp_tee_runtime::{AggregationAlgorithm, ComputationParams, InMemoryTee, TeeGuard};

    let mut tee = InMemoryTee::default();
    tee.initialize().unwrap();

    let size = 10000 * 4; // 10000 f32s
    let p1 = tee.allocate_memory(size).unwrap();
    let data = vec![1.23_f32; 10000];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    tee.write_data(p1, &bytes).unwrap();

    c.bench_function("tee_execute_computation", |b| {
        b.iter(|| {
            tee.execute_computation(
                black_box(&[p1.cast_const()]),
                black_box(&ComputationParams {
                    algorithm: AggregationAlgorithm::FederatedAveraging,
                }),
            )
            .unwrap()
        })
    });
}

fn benchmark_multi_krum(c: &mut Criterion) {
    let input = vec![
        vec![1.0_f32, 2.0, 3.0, 4.0],
        vec![1.1_f32, 2.1, 3.1, 4.1],
        vec![0.9_f32, 1.9, 2.9, 3.9],
        vec![50.0_f32, -50.0, 50.0, -50.0],
    ];

    c.bench_function("multi_krum", |b| {
        b.iter(|| multi_krum(black_box(&input), black_box(1)))
    });

    // A larger, more realistic benchmark for multi_krum
    let mut large_input = Vec::new();
    for i in 0..50 {
        large_input.push(vec![i as f32; 128]);
    }

    c.bench_function("multi_krum_large_50_clients", |b| {
        b.iter(|| multi_krum(black_box(&large_input), black_box(10)))
    });
}

fn benchmark_packet_flow_simulation(c: &mut Criterion) {
    let packets = 1_000_000usize;
    let payload = [1_u8; 64];

    c.bench_function("simulated_packet_pointer_pass_1m", |b| {
        b.iter(|| {
            let mut bytes_seen = 0usize;
            for _ in 0..packets {
                bytes_seen += black_box(payload.as_ptr() as usize & 0x1) + black_box(payload.len());
            }
            bytes_seen
        })
    });
}

criterion_group!(
    benches,
    benchmark_ring_buffer_parsing,
    benchmark_fedavg,
    benchmark_fedavg_large,
    benchmark_tee_encode_decode,
    benchmark_multi_krum,
    benchmark_packet_flow_simulation
);
criterion_main!(benches);
