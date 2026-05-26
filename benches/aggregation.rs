use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smp_tee_runtime::{federated_averaging, multi_krum};

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
}

fn benchmark_packet_flow_simulation(c: &mut Criterion) {
    let packets = 1_000_000usize;
    let payload = vec![1_u8; 64];

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
    benchmark_fedavg,
    benchmark_multi_krum,
    benchmark_packet_flow_simulation
);
criterion_main!(benches);
