use smp_tee_runtime::{federated_averaging, InMemoryTee, TeeGuard};

fn main() {
    let gradients = vec![vec![1.0_f32, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
    let mut tee = InMemoryTee::default();
    tee.initialize()
        .expect("failed to initialize in-memory TEE");

    if let Some(avg) = federated_averaging(&gradients) {
        println!("fedavg result: {:?}", avg);
    }
}
