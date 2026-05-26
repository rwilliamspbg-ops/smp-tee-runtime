# smp-tee-runtime

A hardened, minimal Rust runtime for federated-learning aggregation inside TEEs (SGX/TDX/SEV-SNP/Nitro).

## Usage

Build and test the crate locally:

```bash
cargo build
cargo test
cargo run
```

Run the example flows that demonstrate the public API:

```bash
cargo run --example basic_tee_call
cargo run --example xdp_integration
```

Run the benchmark suite that tracks aggregation and ingress simulation cost:

```bash
cargo bench --bench aggregation
```

## Repository layout

```text
smp-tee-runtime/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   ├── tee_interface/
│   │   ├── mod.rs
│   │   └── traits.rs
│   ├── data_pipeline/
│   │   ├── mod.rs
│   │   └── xdp_ingress.rs
│   └── aggregation/
│       ├── mod.rs
│       └── multi_krum.rs
├── build-scripts/
├── examples/
│   ├── basic_tee_call.rs
│   └── xdp_integration.rs
├── benches/
│   └── aggregation.rs
├── tests/
│   └── end_to_end.rs
├── CONTRIBUTING.md
└── SECURITY.md
```

### Targeted builds

- SGX/TDX: `cargo build --target <sgx-specific-toolchain>`
- SEV-SNP: build inside an SNP-enabled guest VM/toolchain environment.

## Performance Tracking

The table below records the current Criterion results for the shipped benchmark target. Re-run `cargo bench --bench aggregation` after performance-sensitive changes and update the values.

| Benchmark | Current result | What it measures |
| --- | --- | --- |
| `federated_averaging` | 35.325 ns to 36.458 ns | Mean aggregation over a small in-memory batch |
| `multi_krum` | 3.5111 ns to 3.6398 ns | Robust aggregation selection for a small candidate set |
| `simulated_packet_pointer_pass_1m` | 630.24 µs to 644.61 µs | Pointer-passing overhead for a 1M-packet ingress simulation |

## Example end-to-end flow

```bash
cargo run --example xdp_integration
```

This demonstrates: XDP-like ingress packet view -> TEE memory write -> aggregation -> output.

## Library Usage

Use the public API directly when embedding the runtime in another Rust crate:

```rust
use smp_tee_runtime::{AggregationAlgorithm, ComputationParams, InMemoryTee, TeeGuard};

let mut tee = InMemoryTee::default();
tee.initialize().expect("TEE init failed");

let left = tee.allocate_memory(8).expect("left allocation failed");
let right = tee.allocate_memory(8).expect("right allocation failed");

let _result = tee
	.execute_computation(
		&[left.cast_const(), right.cast_const()],
		&ComputationParams {
			algorithm: AggregationAlgorithm::FederatedAveraging,
		},
	)
	.expect("aggregation failed");
```
