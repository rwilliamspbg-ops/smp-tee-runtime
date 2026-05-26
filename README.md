# smp-tee-runtime

A hardened, minimal Rust runtime for federated-learning aggregation inside TEEs (SGX/TDX/SEV-SNP/Nitro).

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
├── CONTRIBUTING.md
└── SECURITY.md
```

## Build

```bash
cargo build
cargo test
cargo run
```

### Targeted builds

- SGX/TDX: `cargo build --target <sgx-specific-toolchain>`
- SEV-SNP: build inside an SNP-enabled guest VM/toolchain environment.

## Example end-to-end flow

```bash
cargo run --example xdp_integration
```

This demonstrates: XDP-like ingress packet view -> TEE memory write -> aggregation -> output.
