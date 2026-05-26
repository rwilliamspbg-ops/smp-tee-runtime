# Contributing

## Development setup

```bash
cargo fmt
cargo test
cargo test --examples
cargo bench --bench aggregation
```

## Documentation updates

- Update [README.md](README.md) when the public usage flow changes.
- Refresh the performance tracking table in [README.md](README.md) after benchmark-affecting changes.
- Keep [build-scripts/README.md](build-scripts/README.md) aligned with any new target-specific helper scripts.

## Pull requests

- Keep changes focused and minimal.
- Add or update tests for behavior changes.
- Document security-impacting assumptions in `SECURITY.md`.
