# Security Policy

## Threat model assumptions

- The host OS and hypervisor are untrusted for plaintext model data.
- Memory returned by `TeeGuard::allocate_memory` is treated as enclave-confined and unreadable by the host.
- The `TeeGuard` implementation is responsible for preserving confidentiality and integrity boundaries.

## Reporting

Please report vulnerabilities via private security advisories in this repository.
