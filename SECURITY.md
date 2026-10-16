# Security Policy

## Supported Artifacts

Only official evaluation packages and native shared libraries (`libsilicium_ann_x86_64.so`) matching the cryptographic SHA-256 digests listed in `checksums.sha256` are supported for benchmark verification.

## Integrity Verification

Before executing any benchmark or loading the native shared library via C-ABI, verify file integrity:

```bash
sha256sum -c checksums.sha256
```

## Reporting a Vulnerability

If you discover a memory safety issue, C-ABI boundary vulnerability, or artifact integrity discrepancy, please open a confidential security advisory on the upstream evaluation repository or contact the Silicium Architecture Research Team with a minimal reproduction script and hardware environment details.
