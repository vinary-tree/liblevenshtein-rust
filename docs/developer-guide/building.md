# Building liblevenshtein

**Version**: 0.9.1
**Last Updated**: 2026-06-19

Comprehensive guide for building the liblevenshtein CLI and libraries.

## Quick Start

### Build Everything (Recommended)
```bash
# Release build with all features - CLI + Library
cargo build --release --all-features

# Outputs:
# - CLI Binary:       target/release/liblevenshtein
# - Shared Library:   target/release/libliblevenshtein.so (Linux)
# - Static Library:   target/release/libliblevenshtein.rlib
```

### Install CLI Globally
```bash
# Install with all features (compression, protobuf)
cargo install --path . --features cli,compression,protobuf

# Now available system-wide:
liblevenshtein --help
liblevenshtein repl --dict words.bin.gz --format bincode-gz
liblevenshtein query "test" --dict /usr/share/dict/words -m 2
liblevenshtein convert words.txt words.bin.gz --to-format bincode-gz
```

---

## Building the CLI

### Release Build (Optimized) - v0.2.0
```bash
# With all features (recommended)
cargo build --bin liblevenshtein --release --features cli,compression,protobuf

# Binary location: target/release/liblevenshtein
# Size: ~20 MB (with debug symbols for profiling)
```

### Development Build (Fast iteration)
```bash
cargo build --bin liblevenshtein --features cli,serialization

# Binary location: target/debug/liblevenshtein
# Much faster compile times, but slower execution
```

### Minimal CLI (No Serialization)
```bash
cargo build --bin liblevenshtein --release --features cli

# Smaller binary, but limited to text format dictionaries
```

### With Compression Only
```bash
cargo build --bin liblevenshtein --release --features cli,compression

# Includes bincode-gz and json-gz formats, excludes protobuf
```

### Strip Debug Symbols (Smallest Binary)
```bash
cargo build --bin liblevenshtein --release --features cli,compression,protobuf
strip target/release/liblevenshtein

# Reduces size from ~20 MB to ~5 MB
```

---

## Building the Library

### Static Library (Rust-to-Rust)
```bash
# Build Rust static library (.rlib)
cargo build --release

# Location: target/release/libliblevenshtein.rlib
# Use in other Rust projects by adding to Cargo.toml:
# [dependencies]
# liblevenshtein = { path = "../liblevenshtein-rust" }
```

### Shared/Dynamic Library (FFI/Other Languages)
```bash
# Build shared library (.so/.dylib/.dll)
cargo build --release

# Linux:   target/release/libliblevenshtein.so
# macOS:   target/release/libliblevenshtein.dylib
# Windows: target/release/liblevenshtein.dll

# Size: ~24 MB (includes all dependencies)
```

### C-Compatible Static Archive
```bash
# For linking with C/C++ programs
cargo rustc --release --crate-type=staticlib

# Creates: target/release/libliblevenshtein.a (Linux/macOS)
#          target/release/liblevenshtein.lib (Windows)
```

---

## Feature Flags

### Available Features

| Feature | Description | CLI Required | Size Impact |
|---------|-------------|--------------|-------------|
| `cli` | Command-line interface and REPL | Yes | +2 MB |
| `serialization` | Bincode, JSON support | No | +500 KB |
| `compression` | Gzip compression | No | +300 KB |
| `protobuf` | Protocol Buffers support | No | +1 MB |

### Feature Combinations

```bash
# Library only (minimal)
cargo build --release
# ~3 MB, core functionality only

# Library + Serialization
cargo build --release --features serialization
# ~3.5 MB, adds Bincode/JSON support

# Library + Serialization + Compression
cargo build --release --features serialization,compression
# ~3.8 MB, adds gzip compression

# Library + All Serialization
cargo build --release --features serialization,compression,protobuf
# ~4.8 MB, all format support

# CLI + Serialization
cargo build --bin liblevenshtein --release --features cli,serialization
# ~19 MB CLI binary

# CLI + Compression (v0.2.0 - recommended)
cargo build --bin liblevenshtein --release --features cli,compression
# ~19.5 MB CLI binary with gzip support

# CLI + All Features
cargo build --bin liblevenshtein --release --all-features
# ~20 MB CLI binary with everything
```

---

## Optimization Builds

### Maximum Performance (CPU-specific)
```bash
# CPU-specific tuning for maximum performance
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli,compression,protobuf

# 5-15% performance improvement on your specific CPU
# Binary won't be portable to other CPU architectures
```

### Link-Time Optimization (Already Enabled)
```toml
# Already configured in Cargo.toml:
[profile.release]
opt-level = 3
lto = true           # Link-time optimization
codegen-units = 1    # Single codegen unit for better optimization
```

### Size-Optimized Build
```bash
# Build for minimum size
RUSTFLAGS="-C opt-level=z -C lto=fat -C codegen-units=1" \
  cargo build --release --features cli,serialization

# Then strip symbols
strip target/release/liblevenshtein

# Results in ~4-5 MB binary (smaller but slightly slower)
```

---

## Platform-Specific Builds

### Linux
```bash
# Standard build
cargo build --release --features cli,compression,protobuf

# Statically linked (no glibc dependency)
cargo build --release --target x86_64-unknown-linux-musl \
  --features cli,compression,protobuf
```

### macOS
```bash
# Standard build
cargo build --release --features cli,compression,protobuf

# Universal binary (Intel + Apple Silicon)
rustup target add x86_64-apple-darwin aarch64-apple-darwin
cargo build --release --target x86_64-apple-darwin --features cli,compression,protobuf
cargo build --release --target aarch64-apple-darwin --features cli,compression,protobuf
lipo -create -output liblevenshtein \
  target/x86_64-apple-darwin/release/liblevenshtein \
  target/aarch64-apple-darwin/release/liblevenshtein
```

### Windows
```bash
# Standard build
cargo build --release --features cli,compression,protobuf

# Outputs:
# - target/release/liblevenshtein.exe
# - target/release/liblevenshtein.dll
# - target/release/liblevenshtein.dll.lib (import library)
```

---

## Cross-Compilation

### Install Cross-Compilation Tool
```bash
cargo install cross
```

### Build for Different Targets
```bash
# Linux ARM64 (e.g., Raspberry Pi)
cross build --release --target aarch64-unknown-linux-gnu \
  --features cli,compression,protobuf

# Linux ARMv7 (older ARM devices)
cross build --release --target armv7-unknown-linux-gnueabihf \
  --features cli,compression,protobuf

# Windows from Linux
cross build --release --target x86_64-pc-windows-gnu \
  --features cli,compression,protobuf
```

---

## Development Builds

### Fast Iteration (Debug Mode)
```bash
# Fastest compile time, slowest execution
cargo build --features cli,serialization
```

### Check Without Building
```bash
# Verify code compiles (much faster than full build)
cargo check --all-features
```

### Build Tests
```bash
# Build all tests
cargo test --no-run --all-features

# Run tests
cargo test --all-features
```

### Build Benchmarks
```bash
# Build benchmarks
cargo bench --no-run

# Run benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench
```

---

## Artifacts Location

After a release build, you'll find:

```
target/release/
├── liblevenshtein              # CLI binary (19 MB)
├── libliblevenshtein.so        # Shared library (24 MB, Linux)
├── libliblevenshtein.rlib      # Static Rust library (used by cargo)
└── deps/
    └── libliblevenshtein-*.rlib # Actual static library with hash
```

---

## Verification

### Verify CLI Build
```bash
./target/release/liblevenshtein --version
./target/release/liblevenshtein --help
./target/release/liblevenshtein config
```

### Verify Library Exports (Linux)
```bash
nm -D target/release/libliblevenshtein.so | grep " T " | head -10
objdump -T target/release/libliblevenshtein.so | head -20
```

### Check Binary Size
```bash
ls -lh target/release/liblevenshtein
ls -lh target/release/libliblevenshtein.so

# With breakdown
cargo bloat --release --features cli --features serialization -n 20
```

### Verify Dependencies
```bash
# Linux
ldd target/release/liblevenshtein

# macOS
otool -L target/release/liblevenshtein

# Windows
dumpbin /dependents target/release/liblevenshtein.exe
```

---

## Troubleshooting

### Link Errors
```bash
# If you get linker errors, try:
cargo clean
cargo build --release --features cli --features serialization
```

### Out of Memory During Build
```bash
# Reduce parallel jobs
cargo build --release --features cli --features serialization -j 2
```

### PathMap Dependency Issues
```bash
# Ensure PathMap is available
cd ../PathMap && cargo build --release
cd ../liblevenshtein-rust
cargo build --release --features cli --features serialization
```

### Feature Not Found
```bash
# List all available features
cargo metadata --no-deps --format-version 1 | jq '.packages[0].features'

# Or check Cargo.toml [features] section
grep -A 10 "\[features\]" Cargo.toml
```

---

## Performance Profiling

### Build for Profiling
```bash
# Already enabled in release profile
cargo build --release --features cli --features serialization

# The debug = true setting in [profile.release] includes symbols for profiling
```

### Generate Flamegraphs
```bash
cargo install flamegraph
RUSTFLAGS="-C target-cpu=native" cargo flamegraph --bin liblevenshtein \
  --features cli,compression,protobuf -- query "test" --dict /usr/share/dict/words -m 2
```

---

## Clean Builds

```bash
# Remove all build artifacts
cargo clean

# Remove specific target
cargo clean --release

# Remove and rebuild
cargo clean && cargo build --release --all-features
```

---

## Summary

**Most Common Use Cases:**

1. **Install CLI for daily use:**
   ```bash
   cargo install --path . --features cli,compression,protobuf
   ```

2. **Build optimized library for embedding:**
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release --features serialization,compression
   ```

3. **Development:**
   ```bash
   cargo build --features cli,serialization
   ```

4. **Everything (CLI + Library):**
   ```bash
   cargo build --release --all-features
   ```
