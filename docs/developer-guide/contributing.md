# Contributing to liblevenshtein-rust

**Version**: 0.9.1
**Last Updated**: 2026-06-19

Thank you for your interest in contributing to liblevenshtein-rust!

## Development Setup

### Install Git Hooks (Recommended)

After cloning the repository, install the git hooks to prevent common mistakes:

```bash
./scripts/install-git-hooks.sh
```

This installs hooks that:
- Prevent accidentally committing local development overrides (e.g., uncommented `[patch]` sections)
- Ensure Cargo.toml uses git dependencies instead of local paths

See [.githooks/README.md](../../.githooks/README.md) for more details.

### Prerequisites

- Rust 1.70 or later
- Git
- Protocol Buffers compiler (optional, for `protobuf` feature)
  - Linux: `sudo apt-get install protobuf-compiler`
  - macOS: `brew install protobuf`
  - Windows: Download from [protobuf releases](https://github.com/protocolbuffers/protobuf/releases)

### Building

The project automatically fetches PathMap from GitHub. Just build:

```bash
# Build with CPU-specific optimizations
cargo build --all-features
```

**For local PathMap development** (optional):

If you need to modify PathMap, uncomment the `[patch]` section in `Cargo.toml`:

```bash
# Clone PathMap as a sibling directory
cd ..
git clone https://github.com/Adam-Vandervorst/PathMap.git PathMap
cd liblevenshtein-rust

# Uncomment the [patch] section at the end of Cargo.toml
# [patch.'https://github.com/Adam-Vandervorst/PathMap.git']
# pathmap = { path = "../PathMap" }

cargo build --all-features
```

See [building.md](building.md) for comprehensive build instructions.

### Running Tests

```bash
# Run all tests
RUSTFLAGS="-C target-cpu=native" cargo test --all-features

# Run tests for specific features
RUSTFLAGS="-C target-cpu=native" cargo test --features compression,protobuf
```

### Running Examples

```bash
# Code completion demo
RUSTFLAGS="-C target-cpu=native" cargo run --example code_completion_demo

# DynamicDawg demo
RUSTFLAGS="-C target-cpu=native" cargo run --example dynamic_dawg_demo

# Contextual filtering
RUSTFLAGS="-C target-cpu=native" cargo run --example advanced_contextual_filtering

# Dynamic dictionary updates
cargo run --example dynamic_dictionary
```

### Benchmarks

```bash
# Run all benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench

# Run specific benchmark suite
RUSTFLAGS="-C target-cpu=native" cargo bench --bench serialization_benchmarks --features compression,protobuf
RUSTFLAGS="-C target-cpu=native" cargo bench --bench filtering_prefix_benchmarks
```

## Code Style

- Follow Rust standard formatting (`cargo fmt`)
- Ensure clippy passes (`cargo clippy --all-features`)
- Add documentation for public APIs
- Include tests for new functionality
- Write clear commit messages following conventional commits format
- Update CHANGELOG.md for user-facing changes

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Update documentation as needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Areas for Contribution

See `FUTURE_ENHANCEMENTS.md` for detailed planned features.

### High Priority

- Additional algorithm variants (e.g., Damerau-Levenshtein with bounded deletions)
- Further performance optimizations (SIMD, parallel queries)
- Improve test coverage for edge cases
- FFI bindings for C/C++ integration

### Medium Priority

- Additional serialization formats (MessagePack, CBOR)
- Dictionary builder optimizations
- CLI batch processing and watch mode belong in
  [`liblevenshtein-rust-cli`](https://github.com/vinary-tree/liblevenshtein-rust-cli)
- More comprehensive benchmarking suite

### Documentation

- API documentation improvements
- Usage tutorials and cookbooks
- Performance tuning guides
- Algorithm comparison studies
- Integration examples for common frameworks

### Examples

- Web service integration (Actix, Axum)
- IDE plugin demonstration
- Real-time search applications
- Multi-language dictionary support

## Questions?

Open an issue on GitHub!

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
