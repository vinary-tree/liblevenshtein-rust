# Building the library

`liblevenshtein` 0.10 is a library-only crate. The executable, REPL, and
document/archive grep stack live in
[`liblevenshtein-rust-cli`](https://github.com/vinary-tree/liblevenshtein-rust-cli).

Keep the development repositories as siblings because the manifests use local
paths while also declaring crates.io versions:

```text
workspace/
├── llattice/
├── libdictenstein/
├── liblevenshtein-rust/
└── liblevenshtein-rust-cli/
```

Common library checks:

```bash
cargo check
cargo test
cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo doc --all-features --no-deps
```

Features are additive library capabilities. `serialization`, `compression`,
and `protobuf` control dictionary persistence; `phonetic-rules` and
`parallel-grep` control reusable phonetic engines; `wasm` and `ffi`
control bindings. There is no `cli` or filesystem/document grep feature.

To build the command:

```bash
cd ../liblevenshtein-rust-cli
cargo build --release
./target/release/liblevenshtein --help
```
