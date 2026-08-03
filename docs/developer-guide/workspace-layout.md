# Cargo workspace layout

The repository contains two independently verified Cargo workspaces:

| Workspace | Manifest | Purpose |
|---|---|---|
| Library and binaries | `Cargo.toml` | The `liblevenshtein` package, examples, tests, and benchmarks |
| Procedural macros | `liblevenshtein-macros/Cargo.toml` | Compile-time phonetic-rule macros and their documentation tests |

## Why the macro crate is independent

The library emits both an `rlib` and a `cdylib`. The macro package also depends
on the library so it can reuse the phonetic parser. If both packages are members
of one Cargo workspace, `cargo test --workspace` builds two root-package units
that try to emit the same fixed-name dynamic-library artifact. Cargo reports a
filename collision even though the dependency graph itself is valid.

The root manifest therefore has `members = ["."]` and explicitly excludes
`liblevenshtein-macros`. The macro manifest declares its own workspace. This is
an artifact-ownership boundary, not a source-code boundary: the macro still uses
the local library path dependency.

## Required verification

Run both workspaces before merging changes that affect parsing, phonetic rules,
public macro syntax, or Cargo metadata:

```bash
cargo test --workspace --all-features
cargo test --manifest-path liblevenshtein-macros/Cargo.toml --all-features
```

The CI matrix runs the second command explicitly. Do not add the macro package
back to the root workspace merely to make `--workspace` discover it; that
reintroduces the dynamic-library collision. New sibling packages should be
assigned to exactly one workspace and documented here.

[← Developer Guide](README.md)
