# Publishing

`liblevenshtein` and `liblevenshtein-cli` use lockstep minor versions,
starting with 0.10.0, but are published from separate repositories.

Release order:

1. Publish and tag `libdictenstein` if its required version changed.
2. Publish `liblevenshtein` to crates.io and create its library-artifact
   release (`rlib` and `cdylib` archives).
3. Publish `liblevenshtein-cli` to crates.io only after the matching
   `liblevenshtein` version is available.
4. Create the CLI GitHub release with executable archives and Debian, RPM, and
   Arch Linux packages.

Before tagging the library:

```bash
cargo test --all-features
cargo package
cargo publish --dry-run
```

The release workflow removes development-only sibling `path` keys while
retaining version requirements. Never publish the executable from this
repository; OS packages and binary archives belong to the CLI repository.
