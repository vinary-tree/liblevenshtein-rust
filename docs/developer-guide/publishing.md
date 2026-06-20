# Publishing to crates.io

This document outlines the requirements and steps for publishing `liblevenshtein` to crates.io.

**Version:** 0.9.1
**Last Updated:** 2026-06-19

---

## Current Status

❌ **Cannot publish yet** - PathMap dependency must be published to crates.io first

## Blockers

### 1. PathMap Dependency (BLOCKER)

**Issue:** The `pathmap` dependency currently points to a Git repository instead of a crates.io version.

**Current configuration (Cargo.toml:16):**
```toml
pathmap = { git = "https://github.com/Adam-Vandervorst/PathMap.git", branch = "master" }
```

**Required configuration:**
```toml
pathmap = "x.y.z"  # Where x.y.z is a published crates.io version
```

**Resolution options:**

1. **Contact PathMap maintainer** (Adam Vandervorst)
   - Ask them to publish PathMap to crates.io
   - Wait for publication
   - Update dependency to use crates.io version

2. **Fork and publish PathMap**
   - Fork https://github.com/Adam-Vandervorst/PathMap
   - Publish under a different name (e.g., `pathmap-rs`, `pathmap-trie`)
   - Update dependency to use your published version
   - Consider contributing back to upstream

3. **Remove PathMap dependency** (NOT RECOMMENDED)
   - Major architectural change
   - Would require implementing alternative trie structure
   - Significant performance implications

**Recommended:** Option 1 (contact maintainer) or Option 2 (fork and publish)

---

## Requirements Checklist

### Required Metadata ✓

All required metadata is already present in `Cargo.toml`:

- ✓ `name` = "liblevenshtein"
- ✓ `version` = "0.3.0"
- ✓ `edition` = "2021"
- ✓ `authors` = ["Dylon Edwards <dylon.devo@gmail.com>"]
- ✓ `license` = "Apache-2.0"
- ✓ `description` = "Levenshtein/Universal Automata for approximate string matching using various dictionary backends"
- ✓ `repository` = "https://github.com/universal-automata/liblevenshtein-rust"
- ✓ `readme` = "README.md"
- ✓ `keywords` = ["levenshtein", "edit-distance", "spell-checking", "fuzzy-search", "automata"] (5/5 max)
- ✓ `categories` = ["algorithms", "data-structures", "text-processing"] (3/5 max)

### Optional Metadata (Recommended)

Consider adding these fields to `Cargo.toml`:

```toml
homepage = "https://github.com/universal-automata/liblevenshtein-rust"
documentation = "https://docs.rs/liblevenshtein"
```

### Required Files ✓

- ✓ `README.md` - Comprehensive documentation exists
- ✓ `LICENSE` - Apache-2.0 license file exists
- ✓ `CHANGELOG.md` - Detailed changelog exists

### All Dependencies from crates.io

Current status:
- ✓ `smallvec` = "1.13" (crates.io)
- ✓ `thiserror` = "2.0" (crates.io)
- ✓ `serde` = "1.0" (crates.io)
- ✓ `bincode` = "1.3" (crates.io)
- ✓ `serde_json` = "1.0" (crates.io)
- ✓ `prost` = "0.12" (crates.io)
- ✓ `bytes` = "1.5" (crates.io)
- ✓ `flate2` = "1.0" (crates.io)
- ✓ `clap` = "4.4" (crates.io)
- ✓ `anyhow` = "1.0" (crates.io)
- ✓ `rustyline` = "13.0" (crates.io)
- ✓ `colored` = "2.0" (crates.io)
- ✓ `dirs` = "5.0" (crates.io)
- ✓ `criterion` = "0.5" (crates.io)
- ✓ `tempfile` = "3.8" (crates.io)
- ✓ `prost-build` = "0.12" (crates.io)
- ❌ `pathmap` - Git dependency (BLOCKER)

---

## Pre-Publication Steps

### 1. Set up crates.io account

```bash
# 1. Create account at https://crates.io (if you don't have one)
# 2. Go to Account Settings → API Tokens
# 3. Generate a new API token
# 4. Log in with cargo:
cargo login <your-api-token>
```

Your API token will be stored in `~/.cargo/credentials.toml`

### 2. Resolve PathMap dependency

Choose one of the resolution options outlined in the Blockers section above.

**Example: If PathMap gets published as version 0.1.0:**

Edit `Cargo.toml` line 16:
```toml
# Before:
pathmap = { git = "https://github.com/Adam-Vandervorst/PathMap.git", branch = "master" }

# After:
pathmap = "0.1.0"
```

### 3. Verify package builds cleanly

```bash
# Build with all features
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo build --all-features

# Run all tests
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo test --all-features

# Run clippy
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo clippy --all-features --no-deps -- -D warnings

# Check formatting
cargo fmt -- --check
```

### 4. Update version if needed

The current version is `0.3.0`. If you need to bump the version:

1. Update `Cargo.toml` line 3: `version = "x.y.z"`
2. Update `packaging/arch/PKGBUILD` line 4: `pkgver=x.y.z`
3. Update `CHANGELOG.md` with new version section
4. Commit: `git commit -am "chore: Bump version to x.y.z"`
5. Tag: `git tag -a vx.y.z -m "Release vx.y.z"`
6. Push: `git push origin master && git push origin vx.y.z`

### 5. Clean up commented [patch] section (if uncommenting for local dev)

Ensure lines 122-127 in `Cargo.toml` are commented out:

```toml
# For local development: uncomment this section to use a local PathMap checkout
# instead of the GitHub repository. This allows you to test changes to PathMap
# without pushing to GitHub first.
#
# [patch.'https://github.com/Adam-Vandervorst/PathMap.git']
# pathmap = { path = "../PathMap" }
```

**Important:** The pre-commit git hook will prevent you from committing if this is uncommented.

---

## Publication Steps

### 1. Dry run (test package creation)

```bash
cargo publish --dry-run
```

This will:
- ✓ Verify all metadata is correct
- ✓ Check that all dependencies are from crates.io
- ✓ Build the package
- ✓ Create a `.crate` file (but not upload it)
- ✓ Show what files will be included

**Expected output:**
```
   Packaging liblevenshtein v0.3.0 (/path/to/liblevenshtein-rust)
   Verifying liblevenshtein v0.3.0 (/path/to/liblevenshtein-rust)
   Compiling liblevenshtein v0.3.0 (/path/to/liblevenshtein-rust/target/package/liblevenshtein-0.3.0)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
    Packaged X files, Y.Y MiB (Z.Z MiB compressed)
```

### 2. Review packaged files

The dry run creates a package in `target/package/liblevenshtein-0.3.0/`. Review it:

```bash
# List all files that will be published
tar -tzf target/package/liblevenshtein-0.3.0.crate

# Extract and review
cd target/package/liblevenshtein-0.3.0/
ls -la
```

**Files to verify:**
- ✓ All source files (`src/`)
- ✓ README.md
- ✓ LICENSE
- ✓ CHANGELOG.md
- ✓ Cargo.toml
- ✓ build.rs (if using protobuf feature)

**Files that should NOT be included:**
- ❌ `.git/` directory
- ❌ `target/` directory
- ❌ `.github/` workflows
- ❌ Local test files
- ❌ `.pkg.tar.zst`, `.deb`, `.rpm` packages

### 3. Publish to crates.io

```bash
cargo publish
```

**Important:** This is irreversible! Once published:
- ✓ Version numbers are permanent (cannot reuse)
- ✓ Can yank versions but cannot delete them
- ✓ Cannot modify already-published code

**Expected output:**
```
    Updating crates.io index
   Packaging liblevenshtein v0.3.0 (/path/to/liblevenshtein-rust)
   Verifying liblevenshtein v0.3.0 (/path/to/liblevenshtein-rust)
   Compiling liblevenshtein v0.3.0 (/path/to/liblevenshtein-rust/target/package/liblevenshtein-0.3.0)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
   Uploading liblevenshtein v0.3.0 (/path/to/liblevenshtein-rust)
```

### 4. Verify publication

1. **Check crates.io:** https://crates.io/crates/liblevenshtein
2. **Wait for docs to build:** https://docs.rs/liblevenshtein (can take 5-15 minutes)
3. **Test installation:**
   ```bash
   cargo install liblevenshtein --features cli,compression,protobuf
   liblevenshtein --version
   ```

---

## Post-Publication

### Update GitHub Release

Once published, update the GitHub release for v0.3.0 to mention crates.io:

```markdown
## Installation

### From crates.io

```bash
# Library only
cargo add liblevenshtein

# With CLI tool
cargo install liblevenshtein --features cli,compression,protobuf
```

### From GitHub Releases

Download pre-built binaries for your platform from the Assets below.
```

### Announce

Consider announcing the release:
- GitHub Discussions
- Rust community forums (users.rust-lang.org)
- Reddit (/r/rust)
- Twitter/X

---

## Troubleshooting

### Error: "git dependencies are not allowed"

**Problem:** PathMap dependency still points to Git repository

**Solution:** Update `Cargo.toml` line 16 to use crates.io version:
```toml
pathmap = "x.y.z"
```

### Error: "failed to verify package tarball"

**Problem:** Package doesn't build correctly when extracted

**Solution:**
1. Check that all necessary files are included (not in `.gitignore` or excluded)
2. Verify `build.rs` works correctly for protobuf feature
3. Run `cargo publish --dry-run` and review extracted package in `target/package/`

### Error: "crate name is already taken"

**Problem:** A crate named `liblevenshtein` already exists on crates.io

**Solution:**
1. Check https://crates.io/crates/liblevenshtein
2. If it's your crate, you need to bump the version
3. If it's someone else's, you need to:
   - Contact them to transfer ownership, OR
   - Choose a different name (e.g., `liblevenshtein-rs`, `levenshtein-automata`)

### Error: "authentication required"

**Problem:** Not logged in to crates.io

**Solution:**
```bash
cargo login <your-api-token>
```

### Error: "file size limit exceeded"

**Problem:** Package is too large (10 MB limit)

**Solution:**
1. Exclude large files using `.gitignore` or `exclude` in Cargo.toml
2. Remove large test files or benchmarks data
3. Consider splitting into multiple crates

---

## Package Size Optimization

If the package is large, you can exclude files:

Add to `Cargo.toml`:
```toml
[package]
# ... existing fields ...
exclude = [
    ".github/*",
    "docs/archive/*",
    "packaging/*",
    "*.sh",
    "target/*",
]
```

Or use `include` to be explicit about what to include:
```toml
[package]
# ... existing fields ...
include = [
    "src/**/*",
    "benches/**/*",
    "examples/**/*",
    "tests/**/*",
    "build.rs",
    "proto/**/*",
    "Cargo.toml",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
]
```

---

## Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR (x.0.0):** Breaking changes (incompatible API changes)
- **MINOR (0.x.0):** New features (backwards-compatible)
- **PATCH (0.0.x):** Bug fixes (backwards-compatible)

**Examples:**
- Adding a new public function: `0.3.0` → `0.4.0` (MINOR)
- Fixing a bug: `0.3.0` → `0.3.1` (PATCH)
- Changing function signature: `0.3.0` → `1.0.0` (MAJOR)

---

## References

- [The Cargo Book - Publishing on crates.io](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [crates.io Policies](https://crates.io/policies)
- [Semantic Versioning](https://semver.org/)
- [Package Metadata Reference](https://doc.rust-lang.org/cargo/reference/manifest.html)

---

## Quick Reference

**Dry run:**
```bash
cargo publish --dry-run
```

**Publish:**
```bash
cargo publish
```

**Yank a version (emergency only):**
```bash
cargo yank --vers 0.3.0
```

**Un-yank a version:**
```bash
cargo yank --vers 0.3.0 --undo
```

**Check what would be included:**
```bash
cargo package --list
```
