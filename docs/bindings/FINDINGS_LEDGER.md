# Language-Binding Findings Ledger — liblevenshtein-rust

Scientific ledger for the ABI / language-binding scrutiny program (plan:
`thoroughly-test-and-document-deep-nova`). One entry per confirmed finding.
Sibling repos keep their own ledgers (`libdictenstein/docs/bindings/FINDINGS_LEDGER.md`,
`lling-llang/docs/scientific-ledger/bindings-findings-ledger.md`,
`duallity/docs/scientific-ledger/bindings-findings-ledger.md`); formal-verification
findings go to `docs/verification/FINDINGS_LEDGER.md`, not here.

Entry schema: `Finding <ID> | date | component | class {completeness, correctness,
performance, hygiene, version-pin} | severity | evidence | analysis | fix (commit
or "ledger-only") | verification | status`.

## Index

| ID | Component | Class | Severity | Status | Fix |
|---|---|---|---|---|---|
| LLEV-B1 | `src/ffi/mod.rs` doc example | correctness (docs) | medium | OPEN → W3 | — |
| LLEV-B2 | `.gitignore` / staged natives | hygiene | high | FIXED | `de5caee` |
| LLEV-B3 | root + macros `Cargo.lock` | hygiene | medium | FIXED | `7475484` |
| LLEV-B4 | `bindings/javascript-runtime/rust/{wasi,browser}.rs` | correctness | high | OPEN → W3 | — |
| LLEV-B5 | `src/ffi/{index,phonetic}.rs` cfg warnings | hygiene | low | FIXED | `7475484` |
| LLEV-B6 | FFI `VtStatus` discriminant reads | correctness (UB) | high | OPEN → W3 | — |
| LLEV-B7 | `VtOptionalU64.reserved` validation (F2) | correctness | medium | OPEN → W3 | — |
| LLEV-B8 | paging-acceptance asymmetry (F3, llev side) | correctness | medium | OPEN → W3 | — |
| LLEV-B9 | family version pins | version-pin | info | LEDGER-ONLY | ledger-only |
| LLEV-B10 | `vinary-tree-interop` `rust-version` | hygiene | low | FIXED | `d895183` |
| LLEV-B11 | swift facade phonetic/threshold surface | completeness | medium | OPEN → W7 | — |
| LLEV-B12 | javascript-runtime Node N-API threshold + size/len symbols | correctness | high | OPEN → W3 | — |
| LLEV-B13 | dotnet threshold overload asymmetry | completeness | low | OPEN → W7 | — |
| LLEV-B14 | python `pattern_size`/`rules_len` | completeness | low | OPEN → W7 | — |
| LLEV-B15 | `docs/diagrams` .dot/.asy render drift | hygiene | low | OPEN → W3 | — |

## Findings

### LLEV-B1 — stale `llev_index_*` C example in `src/ffi/mod.rs`
- **Date**: 2026-08-08 · **Component**: `src/ffi/mod.rs:34-50` · **Class**: correctness (documentation) · **Severity**: medium
- **Evidence**: `rg -n 'llev_index_' src/ffi/mod.rs` → module-level C example demonstrates `LlevIndex`/`llev_index_new`/`llev_index_insert`/`llev_index_query`; `rg -n 'llev_index_' include/ src/ffi/*.rs bindings/api.json` → the symbols exist nowhere else (retired by the dictionary-ownership migration; `scripts/check-bindings.py` even forbids `llev_index_` in publishable facades but does not scan this doc comment).
- **Analysis**: the first C example a binding author reads demonstrates an API that no longer exists; it predates the resource-ABI migration.
- **Fix**: scheduled wave W3 (rewrite to the `llev_transducer_new` → cursor/lease flow, kept in sync with `docs/bindings/c-abi-reference.md`).
- **Verification (planned)**: rewritten example compiled in the C CI lane; `check-bindings.py` doc-comment scan extended to forbid retired symbols here.
- **Status**: OPEN → W3.

### LLEV-B2 — 32 MB staged native libraries and Lein scratch not ignored
- **Date**: 2026-08-08 · **Component**: `.gitignore`, `bindings/dotnet/native/runtimes/`, `bindings/ruby/lib/vinary_tree/liblevenshtein/native/`, `bindings/clojure/.lein-failures` · **Class**: hygiene · **Severity**: high (repo pollution hazard)
- **Evidence**: `du -sh` → two 32 MB `libliblevenshtein.so` staging artifacts; `git check-ignore` (before fix) matched none of the three paths; a blanket `git add bindings/` would have committed 64 MB of ELF objects.
- **Analysis**: the packaging scripts stage prebuilt natives into facade trees; the staging destinations were never added to `.gitignore`.
- **Fix**: commit `de5caee` — directory-level ignores for both staging destinations + `**/.lein-failures`; verified `bindings/ruby/.../native.rb` (source) remains tracked.
- **Verification**: `git check-ignore -v` matches all three paths; baseline commit `7475484` staged zero `.so` files (checked via `git diff --cached --numstat` binary listing).
- **Status**: FIXED.

### LLEV-B3 — whitelisted lockfiles present but never added
- **Date**: 2026-08-08 · **Component**: `Cargo.lock`, `liblevenshtein-macros/Cargo.lock` · **Class**: hygiene · **Severity**: medium (CI reproducibility)
- **Evidence**: `.gitignore:32-36` explicitly whitelists exactly these two lockfiles (`!/Cargo.lock`, `!/liblevenshtein-macros/Cargo.lock`) "used by CI", yet `git ls-files` showed neither tracked.
- **Fix**: staged and committed in baseline `7475484`.
- **Verification**: `git ls-files Cargo.lock liblevenshtein-macros/Cargo.lock` lists both.
- **Status**: FIXED.

### LLEV-B4 — panic discipline gap at the WASM boundary
- **Date**: 2026-08-08 · **Component**: `bindings/javascript-runtime/rust/src/wasi.rs` (~39 `lock().unwrap()`-class sites), `bindings/javascript-runtime/rust/src/browser.rs` (6 sites: `.expect` :33, `unreachable!` :155, vtable-fn `unwrap`s :393/:408/:426, `.unwrap` :784) · **Class**: correctness · **Severity**: high
- **Evidence**: `rg -n 'unwrap\(\)|expect\(|unreachable!' bindings/javascript-runtime/rust/src/*.rs` (non-test code).
- **Analysis**: every native `llev_*`/`ldict_*` entry point routes through `catch_unwind` and returns a status; the WASM runtime crate instead panics, which traps/aborts the instance — the one surface without the family's panic-containment discipline. A poisoned registry mutex or a provider misuse should surface as a status/JsError, not kill the instance.
- **Fix**: scheduled wave W3 — map to status codes/`JsError`, add no-panic regression tests, and add a `check-bindings.py` grep gate over non-test `wasi.rs`/`browser.rs` for `unwrap!/expect(/unreachable!/panic!`.
- **Status**: OPEN → W3.

### LLEV-B5 — cfg-dependent warnings in the FFI modules
- **Date**: 2026-08-08 · **Component**: `src/ffi/index.rs` (unused `LLEV_BUILD_FEATURE_PHONETIC` import + `unused_mut` in `llev_build_features`), `src/ffi/phonetic.rs` (3 imports unused without `bindings-phonetic`) · **Class**: hygiene · **Severity**: low
- **Evidence**: `cargo check --features ffi` (pre-fix) → 4 warnings; `--features native-bindings-full` → 0 (imports used only under the phonetic cfg).
- **Fix**: baseline `7475484` — `llev_build_features` now composes via `cfg!(feature = "bindings-phonetic")` (both constants referenced in every configuration); phonetic imports split under `#[cfg(feature = "bindings-phonetic")]`.
- **Verification**: `cargo check --features ffi` and `--features native-bindings-full` both emit 0 warnings.
- **Status**: FIXED.

### LLEV-B6 — out-of-range provider status is instant UB
- **Date**: 2026-08-08 · **Component**: FFI callback returns read as `#[repr(u32)]` `VtStatus` · **Class**: correctness (undefined behavior) · **Severity**: high
- **Evidence**: `vinary-tree-interop/src/lib.rs` defines `VtStatus` as a fieldless `#[repr(u32)]` enum; consumer callback sites receive it by value from arbitrary foreign providers. A provider returning e.g. `42` makes the Rust-side value an invalid enum discriminant — UB before any check can run.
- **Analysis**: the ABI type crossing the trust boundary must be received as raw `u32` and validated before conversion (mirrors how `LlevStatus::try_from(u32)` already works in `src/ffi/generated.rs`).
- **Fix**: scheduled wave W3 — receive `u32` at every callback boundary, validate, convert (`TryFrom`), map failures to `ProviderError`; adversarial fault-provider test follows the fix (the pre-fix behavior is untestable — it is UB).
- **Status**: OPEN → W3.

### LLEV-B7 — `VtOptionalU64.reserved` accepted unvalidated (pre-registered F2)
- **Date**: 2026-08-08 · **Component**: `src/bindings.rs` value decoding · **Class**: correctness · **Severity**: medium
- **Evidence**: lling-llang's consumer validates `VtWfstArc.reserved == 0`; llev's `VtOptionalU64` path checks only `has_value ∈ {0,1}` — asymmetric application of the interop "reserved fields must be zero" law (`vinary-tree-interop/src/lib.rs` doc).
- **Fix**: scheduled wave W3 under invariant VT-ABI-5, with the fault-provider test.
- **Status**: OPEN → W3.

### LLEV-B8 — paging-acceptance asymmetry across consumers (pre-registered F3, llev side)
- **Date**: 2026-08-08 · **Component**: `src/bindings.rs` `expanded_edges` acceptance checks · **Class**: correctness · **Severity**: medium
- **Evidence**: llev checks `total < start + written`; lling-llang adds `offset > total` and an in-loop progress check; duallity uses saturating adds with a slightly different predicate — three subtly different acceptance predicates for one interop paging law.
- **Fix**: scheduled wave W3 — harmonize all three consumers to the single proven predicate from `docs/verification/abi/theories/ConsumerAcceptance.v` (each sibling patches its own copy in its own wave).
- **Status**: OPEN → W3 (llev), W4 (lling), W5 (duallity).

### LLEV-B9 — family version-pin inconsistencies (ledger-only per user decision)
- **Date**: 2026-08-08 · **Component**: `bindings/related-projects.json`, `.github/workflows/release.yml`, crates.io state · **Class**: version-pin · **Severity**: informational
- **Evidence**: crates.io `liblevenshtein` = 0.9.1 vs local 0.10.0; `vinary-tree-interop` 0.1.0 unpublished (crates.io index NoSuchKey) while sibling workflows pin it; `related-projects.json` + `release.yml` pin `duallity v0.3.0` — `git -C ../duallity tag` ends at v0.2.0; `lling-llang v0.2.0` tag predates its binding tree (committed 2026-08-08).
- **Analysis**: release execution is out of scope for this program; the pins document the *intended* release chain (interop → libdictenstein → liblevenshtein → lling-llang → duallity).
- **Fix**: ledger-only.
- **Status**: LEDGER-ONLY.

### LLEV-B10 — interop crate `rust-version` predates `offset_of!`
- **Date**: 2026-08-08 · **Component**: `vinary-tree-interop/Cargo.toml` · **Class**: hygiene · **Severity**: low
- **Evidence**: `rust-version = "1.70"`; `core::mem::offset_of!` stabilized in 1.77; the W1 layout-contract tests use it; the family MSRV gate is 1.95.
- **Fix**: commit `d895183` — bumped to 1.95 alongside the layout tests (crate unpublished, so no compatibility impact).
- **Verification**: `cargo test --locked -p vinary-tree-interop` green (27 tests) under the workspace toolchain; MSRV leg covers the workspace with `--features ffi`.
- **Status**: FIXED.

### LLEV-B11 — swift facade lacks the phonetic rule-set and threshold surface
- **Date**: 2026-08-08 · **Component**: `bindings/swift/liblevenshtein/Sources/Liblevenshtein/Liblevenshtein.swift` · **Class**: completeness · **Severity**: medium
- **Evidence**: `bindings/conformance/completeness-matrix.tsv` (swift 17/35, 10 FINDING nulls): no `PhoneticRuleSet` facade (`llev_phonetic_rules_parse/builtin/free/len/apply`, `llev_owned_string_free` unbound), no `llev_phonetic_pattern_size`, and all three `llev_*_distance_threshold` functions missing — every other Tier-2/3 facade exposes these.
- **Fix**: scheduled wave W7 (uniform per-language completeness) — add the missing Swift wrappers + tests.
- **Status**: OPEN → W7.

### LLEV-B12 — Node N-API runtime omits symbols its type declarations promise
- **Date**: 2026-08-08 · **Component**: `bindings/javascript-runtime/native/src/addon.cc`, `native.mjs`/`native.cjs` vs `index.d.ts` · **Class**: correctness · **Severity**: high
- **Evidence**: `index.d.ts` declares `levenshteinDistanceThreshold`/`damerauDistanceThreshold`/`trueDamerauDistanceThreshold`; the browser-WASM path implements them (`rust/src/browser.rs` via `runtime-factory.mjs`) but the default Node N-API path binds no `llev_*_threshold` — the typed members are `undefined` at runtime on Node. `pattern_size`/`rules_len` are likewise unbound in `addon.cc`, which also blocks the project JS facade (its two FINDING nulls share this root cause).
- **Analysis**: a TypeScript consumer compiles clean against members that do not exist on the Node default path — a runtime `TypeError` the type system promised away.
- **Fix**: scheduled wave W3 (umbrella-runtime work): bind the five missing symbols in `addon.cc`, surface them through `native.mjs`/`native.cjs`, and add contract tests asserting runtime presence of every `index.d.ts` member on every runtime path.
- **Status**: OPEN → W3.

### LLEV-B13 — dotnet threshold overloads only cover standard Levenshtein
- **Date**: 2026-08-08 · **Component**: `bindings/dotnet/src/VinaryTree.Liblevenshtein/Distance.cs` · **Class**: completeness · **Severity**: low
- **Evidence**: completeness matrix (dotnet 25/35): `llev_damerau_distance_threshold` and `llev_true_damerau_distance_threshold` unbound while `llev_distance_threshold` has an overload — asymmetric.
- **Fix**: scheduled wave W7.
- **Status**: OPEN → W7.

### LLEV-B14 — python facade misses `pattern_size`/`rules_len`
- **Date**: 2026-08-08 · **Component**: `bindings/python/src/liblevenshtein/_native.py` · **Class**: completeness · **Severity**: low
- **Evidence**: completeness matrix (python 20/35 with 2 FINDING nulls): `llev_phonetic_pattern_size` and `llev_phonetic_rules_len` unbound while the peer Tier-1 JVM facade and all Tier-2/3 facades expose them.
- **Fix**: scheduled wave W7.
- **Status**: OPEN → W7.

### LLEV-B15 — pre-existing diagram render drift outside the bindings suite
- **Date**: 2026-08-08 · **Component**: `docs/diagrams` (21 `.dot`/`.asy` sources) · **Class**: hygiene · **Severity**: low
- **Evidence**: `bash docs/diagrams/render.sh --check` reports 21 drifted renders, all Graphviz/Asymptote (renderer-version skew between the committed SVGs and graphviz 15.1.1 / current asy); present at baseline `ae0d6a5`; zero under `docs/diagrams/bindings/` (the new suite renders byte-stable PlantUML).
- **Fix**: scheduled wave W3 (re-render the drifted sources with the current toolchain in a dedicated commit so `render.sh --check` returns to a meaningful zero).
- **Status**: OPEN → W3.
