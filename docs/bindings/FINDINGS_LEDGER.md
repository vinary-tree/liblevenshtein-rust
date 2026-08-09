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
| LLEV-B1 | `src/ffi/mod.rs` doc example | correctness (docs) | medium | FIXED | `8c3c654` |
| LLEV-B2 | `.gitignore` / staged natives | hygiene | high | FIXED | `de5caee` |
| LLEV-B3 | root + macros `Cargo.lock` | hygiene | medium | FIXED | `7475484` |
| LLEV-B4 | `bindings/javascript-runtime/rust/{wasi,browser}.rs` | correctness | high | FIXED | `457d745` |
| LLEV-B5 | `src/ffi/{index,phonetic}.rs` cfg warnings | hygiene | low | FIXED | `7475484` |
| LLEV-B6 | FFI `VtStatus` discriminant reads | correctness (UB) | high | FIXED | `e42485c` (+family) |
| LLEV-B7 | `VtOptionalU64.reserved` validation (F2) | correctness | medium | FIXED | `8c3c654` |
| LLEV-B8 | paging-acceptance asymmetry (F3, llev side) | correctness | medium | FIXED | `65eb4a2` |
| LLEV-B9 | family version pins | version-pin | info | LEDGER-ONLY | ledger-only |
| LLEV-B10 | `vinary-tree-interop` `rust-version` | hygiene | low | FIXED | `d895183` |
| LLEV-B11 | swift facade phonetic/threshold surface | completeness | medium | FIXED | `95b4cdb` |
| LLEV-B12 | javascript-runtime Node N-API threshold + size/len symbols | correctness | high | FIXED | `622e4f6` |
| LLEV-B13 | dotnet threshold overload asymmetry | completeness | low | FIXED | `d0481d9` |
| LLEV-B14 | python `pattern_size`/`rules_len` | completeness | low | FIXED | `26e14ef` |
| LLEV-B15 | `docs/diagrams` .dot/.asy render drift | hygiene | low | OPEN → W3 | — |
| LLEV-B16 | FFI reducer/callback status wire (raw `u32`) | correctness (UB) | medium | FIXED | `dad4429` |
| LLEV-B17 | cursor fault window discards the in-flight batch | correctness (completeness-under-fault) | low | LEDGER + DOCS | ledger-only |
| LLEV-B18 | `llev_distance*` reject `(NULL, 0)` empty operands | correctness | low | FIXED | (this commit) |

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
- **Date**: 2026-08-08 · **Component**: `bindings/javascript-runtime/rust/src/wasi.rs` (36 sites: 33 `lock().unwrap()` + 3 vtable-fn `unwrap`s), `bindings/javascript-runtime/rust/src/browser.rs` (6 sites: `.expect` :33, `unreachable!` :155, vtable-fn `unwrap`s :393/:408/:426, `.unwrap` :784) · **Class**: correctness · **Severity**: high
- **Evidence**: `rg -n 'unwrap\(\)|expect\(|unreachable!' bindings/javascript-runtime/rust/src/*.rs` (non-test code). During the fix a second latent defect surfaced: neither module compiled for its wasm32 target after the `e42485c` raw-u32 wire rule (four `status != VtStatus::Ok` comparisons per file against now-`u32` callback returns); native `cargo check` skips both cfg-gated modules, which is how the break went unseen.
- **Analysis**: every native `llev_*`/`ldict_*` entry point routes through `catch_unwind` and returns a status; the WASM runtime crate instead panicked, which traps/aborts the instance — the one surface without the family's panic-containment discipline. A poisoned registry mutex or a provider misuse should surface as a status/JsError, not kill the instance.
- **Fix**: commit `457d745`, zero panic paths remain (no justified residuals were needed):
  - *wasi.rs mutex poisoning (33 sites)*: recover via `unwrap_or_else(PoisonError::into_inner)` behind one `locked_registry()` chokepoint. Per-site justification (identical for all 33 because they guard one object): the registry is a plain handle table — `next: u32` counter, `HashMap<u32, Handle>`, error `Vec<u8>` — with no invariant spanning a critical section; every operation leaves it structurally valid mid-unwind, so observing post-poison state is sound.
  - *wasi.rs vtable-fn `unwrap`s (3 sites: `start`/`state_info`/`state_arcs`)*: `Option::ok_or` into the existing `FAILURE` (`u32::MAX`) + `vt_error_pointer` message convention.
  - *wasi.rs hardening*: null-output-pointer guards on `vt_wfst_start`/`vt_dictionary_get_text` (previously insta-UB writes on a null pointer, now a status); raw-status decode via `VtStatus::from_raw` with out-of-range mapped to an explicit provider error (also the compile fix).
  - *browser.rs `.expect` :33*: `property()` now returns `Result<(), JsError>` and every caller propagates (`lookup`/`match_value`/`state`/`next` re-typed to `Result`); a `false` `Reflect::set` is reported, never swallowed.
  - *browser.rs `unreachable!` :155*: the guarded U64 match arm returns the guard's own `JsError` instead.
  - *browser.rs vtable-fn `unwrap`s :393/:408/:426*: `ok_or_else` `JsError`s; raw statuses decoded via a `require_ok(from_raw)` helper mirroring the wasi side.
  - *browser.rs `.unwrap` :784*: cursor `Option` restructured into let-else; the adjacent batch index read also moved from panicking indexing to a checked `get`.
  - *Gate*: `scripts/check-bindings.py` now scans non-test lines of exactly these two files for `unwrap()`/`expect(`/`unreachable!`/`panic!`/`todo!`/`unimplemented!`, stripping string/char literals and comments and tracking `#[cfg(test)]` item brace depth.
- **Verification**: `cargo check -p vinary-tree-js-runtime` on native, `wasm32-unknown-unknown`, and `--no-default-features --features wasi` on `wasm32-wasip1` (the wasm targets failed to compile before the fix); full runtime conformance green post-fix on all three paths (`node --test` native 7/7, browser-wasm 4/4, wasi 3/3 — no functional regressions); gate verified green on the fixed files, red on a transient production `unwrap()` mutant, green on a transient `#[cfg(test)]`-scoped mutant. Reviewed residual implicit-panic surface: the `put_u32`/`put_u64`/`put_f64` slice writes index buffers resized from the same lengths the loops iterate (in-bounds by construction), and allocation failure aborts rather than unwinds on wasm (outside status-reporting reach by design).
- **Status**: FIXED.

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
- **Fix**: wave W3, family-wide while the interop crate is unpublished — all 13 interop callback signatures now return raw `u32` on the Rust side; `VtStatus` gained const `to_raw`/`from_raw` and the crate documents the wire rule. Consumers decode at their chokepoints (llev `status()` → `InvalidProviderOutput` on out-of-range; lling `check_status()` likewise; duallity `status()` + its three adapter fault sites); producers encode via `.to_raw()` shims over typed inner fns. Commits: llev/interop `e42485c`, libdictenstein `1aea856`, lling-llang `0a98513`, duallity `e84b2ac`. The C header is unchanged (C enums are integer-typed; byte-identical ABI).
- **Verification**: interop `wire_round_trip_covers_exactly_the_published_range` (0..=8 bijective, everything else refuses); full binding/ffi suites green in all four repos post-conversion; the W3 fault-provider test exercises an out-of-range status end-to-end (now a VALUE, so it is testable).
- **Status**: FIXED.

### LLEV-B7 — `VtOptionalU64.reserved` accepted unvalidated (pre-registered F2)
- **Date**: 2026-08-08 · **Component**: `src/bindings.rs` value decoding · **Class**: correctness · **Severity**: medium
- **Evidence**: lling-llang's consumer validates `VtWfstArc.reserved == 0`; llev's `VtOptionalU64` path checks only `has_value ∈ {0,1}` — asymmetric application of the interop "reserved fields must be zero" law (`vinary-tree-interop/src/lib.rs` doc).
- **Fix**: commit `8c3c654` — the value-decode path now rejects `value.reserved != [0; 7]` (`src/bindings.rs:591`) and the resource-value base rejects `base.reserved != 0` (`src/bindings.rs:317`), both mapping to `InvalidProviderOutput("reserved bytes were not zero")`, under invariant VT-ABI-5.
- **Verification**: `tests/ffi_provider_fault_injection.rs` pins a nonzero-`reserved` reply to `ProviderError` with message "reserved bytes were not zero".
- **Status**: FIXED.

### LLEV-B8 — paging-acceptance asymmetry across consumers (pre-registered F3, llev side)
- **Date**: 2026-08-08 · **Component**: `src/bindings.rs` `expanded_edges` acceptance checks · **Class**: correctness · **Severity**: medium
- **Evidence**: llev checks `total < start + written`; lling-llang adds `offset > total` and an in-loop progress check; duallity uses saturating adds with a slightly different predicate — three subtly different acceptance predicates for one interop paging law.
- **Fix**: commit `65eb4a2` (llev side) — `expanded_edges` (`src/bindings.rs:428`) now runs the single `accepts_dec` predicate proved in `docs/verification/abi/theories/ConsumerAcceptance.v`: `page_len == written`, `written <= capacity`, `written <= total.saturating_sub(start)`, a progress conjunct, and `total <= max_total` realized *structurally* (the consumer never sizes an allocation from the provider-claimed `total`, closing the preallocation-abort vector LLEV-B8/finding LLEV-B8). lling and duallity patch their own copies in W4/W5.
- **Verification**: `tests/abi_paging_correspondence.rs::adversarial_pages_are_rejected_without_aborting` (Overfill / PastEnd / StalledProgress / InflatedTotal each surface as a provider error with no allocation abort) and `honest_paging_is_accepted_and_lossless` (the same predicate certified by the Coq proof).
- **Status**: FIXED (llev); lling → W4, duallity → W5.

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
- **Fix**: commit `95b4cdb` — `Liblevenshtein.swift` gains the three `EditDistance` threshold overloads (`levenshtein`/`damerauOSA`/`damerauLevenshtein`, each `-> Int?` with `nil` for the native `usize::MAX`/`usize::MAX-1` sentinels, via a shared `bounded` helper), `PhoneticPattern.size() -> (states, transitions)`, and a full `PhoneticRuleSet` class (`parse`/`builtin`/`count`/`apply`/`close`) with a `PhoneticRuleSetKind` enum; `apply` copies the `LlevOwnedString` then releases it with `llev_owned_string_free`. Swift rises 17/35 → 27/35 and its enum count 2/4 → 3/4, matching the peer Tier-2/3 reasoned-absence set exactly.
- **Verification**: the SwiftPM Integration executable pins the threshold within/exceeded results (`ca→abc` separating OSA 3 from unrestricted Damerau 2), the pattern size (positive states/transitions), the built-in english-orthography set, and a parsed 2-rule set whose `apply("phgh") == "f"` (DSL verified against the native library); `swift run` green end-to-end with both native libs linked. `python3 scripts/check-bindings.py --check` now reports **0 findings** across all 15 facades.
- **Status**: FIXED.

### LLEV-B12 — Node N-API runtime omits symbols its type declarations promise
- **Date**: 2026-08-08 · **Component**: `bindings/javascript-runtime/native/src/addon.cc`, `native.mjs`/`native.cjs` vs `index.d.ts` · **Class**: correctness · **Severity**: high
- **Evidence**: `index.d.ts` declares `levenshteinDistanceThreshold`/`damerauDistanceThreshold`/`trueDamerauDistanceThreshold`; the browser-WASM path implements them (`rust/src/browser.rs` via `runtime-factory.mjs`) but the default Node N-API path binds no `llev_*_threshold` — the typed members are `undefined` at runtime on Node. `pattern_size`/`rules_len` are likewise unbound in `addon.cc`, which also blocks the project JS facade (its two FINDING nulls share this root cause).
- **Analysis**: a TypeScript consumer compiles clean against members that do not exist on the Node default path — a runtime `TypeError` the type system promised away.
- **Fix**: commit `622e4f6` — `addon.cc` binds all five (`levenshteinDistanceThreshold`/`damerauDistanceThreshold`/`trueDamerauDistanceThreshold` decode the C sentinels — `SIZE_MAX` invalid-UTF-8 → thrown `Error`, `SIZE_MAX-1` exceeded → `undefined` — matching the browser-wasm `Option` mapping exactly; `patternSize` → `{states, transitions}`; `rulesLen` → enabled-rule count); `native.mjs`/`native.cjs` re-export the thresholds beside their siblings and add `PhoneticPattern.size`/`PhoneticRuleSet.size` getters; `browser.rs` gains the same two getters so the shared `index.d.ts` additions (`AutomatonSize` + the two `size` members) hold on the wasm path; the project facade `bindings/javascript/index.d.ts` declares the two members its pass-through instances now carry; `api-surface-map.json` converts the 5 runtime + 2 project-facade FINDING nulls to real symbols (findings 21 → 14, all remaining scheduled W7) and `completeness-matrix.tsv` is regenerated.
- **Verification**: `npm run build:native` (-Werror clean) + `node --test test/native.test.mjs` 7/7, including three new suites: threshold early-exit semantics pinned to the unthresholded functions (with `ca→abc` separating OSA 3 from unrestricted Damerau-Levenshtein 2), size behavior (exact 2-rule count for `ph -> f; gh -> ;`, closed-handle throws), and the structural regression — a nesting-aware `index.d.ts` scan asserting every declared method is `typeof === "function"` and every declared property exists on live native instances (proved red under a transient mutant deleting one threshold re-export). CJS path spot-checked for all five; browser-wasm path returns identical values for all five members; `python3 scripts/check-bindings.py --check` green over the regenerated matrix.
- **Status**: FIXED.

### LLEV-B13 — dotnet threshold overloads only cover standard Levenshtein
- **Date**: 2026-08-08 · **Component**: `bindings/dotnet/src/VinaryTree.Liblevenshtein/Distance.cs` · **Class**: completeness · **Severity**: low
- **Evidence**: completeness matrix (dotnet 25/35): `llev_damerau_distance_threshold` and `llev_true_damerau_distance_threshold` unbound while `llev_distance_threshold` has an overload — asymmetric.
- **Fix**: commit `d0481d9` — `Native.cs` binds both missing symbols and `Distance` gains `Damerau`/`TrueDamerau` threshold overloads through a shared `ThresholdCall` helper; the overloads pass the native result through, so the exceeded-bound sentinel is `nuint.MaxValue - 1` (not `threshold + 1` — the XML docs are corrected to say so). dotnet rises 25/35 → 27/35.
- **Verification**: `dotnet run --framework net10.0` green (net8.0 covered by CI) with the conformance program pinning within-bound and exceeded-bound results for all three overloads and using `ca→abc` (OSA 3 vs unrestricted Damerau 2) to separate the true-Damerau overload from OSA.
- **Status**: FIXED.

### LLEV-B14 — python facade misses `pattern_size`/`rules_len`
- **Date**: 2026-08-08 · **Component**: `bindings/python/src/liblevenshtein/_native.py` · **Class**: completeness · **Severity**: low
- **Evidence**: completeness matrix (python 20/35 with 2 FINDING nulls): `llev_phonetic_pattern_size` and `llev_phonetic_rules_len` unbound while the peer Tier-1 JVM facade and all Tier-2/3 facades expose them.
- **Fix**: commit `26e14ef` — `_native.py` binds both `ctypes` prototypes; `PhoneticPattern.size` returns a `(states, transitions)` tuple and `PhoneticRuleSet.__len__` surfaces the enabled-rule count through the Pythonic `len()` protocol. python rises 20/35 → 22/35.
- **Verification**: `pytest -q bindings/python/tests` 5/5 (two new tests pin the size-tuple shape with positive states/transitions and `len()` over the built-in english-orthography set plus an `apply()` round-trip).
- **Status**: FIXED.

### LLEV-B15 — pre-existing diagram render drift outside the bindings suite
- **Date**: 2026-08-08 · **Component**: `docs/diagrams` (21 `.dot`/`.asy` sources) · **Class**: hygiene · **Severity**: low
- **Evidence**: `bash docs/diagrams/render.sh --check` reports 21 drifted renders, all Graphviz/Asymptote (renderer-version skew between the committed SVGs and graphviz 15.1.1 / current asy); present at baseline `ae0d6a5`; zero under `docs/diagrams/bindings/` (the new suite renders byte-stable PlantUML).
- **Fix**: scheduled wave W3 (re-render the drifted sources with the current toolchain in a dedicated commit so `render.sh --check` returns to a meaningful zero).
- **Status**: OPEN → W3.

### LLEV-B16 — FFI reducer/callback status wire read as a typed enum (twin of B6)
- **Date**: 2026-08-08 · **Component**: `src/ffi/index.rs` batch reducer + interop callback returns · **Class**: correctness (UB) · **Severity**: medium
- **Evidence**: the batch reducer callback and several interop callback returns moved the status across the wire as a typed `#[repr(u32)]` enum; a foreign reducer (or callback) returning an out-of-range discriminant is read directly into the enum — instant undefined behavior, the same class as LLEV-B6 on the provider-callback path.
- **Analysis**: the ABI status wire must be a raw `u32` validated *before* any enum conversion; a typed return type silently assumes the foreign side only ever produces in-range discriminants.
- **Fix**: commit `dad4429` — the reducer callback wire is a raw `u32` decoded via `LlevStatus::try_from` (out-of-range → `InvalidArgument`, "batch reducer returned an out-of-range status"); interop callback returns fold through `VtStatus::to_raw()` shims over typed `_status` inner fns; the commit also corrected the cross-read findings noted against B6/B8.
- **Verification**: `tests/ffi_reducer_laws.rs` (all 11 valid non-Ok/End statuses returned verbatim; raw 13 / 42 / `u32::MAX` → `InvalidArgument`) and the reducer-wire encode in `tests/ffi_resource_snapshot_semantics.rs`; registered as invariant family LLEV-STAT-6 (`ABI_INVARIANTS.tsv`) for the End-from-callback fold.
- **Status**: FIXED.

### LLEV-B17 — the cursor fault window discards the in-flight batch
- **Date**: 2026-08-09 · **Component**: `src/bindings.rs` `QueryCursor` fault channel / `src/ffi/index.rs` batch fill · **Class**: correctness (completeness-under-fault) · **Severity**: low
- **Evidence**: surfaced by the W3 T2 consumer suite. When a provider callback faults partway through assembling a batch, the consumer discards the *whole* in-progress batch and surfaces the fault; the fault channel is take-once, so the next advance resumes the cursor **empty**. A fault while producing match `b` therefore also costs the same poisoned pull's already-visited `c` (pinned by `provider_fault_is_taken_once_then_the_cursor_resumes`), and a fault mid-batch at the safe layer drops that batch's already-gathered prefix.
- **Analysis**: the behavior is deterministic and memory-safe, and it does **not** violate the proven consumer laws. `CursorSnapshotSemantics.v` `emitted_subset_captured` (VT-SNAP-1) and `latched_fault_surfaces_next` (VT-SNAP-3) require that every emitted match is a member of the captured revision and that no match is fabricated *past* a fault; neither requires delivery of matches gathered *before* the fault. Treating a provider fault as terminal for the in-progress batch — rather than skipping the faulting node and continuing — is the safe, simple contract: it never fabricates, never double-delivers, and always surfaces the error exactly once. Delivering the gathered prefix and then re-surfacing the fault on a later call would require stashing a pending fault and is a semantics change beyond this program's pre-registered scope (F1–F5), so it is intentionally not taken.
- **Fix**: ledger + docs. Recorded as a **behavioral contract**: *a provider fault is terminal for the batch it interrupts; matches gathered during the faulting pull (including any the traversal advanced past but had not yet emitted) are not delivered, and the fault surfaces exactly once (take-once) before the cursor resumes empty.* No code change. A consumer that must not lose matches under provider faults should either pull with capacity 1 (so at most one match is ever in flight) or treat any `Provider` error as "traversal incomplete — restart from a fresh query over a re-captured snapshot".
- **Verification**: `tests/binding_snapshot_semantics.rs::provider_fault_is_taken_once_then_the_cursor_resumes` (take-once + resume-empty) and `tests/abi_paging_correspondence.rs::adversarial_pages_are_rejected_without_aborting` (no abort / no fabrication under adversarial pages); registered as invariant LLEV-CUR-1 (`ABI_INVARIANTS.tsv`, test-pinned over the finite fault-channel FSM).
- **Status**: LEDGER + DOCS (behavioral contract; no code change).

### LLEV-B18 — `llev_distance*` reject a null pointer for an empty operand
- **Date**: 2026-08-09 · **Component**: `src/ffi/mod.rs` `cbuf_to_str` (used by all six `src/ffi/distance.rs` functions) · **Class**: correctness · **Severity**: low
- **Evidence**: surfaced by the W7 C8 property suites across the facades. `cbuf_to_str` checked `ptr.is_null()` *before* handling `len == 0`, so `llev_distance(NULL, 0, NULL, 0)` returned the invalid-input sentinel `usize::MAX` instead of `0`. The defect is reachable from any facade whose host runtime materializes an empty string as a null data pointer with length 0: runtime-confirmed in the dotnet facade (`fixed (byte* p = new byte[0])` yields null) and the go facade (`(&slice[0])` is nil for an empty slice), where `Distance("", "")` returned the sentinel. The transducer *query* path (`src/ffi/index.rs`, `len == 0` checked first) had always accepted `(NULL, 0)` as an empty query, so the two surfaces disagreed on the same input class.
- **Analysis**: a zero length denotes the empty string irrespective of the pointer, and `slice::from_raw_parts(NULL, 0)` is undefined behavior even at zero length — so the null branch must be reached only when `len > 0`. Ordering the `len == 0` short-circuit first both fixes the empty-operand result and removes the latent UB, and harmonizes the standalone distance functions with the query path. A null pointer with a *non-zero* length remains an invalid operand and must keep returning `usize::MAX`.
- **Fix**: this commit — `cbuf_to_str` returns `Some("")` when `len == 0` before testing the pointer; `(NULL, len > 0)` still returns `None` (→ `usize::MAX`). Test-only remit made the finder scope its distance properties to non-empty inputs and document the gap; that scoping can now be lifted for the affected facades since the root cause is fixed at the ABI. (The facade-side `.md` notes remain accurate — passing a non-null buffer for empty was, and is, also correct — so they need no revision.)
- **Verification**: `src/ffi/distance.rs::tests::ffi_distance_treats_null_zero_length_as_empty_string` — `(NULL, 0)` empty-to-empty is `0` across all six variants; a null empty operand pairs symmetrically with a non-empty operand at cost = its length; threshold bounds still hold; and `(NULL, len > 0)` still returns `usize::MAX`. Full `cargo test --lib --features ffi ffi::distance::tests` green (5/5).
- **Status**: FIXED.
