# Family ABI Invariant Traceability

> **Scope.** The `vinary-tree` family's cross-project ABI (the `#![no_std]`
> `vinary-tree-interop` resource contract — `VtResource`, `vt.dictionary.v1`,
> `vt.scalar-wfst.1` — and the per-project C ABIs and language facades built on
> it) is verified against a traceability spine: every ABI invariant has a
> **formal home** (a Rocq theorem, a TLA+ property, an SMT check, or a Verus
> contract), a **strength** on the proof ladder, a **mirroring executable
> test** (or is a formal-only algebra result), and the **gates** that run both.
> This document is the family-level index over the five per-repo registries; it
> lives in the interop host (`liblevenshtein-rust`) and cross-links each repo's
> own registry, which remains the authoritative source for its rows.

As of the closure pass, the family carries **98 registered ABI
invariants** across five repositories, each checked by a per-repo registry
gate that enforces model↔invariant↔test↔gate closure.

## Per-repository registries

| Repository | Registry | Invariants | Registry gate |
|---|---|---|---|
| liblevenshtein-rust (interop host + consumer) | `liblevenshtein-rust/docs/verification/ABI_INVARIANTS.tsv` | 39 | `scripts/check-abi-invariants.py` |
| libdictenstein (producer) | `libdictenstein/formal-verification/ABI_INVARIANTS.tsv` | 18 | `verify-formal-correspondence.sh` |
| lling-llang (WFST) | `lling-llang/proofs/doc/abi-invariants.tsv` | 21 | `scripts/check-abi-invariants.py` |
| duallity (integration) | `duallity/proofs/doc/abi-invariants.tsv` | 14 | `scripts/check-abi-invariants.py` |
| llattice (lattice leaf) | `llattice/proofs/doc/lattice-invariants.tsv` | 6 | `scripts/check-lattice-invariants.py` |
| **Total** | | **98** | |

## Proof-strength distribution

The strength ladder (strongest first): `rocq-unbounded` > `tlaps-unbounded` >
`apalache-inductive` > `verus` > `smt-dual` > `tlc-bounded` > `test-pinned`.

| Strength | Count |
|---|---|
| `rocq-unbounded` | 52 |
| `verus` | 3 |
| `smt-dual` | 8 |
| `tlc-bounded` | 23 |
| `test-pinned` | 12 |

## Formal-home distribution (spec kind)

| Spec kind | Count |
|---|---|
| `coq` | 52 |
| `tla` | 23 |
| `test` | 12 |
| `smt` | 8 |
| `verus` | 3 |

## Invariant ID families

| Prefix | Count | Meaning |
|---|---|---|
| `VT-*` | 27 | interop resource ABI (lifecycle, query-interface, paging, layout, call gate, snapshot) |
| `LLING-*` | 21 | lling-llang WFST ABI (weight-domain semirings, weight bridge, composition, product registry, gate, status) |
| `LDICT-*` | 18 | libdictenstein producer ABI (snapshot capture, paging, lifecycle, status) |
| `DUAL-*` | 14 | duallity integration ABI (state codec, snapshot capture-once, adapter laws, fzf bound, status) |
| `LLEV-*` | 12 | liblevenshtein consumer ABI (batch lease, arena, status, cursor fault channel) |
| `LATT-*` | 6 | llattice algebra (lattice laws, float caveat) |

## Verifying the whole family

Each repository runs its own registry checker plus its proof/model gates:

- **liblevenshtein-rust**: `python3 scripts/check-abi-invariants.py` +
  `scripts/verify-formal.sh audit trusted coq-trusted tla verus smt`.
- **libdictenstein**: `scripts/verify-formal-correspondence.sh` (folds in the
  registry checker) + `scripts/verify-unsafe-boundary-inventory.sh`.
- **lling-llang**: `bash proofs/verify.sh` (Coq build + proof-check +
  `scripts/check-abi-invariants.py` + TLC models incl. mutants).
- **duallity**: `bash proofs/verify.sh` (Coq build + proof-check +
  `scripts/check-abi-invariants.py` + the capture-once TLC model + mutant).
- **llattice**: `bash proofs/verify.sh` (Coq build + proof-check +
  `scripts/check-lattice-invariants.py`).

Every `test-backed` row names a real test function that the repo's test suite
runs; every `formal-only` row (marked `-` in the test columns) is a pure
Rocq/TLA+ result whose evidence is the machine-checked proof itself. No row
may drift from its spec or test without its registry gate failing.

## W8 empirical corroboration

The formal invariants are corroborated dynamically by the wave-W8 performance and
dynamic-analysis artifacts (liblevenshtein-rust), which measure at machine level
what the proofs certify abstractly. These are corroboration, not a second formal
home — the machine-checked proof remains the primary evidence for each invariant.

| Invariant family | Empirical corroboration |
|---|---|
| `VT-PAGE-*` (paging law) | `tests/ffi_boundary_census.rs` measures `next_batch_calls == ceil(M/cap)+1`; `PERFORMANCE_EXPERIMENTS.md` E2 |
| `VT-SNAP-*` (constant-time capture) | census `snapshot_calls == 1` per query; `benches/ffi_boundary_benchmarks.rs` resource-handoff flat curve |
| `LLEV-ARENA-*` (arena reuse, no warm realloc) | `bindings/c/tests/arena_profile.c` (zero-copy packing + base reuse, run under ASan) + `scripts/profile-ffi-arena.sh` (valgrind massif/dhat) |
| `LLEV-LEASE-*`, `VT-LIFE-*` | census refcount-ledger balance; `scripts/run-sanitizers.sh` (ASan/TSan), plus per-repo ASan legs in lling-llang and duallity |

Performance decisions (batch-size default, ctypes-vs-cffi) are recorded in
`docs/bindings/PERFORMANCE_EXPERIMENTS.md`.

## DOI verification sweep (W9)

Every DOI across the five repositories' docs is resolved against `doi.org`
(first-hop resolution: `301`/`302` = a registered, resolvable DOI). Result:
**every citation DOI resolves**, save one intentional negative control.

- **1 intentional negative control** — `10.1145/9999999.9999999` in
  `docs/theory/snapshot-semantics.md`, an explicit "invalid DOI → 404" example
  demonstrating the verification methodology. Correctly non-resolving by design.
- **8 corrected citations.** The sweep initially found eight non-resolving DOIs in
  pre-existing algorithm/architecture/archive documents (none in the ABI/binding
  docs this program produced). Each was corrected to its authoritative DOI —
  identified via the Crossref bibliographic API and confirmed to resolve
  (`302`) against `doi.org`:

  | Paper (location) | Corrected DOI | Correction |
  |---|---|---|
  | Schulz & Mihov 2002, *Fast String Correction with Levenshtein Automata* (`docs/verification/INDEX.md` and `docs/algorithms/02-levenshtein-automata/`) | `10.1007/s10032-002-0082-8` | both docs cited wrong Springer DOIs for the IJDAR paper; unified to the canonical, resolving one |
  | Hinze & Jeuring, *Generic Haskell: Applications* (`docs/algorithms/06-zipper-navigation/`) | `10.1007/978-3-540-45191-4_2` | wrong LNCS chapter DOI |
  | Oommen & Loke 1997, *Pattern Recognition of Strings…* (`docs/algorithms/02-levenshtein-automata/`) | `10.1016/S0031-3203(96)00101-X` | transposed digits in the Elsevier DOI |
  | Yata et al., *Fast String Matching with Space-Efficient Word Graphs* (`docs/algorithms/01-dictionary-layer/…/double-array-trie.md`) | `10.1109/innovations.2008.4781726` | wrong DOI and year (IEEE Innovations **2008**, not 2009) |
  | Arulraj et al. 2018, *BzTree* (`libdictenstein/docs/design/history/`) | `10.1145/3187009.3164147` | wrong ACM DOI |
  | Blumer et al. 1987, *Complete Inverted Files…* (`libdictenstein/docs/architecture/optimization-roadmap.md`) | `10.1145/28869.28873` | wrong JACM DOI |
  | Bakhturina et al. 2021, *NeMo Inverse Text Normalization* (`duallity/docs/archive/references/papers.md`) | `10.21437/Interspeech.2021-1571` | wrong Interspeech proceedings number |

All DOIs across the family now resolve (the negative control excepted). The sweep
is the falsifiable artifact — re-run the resolution to reproduce.
