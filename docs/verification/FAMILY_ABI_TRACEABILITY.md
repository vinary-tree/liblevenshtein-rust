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

