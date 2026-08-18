# vinary-tree-interop

**One ABI, zero algorithms.** `vinary-tree-interop` is the stable C ABI the
vinary-tree family uses to hand *live* resources — dictionaries and weighted
finite-state transducers — between independently built libraries and
language bindings. The crate is `#![no_std]`, dependency-free, and contains
**layouts and constants only**: no functions, no allocator, no I/O, nothing
executable. Every function that exists at this boundary is a pointer inside
a provider-supplied vtable. Project crates own the safe wrappers and the
concrete resource implementations; this crate owns the bytes they agree on.

That inversion is what makes the family modular: a consumer that reads only
[`include/vinary_tree_interop.h`](include/vinary_tree_interop.h) (or the
bit-identical Rust definitions in [`src/lib.rs`](src/lib.rs)) can consume
resources from any provider — another family repo, another *version* of a
family repo, or a third party — without linking against it and without
touching Rust's unstable dynamic-library ABI.

## The two-word resource

A resource is exactly two machine words:

```c
typedef struct VtResource {
    void* context;                          /* provider-defined state  */
    const struct VtResourceVTable* vtable;  /* provider-owned, immutable */
} VtResource;
```

Handing one across the boundary is $`\mathcal{O}(1)`$ — sixteen bytes on a
64-bit target, never a serialization. The base vtable carries a
reference-counting pair (`retain` / `release`) and versioned capability
discovery (`query_interface`), the portable core of COM's `IUnknown`.
Three laws do the heavy lifting (stated precisely, with their proofs and
pins, in the [ABI reference](docs/abi-reference.md)):

- **Two-word law** — `sizeof(VtResource)` is exactly two pointers; asserted
  at compile time in Rust and C++ and at runtime on the JVM.
- **Copy-is-not-retain** — copying the two words is free and confers no
  ownership; each *owned* copy is paid for with one `retain` and settled
  with exactly one `release`.
- **Snapshot capture is** $`\mathcal{O}(1)`$ — mutable dictionaries hand
  consumers an immutable, structurally shared revision at operation start;
  copying the dictionary or holding a long-lived read lock violates the
  interface contract. Batched paging (`node_edges` / `state_arcs`, 256
  entries recommended) keeps boundary crossings to
  $`\lceil \deg(v) / 256 \rceil`$ per node expansion.

![Class model of the published ABI: the two-word VtResource, its base VtResourceVTable, the dictionary and scalar-WFST interface vtables discovered through query_interface, the caller-owned page payload types, and the shared status/domain enums.](../docs/diagrams/bindings/vt-structs-class.svg)

## Interface catalog

Interfaces are named by 16-byte identifiers compared byte-for-byte and
negotiated through `query_interface` with a consumer-supplied minimum
version:

| Interface ID | Version | Payloads | What it is |
|---|---|---|---|
| `vt.dictionary.v1` | 1 (`VT_DICTIONARY_INTERFACE_VERSION`) | `VtDictionaryEdge`, `VtOptionalU64` | Immutable dictionary snapshots: root/final/transition queries plus batched edge paging, over byte, Unicode-scalar, or `u64` label domains, with optional `u64` values. |
| `vt.dict.visit.v1` | 1 (`VT_DICTIONARY_VISIT_INTERFACE_VERSION`) | `VtDictionaryEdge` | Optional fused finality and edge-page inspection for callback-based dictionary traversal. |
| `vt.dict.graph.v1` | 1 (`VT_DICTIONARY_GRAPH_INTERFACE_VERSION`) | `VtDictionaryGraphNode`, `VtDictionaryGraphEdge`, `VtDictionaryGraphView` | Optional compact immutable snapshot graph. A consumer validates the complete borrowed view once, retains its owner, and thereafter traverses dense node/edge arrays without provider callbacks or consumer cache publication. |
| `vt.snapshot.id.1` | 1 (`VT_SNAPSHOT_IDENTITY_INTERFACE_VERSION`) | `VtSnapshotIdentity` | Optional process-local immutable producer/revision identity for safely sharing derived state across separately retained views of the same snapshot. |
| `vt.scalar-wfst.1` | 1 (`VT_WFST_INTERFACE_VERSION`) | `VtWfstArc` | Immutable scalar-weighted FSTs: start/finality/arc paging with `f64` weights in one of seven declared semirings, epsilon labels encoded by flag (never by magic value), lazy and acyclic capability claims. |

Base protocol version: `VT_ABI_VERSION` = 1. The full change rules — what
may be added, what forks an identity, and the four distinct version
counters — are the [ABI evolution policy](docs/abi-evolution.md).

## Who produces and consumes what

| Repository | Dictionary + optional graph | Scalar WFST | Notes |
|---|---|---|---|
| [libdictenstein](https://github.com/vinary-tree/libdictenstein) | **produces** | — | Dictionary resources publish the base interface; immutable DynamicDawg snapshots additionally publish compact graphs in all three unit domains. Other backends retain the callback fallback until they can expose the same immutable representation without copying at query start. |
| [liblevenshtein](https://github.com/vinary-tree/liblevenshtein-rust) | **consumes** | — | Validates compact graphs at snapshot acquisition and routes every applicable automaton through the shared captured-graph traversal seam; falls back to fused or paged callbacks for older providers. |
| [duallity](https://github.com/vinary-tree/duallity) | **consumes base dictionary** | **produces** | Builds Levenshtein/fuzzy WFSTs *from* consumed dictionary resources. |
| [lling-llang](https://github.com/vinary-tree/lling-llang) | — | **produces + consumes** | Publishes vector WFSTs and lazily composes consumed ones. |
| umbrella JS runtime ([`bindings/javascript-runtime`](../bindings/javascript-runtime)) | hosts | hosts | Depends on all four projects plus this crate; the one sanctioned all-of-family surface (Node, WASI, browser). |

## Documentation

| Document | Contents |
|---|---|
| [docs/abi-reference.md](docs/abi-reference.md) | The annotated, literate walk of the entire header: every declaration quoted and explained, the refcount/paging/two-word/snapshot laws in display math, the seven semirings defined, and a complete minimal C provider that compiles under `-std=c17 -Wall -Wextra -Werror`. |
| [docs/abi-evolution.md](docs/abi-evolution.md) | The four version counters and their jurisdictions, additive-versus-breaking rules per construct, worked examples (add an op, add a weight domain, retire a flag), the decision table, and the current family compatibility matrix. |
| [docs/security-model.md](docs/security-model.md) | The family trust model: zones, the panic/exception containment law with file:line evidence, threading-by-claim, the input-validation duty table grounded in confirmed findings, exhaustion vectors, WASI capability policy, and explicit non-goals. |

## Language packages

Nine language-native mirrors of the interop structs and constants live under
[`bindings/`](bindings/), so non-C ecosystems consume the ABI idiomatically
without generating from the header at build time:

[Fortran](bindings/fortran) · [Go](bindings/go) · [Haskell](bindings/haskell) ·
[JavaScript](bindings/javascript) · [JVM](bindings/jvm) · [Lua](bindings/lua) ·
[OCaml](bindings/ocaml) · [Python](bindings/python) · [Swift](bindings/swift)

Each mirror is layout-checked against the canonical definitions by the
repository's binding gates (`scripts/generate-bindings.py` and
`scripts/check-bindings.py` at the repo root).

## The executable contract

Three test suites in [`tests/`](tests/) are the normative documents' teeth;
CI runs them on every push (`cargo test --locked -p vinary-tree-interop`):

- [`tests/layout_contract.rs`](tests/layout_contract.rs) — exact sizes,
  alignments, and field offsets for every published type on the 64-bit and
  32-bit ARM tiers; the two-word law; byte-exact interface identifiers; the
  `Option<extern "C" fn>` null niche that makes NULL-slot vtables one ABI.
- [`tests/discriminant_pins.rs`](tests/discriminant_pins.rs) — every enum
  discriminant and flag bit pinned exactly *and* matched wildcard-free, so
  adding a variant fails compilation until the evolution policy is
  consulted; zeroed reserved-field defaults.
- [`tests/vtable_evolution.rs`](tests/vtable_evolution.rs) — the
  `query_interface` negotiation surface against a hand-rolled provider,
  including the additive-evolution proof: a future, larger vtable remains
  consumable through its v1 prefix, discovered via `struct_size`.

## Crate facts

| Fact | Value |
|---|---|
| Version | 0.1.0 |
| Rust | edition 2021, `rust-version` 1.95, `#![no_std]` |
| Dependencies | none |
| License | Apache-2.0 |
