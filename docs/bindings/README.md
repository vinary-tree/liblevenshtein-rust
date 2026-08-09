# Language-binding documentation hub

The map of everything written about liblevenshtein's ABI and language
bindings, in reading order. Two layers exist and never duplicate each
other: the **family canon** (hosted with the `vinary-tree-interop` crate in
this repository) specifies the shared resource ABI every vinary-tree
project speaks; the **project corpus** (this directory and its satellites)
specifies what liblevenshtein builds above it — the `llev_*` C surface, the
resource consumer, the cursor laws, and the JS/WASM topology.

## Reading order

1. **Decide-and-orient:** [`docs/language-bindings.md`](../language-bindings.md)
   — the architecture decision (small versioned C resource ABI + generated
   constants + hand-written facades; why not UniFFI), the three layers, the
   snapshot/marshalling contracts, tiers, distribution, and platform
   policy.
2. **The family canon** (normative, shared by all four projects):
   [interop README](../../vinary-tree-interop/README.md) — the portal ·
   [ABI reference](../../vinary-tree-interop/docs/abi-reference.md) — the
   annotated header walk with the refcount/paging/snapshot laws ·
   [evolution policy](../../vinary-tree-interop/docs/abi-evolution.md) —
   the four version counters and the additive-versus-fork rules ·
   [security model](../../vinary-tree-interop/docs/security-model.md) —
   zones, containment, validation duties.
3. **The project corpus** (this layer):

| Document | What it specifies |
|---|---|
| [c-abi-reference.md](c-abi-reference.md) | All 35 `llev_*` functions: signatures, preconditions, exact returnable status sets, ownership, thread safety, complexity; the 13-value status table and its `VtStatus` mapping; the lease protocol with literate batch-loop and reducer pseudocode; a compile-checked complete C consumer. |
| [resource-consumer.md](resource-consumer.md) | The safe-Rust layer under the C ABI: intake (retain-validate-else-release), `ForeignNode` domains, the `CallGate` (VT-GATE-1..3), the status wire rule and fault latch, the total `BindingError` map, and the two-pass arena fixup. |
| [wasm-topology.md](wasm-topology.md) | The JS exception to modular packaging: the `@vinary-tree/vinary-tree` umbrella, the three runtime paths, the runtime-identity guard, WASI preopen policy, and panic-versus-status discipline. |
| [../theory/snapshot-semantics.md](../theory/snapshot-semantics.md) | The cursor laws S1-S6 as display math, the $`\mathcal{O}(1)`$-capture argument from path-copied revisions, the partial-persistence classification, the refcount lineage, and the law ↔ model ↔ test correspondence table. |
| [../security/binding-trust-model.md](../security/binding-trust-model.md) | The family trust model instantiated for this consumer: `boundary()`, the bounded error channel, the decoded status wire, lease-refusal as UAF prevention, duty status per hostile-input class. |
| [FINDINGS_LEDGER.md](FINDINGS_LEDGER.md) | The scientific ledger of confirmed binding findings (LLEV-B1…), append-only, with fix commits and verification. |
| [../releasing-language-bindings.md](../releasing-language-bindings.md) | The release process: publish-order DAG, registry coordinates, credentials, pin-coherence preconditions, gates. |

4. **Machine-readable governance** (the sources the gates enforce):

| Artifact | Role |
|---|---|
| [`bindings/api.json`](../../bindings/api.json) | The single source of truth: versions, status/algorithm/order enums, the 35 modeled `cFunctions`, marshalling and snapshot law strings, forbidden owned objects, the canonical snapshot fixture. `scripts/generate-bindings.py` emits the headers/constants; `--check` pins them in CI. |
| [`bindings/api-surface-map.json`](../../bindings/api-surface-map.json) | The per-facade completeness model driving the coverage matrix. |
| [`bindings/conformance/`](../../bindings/conformance) | Generated conformance fixtures (query-start snapshot TSV, completeness matrix) every language suite replays. |
| [`scripts/check-bindings.py`](../../scripts/check-bindings.py) | The contract gate: symbol parity model ↔ Rust ↔ header, forbidden retired APIs, umbrella identity guard, coordinates, feature-alias policy. |
| [`docs/verification/ABI_INVARIANTS.tsv`](../verification/ABI_INVARIANTS.tsv) | The canonical invariant registry (VT-LIFE, VT-QI, VT-GATE, VT-ABI, and the wave-W3 rows as they land) tying each law to its model, test, and gate. |

5. **Diagrams:** the binding suite lives in
   [`docs/diagrams/bindings/`](../diagrams/bindings) (sources + committed
   SVGs, rendered by `docs/diagrams/render.sh bindings`): the four canon
   diagrams (vt-structs class, interface negotiation, evolution timeline,
   trust zones) and the ten project diagrams (three-layer architecture,
   family data flow, registry topology, resource handoff, lease lifecycle,
   cursor-lease FSM, resource-lifecycle FSM, reducer flow, call-gate
   serialization, WASM umbrella deployment).

## The family, one hop away

Per the separation-of-concerns rule, each repository documents its own ABI
surface; these are the sibling entry points this corpus cites:

- **vinary-tree-interop** (hosted here) — the canon trio above.
- **libdictenstein** (producer): [binding hub](https://github.com/vinary-tree/libdictenstein/blob/master/docs/bindings/README.md) ·
  [`ldict_*` C-ABI reference](https://github.com/vinary-tree/libdictenstein/blob/master/docs/bindings/c-abi-reference.md) ·
  [resource producer](https://github.com/vinary-tree/libdictenstein/blob/master/docs/bindings/resource-producer.md) ·
  [FFI boundary security](https://github.com/vinary-tree/libdictenstein/blob/master/docs/security/ffi-boundary.md)
- **lling-llang** (WFST producer + consumer): [`lling_*` C-ABI reference](https://github.com/vinary-tree/lling-llang/blob/master/docs/api/c-abi-reference.md) ·
  [resource ABI architecture](https://github.com/vinary-tree/lling-llang/blob/master/docs/architecture/resource-abi.md) ·
  [ABI trust model](https://github.com/vinary-tree/lling-llang/blob/master/docs/security/abi-trust-model.md)
- **duallity** (dictionary consumer, WFST producer): [resource ABI and bindings](https://github.com/vinary-tree/duallity/blob/master/docs/architecture/06-resource-abi-and-bindings.md) ·
  [language-bindings guide](https://github.com/vinary-tree/duallity/blob/master/docs/guides/07-language-bindings.md) ·
  [threat model](https://github.com/vinary-tree/duallity/blob/master/docs/security/threat-model.md)

(Sibling links are absolute — these are separate repositories; the sibling
documents land with their own waves of this program.)
