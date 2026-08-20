# Vinary Tree OCaml interop binding

This package exposes the language-native representation of the stable Vinary Tree resource ABI. It is the neutral handoff layer used by dictionary, automaton, and WFST packages; it owns no algorithm-specific policy.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | OCaml |
| Languages/runtime | OCaml 5 through dune/opam |
| Support tier | Tier 3 |
| Distribution | opam package `vinary-tree-interop` |
| Native boundary | This adapter represents the two-word `VtResource` capability and its versioned interfaces; it does not implement a dictionary or automaton. |
| Canonical facade source | [`vinary-tree-interop/bindings/ocaml`](../../../vinary-tree-interop/bindings/ocaml) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/abi-reference.md) before implementing a custom provider
and the [family hub](../../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../../docs/diagrams/bindings/interface-negotiation-activity.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/ocaml/test/snapshot.ml`](../../../bindings/ocaml/test/snapshot.ml). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
opam exec -- dune runtest --root bindings/ocaml
```

Examples deliberately construct or receive resources through public project
packages. They never import private Rust modules, depend on object layout, or
reach behind the stable C/resource ABIs.

## Public API and data model

The idiomatic facade groups the stable surface into these concepts:

| Concept | Semantics |
|---|---|
| `VtResource` | Two pointer-sized words: an opaque context and a base vtable. A borrowed value transfers no ownership. |
| Base vtable | `struct_size`, ABI version, retain, release, and `query_interface`; it is the only mandatory interface. |
| Dictionary interface | Snapshot capture, node paging, finality, optional values, unit/value domains, and capability flags. |
| Dictionary entries interface | Optional finite lexicographic stream over one captured revision, with bounded arena batches, exact generation leases, cancellation, and a reducer path. |
| Scalar-WFST interface | Snapshot capture, start state, final weights, paged arcs, label/weight domains, and capability flags. |

Unit and value domains are explicit enum fields on the discovered interface; adapters must never infer them from host container types. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[family resource ABI reference](../../docs/abi-reference.md). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by [`bindings/api.json`](../../../bindings/api.json) and the generated interop constants.

## Ownership, snapshots, and resource handoff

Use the explicit `close` functions or `Fun.protect`; GC finalizers are only a last-resort retain release. Close every entries cursor, and release its current generation before advancing or closing it.

A borrowed resource becomes owned only after a successful `retain`. Interface
discovery does not transfer ownership, and a failed validation must release any
retain already acquired. A captured snapshot owns an independent revision and
may outlive the producing project handle. Release exactly once for every
successful retain; never release an unretained borrowed pair. An entries cursor is move-only and owns its
captured revision until `close`. Exactly one generation may be live: release
that exact generation before advancing, reducing, or closing; reducer batch
views expire when their callback returns.

Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena.

## Errors and failure containment

Interop validation failures preserve `VtStatus`; project facades map that status into their own public error currency.

Null resource words, truncated vtables, incompatible interface identities or versions, invalid domains, forged node/state identifiers, malformed page counts or entry arenas, stale or mismatched batch generations, live-batch conflicts, provider faults, and contained panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

A retained resource may cross threads only when its advertised interface flags permit it. Retain and release remain balanced under every failure path. One entries cursor and its live batch are single-consumer; reducer callbacks must not reenter that cursor.

Snapshot capture is a linearization point, not a dictionary-wide query lock.
First-party immutable snapshots can be walked concurrently. A foreign provider
that does not advertise parallel callbacks is serialized at its callback gate;
the host language must not add a weaker promise.

## Performance and marshalling

- Pass the two-word resource by value; do not serialize or copy the graph.
- Capture one immutable snapshot and page nodes/arcs through bounded buffers.
- Negotiate entries-v1 when exact lexicographic enumeration is needed; honor all entry/unit/value limits on every batch.
- Cache a validated optional interface only while the owning resource remains retained.
- Respect capability flags before enabling parallel callback entry.
- Prefer a compact immutable graph interface when advertised; retain the paged callback fallback for compatibility.

## Security model

Treat a foreign resource provider and all user-controlled queries as untrusted
inputs. Validate lengths before allocation, preserve paging bounds, reject
unknown enum values, contain callbacks/panics at the boundary, and never trust
capability flags until interface negotiation succeeds. The normative duties
are in the [binding trust model](../../docs/security-model.md).

## Compatibility and troubleshooting

The project ABI revision, family ABI version, interface identity/version,
package version, and umbrella-runtime version are independent counters. Follow
the [ABI evolution policy](../../docs/abi-evolution.md); never infer compatibility from a
package version alone.

When loading fails, check—in order—the documented runtime/toolchain version,
CPU/OS artifact, native-access permission, loader search path, dependent
interop package pin, and process-wide JavaScript runtime identity. When a query
fails after construction, report the typed status and copied diagnostic before
reducing the case to the smallest dictionary/query pair.

## Maintainer checklist

1. Update the interop generator model before changing a layout, identifier, flag, or enum.
2. Regenerate headers/constants and the API coverage matrix.
3. Extend the canonical executable example and negative-path tests.
4. Run the language package, snapshot, leak, property, and cross-project suites.
5. Verify package staging contains this guide and uses coherent sibling pins.
6. Render diagrams headlessly and run the documentation/link/math gates.

<!-- END GENERATED BINDING OPERATIONS -->
