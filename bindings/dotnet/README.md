# Vinary Tree .NET bindings

`Liblevenshtein` is the NuGet distribution for the idiomatic, streaming
`VinaryTree.Liblevenshtein` .NET namespace over the
stable native ABI. `VinaryTree.Interop` contains the shared two-word retained
resource contract used by independently packaged dictionary producers.

The package targets .NET 8 (the oldest supported LTS) and uses the latest C#
language standard. Query enumerators retain the query-start dictionary revision
and lease only one native result batch at a time. Dispose transducers and
enumerators deterministically; `SafeHandle` supplies the leak-safe fallback.

Build and test with `dotnet run --project tests/VinaryTree.Liblevenshtein.Tests`.
NuGet packaging is produced with `dotnet pack -c Release`.

Dictionary producers implementing `IDictionaryResource` gain immutable native
.NET collection views from `VinaryTree.Interop`:

```csharp
DictionarySnapshot snapshot = dictionary.SnapshotEntries();
DictionaryKey key = DictionaryKey.FromString("café");
bool present = snapshot.Entries.TryGetValue(key, out ulong? value);

using DictionaryEntryStream entries = dictionary.StreamEntries();
foreach (DictionaryEntry entry in entries)
{
    Console.WriteLine(entry);
    if (ShouldStop(entry)) break;
}
```

`DictionarySnapshot` implements `IReadOnlyCollection<DictionaryEntry>` and
exposes ordered `IReadOnlySet<DictionaryKey>` and
`IReadOnlyDictionary<DictionaryKey, ulong?>` views. `TryGetValue` distinguishes
an absent key from a present key with no value. Keys preserve arbitrary bytes,
Unicode scalars, and the complete `ulong` domain with value equality. Snapshot
metadata carries exact length and revision identity. All three views share one
ordered entry array and use binary-search lookup; they do not build duplicate
output-sized hash tables or take a lazy-initialization lock. Set-relation calls
accept every ordinary `IEnumerable<DictionaryKey>` and allocate temporary
state only when the requested relation requires de-duplicating that external
sequence. `OpenEntryEnumerator` and `StreamEntries` own a native cursor;
dispose them after early termination. Enumeration copies and releases each
bounded batch before yielding managed entries, and cleanup cancels unread work.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | .NET |
| Languages/runtime | .NET 8+ and current C# |
| Support tier | Tier 2 |
| Distribution | NuGet package `Liblevenshtein` |
| Native boundary | Source-generated P/Invoke reaches the stable C ABI; `VinaryTree.Interop` carries retained resource handles between packages. |
| Canonical facade source | [`bindings/dotnet/src/VinaryTree.Liblevenshtein`](../../bindings/dotnet/src/VinaryTree.Liblevenshtein) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/dotnet/tests/VinaryTree.Liblevenshtein.Tests/Program.cs`](../../bindings/dotnet/tests/VinaryTree.Liblevenshtein.Tests/Program.cs). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
dotnet run --project bindings/dotnet/tests/VinaryTree.Liblevenshtein.Tests
```

Examples deliberately construct or receive resources through public project
packages. They never import private Rust modules, depend on object layout, or
reach behind the stable C/resource ABIs.

## Public API and data model

The idiomatic facade groups the stable surface into these concepts:

| Concept | Semantics |
|---|---|
| Dictionary resource | A retained `vt.dictionary.v1` capability. Construction and mutation belong to a producer such as libdictenstein. |
| Transducer | Immutable query configuration plus a retained dictionary provider; construction is constant-time with respect to dictionary size. |
| Query cursor | A one-shot traversal over the immutable dictionary revision captured at query start. |
| Match/batch | Owned matches are stable host values; a borrowed batch is valid only inside its documented callback or lease interval. |

`string`, `ReadOnlySpan<byte>`, and `ReadOnlySpan<ulong>` select Unicode, byte, and token domains. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[`llev_*` C ABI reference](../../docs/bindings/c-abi-reference.md). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by [`bindings/api-surface-map.json`](../../bindings/api-surface-map.json) and the [generated completeness matrix](../../bindings/conformance/completeness-matrix.tsv).

## Ownership, snapshots, and resource handoff

Use `using`/`await using`-style lexical ownership for disposable handles. `SafeHandle` protects exceptional paths but does not replace disposal.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena.

## Errors and failure containment

Non-OK statuses become typed .NET exceptions with status and native diagnostic properties.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

Independent handles are safe across tasks. One enumerator is single-consumer; cancellation or early exit must dispose it to release the batch lease.

Snapshot capture is a linearization point, not a dictionary-wide query lock.
First-party immutable snapshots can be walked concurrently. A foreign provider
that does not advertise parallel callbacks is serialized at its callback gate;
the host language must not add a weaker promise.

## Performance and marshalling

- Reuse transducers for repeated queries against the same resource.
- Prefer streaming cursors to whole-result materialization.
- Prefer batch/reducer APIs when per-match boundary crossings dominate.
- Keep Unicode, byte, and token domains explicit to avoid transcoding.
- Measure native, WASM, and WASI paths independently; they have different
  startup and marshalling costs but identical query semantics.

No host wrapper should cache unbounded query results. Applications that add a
memo use a revision key and a hard entry/weight bound; eviction may be
approximate because all values remain derivable from the retained snapshot.

## Security model

Treat a foreign resource provider and all user-controlled queries as untrusted
inputs. Validate lengths before allocation, preserve paging bounds, reject
unknown enum values, contain callbacks/panics at the boundary, and never trust
capability flags until interface negotiation succeeds. The normative duties
are in the [binding trust model](../../docs/security/binding-trust-model.md).

## Compatibility and troubleshooting

The project ABI revision, family ABI version, interface identity/version,
package version, and umbrella-runtime version are independent counters. Follow
the [ABI evolution policy](https://github.com/vinary-tree/vinary-tree-interop/blob/master/docs/abi-evolution.md); never infer compatibility from a
package version alone.

When loading fails, check—in order—the documented runtime/toolchain version,
CPU/OS artifact, native-access permission, loader search path, dependent
interop package pin, and process-wide JavaScript runtime identity. When a query
fails after construction, report the typed status and copied diagnostic before
reducing the case to the smallest dictionary/query pair.

## Maintainer checklist

1. Update the machine-readable binding model before changing a public symbol.
2. Regenerate headers/constants and the API coverage matrix.
3. Extend the canonical executable example and negative-path tests.
4. Run the language package, snapshot, leak, property, and cross-project suites.
5. Verify package staging contains this guide and uses coherent sibling pins.
6. Render diagrams headlessly and run the documentation/link/math gates.

<!-- END GENERATED BINDING OPERATIONS -->
