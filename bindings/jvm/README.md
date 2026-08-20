# JVM binding

This binding uses the finalized Foreign Function & Memory API (JDK 22+) over
the stable C ABI. JDK 25 LTS is the primary build/test runtime, JDK 26 is also
tested, and bytecode targets Java 22. Java, Kotlin, and Scala consume the same
Java API. Clojure users should normally depend on the idiomatic Clojars facade,
`io.vinarytree/liblevenshtein-clojure`, which delegates to this
artifact without introducing another native boundary.

The published `io.vinarytree:liblevenshtein` JAR contains native libraries for
Linux x86_64/aarch64, macOS aarch64, and Windows x86_64. They are extracted to
a private temporary file and loaded automatically. Source-tree development may
instead supply the library through `java.library.path`.

Run with native access enabled:

```text
# Class path
--enable-native-access=ALL-UNNAMED

# Module path (automatic module name)
--enable-native-access=io.vinarytree.liblevenshtein
```

`Transducer` accepts `io.vinarytree.interop.DictionaryResource`; the
resource normally comes from libdictenstein and is retained in O(1).
Liblevenshtein exposes no dictionary constructors or CRUD. `QueryCursor` is
lazy, closeable, batch-aware, and retains the immutable dictionary
revision visible at query start. It never holds a read lock while JVM code
processes results. The ordinary iterator materializes one batch at a time;
`forEachBatch` exposes callback-scoped `MemorySegment` views for the
allocation-minimizing path. `BorrowedMatchBatch` provides direct indexed
`distance`, `id`, `unitDomain`, `termLength`, `byteLength`, `bytes`, `utf8`, and
`u64` accessors that do not allocate a per-match wrapper; `get(index)` remains
the convenient object view. Every borrowed view expires when its batch callback
returns. Cursors and transducers use `Cleaner` only as a leak-safety fallback;
applications should close them deterministically.

Dictionary resources also provide natural immutable collections through the
neutral interop artifact. No Kotlin or Scala runtime is pulled into the Java
artifact:

```java
DictionarySnapshot snapshot = dictionary.entriesSnapshot();
boolean presentWithoutValue =
    snapshot.entries().containsKey(DictionaryKey.unicode("café"))
        && snapshot.entries().get(DictionaryKey.unicode("café")).isEmpty();

try (var entries = dictionary.entryStream()) {
    entries.limit(20).forEach(System.out::println);
}
```

```kotlin
val snapshot = dictionary.entriesSnapshot()
val present = DictionaryKey.unicode("café") in snapshot.keys()
dictionary.entryStream().use { entries ->
    entries.limit(20).forEach(::println)
}
```

```scala
import scala.jdk.CollectionConverters.*
import scala.util.Using

val snapshot = dictionary.entriesSnapshot()
val first = Using.resource(dictionary.entryIterator()) { cursor =>
  cursor.asScala.take(20).toVector
}
```

`DictionaryKey` preserves arbitrary bytes, exact Unicode scalars, and raw
unsigned-64 token bits with value equality. Snapshot Set/Map and ordered-entry
views are unmodifiable and retain native lexicographic order. A map contains
every present key, so `containsKey` separates absence from an
`Optional.empty()` present-unvalued entry. Exact length and snapshot identity
are captured in `DictionarySnapshot.metadata()`.

<!-- BEGIN GENERATED BINDING OPERATIONS; DO NOT EDIT -->

## Support and package contract

| Property | Contract |
|---|---|
| Binding | JVM |
| Languages/runtime | Java 22+, Kotlin, and Scala |
| Support tier | Tier 1 |
| Distribution | Maven coordinate `io.vinarytree:liblevenshtein` |
| Native boundary | The finalized Foreign Function & Memory API calls the stable C ABI. Kotlin and Scala consume the same Java classes and ownership contracts. |
| Canonical facade source | [`bindings/jvm/src/main/java/io/vinarytree/liblevenshtein`](../../bindings/jvm/src/main/java/io/vinarytree/liblevenshtein) |

The support tier controls release gating, not semantic quality: every tier has
the same snapshot, ownership, status, and ABI compatibility laws. Consult the
[binding architecture](../../docs/language-bindings.md) before implementing a custom provider
and the [family hub](../../docs/bindings/README.md) when combining independently packaged projects.

![The host-language facade crosses one project ABI and retains a versioned family resource rather than sharing Rust object layouts.](../../docs/diagrams/bindings/three-layer-architecture.svg)

## Executable example and verification

The repository's canonical executable example is
[`bindings/jvm/src/smoke/java/io/vinarytree/liblevenshtein/ResourceSnapshotSmoke.java`](../../bindings/jvm/src/smoke/java/io/vinarytree/liblevenshtein/ResourceSnapshotSmoke.java). It exercises the same public package a user
installs and is run by the binding CI with:

```sh
./gradlew -p bindings/jvm test
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

Java `String` is UTF-8 encoded for Unicode queries; byte arrays and `long[]` select the byte and packed-token domains. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

For the exhaustive native function contract—including exact preconditions,
returnable statuses, complexity, and thread-safety—use the
[`llev_*` C ABI reference](../../docs/bindings/c-abi-reference.md). The facade
source linked above is the authoritative idiomatic symbol inventory; its
exhaustive coverage is governed by [`bindings/api-surface-map.json`](../../bindings/api-surface-map.json) and the [generated completeness matrix](../../bindings/conformance/completeness-matrix.tsv).

## Ownership, snapshots, and resource handoff

Use try-with-resources, Kotlin `use`, or Scala `Using.resource`. `Cleaner` is only a leak-safety fallback.

A transducer retains the provider resource, and a query retains the revision
visible at query start. Closing the original dictionary or publishing later
mutations cannot invalidate that query. Acquisition either completes with one
owned retain or fails with no ownership transfer. Teardown order is therefore
free across dictionary, transducer, and completed query handles.

Borrowed results are intentionally lexical. Copy data that must outlive the
callback; retaining a raw address, slice, memory segment, or foreign pointer is
an API violation even when the next operation happens to reuse the same arena.

## Errors and failure containment

Every non-OK status becomes a Java exception containing the thread-local native diagnostic; callback exceptions still close the active lease.

Malformed utf-8, unsupported unit domains, incompatible resource versions, closed handles, invalid bounds, allocation failures, provider faults, and contained rust panics are distinct failures. Never parse diagnostic prose to
branch on an error: inspect the typed status/exception first and treat the
message as human context. Diagnostics must be copied before another native
call on the same thread.

## Concurrency and reentrancy

Transducers and independent cursors are reentrant. One cursor is single-consumer; borrowed `MemorySegment` views expire when their batch callback returns.

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
the [ABI evolution policy](../../vinary-tree-interop/docs/abi-evolution.md); never infer compatibility from a
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
