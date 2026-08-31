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

A Java caller receives a dictionary resource from its chosen storage package,
then uses ordinary try-with-resources for both the reusable transducer and each
one-shot query cursor:

```java
import io.vinarytree.interop.DictionaryResource;
import io.vinarytree.liblevenshtein.Algorithm;
import io.vinarytree.liblevenshtein.Match;
import io.vinarytree.liblevenshtein.QueryOrder;
import io.vinarytree.liblevenshtein.Transducer;
import java.util.ArrayList;
import java.util.List;

final class FuzzyExample {
    private FuzzyExample() {}

    static List<Match> fuzzy(DictionaryResource dictionary, String query) {
        try (var transducer = new Transducer(dictionary, Algorithm.TRANSPOSITION);
                var matches = transducer.query(query, 2, QueryOrder.DISTANCE_THEN_TERM)) {
            var result = new ArrayList<Match>();
            matches.forEachRemaining(result::add);
            return List.copyOf(result);
        }
    }
}
```

Phonetic automata and rewrite rules are reusable native resources too. Failures
carry both an enum suitable for control flow and the exact raw status for
forward-compatible diagnostics:

```java
import io.vinarytree.liblevenshtein.NativeException;
import io.vinarytree.liblevenshtein.PhoneticPattern;
import io.vinarytree.liblevenshtein.PhoneticRuleSet;
import io.vinarytree.liblevenshtein.PhoneticRuleSetKind;
import io.vinarytree.liblevenshtein.Status;

final class PhoneticExample {
    private PhoneticExample() {}

    static String normalizeIfAccepted(String input) {
        try (var pattern = PhoneticPattern.compileLlre(
                        "@name \"Greeting\"\n^hello$");
                var rules = PhoneticRuleSet.builtin(
                        PhoneticRuleSetKind.ENGLISH_ORTHOGRAPHY)) {
            return pattern.matches(input) ? rules.apply(input) : input;
        } catch (NativeException failure) {
            String detail =
                    "native status " + failure.statusCode() + ": " + failure.getMessage();
            if (failure.status() == Status.UNSUPPORTED) {
                throw new UnsupportedOperationException(detail, failure);
            }
            throw new IllegalStateException(detail, failure);
        }
    }
}
```

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

### Automaton selection

| Algorithm | Edit semantics | Metric? | Typical use |
|---|---|---:|---|
| Standard | Insert, delete, and substitute | yes | General spelling correction |
| Transposition | Optimal string alignment with adjacent swaps | no | Typographical swaps when metric-tree laws are unnecessary |
| Merge and split | Standard edits plus symmetric two-to-one and one-to-two edits | yes | Optical character recognition and segmentation errors |
| Damerau-Levenshtein | Unrestricted, history-composable adjacent transpositions | yes | True Damerau matching and metric indexes |

The transposition and unrestricted Damerau variants are deliberately distinct:
for example, optimal string alignment assigns distance 3 from `CA` to `ABC`,
while unrestricted Damerau-Levenshtein assigns distance 2. Select the algorithm
when constructing the transducer; all query domains and snapshot laws remain the
same.

Java `String` is UTF-8 encoded for Unicode queries; byte arrays and `long[]` select the byte and packed-token domains. Empty terms, embedded zero bytes, non-ASCII text, and the full
unsigned 64-bit identifier range are represented explicitly; no facade may use
a sentinel value that removes a valid input from the domain.

### Facade symbol index

This table is generated from the same exhaustive model as the binding
conformance gate. A public symbol may implement several ABI operations when
the host language expresses domain or lifecycle choices with overloads,
variants, protocols, or methods.

| Public symbol | Backing native operation(s) | Capability |
|---|---|---|
| `NativeException` | `llev_last_error_message` | typed failure diagnostics |
| `NativeException.status` | `llev_last_error_message` | typed failure diagnostics |
| `NativeException.statusCode` | `llev_last_error_message` | typed failure diagnostics |
| `PhoneticPattern.close` | `llev_phonetic_pattern_free` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.compileLlre` | `llev_phonetic_pattern_compile_llre` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.compileRegex` | `llev_phonetic_pattern_compile_regex` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.matches` | `llev_phonetic_pattern_matches` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticPattern.size` | `llev_phonetic_pattern_size` | compiled phonetic-pattern lifecycle and matching |
| `PhoneticRuleSet.apply` | `llev_owned_string_free`, `llev_phonetic_rules_apply` | owned result-string release; phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.builtin` | `llev_phonetic_rules_builtin` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.close` | `llev_phonetic_rules_free` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.parse` | `llev_phonetic_rules_parse` | phonetic rule-set lifecycle and rewriting |
| `PhoneticRuleSet.size` | `llev_phonetic_rules_len` | phonetic rule-set lifecycle and rewriting |
| `QueryCache` | `llev_query_cache_new` | project ABI operation |
| `QueryCache.clear` | `llev_query_cache_clear` | project ABI operation |
| `QueryCache.close` | `llev_query_cache_free` | project ABI operation |
| `QueryCache.query` | `llev_query_cache_query_utf8`, `llev_query_cache_query_bytes`, `llev_query_cache_query_u64` | project ABI operation |
| `QueryCache.resetStats` | `llev_query_cache_reset_stats` | project ABI operation |
| `QueryCache.stats` | `llev_query_cache_stats` | project ABI operation |
| `QueryCursor.close` | `llev_query_cursor_free` | streaming result traversal and batch leases |
| `QueryCursor.forEachBatch` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `QueryCursor.next` | `llev_query_cursor_next_batch`, `llev_query_cursor_release_batch` | streaming result traversal and batch leases |
| `Status.code` | `llev_last_error_message` | typed failure diagnostics |
| `Transducer` | `llev_transducer_new` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer.close` | `llev_transducer_free` | transducer lifecycle, snapshot, or domain metadata |
| `Transducer.query` | `llev_transducer_query_utf8`, `llev_transducer_query_bytes`, `llev_transducer_query_u64`, `llev_transducer_query_pattern` | domain-preserving dictionary query; phonetic-pattern dictionary query |
| `Transducer.snapshot` | `llev_transducer_snapshot` | transducer lifecycle, snapshot, or domain metadata |

### Public types and traversal protocols

| Facade type or protocol | Purpose | Exposure note |
|---|---|---|
| `Status` | Typed native status or error carrier | generated from bindings/api.json; NativeException.status returns this enum and preserves unknown raw values separately |
| `Algorithm` | Edit-distance algorithm selection | Public facade type |
| `QueryOrder` | Result traversal ordering | Public facade type |
| `PhoneticRuleSetKind` | Built-in phonetic rule-set selection | Public facade type |
| `QueryCursor.next` | One-shot owned-result iteration | Public facade protocol |
| `QueryCursor.forEachBatch` | Bounded batch/reducer traversal | Public facade protocol |

Native operations omitted from the public-symbol table are deliberately
encapsulated by the facade. The generated completeness matrix records every
such operation with its reviewed rationale; an unreasoned absence fails CI.

### Intended usage paths

| Need | Use | Rationale |
|---|---|---|
| Repeated fuzzy queries | Reuse one transducer and create a fresh cursor per query | Construction retains a provider in constant time; each cursor captures its own immutable revision. |
| Ordinary streaming | The facade iterator protocol | It materializes bounded owned values and supports early termination with deterministic close. |
| Maximum result throughput | The facade batch/reducer protocol | It amortizes the foreign boundary and keeps borrowed views inside one lexical lease. |
| Repeated phonetic matching | Compile a phonetic pattern once, then query or match repeatedly | Compilation is separated from traversal and the compiled handle is immutable. |
| Repeated phonetic rewriting | Parse or select a rule set once, then apply it repeatedly | Rule validation and allocation are amortized while each returned string remains independently owned. |
| Cross-project dictionaries | Pass the retained dictionary resource directly | The versioned resource preserves snapshot identity without serialization or shared Rust layout. |

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
