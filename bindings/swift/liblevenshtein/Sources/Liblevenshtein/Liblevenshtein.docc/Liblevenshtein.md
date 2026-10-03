# Liblevenshtein

Search a dictionary for nearby strings with reusable edit-distance automata,
streaming result cursors, and optional bounded caching.

## Overview

`Liblevenshtein` is the SwiftPM product for the liblevenshtein native engine.
It consumes a `VinaryTreeInterop.DictionaryResource` from a separate producer,
typically libdictenstein. The dictionary remains owned by that producer; a
``Transducer`` retains a native resource instead of copying every entry. Each
``QueryCursor`` retains the immutable dictionary revision visible at query
creation, so later dictionary mutations do not change an in-flight search.

The three query domains are Unicode text, arbitrary bytes, and unsigned
64-bit token sequences. A match retains its original domain in ``MatchTerm``.
No identifier or token value is reserved as a sentinel.

```swift
import Liblevenshtein
import VinaryTreeInterop

func suggestions(
    from dictionary: some DictionaryResource,
    for input: String
) throws -> [Match] {
    let transducer = try Transducer(dictionary: dictionary, algorithm: .standard)
    defer { transducer.close() }
    let cursor = try transducer.query(
        input,
        maximumDistance: 2,
        order: .distanceThenTerm
    )
    defer { cursor.close() }

    var matches: [Match] = []
    while true {
        let batch = try cursor.nextBatch(maximum: 256)
        if batch.isEmpty { return matches }
        matches.append(contentsOf: batch)
    }
}
```

The throwing batch API preserves recoverable native errors. The ordinary
`Sequence` interface is convenient when the caller accepts a trap on a late
native failure, because `IteratorProtocol.next()` cannot throw. Close a
cursor explicitly when iteration stops early; `deinit` is only a fallback.

For repeated queries, retain one ``Transducer`` and create a fresh cursor per
query. ``QueryCache`` optionally stores complete results under hard entry and
weight bounds. A cache is exclusive: create one per worker instead of sharing
it across concurrent callers. Inspect ``QueryCacheStats`` to measure hits,
admissions, evictions, and current residency.

``EditDistance`` provides standalone distances without a dictionary.
``PhoneticPattern`` and ``PhoneticRuleSet`` compile or parse reusable phonetic
operations; explicitly close these native resources after use.

## Topics

### Search and caching

- ``Transducer``
- ``QueryCursor``
- ``Match``
- ``MatchTerm``
- ``QueryCache``
- ``QueryCacheStats``

### Other automata and error handling

- ``EditDistance``
- ``PhoneticPattern``
- ``PhoneticRuleSet``
- ``LiblevenshteinError``
- ``Status``

### Selection and configuration

- ``Algorithm``
- ``QueryOrder``
- ``PhoneticRuleSetKind``
- ``OperationApplicability``
- ``UniversalVariant``
