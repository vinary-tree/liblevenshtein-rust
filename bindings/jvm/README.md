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
