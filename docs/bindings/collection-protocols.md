# Collection-protocol parity and native Rust idioms

Status: **approved design and implementation roadmap; the protocol adapters in
this document are not all shipped yet.** The per-language guides describe the
current API. This document defines the target, the performance model, and the
gates that must be satisfied before a guide may claim collection support.

The goal is broader than making a dictionary iterable. A Vinary dictionary may
store byte strings, Unicode scalar sequences, or `u64` sequences; a final key
may have no mapped value, or a present `u64` value; and a mutable producer uses
immutable query-start snapshots. Host collection interfaces must preserve those
semantics without forcing the native implementation through an FFI-shaped API.

## Design decision

The pure Rust API is the semantic and performance baseline. One generic,
snapshot-pinned entry traversal in `libdictenstein` serves every applicable
automaton and storage backend. The family ABI exposes that facility in optional,
batched form. Foreign facades then provide two products:

1. an ordinary host collection view whose iterator owns only host memory and
   therefore cannot leak a native snapshot when a loop exits early; and
2. an explicitly closeable streaming traversal for large dictionaries, with a
   lexical helper such as Java try-with-resources, Kotlin `use`, C# `using`, or
   a Python context manager.

The ordinary view materializes one immutable revision before returning its
iterator. The streaming view retains one immutable snapshot and leases bounded
batches. Both observe the same revision and lexical order. Membership remains a
direct dictionary lookup rather than an iteration scan.

### Native shape, shared laws

Adoption is an acceptance criterion. The shared implementation standardizes
laws, test fixtures, and native batches—not public spelling. Each facade must
use the names, protocols, generics, errors, lifecycle constructs, documentation
style, and package conventions its users already expect. Ordinary callers
should not need to understand handles, vtables, leases, status codes, or the C
ABI. Those remain available only through an explicitly labeled expert layer.

Representative "pit of success" syntax is `for entry in &dictionary` and
iterator combinators in Rust, enhanced `for`/`Set`/`Map` and streams in Java,
`for`/LINQ and `using` in C#, collection ABCs and `with` in Python,
`for...of` and explicit resource management in JavaScript, ranges in C++,
`range`-compatible iterator functions in Go, `Sequence` in Swift, and
`Enumerable#each` in Ruby. Naming and return types should follow the host
ecosystem even when that means the facades are intentionally not isomorphic.

![A single native snapshot traversal feeds optimized Rust iterators, a batched optional ABI, and leak-safe host collection adapters.](../diagrams/bindings/collection-view-snapshot-flow.svg)

## Confirmed current gaps

The existing Rust producer already has useful `iter*` methods and several
borrowed `IntoIterator` implementations, but the surface is not uniform across
all automata. Some persistent byte and character iterators collect the complete
result before yielding, only selected types implement `FromIterator`, and the
generic iterator types do not yet advertise all sound standard traits and size
hints. `liblevenshtein` query result types implement `Iterator`, but collection
construction and traversal belong primarily to `libdictenstein`.

The current foreign facades expose resource ownership and lookup operations,
not complete collection protocols:

| Surface | Shipped idiom | Collection-parity gap |
|---|---|---|
| Rust producer | Backend-specific `iter*`; borrowed `IntoIterator` on selected in-memory types; limited `FromIterator` | Uniform borrowed/snapshot/owned iteration, `Extend`/fallible construction, iterator metadata, all automata and unit domains |
| Rust consumer | Lazy query iterators and reducers | Consistent `FusedIterator`, truthful `size_hint`, borrowed/owned result forms, collection conversion documentation |
| Java/Kotlin/Scala | `AutoCloseable`; try-with-resources, `use`, and `Using` | No `Set`, `Map`, `Iterable`, `Iterator`, `Spliterator`, or stream view |
| Clojure | Explicit resource facade | No reducible/sequence abstraction with bounded native lifetime |
| .NET | `IDisposable` | No `IReadOnlySet`, `IReadOnlyDictionary`, `IEnumerable`, or async/cancellable stream |
| Python | `len`, membership, lookup, context manager | No `collections.abc.Set`/`Mapping` view or iterator |
| JavaScript/TypeScript | Disposable resource facade | No iterable/async-iterable collection view |
| Ruby | Closeable object | No `Enumerable` entry/key traversal |
| C++ | RAII wrapper | No standard range/view surface |
| Go | Explicit `Close` | No idiomatic iterator sequence or range helper |
| Swift | Explicit lifecycle wrapper | No `Sequence`/iterator view |
| Fortran, OCaml, Haskell, Lua | Explicit facade functions | No language-idiomatic fold/sequence/table traversal |

This table is a gap inventory, not a promise that every mutable host interface
is semantically valid. For example, Java `Map.put` cannot represent every
backend's persistence error contract, so mutable mapping is an explicit adapter
with documented exceptions rather than an unsafe claim that every dictionary is
a general-purpose `Map`.

## Native Rust baseline

### Generic capability traits

Add narrow traits in `libdictenstein` and implement them through shared zipper,
snapshot-root, and compact-graph machinery:

```rust,ignore
pub trait DictionaryEntries: Dictionary {
    type EntryValue: DictionaryValue;
    type Entries<'a>: Iterator<
            Item = DictionaryEntry<
                <Self::Node as DictionaryNode>::Unit,
                Self::EntryValue,
            >,
        >
        + FusedIterator
    where
        Self: 'a;

    fn entries(&self) -> Self::Entries<'_>;
}

pub trait SnapshotEntries: DictionaryEntries {
    type Snapshot: DictionaryEntries + Clone + Send + Sync + 'static;
    fn snapshot(&self) -> Self::Snapshot;
}
```

The exact associated types should be selected during implementation; they must
not force boxing or dynamic dispatch on monomorphic Rust callers. Separate
capability traits are preferred to one oversized trait so read-only, mutable,
mapped, bijective, substring, persistent, and unit-domain-specific automata do
not pay for unsupported operations.

Define one lossless entry type. Its mapped state must distinguish:

- key absent;
- key present without a value; and
- key present with a value.

That distinction already exists in the family ABI and must not collapse into a
single `Option<V>` lookup result when collection adapters need to differentiate
membership from mapped value. Use an enum such as `EntryValue<V> { Unit,
Value(V) }` within an emitted entry; absence is represented by no entry.

### Idiomatic construction and mutation

Provide the following wherever the backend semantics permit:

- `IntoIterator for &DictionaryType` over one immutable revision;
- `IntoIterator for SnapshotType`, allowing an iterator to own its snapshot;
- `FromIterator<K>` and `FromIterator<(K, V)>` for infallible in-memory
  constructors;
- `Extend<K>` and `Extend<(K, V)>` for infallible mutable in-memory backends;
- named `try_from_iter`, `try_extend`, and sorted variants for persistent or
  otherwise fallible backends, because `FromIterator` cannot report I/O or
  durability failure;
- `FromIterator` implementations that route through the optimized bulk builder,
  not a naïve per-key public insertion loop;
- `iter`, `keys`, `entries`, and `values` names with the same semantics across
  byte, scalar, and `u64` domains.

Do not implement `Index`: concurrent mutation and value cloning mean a stable
borrowed `&V` generally cannot be returned soundly. Do not implement `Deref` to
a standard collection or define `Ord`/`Hash` for mutable dictionaries merely to
look collection-like. Prefer explicit `get`, `contains`, and entry capabilities.

### Iterator laws and optimization

All generic iterators must be iterative and stack-safe, preserve deterministic
lexicographic unit order, capture exactly one immutable revision, and avoid a
lock whose lifetime spans user code. The fast path should:

- retain a compact snapshot graph/root once;
- keep a reusable traversal stack and path arena;
- reconstruct or copy a key only for a final node;
- borrow edge batches or native slices where lifetimes permit;
- reuse buffers for visitor/fold APIs;
- provide `FusedIterator` after exhaustion;
- provide a truthful lower and upper `size_hint`; use `ExactSizeIterator` only
  when the captured revision supplies O(1) cardinality and each remaining entry
  is counted without hidden filtering;
- avoid `DoubleEndedIterator` unless reverse traversal is native and does not
  materialize the dictionary;
- make `collect` reserve from a known snapshot cardinality; and
- keep the monomorphic Rust path free of ABI calls, callbacks, atomics per edge,
  and virtual dispatch.

Offer an allocation-reusing visitor/fold in addition to owned-key iteration.
Owned `Vec<Unit>`/`String` items remain the ergonomic default because an
`Iterator` cannot safely yield a reference into a buffer it mutates on the next
call. A lending iterator may be evaluated behind a separate API, but must not
require an unstable Rust feature or infect the ordinary collection surface.

### Query-result idioms

For `liblevenshtein`, audit every query iterator and adapter for:

- `FusedIterator` where exhaustion is terminal;
- exact or conservative `size_hint` implementations;
- `IntoIterator` on query/result owner types where it adds no extra allocation;
- explicit traversal-order documentation;
- `collect::<Vec<_>>()`, reducer, and callback paths that share one engine;
- cancellation-by-drop with no lock held and bounded destructor work; and
- preservation of query-start snapshot semantics for borrowed and owning forms.

Do not add `ExactSizeIterator` to a lazy fuzzy query: accepted cardinality is
unknown without doing the search. Optimize the reducer/visitor path for callers
that do not need owned result materialization.

## Shared native traversal and ABI

### One implementation, optional acceleration

First build a backend-generic `SnapshotEntryTraversal` over the same native
snapshot cursor/compact graph used by optimized query traversal. Backend
specialization is allowed only behind capability hooks and benchmarks: for
example, a persistent ARTrie can traverse its immutable overlay directly while
a provider that exposes only the graph interface uses the shared compact-graph
walker. All applicable automata must pass the same laws.

Then add an optional, versioned family interface (working name
`vt.dictionary.entries.v1`) that drains entries in batches. A batch contains
fixed-size descriptors plus contiguous unit and optional-value arenas. Every
descriptor carries offsets and lengths; the cursor owns its snapshot; one batch
is leased at a time; release is mandatory before the next batch; cancellation
and limits are explicit. Existing `node_edges` traversal remains the compatible
fallback.

The interface must include:

- unit and value domain descriptors;
- deterministic order declaration;
- snapshot identity/revision when supported;
- optional exact cardinality;
- caller-selected maximum batch/arena bounds;
- `next_batch`, `release_batch`, `reduce`, cancel, and close operations;
- the existing bounded status/error model; and
- feature negotiation so older producers and consumers continue to interoperate.

No facade should perform one FFI call per edge or key. Native facades drain
batches; managed expert paths reduce a borrowed batch before release; ordinary
views materialize with geometric growth or exact reservation and then close the
cursor.

## Host-language surfaces

### Java, Kotlin, Scala, and Clojure

Provide explicit views rather than making the closeable dictionary object
itself inherit one ambiguous collection type:

```java
try (Dictionary dictionary = Dictionary.open(...)) {
    Set<String> stable = dictionary.asStringSet(); // host-owned snapshot
    consume(stable);

    try (DictionaryEntryStream<String, OptionalLong> entries =
             dictionary.stringEntries()) {
        entries.forEachRemaining(
            entry -> consume(entry.key(), entry.value()));
    }
}
```

- Java: immutable `Set<String>`/typed key-set and
  `Map<K, OptionalLong>` snapshot views; closeable `Iterator`/`Spliterator` and
  sequential `Stream` for bounded streaming. `trySplit` is enabled only when a
  snapshot can partition at root subtrees without duplicate traversal.
- Kotlin: `Set`, `Map`, `Sequence`, and `use` extensions without duplicating JNI
  logic.
- Scala: immutable `Set`/`Map`, `Iterator`, and `Using.resource` adapters.
- Clojure: reducible/foldable view; `reduce` uses the native batch reducer so it
  does not first create a Java collection.

The legacy Java `Dawg` extending `AbstractSet<String>` establishes desired set
ergonomics, not a requirement to couple native lifetime to Java iterator
lifetime. `AutoCloseable` remains mandatory; `Cleaner` remains a last-resort
safety net, never the normal lifecycle.

### .NET

Expose immutable `IReadOnlySet<TKey>` and `IReadOnlyDictionary<TKey,
OptionalUInt64>` snapshot views, plus `IEnumerable<Entry>` on a materialized
view. Use `IDisposable`/`using` for a closeable batch enumerator. Add
`IAsyncEnumerable` only if traversal actually becomes asynchronous; wrapping a
synchronous native call in a task would add overhead without improving
concurrency.

### Python

Expose read-only `collections.abc.Set` and `Mapping` views with `__len__`,
`__contains__`, `__iter__`, `keys`, `items`, `values`, and `get`. Iteration over
the ordinary view is host-owned. A separate context-managed streaming iterator
supports large dictionaries. Release the GIL around native batch traversal and
reacquire it only while constructing Python objects or invoking Python code.

### JavaScript, TypeScript, and ClojureScript

Expose `[Symbol.iterator]` on immutable materialized views and
`[Symbol.asyncIterator]` only on WASI/worker paths that are genuinely
asynchronous. A closeable native iterator should support explicit resource
management where the runtime implements it, plus `try/finally` everywhere.
TypeScript types distinguish keys, entries, and optional mapped values.

### C++, Go, Swift, Ruby, and the remaining facades

- C++: borrowed/snapshot ranges with an input iterator; a sized range only when
  exact snapshot cardinality is available; RAII cursor ownership.
- Go: callback/range iterator sequences and explicit cancellation. Keep slice
  lifetime rules obvious; copy by default, borrowed batch only in an expert
  callback.
- Swift: `Sequence` for host-owned snapshots and a closeable streaming
  iterator; use `withExtendedLifetime` around native batch borrows.
- Ruby: `Enumerable` with `each` returning an enumerator when no block is
  supplied; ensure `break` closes a streaming cursor.
- OCaml and Haskell: lazy sequence/list-fold adapters whose resource-scoped
  variants cannot escape their bracket/region.
- Lua: `pairs`-style closure backed by a materialized view, plus an explicit
  closeable streaming iterator.
- Fortran: counted batch and callback/fold procedures rather than pretending a
  dynamic dictionary is a native array.

## Snapshot, mutation, and error semantics

Every iterator sees exactly the revision captured at iterator/view creation.
Concurrent inserts, removes, value updates, clear, compaction, checkpoint, or
reopen cannot alter its remaining entries. A mutable collection adapter applies
changes to the live dictionary, while an already-created iterator remains
stable.

Host protocols must translate errors without weakening them:

- infallible lookup protocols may wrap only operations whose backend contract
  is infallible;
- persistent mutation adapters use the host's documented exception/result
  channel;
- partial bulk mutation reports its atomicity explicitly;
- invalid UTF-8 is never silently discarded from a byte-domain collection;
- length conversion checks the host integer range; and
- a close operation is idempotent, while use-after-close is deterministic.

## Performance and correctness gates

Collection support is complete only after all of these gates pass:

1. **Generic Rust laws:** every applicable in-memory and persistent automaton,
   byte/scalar/`u64` domain, unit/mapped value domain, empty key, shared suffix,
   and term-only entry matches a `BTreeMap`/`BTreeSet` reference model.
2. **Snapshot laws:** mutation and compaction stress tests prove revision
   stability, cursor independence, early-drop reclamation, and no lock held
   across consumer code.
3. **Language conformance:** common fixtures test membership, length, order,
   iteration, tri-state values, early loop exit, disposal, exception mapping,
   and concurrent mutation in each facade.
4. **Ecosystem acceptance:** API review checks the relevant language's standard
   collection contracts, naming conventions, package tooling, documentation
   examples, resource-management idioms, and static-analysis expectations. A
   facade that merely transliterates the C ABI does not pass.
5. **Complexity:** snapshot capture remains O(1); traversal is O(nodes + output
   units); auxiliary memory is O(depth + leased batch) for streaming and
   O(output) only for an explicitly materialized view.
6. **Allocation and call budgets:** criterion counters record allocations per
   key/unit and ABI calls per batch; no edge-at-a-time FFI path is accepted.
7. **Performance parity:** compare direct Rust iteration, generic Rust
   traversal, C batches, and every managed facade on the same admitted host.
   Report median, MAD, paired ratios, confidence intervals, throughput, peak
   memory, and early-cancellation cost under the repository's
   [benchmark methodology](../benchmarks/optimization-and-profiling-methodology.md).
8. **Regression thresholds:** each language manifest declares its supported
   protocol and evidence. CI fails on semantic drift; scheduled benchmarks flag
   statistically and practically significant throughput or allocation
   regressions.

Profiles must separate traversal, key materialization, value conversion, FFI
crossings, and host allocation. Use headless AMD uProf, `perf`, and Heaptrack as
described by the benchmark methodology. Optimize only after a causal experiment
identifies a dominant component.

## Implementation work packages

1. **Rust contract and law suite.** Define the lossless entry type and narrow
   capability traits; inventory every automaton; add compile-time trait tests,
   reference-model properties, and snapshot/early-drop laws.
2. **Generic native traversal.** Implement the monomorphic snapshot-pinned
   walker with reusable stack/path storage; route all applicable byte, scalar,
   and `u64` backends through it; retain measured specialization hooks.
3. **Rust idiom completion.** Standardize `iter`/`keys`/`entries`/`values`,
   borrowed and snapshot-owning `IntoIterator`, optimized `FromIterator`,
   `Extend`, fallible variants, iterator metadata, and query-result idioms.
4. **Batched ABI extension.** Specify, model, fuzz, and implement the optional
   versioned entry-batch interface with graph fallback, bounded leases, and
   compatibility tests across repository versions.
5. **Managed collection foundations.** Generate the shared materializer,
   closeable cursor, error translation, cardinality reservation, and early-exit
   cleanup primitives for each runtime family.
6. **Parity-first facades.** Land Java/Kotlin/Scala/Clojure and .NET collection
   views first, including Java `Set` parity and deterministic RAII idioms.
7. **Dynamic and native facades.** Land Python, JavaScript/TypeScript,
   Ruby, C++, Go, and Swift; then Fortran, OCaml, Haskell, and Lua where their
   package tier applies.
8. **Evidence and rollout.** Run the correctness, profiler, allocation,
   scalability, and cross-language matrices; update every guide and
   `bindings/api.json` only for protocols whose gates pass; publish migration
   and compatibility notes.

Each work package is independently reviewable. No binding may introduce a
private graph walker or change snapshot/value semantics to obtain surface-level
idiomaticity. Specialization must show a repeatable, admitted benchmark gain and
retain the common law suite.
