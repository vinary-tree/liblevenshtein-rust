# Optimization propagation across automata and dictionary backends

## Purpose and scope

This document records where the Java-parity campaign's accepted optimizations
apply, where a representation-specific adaptation is required, and where an
optimization is inapplicable. It covers the public query surfaces in
`liblevenshtein-rust`, every production `DictionaryNode` implementation in
`libdictenstein`, the byte, Unicode-scalar, and `u64` unit domains, and the
native and resource-backed execution paths.

The governing rule is semantic and mechanical, not nominal: an optimization is
shared when its required invariant is shared. A backend-specific implementation
is used only when the storage or scheduling invariant differs. Compatibility
defaults remain available on public traits, while production backends override
the hot seam with their native representation.

## Terminology

- **Dictionary node** means one state of a term-indexing graph implementing
  `libdictenstein::DictionaryNode`.
- **Unit domain** means the edge-label type: byte (`u8`), Unicode scalar
  (`char`), or token identifier (`u64`).
- **Native path** means a monomorphized Rust query over a concrete dictionary
  node.
- **Resource path** means a query over a retained `VtResource` and the versioned
  `VtDictionaryVTable` C application binary interface (ABI).
- **Directly reusable** means one generic implementation is used without a
  representation conversion.
- **Adapted** means the causal idea is retained but the storage-specific kernel
  differs.
- **Inapplicable** means the optimization's precondition is false; forcing it
  would add work or change semantics.

## Shared execution architecture

The accepted query optimizations are deliberately layered:

```text
native DictionaryTraversalRoot          retained foreign snapshot
       |                                     |
       |                           vt.dict.graph.v1, when available
       +----------------------+--------------+
                              |
                              v
              TraversalSession<DictionaryNode>
       one owner + session-selected copy-only cursor queue
                              |
                              v
                 query-surface scheduler
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
       packed Standard DFA       dense positional target table
       exact-cost u64 lanes       all unit-cost algorithms
                  |                       |
                  +-----------+-----------+
                              |
                              v
                   match/value materialization
```

`TraversalSession` is the generic ownership boundary. An immutable native or
foreign graph keeps one retained owner and queues one-word copy cursors. Its
cursor union carries either a dense graph position or the backend's associated
native cursor; the session mode is the sole discriminant, so no mode tag is
repeated in frontier entries. DynamicDAWG native cursors preserve `NonNull`
provenance without integer conversion, while DAT cursors remain dense. The
graph field precedes the owner field and is therefore dropped first. A backend
with native stable cursors supplies the same contract without a graph
conversion. The compatibility arm stores owned nodes in a query-local
recycling arena and consumes each slot once. When an owned or graphless foreign
node advertises efficient page addressing, its DFS frame retains the immutable
parent and constructs only one bounded page of accepted child handles at a
time; nodes without that capability preserve the eager compatibility path.
Query schedulers therefore do not duplicate backend ownership or paging logic.

The resource adapter validates an optional compact graph once and routes it
through the same session. Providers without that interface retain the fused
callback and immutable-node-cache fallback. Transition optimizations are
selected independently: eligible short Standard queries use the packed DFA;
Transposition, Merge-and-Split, unrestricted Damerau-Levenshtein, and
ineligible Standard queries share the dense positional engine.

## Accepted optimization matrix

| Optimization | Classification | Shared implementation | Required invariant |
|---|---|---|---|
| Freeze-once sorted minimal construction | Direct for valueless and mapped `DynamicDawg` byte, char, and `u64` constructors | Unit-generic sorted builder in `libdictenstein::dynamic_dawg::core` | Input keys are nondecreasing; duplicate mapped terms retain the last value; one immutable graph is published after minimization |
| Sort then build minimally for unordered bulk input | Direct for valueless and mapped `DynamicDawg` byte, char, and `u64` constructors | Each public unordered constructor delegates to the same unit-generic sorted builder after a stable sort when values are present | Bulk ownership permits reordering; stable duplicate order preserves last-value-wins semantics; concurrent incremental insertion retains its existing path-copying semantics |
| Empty binding-batch construction | Direct for byte, Unicode-scalar, and `u64` binding-owned `DynamicDawg` resources | The existing batch mutation detects an empty backend, validates the complete batch, detects nondecreasing order, and publishes one graph built by the same unit-generic kernel | The safe Rust API validates the complete batch and is atomic; the legacy C ABI freeze-builds the validated prefix on descriptor failure to preserve its prefix-applied contract; resource identity remains stable; older snapshots retain their roots; nonempty batches keep incremental update semantics; valueless terminals remain `None` rather than sentinel values and may therefore be minimized |
| Fast non-cryptographic merge registry | Direct for all sorted DAWG unit domains | Private `FxHashMap` registry in the generic build kernel | Registry keys are internal structural signatures, not attacker-controlled hash-table API inputs |
| Static double-array-trie construction | Adapted for byte and char double-array tries | Double-array-specific arena construction and placement | Array base/check placement is not a DAWG-equivalence registry and cannot reuse Daciuk minimization |
| Borrowed edge visitation | Direct for every production node backend | Generic `DictionaryNode::for_each_edge` seam with backend overrides | Backend can enumerate native edge storage while cloning only the returned child handle |
| Fused finality plus edge visitation | Direct for every query surface that always expands a popped node; compatible fallback elsewhere | `DictionaryNode::visit_edges_and_finality` | The query scheduler inspects finality and outgoing edges in the same logical node step |
| Predicate-first child materialization | Direct for every production node backend and applicable query scheduler | Generic `DictionaryNode::filter_map_edges` and `filter_map_edges_and_finality` seams with storage-specific overrides | The label projection is pure with respect to dictionary storage; a child handle is created exactly once only when projection accepts the label; traversal order and finality observation are unchanged |
| Bounded complete-query memoization | Opt-in and unit/result generic for every repeat-query caller that can supply a dictionary revision | `VersionedQueryCache<V, W>` shares TinyLFU admission, SIEVE victim selection, hard entry/weight bounds, collision-exact query keys, immutable `Arc<[V]>` results, and the allocation-free circular-span transaction planner | The caller advances the revision after any result-affecting mutation; the miss closure computes the exact complete result; approximation changes only residency; one owner or caller-selected sharding preserves the synchronization-free hot path |
| Per-query characteristic-vector cache | Direct for byte, char, and `u64` unit-cost, affine-gap, and f64-weighted queries | Shared `CharacteristicCache<U>` behind `CachedUnitTransitions<U>` and `CachedF64Transitions<U>` | Query is immutable for the iterator lifetime and equality of units determines the characteristic vector; operation costs affect successors, not unit equivalence |
| Class-only characteristic label cache | Direct for byte, char, and `u64` unit-cost queries | `CharacteristicCache<U>` maps hot labels to compact class IDs and owns each pattern once in a central class table | Exact label equality is retained for direct-table collision checks; a pattern is fetched only when the generated transition table misses |
| Epsilon-closed queued states | Direct for all built-in unit-cost, affine-gap, and f64-weighted variants | Generic integer `AutomatonVariant` kernel and the corresponding weighted kernel | Initial and enqueued states are epsilon-closed exactly once |
| Compact generated-state frontiers | Direct for queued built-in unit-cost ordinary, ordered, priority, ranked-value, value-filtered/value-yielding, and prefix-DFS iterators | `GeneratedStateId` and canonical transition rows in `CachedUnitTransitions<U>` | IDs and rows are query-local and append-only; sentinels cannot collide with a valid ID; collision buckets compare canonical position slices; public/stateful navigation may materialize a `State` adapter |
| Snapshot-scoped cursor traversal | Direct for every scheduler whose dictionary walk does not expose child handles; adapted for schedulers that must materialize an accepting node; lazy owned traversal for providers whose first cursor requires a full projection | Generic `TraversalSession<N>` plus session-discriminated `TraversalCursor<N::SnapshotCursor>`; `DictionaryNode::snapshot_cursor_requires_full_projection` communicates capture complexity without disabling cursor snapshots for amortizing resource producers; `DeferredNodeSource<N>` delays accepting-node materialization | One retained revision owns every cursor; graph handles drop before that owner; the session mode proves the active union member; native pointer cursors preserve provenance; an ordinary lookup never pays O(dictionary) merely to obtain its first cursor; owned-fallback slots are consumed or explicitly discarded exactly once |
| Compact priority-query paths | Direct for priority traversal over byte, Unicode-scalar, and `u64` dictionaries | One append-only `ParentPathNode<U>` arena, constant-size `ParentPathKey`, and comparator-aware `ContextHeap<T>` shared with generic result-path reconstruction | Equal logical prefixes share arena keys; the cached first unit resolves cross-root comparisons in O(1); equal first units use the exact parent-chain comparator; accepted results materialize once; the legacy cloned-vector heap remains a benchmark control |
| Compact generic parent-path metadata | Direct for every parent-arena path strategy over byte, Unicode-scalar, and `u64` dictionaries | `ParentPathKey(u32)` and a checked `u32` depth are shared by value, ranked-value, ordered, priority, language, and phonetic traversal | `u32::MAX` remains the root sentinel; append checks reject identifier exhaustion; depth growth is checked; reconstruction converts to `usize` only at allocation boundaries |
| Lazy bounded DFS edge pages | Direct for prefix DFS over compact graphs, native cursors, and owned or graphless foreign nodes advertising efficient page addressing; rejected for exhaustive subsequence DFS | Generic `DfsNodeEdges<N>` selects an eager compatibility source or a `SmallVec`-backed cursor page; DATs, slice-backed suffix automata, PathMap, and graphless foreign providers expose efficient paging, while transparent decorators forward the capability | Finality and total out-degree are captured by a zero-capacity metadata probe; edge order and total count remain stable for the snapshot; one page holds at most eight descriptors without allocation; SCDAWG and overlay/persistent nodes whose merged edge views are not index-addressable remain eager rather than rescan on every page; subsequence retains eager expansion because it has no automaton-dead edge pruning |
| Packed exact-cost edit lanes | Direct for byte, Unicode-scalar, and `u64` Standard queries whose `query_length + 1` lanes at distances 0–3 fit in one `u64` | `PackedEditLaneLayout` stores exact-cost lanes and computes one-to-three-deletion closure with fixed masks and shifts | Standard unit costs and bounded width; prefix sinks remain explicit; all other cases fall back to the positional engine |
| Lazy compact packed DFA | Direct for eligible unrestricted Standard queries on all three unit domains | `PackedStandardDfa<U>` interns reached packed frontiers and stores row-major targets by exact label-equivalence class | The query is immutable; exact label classes preserve collisions; only states reached by the concrete dictionary walk are generated |
| Source-row-local class-zero result reuse | Direct for Standard, OSA, and merge/split packed DFAs across direct, ordered, priority, ranked-value, and value-filtered schedulers | `ExactLabelDfaRow` retains the first class-zero result for the current sibling-edge expansion | Class zero denotes every label absent from the immutable query; the cached dead sentinel and live state ID are exact for one fixed source row and are discarded before the next dictionary node |
| Static prepared packed-row dispatch | Direct for Standard, OSA, and merge/split in every representation-erased scheduler; the direct query kernel already selects a concrete type for the query lifetime | `PreparedUnitCostRow<U>` plus `with_prepared_unit_cost_row!` centralizes once-per-node selection without a vtable | The macro monomorphizes each expansion body for its concrete row; positional and profiling-control fallbacks preserve the representation-erased engine and exact transition semantics |
| Flat dense positional target table | Direct for Standard fallback, Transposition, Merge-and-Split, and unrestricted Damerau-Levenshtein across all query schedulers and unit domains | `DenseGeneratedTargets` owns one fixed-stride row-major matrix beside stable boxed canonical position slices, plus sparse overflow for custom-policy classes | Generated state IDs are dense and query-local; exact classes remain within the query-derived stride; an arbitrary substitution policy cannot widen every existing row |
| Bulk state-position copy | Direct for integer and f64 states with contiguous `Copy` positions | `extend_from_slice` in each state representation | Position layout is contiguous and copying preserves order and bit identity |
| Pinned immutable provider snapshot | Direct for all resource-backed unit domains and algorithms | `ResourceTransducer::snapshot` and provider snapshot reuse | Provider honors immutable revision and stable node-identifier lifetime contracts |
| Revision-memoized producer snapshot | Direct for DynamicDawg byte/char/`u64`, persistent ARTrie byte/char/`u64`/vocabulary, DAT byte/char, and SCDAWG byte/char resources | Unit-agnostic `SnapshotMemo` owns one helpably replaceable single-flight cell per revision; DynamicDAWG keys it from one atomically retained `(root, count, revision)` graph generation, while heterogeneous mutable producers use an RAII writer announcement and invalidate conservatively after any mutation error that could have partially published | DynamicDAWG capture has one `GraphVersion` load as its linearization point and needs no outer writer handshake; other mutable producers validate the active-writer count **before** the final memo-revision load after immutable root/count capture; acquiring a newly published root carries the earlier writer announcement, and acquiring the final writer withdrawal carries the preceding revision advance; writers never wait for snapshotters, takeover bounds damage from a stalled cold initializer, capture is lock-free system-wide and reader-obstruction-free rather than reader-wait-free under unbounded mutation, and immutable backends retain revision zero indefinitely |
| Cross-resource immutable node cache | Direct for graphless providers advertising `vt.snapshot.id.1`; private-cache fallback for older graphless providers; bypassed by providers exposing a validated immutable graph | Process-wide weak node-cache registry keyed by `(producer, revision)` and provider lineage; graph-backed providers instead share the per-resource immutable graph memo | The producer guarantees identity uniqueness and revision immutability; the weak registry does not extend snapshot lifetime; a captured graph is validated once and needs no node cache; absence or rejection of optional interfaces preserves the callback fallback |
| Chunked append-only provider arena | Direct for every producer snapshot node type | Unit- and backend-generic `NodeArena<N>` over `HybridOnceBoxSlots` with a bounded dense prefix, sharded sparse overflow, fallible growth, atomic ID reservation, and per-edge publication cells | IDs are immutable for the snapshot lifetime; reads, growth, and publication are lock-free; a failed reservation consumes no ID; releasing the last snapshot synchronously reclaims its slots |
| Hybrid immutable consumer cache | Direct for every resource-backed node domain | `HybridOnceBoxSlots` shares one bounded dense prefix and sharded sparse overflow across unit domains | Cached entries are immutable and append-only for the query-owner lifetime; losing publications are reclaimed immediately; provider/vtable lineage prevents cross-provider identity aliasing |
| Owner/key split and query-local faults | Direct for all resource cursors | One retained `Arc<Provider>` in `QueryCursor`, copy-only `ProviderRef`/foreign-node keys, and `AtomicTakeBox<BindingError>` per cursor | Cursor field drop order outlives every non-owning key; provider allocation never moves; each query observes only its own callback failure |
| Atomic persistent root cardinality | Direct for persistent ARTrie byte/char/`u64`/vocabulary overlays | Generic counted `AtomicNodePtr<K,V>` publishes immutable root and exact term count in one ArcSwap CAS | Every membership-creating/removing CAS applies `+1`/`-1`; value-only and structural publications apply zero; prebuilt roots seed the count once; root and length are captured from the same revision |
| Stack-paged provider edges | Direct for all foreign node domains, including graphless snapshot providers | Generic `ForeignNode<U>` implements the same `DictionaryNode` paging seam used by native backends and fills a fixed stack page directly from `node_edges` | ABI edge descriptor has a fixed representation; edge order/count are immutable for the captured snapshot; zero-capacity metadata probes construct no children; requested pages are capped at the recommended stack size |
| Fused provider node callback | Direct for providers advertising the optional interface | `VtDictionaryVisitVTable::node_visit` | Provider can return finality and one edge page under one validation/lock operation |
| Immutable provider node cache | Direct for all resource-backed queries | Provider-wide node-ID cache shared by snapshot cursors | Node ID, finality, and edges are immutable within the retained snapshot |
| Direct immutable resource snapshot graph | Direct for byte, Unicode-scalar, and `u64` providers advertising `vt.dict.graph.v1`; optional fallback elsewhere | `VtDictionaryGraphVTable` exposes one immutable node/edge view; producer and consumer revision memos publish their projection/import once, then every scheduler uses `TraversalSession` | The view belongs to a retained immutable snapshot; ranges, targets, label order/domain, flags, reserved bytes, root, and value cursors validate before traversal; mapped values resolve through checked graph-local cursors only at accepting nodes |
| Packed graph finality | Direct for every native or resource scheduler using `SnapshotTraversalGraph<U, H>` | Finality occupies the high bit of the existing 64-bit edge-range descriptor | One node has fewer than $`2^{31}`$ outgoing edges; construction validates the range before publication; the descriptor remains eight bytes |
| Borrowed result-batch consumption | Direct at the JVM cursor API; rejected as the parity-harness default by H-O27 | `QueryCursor.forEachBatch` plus descriptor/byte views; the ordinary harness path continues to materialize managed matches | Consumer finishes reading a borrowed batch before the callback returns; callers choose it for lazy decoding or allocation control rather than assuming a throughput win |
| Lexical JVM whole-query drain | Direct for every JVM transducer algorithm and string, byte, packed-`u64`, and phonetic query form | `NativeQueryBatches` centralizes one confined arena and batch lease; `Transducer.forEachMatch` supplies typed public overloads | The callback is synchronous, each delivered `Match` owns its managed data, and arena/batch cleanup remains deterministic when either native traversal or the consumer throws |

## Dictionary backend inventory

All eighteen production `DictionaryNode` implementations in `libdictenstein`
override `for_each_edge`; none falls through to the compatibility
implementation that boxes `edges()`. The resource node and all nine transparent
node decorators in `liblevenshtein-rust` also preserve the inner hot seam. The
decorators share the monomorphized helpers in `dictionary::node_adapter`, so an
eviction, phonetic-normalization, or resource wrapper cannot accidentally
reintroduce allocation. The inventory is:

| Family | Unit domains and node representations | Query optimization status | Construction status |
|---|---|---|---|
| Dynamic DAWG | `DynamicDawgNode`, `DynamicDawgCharNode`, `DynamicDawgU64Node` | Direct: predicate-first borrowed/fused traversal and generic unit transition cache | Direct: sorted minimal and unordered sort-plus-minimal builders for set, mapped, and empty binding-batch dictionaries |
| Double-array trie | `DoubleArrayTrieNode`, `DoubleArrayTrieCharNode` | Direct: predicate-first borrowed/fused traversal and generic unit transition cache | Adapted: static double-array placement builder |
| Path map | `TrieRefNode`, `TrieRefNodeChar` | Direct: predicate-first borrowed/fused traversal, index-addressed lazy DFS pages, and generic unit transition cache | Inapplicable: path-map persistence and snapshot ownership are not minimal-DAWG construction |
| Suffix automaton | `SuffixNodeHandle`, `SuffixNodeCharHandle` | Direct: predicate-first borrowed/fused traversal, lazy bounded DFS pages, and generic unit transition cache | Inapplicable: suffix-link construction indexes substrings rather than a finite term language |
| SCDAWG | `ScdawgNodeHandle`, `ScdawgCharNodeHandle` | Direct: predicate-first borrowed/fused traversal and generic unit transition cache; eager DFS is retained because the merged compact edge view is not index-addressable | Inapplicable: compact suffix-DAWG topology and end-position semantics differ from term-DAWG minimization |
| Persistent suffix automaton | byte and char persistent node handles | Direct: predicate-first borrowed/fused traversal and generic unit transition cache; eager DFS avoids repeated persistent-map scans | Inapplicable: persistence, suffix links, and durable publication are defining invariants |
| Persistent suffix tree | byte and char persistent node handles | Direct: predicate-first borrowed/fused traversal and generic unit transition cache; eager DFS avoids repeated compressed-edge scans | Inapplicable: compressed suffix edges and durable arena ownership differ |
| Persistent SCDAWG | byte and char persistent node handles | Direct: predicate-first borrowed/fused traversal and generic unit transition cache; eager DFS avoids repeated merged-view scans | Inapplicable: persistent compact-suffix topology differs |
| Persistent ARTrie overlay | generic key-encoding overlay node | Direct: predicate-first borrowed/fused lazy traversal and generic unit transition cache for supported key encodings; ordinary queries avoid the O(dictionary) dense cursor projection, while FFI snapshots may construct and amortize that projection deliberately | Inapplicable: adaptive-radix layout, transactional publication, and persistence replace finite-language minimization |
| Transparent decorators | `AgeNode`, `CostAwareNode`, `LfuNode`, `LruNode`, `LruOptimizedNode`, `MemoryPressureNode`, `NoopNode`, `TtlNode`, and `PhoneticNormalizedNode` | Direct: generic child wrapping delegates borrowed, fused, and predicate-first visitation; mapped decorators delegate `value_at_final` | Inapplicable: decorators do not own construction topology |
| Foreign resource | generic `ForeignNode<U>` | Direct: compact snapshot graph when advertised; graphless providers inherit lazy bounded DFS paging directly from `node_edges`, plus predicate-first fused visitation and immutable node caching | Inapplicable: the provider owns construction |

The persistent byte, char, vocabulary, and `u64` ARTrie public types reach the
query engine through the overlay node implementation; compact and
prefix-compatible `u64` representations retain their encoding-specific
construction and traversal adapters.

The persistent suffix-array helper is a construction component rather than a
`DictionaryNode`: it already uses prefix-doubling over integer ranks and feeds
the persistent suffix indexes. DAWG minimization and node visitation are
therefore inapplicable to that helper, while the suffix automaton/tree/SCDAWG
nodes built from its output inherit the traversal seams listed above.

## Automaton and query-surface inventory

### Unit-cost integer variants

Eligible short Standard Levenshtein queries use `PackedStandardMachine<U>`:
one-word exact-cost lanes feed a lazily generated dense deterministic finite
automaton (DFA). Standard fallbacks, optimal string alignment (the public
`Transposition` variant), merge-and-split, and unrestricted
Damerau-Levenshtein share `CachedUnitTransitions<U>`. That engine owns the
characteristic-class cache, stable canonical generated-state slices, one flat
row-major target table, and the monomorphized `AutomatonVariant` kernel. Queued
unit-cost schedulers retain only a `UnitCostFrontier`, whose representation is
chosen once per query; the schedulers do not branch on the automaton algorithm
or duplicate the transition logic. The materializing adapter remains available
to stateful navigation APIs. Affine-gap queries share characteristic caching
and the epsilon-closed queue invariant while supplying their exact fixed-point
parameters to the statically dispatched `AffineV` kernel. The following public
query surfaces inherit this selection for their unit-cost variants:

- breadth-first traversal and value-yielding traversal;
- ordered, prefix-ordered, ranked-value, and priority traversal;
- value-predicate and value-set filtering;
- prefix-pruned depth-first traversal; and
- zipper traversal (with a materialized state at the navigation boundary).

Match-mode wrappers, filtered-ordered wrappers, and resource cursors delegate to
one of these iterators and therefore inherit the same kernel.

The JVM lexical drain sits above this dispatch rather than inside an automaton
implementation. Consequently, standard Levenshtein, optimal string alignment,
merge-and-split, unrestricted Damerau-Levenshtein, and every specialized
transducer that reaches the common native query-batch ABI share one arena
lifetime and callback loop. No algorithm duplicates the foreign-memory
ownership logic, and adding a new automaton does not require another drain
implementation unless it introduces a genuinely different query descriptor.

### Specialized state machines

| Surface | Shared optimizations | Deliberately separate mechanism |
|---|---|---|
| f64 weighted query | Borrowed/fused node visitation, shared characteristic caching, epsilon-closed queued states, and bulk contiguous state copy | `CachedF64Transitions<U>` retains the weighted successor/pruning kernel because operation costs remain f64 values |
| Contextual query | Borrowed/fused node visitation | Each child column depends on contextual cost callbacks and prefix context |
| Language-product query | Copy-cursor traversal; mapped resource queries use the compact graph and resolve only accepted final values | The frontier is a product of edit and language states, so a unit-only cached transition is insufficient |
| Character phonetic query | Copy-cursor traversal, including the compact resource graph when children are expandable; query-lifetime mode selection retains only the incremental language iterator or the articulatory scanner | Candidate admission uses the phonetic product and optional articulatory costs; the articulatory scanner keeps right-sized local queue/path storage so borrowed custom node types do not acquire a `'static` API bound |
| Byte phonetic query | Inherits language-query cursor traversal | Byte phonetic traversal delegates to the language-product iterator |
| Subsequence query | Shared `DfsNodeEdges` eager source and fused node inspection during explicit DFS-frame construction | It has no edit-distance state transition or automaton-dead edge pruning; the measured canonical workload exhausts most dictionaries, so paged refills regress and are deliberately not selected |
| Generalized and universal automata | Bulk contiguous state operations where representation permits; their existing subsumption kernels remain intact | They expose different state algebras and are not routed through the built-in `AutomatonVariant` transition function |
| Quantized time-series trie | Borrowed/fused dictionary visitation plus the shared unit transition cache for its Levenshtein candidate filter | Elastic DTW, ERP, Fréchet, MSM, and TWED walkers use kernel-specific dynamic-programming columns; they inherit borrowed edge visitation but not string characteristic vectors |
| WallBreaker/SCDAWG query | Borrowed forward-edge visitation during bidirectional extension | Pigeonhole splitting and bounded whole-candidate verification do not maintain a queued Levenshtein `State` |
| Hybrid n-gram/Jaro-Winkler matcher | Inapplicable | Its inverted n-gram index and pairwise similarity stages do not traverse `DictionaryNode` or construct automaton states |
| Incremental, memoized, online, and token phonetic NFAs | Indirect where composed with a dictionary-backed phonetic/language query | Standalone stream matchers consume an input stream rather than a term dictionary, so dictionary construction and node visitation do not apply |

### Conditional expansion exceptions

The ordinary distance-ordered iterator conditionally suppresses child expansion
when a final node is re-observed in a distance bucket that has already passed.
It therefore retains separate finality and edge operations for that branch;
unconditionally invoking the fused method would enqueue descendants that the
existing scheduler intentionally skips. Prefix-ordered traversal always expands
and uses the fused method. Resource-backed ordered queries still benefit from
the immutable node cache, so a repeated finality observation does not repeat a
provider traversal callback.

Phonetic traversal similarly uses a finality-only observation at the maximum
depth, because no outgoing edge may be visited there. This is less work than
forcing fused edge enumeration.

## ABI evolution and compatibility

`VtDictionaryGraphVTable` and `VtDictionaryVisitVTable` are optional,
separately discoverable interfaces, not fields appended to the mandatory
dictionary vtable. A graph-capable immutable snapshot supplies one contiguous
node/edge view and an optional mapped-value lookup. A provider without the
graph interface continues through `node_visit`, when available, or the
original finality and paged-edge callbacks. This preserves binary compatibility
while letting the hottest path avoid callbacks, cache-directory lookups,
atomic child promotion, and provider-arena synchronization altogether.

`ResourceTransducer::snapshot` and the C ABI's
`llev_transducer_snapshot` are additive. A caller that needs a live source
revision may retain the existing behavior; a read-mostly batch can explicitly
pin one immutable revision and share its validated graph or lazily populated
fallback node cache across all cursors.

## Correctness and performance gates

Propagation is complete only when all of the following evidence is green:

1. Exact result signatures agree between direct Rust and the resource path for
   every shared cross-language cell.
2. Unit tests cover byte, Unicode-scalar, and `u64` query units and all four
   public unit-cost algorithms.
3. Specialized query tests cover f64, contextual, language, phonetic, ordered,
   ranked/value-filtered, prefix, subsequence, and zipper behavior.
4. Every production dictionary node and transparent decorator continues to
   compile with its borrowed visitor override; a compatibility-only mock proves
   the default trait implementation, and a decorator probe proves that wrapping
   preserves direct and fused dispatch without calling boxed `edges()`.
5. ABI layout tests and the formal layout manifest prove the base and optional
   vtable representations on 32-bit and 64-bit targets.
6. Construction tests prove sorted-order validation, duplicate/value semantics,
   empty input, and equivalent membership for sorted and unordered builders.
7. Backend-specific benchmark comparisons show no material regression; an
   optimization is reverted or specialized if the abstraction costs more than
   its preregistered acceptance threshold.
8. JVM lexical-drain tests compare lazy and callback result multisets, execute
   at least 10,000 steady-state queries, and prove cleanup after a consumer
   exception for each supported query descriptor family.

The measured causes, experiment decisions, and before/after performance results
are maintained in
[`java-parity-causal-analysis.md`](cross-language/java-parity-causal-analysis.md).
The ABI contracts are specified in
[`abi-reference.md`](../../vinary-tree-interop/docs/abi-reference.md) and
[`abi-evolution.md`](../../vinary-tree-interop/docs/abi-evolution.md).
