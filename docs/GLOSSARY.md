# liblevenshtein-rust Technical Glossary

**Comprehensive reference for implementation, performance, and user-facing terminology**

**Last Updated:** 2026-06-19  ·  **Version:** 0.9.1

---

## About This Glossary

This glossary covers implementation details, performance optimizations, data structures, and user-facing features in liblevenshtein-rust.

**For theoretical algorithm concepts** (Position, Subsumption, Characteristic Vectors, etc.), see the [Levenshtein Automata Glossary](research/levenshtein-automata/glossary.md).

> **Crate-boundary note.** Since v0.9.0 the dictionary backends were extracted to
> the sibling [`libdictenstein`](architecture/overview.md) crate and are re-exported
> here as `#[deprecated]` shims. Some entries below still reference historical
> `src/dictionary/…` paths; those implementations now live in `libdictenstein`.
> Terminology introduced after the 2025-01 revision — phonetics, time-series
> (MSM), the universal/generalized automaton variants, the `.llev`/`.llre` DSLs,
> and the formal-verification vocabulary — is collected in the
> [Terminology added since 2025](#terminology-added-since-2025-phonetics--time-series--automaton-variants--dsls--verification) section.

## How to Use

- **Categories**: Each term is tagged with relevant categories
- **Cross-references**: "See also" links to related terms
- **Code links**: Direct links to source implementations
- **Search**: Use Ctrl+F to find specific terms

## Categories

- **[Algorithm]** - Core algorithmic concepts and patterns
- **[Data Structure]** - Data organization and storage patterns
- **[Performance]** - Optimization techniques and patterns
- **[Memory]** - Memory management strategies
- **[API]** - User-facing features and interfaces
- **[Unicode]** - Character and text handling
- **[Serialization]** - Data persistence formats
- **[Navigation]** - Tree traversal patterns
- **[Caching]** - Cache management and eviction policies

---

## A

### Arc Path Sharing
**Categories:** [Performance], [Memory]

**Definition:** Optimization technique using `Arc<Vec<T>>` to share path data between dictionary nodes without cloning, eliminating expensive allocation during tree traversal.

**Benefits:**
- Eliminates O(depth) cloning overhead
- Reduces memory allocations by ~90%
- Improves cache locality

**Used in:** PathMapDictionary, DictZipper implementations

**Code:** `src/dictionary/pathmap.rs` (now in the `libdictenstein` crate)

**See also:** SmallVec, Lazy Edge Iteration

---

### Arena Allocation
**Categories:** [Performance], [Memory], [Data Structure]

**Definition:** Memory allocation strategy where objects are allocated from a pre-allocated memory block (arena) and freed all at once when the arena is dropped, avoiding per-object deallocation overhead.

**Benefits:**
- 20-25% faster construction for OptimizedDawg
- Better cache locality (sequential allocation)
- No per-object deallocation cost

**Trade-offs:**
- Cannot free individual objects
- Memory held until entire arena drops
- Requires upfront size estimation

**Used in:** OptimizedDawg, State pools

**Code:** `src/dictionary/dawg_optimized.rs` (now in the `libdictenstein` crate)

**See also:** State Pool, Memory Pressure

---

### Auto-Minimization
**Categories:** [Algorithm], [Data Structure], [Performance]

**Definition:** Automatic DAWG minimization triggered when the graph size exceeds a configured growth threshold, maintaining compact representation during bulk insertions.

**Configuration:** Set threshold ratio (e.g., 1.5 = minimize at 50% growth)

**Benefits:**
- 30% faster bulk insertions (1000+ terms)
- Prevents memory bloat during construction
- Maintains query performance

**Trade-offs:**
- Slight overhead for small datasets
- Periodic pauses during minimization

**Used in:** DynamicDawg

**Code:** `src/dictionary/dynamic_dawg.rs` (now in the `libdictenstein` crate)

**See also:** Suffix Sharing, Bloom Filter

---

### AVX2 / AVX-512
**Categories:** [Performance], [SIMD]

**Definition:** Advanced Vector Extensions - Intel/AMD SIMD instruction sets enabling parallel operations on 256-bit (AVX2) or 512-bit (AVX-512) registers.

**Usage in Project:**
- Characteristic vector operations (8x or 16x parallelism)
- Position subsumption checking
- Distance matrix computations

**Detection:** Runtime CPU feature detection via `is_x86_feature_detected!`

**Performance:** 30-64% speedup on supported CPUs

**Code:** [`src/transducer/simd/`](../src/transducer/simd.rs)

**See also:** SSE4.1, Vectorization, Scalar Fallback

---

## B

### BASE and CHECK Arrays
**Categories:** [Data Structure], [Algorithm]

**Definition:** Core arrays in double-array trie implementation. BASE stores the base index for state transitions, CHECK stores parent state for validity verification.

**Algorithm:**
```
transition(state, char) = BASE[state] + char
if CHECK[result] == state: valid transition
```

**Benefits:**
- O(1) state transitions via array indexing
- Compact representation (~8 bytes per state)
- Excellent cache locality

**Used in:** DoubleArrayTrie, DoubleArrayTrieChar

**Code:** `src/dictionary/double_array_trie.rs` (now in the `libdictenstein` crate)

**See also:** Double-Array Trie, Cache Locality

---

### Bloom Filter
**Categories:** [Data Structure], [Performance]

**Definition:** Probabilistic data structure for fast membership testing with no false negatives but possible false positives. Used to accelerate `contains()` operations in DynamicDawg.

**Performance:**
- 88-93% faster `contains()` operations
- ~10% memory overhead
- Configurable capacity

**Configuration:** Set expected capacity (e.g., 10,000 terms)

**Trade-offs:**
- False positives possible (must verify with full check)
- Memory overhead grows with capacity
- Cannot remove elements

**Used in:** DynamicDawg

**Code:** `src/dictionary/dynamic_dawg.rs` (now in the `libdictenstein` crate)

**See also:** Auto-Minimization, Contains Operations

---

### Byte-Level vs Character-Level
**Categories:** [Unicode], [Algorithm]

**Definition:** Fundamental distinction in how strings are processed for distance calculations.

**Byte-Level:** Treats each UTF-8 byte as a unit
- Example: "café" (5 bytes: c, a, f, 0xC3, 0xA9)
- Distance("café", "cafe") = 2 (two bytes differ)
- Used by: DoubleArrayTrie, PathMapDictionary, DawgDictionary

**Character-Level:** Treats each Unicode character as a unit
- Example: "café" (4 characters: c, a, f, é)
- Distance("café", "cafe") = 1 (one character substitution)
- Used by: DoubleArrayTrieChar, PathMapDictionaryChar, DynamicDawgChar

**When to Use:**
- Byte-level: ASCII/Latin-1 text, maximum performance
- Character-level: Multi-language text, correct Unicode semantics

**Performance:** Character-level adds ~5% overhead for UTF-8 decoding

**Code:** `src/dictionary/double_array_trie_char.rs` (now in the `libdictenstein` crate)

**See also:** UTF-8 Decoding, CharUnit Trait, Monomorphization

---

## C

### Cache Locality
**Categories:** [Performance], [Memory]

**Definition:** Property where data accessed together is stored close in memory, minimizing cache misses and improving CPU cache hit rates.

**Impact in Project:**
- DoubleArrayTrie: Sequential BASE/CHECK arrays → excellent locality
- Arena allocation: Sequential object layout → better locality
- PathMap: Pointer chasing → worse locality

**Performance Difference:** 3-30x speedup for DAT vs PathMap queries

**Measurement:** Use `perf stat -e cache-references,cache-misses` to measure

**Code:** All dictionary implementations

**See also:** Double-Array Trie, Arena Allocation, BASE and CHECK Arrays

---

### CharUnit Trait
**Categories:** [API], [Unicode], [Data Structure]

**Definition:** Abstraction trait enabling generic implementations over both `u8` (bytes) and `char` (Unicode characters), allowing byte-level and character-level dictionaries to share code.

**Methods:**
- `from_bytes()` - Parse from UTF-8
- `to_bytes()` - Serialize to UTF-8
- Size, iteration, conversion operations

**Benefits:**
- Zero-cost abstraction via monomorphization
- Single codebase for both variants
- Type-safe character/byte handling

**Used in:** All dictionary backends (generic over `L: CharUnit`)

**Code:** `src/dictionary/char_unit.rs` (now in the `libdictenstein` crate)

**See also:** Byte-Level vs Character-Level, Monomorphization

---

### Checkpoint System
**Categories:** [API], [Algorithm], [Data Structure]

**Definition:** State snapshotting mechanism in ContextualCompletionEngine allowing undo/redo operations for draft text, enabling editor integrations with time-travel debugging.

**Operations:**
- `checkpoint()` - Save current draft state (~116 ns)
- `undo()` - Restore to previous checkpoint
- Multiple checkpoint stack support

**Use Cases:**
- Editor integration (Ctrl+Z support)
- Incremental typing with backtracking
- LSP server implementations

**Performance:** Sub-microsecond checkpoint creation

**Code:** [`src/contextual/engine.rs`](../src/contextual/engine.rs)

**See also:** Draft State, Contextual Completion, Incremental Typing

---

### Contextual Completion
**Categories:** [API], [Algorithm]

**Definition:** Hierarchical scope-aware completion system providing fuzzy matching within lexical scopes (global → module → function → block), with separate draft and finalized term spaces.

**Key Concepts:**
- **Hierarchical Visibility:** Child scopes see parent terms
- **Draft State:** Incremental typing without polluting finalized dictionary
- **Checkpoint/Undo:** Editor-friendly state management
- **Query Fusion:** Search both draft and finalized simultaneously

**Performance:**
- Insert character: ~4 µs
- Query (500 terms, distance 1): ~11.5 µs
- Checkpoint: ~116 ns

**Use Cases:**
- LSP servers (multi-file scope awareness)
- Code editors (context-sensitive completion)
- REPL environments (session-scoped symbols)

**Code:** [`src/contextual/`](../src/contextual/)

**See also:** Hierarchical Visibility, Draft State, Scope-Aware Completion

---

### Cost-Aware Eviction
**Categories:** [Caching], [Performance]

**Definition:** Cache eviction policy that considers both access frequency and computational cost of regenerating entries, prioritizing retention of expensive-to-recompute items.

**Algorithm:** Score = frequency × regeneration_cost

**Benefits:**
- Optimizes total computation time
- Protects expensive operations (distance 3-4 queries)
- Balances hit rate with recomputation cost

**Trade-offs:**
- More complex than LRU
- Requires cost estimation

**Used in:** Planned for FuzzyMap caching

**Code:** [`src/cache/eviction/cost_aware.rs`](../src/cache/eviction/cost_aware.rs)

**See also:** LRU, LFU, Memory Pressure Eviction

---

## D

### Damerau-Levenshtein Distance
**Categories:** [Algorithm], [Edit Operations]

**Definition:** Edit distance metric that includes transpositions (swapping adjacent characters) as a single operation, in addition to standard insertions, deletions, and substitutions.

**Example:**
- Standard Levenshtein: "teh" → "the" = 2 (delete 'e', insert 'e')
- Damerau-Levenshtein: "teh" → "the" = 1 (transpose 'e' and 'h')

**Use Cases:**
- Typing errors (common transpositions: "teh", "recieve")
- Keyboard input corrections

**Implementation:** `Algorithm::Transposition` variant

**Code:** [`src/transducer/transition.rs`](../src/transducer/transition.rs) (Lines 195-319)

**See also:** Transposition, Edit Operations, Standard Algorithm

---

### DashMap
**Categories:** [Data Structure], [Performance]

**Definition:** Concurrent HashMap implementation providing lock-free reads and fine-grained write locking, used for thread-safe caching without global locks.

**Benefits:**
- Lock-free concurrent reads
- Sharded locking for writes (reduced contention)
- No std::sync::RwLock overhead

**Used in:** Fuzzy cache implementations, concurrent query caching

**Code:** External dependency, used in [`src/cache/`](../src/cache/)

**See also:** Thread-Safe Interior Mutability, RwLock

---

### DAWG (Directed Acyclic Word Graph)
**Categories:** [Data Structure], [Algorithm]

**Definition:** Space-efficient trie variant where common suffixes are shared through node merging, creating a directed acyclic graph structure.

**Properties:**
- Acyclic (no loops)
- Deterministic (single path per prefix)
- Suffix sharing (common suffixes use same nodes)

**Variants in Project:**
- **DawgDictionary:** Basic static DAWG
- **OptimizedDawg:** Arena-allocated, 20-25% faster construction
- **DynamicDawg:** Supports runtime insert/remove/minimize

**Memory:** ~13-32 bytes per state (depending on variant)

**Code:** `src/dictionary/dawg.rs` (now in the `libdictenstein` crate)

**See also:** Trie, Suffix Sharing, Auto-Minimization

---

### Double-Array Trie (DAT)
**Categories:** [Data Structure], [Algorithm], [Performance]

**Definition:** Trie implementation using two parallel arrays (BASE and CHECK) for O(1) state transitions with excellent cache locality.

**Structure:**
- BASE[s] stores base index for state s
- CHECK[r] validates that r is valid child of parent
- Transition: next_state = BASE[state] + char

**Benefits:**
- **Fastest queries:** 3x faster than DAWG, 10x faster than PathMap
- **O(1) transitions:** Array indexing, no pointer chasing
- **Compact:** ~8 bytes per state
- **Cache-friendly:** Sequential array access

**Variants:**
- **DoubleArrayTrie:** Byte-level (u8 labels)
- **DoubleArrayTrieChar:** Character-level (char labels, Unicode-aware)

**Trade-offs:**
- Static (no runtime modifications)
- Construction slower than PathMap

**Code:** `src/dictionary/double_array_trie.rs` (now in the `libdictenstein` crate)

**See also:** BASE and CHECK Arrays, Cache Locality, Dictionary Automaton

---

### Draft State
**Categories:** [API], [Algorithm]

**Definition:** Temporary, uncommitted text in ContextualCompletionEngine representing incremental typing (e.g., "local_var" while user is still typing) that exists separately from the finalized dictionary.

**Properties:**
- Not visible to other contexts
- Can be checkpointed and undone
- Convertible to finalized term via `finalize()`
- Participates in fuzzy queries alongside finalized terms

**Use Cases:**
- Real-time completion while typing
- Editor integration (unsaved changes)
- REPL input buffer

**Performance:** ~4 µs per character insertion

**Code:** [`src/contextual/draft.rs`](../src/contextual/draft_buffer.rs)

**See also:** Finalized State, Checkpoint System, Incremental Typing

---

### Dynamic DAWG
**Categories:** [Data Structure], [API]

**Definition:** DAWG variant supporting runtime insert, remove, and minimize operations with thread-safe access via RwLock.

**Features:**
- Insert/remove terms at runtime
- Optional auto-minimization (configurable threshold)
- Optional Bloom filter (88-93% faster contains())
- Thread-safe interior mutability
- Character-level variant (DynamicDawgChar)

**Performance Optimizations:**
- Sorted batch insertion
- Auto-minimization at 50% growth
- Bloom filter for membership tests

**Use Cases:**
- Dynamic dictionaries requiring frequent updates
- Best fuzzy matching performance for dynamic use

**Code:** `src/dictionary/dynamic_dawg.rs` (now in the `libdictenstein` crate)

**See also:** DAWG, Auto-Minimization, Bloom Filter, Thread-Safe Interior Mutability

---

## E

### Edge Label Scanning
**Categories:** [Performance], [SIMD]

**Definition:** SIMD-accelerated operation that scans dictionary edge labels in parallel to find matching characters, using vectorized comparisons.

**Implementation:**
- Load 16-32 labels into SIMD register
- Parallel equality comparison with target character
- Return bitmask of matches
- Process matches sequentially

**Performance:** 20-40% faster edge iteration for high-degree nodes

**Threshold:** Enabled for nodes with 16+ children (empirically tuned)

**Code:** `src/dictionary/simd/edge_lookup.rs` (now in the `libdictenstein` crate)

**See also:** AVX2, Vectorization, Threshold Tuning

---

### Edit Operations
**Categories:** [Algorithm]

**Definition:** Fundamental string transformation operations used to measure Levenshtein distance.

**Standard Operations** (Algorithm::Standard):
- **Insertion:** Add a character (cost = 1)
- **Deletion:** Remove a character (cost = 1)
- **Substitution:** Replace one character with another (cost = 1)

**Extended Operations:**
- **Transposition** (Algorithm::Transposition): Swap adjacent characters (cost = 1)
- **Merge** (Algorithm::MergeAndSplit): Two chars → one char (cost = 1)
- **Split** (Algorithm::MergeAndSplit): One char → two chars (cost = 1)

**Code:** [`src/transducer/transition.rs`](../src/transducer/transition.rs)

**See also:** Levenshtein Distance, Damerau-Levenshtein Distance, Algorithm Variants

---

### Eviction Policy
**Categories:** [Caching], [Performance]

**Definition:** Strategy for removing entries from a cache when capacity is reached, determining which entries to discard.

**Policies Implemented:**
- **LRU (Least Recently Used):** Evict oldest access
- **LFU (Least Frequently Used):** Evict lowest access count
- **TTL (Time To Live):** Evict after expiration time
- **Age-Based:** Evict by creation time
- **Cost-Aware:** Evict by frequency × regeneration cost
- **Memory Pressure:** Evict based on system memory availability

**Code:** [`src/cache/eviction/`](../src/cache/eviction/)

**See also:** LRU, LFU, TTL, Cost-Aware Eviction, Memory Pressure Eviction

---

## F

### Finalized State
**Categories:** [API], [Algorithm]

**Definition:** Committed, permanent terms in ContextualCompletionEngine that are visible across contexts according to hierarchical visibility rules, as opposed to draft state which is context-local.

**Properties:**
- Immutable once finalized
- Visible to child contexts (hierarchical visibility)
- Shared across all transducers using the dictionary
- Cannot be undone (permanent)

**Creation:** Call `finalize()` on a context with draft state

**Code:** [`src/contextual/engine.rs`](../src/contextual/engine.rs)

**See also:** Draft State, Hierarchical Visibility, Contextual Completion

---

### Fuzzy Map
**Categories:** [API], [Data Structure]

**Definition:** Dictionary-like data structure supporting approximate key lookups using Levenshtein distance, returning values for keys within specified edit distance.

**Example:**
```rust
map.get("appl", 1)  // Returns value for "apple"
```

**Variants:**
- **FuzzyMap:** Returns single matching term and value
- **FuzzyMultiMap:** Aggregates values from all matching terms (union operation)

**Performance:** 10-100x faster with during-traversal value filtering vs post-filtering

**Use Cases:**
- User ID lookup with fuzzy name matching
- Tag aggregation across similar terms
- Multi-valued dictionary queries with error tolerance

**Code:** [`src/cache/multimap.rs`](../src/cache/multimap.rs)

**See also:** Value Filtering, Levenshtein Distance, Term-Value Mapping

---

## H

### Hierarchical Visibility
**Categories:** [Algorithm], [API]

**Definition:** Scope visibility rules in ContextualCompletionEngine where child contexts can see parent terms but not sibling or descendant terms, modeling lexical scoping.

**Example:**
```
Global (terms: std::vector, std::string)
  └─ Function (terms: parameter, result)
       └─ Block (terms: local_var)
```
From Block: can see local_var, parameter, result, std::vector, std::string
From Function: can see parameter, result, std::vector, std::string (NOT local_var)

**Implementation:** Context tree with upward traversal

**Code:** [`src/contextual/engine.rs`](../src/contextual/engine.rs)

**See also:** Contextual Completion, Scope-Aware Completion, Context Tree

---

## I

### Imitation Method
**Categories:** [Algorithm], [Performance]

**Definition:** Technique from Chapter 6 of Schulz & Mihov (2002) paper that simulates Levenshtein automaton LEV_n(W) without explicit construction, generating states on-demand during dictionary traversal.

**Benefits:**
- O(|W|) space instead of potentially O(4^n) for materialized automaton
- Lazy evaluation - only compute states actually needed
- No preprocessing phase

**Implementation:** On-the-fly state generation in QueryIterator

**Code:** [`src/transducer/query.rs`](../src/transducer/query.rs) (Lines 86-188)

**See also:** Levenshtein Automaton, Lazy Evaluation, Parallel Traversal

---

### Incremental Typing
**Categories:** [API], [Performance]

**Definition:** Character-by-character text input handling in ContextualCompletionEngine, updating draft state and providing real-time completion suggestions.

**Performance:**
- Per-character insertion: ~4 µs
- Query after each character: ~11.5 µs (500 terms, distance 1)
- Total latency: < 20 µs (submillisecond for interactive use)

**Features:**
- Immediate completion updates
- Checkpoint support for undo
- Query fusion (draft + finalized)

**Use Cases:**
- IDE code completion
- Search-as-you-type
- Command palette fuzzy matching

**Code:** [`src/contextual/engine.rs`](../src/contextual/engine.rs)

**See also:** Draft State, Contextual Completion, Checkpoint System

---

## L

### Lazy Edge Iteration
**Categories:** [Performance], [Algorithm]

**Definition:** Zero-copy edge iteration strategy in PathMap that avoids allocating child vectors, instead providing iterators directly over internal data structures.

**Benefits:**
- 15-50% faster edge iteration
- No temporary allocations
- Reduced memory pressure

**Implementation:** Return `impl Iterator` over internal edge storage

**Code:** `src/dictionary/pathmap.rs` (now in the `libdictenstein` crate)

**See also:** Arc Path Sharing, Zero-Copy

---

### Lazy Evaluation
**Categories:** [Algorithm], [Performance]

**Definition:** Evaluation strategy where query results are computed on-demand as the iterator is consumed, rather than materializing all results upfront.

**Benefits:**
- Constant memory usage (O(1) for iterator state)
- Early termination possible (stop after first N results)
- No wasted work for unused results

**Implementation:** QueryIterator and OrderedQueryIterator are lazy iterators

**Code:** [`src/transducer/query.rs`](../src/transducer/query.rs)

**See also:** Imitation Method, Iterator Pattern

---

### Levenshtein Distance
**Categories:** [Algorithm]

**Definition:** Edit distance metric measuring the minimum number of single-character edits (insertions, deletions, substitutions) required to transform one string into another.

**Example:** distance("kitten", "sitting") = 3
1. kitten → sitten (substitute 'k' → 's')
2. sitten → sittin (substitute 'e' → 'i')
3. sittin → sitting (insert 'g')

**Variants:**
- **Standard:** Insert, delete, substitute
- **Damerau:** + transposition
- **Generalized:** + merge, split

**Code:** Core algorithm spans [`src/transducer/`](../src/transducer/)

**See also:** Edit Operations, Damerau-Levenshtein Distance, Wagner-Fischer Algorithm

---

### LFU (Least Frequently Used)
**Categories:** [Caching], [Performance]

**Definition:** Cache eviction policy that removes entries with the lowest access count when capacity is reached.

**Benefits:**
- Protects frequently-accessed entries
- Good for workloads with hot keys

**Trade-offs:**
- No temporal locality (old frequent items stay)
- Requires frequency counter maintenance
- "Cache pollution" from historical access patterns

**Implementation:** Access counter per entry

**Code:** [`src/cache/eviction/lfu.rs`](../src/cache/eviction/lfu.rs)

**See also:** LRU, Eviction Policy, Cost-Aware Eviction

---

### LRU (Least Recently Used)
**Categories:** [Caching], [Performance]

**Definition:** Cache eviction policy that removes the entry with the oldest last-access time when capacity is reached.

**Benefits:**
- Simple implementation
- Good temporal locality
- Industry-standard baseline

**Trade-offs:**
- Ignores access frequency
- Can evict expensive-to-recompute entries

**Implementation:** Timestamp tracking per entry

**Code:** [`src/cache/eviction/lru.rs`](../src/cache/eviction/lru.rs)

**See also:** LFU, Eviction Policy, Temporal Locality

---

## M

### Memory Pressure Eviction
**Categories:** [Caching], [Performance], [Memory]

**Definition:** Adaptive cache eviction policy that monitors system memory availability and aggressively evicts entries when memory pressure is high.

**Benefits:**
- Prevents OOM crashes
- System-aware resource management
- Automatic adaptation to available memory

**Implementation:**
- Monitor system memory (sysinfo crate)
- Aggressive eviction above threshold (e.g., 80% usage)
- Standard eviction below threshold

**Code:** [`src/cache/eviction/memory_pressure.rs`](../src/cache/eviction/memory_pressure.rs)

**See also:** Eviction Policy, LRU, System Memory Monitoring

---

### Monomorphization
**Categories:** [Performance], [Unicode]

**Definition:** Rust compiler optimization that generates specialized code for each concrete type used with a generic function, enabling zero-cost abstractions.

**Impact in Project:**
- CharUnit trait: Separate optimized code for `u8` and `char` variants
- No runtime polymorphism overhead
- Identical performance to hand-written specialized versions

**Trade-off:** Increased binary size (duplicate code for each type)

**Code:** All generic dictionary implementations over `L: CharUnit`

**See also:** CharUnit Trait, Byte-Level vs Character-Level, Zero-Cost Abstraction

---

## O

### Ordered Query
**Categories:** [API], [Algorithm]

**Definition:** Query variant that returns results sorted by edit distance first, then lexicographically, using priority queue-based traversal.

**Example Output:**
```
Query: "aple", distance 2
Results (in order):
  - "ape" (distance 1)
  - "apple" (distance 1)
  - "apply" (distance 2)
```

**Implementation:** `query_ordered()` returns `OrderedQueryIterator`

**Performance:** Slightly slower than unordered due to priority queue overhead

**Code:** [`src/transducer/query_ordered.rs`](../src/transducer/ordered_query.rs)

**See also:** Query Iterator, Lazy Evaluation

---

## P

### Parallel Traversal
**Categories:** [Algorithm], [Performance]

**Definition:** Simultaneous navigation of dictionary automaton A^D and Levenshtein automaton LEV_n(W), advancing through both in lockstep during query execution.

**Algorithm:**
```
dict_node = dictionary.root()
automaton_state = initial_state()

for each dict_edge in dictionary:
    dict_node' = follow_edge(dict_node, char)
    automaton_state' = transition(automaton_state, char)

    if both_accepting(dict_node', automaton_state'):
        yield current_word
```

**Complexity:** O(|D|) where |D| is total dictionary edges

**Code:** [`src/transducer/query.rs`](../src/transducer/query.rs)

**See also:** Imitation Method, Dictionary Automaton, Levenshtein Automaton

---

### PathMap
**Categories:** [Data Structure], [API]

**Definition:** High-performance trie implementation with structural sharing and zero-copy path access, supporting dynamic updates through interior mutability.

**Features:**
- Structural sharing (shared subtrees)
- Zero-copy path iteration
- Thread-safe insert/delete
- Value-mapped variant support

**Performance:**
- Fastest dynamic backend for modifications
- 3-10x slower queries than DoubleArrayTrie (pointer chasing)

**Variants:**
- **PathMapDictionary:** Byte-level
- **PathMapDictionaryChar:** Character-level (Unicode)

**Code:** `src/dictionary/pathmap.rs` (now in the `libdictenstein` crate)

**See also:** Arc Path Sharing, Lazy Edge Iteration, Dynamic DAWG

---

### PGO (Profile-Guided Optimization)
**Categories:** [Performance]

**Definition:** Compiler optimization technique using runtime profiling data to guide code generation, optimizing hot paths and reducing cold path overhead.

**Usage:**
```bash
# Generate profile
RUSTFLAGS="-C profile-generate=/tmp/pgo" cargo build --release
./target/release/liblevenshtein benchmark
# Use profile
RUSTFLAGS="-C profile-use=/tmp/pgo -C llvm-args=-pgo-warn-missing-function" cargo build --release
```

**Benefits:** 10-15% performance improvement on hot query paths

**Code:** Build system only (no source changes)

**See also:** Performance Optimization, Benchmarking

---

### Position Subsumption
**Categories:** [Algorithm], [SIMD]

**Definition:** SIMD-accelerated operation checking if one position subsumes another (i#e ⊑ j#f), processing multiple positions in parallel.

**Subsumption Rule:** i#e ⊑ j#f iff (e < f) ∧ (|j-i| ≤ f-e)

**Vectorization:** Load 4-8 positions, perform parallel comparisons

**Performance:** 40-60% faster subsumption checking in state operations

**Code:** [`src/transducer/simd/subsumption.rs`](../src/transducer/simd.rs)

**See also:** Subsumption (theory glossary), AVX2, State Operations

---

## Q

### Query Fusion
**Categories:** [Algorithm], [API]

**Definition:** Technique in ContextualCompletionEngine that searches both draft and finalized term spaces simultaneously in a single traversal, avoiding separate queries.

**Benefits:**
- Single dictionary traversal
- No result merging overhead
- Atomic view of both spaces

**Implementation:** Unified search with is_draft flag in results

**Code:** [`src/contextual/engine.rs`](../src/contextual/engine.rs)

**See also:** Draft State, Finalized State, Contextual Completion

---

### Query Iterator
**Categories:** [API], [Algorithm]

**Definition:** Lazy iterator implementing the Imitation Method, generating fuzzy matching results on-demand through parallel traversal of dictionary and Levenshtein automaton.

**Methods:**
- `query(term, distance)` → basic iterator
- `query_with_distance(term, distance)` → includes distance in results
- `query_ordered(term, distance)` → sorted results

**Performance:** O(|D|) traversal, constant memory

**Code:** [`src/transducer/query.rs`](../src/transducer/query.rs)

**See also:** Lazy Evaluation, Parallel Traversal, Imitation Method

---

## R

### Referential Transparency
**Categories:** [Algorithm], [Navigation]

**Definition:** Property of zipper navigation where operations produce new zippers without modifying existing ones, enabling safe concurrent access and time-travel debugging.

**Benefits:**
- Thread-safe without locking
- Enables undo/redo
- Composable operations

**Implementation:** All zipper operations return new zipper instances

**Code:** `src/dictionary/pathmap_zipper.rs` (now in the `libdictenstein` crate)

**See also:** Zipper Pattern, Immutable Navigation

---

### Runtime CPU Feature Detection
**Categories:** [Performance], [SIMD]

**Definition:** Technique for detecting available SIMD instruction sets at runtime, enabling adaptive code paths based on CPU capabilities.

**Implementation:**
```rust
if is_x86_feature_detected!("avx2") {
    // Use AVX2 implementation
} else if is_x86_feature_detected!("sse4.1") {
    // Use SSE4.1 implementation
} else {
    // Use scalar fallback
}
```

**Benefits:**
- Single binary works on all CPUs
- Optimal performance on newer CPUs
- Graceful degradation on older CPUs

**Code:** [`src/transducer/simd/mod.rs`](../src/transducer/simd.rs)

**See also:** AVX2, SSE4.1, Scalar Fallback

---

### RwLock (Reader-Writer Lock)
**Categories:** [Performance], [API]

**Definition:** Synchronization primitive allowing multiple concurrent readers OR single writer, used for thread-safe dictionary access in dynamic backends.

**Benefits:**
- Multiple queries can run concurrently (shared read lock)
- Exclusive write lock for modifications
- No data races

**Trade-offs:**
- Write operations block all readers
- Lock contention on high write frequency

**Used in:** DynamicDawg, PathMapDictionary

**Code:** Standard library `std::sync::RwLock`

**See also:** Thread-Safe Interior Mutability, DashMap, Dynamic DAWG

---

## S

### Scalar Fallback
**Categories:** [Performance], [SIMD]

**Definition:** Non-vectorized implementation serving as fallback when SIMD instructions are unavailable or inappropriate (e.g., small data sizes).

**Usage:**
- CPU lacks required instruction set (no AVX2/SSE4.1)
- Data size below vectorization threshold
- Architecture doesn't support SIMD (ARM without NEON)

**Performance:** Typically 2-3x slower than SIMD, but still optimized scalar code

**Code:** All SIMD modules include scalar fallback paths

**See also:** Runtime CPU Feature Detection, AVX2, SSE4.1

---

### Scope-Aware Completion
**Categories:** [API], [Algorithm]

**Definition:** Completion system that respects lexical scoping rules, only suggesting terms visible in the current scope based on hierarchical visibility.

**Example:**
```
global scope: std::vector, std::string
function scope: parameter, result
block scope: local_var

Query in block: sees all three scopes
Query in function: sees function + global (NOT block)
```

**Implementation:** Context tree with upward traversal

**Code:** [`src/contextual/engine.rs`](../src/contextual/engine.rs)

**See also:** Hierarchical Visibility, Contextual Completion, Context Tree

---

### SmallVec
**Categories:** [Performance], [Memory]

**Definition:** Optimization data structure that stores small collections inline (on stack) and spills to heap only when size exceeds threshold, reducing allocations for common small sizes.

**Configuration:** `SmallVec<[T; N]>` where N is inline capacity

**Benefits:**
- Zero allocations for size ≤ N
- Better cache locality (stack storage)
- Reduced allocator pressure

**Used in:** State storage, edge lists, position vectors

**Code:** External dependency, used throughout [`src/transducer/`](../src/transducer/)

**See also:** Arena Allocation, Memory Pressure

---

### Sorted Batch Insertion
**Categories:** [Algorithm], [Performance]

**Definition:** Optimization in DynamicDawg where inserting pre-sorted terms enables efficient construction without repeated minimization.

**Algorithm:**
1. Sort terms lexicographically
2. Insert in order
3. Single minimization at end

**Performance:** 30-50% faster for bulk inserts (1000+ terms)

**Code:** `src/dictionary/dynamic_dawg.rs` (now in the `libdictenstein` crate)

**See also:** Auto-Minimization, Dynamic DAWG

---

### SSE4.1
**Categories:** [Performance], [SIMD]

**Definition:** Streaming SIMD Extensions 4.1 - Intel/AMD instruction set enabling parallel operations on 128-bit registers (4x f32 or 4x i32).

**Usage in Project:**
- Characteristic vector operations
- Position comparisons
- Fallback when AVX2 unavailable

**Detection:** Runtime CPU feature detection

**Performance:** 20-30% speedup vs scalar

**Code:** [`src/transducer/simd/`](../src/transducer/simd.rs)

**See also:** AVX2, Vectorization, Scalar Fallback

---

### State Pool
**Categories:** [Performance], [Memory], [Data Structure]

**Definition:** Object pool pattern for reusing allocated state objects (Position sets) across queries, eliminating allocation overhead in hot paths.

**Benefits:**
- Eliminates per-query allocations
- Exceptional performance gains (30-50% speedup)
- Reduced GC pressure

**Usage:** Pass `&mut StatePool` to query operations

**Code:** [`src/transducer/state_pool.rs`](../src/transducer/pool.rs)

**See also:** Arena Allocation, Memory Pressure

---

### Suffix Automaton
**Categories:** [Data Structure], [Algorithm]

**Definition:** Trie variant optimized for substring/infix matching, where any path through the automaton represents a valid substring.

**Use Cases:**
- Find patterns anywhere in text (not just prefixes)
- Substring search queries
- Pattern matching

**Variants:**
- **SuffixAutomaton:** Byte-level
- **SuffixAutomatonChar:** Character-level (Unicode)

**Trade-offs:**
- No prefix matching support (`.prefix()` unavailable)
- Different query semantics than other backends

**Code:** `src/dictionary/suffix_automaton.rs` (now in the `libdictenstein` crate)

**See also:** DAWG, Trie, Infix Matching

---

### Suffix Sharing
**Categories:** [Algorithm], [Data Structure]

**Definition:** DAWG optimization where nodes with identical right-languages (same set of possible continuations) are merged, reducing memory usage.

**Example:**
```
"cat" and "bat" share suffix "at"
"testing" and "resting" share suffix "esting"
```

**Benefits:**
- 20-50% memory reduction vs unshared trie
- Enables DAWG compact representation

**Used in:** All DAWG variants

**Code:** `src/dictionary/dynamic_dawg.rs` (now in the `libdictenstein` crate)

**See also:** DAWG, Auto-Minimization

---

## T

### Temporal Locality
**Categories:** [Performance], [Caching]

**Definition:** Property where recently accessed data is likely to be accessed again soon, exploited by LRU caching and CPU cache management.

**Impact:**
- LRU eviction leverages temporal locality
- CPU caches exploit temporal locality automatically
- Hot query patterns benefit from caching

**Measurement:** Cache hit rate over time

**See also:** Cache Locality, LRU, Eviction Policy

---

### Term-Value Mapping
**Categories:** [API], [Data Structure]

**Definition:** Association of values with dictionary terms, enabling fuzzy lookup of metadata alongside approximate string matching.

**Example:**
```rust
dict.insert_with_value("apple", 42);
dict.insert_with_value("application", vec![1, 2, 3]);
```

**Supported Backends:**
- DynamicDawg<V>
- DynamicDawgChar<V>
- PathMapDictionary<V>
- PathMapDictionaryChar<V>

**Common Value Types:**
- Frequencies: `u32`, `u64`
- IDs: `HashSet<u32>`
- Metadata: Custom structs

**Code:** All dictionary implementations with generic `V` parameter

**See also:** Fuzzy Map, Value Filtering

---

### Thread-Safe Interior Mutability
**Categories:** [API], [Performance]

**Definition:** Pattern using RwLock or similar primitives to enable concurrent read access and exclusive write access to shared data structures.

**Implementation:**
```rust
Arc<RwLock<Dictionary>>
```

**Benefits:**
- Multiple queries run concurrently (read locks)
- Safe modifications (write locks)
- No data races

**Used in:** DynamicDawg, PathMapDictionary

**Code:** All dynamic dictionary backends

**See also:** RwLock, DashMap, Dynamic DAWG

---

### Threshold Tuning
**Categories:** [Performance], [SIMD]

**Definition:** Data-driven optimization technique where SIMD algorithms are enabled only above empirically determined data size thresholds, below which scalar code is faster due to setup overhead.

**Example:** Edge label scanning uses SIMD only for 16+ edges

**Methodology:**
1. Benchmark both implementations across sizes
2. Identify crossover point
3. Use conditional dispatch

**Benefits:**
- Optimal performance across all input sizes
- No SIMD overhead for small data

**Code:** All SIMD implementations include threshold checks

**See also:** Runtime CPU Feature Detection, Scalar Fallback

---

### Transposition
**Categories:** [Algorithm], [Edit Operations]

**Definition:** Edit operation swapping two adjacent characters, implemented in Algorithm::Transposition variant (Damerau-Levenshtein distance).

**Example:** "teh" → "the" (transpose 'e' and 'h')

**Cost:** 1 edit operation

**Implementation:** Uses special t-position (i#e_t) to track transposition state

**Code:** [`src/transducer/transition.rs`](../src/transducer/transition.rs) (Table 7.1, Lines 195-319)

**See also:** Damerau-Levenshtein Distance, Edit Operations, t-position (theory glossary)

---

### TTL (Time To Live)
**Categories:** [Caching], [Performance]

**Definition:** Cache eviction policy that removes entries after a fixed duration from insertion, regardless of access patterns.

**Configuration:** Set expiration duration (e.g., 5 minutes)

**Use Cases:**
- Session-based caching
- Rate limiting
- Temporary data with known lifetime

**Trade-offs:**
- Can evict frequently-used entries
- Requires timestamp tracking

**Code:** [`src/cache/eviction/ttl.rs`](../src/cache/eviction/ttl.rs)

**See also:** LRU, Age-Based Eviction, Eviction Policy

---

## U

### UTF-8 Decoding
**Categories:** [Unicode], [Performance]

**Definition:** Process of parsing multi-byte UTF-8 sequences into Unicode code points (char values) for character-level dictionary operations.

**Performance Impact:**
- ~5% overhead for character-level variants
- Validates UTF-8 correctness
- Required for proper Unicode semantics

**Implementation:** Rust's built-in `str::chars()` iterator

**Used in:** All `*Char` dictionary variants

**Code:** `src/dictionary/char_unit.rs` (now in the `libdictenstein` crate)

**See also:** Byte-Level vs Character-Level, CharUnit Trait, Monomorphization

---

## V

### Value Filtering
**Categories:** [API], [Performance]

**Definition:** Optimization technique filtering dictionary entries during traversal based on associated values, dramatically faster than post-filtering results.

**Example:**
```rust
// Filter by scope ID during traversal (10-100x faster)
transducer.query_by_value_set("var", 1, &visible_scopes)

// vs post-filtering (slow)
transducer.query("var", 1).filter(|t| visible_scopes.contains(&t.scope))
```

**Performance:** 10-100x speedup by pruning traversal early

**Use Cases:**
- Scope-based code completion
- Access control filtering
- Multi-tenancy isolation

**Code:** [`src/transducer/query.rs`](../src/transducer/query.rs)

**See also:** Term-Value Mapping, Fuzzy Map, Scope-Aware Completion

---

### Vectorization
**Categories:** [Performance], [SIMD]

**Definition:** Optimization technique using SIMD instructions to process multiple data elements in parallel with a single instruction.

**Example:** Compare 16 characters simultaneously with AVX2

**Benefits:**
- 4-16x parallelism (depending on instruction set)
- 20-64% overall performance improvement
- Exploits modern CPU capabilities

**Implementation:** Manual SIMD intrinsics in hot paths

**Code:** [`src/transducer/simd/`](../src/transducer/simd.rs)

**See also:** AVX2, SSE4.1, SIMD

---

## W

### Wagner-Fischer Algorithm
**Categories:** [Algorithm]

**Definition:** Dynamic programming algorithm for computing Levenshtein distance using a (|W|+1) × (|V|+1) matrix where cell [i,j] contains distance between W[0..i] and V[0..j].

**Complexity:** O(|W| × |V|) time, O(|W| × |V|) space (O(min(|W|, |V|)) with optimization)

**Relation to Project:** Levenshtein automata achieve O(|D|) by avoiding per-word distance computation

**Code:** Not directly implemented (automata approach avoids this)

**See also:** Levenshtein Distance, Parallel Traversal

---

## Z

### Zero-Cost Abstraction
**Categories:** [Performance], [API]

**Definition:** Programming language feature where high-level abstractions compile to the same machine code as hand-written low-level code, with no runtime overhead.

**Examples in Project:**
- CharUnit trait (monomorphized to u8 or char)
- Generic dictionary implementations
- Iterator abstractions

**Rust Guarantee:** Abstraction incurs no additional runtime cost vs manual implementation

**Code:** All generic code over `L: CharUnit` or `V: DictionaryValue`

**See also:** Monomorphization, CharUnit Trait

---

### Zero-Copy
**Categories:** [Performance], [Memory]

**Definition:** Optimization avoiding data copying by using references, views, or sharing mechanisms.

**Examples in Project:**
- Lazy edge iteration (iterator over internal storage)
- Arc path sharing (reference-counted paths)
- String slicing instead of cloning

**Benefits:**
- Eliminates allocation overhead
- Reduces memory usage
- Improves cache efficiency

**Code:** PathMap lazy edge iteration, Arc path sharing

**See also:** Lazy Edge Iteration, Arc Path Sharing

---

### Zipper Pattern
**Categories:** [Navigation], [Algorithm], [Data Structure]

**Definition:** Functional data structure pattern (from Huet 1997) providing efficient cursor-based tree navigation with context preservation, enabling immutable updates and backtracking.

**Properties:**
- **Focus:** Current position in tree
- **Context:** Path from root with unvisited siblings
- **Referentially transparent:** Operations return new zippers
- **Efficient:** O(1) navigation operations

**Variants in Project:**
- **DictZipper / ValuedDictZipper:** Dictionary navigation
- **AutomatonZipper:** Levenshtein automaton state tracking
- **PathMapZipper:** PathMap-specific implementation
- **IntersectionZipper:** Parallel dictionary + automaton traversal

**Use Cases:**
- Custom traversal algorithms (DFS, A*, beam search)
- Time-travel debugging
- Undo/redo systems
- Hierarchical data navigation

**Code:**
- `src/dictionary/pathmap_zipper.rs` (now in the `libdictenstein` crate)
- [`src/transducer/intersection_zipper.rs`](../src/transducer/intersection_zipper.rs)

**See also:** Referential Transparency, Immutable Navigation, Context-Preserving Traversal

---

## Terminology added since 2025 (phonetics · time series · automaton variants · DSLs · verification)

The terms below cover subsystems introduced or substantially expanded after the
2025-01 revision. Each is defined before use elsewhere in the documentation.

### Automaton variants

#### Parameterized (Lazy) Automaton
**Categories:** [Algorithm]

**Definition:** The default query engine. It *simulates* the Levenshtein automaton `A(W, k)` whose states are reduced *sets* of positions `⟨i, e⟩`, materialising each state on first visit during the dictionary walk — there is no precompiled DFA. Equivalent to the academic "parameterized automaton" / Schulz–Mihov imitation method.

**Code:** [`src/transducer/{query,state,transition,pool}.rs`](../src/transducer/) · **See also:** Imitation Method, Universal Levenshtein Automaton, Generalized Automaton, Characteristic Vector

#### Universal Levenshtein Automaton
**Categories:** [Algorithm]

**Definition:** A parameter-free deterministic automaton precomputed once for a fixed `k` and reused for any query word (Mitankin 2005). The crate offers it as an eager alternative to the lazy engine when `k` is fixed and queries are numerous.

**Code:** [`src/transducer/universal/`](../src/transducer/universal/) · **See also:** Parameterized Automaton, Subsumption

#### Generalized Automaton
**Categories:** [Algorithm]

**Definition:** A runtime-configurable automaton whose edit operations are supplied as an `OperationSet` at run time (rather than a compile-time marker type), enabling weighted (`f64`) and phonetic edit costs at ~10–20 % overhead.

**Code:** [`src/transducer/generalized/`](../src/transducer/generalized/) · **See also:** OperationSet, Articulatory Distance

#### OperationSet / SubstitutionSet / SubstitutionPolicy
**Categories:** [Algorithm], [API]

**Definition:** `OperationSet` enumerates the edit operations a generalized automaton may apply. `SubstitutionSet` restricts *which* character substitutions are permitted (presets: `phonetic_basic`, `keyboard_qwerty`, `leet_speak`, `ocr_friendly`); `SubstitutionPolicy` (`Unrestricted` — a zero-sized default — or `Restricted`) selects the policy at the type level.

**Code:** [`src/transducer/{algorithm,substitution_set,substitution_policy}.rs`](../src/transducer/) · **See also:** Edit Operations, Restricted Substitutions

#### Myers Bit-Parallel Distance
**Categories:** [Algorithm], [Performance]

**Definition:** Myers' (1999) bit-vector dynamic-programming algorithm computing edit distance in `𝒪(⌈m∕w⌉ · n)` for machine word width `w`. `standard_distance` dispatches to it for short (`≤ 64`-byte) ASCII inputs.

**Code:** [`src/distance/myers.rs`](../src/distance/myers.rs) · **DOI:** [10.1145/316542.316550](https://doi.org/10.1145/316542.316550) · **See also:** SIMD, Scalar Fallback

#### WallBreaker / Pigeonhole Filter
**Categories:** [Algorithm]

**Definition:** A strategy for large error bounds (`k ≥ 5`): split the query into `k + 1` pieces; by the pigeonhole principle at least one piece is error-free, so it can be located exactly via the SCDAWG and extended/verified. Avoids the state-space blow-up of large-`k` automata.

**Code:** [`src/wallbreaker/`](../src/wallbreaker/) · **See also:** SCDAWG

#### SCDAWG (Symmetric Compact DAWG)
**Categories:** [Data Structure]

**Definition:** A bidirectional compact DAWG indexing every substring, supporting forward extension and suffix links so a matched region can grow left and right. Backs WallBreaker piece location.

**Code:** `Scdawg` / `ScdawgChar` in `libdictenstein` · **See also:** WallBreaker, DAWG

### Phonetic matching

#### IPA (International Phonetic Alphabet)
**Categories:** [Unicode], [API]

**Definition:** A standardized symbol set for the sounds of spoken language; the crate uses IPA for language-agnostic syllabification and articulatory-feature comparison.

**Code:** [`src/phonetic/ipa_syllable.rs`](../src/phonetic/ipa_syllable.rs) · **See also:** Articulatory Feature Distance, Syllabification

#### Articulatory Feature Distance
**Categories:** [Algorithm]

**Definition:** A pronunciation-aware distance in which phonemes are vectors of articulatory features (place and manner of articulation, voicing), and the substitution cost between two phonemes is their feature-vector distance — so `/p/`↔`/b/` (a voicing flip) costs less than `/p/`↔`/k/`.

**Code:** [`src/phonetic/feature_distance.rs`](../src/phonetic/feature_distance.rs), [`src/transducer/articulatory_costs.rs`](../src/transducer/articulatory_costs.rs) · **See also:** Generalized Automaton, IPA

#### Phonetic Normalization
**Categories:** [Algorithm], [API]

**Definition:** Rewriting a term to a canonical phonetic form (via the rule engine, in 53 languages) before fuzzy matching, so that orthographically different but sound-alike terms collide. Exposed as `PhoneticNormalizedDictionary(Char)`.

**Code:** `src/dictionary/phonetic_normalized.rs` (now in the `libdictenstein` crate), [`src/phonetic/application.rs`](../src/phonetic/application.rs) · **See also:** Soundex, NFA Product

#### Phonetic Algorithms (Soundex · Metaphone · NYSIIS · Caverphone · Cologne · Daitch–Mokotoff · Beider–Morse)
**Categories:** [Algorithm]

**Definition:** Classical phonetic-encoding schemes that map a word to a code approximating its pronunciation, enabling sound-alike grouping. Each has a dedicated reference under [`docs/phonetic-extraction/`](phonetic-extraction/).

**See also:** Phonetic Normalization

#### NFA Product (Phonetic ∩ Levenshtein)
**Categories:** [Algorithm]

**Definition:** The product automaton of a phonetic-pattern NFA (built by Thompson construction) with the Levenshtein automaton `A(W, k)`; running it against the dictionary yields candidates that match the pattern *and* lie within edit distance `k`.

**Code:** [`src/phonetic/nfa/product.rs`](../src/phonetic/nfa/product.rs) · **See also:** Thompson Construction, `.llre`

#### Thompson Construction
**Categories:** [Algorithm]

**Definition:** The classical construction of an ε-NFA from a regular expression by structural induction (concatenation, alternation, Kleene star). Because the result is an NFA simulated in linear time, the `.llre` engine is **ReDoS-resistant by construction**.

**Code:** [`src/phonetic/nfa/thompson.rs`](../src/phonetic/nfa/thompson.rs) · **See also:** `.llre`, NFA Product

### DSLs & formats

#### `.llev`
**Categories:** [API], [Serialization]

**Definition:** The LibLevenshtein phonetic-rule file format: a source language for phonetic rewrite rule-sets, compiled (lexer → AST → ruleset → compiled) and applied via `apply_rules_seq`.

**Code:** [`src/phonetic/llev/`](../src/phonetic/llev/) · grammar: [`docs/grammar/llev.ebnf`](grammar/llev.ebnf) · **See also:** Phonetic Normalization

#### `.llre` (LibLevenshtein Regex Expression)
**Categories:** [API]

**Definition:** A regular-expression file format compiled (lexer → parser → AST → symbol expander → NFA compiler) to an NFA for phonetic/pattern matching. ReDoS-resistant via Thompson/Glushkov construction.

**Code:** [`src/phonetic/llre/`](../src/phonetic/llre/) · grammar: [`docs/grammar/llre.ebnf`](grammar/llre.ebnf) · **See also:** Thompson Construction, NFA Product

#### EBNF (Extended Backus–Naur Form)
**Categories:** [API]

**Definition:** The metasyntax used to specify the `.llev`, `.llre`, and regex grammars under [`docs/grammar/`](grammar/).

**See also:** `.llev`, `.llre`

### Time series

#### MSM (Move–Split–Merge)
**Categories:** [Algorithm]

**Definition:** A metric for real-valued time series built from three unit-cost-parameterized edits — **Move** (change a value, cost `∣xᵢ − y∣`), **Split** (one value → two), and **Merge** (two adjacent values → one). Unlike DTW, MSM is a true metric (satisfies the triangle inequality), enabling lower-bound pruning.

**Code:** [`src/time_series/msm.rs`](../src/time_series/msm.rs) · **DOI:** [10.1109/TKDE.2012.88](https://doi.org/10.1109/TKDE.2012.88) · **See also:** TimeSeriesIndex, DTW

#### DTW (Dynamic Time Warping)
**Categories:** [Algorithm]

**Definition:** A classical elastic time-series similarity measure; included here for contrast — DTW is *not* a metric (no triangle inequality), whereas MSM is, which is why this crate indexes with MSM.

**See also:** MSM

#### TimeSeriesIndex / HybridSearchIndex
**Categories:** [Data Structure], [API]

**Definition:** `TimeSeriesIndex` indexes quantized/encoded series in a `DynamicDawg`; `HybridSearchIndex` adds a two-stage search — a cheap lower-bound filter (`length_lb`, `euclidean_lb`, `l1_lb`, `combined_lb`) followed by exact MSM verification, optionally in parallel with rayon.

**Code:** [`src/time_series/{trie_index,hybrid_search,lower_bounds}.rs`](../src/time_series/) · **See also:** MSM, SAX Encoding

#### SAX Encoding
**Categories:** [Algorithm], [Data Structure]

**Definition:** Symbolic Aggregate approXimation — one of the `QuantizationConfig` encodings (alongside delta and float quantization) that turns a numeric series into a discrete symbol string so it can be stored in a trie/DAWG.

**Code:** [`src/time_series/encoding.rs`](../src/time_series/encoding.rs) · **See also:** TimeSeriesIndex

### Crates & verification

#### libdictenstein
**Categories:** [API], [Data Structure]

**Definition:** The sibling crate (path dependency, v0.2) that owns all dictionary backends and the `Dictionary`/`DictionaryNode`/`MappedDictionary` traits, plus SIMD + bloom-filter pruning. Extracted from liblevenshtein in v0.9.0; the old types are re-exported here as deprecation shims.

**See also:** Deprecation Shim, Architecture Overview

#### duallity
**Categories:** [API]

**Definition:** An external, optional crate providing WFST (weighted finite-state transducer) / language-model composition. Referenced by liblevenshtein for WFST integration but **not** a build dependency of this crate.

**See also:** WFST

#### Deprecation Shim
**Categories:** [API]

**Definition:** A `#[deprecated]` re-export in `src/dictionary/` (and the prelude) that forwards a historical liblevenshtein dictionary type to its new home in `libdictenstein`, preserving source compatibility across the 0.9.0 extraction.

**Code:** `src/dictionary/mod.rs` (now in the `libdictenstein` crate) · **See also:** libdictenstein

#### Rocq / Coq · TLA+ · trusted/partial/legacy profile
**Categories:** [Algorithm]

**Definition:** The formal-verification toolchain. **Rocq** (formerly Coq) machine-checked `.v` theories prove metric and algorithmic properties; **TLA+** specifications model-check concurrent and query behaviour. The verification **profile** (trusted / partial / legacy) recorded in [`FORMAL_VERIFICATION_MANIFEST.tsv`](verification/FORMAL_VERIFICATION_MANIFEST.tsv) is the declared source of truth for what is proved versus assumed.

**See also:** [Verification](verification/)

---

## Cross-References

### By Category

**Algorithm:** Imitation Method, Parallel Traversal, Query Fusion, Scope-Aware Completion, Wagner-Fischer Algorithm, Auto-Minimization, Suffix Sharing, Sorted Batch Insertion

**Data Structure:** Arena Allocation, BASE and CHECK Arrays, Bloom Filter, DAWG, Double-Array Trie, Dynamic DAWG, PathMap, SmallVec, State Pool, Suffix Automaton, Zipper Pattern

**Performance:** Arc Path Sharing, Cache Locality, Edge Label Scanning, Lazy Edge Iteration, Lazy Evaluation, Monomorphization, PGO, Runtime CPU Feature Detection, Scalar Fallback, Threshold Tuning, Vectorization, Zero-Copy

**Memory:** Arena Allocation, Memory Pressure Eviction, SmallVec, State Pool, Zero-Copy

**API:** CharUnit Trait, Checkpoint System, Contextual Completion, Draft State, Finalized State, Fuzzy Map, Ordered Query, Query Iterator, RwLock, Term-Value Mapping, Thread-Safe Interior Mutability, Value Filtering

**Unicode:** Byte-Level vs Character-Level, CharUnit Trait, UTF-8 Decoding, Monomorphization

**Caching:** Cost-Aware Eviction, DashMap, Eviction Policy, LFU, LRU, Memory Pressure Eviction, TTL, Temporal Locality

**SIMD:** AVX2, AVX-512, Edge Label Scanning, Position Subsumption, Runtime CPU Feature Detection, Scalar Fallback, SSE4.1, Threshold Tuning, Vectorization

---

## Additional Resources

- **Theoretical Concepts:** See [Levenshtein Automata Glossary](research/levenshtein-automata/glossary.md)
- **Algorithm Documentation:** [Complete Algorithm Docs](research/levenshtein-automata/README.md)
- **Implementation Mapping:** [Code-to-Paper Correspondence](research/levenshtein-automata/implementation-mapping.md)
- **User Guide:** [Features Overview](user-guide/features.md)
- **Developer Guide:** [Architecture](developer-guide/architecture.md)

---

**Contributing:** To add new terms, maintain alphabetical order and include all standard fields (definition, benefits, trade-offs, code references, see also).

**Last Updated:** 2026-06-19
