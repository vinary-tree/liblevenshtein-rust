# Contextual Completion Implementation Roadmap

## Document Status
- **Status**: Design Approved
- **Parent**: [contextual-completion-zipper.md](./contextual-completion-zipper.md)
- **Created**: 2025-10-31
- **Last Updated**: 2025-10-31

## Overview

This document provides a detailed, phase-by-phase implementation plan for the Contextual Completion Engine with hierarchical zippers.

## Implementation Phases

### Phase 1: Core Zipper Traits and Basic Implementations
**Goal**: Establish foundational zipper abstractions and implement for target backends

**Duration**: 1-2 weeks

#### Tasks

##### 1.1: Define Zipper Traits
- **File**: `src/dictionary/zipper.rs` (new)
- **What to build**:
  - `DictionaryZipper` trait with methods: `is_final()`, `descend()`, `children()`, `path()`
  - `ValuedDictionaryZipper` trait extending `DictionaryZipper` with `value()` method
  - Comprehensive trait documentation with examples
- **Tests**: `tests/dictionary_zipper_test.rs`
  - Test trait bounds and generic usage
  - Verify Clone behavior

##### 1.2: Implement PathMapZipper
- **File**: `src/dictionary/pathmap_zipper.rs` (new)
- **What to build**:
  - `PathMapZipper<V>` struct wrapping `Arc<RwLock<PathMap<V>>>` + path
  - Implement `DictionaryZipper` trait using PathMap's native zippers
  - Implement `ValuedDictionaryZipper` trait
  - Lock-per-operation pattern with `with_zipper()` helper
- **Tests**: `tests/pathmap_zipper_test.rs`
  - Basic navigation (root → children → grandchildren)
  - Value access for finalized nodes
  - Path reconstruction accuracy
  - Clone and Arc sharing behavior
  - Edge cases: invalid descend, empty path, non-existent children

##### 1.3: Implement DoubleArrayTrieZipper
- **File**: `src/dictionary/dat_zipper.rs` (new)
- **What to build**:
  - `DoubleArrayTrieZipper` struct with `node_index` and `path`
  - Implement `DictionaryZipper` trait using DAT's transition array
  - Optimize for Copy semantics where possible
- **Tests**: `tests/dat_zipper_test.rs`
  - Navigation through DAT structure
  - Finality detection
  - Children iteration
  - Path reconstruction
  - Edge cases: root node, leaf nodes, no children

##### 1.4: Update Module Exports
- **File**: `src/dictionary/mod.rs`
- **What to do**:
  - Add `pub mod zipper;`
  - Export `DictionaryZipper` and `ValuedDictionaryZipper` traits
  - Update `prelude.rs` to include zipper traits

**Deliverables**:
- ✅ Zipper traits defined and documented
- ✅ PathMapZipper implementation with full test coverage
- ✅ DoubleArrayTrieZipper implementation with full test coverage
- ✅ All tests passing

---

### Phase 2: Automaton and Intersection Zippers
**Goal**: Build zipper layer for Levenshtein automaton and compose with dictionary zippers

**Duration**: 1-2 weeks

#### Tasks

##### 2.1: Create AutomatonZipper
- **File**: `src/transducer/automaton_zipper.rs` (new)
- **What to build**:
  - `AutomatonZipper` struct with `state`, `query`, `max_distance`, `algorithm`
  - `new()` method initializing state with all distance thresholds
  - `transition()` method using existing `transition_state_pooled()`
  - `min_distance()` and `infer_distance()` methods
  - `is_viable()` helper for early termination
- **Tests**: `tests/automaton_zipper_test.rs`
  - State initialization correctness
  - Transition accuracy for all algorithm variants (Standard, Transposition, MergeAndSplit)
  - Distance computation verification
  - StatePool integration and reuse verification
  - Edge cases: max_distance exceeded, empty query, long query

##### 2.2: Create IntersectionZipper
- **File**: `src/transducer/intersection_zipper.rs` (new)
- **What to build**:
  - `IntersectionZipper<D: DictionaryZipper>` struct
  - `new()` constructor for root zipper
  - `is_match()`, `distance()`, `depth()` methods
  - `term()` method for path reconstruction
  - `children()` iterator combining dict + automaton transitions
  - Reuse existing `PathNode` type for parent chains
- **Tests**: `tests/intersection_zipper_test.rs`
  - Root zipper creation
  - Child navigation with combined transitions
  - Match detection accuracy
  - Distance computation correctness
  - Path reconstruction (term extraction)
  - Edge cases: no matches, all children match, partial matches

##### 2.3: Create ZipperQueryIterator
- **File**: `src/transducer/zipper_query_iterator.rs` (new)
- **What to build**:
  - `ZipperQueryIterator<D: DictionaryZipper>` struct with BFS queue
  - Implement `Iterator` trait yielding `(String, usize)` tuples
  - Integrate `StatePool` for state reuse
  - Match existing `QueryIterator` API surface
- **Tests**: `tests/zipper_query_iterator_test.rs`
  - Compare results with existing `QueryIterator` (should be identical)
  - Verify lazy evaluation (iterator doesn't compute all results upfront)
  - Test early termination (`.take(5)`)
  - Benchmark vs existing iterator

##### 2.4: Benchmark Comparison
- **File**: `benches/zipper_vs_node_benchmark.rs` (new)
- **What to benchmark**:
  - `QueryIterator` (node-based) vs `ZipperQueryIterator` (zipper-based)
  - PathMapDictionary backend (where zipper should excel)
  - Various query lengths, distances, dictionary sizes
  - Memory allocation patterns (use `dhat` or similar)
- **Success criteria**: Zipper-based within 10% of node-based performance

**Deliverables**:
- ✅ AutomatonZipper with full algorithm support
- ✅ IntersectionZipper composing dictionary + automaton
- ✅ ZipperQueryIterator with Iterator trait
- ✅ Benchmark results showing parity with existing implementation
- ✅ All tests passing

---

### Phase 3: Draft Management with Rollback and Checkpoints
**Goal**: Build draft buffer infrastructure for character-level insertion and undo/redo

**Duration**: 1 week

#### Tasks

##### 3.1: Create DraftBuffer
- **File**: `src/contextual/draft_buffer.rs` (new directory: `src/contextual/`)
- **What to build**:
  - `DraftBuffer` struct with `VecDeque<char>` for `𝒪(1)` push/pop
  - Methods: `push()`, `pop()`, `truncate()`, `len()`, `is_empty()`, `to_string()`
  - Store context ID and creation timestamp
- **Tests**: `tests/draft_buffer_test.rs`
  - Push and pop operations
  - Truncate to arbitrary position
  - String conversion accuracy
  - Edge cases: empty buffer, unicode characters, emoji

##### 3.2: Create Checkpoint System
- **File**: `src/contextual/checkpoint.rs` (new)
- **What to build**:
  - `Checkpoint` struct with `id`, `context`, `position`
  - Checkpoint manager with ID generation (AtomicU64)
  - Checkpoint storage (DashMap for thread-safety)
  - Methods: `create()`, `restore()`, `remove()`, `list()`
- **Tests**: `tests/checkpoint_test.rs`
  - Checkpoint creation and restoration
  - Multiple checkpoints per context
  - Checkpoint invalidation
  - Concurrent checkpoint operations

##### 3.3: Create ContextTree
- **File**: `src/contextual/context_tree.rs` (new)
- **What to build**:
  - `ContextTree` struct with parent/children maps
  - Methods: `insert()`, `remove_subtree()`, `ancestors()`, `contains()`
  - Hierarchical visibility resolution
- **Tests**: `tests/context_tree_test.rs`
  - Tree construction (insert with parents)
  - Ancestor traversal
  - Subtree removal
  - Edge cases: circular parents (should error), orphan nodes

##### 3.4: Integration Test
- **File**: `tests/draft_lifecycle_test.rs` (new)
- **What to test**:
  - Full lifecycle: insert → checkpoint → rollback → restore → finalize
  - Multiple contexts with independent drafts
  - Checkpoint expiration after finalization
  - Edge cases: rollback past start, restore to invalid checkpoint

**Deliverables**:
- ✅ DraftBuffer with character-level operations
- ✅ Checkpoint system with undo/redo support
- ✅ ContextTree with hierarchical visibility
- ✅ Integration tests covering full draft lifecycle
- ✅ All tests passing

---

### Phase 4: Contextual Completion Engine
**Goal**: Assemble all pieces into the high-level ContextualCompletionEngine API

**Duration**: 2 weeks

#### Tasks

##### 4.1: Create ContextualCompletionEngine
- **File**: `src/contextual/engine.rs` (new)
- **What to build**:
  - `ContextualCompletionEngine` struct with:
    - `transducer: Transducer<PathMapDictionary<Vec<ContextId>>>`
    - `drafts: Arc<DashMap<ContextId, DraftBuffer>>`
    - `context_tree: Arc<RwLock<ContextTree>>`
    - `checkpoints: Arc<DashMap<u64, Checkpoint>>`
    - `checkpoint_counter: Arc<AtomicU64>`
  - Constructors: `new()`, `with_algorithm()`
  - Context management: `create_context()`, `remove_context()`, `get_ancestors()`, `get_visible_contexts()`
- **Tests**: `tests/engine_construction_test.rs`
  - Engine creation with different algorithms
  - Context creation and removal
  - Visibility resolution

##### 4.2: Implement Character-Level Insertion
- **What to build**:
  - `insert_char()`, `insert_str()`, `get_draft()`, `has_draft()`
  - Context validation before insertion
  - Thread-safe draft access with DashMap
- **Tests**: `tests/engine_insertion_test.rs`
  - Single character insertion
  - Multi-character insertion
  - Unicode handling
  - Concurrent insertions in different contexts

##### 4.3: Implement Rollback and Checkpoints
- **What to build**:
  - `rollback_char()`, `rollback_n()`, `clear_draft()`
  - `checkpoint()`, `restore()`, `remove_checkpoint()`, `get_checkpoints()`
- **Tests**: `tests/engine_rollback_test.rs`
  - Character rollback
  - Checkpoint restoration
  - Multiple undo/redo operations
  - Edge cases: rollback empty buffer, restore invalid checkpoint

##### 4.4: Implement Finalization and Discard
- **What to build**:
  - `finalize()` - move draft to dictionary with context IDs
  - `finalize_direct()` - direct insertion without draft
  - `discard()` - remove draft without finalizing
- **Tests**: `tests/engine_finalization_test.rs`
  - Draft finalization
  - Direct finalization
  - Discard operation
  - Checkpoint cleanup after finalization

##### 4.5: Implement Query Fusion
- **What to build**:
  - `complete()` - main query method
  - `complete_finalized()` - query only finalized terms
  - `complete_drafts()` - query only drafts
  - Internal: `query_finalized()`, `query_drafts()`, `merge_completions()`
  - Hierarchical visibility filtering
- **Tests**: `tests/engine_query_test.rs`
  - Query with finalized terms only
  - Query with drafts only
  - Query with both (fusion)
  - Hierarchical visibility (child sees parent terms)
  - Deduplication (draft overrides finalized with same term)

##### 4.6: Create Completion Type
- **File**: `src/contextual/completion.rs` (new)
- **What to build**:
  - `Completion` struct with `term`, `distance`, `is_draft`, `contexts`
  - Implement `Ord` for sorting (distance → draft status → term)
  - Helper constructors: `finalized()`, `draft()`
- **Tests**: `tests/completion_test.rs`
  - Sorting behavior
  - Equality and hashing
  - Display formatting

**Deliverables**:
- ✅ ContextualCompletionEngine with full API
- ✅ Character-level insertion and rollback
- ✅ Checkpoint-based undo/redo
- ✅ Finalization and discard
- ✅ Query fusion with hierarchical visibility
- ✅ All tests passing
- ✅ API documentation with examples

---

### Phase 5: Transducer Integration and Backward Compatibility
**Goal**: Integrate with existing Transducer API without breaking changes

**Duration**: 1 week

#### Tasks

##### 5.1: Add Transducer Extension Methods
- **File**: `src/transducer/mod.rs`
- **What to add**:
  - `impl<D> Transducer<D> where D: MappedDictionary<Value = Vec<ContextId>>`
  - `contextual_engine(self) -> ContextualCompletionEngine` - consume transducer
  - `query_with_zipper()` - internal method for zipper-based queries
- **Tests**: `tests/transducer_integration_test.rs`
  - Create engine from transducer
  - Verify transducer is consumed (ownership transfer)
  - Query using zipper internally (PathMap backend)

##### 5.2: Refactor QueryIterator to Use Zippers (PathMap only)
- **File**: `src/transducer/query.rs`
- **What to change**:
  - Add conditional compilation or trait-based dispatch
  - Use `ZipperQueryIterator` for PathMapDictionary backend
  - Keep existing node-based iterator for other backends
  - Maintain identical public API
- **Tests**: Existing `tests/query_comprehensive_test.rs` should pass unchanged
  - Run all existing query tests with PathMap backend
  - Verify results are identical
  - Benchmark to ensure no regression

##### 5.3: Update Examples
- **File**: `examples/contextual_completion.rs` (new)
- **What to show**:
  - Basic usage: insert characters, query, finalize
  - Hierarchical scopes: global → module → function
  - Undo/redo with checkpoints
  - Concurrent usage with Arc
- **Run**: `cargo run --example contextual_completion`

##### 5.4: Update Documentation
- **Files to update**:
  - `README.md` - add brief mention of contextual completion
  - `docs/features.md` - add section on contextual completion
  - API docs in source files with `///` comments
- **What to document**:
  - When to use contextual completion vs regular query
  - Performance characteristics
  - Thread-safety guarantees
  - Example code snippets

**Deliverables**:
- ✅ Transducer integration methods
- ✅ QueryIterator uses zippers for PathMap (backward compatible)
- ✅ Example code demonstrating usage
- ✅ Updated documentation
- ✅ All existing tests still pass

---

### Phase 6: Optimization, Benchmarking, and Polish
**Goal**: Optimize hot paths and ensure production-ready quality

**Duration**: 1-2 weeks

#### Tasks

##### 6.1: Profiling and Hot Path Optimization
- **Tools**: `cargo flamegraph`, `perf`, `criterion`
- **What to profile**:
  - `complete()` with various context counts and draft counts
  - `insert_char()` rapid insertion (simulate typing)
  - `finalize()` with large dictionaries
- **Optimizations to try**:
  - Cache compiled automaton zippers for repeated queries
  - Bloom filter for draft existence checks
  - Optimize PathNode allocation (consider arena allocator)
  - Reduce lock contention (measure with `parking_lot` crate)

##### 6.2: Comprehensive Benchmarks
- **File**: `benches/contextual_completion_benchmarks.rs` (new)
- **Benchmarks**:
  - `insert_char` throughput (ops/sec)
  - `rollback_char` throughput
  - `checkpoint + restore` latency
  - `complete()` with varying:
    - Dictionary size (1K, 10K, 100K terms)
    - Draft count (0, 10, 100)
    - Context depth (1, 5, 10 levels)
    - Max distance (1, 2, 3)
  - Memory usage (RSS, heap allocations)
- **Baseline**: Compare against regular `Transducer::query()`

##### 6.3: Concurrency Benchmarks
- **File**: `benches/concurrent_completion_benchmarks.rs` (new)
- **Benchmarks**:
  - Concurrent queries (8 threads) with no writes
  - Concurrent insertions (8 threads) in different contexts
  - Mixed workload (50% query, 50% insert)
  - Lock contention analysis
- **Metrics**: Throughput, latency percentiles (p50, p95, p99)

##### 6.4: Stress Testing
- **File**: `tests/stress_test.rs` (new, maybe `#[ignore]` by default)
- **Tests**:
  - 10K contexts with 1M total terms
  - 1K checkpoints across 100 contexts
  - Rapid insert/rollback cycles (100K operations)
  - Memory leak detection (run with `valgrind` or `miri`)

##### 6.5: Documentation Polish
- **What to add**:
  - Architecture diagram in ASCII art
  - Complexity analysis (Big-O) for all operations
  - Memory layout diagrams
  - Thread-safety guarantees table
  - Migration guide for users of existing API
  - FAQ section

##### 6.6: Clippy and Rustfmt
- **Run**:
  - `cargo clippy -- -D warnings` (fix all warnings)
  - `cargo fmt --all` (format all code)
  - `cargo doc --no-deps --open` (verify doc links)

**Deliverables**:
- ✅ Profile-guided optimizations applied
- ✅ Comprehensive benchmark suite
- ✅ Stress tests passing
- ✅ Performance within targets (NFR2: < 10% overhead)
- ✅ Memory usage within targets (NFR3: < 1KB per context)
- ✅ Documentation polished and complete
- ✅ All clippy warnings fixed

---

## Success Criteria

### Functional
- [ ] All functional requirements (FR1-FR9) implemented
- [ ] All API methods documented with examples
- [ ] All tests passing (unit, integration, stress)
- [ ] Example code runs successfully

### Non-Functional
- [ ] NFR1: Thread-safe (verified with concurrency tests)
- [ ] NFR2: Performance < 10% overhead (verified with benchmarks)
- [ ] NFR3: Memory < 1KB per context (verified with profiling)
- [ ] NFR4: Lock-free or fine-grained locking (DashMap, lock-per-operation)
- [ ] NFR5: Backward compatible (existing tests still pass)
- [ ] NFR6: Zero-cost abstractions (verified with benchmarks, no trait objects)

### Quality
- [ ] Test coverage > 90% (use `tarpaulin`)
- [ ] No clippy warnings
- [ ] All public items documented
- [ ] Example code in documentation
- [ ] Architecture documented

## Risk Mitigation

### Risk: Performance Regression
- **Mitigation**: Benchmark early and often, optimize hot paths
- **Fallback**: Make zipper-based queries opt-in, keep node-based default

### Risk: Lifetime Complexity with Zippers
- **Mitigation**: Use lock-per-operation pattern with path storage instead of persistent zippers
- **Fallback**: Clone nodes instead of borrowing (trade memory for simplicity)

### Risk: Merge Conflicts with Main Branch
- **Mitigation**: Develop in feature branch, sync frequently, small PRs
- **Fallback**: Rebase regularly, use feature flags

### Risk: Lock Contention in High-Concurrency Scenarios
- **Mitigation**: Use DashMap for per-context parallelism, RwLock for dictionary
- **Fallback**: Implement optimistic locking or lock-free structures if needed

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Core Zipper Traits | 1-2 weeks | None |
| Phase 2: Automaton & Intersection | 1-2 weeks | Phase 1 |
| Phase 3: Draft Management | 1 week | None (parallel with Phase 2) |
| Phase 4: Contextual Engine | 2 weeks | Phases 1-3 |
| Phase 5: Transducer Integration | 1 week | Phase 4 |
| Phase 6: Optimization & Polish | 1-2 weeks | Phase 5 |
| **Total** | **7-10 weeks** | |

## Next Steps

1. Create feature branch: `git checkout -b feature/contextual-completion`
2. Create directory structure: `mkdir -p src/contextual tests/contextual benches`
3. Start Phase 1: Define zipper traits
4. Update progress in [contextual-completion-progress.md](./contextual-completion-progress.md)
