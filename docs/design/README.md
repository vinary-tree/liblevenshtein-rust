# Design Documents

Technical specifications and design documents for major features and algorithms.

## Contents

### [Lazy Ordered-Cost Product Automata](../theory/lazy-ordered-cost-product-automata.md)
Mathematical foundation shared by the specialized implementations:
- query-specialized weighted residuals and an exact applicability criterion
- ordered path algebras, continuation simulations, and canonical antichains
- exact versus abstract synchronized products and exact leaf authority
- coalgebraic online stability and a separate metric-qualification layer

### [Lazy Synchronized Products and Stable Online Automata](lazy-online-products.md)
Defining execution architecture across string, language, and time-series queries:
- why product names the on-demand machine while intersection names its accepted language
- compact canonical state IDs and observed-label transition caches
- proved-only subsumption and exact leaf authority
- fixed-query unknown-stream memory, stack safety, fail-closed limits, and audit matrix
- formal-to-property-test traceability and optimization constraints

### [Crate Boundary and Pruning Duality](crate-boundary-and-prune-duality.md)
Placement rule for future measures:
- why non-negative min-plus costs stay in liblevenshtein
- why gain-valued fzf scoring crosses through duallity to lling-llang
- the corrected unstarted local-alignment upper bound
- why the balanced prefix visitor is DFS rather than BFS

### [Class-A Alignment Presets](class-a-presets.md)
Exact configuration-based string measures:
- Hamming, insertion/deletion, and directional bounded-skip semantics
- public reference distances and generalized-grid conformance
- operation-set validation and resource ceiling
- rejected specialized-walker decision and formal/property evidence

### [True-Damerau Streaming](true-damerau-streaming.md)
History-dependent string-automaton design:
- Lowrance–Wagner entry/extension/resolution refinement
- `DamerauPending` payload and kind-aware subsumption
- explicit representation/resource boundaries
- heterogeneous formal and empirical evidence

### [Elastic Kernels](elastic-kernels.md)
Generic exact time-series retrieval design:
- `ElasticKernel` and `ElasticTransducer<K,V>` ownership boundary
- K1–K4 subtree, candidate, and exact-rescoring obligations
- additive and bottleneck cost-monoid support
- literate range/kNN algorithms, security boundaries, and multi-tool verification

### [`PositionKind` and `AutomatonVariant`](automaton-variant-seam.md)
Monomorphized legacy-automaton seam:
- typed continuation languages in the unchanged 24-byte `Position`
- one runtime selection per dictionary edge
- variant-specific successor, epsilon, subsumption, finish, and window contracts
- property, formal, disassembly, and pre-registered zero-cost gates

### [Ordered Cost Monoid](cost-monoid.md)
Bounded dynamic-programming cost design:
- exact L1–L7 algebra and dominance argument
- unit, weighted, and bottleneck carriers
- checked decimal-to-integer `CostScale`
- IEEE-754 trust boundary and multi-tool verification

### [Generalized-Automaton Repair](generalized-automaton-repair.md)
Exact operation-driven acceptance:
- sparse alignment graph over arbitrary consuming operations
- exact cumulative fractional weights via `CostScale`
- correct Hamming, indel, bounded-skip, and empty-side semantics
- compatibility, resource, formal-proof, and behavior-change boundaries

### [Language Product](language-product.md)
Generic distance-to-language product design:
- `LanguageAutomaton<U>` set-transition contract
- Cost-indexed frontier and merge proof
- Iterative dictionary intersection
- Resource policy and multi-tool formal evidence

### [Dynamic DAWG](dynamic-dawg.md)
Design specification for the dynamic directed acyclic word graph:
- Incremental construction algorithm
- Mutation operations
- Memory efficiency considerations
- Trade-offs vs. static DAWG

### [Hierarchical Correction](hierarchical-correction.md)
Design for hierarchical error correction in fuzzy matching:
- Multi-level correction strategies
- Priority-based candidate selection
- Performance optimization techniques

### [Prefix Matching](prefix-matching.md)
Prefix-based search optimization design:
- Trie-based prefix matching
- Early termination strategies
- Integration with Levenshtein automata

### [Protobuf Dictionary and Operation-Set Persistence](protobuf-serialization.md)
Protocol buffer serialization format specification:
- Schema design
- Backward compatibility guarantees
- Size optimization strategies
- Cross-language interoperability

### [Suffix Automaton](suffix-automaton.md)
Suffix automaton implementation design:
- Construction algorithm
- Online building process
- Space-time trade-offs
- Use cases and applications

### [Contextual Completion Progress](contextual-completion-progress.md)
Complete implementation history and status of the contextual code completion engine:
- Phase-by-phase implementation tracking (Phases 1-6)
- Architecture overview (ContextualCompletionEngine, draft state, finalized terms)
- Performance benchmarks and optimization results
- API documentation and usage examples
- Future enhancement roadmap

### [Zipper vs Node Performance](zipper-vs-node-performance.md)
Historical performance analysis plus a current architecture note:
- the superseded 2025 full-state zipper baseline
- the 2026 shared compact product scheduler
- query-first edge projection and opaque path-context erasure
- snapshot, relative-path, ordering, and stack-safety gates
- the [current bounded engineering check](../benchmarks/lazy-product-engineering-check-2026-08-29.md)

## Performance Characteristics

**Historical 2025 contextual-completion measurements** (superseded query
architecture):
- Insert character: ~4 µs (12 M chars/sec)
- Checkpoint: ~116 ns per operation
- Query (500 terms, distance 1): ~11.5 µs
- Query (distance 2): ~309 µs
- Thread-safe, hierarchical context support

**Current compact PathMap product** (115-term bounded engineering check,
2026-08-29):
- cutoff 0: node 1.1417 µs; zipper 1.1474 µs
- cutoff 1: node 4.1509 µs; zipper 4.2055 µs
- cutoff 2: node 11.794 µs; zipper 11.480 µs
- zipper traversal is within 1.3% of direct-node traversal in every measured
  query; these one-host figures are engineering checks, not portable latency
  promises

## Related Documentation

- [Developer Guide](../developer-guide/README.md) - Implementation guidelines
- [Research](../research/README.md) - Performance research and analysis
