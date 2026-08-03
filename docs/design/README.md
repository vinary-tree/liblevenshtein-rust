# Design Documents

Technical specifications and design documents for major features and algorithms.

## Contents

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
Comprehensive performance analysis of zipper-based vs node-based query iteration:
- Benchmark results (1.66-1.97× performance difference)
- Root cause analysis (indirection, locks, allocations, cache effects)
- Architectural trade-offs and benefits
- Use case recommendations (when to use each approach)
- Future optimization opportunities

## Performance Characteristics

**Contextual Completion Engine** (Zipper-based):
- Insert character: ~4 µs (12 M chars/sec)
- Checkpoint: ~116 ns per operation
- Query (500 terms, distance 1): ~11.5 µs
- Query (distance 2): ~309 µs
- Thread-safe, hierarchical context support

**Simple Fuzzy Matching** (Node-based):
- Query (distance 1): ~53 µs (1.88× faster)
- Query (distance 2): ~156 µs (1.97× faster)
- Best for single-dictionary, high-throughput scenarios

## Related Documentation

- [Developer Guide](../developer-guide/README.md) - Implementation guidelines
- [Research](../research/README.md) - Performance research and analysis
