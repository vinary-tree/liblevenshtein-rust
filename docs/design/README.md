# Design Documents

Technical specifications and design documents for major features and algorithms.

## Contents

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

### [Protobuf Serialization](protobuf-serialization.md)
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
