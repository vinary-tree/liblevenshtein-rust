# Documentation Index

This index shows what's documented and where to find it. The documentation is organized by algorithmic layers, with detailed implementation guides for core components.

## Quick Navigation by Topic

### Dictionary Types

**All dictionary types** are documented in [01-dictionary-layer/README.md](01-dictionary-layer/README.md), which includes:

| Dictionary Type | Coverage | Location |
|-----------------|----------|----------|
| **DoubleArrayTrie** | ⭐ Comprehensive (recommended default) | [Detailed guide](01-dictionary-layer/implementations/double-array-trie.md) |
| **DoubleArrayTrieChar** | ⭐ Comprehensive (Unicode support) | [Detailed guide](01-dictionary-layer/implementations/double-array-trie-char.md) |
| **DynamicDawg** | ⭐ Comprehensive | [Detailed guide](01-dictionary-layer/implementations/dynamic-dawg.md) |
| **DynamicDawgChar** | ⭐ Comprehensive (Unicode + dynamic) | [Detailed guide](01-dictionary-layer/implementations/dynamic-dawg-char.md) |
| **SuffixAutomaton** | ⭐ Comprehensive (substring matching) | [Detailed guide](01-dictionary-layer/implementations/suffix-automaton.md) |
| **SuffixAutomatonChar** | 📝 Planned (Unicode substring matching) | [Layer README](01-dictionary-layer/README.md) |
| **PathMapDictionary** | ⭐ Comprehensive (persistent trie) | [Detailed guide](01-dictionary-layer/implementations/pathmap-dictionary.md) |
| **PathMapDictionaryChar** | ✅ Overview + comparison | [Layer README](01-dictionary-layer/README.md) |

**Key sections in Dictionary Layer README**:
- Decision flowchart for choosing a dictionary type
- Detailed comparison table (performance, features, use cases)
- Benchmarks for all types
- Memory characteristics
- Common use cases with code examples

### Levenshtein Automata

**All algorithm variants** are documented in [02-levenshtein-automata/README.md](02-levenshtein-automata/README.md):

| Algorithm Variant | Coverage | Location |
|-------------------|----------|----------|
| **Standard** | ⭐ Complete theory + examples | [Automata README](02-levenshtein-automata/README.md#1-standard-classic-levenshtein) |
| **Transposition** | ⭐ Complete theory + examples | [Automata README](02-levenshtein-automata/README.md#2-transposition-damerau-levenshtein) |
| **Merge-and-Split** | ⭐ Complete theory + examples | [Automata README](02-levenshtein-automata/README.md#3-merge-and-split) |

**Key sections**:
- Levenshtein distance theory with examples
- Finite automata approach explanation
- State representation and transitions
- Algorithm selection guide
- Performance analysis
- 6 detailed usage examples

### Zipper Pattern

Comprehensive documentation in [06-zipper-navigation/README.md](06-zipper-navigation/README.md):

| Zipper Type | Coverage |
|-------------|----------|
| **DoubleArrayTrieZipper** | ⭐ Complete with examples |
| **DoubleArrayTrieCharZipper** | ⭐ Complete with examples |
| **DynamicDawgZipper** | ✅ Implemented (byte-level DAWG navigation) |
| **DynamicDawgCharZipper** | ✅ Implemented (character-level DAWG navigation) |
| **SuffixAutomatonZipper** | ✅ Implemented (byte-level substring navigation) |
| **SuffixAutomatonCharZipper** | ✅ Implemented (Unicode substring navigation) |
| **PathMapZipper** | ⭐ Complete with examples |
| **AutomatonZipper** | ⭐ Complete (Levenshtein state navigation) |
| **IntersectionZipper** | ⭐ Complete (dictionary + automaton composition) |

**Includes**:
- Huet's zipper theory
- 8 detailed usage examples
- Performance analysis
- Advanced patterns (DFS, BFS, parallel exploration)
- Automaton-based fuzzy matching
- Query composition patterns

### Value Storage

Comprehensive documentation in [09-value-storage/README.md](09-value-storage/README.md):

**Covers**:
- Architecture with diagrams
- 5 detailed use cases (code completion, categorization, fuzzy maps, metadata, hierarchical scopes)
- Performance analysis (10-100x speedup)
- Implementation details for all backends

### Contextual Completion

Comprehensive implementation for IDE-style code completion in [docs/design](../design/README.md):

| Document | Coverage | Location |
|----------|----------|----------|
| **Design Specification** | ⭐ Complete architecture | [contextual-completion-zipper.md](../design/contextual-completion-zipper.md) |
| **API Reference** | ⭐ Complete API docs | [contextual-completion-api.md](../design/contextual-completion-api.md) |
| **Implementation Roadmap** | ⭐ Complete 6-phase plan | [contextual-completion-roadmap.md](../design/contextual-completion-roadmap.md) |
| **Progress Tracking** | ⭐ 100% Complete (all phases) | [contextual-completion-progress.md](../design/contextual-completion-progress.md) |
| **Performance Analysis** | ⭐ Zipper vs Node comparison | [zipper-vs-node-performance.md](../design/zipper-vs-node-performance.md) |

**Features**:
- Draft text management with character-level insertion/deletion
- Checkpoint-based undo/redo system
- Hierarchical scope visibility
- Query fusion (draft + finalized terms)
- Automaton-based fuzzy matching
- Thread-safe concurrent access
- **93 tests, 0 failures** ✅

### Core Algorithms

| Layer | Status | Location |
|-------|--------|----------|
| **Intersection/Traversal** | ⭐ Complete | [03-intersection-traversal/README.md](03-intersection-traversal/README.md) |
| **SIMD Optimization** | ⭐ Complete | [05-simd-optimization/README.md](05-simd-optimization/README.md) |
| **Distance Calculation** | 📝 Covered in Automata README | [02-levenshtein-automata/README.md](02-levenshtein-automata/README.md) |
| **Contextual Completion** | ⭐ Complete (feature flag) | [docs/design/contextual-completion-*.md](../design/README.md) |

## Documentation by Component

### Layer 1: Dictionary Layer
**File**: [01-dictionary-layer/README.md](01-dictionary-layer/README.md)

**Contents** (~800 lines):
- Overview of all 9 dictionary implementations
- Quick selection flowchart
- Detailed comparison table
- Performance benchmarks (construction, queries, fuzzy search)
- Memory characteristics
- Thread safety details
- Common use cases with examples
- Integration with automata

**Implementation Guides**:
- [DoubleArrayTrie](01-dictionary-layer/implementations/double-array-trie.md) (~1000 lines)
  - BASE/CHECK algorithm theory
  - Data structure diagrams
  - Construction algorithm
  - 8 usage examples
  - Performance analysis
  - Advanced topics (serialization, Redis, memory-mapping)

- [DoubleArrayTrieChar](01-dictionary-layer/implementations/double-array-trie-char.md) (~950 lines)
  - Why character-level matters (UTF-8 problem)
  - Unicode fundamentals
  - 9 usage examples (emoji, CJK, accented characters)
  - Performance overhead analysis
  - Advanced Unicode topics (normalization, grapheme clusters)

- [DynamicDawg](01-dictionary-layer/implementations/dynamic-dawg.md) (~1100 lines)
  - Runtime DAWG with insert/remove operations
  - Reference counting for safe node deletion
  - Incremental minimization algorithm
  - 8 usage examples
  - Performance analysis vs static DAWG

- [DynamicDawgChar](01-dictionary-layer/implementations/dynamic-dawg-char.md) (~1000 lines)
  - Character-level (Unicode) dynamic DAWG
  - Correct edit distance for multi-byte characters
  - 8 usage examples with emoji, CJK text
  - Performance overhead (5-8% vs byte-level)

- [SuffixAutomaton](01-dictionary-layer/implementations/suffix-automaton.md) (~1050 lines)
  - Substring matching (not just prefix)
  - Online construction algorithm
  - Endpos equivalence and suffix links
  - 8 usage examples (code search, log analysis, DNA sequences)
  - $`\le 2n-1`$ states for $`n`$-character text

- SuffixAutomatonChar (📝 Planned, ~1100 lines)
  - Character-level (Unicode) substring matching
  - Correct multi-byte UTF-8 handling
  - Planned: 8 usage examples with emoji, CJK, accented text
  - Planned: Performance overhead analysis (5-8% vs byte-level)

- [PathMapDictionary](01-dictionary-layer/implementations/pathmap-dictionary.md) (~900 lines)
  - Persistent trie with structural sharing
  - Immutable data structure benefits
  - PathMap library integration
  - 8 usage examples (prototyping, metadata storage)
  - Performance vs specialized tries (2-3x slower but simpler)

### Layer 2: Levenshtein Automata
**File**: [02-levenshtein-automata/README.md](02-levenshtein-automata/README.md)

**Contents** (~850 lines):
- Levenshtein distance theory
- Finite automata approach
- All three algorithm variants (Standard, Transposition, Merge-and-Split)
- State representation
- 6 usage examples
- Performance analysis
- Algorithm selection guide

### Layer 3: Intersection & Traversal
**File**: [03-intersection-traversal/README.md](03-intersection-traversal/README.md)

**Contents** (~700 lines):
- Synchronized traversal algorithm
- State representation with AutomatonZipper
- Traversal strategies (DFS, BFS, priority queue)
- Optimization techniques
- 4 usage examples
- Performance analysis

### Layer 5: SIMD Optimization
**File**: [05-simd-optimization/README.md](05-simd-optimization/README.md)

**Contents** (~900 lines):
- SIMD fundamentals
- Edge label scanning with AVX2/SSE4.1
- Benchmark results (2-4x speedup)
- Implementation details
- Runtime detection
- Platform support
- Performance tuning guide

### Layer 6: Zipper Navigation
**File**: [06-zipper-navigation/README.md](06-zipper-navigation/README.md)

**Contents** (~850 lines):
- Huet's zipper theory
- DictZipper and ValuedDictZipper traits
- All zipper implementations (DoubleArrayTrie, PathMap, Automaton, Intersection)
- 8 detailed usage examples
- Performance analysis
- Advanced patterns
- Zipper-based query iteration

**Zipper Implementations**:
- `DoubleArrayTrieZipper` and `DoubleArrayTrieCharZipper` (index-based navigation)
- `PathMapZipper` (path-based navigation with structural sharing)
- `AutomatonZipper` (Levenshtein state transitions)
- `IntersectionZipper` (dictionary + automaton composition)
- Full support for valued zippers (accessing term metadata)

### Layer 9: Value Storage
**File**: [09-value-storage/README.md](09-value-storage/README.md)

**Contents** (~900 lines):
- Architecture explanation with diagrams
- Memory layout visualization
- Complete trait documentation
- 5 extensive use cases
- Performance analysis (10-100x speedup)
- Implementation details for all backends
- Best practices guide

### Main Overview
**File**: [README.md](README.md)

**Contents** (~450 lines):
- Architecture diagram showing all 9 layers
- Quick start examples
- Performance comparison tables
- Use case decision trees
- Navigation hub with links to all layers

## What's NOT Yet Documented (Future Work)

The following could be added if needed:

### Additional Dictionary Implementation Guides

Detailed implementation guides could be added for these dictionary types:

- [ ] **SuffixAutomatonChar** implementation details (Unicode substring matching, ~1100 lines)
  - Priority: HIGH - Character-level variant of implemented SuffixAutomaton
  - Would match pattern of other *Char guides (DoubleArrayTrieChar, DynamicDawgChar)


- [ ] PathMapDictionaryChar implementation details (Unicode variant)
  - Priority: LOW - PathMapDictionary handles Unicode naturally

**Note**: The layer README provides comprehensive coverage for choosing and using all dictionary types, so these guides are supplementary for users who need deep implementation details.

### Additional Layers

- [ ] Distance Calculation (currently covered within Automata layer)
- [ ] Caching Layer (lower priority - straightforward wrapper)
- [ ] Layer 4: State Pooling (optimization detail, covered in transducer docs)
- [ ] Layer 7: Query Iteration (covered in zipper navigation)
- [ ] Layer 8: Transducer (main API, well-documented in source)

## How to Find What You Need

### "How do I choose a dictionary type?"
→ [01-dictionary-layer/README.md](01-dictionary-layer/README.md) (decision flowchart + comparison table)

### "How do I use DoubleArrayTrie?"
→ [01-dictionary-layer/implementations/double-array-trie.md](01-dictionary-layer/implementations/double-array-trie.md) (8 examples)

### "How do I handle Unicode text?"
→ [01-dictionary-layer/implementations/double-array-trie-char.md](01-dictionary-layer/implementations/double-array-trie-char.md) (complete guide)

### "How does fuzzy matching work?"
→ [02-levenshtein-automata/README.md](02-levenshtein-automata/README.md) (theory + algorithms)

### "Should I use Standard or Transposition algorithm?"
→ [02-levenshtein-automata/README.md#algorithm-selection-guide](02-levenshtein-automata/README.md#algorithm-selection-guide)

### "How do I store values with terms?"
→ [09-value-storage/README.md](09-value-storage/README.md) (complete architecture guide)

### "How do I navigate the trie with zippers?"
→ [06-zipper-navigation/README.md](06-zipper-navigation/README.md) (8 examples)

### "How do DynamicDawg and PathMapDictionary compare?"
→ [01-dictionary-layer/README.md#detailed-comparison-table](01-dictionary-layer/README.md#detailed-comparison-table)

### "What's the performance difference between dictionary types?"
→ [01-dictionary-layer/README.md#performance-benchmarks](01-dictionary-layer/README.md#performance-benchmarks)

### "How does SIMD optimization work?"
→ [05-simd-optimization/README.md](05-simd-optimization/README.md) (complete guide)

### "How do I build code completion with draft text and undo?"
→ [docs/design/contextual-completion-api.md](../design/contextual-completion-api.md) (complete API + examples)

### "How does contextual completion work internally?"
→ [docs/design/contextual-completion-zipper.md](../design/contextual-completion-zipper.md) (architecture + design)

## Documentation Statistics

- **Total markdown files**: 20+ (algorithms + design docs)
- **Total documentation lines**: ~11,000+ lines (excluding planned SuffixAutomatonChar guide)
- **Algorithm layer docs**: ~8,200 lines
- **Design docs**: ~3,000+ lines (contextual completion)
- **Code examples**: 70+ complete, runnable examples
- **Benchmarks**: Comprehensive performance data across all components
- **Test coverage**: 316+ tests with detailed validation
- **Diagrams**: ASCII art for data structures and algorithms
- **Academic references**: 30+ papers and resources
- **Implementation guides**: 6 detailed dictionary guides (~5,900 lines)

## Contributing Documentation

If you'd like to add documentation for components not yet covered in detail:

1. Follow the existing structure (see any layer README as template)
2. Include: Theory, Usage Examples, Performance Analysis, References
3. Add diagrams where helpful (ASCII art is fine)
4. Link from/to related documentation
5. Update this index

---

**Last Updated**: 2025-11-05
**Coverage**: Core components + major features comprehensively documented

**Recent Additions**:
- ✅ DynamicDawg and DynamicDawgChar implementation guides (2025-11-04)
- ✅ SuffixAutomaton implementation guide (2025-11-04)
- ✅ PathMapDictionary implementation guide (2025-11-04)
- ✅ Contextual Completion complete implementation - all 6 phases (2025-11-04)
- ✅ AutomatonZipper and IntersectionZipper documentation (2025-11-04)
- ✅ ValuedDictZipper support for all major dictionary types (2025-11-04)
- ✅ SuffixAutomatonChar implementation + zipper (2025-11-05)
- ✅ 4 new zippers with 46 additional tests (2025-11-05)
- ✅ Version 0.6.0 released (2025-11-05)

**Project Completeness**:
- Dictionary Layer: ✅ 95% (6/7 recommended types have detailed guides; SuffixAutomatonChar planned)
- Levenshtein Automata: ✅ 100% (all variants documented)
- Zipper Navigation: ✅ 100% (all major zippers documented)
- Value Storage: ✅ 100% (architecture + all backends)
- Contextual Completion: ✅ 100% (production-ready, fully tested with PathMapDictionary)
- SIMD Optimization: ✅ 100% (comprehensive guide)
- Intersection/Traversal: ✅ 100% (complete documentation)
