# WallBreaker Implementation Plan

## Document Purpose

This document provides detailed, phase-by-phase implementation plans for three different approaches to adding WallBreaker functionality to liblevenshtein-rust. Each approach represents a different trade-off between implementation effort, performance, and architectural complexity.

## Table of Contents

1. [Option A: Full SCDAWG Implementation](#option-a-full-scdawg-implementation)
2. [Option B: Hybrid Approach](#option-b-hybrid-approach-recommended)
3. [Option C: Index-Based Approach](#option-c-index-based-approach)
4. [Common Components](#common-components)
5. [Testing Strategy](#testing-strategy)

---

## Option A: Full SCDAWG Implementation

**Effort:** 21-31 weeks (5-8 months full-time)
**Performance:** Maximum (matches paper's theoretical guarantees)
**Complexity:** High
**Recommendation:** For long-term, production deployment where performance is critical

### Phase 1: Foundation (4-6 weeks)

#### Task 1.1: Extend Dictionary Traits (1 week)
**File:** `/src/dictionary/mod.rs`

```rust
/// New trait for dictionaries supporting substring queries
pub trait SubstringDictionary: Dictionary {
    /// Find all occurrences of an exact substring in dictionary words
    fn find_exact_substring(&self, pattern: &str) -> Vec<SubstringMatch>;

    /// Check if a substring exists in any dictionary word
    fn contains_substring(&self, pattern: &str) -> bool {
        !self.find_exact_substring(pattern).is_empty()
    }
}

/// Represents an exact substring match in a dictionary word
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstringMatch {
    /// The complete dictionary term containing the substring
    pub term: String,
    /// Starting position of the match in the term
    pub start_pos: usize,
    /// Ending position (exclusive) of the match in the term
    pub end_pos: usize,
    /// Reference to the dictionary node at the match position
    pub node_position: NodePosition,
}

/// Opaque handle to a position within the dictionary structure
#[derive(Debug, Clone)]
pub struct NodePosition {
    // Implementation-specific position data
    inner: Box<dyn Any + Send + Sync>,
}
```

**Tests:**
- Trait compilation
- Default implementations
- API usability

#### Task 1.2: Implement Substring Search for Existing Backends (2-3 weeks)

**Priority Order:**
1. **SuffixAutomaton** (easiest - infrastructure exists)
   - File: `/src/dictionary/suffix_automaton.rs`
   - Leverage existing suffix links
   - Expose through public API

2. **DynamicDawg** (supports runtime queries)
   - File: `/src/dictionary/dynamic_dawg.rs`
   - Build substring search on existing structure
   - May require auxiliary data structure

3. **DoubleArrayTrie** (performance-critical)
   - File: `/src/dictionary/double_array_trie.rs`
   - Challenge: no inherent substring support
   - May need offline preprocessing

**Deliverable:** At least one backend (SuffixAutomaton) with full substring search

#### Task 1.3: Pattern Splitting Utilities (1 week)
**File:** `/src/transducer/pattern_splitting.rs` (new)

```rust
/// Split a pattern into b+1 pieces for WallBreaker algorithm
pub fn split_pattern(pattern: &[u8], max_distance: usize) -> Vec<Vec<u8>> {
    let num_pieces = max_distance + 1;
    let piece_size = pattern.len() / num_pieces;

    // Baseline: equal-size pieces. Future scientific variants can choose
    // split points based on:
    // - Character frequency
    // - Expected match likelihood
    // - Boundary conditions

    // Simple equal division for now
    pattern.chunks(piece_size.max(1))
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Analyze split quality and suggest optimizations
pub fn analyze_split(pattern: &[u8], pieces: &[Vec<u8>]) -> SplitQuality {
    // Heuristics for split effectiveness
}
```

**Tests:**
- Various pattern lengths
- Different error bounds
- Edge cases (pattern shorter than b+1)

#### Task 1.4: Documentation & Examples (1 week)
- API documentation
- Usage examples
- Integration guide

---

### Phase 2: SCDAWG Backend (8-12 weeks)

#### Task 2.1: Research & Design (2 weeks)

**Research Tasks:**
- Study SCDAWG construction algorithms
- Analyze memory vs. performance trade-offs
- Review reference implementations (if available)
- Design node structure and edge storage

**Deliverable:** Design document with:
- Node structure specification
- Construction algorithm pseudo-code
- Memory layout analysis
- Performance estimates

#### Task 2.2: Core SCDAWG Implementation (4-6 weeks)
**File:** `/src/dictionary/scdawg.rs` (new)

```rust
/// Symmetric Compact Directed Acyclic Word Graph
pub struct SCDawgDictionary {
    nodes: Vec<SCDawgNode>,
    root_index: usize,
    term_count: usize,
}

/// Node in SCDAWG with bidirectional edges
pub struct SCDawgNode {
    /// Forward edges: character → child node index
    forward_edges: Vec<(u8, usize)>,

    /// Backward edges: character → parent node index
    backward_edges: Vec<(u8, usize)>,

    /// Suffix link for construction and navigation
    suffix_link: Option<usize>,

    /// Prefix link for reverse navigation
    prefix_link: Option<usize>,

    /// Maximum string length reachable from this node
    max_length: usize,

    /// Is this node a word boundary?
    is_final: bool,

    /// Optional value storage
    value: Option<()>,  // Generic later
}

impl SCDawgDictionary {
    /// Construct SCDAWG from sorted word list
    pub fn from_sorted_terms(terms: &[&str]) -> Self {
        // Inenaga et al. algorithm with bidirectional edges
    }
}
```

**Sub-tasks:**
1. Basic node structure (1 week)
2. Forward edge construction (1-2 weeks)
3. Backward edge construction (1-2 weeks)
4. Suffix/prefix link computation (1 week)
5. Optimization & validation (1 week)

#### Task 2.3: Bidirectional Traversal Traits (2 weeks)
**File:** `/src/dictionary/mod.rs`

```rust
/// Extended trait for nodes supporting bidirectional traversal
pub trait BidirectionalDictionaryNode: DictionaryNode {
    /// Get all parent nodes (reverse transitions)
    fn reverse_transition(&self, label: Self::Unit) -> Vec<Self>;

    /// Iterate over all incoming edges
    fn reverse_edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_>;

    /// Get current position/depth in the trie
    fn depth(&self) -> Option<usize>;
}
```

**Implementation:**
- Implement for `SCDawgNode`
- Document semantics
- Add examples

#### Task 2.4: Integration & Testing (2-4 weeks)
- Dictionary trait implementation
- SubstringDictionary implementation
- Comprehensive test suite
- Benchmark against existing backends

---

### Phase 3: WallBreaker Algorithm (6-10 weeks)

#### Task 3.1: Extension Filters (3-4 weeks)
**File:** `/src/transducer/wallbreaker_filters.rs` (new)

```rust
/// Filter for controlling left-extension from a match point
pub struct LeftExtensionFilter {
    remaining_pattern: Vec<u8>,  // Pattern piece to the left
    error_budget: usize,          // Errors allowed for this extension
    algorithm: Algorithm,
}

impl LeftExtensionFilter {
    /// Check if a reverse transition is valid
    pub fn accepts_reverse_transition(
        &self,
        current_state: &State,
        dict_unit: u8,
    ) -> Option<State> {
        // Similar to forward transition but in reverse
        // Uses "reversed" Levenshtein filter from paper
    }
}

/// Symmetric filter for right-extension
pub struct RightExtensionFilter {
    // Similar to LeftExtensionFilter
}
```

**Tests:**
- Correctness vs. standard filter
- Edge cases (boundaries, empty patterns)
- Performance benchmarks

#### Task 3.2: Match Extension Logic (2-3 weeks)
**File:** `/src/transducer/wallbreaker_extension.rs` (new)

```rust
/// Represents a partial match being extended
pub struct PartialMatch<N: BidirectionalDictionaryNode> {
    node: N,
    matched_text: String,
    errors_used: usize,
    left_pattern: Vec<u8>,
    right_pattern: Vec<u8>,
}

/// Extend a match leftward
pub fn extend_left<N: BidirectionalDictionaryNode>(
    partial: PartialMatch<N>,
    filter: &LeftExtensionFilter,
) -> Vec<PartialMatch<N>> {
    let mut extensions = Vec::new();

    for (label, parent_node) in partial.node.reverse_edges() {
        if let Some(next_state) = filter.accepts_reverse_transition(&current_state, label) {
            extensions.push(PartialMatch {
                node: parent_node,
                matched_text: format!("{}{}", label as char, partial.matched_text),
                errors_used: next_state.min_distance().unwrap_or(0),
                left_pattern: /* updated */,
                right_pattern: partial.right_pattern.clone(),
            });
        }
    }

    extensions
}

/// Symmetric function for right extension
pub fn extend_right<N: BidirectionalDictionaryNode>(...) { }
```

#### Task 3.3: WallBreaker Query Iterator (2-3 weeks)
**File:** `/src/transducer/wallbreaker_query.rs` (new)

```rust
/// Iterator implementing WallBreaker search strategy
pub struct WallBreakerQueryIterator<N: BidirectionalDictionaryNode> {
    /// Pattern split into b+1 pieces
    pattern_pieces: Vec<Vec<u8>>,

    /// Exact matches found for each piece
    piece_matches: Vec<Vec<SubstringMatch>>,

    /// Pending partial matches being extended
    pending: VecDeque<PartialMatch<N>>,

    /// Maximum allowed edit distance
    max_distance: usize,

    /// Distance algorithm to use
    algorithm: Algorithm,

    /// Results cache (deduplicate)
    seen: FxHashSet<String>,
}

impl<N: BidirectionalDictionaryNode> Iterator for WallBreakerQueryIterator<N> {
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(partial) = self.pending.pop_front() {
            // If fully extended (no remaining pattern), verify and return
            if partial.left_pattern.is_empty() && partial.right_pattern.is_empty() {
                if partial.node.is_final() && partial.errors_used <= self.max_distance {
                    if self.seen.insert(partial.matched_text.clone()) {
                        return Some(Candidate {
                            term: partial.matched_text,
                            distance: partial.errors_used,
                        });
                    }
                }
                continue;
            }

            // Extend left if needed
            if !partial.left_pattern.is_empty() {
                let filter = LeftExtensionFilter::new(
                    &partial.left_pattern,
                    self.max_distance - partial.errors_used,
                    self.algorithm,
                );
                self.pending.extend(extend_left(partial.clone(), &filter));
            }

            // Extend right if needed
            if !partial.right_pattern.is_empty() {
                let filter = RightExtensionFilter::new(
                    &partial.right_pattern,
                    self.max_distance - partial.errors_used,
                    self.algorithm,
                );
                self.pending.extend(extend_right(partial, &filter));
            }
        }

        None
    }
}
```

---

### Phase 4: Integration & Testing (4-6 weeks)

#### Task 4.1: Public API Integration (1-2 weeks)
**File:** `/src/transducer/mod.rs`

```rust
impl<D> Transducer<D>
where
    D: SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
{
    /// Query using WallBreaker algorithm
    pub fn query_wallbreaker(
        &self,
        term: &str,
        max_distance: usize,
    ) -> WallBreakerQueryIterator<D::Node> {
        WallBreakerQueryIterator::new(
            self.dictionary.root(),
            term,
            max_distance,
            self.algorithm,
        )
    }

    /// Query with automatic strategy selection
    pub fn query_auto(
        &self,
        term: &str,
        max_distance: usize,
    ) -> Box<dyn Iterator<Item = Candidate>> {
        // Choose traditional vs WallBreaker based on heuristics
        if max_distance >= 4 && term.len() >= 50 {
            Box::new(self.query_wallbreaker(term, max_distance))
        } else {
            Box::new(self.query(term, max_distance))
        }
    }
}
```

#### Task 4.2: Comprehensive Testing (2-3 weeks)

**Unit Tests:**
- Pattern splitting correctness
- Substring search accuracy
- Extension filter correctness
- Bidirectional traversal

**Integration Tests:**
- Full query end-to-end
- Correctness vs. traditional algorithm
- Edge cases (empty patterns, max_distance=0, etc.)
- Large dictionaries

**Property-Based Tests:**
```rust
#[quickcheck]
fn wallbreaker_matches_traditional(pattern: String, max_dist: usize) -> bool {
    let dict = build_test_dictionary();
    let traditional_results: HashSet<_> =
        dict.query(&pattern, max_dist).map(|c| c.term).collect();
    let wallbreaker_results: HashSet<_> =
        dict.query_wallbreaker(&pattern, max_dist).map(|c| c.term).collect();

    traditional_results == wallbreaker_results
}
```

#### Task 4.3: Performance Benchmarking (1-2 weeks)

**Benchmark Suite:**
- Vary error bounds (1, 2, 4, 8, 16)
- Vary pattern lengths (10, 50, 100, 200 chars)
- Vary dictionary sizes (1K, 10K, 100K, 1M terms)
- Compare: Traditional vs. WallBreaker vs. Paper results

**File:** `/benches/wallbreaker_benchmarks.rs`

```rust
fn bench_wallbreaker_vs_traditional(c: &mut Criterion) {
    let mut group = c.benchmark_group("wallbreaker_comparison");

    for max_dist in [1, 2, 4, 8, 16] {
        for pattern_len in [10, 50, 100] {
            let pattern = generate_pattern(pattern_len);

            group.bench_with_input(
                BenchmarkId::new("traditional", format!("d{}_p{}", max_dist, pattern_len)),
                &(&dict, &pattern, max_dist),
                |b, (d, p, dist)| {
                    b.iter(|| d.query(p, *dist).collect::<Vec<_>>())
                },
            );

            group.bench_with_input(
                BenchmarkId::new("wallbreaker", format!("d{}_p{}", max_dist, pattern_len)),
                &(&dict, &pattern, max_dist),
                |b, (d, p, dist)| {
                    b.iter(|| d.query_wallbreaker(p, *dist).collect::<Vec<_>>())
                },
            );
        }
    }
}
```

#### Task 4.4: Documentation & Examples (1 week)
- Algorithm explanation
- API documentation
- Usage examples
- Performance characteristics guide
- Migration guide

---

## Option B: Hybrid Approach (RECOMMENDED)

**Effort:** 6-9 weeks
**Performance:** 60-70% of full SCDAWG
**Complexity:** Medium
**Recommendation:** Best effort/benefit ratio, good for most use cases

### Overview

Reuse existing SuffixAutomaton for substring search, but use simplified bidirectional extension:
- **Forward extension:** Use existing automaton infrastructure
- **Backward extension:** Reconstruct from matched term strings (slower but functional)

### Phase 1: Foundation (2-3 weeks)

#### Task 1.1: Extend SuffixAutomaton for Substring Search (1-2 weeks)
**File:** `/src/dictionary/suffix_automaton.rs`

SuffixAutomaton already has most infrastructure needed. Expose substring search:

```rust
impl SuffixAutomaton {
    /// Find all exact substring matches
    pub fn find_substring(&self, pattern: &str) -> Vec<SubstringMatch> {
        let mut matches = Vec::new();
        let pattern_bytes = pattern.as_bytes();

        // Traverse from root matching pattern
        let mut node_idx = 0;
        for &byte in pattern_bytes {
            if let Some(&next_idx) = self.nodes[node_idx]
                .edges
                .iter()
                .find(|(label, _)| *label == byte)
                .map(|(_, idx)| idx)
            {
                node_idx = next_idx;
            } else {
                return matches;  // No match
            }
        }

        // Pattern matched! Now find all terms containing this substring
        self.collect_terms_containing(node_idx, pattern, &mut matches);
        matches
    }

    fn collect_terms_containing(
        &self,
        node_idx: usize,
        matched_substring: &str,
        matches: &mut Vec<SubstringMatch>,
    ) {
        // Use suffix links to find all occurrences
        // Reconstruct full terms
    }
}
```

**Advantage:** SuffixAutomaton already built for this!

#### Task 1.2: Pattern Splitting (1 week)
Same as Option A, Task 1.3

---

### Phase 2: Hybrid WallBreaker (3-4 weeks)

#### Task 2.1: Simplified Bidirectional Extension (2-3 weeks)
**File:** `/src/transducer/hybrid_wallbreaker.rs` (new)

```rust
/// Hybrid approach: no true bidirectional dictionary needed
pub struct HybridWallBreakerIterator<N: DictionaryNode> {
    // ... similar to full WallBreaker
}

/// Extend left by reconstructing from term string
fn extend_left_hybrid(
    dict: &impl Dictionary,
    matched_term: &str,
    match_start: usize,
    remaining_left_pattern: &[u8],
    error_budget: usize,
) -> Vec<PartialMatch> {
    let left_text = &matched_term[..match_start];

    // Use standard Levenshtein distance on strings
    if levenshtein_distance(left_text.as_bytes(), remaining_left_pattern) <= error_budget {
        // Valid extension
    }

    // No need for node-by-node traversal
    // Trade-off: slower but works without bidirectional dictionary
}

/// Forward extension uses normal automaton (fast)
fn extend_right_hybrid(
    node: N,
    remaining_right_pattern: &[u8],
    error_budget: usize,
    algorithm: Algorithm,
) -> Vec<PartialMatch<N>> {
    // Use existing transition_state infrastructure
    // This is fast and reuses existing code
}
```

**Key Insight:** We don't need perfect symmetry. Backward can be slower since we only do it for exact matches (small number).

#### Task 2.2: Integration (1 week)
- Public API
- Query iterator
- Result deduplication

---

### Phase 3: Testing & Validation (2 weeks)

Same as Option A Phase 4, but:
- Focus on correctness
- Document performance trade-offs
- Benchmark shows where hybrid approach wins

---

## Option C: Index-Based Approach

**Effort:** 3-4 weeks
**Performance:** 40-50% of full
**Complexity:** Low
**Recommendation:** Quick win, proof-of-concept

### Phase 1: Implementation (2-3 weeks)

#### Task 1.1: Build Substring Index (1 week)
**File:** `/src/dictionary/substring_index.rs` (new)

```rust
/// Auxiliary index for substring queries
pub struct SubstringIndex {
    /// Map: substring → list of (term, position)
    index: HashMap<String, Vec<(String, usize)>>,
    min_substring_len: usize,
}

impl SubstringIndex {
    /// Build index from dictionary
    pub fn from_dictionary(dict: &impl Dictionary, min_len: usize) -> Self {
        let mut index = HashMap::new();

        for term in dict.iter_terms() {
            // Index all substrings of length >= min_len
            for start in 0..term.len() {
                for end in (start + min_len)..=term.len() {
                    let substring = &term[start..end];
                    index
                        .entry(substring.to_string())
                        .or_insert_with(Vec::new)
                        .push((term.clone(), start));
                }
            }
        }

        SubstringIndex { index, min_substring_len: min_len }
    }

    pub fn lookup(&self, pattern: &str) -> Vec<(String, usize)> {
        self.index.get(pattern).cloned().unwrap_or_default()
    }
}
```

#### Task 1.2: Two-Phase Query (1 week)
**File:** `/src/transducer/indexed_wallbreaker.rs` (new)

```rust
pub struct IndexedWallBreakerIterator {
    candidates: Vec<(String, usize)>,  // From index lookup
    current: usize,
    query: Vec<u8>,
    max_distance: usize,
}

impl Iterator for IndexedWallBreakerIterator {
    type Item = Candidate;

    fn next(&mut self) -> Option<Candidate> {
        while self.current < self.candidates.len() {
            let (term, _pos) = &self.candidates[self.current];
            self.current += 1;

            // Phase 2: Verify with standard Levenshtein
            let distance = levenshtein_distance(term.as_bytes(), &self.query);
            if distance <= self.max_distance {
                return Some(Candidate {
                    term: term.clone(),
                    distance,
                });
            }
        }
        None
    }
}
```

**Advantage:** Simple, works with any dictionary backend

#### Task 1.3: Integration & Optimization (1 week)
- API integration
- Memory optimization (compress index)
- Cache frequently queried substrings

---

### Phase 2: Testing (1 week)

- Correctness tests
- Performance benchmarks
- Memory usage analysis

---

## Common Components

### Testing Infrastructure

**Required for all options:**

```rust
// File: /tests/wallbreaker_correctness.rs

/// Generate test cases with known distances
fn generate_test_cases() -> Vec<(String, String, usize)> {
    // (pattern, term, distance)
}

/// Verify WallBreaker matches traditional algorithm
fn test_correctness_against_traditional() {
    let dict = load_test_dictionary();
    for (pattern, expected_term, expected_dist) in generate_test_cases() {
        let traditional: HashSet<_> = dict.query(&pattern, expected_dist)
            .map(|c| c.term)
            .collect();

        let wallbreaker: HashSet<_> = dict.query_wallbreaker(&pattern, expected_dist)
            .map(|c| c.term)
            .collect();

        assert_eq!(traditional, wallbreaker,
                   "Mismatch for pattern: {}", pattern);
    }
}
```

### Documentation Template

All options need:
1. API docs with examples
2. Algorithm explanation
3. Performance characteristics
4. When to use vs. traditional
5. Migration guide

---

## Timeline Comparison

| Phase | Option A (Full) | Option B (Hybrid) | Option C (Index) |
|-------|----------------|-------------------|------------------|
| Foundation | 4-6 weeks | 2-3 weeks | N/A |
| Core Implementation | 8-12 weeks | 3-4 weeks | 2-3 weeks |
| Integration | 6-10 weeks | N/A | N/A |
| Testing | 4-6 weeks | 2 weeks | 1 week |
| **TOTAL** | **21-31 weeks** | **6-9 weeks** | **3-4 weeks** |

---

## Risk Assessment

### Option A Risks
- **High:** SCDAWG construction complexity
- **Medium:** Memory overhead
- **Low:** API design

### Option B Risks
- **Medium:** Backward extension performance
- **Low:** Integration complexity
- **Low:** Correctness

### Option C Risks
- **High:** Index memory overhead
- **Medium:** Performance limitations
- **Low:** Implementation complexity

---

## Recommendation

**Start with Option B (Hybrid Approach)** because:
1. Delivers 60-70% of full benefits in 25-40% of the time
2. Proves WallBreaker value before deep investment
3. Can upgrade to Option A later if needed
4. Low risk, good ROI

**Consider Option C** for:
- Quick proof-of-concept
- Validating WallBreaker applicability
- Projects with relaxed performance requirements

**Reserve Option A** for:
- Production systems with demanding performance needs
- After Option B validation shows clear benefits
- Long-term strategic implementation

---

**Last Updated:** 2025-11-06
**Next Steps:** Select approach and begin Phase 1 implementation
