# Research Initiatives

**Status**: Planning Phase
**Last Updated**: 2025-11-11
**Priority**: Future work (post-Phase 6 completion)

---

## Overview

This document outlines three research initiatives identified during the OptimizedDawg deprecation analysis (2025-11-11). Each represents a substantial multi-week effort requiring careful design, implementation, and empirical validation.

### Quick Reference

| Initiative | Effort | Priority | Potential Impact |
|-----------|--------|----------|------------------|
| [SIMD Edge Search](#1-simd-edge-search-in-dynamicdawg) | 2-4 weeks | Medium | 5-15% query speedup (estimated) |
| [Hybrid Storage](#2-hybrid-arenaper-node-storage) | 4-6 weeks | Low | 20-30% memory reduction for immutable dicts |
| [DAT/DAWG Hybrid](#3-datdawg-hybrid-with-suffix-sharing) | 8-12 weeks | High | Best-in-class space/time balance |

---

## 1. SIMD Edge Search in DynamicDawg

### Motivation

OptimizedDawg contains SIMD-optimized edge search logic that could potentially benefit DynamicDawg's query performance. Before deprecating OptimizedDawg entirely, we should extract this optimization and benchmark it.

### Background

**Current Implementation (OptimizedDawg)**:
- Uses SIMD instructions (AVX2/SSE4.2) for parallel edge comparison
- Located in `src/dictionary/dawg_optimized.rs`
- Processes 16-32 edges simultaneously
- Falls back to binary search for small edge counts

**Target (DynamicDawg)**:
- Currently uses binary search for edge lookups
- Located in `src/dictionary/dynamic_dawg.rs`
- Per-node edge storage (not arena-based)

### Hypothesis

**H₀**: SIMD edge search provides ≥10% query speedup for DynamicDawg at typical edge counts (4-8 edges/node)

**Alternative outcomes**:
- SIMD overhead dominates at small edge counts → no benefit
- CPU frequency scaling negates SIMD advantage → marginal benefit
- Memory access patterns matter more than comparison speed → no benefit

### Research Questions

1. **What is the edge count distribution in typical dictionaries?**
   - Measure on big.txt corpus (32K words)
   - Measure on system dictionary (/usr/share/dict/words)
   - Identify optimal SIMD threshold

2. **What is the SIMD performance breakeven point?**
   - Benchmark SIMD vs binary search at edge counts: 2, 4, 8, 16, 32, 64
   - Measure on both hot and cold cache conditions

3. **What is the end-to-end query impact?**
   - Full transducer queries at distances 1, 2, 3
   - Realistic workload (Zipfian distribution)
   - Compare to baseline DynamicDawg

### Methodology

#### Phase 1: Analysis (Week 1)
1. **Profile current DynamicDawg** to identify edge lookup hotspots
   ```bash
   RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
     cargo flamegraph --bench backend_comparison -- --profile-time=60
   ```
2. **Measure edge count distribution** in typical dictionaries
   ```rust
   // Add instrumentation to DynamicDawg
   let mut edge_counts = HashMap::new();
   for node in dict.nodes() {
       *edge_counts.entry(node.edges().len()).or_insert(0) += 1;
   }
   ```
3. **Analyze flamegraph** to determine if edge search is a bottleneck
   - If edge search < 5% of total time → SKIP (won't impact end-to-end)
   - If edge search ≥ 5% of total time → PROCEED to Phase 2

#### Phase 2: Microbenchmark (Week 2)
1. **Extract SIMD logic** from OptimizedDawg
   ```rust
   // Create benches/edge_search_comparison.rs
   fn bench_edge_search_simd(c: &mut Criterion) { ... }
   fn bench_edge_search_binary(c: &mut Criterion) { ... }
   ```
2. **Benchmark at various edge counts**: 2, 4, 8, 16, 32, 64
3. **Test cache sensitivity**: warm vs cold cache
4. **Determine breakeven point**: edge count where SIMD becomes faster

**Decision criteria**:
- If SIMD faster at typical edge counts (4-8) → PROCEED to Phase 3
- If SIMD slower or marginal (< 5% improvement) → ABANDON

#### Phase 3: Integration (Week 3)
1. **Implement SIMD edge search in DynamicDawg**
   ```rust
   #[cfg(target_feature = "avx2")]
   fn find_edge_simd(&self, label: u8) -> Option<usize> { ... }

   fn find_edge(&self, label: u8) -> Option<usize> {
       if self.edges.len() >= SIMD_THRESHOLD {
           self.find_edge_simd(label)
       } else {
           self.find_edge_binary(label)
       }
   }
   ```
2. **Add feature flag**: `simd-edge-search` (opt-in)
3. **Ensure correctness**: All tests pass with SIMD enabled

#### Phase 4: Validation (Week 4)
1. **End-to-end benchmarks** on corpus workloads
   ```bash
   # Baseline
   cargo bench --bench corpus_benchmarks --features rand

   # With SIMD
   cargo bench --bench corpus_benchmarks --features rand,simd-edge-search
   ```
2. **Compare results** using Criterion's baseline comparison
3. **Profile with flamegraph** to verify optimization is active

**Decision criteria**:
- If ≥10% query speedup → MERGE and enable by default
- If 5-10% speedup → MERGE but keep as opt-in feature
- If <5% speedup → ABANDON (not worth maintenance burden)

### Success Criteria

**Minimum viable improvement**: 10% query speedup at distance=2 on realistic workload

**Regression tolerance**: No memory usage increase, no construction time regression

### Resources Required

- **Time**: 2-4 weeks (depending on Phase 1 outcome)
- **Hardware**: AVX2-capable CPU (already available: Xeon E5-2699 v3)
- **Tools**: Criterion, flamegraph, perf
- **Corpus**: big.txt (already downloaded)

### Risks

1. **SIMD not beneficial**: Overhead dominates at typical edge counts → 2 weeks wasted
2. **Cache effects dominate**: Memory access patterns matter more than SIMD → optimization ineffective
3. **Portability issues**: SIMD requires AVX2, not available on all platforms → feature flag complexity

### References

- OptimizedDawg SIMD implementation: `src/dictionary/dawg_optimized.rs:250-320`
- DynamicDawg edge lookup: `src/dictionary/dynamic_dawg.rs:180-200`
- Edge count analysis script: (to be created)

---

## 2. Hybrid Arena/Per-Node Storage

### Motivation

OptimizedDawg's arena-based edge storage reduces memory overhead for immutable dictionaries but is incompatible with DynamicDawg's mutability. A hybrid approach could offer space savings for read-only dictionaries while maintaining the DynamicDawg API.

### Background

**Current Storage Models**:

1. **DynamicDawg (Per-Node)**:
   - Each node owns its edges: `Vec<Edge>`
   - Advantages: Mutable, straightforward
   - Disadvantages: 24 bytes overhead per node (Vec capacity + len + ptr)

2. **OptimizedDawg (Arena)**:
   - All edges in single `Vec<Edge>`, nodes store offset+length
   - Advantages: 8 bytes overhead per node
   - Disadvantages: Immutable, incompatible with insert/remove

3. **DoubleArrayTrie (BASE+CHECK)**:
   - Separate arrays for all nodes
   - Advantages: O(1) transitions, 8 bytes/state
   - Disadvantages: Slow construction, no suffix sharing

### Hypothesis

**H₀**: A hybrid storage model can reduce memory usage by 20-30% for immutable dictionaries while maintaining DynamicDawg's query performance

**Design sketch**:
```rust
enum DawgStorage {
    Mutable(Vec<Node>),        // Per-node Vec<Edge>
    Immutable(Arena<Edge>),    // Arena with offset+length
}

impl DynamicDawg {
    fn compact_to_immutable(&mut self) {
        // Convert Mutable → Immutable storage
        // Freezes dictionary (no more mutations)
    }
}
```

### Research Questions

1. **What is the memory overhead breakdown?**
   - Measure current DynamicDawg memory usage per component
   - Estimate savings from arena storage
   - Calculate break-even dictionary size

2. **What is the performance impact of dual storage modes?**
   - Benchmark query performance: Mutable vs Immutable storage
   - Measure cache effects (arena may improve locality)

3. **What is the API complexity increase?**
   - How to expose immutable conversion?
   - How to prevent mutations after compaction?
   - Type system enforcement vs runtime checks?

### Methodology

#### Phase 1: Analysis (Weeks 1-2)
1. **Memory profiling** of current DynamicDawg
   ```bash
   valgrind --tool=massif cargo test --release --test memory_usage
   ```
2. **Component breakdown**:
   - Node struct overhead
   - Edge Vec capacity waste
   - Alignment padding
3. **Estimate theoretical savings** from arena storage

**Decision criteria**:
- If potential savings ≥ 20% → PROCEED to Phase 2
- If potential savings < 20% → ABANDON (not worth complexity)

#### Phase 2: Prototype (Weeks 3-4)
1. **Implement hybrid storage** in feature branch
   ```rust
   struct DynamicDawgImmutable {
       nodes: Vec<NodeCompact>,
       edges: Vec<Edge>,
   }

   impl Dictionary for DynamicDawgImmutable { ... }
   ```
2. **Implement conversion** from DynamicDawg
   ```rust
   impl DynamicDawg {
       fn into_immutable(self) -> DynamicDawgImmutable { ... }
   }
   ```
3. **Verify correctness**: All tests pass with immutable storage

#### Phase 3: Benchmarking (Week 5)
1. **Memory benchmarks**
   ```bash
   cargo bench --bench memory_footprint
   ```
2. **Query performance benchmarks**
   ```bash
   cargo bench --bench corpus_benchmarks
   ```
3. **Construction overhead**
   ```bash
   cargo bench --bench construction_benchmarks
   ```

**Decision criteria**:
- If ≥20% memory reduction AND no query regression → PROCEED to Phase 4
- If <20% memory reduction OR >5% query regression → ABANDON

#### Phase 4: API Design (Week 6)
1. **Design user-facing API**
   - Option A: Separate type (`DynamicDawgImmutable`)
   - Option B: Mode flag (runtime check)
   - Option C: Type state pattern (compile-time enforcement)
2. **Document API** with examples
3. **Integration testing** with transducers

### Success Criteria

**Minimum viable improvement**: 20% memory reduction with no query performance regression

**API requirements**: Clear, type-safe, hard to misuse

### Resources Required

- **Time**: 4-6 weeks
- **Tools**: Valgrind (massif), heaptrack, Criterion
- **Complexity**: Medium-High (dual storage modes increase code complexity)

### Risks

1. **Insufficient savings**: Overhead from dual modes negates memory savings
2. **API complexity**: Hard to explain when to use immutable vs mutable
3. **Maintenance burden**: Two code paths to maintain and test
4. **Cache effects unpredictable**: Arena storage may hurt or help cache performance

### References

- OptimizedDawg arena storage: `src/dictionary/dawg_optimized.rs:50-100`
- DynamicDawg node structure: `src/dictionary/dynamic_dawg.rs:30-60`
- Memory profiling guide: `docs/benchmarks/MEMORY_PROFILING.md` (to be created)

---

## 3. DAT/DAWG Hybrid with Suffix Sharing

### Motivation

Combine the strengths of Double-Array Trie (O(1) transitions) with DAWG (suffix sharing for space efficiency). This represents a novel data structure that could offer best-in-class space/time characteristics.

### Background

**Double-Array Trie (DAT)**:
- O(1) state transitions via BASE+CHECK arrays
- 8 bytes per state (BASE[4] + CHECK[4])
- No suffix sharing → redundant storage for common endings

**DAWG (Directed Acyclic Word Graph)**:
- Suffix sharing → minimal states
- O(log k) transitions via binary search (k = edge count)
- Space-efficient but slower queries

**MP DAT (Minimal Prefix DAT)**:
- DAT for prefix trie + TAIL array for suffixes
- Different from DAT/DAWG hybrid (no suffix sharing in DAT part)

### Hypothesis

**H₀**: A true DAT/DAWG hybrid can achieve:
- Query speed: Within 2× of pure DAT (O(1) transitions with suffix compression overhead)
- Space efficiency: Within 2× of pure DAWG (BASE+CHECK overhead with suffix sharing)
- Construction time: Better than DoubleArrayTrie (fewer states due to suffix sharing)

**Novel contribution**: First implementation combining BASE+CHECK arrays with suffix sharing

### Research Questions

1. **How to detect suffix sharing opportunities in BASE+CHECK array?**
   - Suffix detection algorithm compatible with DAT structure
   - Hash-based suffix equivalence checking
   - Trade-off: detection cost vs space savings

2. **How to represent shared suffixes in BASE+CHECK framework?**
   - Option A: Special CHECK values indicating suffix pointers
   - Option B: Separate suffix table (like MP DAT's TAIL)
   - Option C: Virtual states with redirection

3. **What are the construction complexity implications?**
   - Can we build incrementally or require batch construction?
   - How does suffix sharing interact with BASE array compaction?

4. **What are the real-world space/time trade-offs?**
   - Benchmark on multiple corpora (big.txt, system dictionary)
   - Compare to both DAT and DAWG baselines

### Methodology

#### Phase 1: Literature Review & Design (Weeks 1-3)
1. **Literature survey**:
   - Read DAT papers (Aoe 1989, Yata 2007, etc.)
   - Read DAWG papers (Crochemore 1985, etc.)
   - Search for existing DAT+DAWG combinations
2. **Design document**:
   - Array layout (BASE, CHECK, SUFFIX?)
   - State transition algorithm
   - Suffix detection algorithm
   - Construction algorithm
3. **Complexity analysis**:
   - Theoretical bounds on space/time
   - Expected performance on real dictionaries

**Deliverable**: `docs/research/DAT_DAWG_HYBRID_DESIGN.md` (30+ pages)

#### Phase 2: Prototype Implementation (Weeks 4-7)
1. **Implement core data structure**
   ```rust
   pub struct DatDawgHybrid {
       base: Vec<i32>,
       check: Vec<i32>,
       suffix_map: HashMap<Vec<u8>, u32>,
       // ...
   }
   ```
2. **Implement construction algorithm**
   - Build prefix trie with BASE+CHECK
   - Detect and merge suffix sharing opportunities
   - Compact BASE array
3. **Implement query algorithm**
   - Follow BASE+CHECK for prefix
   - Resolve suffix pointers
   - Verify final string

#### Phase 3: Correctness Validation (Week 8)
1. **Comprehensive testing**:
   - Unit tests for edge cases
   - Property-based tests (all inserted terms retrievable)
   - Fuzzing with random dictionaries
2. **Equivalence testing**:
   ```rust
   // Verify same results as reference DAWG
   for term in dictionary {
       assert_eq!(dat_dawg.contains(term), dawg.contains(term));
   }
   ```

#### Phase 4: Performance Benchmarking (Weeks 9-10)
1. **Construction benchmarks** vs DAT, DAWG, DynamicDawg
2. **Query benchmarks** vs all backends
3. **Memory measurements** vs all backends
4. **Flamegraph analysis** to identify bottlenecks

#### Phase 5: Optimization (Weeks 11-12)
1. **Based on flamegraph results**:
   - Optimize hot paths
   - Cache-friendly data layout
   - SIMD opportunities?
2. **Re-benchmark** after optimizations
3. **Document findings**

### Success Criteria

**Space efficiency**: Within 2× of DynamicDawg (best current space)
**Query speed**: Within 2× of DoubleArrayTrie (O(1) transitions)
**Construction time**: Better than DoubleArrayTrie (fewer states)

**Novel contribution**: Publishable algorithm (conference/journal paper)

### Resources Required

- **Time**: 8-12 weeks (full research project)
- **Expertise**: Deep understanding of DAT and DAWG algorithms
- **Tools**: All existing benchmark infrastructure
- **Corpus**: Multiple test corpora (big.txt, system dict, specialized domains)

### Risks

1. **No practical benefit**: Overhead of suffix resolution negates O(1) transition advantage
2. **Construction intractable**: Suffix detection + BASE compaction too expensive
3. **Already exists**: May find existing implementation in literature
4. **Complexity not justified**: May perform worse than simple DynamicDawg

### Success Probability

**Estimated**: 40-60%

**Why uncertain**:
- Novel data structure (no prior implementation known)
- Trade-offs may not favor hybrid approach
- Implementation complexity very high

**Mitigation**: Phase 1 design phase includes feasibility analysis with early exit criteria

### References

- DAT literature: `docs/research/DOUBLE_ARRAY_TRIE_LITERATURE.md` (to be created)
- DAWG literature: `docs/research/DAWG_LITERATURE.md` (to be created)
- MP DAT comparison: `docs/research/evaluation-methodology/MP_DAT_IMPLEMENTATION_PLAN.md`

---

## Prioritization Framework

### Evaluation Criteria

| Initiative | ROI | Risk | Complexity | Time | Total Score |
|-----------|-----|------|------------|------|-------------|
| SIMD Edge Search | 3/5 | 2/5 | 2/5 | 4/5 | 11/20 |
| Hybrid Storage | 2/5 | 3/5 | 4/5 | 3/5 | 12/20 |
| DAT/DAWG Hybrid | 5/5 | 4/5 | 5/5 | 1/5 | 15/20 |

**Scoring**:
- ROI: Return on investment (impact if successful)
- Risk: Probability of failure (higher = riskier)
- Complexity: Implementation difficulty (higher = harder)
- Time: Time efficiency (higher = faster completion)

### Recommended Order

**If goal is quick wins**: SIMD Edge Search → Hybrid Storage → DAT/DAWG Hybrid

**If goal is novel contribution**: DAT/DAWG Hybrid → SIMD Edge Search → Hybrid Storage

**If goal is practical improvement**: SIMD Edge Search → (skip others unless SIMD succeeds)

---

## Getting Started

### Prerequisites

All initiatives require:
- ✅ Phase 6 complete (100% feature parity)
- ✅ OptimizedDawg deprecated (reduces maintenance burden)
- ✅ Corpus infrastructure ready (big.txt, holbrook.dat)
- ✅ Benchmark suite established

### Before Beginning Any Initiative

1. **Review this document** thoroughly
2. **Create detailed design document** in `docs/research/[initiative-name]/`
3. **Estimate time commitment** realistically
4. **Get stakeholder approval** (if applicable)
5. **Create feature branch**: `research/[initiative-name]`

### Template Structure

Each initiative should have:
```
docs/research/[initiative-name]/
├── README.md              # Overview and motivation
├── DESIGN.md              # Detailed design document
├── METHODOLOGY.md         # Research methodology
├── RESULTS.md             # Benchmark results and findings
├── CONCLUSION.md          # Final analysis and decision
└── references/            # Papers, articles, prior art
```

---

## Questions?

Contact: (project maintainer contact info)

**Last reviewed**: 2025-11-11
**Next review**: After first initiative completion
