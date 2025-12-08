# MORK FuzzySource Adapter for Approximate String Matching

This document describes how MORK (MeTTa Optimal Reduction Kernel) integrates with liblevenshtein as an external library to enable fuzzy pattern matching in MeTTa queries over PathMap-backed knowledge graphs.

## Table of Contents

1. [Integration Architecture](#integration-architecture)
2. [Executive Summary](#executive-summary)
3. [Architecture Overview](#architecture-overview)
4. [MeTTa Query Examples](#metta-query-examples)
5. [Integration Phases](#integration-phases)
6. [Extended Architecture Layers](#extended-architecture-layers)
7. [MORK Pattern Matching Synergies](#mork-pattern-matching-synergies)
8. [Implementation Guide](#implementation-guide)
9. [Related Documentation](#related-documentation)

---

## Integration Architecture

### What This Is NOT

liblevenshtein is **not embedded** into MORK. MORK does not contain liblevenshtein source code or directly implement fuzzy matching algorithms. The fuzzy matching logic remains in the liblevenshtein library.

### What This IS

Three separate integration layers working together:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: MeTTa Query Syntax                                    │
│  !(match &space (fuzzy "colr" 2 $result) $result)              │
│  User-facing query language for fuzzy matching                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: MORK FuzzySource Adapter                              │
│  MORK/kernel/src/fuzzy_source.rs                               │
│  Implements Source trait, wraps liblevenshtein transducer       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: liblevenshtein/PathMap Integration                    │
│  liblevenshtein-rust/src/dictionary/pathmap.rs                 │
│  PathMapDictionary backend, shared vocabulary with MORK         │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Location | Purpose |
|-------|----------|---------|
| **PathMap Dictionary** | liblevenshtein-rust | liblevenshtein uses PathMap as storage backend |
| **FuzzySource Adapter** | MORK/kernel/src/fuzzy_source.rs | MORK queries liblevenshtein via Source trait |
| **MeTTa Syntax** | MeTTa queries | User-facing `(fuzzy ...)` query syntax |

### Data Flow

```
MeTTa Query: (fuzzy "colr" 2 $result)
       │
       ▼
MORK query_multi_raw()
       │
       ▼
FuzzySource.query()  ←── MORK adapter (implements Source trait)
       │
       ▼
liblevenshtein::Transducer.query()  ←── External library call
       │
       ▼
PathMapDictionary  ←── Shared storage backend
       │
       ▼
PathMap (mmap)  ←── Same data as MORK's BTMSource/ACTSource
```

### Why This Design?

1. **Separation of concerns**: MORK handles pattern matching, liblevenshtein handles fuzzy matching
2. **Shared storage**: PathMap avoids vocabulary duplication between MORK and liblevenshtein
3. **Clean interfaces**: Source trait provides uniform query abstraction for all data sources
4. **Independent development**: Each project can evolve independently
5. **No code duplication**: Fuzzy matching algorithms live only in liblevenshtein

---

## Executive Summary

**Goal**: Integrate liblevenshtein's approximate string matching into MORK to enable fuzzy pattern matching in MeTTa queries over PathMap-backed knowledge graphs.

**Key Finding**: The architectural alignment is strong across all three projects:
- All share zipper-based traversal patterns
- liblevenshtein already has a `pathmap-backend` feature
- MORK's pattern matching (`match2()`, `unify()`) naturally extends to NFA/CFG/structural correction

**Benefits**:
- **Fuzzy symbol matching**: MeTTa queries match "approximately equal" symbols
- **Typo tolerance**: Knowledge graphs tolerate data entry errors
- **Phonetic matching**: Sound-alike queries via verified phonetic rules
- **Ranked results**: Return best matches first with weighted scores
- **Compositional patterns**: WFST composition enables complex fuzzy patterns

---

## Architecture Overview

### Integration Points

```
                         MORK Query Pipeline
                                 |
            +--------------------+--------------------+
            |                    |                    |
    BTMSource (exact)    ACTSource (exact)    FuzzySource (new)
            |                    |                    |
            v                    v                    v
    ReadZipperUntracked  ACTMmapZipper    TransducerZipper (new)
                                              |
                                    +---------+---------+
                                    |         |         |
                                 Standard  Phonetic   Lattice
                                    |         |         |
                                    v         v         v
                              liblevenshtein transducer
                                    |
                                    v
                             PathMapDictionary
```

### Component Locations

| Component | Location | Purpose |
|-----------|----------|---------|
| PathMap | `/home/dylon/Workspace/f1r3fly.io/PathMap/` | Shared trie-based key-value store |
| MORK | `/home/dylon/Workspace/f1r3fly.io/MORK/` | MeTTa query engine and pattern matcher |
| liblevenshtein | `/home/dylon/Workspace/f1r3fly.io/liblevenshtein-rust/` | Approximate string matching automata |

### Three-Tier Correction Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Tier 1: Lexical (liblevenshtein)                        │
│   FST/Levenshtein automata → Word lattice               │
│   Files: src/transducer/, src/lattice/, src/wfst/       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Tier 2: Syntactic (MORK)                                │
│   - CFG rules compiled to pattern/template pairs        │
│   - query_multi_i() matches against lattice             │
│   - transform_multi_multi_() applies corrections        │
│   - Output: Valid parse forest + corrections            │
│   Files: kernel/src/sources.rs, kernel/src/space.rs     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Tier 3: Semantic (Type checker / LLM)                   │
│   Final ranking and validation                          │
└─────────────────────────────────────────────────────────┘
```

---

## MeTTa Query Examples

This section demonstrates how FuzzySource enables fuzzy pattern matching in MeTTa queries. The key benefit is that fuzzy matching participates directly in MORK's unification, allowing single-pass queries that combine fuzzy and exact constraints.

### Knowledge Base Setup

```metta
; Entity definitions with names and types
(= (entity-name "e1") "color")
(= (entity-name "e2") "colour")
(= (entity-name "e3") "collar")
(= (entity-name "e4") "blue")
(= (entity-type "e1") "property")
(= (entity-type "e2") "property")
(= (entity-type "e3") "object")
(= (entity-type "e4") "property")

; Person names (for phonetic examples)
(= (person-name "p1") "Stephen")
(= (person-name "p2") "Stefan")
(= (person-name "p3") "Stephan")
(= (person-name "p4") "Steve")
```

### Basic Fuzzy Matching (Phase A)

```metta
; Find entities with names within edit distance 2 of "colr" (a typo)
!(match &space
    (= (entity-name $entity) (fuzzy "colr" 2))
    $entity)
; Returns: "e1" "e2" "e3"
; Matches: "color" (dist=1), "colour" (dist=2), "collar" (dist=2)
; Does NOT match: "blue" (dist=4 > 2)
```

### Variable Binding with Fuzzy Results

```metta
; Get both the entity ID and the actual matched name
!(match &space
    (= (entity-name $entity) (fuzzy "colr" 2 $matched-name))
    ($entity $matched-name))
; Returns: ("e1" "color") ("e2" "colour") ("e3" "collar")
```

### Combining Fuzzy with Exact Constraints

The real power of FuzzySource in MORK is combining fuzzy matching with additional pattern constraints in a single query:

```metta
; Find entities with names similar to "colr" AND type = "property"
!(match &space
    (= (entity-name $entity) (fuzzy "colr" 2 $name))
    (= (entity-type $entity) "property")
    ($entity $name))
; Returns: ("e1" "color") ("e2" "colour")
; "collar" excluded because entity-type is "object", not "property"
```

This query demonstrates single-pass composition:
1. FuzzySource yields candidates: `color`, `colour`, `collar`
2. For each candidate, MORK checks the type constraint
3. Only entities satisfying BOTH constraints are returned

### Ranked Results (Phase B)

```metta
; Get top 5 matches with edit distances for ranking
!(match &space
    (= (entity-name $entity) (fuzzy-ranked "colr" 2 5 $name $distance))
    ($entity $name $distance))
; Returns: ("e1" "color" 1) ("e2" "colour" 2) ("e3" "collar" 2)
; Results ordered by edit distance (closest matches first)
```

### Phonetic Matching (Phase C)

```metta
; Find person names that sound like "Steven" (phonetic similarity)
!(match &space
    (= (person-name $id) (fuzzy-phonetic "Steven" english $matched))
    ($id $matched))
; Returns: ("p1" "Stephen") ("p2" "Stefan") ("p3" "Stephan") ("p4" "Steve")
; Phonetic rules: "ph" ≈ "f", "v" ≈ "ph", final vowel variations
```

### Complex Multi-Constraint Queries

```metta
; Find properties with names similar to colors (combining multiple patterns)
!(match &space
    (= (entity-name $e1) (fuzzy "colr" 2 $name1))
    (= (entity-type $e1) "property")
    (= (entity-name $e2) (fuzzy "blu" 2 $name2))
    (= (entity-type $e2) "property")
    (($e1 $name1) ($e2 $name2)))
; Returns all pairs of (color-like property, blue-like property)
```

### Why FuzzySource in MORK (vs MeTTaTron-Only)

Without FuzzySource, MeTTaTron would need to orchestrate multiple MORK queries:

```
WITHOUT FuzzySource (MeTTaTron orchestration):
  1. MeTTaTron expands (fuzzy "colr" 2) → ["color", "colour", "collar", ...]
  2. For EACH candidate, MeTTaTron calls MORK:
     !(match &space (= (entity-name $e) "color") (= (entity-type $e) "property") ...)
     !(match &space (= (entity-name $e) "colour") (= (entity-type $e) "property") ...)
     !(match &space (= (entity-name $e) "collar") (= (entity-type $e) "property") ...)
  3. MeTTaTron combines results

WITH FuzzySource (single MORK query):
  !(match &space
      (= (entity-name $e) (fuzzy "colr" 2))
      (= (entity-type $e) "property")
      $e)
```

| Aspect | FuzzySource in MORK | MeTTaTron Orchestration |
|--------|---------------------|------------------------|
| **Query execution** | Single pass | Multiple round-trips |
| **Ranking** | Native (distance in results) | Must be reconstructed |
| **Optimization** | MORK optimizer sees full query | Each sub-query optimized separately |
| **Lattice efficiency** | O(K×N) edge processing | O(K^N) path enumeration |
| **Variable binding** | Unified across constraints | Per-query, then merged |

---

## Integration Phases

### Phase A: FuzzySource (Milestone 1)

**Goal**: Enable fuzzy symbol matching in MORK queries using liblevenshtein's existing transducer.

**Acceptance Criteria**:
1. MORK queries support fuzzy pattern specifiers: `(FUZZY max_dist symbol)`
2. FuzzySource integrates with `query_multi_raw` via Source trait
3. All algorithms: Standard, Transposition, MergeAndSplit
4. Results include edit distance for ranking
5. Latency: <10ms for typical queries

**Files to Create/Modify**:
- `MORK/kernel/src/fuzzy_source.rs` - FuzzySource implementation
- `MORK/kernel/src/fuzzy_zipper.rs` - Zipper adapter for transducer
- `MORK/kernel/src/sources.rs` - Add FuzzySource to ASource enum
- `MORK/kernel/Cargo.toml` - Add liblevenshtein dependency

**Example Usage**:
```metta
; Query with fuzzy symbol matching (distance ≤ 2)
!(match &space (fuzzy "colr" 2 $result) $result)
; Returns: color, colour, collar, ...
```

**Details**: See [fuzzy_source.md](./fuzzy_source.md)

---

### Phase B: Lattice Infrastructure (Milestone 2)

**Goal**: Return structured lattices instead of flat iterators for ranked multi-candidate results.

**Acceptance Criteria**:
1. Transducer returns `Lattice` DAG structure
2. Lattice supports n-best path extraction
3. Edge weights: edit distance + phonetic costs
4. Latency: <50ms for lattice construction

**Files to Create/Modify**:
- `liblevenshtein-rust/src/lattice/mod.rs` - Lattice core module
- `liblevenshtein-rust/src/lattice/builder.rs` - LatticeBuilder
- `liblevenshtein-rust/src/lattice/path_iterator.rs` - Path extraction
- `liblevenshtein-rust/src/transducer/mod.rs` - Add `query_lattice()`
- `MORK/kernel/src/lattice_zipper.rs` - Lattice-to-zipper adapter

**Example Usage**:
```metta
; Query with ranked fuzzy matching
!(match &space (fuzzy-ranked "phone" 3 5) $results)  ; top 5 within dist 3
; Returns: [(phone 0.0) (fone 0.3) (phon 0.5) (phones 1.0) (foam 2.1)]
```

**Details**: See [lattice_integration.md](./lattice_integration.md)

---

### Phase C: Full WFST Integration (Milestone 3)

**Goal**: Complete WFST implementation with weighted transitions, phonetic NFA composition, and FST composition operators.

**Acceptance Criteria**:
1. Weighted transitions with configurable cost functions
2. NFA phonetic regex compilation and composition
3. Phonetic-aware lattice generation via verified rules
4. Latency: <100ms for phonetic-expanded lattice

**Files to Create/Modify**:
- `liblevenshtein-rust/src/wfst/mod.rs` - WFST module root
- `liblevenshtein-rust/src/wfst/weight.rs` - Tropical semiring weights
- `liblevenshtein-rust/src/wfst/nfa.rs` - Phonetic NFA (Thompson's)
- `liblevenshtein-rust/src/wfst/composition.rs` - FST ∘ FST composition
- `MORK/kernel/src/fuzzy_source.rs` - Add WFST support

**Example Usage**:
```metta
; Complex phonetic pattern matching
!(match &space
    (wfst-query
        (pattern "(ph|f)(one|oan)")
        (max-dist 2)
        (phonetic english)
        (top-k 10))
    $results)
```

**Details**: See [wfst_composition.md](./wfst_composition.md)

---

### Phase D: MORK-Based Grammar Correction (Future)

**Goal**: Use MORK's pattern matching as the rule engine for CFG-based grammatical error correction.

**Files to Create/Modify**:
- `liblevenshtein-rust/src/grammar/cfg_compiler.rs` - Compile CFG → MORK patterns
- `liblevenshtein-rust/src/grammar/error_grammar.rs` - Error productions with costs
- `liblevenshtein-rust/src/grammar/lattice_parser.rs` - MORK query_multi_i() integration
- `MORK/kernel/src/space.rs` - Add cost tracking to Space

**Details**: See [grammar_correction.md](./grammar_correction.md)

### Phase D Extensions for Dialogue

Phase D grammar correction can be enhanced for dialogue-aware processing:

**Dialogue-Aware Grammar Rules**:
```metta
; Context-sensitive article correction
Pattern:  (np (dt "a") (n ?N))
Template: (np (dt (context-article ?N &dialogue)) (n ?N))
; Uses dialogue context to determine "a" vs "an" vs "the"

; Pronoun resolution via coreference
Pattern:  (pronoun ?P)
Template: (resolved-entity (resolve-coref ?P &entity-registry))
```

**Integration with Dialogue Layer**:
- `query_multi_i()` can access dialogue history via PathMap
- Entity salience affects pattern matching priority
- Speaker vocabulary personalizes dictionary lookups

---

## Extended Architecture Layers

The three-tier WFST core is extended with additional layers for conversational and LLM agent support:

| Layer | Components | MORK Integration |
|-------|------------|------------------|
| **Dialogue Context** | Turn Tracker, Entity Registry, Topic Graph | MORK patterns for coreference, topic queries |
| **Pragmatic Reasoning** | Speech Act Classifier, Implicature Resolver | MORK rules for indirect speech acts |
| **LLM Integration** | Preprocessor, Postprocessor, Hallucination Detector | MORK fact queries for verification |
| **Agent Learning** | Feedback Collector, Pattern Learner | MORK patterns for learned corrections |

These layers build on top of the three-tier WFST architecture, using PathMap as the shared storage layer and MORK for pattern-based queries.

**See**:
- [Dialogue Context Layer](../../mettail/dialogue/README.md) - Coreference resolution and topic tracking
- [LLM Integration](../../mettail/llm-integration/README.md) - Prompt preprocessing and response validation
- [Agent Learning](../../mettail/agent-learning/README.md) - Feedback integration and personalization

---

## MORK Pattern Matching Synergies

MORK's pattern matching capabilities directly support NFA, CFG, and structural correction work.

### NFA Representation

MORK can encode NFA states and transitions as S-expressions:

```metta
; NFA state encoding
(state q0 [(trans a q0) (trans b q1)])
(state q1 [(trans b q1) (trans ε acc)])
(accepting acc)

; Pattern to find epsilon closure
Pattern: (state ?Q [(trans ε ?R) . ?rest])
Result: Bindings {?Q → q1, ?R → acc}
```

MORK's recursive `match2()` function naturally handles state graph traversal.

### CFG as Pattern/Template Pairs

CFG production rules map directly to MORK's transform mechanism:

```metta
; CFG Rule: NP → DT N
Pattern:  (np (dt ?D) (n ?N))
Template: (noun_phrase ?D ?N)

; Error Production: Article error
Pattern:  (np (dt "a") (n ?N))          ; where is_vowel_initial(?N)
Template: (np (dt "an") (n ?N))
Cost: 0.5
```

MORK's `transform_multi_multi_()` (space.rs:1221) is designed exactly for this pattern.

### Efficient Lattice Processing

MORK's `query_multi_i()` handles lattice inputs efficiently:
- **O(K×N)** edge processing instead of **O(K^N)** path enumeration
- Native PathMap storage with trie-based memoization
- Variable binding via De Bruijn levels for feature propagation

### Key MORK Functions for Integration

| Function | Location | Purpose |
|----------|----------|---------|
| `match2()` | expr/src/lib.rs:921 | Recursive structural pattern matching |
| `unify()` | expr/src/lib.rs:1849 | Robinson's unification with variable binding |
| `query_multi_i()` | kernel/src/space.rs:992 | Multi-source query with lattice support |
| `transform_multi_multi_()` | kernel/src/space.rs:1221 | Pattern → template transformation |

---

## Implementation Guide

### Prerequisites

1. **Rust toolchain**: Latest stable (1.75+)
2. **PathMap**: Built with default features
3. **MORK**: Built with `grounding` feature
4. **liblevenshtein**: Built with `pathmap-backend` feature

### Dependencies

Add to `MORK/kernel/Cargo.toml`:
```toml
[dependencies]
liblevenshtein = { path = "../../../liblevenshtein-rust", features = ["pathmap-backend"] }

[features]
fuzzy = []  # Enable fuzzy matching
```

### Key Interfaces

```rust
// FuzzySource configuration
pub struct FuzzyConfig {
    pub max_distance: usize,
    pub algorithm: Algorithm,  // Standard, Transposition, MergeAndSplit
    pub include_exact: bool,
}

// Lattice structures (Phase B)
pub struct Lattice {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    start: NodeId,
    end: NodeId,
    vocab: IndexMap<Arc<str>, VocabId>,
}

// Weighted transducer trait (Phase B)
pub trait WeightedTransducer<D: Dictionary> {
    fn query_lattice(&self, term: &str, max_distance: usize) -> Lattice;
    fn query_nbest(&self, term: &str, n: usize, max_distance: usize) -> Vec<ScoredCandidate>;
}

// WFST semiring (Phase C)
pub trait Semiring: Clone + Copy + PartialEq {
    const ZERO: Self;
    const ONE: Self;
    fn times(self, other: Self) -> Self;
    fn plus(self, other: Self) -> Self;
}
```

---

## Related Documentation

### In liblevenshtein-rust

- [WFST Architecture](../wfst/architecture.md) - Overall WFST design
- [Lattice Data Structures](../wfst/lattice_data_structures.md) - Lattice implementation spec
- [CFG Grammar Correction](../wfst/cfg_grammar_correction.md) - Grammar-based correction
- [Grammar Correction Design](../design/grammar-correction/MAIN_DESIGN.md) - Full correction pipeline

### In MORK

- `kernel/src/sources.rs` - Source trait and existing implementations
- `kernel/src/space.rs` - Query execution and transformation
- `expr/src/lib.rs` - Pattern matching and unification

### In PathMap

- `pathmap-book/src/` - PathMap documentation
- `src/zipper.rs` - Zipper trait definitions

### Extended Correction Layers

- [Dialogue Context Layer](../../mettail/dialogue/README.md) - Coreference resolution and topic tracking
- [LLM Integration](../../mettail/llm-integration/README.md) - Prompt preprocessing and response validation
- [Agent Learning](../../mettail/agent-learning/README.md) - Feedback integration and personalization
- [Correction WFST Architecture](../../mettail/correction-wfst/01-architecture-overview.md) - Full three-tier architecture overview

---

## Phase Dependencies

```
Phase A (FuzzySource)
    |
    | [FuzzySource working, all algorithms, basic results]
    v
Phase B (Lattice)
    |
    | [Lattice DAG, n-best paths, weighted edges]
    v
Phase C (Full WFST)
    |
    | [Phonetic NFA, composition, verified rules]
    v
Phase D (Grammar)
    |
    | [CFG via MORK patterns, structural correction]
    v
[Future: Neural LM Integration]
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PathMap zipper API incompatibility | Prototype zipper adapter early; fallback to copy-based |
| WFST composition complexity | Start with simplified composition; optimize later |
| Phonetic NFA explosion | Limit accepted string length; lazy expansion |
| Performance degradation | Benchmark each phase; set latency gates |
| Memory overhead from lattices | Lazy path iteration; arena allocation |

---

## Changelog

- **2025-12-08**: Added comprehensive MeTTa Query Examples section
  - Added knowledge base setup examples with entities and types
  - Added basic fuzzy matching, variable binding, and constraint composition examples
  - Added ranked results (Phase B) and phonetic matching (Phase C) previews
  - Added architectural comparison: FuzzySource in MORK vs MeTTaTron-only approach
- **2024-12-06**: Extended architecture documentation
  - Added Extended Architecture Layers section for dialogue/LLM/agent learning
  - Added Phase D Extensions for Dialogue with dialogue-aware grammar rules
  - Added cross-references to MeTTaIL extended correction documentation
- **2024-12-05**: Initial documentation created
  - Phases A-D defined with acceptance criteria
  - MORK synergies documented
