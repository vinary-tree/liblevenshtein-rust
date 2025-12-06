# Unified Correction WFST Architecture

This document introduces the multi-tier Weighted Finite State Transducer (WFST)
architecture for correction across written, spoken, and programming languages.
The architecture includes the foundational three-tier WFST plus additional layers
for dialogue context, LLM integration, and adaptive learning.

**Sources**:
- liblevenshtein: `/home/dylon/Workspace/f1r3fly.io/liblevenshtein-rust/`
- MORK: `/home/dylon/Workspace/f1r3fly.io/MORK/`
- MeTTaTron: `/home/dylon/Workspace/f1r3fly.io/MeTTa-Compiler/`

**Related Integration Docs**:
- [MORK Integration Overview](../../integration/mork/README.md) - Phase A-D implementation
- [FuzzySource Implementation](../../integration/mork/fuzzy_source.md) - Phase A details
- [Lattice Integration](../../integration/mork/lattice_integration.md) - Phase B details
- [WFST Composition](../../integration/mork/wfst_composition.md) - Phase C details
- [Grammar Correction](../../integration/mork/grammar_correction.md) - Phase D details
- [PathMap Integration](../../integration/pathmap/README.md) - Shared storage layer

**Extended Architecture Docs**:
- [Dialogue Context Layer](../dialogue/README.md) - Discourse semantics and coreference
- [LLM Integration Layer](../llm-integration/README.md) - Prompt preprocessing and response validation
- [Agent Learning Layer](../agent-learning/README.md) - Feedback collection and online learning

**Original WFST Documentation** (detailed implementation specs):
- [WFST Architecture](../../wfst/architecture.md) - Complete system design (~2400 lines)
- [CFG Grammar Correction](../../wfst/cfg_grammar_correction.md) - Error grammar formalism (~1900 lines)
- [Lattice Parsing](../../wfst/lattice_parsing.md) - Earley parsing on lattices (~1050 lines)
- [Lattice Data Structures](../../wfst/lattice_data_structures.md) - Rust implementations (~550 lines)
- [NFA Phonetic Regex](../../wfst/nfa_phonetic_regex.md) - Phonetic pattern matching
- [References](../../wfst/references/papers.md) - 35+ cited papers

**Programming Language Correction** (5-layer design with SMT repair):
- [Grammar Correction Design](../../design/grammar-correction/README.md) - Navigation guide
- [Complete Design Document](../../design/grammar-correction/MAIN_DESIGN.md) - 5-layer architecture (~5100 lines)
- [Theoretical Analysis](../../design/grammar-correction/theoretical-analysis/README.md) - 28+ formal theorems

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Extended Architecture Overview](#extended-architecture-overview)
3. [Three-Tier WFST Core](#three-tier-wfst-core)
4. [Dialogue Context Layer](#dialogue-context-layer)
5. [LLM Integration Layer](#llm-integration-layer)
6. [Agent Learning Layer](#agent-learning-layer)
7. [MORK Integration Phases](#mork-integration-phases)
8. [Why Layered Correction?](#why-layered-correction)
9. [PathMap as Universal Storage](#pathmap-as-universal-storage)
10. [Performance Considerations](#performance-considerations)

---

## Problem Statement

Error correction spans multiple domains with distinct requirements:

| Domain | Error Types | Correction Needs |
|--------|-------------|------------------|
| **Written Text** | Typos, spelling, grammar | Dictionary lookup, context |
| **Spoken Language** | Phonetic confusion, homophones | Phoneme similarity, ASR lattices |
| **Programming Languages** | Syntax errors, type mismatches | Grammar validation, semantic types |

A unified architecture must handle all these while maintaining:
- **Efficiency**: Real-time correction for interactive use
- **Accuracy**: High precision without false corrections
- **Extensibility**: Easy addition of new languages/domains

---

## Extended Architecture Overview

The complete correction architecture extends the three-tier WFST core with additional layers
for dialogue context, LLM integration, and adaptive learning. This multi-layer design enables:

- **Conversational correction**: Understanding context across dialogue turns
- **LLM agent integration**: Pre/post-processing for language model interactions
- **Personalization**: Learning from user feedback and preferences

```
┌─────────────────────────────────────────────────────────────────────────┐
│          EXTENDED CORRECTION ARCHITECTURE (Full Stack)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    DIALOGUE CONTEXT LAYER                          │ │
│  │  Turn History │ Entity Registry │ Topic Graph │ Speaker Models     │ │
│  │  [Discourse semantics, coreference resolution, topic tracking]     │ │
│  │  See: ../dialogue/README.md                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌───────────────────────────┼────────────────────────────────────────┐ │
│  │                           ▼                                        │ │
│  │               THREE-TIER WFST CORE                                 │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │ │
│  │  │ Tier 1:      │→ │ Tier 2:      │→ │ Tier 3:      │             │ │
│  │  │ Lexical      │  │ Syntactic    │  │ Semantic     │             │ │
│  │  │ (libleven.)  │  │ (MORK/CFG)   │  │ (MeTTaIL)    │             │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │ │
│  │  [Edit distance, phonetic rules, grammar validation, type checking]│ │
│  │  See: #three-tier-wfst-core below                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌───────────────────────────┼────────────────────────────────────────┐ │
│  │                           ▼                                        │ │
│  │               PRAGMATIC REASONING LAYER                            │ │
│  │  Speech Act Classifier │ Implicature Resolver │ Relevance Ranker   │ │
│  │  [Intent detection, indirect speech acts, contextual relevance]    │ │
│  │  See: ../dialogue/04-pragmatic-reasoning.md                        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌───────────────────────────┼────────────────────────────────────────┐ │
│  │                           ▼                                        │ │
│  │               LLM INTEGRATION LAYER                                │ │
│  │  ┌────────────────────────────────────────────────────────────┐   │ │
│  │  │ PROMPT PREPROCESSING                                       │   │ │
│  │  │ Correction → Coreference → Context Injection → RAG         │   │ │
│  │  └────────────────────────────┬───────────────────────────────┘   │ │
│  │                               ▼                                    │ │
│  │                       ┌──────────────┐                            │ │
│  │                       │   LLM API    │                            │ │
│  │                       └──────┬───────┘                            │ │
│  │                              ▼                                     │ │
│  │  ┌────────────────────────────────────────────────────────────┐   │ │
│  │  │ RESPONSE POSTPROCESSING                                    │   │ │
│  │  │ Coherence Check → Fact Verification → Hallucination Detect │   │ │
│  │  └────────────────────────────────────────────────────────────┘   │ │
│  │  See: ../llm-integration/README.md                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌───────────────────────────┼────────────────────────────────────────┐ │
│  │                           ▼                                        │ │
│  │               AGENT LEARNING LAYER                                 │ │
│  │  Feedback Collection │ Pattern Learning │ User Preferences         │ │
│  │  Online Learning │ Threshold Adaptation │ Model Versioning         │ │
│  │  [Adaptive correction weights, personalized dictionaries]          │ │
│  │  See: ../agent-learning/README.md                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Layer Summary

| Layer | Components | Purpose |
|-------|------------|---------|
| **Dialogue Context** | Turn History, Entity Registry, Topic Graph | Multi-turn conversation tracking |
| **WFST Core** | Lexical, Syntactic, Semantic Tiers | Fundamental correction pipeline |
| **Pragmatic** | Speech Acts, Implicatures, Relevance | Intent understanding |
| **LLM Integration** | Preprocessing, Postprocessing | LLM agent support |
| **Agent Learning** | Feedback, Patterns, Preferences | Adaptive personalization |

---

## Three-Tier WFST Core

The foundational correction system uses three progressively refined tiers:

```
┌─────────────────────────────────────────────────────────────────────┐
│           UNIFIED CORRECTION WFST ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT: Erroneous text (written/spoken/code)                        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Tier 1: Lexical Correction                 │   │
│  │                       (liblevenshtein)                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│  │  │ Edit Dist.  │  │  Phonetic   │  │   Custom    │          │   │
│  │  │ Automata    │  │   Rules     │  │   Weights   │          │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │   │
│  │         └─────────────────┼─────────────────┘                │   │
│  │                           ▼                                   │   │
│  │              ┌────────────────────────┐                      │   │
│  │              │   Candidate Lattice    │                      │   │
│  │              └───────────┬────────────┘                      │   │
│  └──────────────────────────┼───────────────────────────────────┘   │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Tier 2: Syntactic Validation                 │   │
│  │                     (CFG + MORK/PathMap)                      │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │                     MORK Space                          │ │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │   │
│  │  │  │  Grammar    │  │  Pattern    │  │   Bloom +   │     │ │   │
│  │  │  │  Rules      │  │  Matching   │  │   LRU       │     │ │   │
│  │  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │ │   │
│  │  │         └─────────────────┼─────────────────┘           │ │   │
│  │  │                           ▼                             │ │   │
│  │  │               ┌─────────────────────┐                   │ │   │
│  │  │               │      PathMap        │                   │ │   │
│  │  │               │  (Shared Storage)   │                   │ │   │
│  │  │               └──────────┬──────────┘                   │ │   │
│  │  └──────────────────────────┼──────────────────────────────┘ │   │
│  │                             ▼                                 │   │
│  │              ┌────────────────────────┐                      │   │
│  │              │  Syntactically Valid   │                      │   │
│  │              │     Candidates         │                      │   │
│  │              └───────────┬────────────┘                      │   │
│  └──────────────────────────┼───────────────────────────────────┘   │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                Tier 3: Semantic Type Checking                 │   │
│  │                (MeTTaIL / MeTTaTron / Rholang)                │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │                    MeTTaTron                            │ │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │   │
│  │  │  │   MeTTa     │  │   Type      │  │  Behavioral │     │ │   │
│  │  │  │   Atomspace │  │   Checking  │  │   Predicates│     │ │   │
│  │  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │ │   │
│  │  │         └─────────────────┼─────────────────┘           │ │   │
│  │  │                           ▼                             │ │   │
│  │  │        ┌──────────────────────────────────┐             │ │   │
│  │  │        │ OSLF Predicate Evaluation        │             │ │   │
│  │  │        │ (structural + behavioral types)  │             │ │   │
│  │  │        └───────────────┬──────────────────┘             │ │   │
│  │  └────────────────────────┼────────────────────────────────┘ │   │
│  │                           │                                   │   │
│  │  ┌────────────────────────┼────────────────────────────────┐ │   │
│  │  │                   Rholang Bridge                        │ │   │
│  │  │  PathMap <-> MeTTa State <-> Rholang Par                │ │   │
│  │  │  (Enables cross-language semantic checking)              │ │   │
│  │  └────────────────────────┼────────────────────────────────┘ │   │
│  │                           ▼                                   │   │
│  │              ┌────────────────────────┐                      │   │
│  │              │  Semantically Valid    │                      │   │
│  │              │     Corrections        │                      │   │
│  │              └───────────┬────────────┘                      │   │
│  └──────────────────────────┼───────────────────────────────────┘   │
│                             ▼                                        │
│  OUTPUT: Ranked corrections with confidence scores                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Tier Summary

| Tier | Component | Purpose | Speed |
|------|-----------|---------|-------|
| **1** | liblevenshtein | Lexical candidates via edit distance | Fastest |
| **2** | MORK/PathMap | Syntactic filtering via CFG | Fast |
| **3** | MeTTaIL/Rholang | Semantic type checking | Thorough |

---

## Dialogue Context Layer

The dialogue context layer extends correction capabilities for multi-turn conversations,
enabling context-aware corrections that consider the full discourse history.

**Full documentation**: [Dialogue Context Documentation](../dialogue/README.md)

### Components

| Component | Purpose | PathMap Key |
|-----------|---------|-------------|
| **Turn Tracker** | Conversation history with sliding window | `/dialogue/{id}/turn/` |
| **Entity Registry** | Cross-turn entity tracking and coreference | `/dialogue/{id}/entity/` |
| **Topic Graph** | Discourse structure and topic continuity | `/dialogue/{id}/topic/` |
| **Speaker Models** | Per-participant vocabulary and style | `/dialogue/{id}/speaker/` |

### Key Capabilities

1. **Coreference Resolution**: Resolves pronouns and references across turns
   - Pronoun resolution: "it" → "the document"
   - Definite description binding: "the file" → specific file entity
   - See: [Coreference Resolution](../dialogue/02-coreference-resolution.md)

2. **Discourse Coherence**: Validates corrections maintain conversation flow
   - Coherence relations (Elaboration, Question-Answer, Contrast)
   - Topic continuity checking
   - See: [Discourse Semantics](../dialogue/01-discourse-semantics.md)

3. **Topic Management**: Tracks and validates topic shifts
   - Topic extraction and clustering
   - Drift detection and validation
   - See: [Topic Management](../dialogue/03-topic-management.md)

### Integration with WFST Core

```
Dialogue Context → WFST Core
─────────────────────────────
• Entity salience affects candidate ranking
• Topic keywords influence lexical tier
• Speaker vocabulary personalizes dictionary
• Discourse coherence validates semantic tier
```

---

## LLM Integration Layer

The LLM integration layer provides preprocessing and postprocessing for language model
interactions, ensuring corrected input and validated output.

**Full documentation**: [LLM Integration Documentation](../llm-integration/README.md)

### Preprocessing Pipeline

Transforms user input before LLM processing:

```
User Input → Correction → Coreference → Context Injection → RAG → LLM Prompt
```

| Stage | Function | Documentation |
|-------|----------|---------------|
| **Correction** | Three-tier WFST fixes errors | This document |
| **Coreference** | Resolves references using dialogue context | [02-coreference-resolution.md](../dialogue/02-coreference-resolution.md) |
| **Context Injection** | Formats dialogue history for prompt | [04-context-injection.md](../llm-integration/04-context-injection.md) |
| **RAG** | Retrieves relevant knowledge | [04-context-injection.md](../llm-integration/04-context-injection.md) |

**Documentation**: [Prompt Preprocessing](../llm-integration/01-prompt-preprocessing.md)

### Postprocessing Pipeline

Validates and corrects LLM responses:

```
LLM Response → Coherence → Fact Check → Hallucination → Correction → Final Output
```

| Stage | Function | Documentation |
|-------|----------|---------------|
| **Coherence Check** | Validates response addresses query | [02-output-postprocessing.md](../llm-integration/02-output-postprocessing.md) |
| **Fact Verification** | Checks factual claims against knowledge | [03-hallucination-detection.md](../llm-integration/03-hallucination-detection.md) |
| **Hallucination Detection** | Identifies fabricated content | [03-hallucination-detection.md](../llm-integration/03-hallucination-detection.md) |
| **Correction** | Three-tier WFST fixes errors | This document |

**Documentation**: [Output Postprocessing](../llm-integration/02-output-postprocessing.md)

### Hallucination Types

| Type | Detection Method | Example |
|------|------------------|---------|
| **Fabricated Fact** | Knowledge base mismatch | Invented statistics |
| **Nonexistent Entity** | Entity registry lookup | Made-up person names |
| **Temporal Error** | Timeline validation | Wrong date claims |
| **Contradiction** | Dialogue consistency | Conflicting statements |

**Documentation**: [Hallucination Detection](../llm-integration/03-hallucination-detection.md)

---

## Agent Learning Layer

The agent learning layer provides adaptive correction through feedback collection,
pattern learning, and online weight updates.

**Full documentation**: [Agent Learning Documentation](../agent-learning/README.md)

### Components

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| **Feedback Collection** | Capture user responses to corrections | [01-feedback-collection.md](../agent-learning/01-feedback-collection.md) |
| **Pattern Learning** | Extract error patterns from feedback | [02-pattern-learning.md](../agent-learning/02-pattern-learning.md) |
| **User Preferences** | Model individual user characteristics | [03-user-preferences.md](../agent-learning/03-user-preferences.md) |
| **Online Learning** | Incremental weight and threshold updates | [04-online-learning.md](../agent-learning/04-online-learning.md) |

### Feedback Flow

```
User Action → Signal Detection → Normalization → Learning Update
───────────────────────────────────────────────────────────────
Accept (fast)     → Strong positive   → +0.8 to +1.0
Accept (slow)     → Weak positive     → +0.3 to +0.5
Modify            → Correction signal → Pattern extraction
Reject            → Negative signal   → -0.8 to -1.0
Ignore            → Weak negative     → -0.1 to -0.3
```

### Learned Adaptations

| Adaptation | Scope | Effect |
|------------|-------|--------|
| **Edit Weights** | Global/User | Character-level substitution costs |
| **Feature Weights** | Global/User | Ranking factor importance |
| **Thresholds** | User/Domain | Correction confidence cutoffs |
| **Vocabulary** | User | Personal dictionary additions |
| **Patterns** | Global | Recognized error→correction pairs |

### PathMap Storage

```
/learning/
    /patterns/                 # Learned error patterns
    /user/{user_id}/          # Per-user profiles
        /vocabulary/          # Personal dictionary
        /weights/             # Personalized weights
        /thresholds/          # Confidence thresholds
    /models/                  # Version-controlled models
        /current/             # Active model
        /checkpoints/         # Historical snapshots
```

---

## MORK Integration Phases

The three-tier architecture is implemented through four progressive phases, each building
on the previous. See the [MORK Integration Overview](../../integration/mork/README.md) for
complete implementation details.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MORK Integration Phases                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase A: FuzzySource Trait                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Trait abstraction for fuzzy dictionary backends          │   │
│  │  • PathMap + DAWG + DoubleArrayTrie implementations         │   │
│  │  • Integration point: liblevenshtein → MORK                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  Phase B: Lattice Infrastructure                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Weighted DAG for multi-candidate representation          │   │
│  │  • K-best path extraction (Dijkstra-based)                  │   │
│  │  • LatticeZipper for MORK ProductZipper integration         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  Phase C: WFST Composition                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Semiring weights (Tropical, Log, Probability)            │   │
│  │  • Phonetic NFA via Thompson's construction                 │   │
│  │  • FST ∘ FST ∘ Trie composition operators                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  Phase D: Grammar Correction                                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • CFG rules as pattern/template pairs                      │   │
│  │  • MORK match2() for structural matching                    │   │
│  │  • query_multi_i() for O(K×N) lattice processing            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase A: FuzzySource Trait

**Documentation**: [FuzzySource Implementation](../../integration/mork/fuzzy_source.md)

The `FuzzySource` trait provides a unified interface for fuzzy dictionary lookups across
different storage backends:

```rust
/// Unified trait for fuzzy dictionary sources.
pub trait FuzzySource {
    /// Query with fuzzy matching up to max_distance.
    fn fuzzy_lookup(&self, query: &[u8], max_distance: u8)
        -> impl Iterator<Item = (Vec<u8>, u8)>;
}
```

**Implementations**:
- `PathMap`: Trie-based storage with zipper navigation
- `DynamicDawg` / `DynamicDawgChar`: SIMD-optimized for runtime updates
- `DoubleArrayTrie` / `DoubleArrayTrieChar`: Optimized for static dictionaries

**Integration Point**: Tier 1 (Lexical Correction) uses FuzzySource for candidate generation.

### Phase B: Lattice Infrastructure

**Documentation**: [Lattice Integration](../../integration/mork/lattice_integration.md)

Lattices represent the space of correction candidates as weighted directed acyclic graphs:

```
Query Term: "teh"
    │
    ▼
Transducer::query_lattice()
    │
    │ Builds DAG of candidates with weighted edges
    ▼
Lattice { nodes, edges, vocab }
    │
    ▼
LatticeZipper (MORK adapter)
    │
    │ Iterates paths by total weight
    ▼
ProductZipper → Unification → Ranked Results
```

**Key Components**:
- `Lattice`: Core DAG structure with vocabulary deduplication
- `LatticeBuilder`: Incremental construction API
- `PathIterator` / `k_best()`: Path extraction algorithms
- `LatticeZipper`: Adapter for MORK's ProductZipper

**Integration Point**: Bridge between Tier 1 and Tier 2.

### Phase C: WFST Composition

**Documentation**: [WFST Composition](../../integration/mork/wfst_composition.md)

Full Weighted Finite State Transducer infrastructure with phonetic NFA composition:

```
Query Pattern: "(ph|f)(o|oa)(n|ne)"
    │
    ▼
PhoneticNfa::compile()      ← Thompson's construction
    │
    ▼
ComposedAutomaton::new(phonetic_nfa, levenshtein, dictionary)
    │
    │ FST ∘ FST ∘ Trie composition
    ▼
Lattice with phonetic-weighted edges
```

**Key Concepts**:

| Semiring | ⊕ (combine) | ⊗ (extend) | Use Case |
|----------|-------------|------------|----------|
| **Tropical** | min | + | Shortest path (Viterbi) |
| **Log** | log-sum-exp | + | Probabilistic (forward-backward) |
| **Probability** | + | × | Raw probabilities |

**Integration Point**: Tier 1 phonetic expansion before Tier 2 filtering.

### Phase D: Grammar Correction

**Documentation**: [Grammar Correction](../../integration/mork/grammar_correction.md)

CFG-based error correction using MORK's pattern matching as the rule engine:

```metta
; CFG Rule: Subject-Verb Agreement Error
Pattern:  (s (np ?Subj :number singular) (vp (v ?V :number plural) ?Rest))
Template: (s (np ?Subj :number singular) (vp (v (singularize ?V)) ?Rest))
Cost: 1.0
```

**Key MORK Functions**:

| Function | Location | Purpose |
|----------|----------|---------|
| `match2()` | expr/src/lib.rs:921 | Recursive structural matching |
| `unify()` | expr/src/lib.rs:1849 | Variable binding + constraints |
| `query_multi_i()` | kernel/src/space.rs:992 | O(K×N) lattice queries |
| `transform_multi_multi_()` | kernel/src/space.rs:1221 | Pattern→template application |

**Integration Point**: Tier 2 (Syntactic Validation) rule engine.

### Phase Integration Summary

| Phase | Tier | Primary Function | Output |
|-------|------|------------------|--------|
| **A** | 1 | Fuzzy lookup | Raw candidates |
| **B** | 1→2 | Lattice construction | Weighted DAG |
| **C** | 1 | Phonetic expansion | Expanded candidates |
| **D** | 2 | Grammar filtering | Valid corrections |

---

## Why Layered Correction?

### Progressive Refinement

Each tier reduces the candidate set before the next:

```
Input Error: "teh" in "teh cat sat"
    │
    ▼ Tier 1 (Lexical)
Candidates: [the, tea, ten, tee, tech, ...]  (~100 candidates)
    │
    ▼ Tier 2 (Syntactic)
Valid in context: [the, tea]  (grammar allows determiner or noun)
    │
    ▼ Tier 3 (Semantic)
Best correction: "the"  (matches "cat sat" semantic context)
```

### Computational Efficiency

| Tier | Complexity | Candidates |
|------|------------|------------|
| 1 | O(n × d) | Generate many |
| 2 | O(n × g) | Filter structurally |
| 3 | O(n × t) | Verify semantically |

Where:
- n = number of candidates
- d = edit distance bound
- g = grammar size
- t = type checking cost

By filtering at each tier, expensive semantic checks only run on valid candidates.

### Separation of Concerns

Each tier has distinct expertise:

| Tier | Knowledge Required |
|------|-------------------|
| 1 | Character/phoneme similarity |
| 2 | Language grammar |
| 3 | Type system, domain semantics |

---

## PathMap as Universal Storage

PathMap serves as the shared storage layer across all tiers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PathMap Integration                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  liblevenshtein │                                            │
│  │  Dictionary     │──────┐                                     │
│  └─────────────────┘      │                                     │
│                           │                                     │
│  ┌─────────────────┐      │      ┌─────────────────────────┐   │
│  │  MORK Grammar   │──────┼─────>│       PathMap           │   │
│  │  Rules          │      │      │  (Trie-based Storage)   │   │
│  └─────────────────┘      │      └─────────────────────────┘   │
│                           │                 │                   │
│  ┌─────────────────┐      │                 │                   │
│  │  MeTTa Type     │──────┘                 ▼                   │
│  │  Predicates     │           ┌────────────────────────────┐  │
│  └─────────────────┘           │  Shared Query Interface     │  │
│                                │  - Pattern matching          │  │
│                                │  - Fuzzy lookup              │  │
│                                │  - Type queries              │  │
│                                └────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Benefits

1. **No Serialization Overhead**: All tiers operate on same data format
2. **Shared Indexing**: Pattern matching works uniformly
3. **Cross-Tier Queries**: Type predicates can reference grammar rules
4. **Merkleization**: Content-addressed caching across tiers

---

## Performance Considerations

### Tier 1 Optimizations

- **SIMD**: DynamicDawg uses SIMD for parallel character comparison
- **Bloom Filter**: Fast negative lookups before trie traversal
- **Lazy Iteration**: Candidates generated on-demand

### Tier 2 Optimizations

- **Lattice Parsing**: 3-10x speedup over exhaustive enumeration
- **LRU Cache**: Hot grammar rules cached
- **Incremental Parsing**: Reuse partial parses for nearby errors

### Tier 3 Optimizations

- **Predicate Caching**: Common type queries cached
- **Lazy Evaluation**: Type checking on-demand
- **Parallel Checking**: Independent candidates checked in parallel

### Memory Budget

| Component | Typical Size |
|-----------|-------------|
| Dictionary (English) | 50-100 MB |
| Grammar (Programming Language) | 10-50 MB |
| Type Predicates | 5-20 MB |
| Working Set (LRU) | 10-50 MB |

---

## Summary

The extended correction architecture provides:

1. **Comprehensive Correction**: Lexical, syntactic, and semantic (three-tier WFST core)
2. **Conversational Support**: Multi-turn dialogue context and coreference resolution
3. **LLM Integration**: Pre/post-processing for language model agent interactions
4. **Adaptive Learning**: Feedback-driven personalization and online weight updates
5. **Efficient Filtering**: Each tier reduces candidates before expensive processing
6. **Unified Storage**: PathMap as shared layer across all components

### Core Integration Points

| From | To | Interface |
|------|-----|-----------|
| **liblevenshtein** | PathMap | FuzzySource trait |
| **MORK** | PathMap | Native storage backend |
| **MeTTaTron** | PathMap | Type predicate storage |
| **Rholang** | PathMap | Par conversion |
| **Dialogue Context** | WFST Core | Entity salience, speaker vocab |
| **LLM Layer** | WFST Core | Pre/post-processing pipeline |
| **Agent Learning** | All Layers | Adaptive weights and thresholds |

### Layer Dependencies

```
Dialogue Context Layer
        │
        ▼
Three-Tier WFST Core ←────────────────────────────┐
        │                                          │
        ▼                                          │
Pragmatic Reasoning Layer                          │
        │                                          │
        ▼                                          │
LLM Integration Layer                              │
        │                                          │
        ▼                                          │
Agent Learning Layer ──────────────────────────────┘
        │                  (feedback loop)
        ▼
   PathMap Storage
```

---

## References

### WFST Core Documentation

- See [02-tier1-lexical-correction.md](./02-tier1-lexical-correction.md) for Tier 1 details
- See [03-tier2-syntactic-validation.md](./03-tier2-syntactic-validation.md) for Tier 2 details
- See [04-tier3-semantic-type-checking.md](./04-tier3-semantic-type-checking.md) for Tier 3 details
- See [05-data-flow.md](./05-data-flow.md) for complete data flow
- See [06-integration-possibilities.md](./06-integration-possibilities.md) for use cases

### Extended Layer Documentation

- See [Dialogue Context Layer](../dialogue/README.md) for conversation tracking
- See [LLM Integration Layer](../llm-integration/README.md) for agent support
- See [Agent Learning Layer](../agent-learning/README.md) for adaptive correction

### Integration Documentation

- See [MORK Integration](../../integration/mork/README.md) for Phases A-D
- See [PathMap Integration](../../integration/pathmap/README.md) for shared storage
- See [bibliography.md](../reference/bibliography.md) for complete references
