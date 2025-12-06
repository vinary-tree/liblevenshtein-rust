# WFST-Based Text Normalization Architecture for liblevenshtein-rust

**Status**: Design Document
**Last Updated**: 2025-11-20
**Author**: Research synthesis from arXiv, ACL Anthology, NVIDIA NeMo, Google Sparrowhawk

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architectural Overview](#architectural-overview)
3. [Three-Tier Hybrid Approach](#three-tier-hybrid-approach)
4. [Six-Layer Pipeline Design](#six-layer-pipeline-design)
5. [Composition Operators](#composition-operators)
6. [Weight Schemes and Scoring](#weight-schemes-and-scoring)
7. [Lattice Representation](#lattice-representation)
8. [Integration with liblevenshtein-rust](#integration-with-liblevenshtein-rust)
9. [Integration with MORK](#integration-with-mork-metta-optimal-reduction-kernel)
10. [Integration with Large Language Models](#integration-with-large-language-models)
11. [Comparison with Industry Systems](#comparison-with-industry-systems)
12. [Performance Characteristics](#performance-characteristics)
13. [Deployment Modes](#deployment-modes)
14. [References](#references)

---

## Executive Summary

### The Problem

Text normalization for noisy user-generated content (SMS, chat, social media) requires correcting multiple error types:

1. **Spelling errors**: Typos, character-level mistakes ("teh" → "the")
2. **Phonetic errors**: Sound-based misspellings ("fone" → "phone")
3. **Grammatical errors**: Syntax mistakes requiring context-free parsing
4. **Semantic ambiguities**: Word sense disambiguation ("bank" = river vs financial)

### Industry Consensus (2020-2025)

**Key Finding**: Neither pure FST nor pure neural approaches are optimal.

- **Pure FST**: Fast, deterministic, but limited to regular languages (cannot handle nested syntax)
- **Pure Neural**: Context-aware, but prone to "unrecoverable errors" and hallucination
- **Hybrid FST + Neural**: Industry standard (NVIDIA NeMo, Google Sparrowhawk)

**Quote from NVIDIA NeMo (arXiv:2104.05055)**:
> "Low tolerance towards unrecoverable errors is the main reason why most ITN systems in production are still largely rule-based using WFSTs"

### liblevenshtein-rust's Unique Advantage

**Three-Tier Hybrid Architecture** (FST + CFG + Neural):

```
┌─────────────────────────────────────────────────────────┐
│ Tier 1: Regular (FST/NFA)                               │
│ - Spelling correction (Levenshtein automata)            │
│ - Phonetic normalization (NFA regex)                    │
│ - Morphological variants                                │
│ - Complexity: O(n)                                      │
│ - Deterministic: Yes                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Tier 2: Context-Free (CFG) ← UNIQUE TO liblevenshtein  │
│ - Grammar correction (subject-verb agreement)           │
│ - Phrase structure (nested dependencies)                │
│ - Article selection (a/an based on phonology)           │
│ - Complexity: O(n³) CYK, O(n²) average Earley          │
│ - Deterministic: Yes (with PCFG for disambiguation)     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Tier 3: Neural (Optional)                               │
│ - Semantic disambiguation                               │
│ - Discourse-level coherence                             │
│ - Style transfer                                        │
│ - Complexity: O(n²) transformer                         │
│ - Deterministic: No (probabilistic)                     │
└─────────────────────────────────────────────────────────┘
```

**Key Differentiator**: Tiers 1-2 are **purely symbolic** (deterministic, verifiable, no training data needed). Only Tier 3 requires neural networks.

**Industry Comparison**:
- **NVIDIA NeMo**: FST + Neural (no CFG layer)
- **Google Sparrowhawk**: FST only (text normalization, no grammar)
- **liblevenshtein-rust**: FST + CFG + Neural (full Chomsky hierarchy coverage)

### Why This Matters

**Chomsky Hierarchy Coverage**:

| Language Class | Formalism | Can Handle | Cannot Handle |
|----------------|-----------|------------|---------------|
| **Type 3: Regular** | FST/NFA | Spelling, phonetic, morphology | Nested syntax (a^n b^n) |
| **Type 2: Context-Free** | CFG | Syntax, phrase structure, agreement | Semantic dependencies |
| **Type 1: Context-Sensitive** | Neural LM | Semantics, discourse | (requires large training data) |

**liblevenshtein-rust covers all three levels**, allowing deterministic symbolic correction where possible, and falling back to neural only when necessary.

---

## Architectural Overview

### Design Principles

1. **Symbolic-first**: Use deterministic algorithms (FST, CFG) before neural models
2. **Compositional**: Layers compose via well-defined interfaces (lattices, parse forests)
3. **Modular**: Each layer independently testable and swappable
4. **Performance-aware**: O(n) for FST, O(n³) for CFG, optional neural for accuracy
5. **Formally verified**: Phonetic rules proven in Coq, grammar rules symbolic

### Data Flow

```
Input Text: "i seen a elephant yesterday"
            ↓
[Detection] Identify potential errors
            ↓ ["seen", "a elephant"]
[NFA Phonetic] Expand phonetic variants
            ↓ Lattice: {seen, scene, sean} × {a, an}
[Levenshtein] Add edit distance corrections
            ↓ Extended lattice (phonetic + spelling)
[CFG Grammar] Apply error grammar rules
            ↓ Rule: DT[a] NP[+vowel_initial] → DT[an] NP
            ↓ Grammatically valid lattice
[Neural LM] (optional) Rank candidates by fluency
            ↓ "seen" vs "saw" (requires context understanding)
[Selection] Choose best candidate
            ↓
Output: "i saw an elephant yesterday"
```

---

## Three-Tier Hybrid Approach

### Tier 1: Regular Languages (FST/NFA)

**Capabilities**:
- Character-level edit operations (insertion, deletion, substitution, transposition)
- Phonetic transformations (ph→f, c→s, gh→∅)
- Morphological variants (plurals, verb conjugations if context-free)
- Lexical normalization ("u" → "you", "4" → "for")

**Algorithms**:
- **Levenshtein automaton**: Accepts all strings within edit distance n
- **NFA phonetic regex**: Compiles patterns like `(ph|f)` to non-deterministic automaton
- **FST composition**: Chains transducers (T1 ∘ T2 ∘ ... ∘ Tn)

**Complexity**: O(n) with pre-compiled automata

**Example**:
```
Input: "fone"
NFA regex: (ph|f)(o|oa)(n|ne)
Levenshtein: edit_distance("fone", dict) ≤ 2
Intersection: {"phone", "fone", "phones"} ∩ dictionary
Output lattice: [("phone", cost=1.0), ("phones", cost=1.2)]
```

**Limitations**:
- ❌ Cannot handle nested structures (balanced parentheses)
- ❌ Cannot count (subject-verb agreement if distance > trigram)
- ❌ No semantic understanding

### Tier 2: Context-Free Languages (CFG)

**Capabilities**:
- Subject-verb agreement ("they was" → "they were")
- Article selection ("a apple" → "an apple")
- Nested phrase structures (PP, NP, VP)
- Tense consistency within clauses
- Auxiliary verb selection ("can able" → "can" or "is able")

**Algorithms**:
- **CYK parsing**: O(n³), requires Chomsky Normal Form
- **Earley parsing**: O(n³) worst case, O(n²) average, handles arbitrary CFG
- **Probabilistic CFG**: Assigns probabilities to ambiguous parses

**Error Grammar Approach**:
```coq
(* Well-formed rule *)
S → NP[num=sg] VP[num=sg]
S → NP[num=pl] VP[num=pl]

(* Error production with correction *)
S → NP[num=sg] VP[num=pl]
    { ERROR: "Subject-verb agreement mismatch"
      FIX: Rewrite VP to singular form }

S → NP[num=pl] VP[num=sg]
    { ERROR: "Subject-verb agreement mismatch"
      FIX: Rewrite VP to plural form }
```

**Complexity**: O(n³) for CYK, O(n²) average for Earley

**Example**:
```
Input: "the cat run fast"

CFG Parse:
  S
  ├── NP[num=sg]
  │   ├── DT: "the"
  │   └── N[num=sg]: "cat"
  └── VP[num=pl] ← ERROR (expects sg)
      ├── V[num=pl]: "run"
      └── ADV: "fast"

Error detected: NP[sg] + VP[pl] mismatch
Correction: Change "run" → "runs"
Output: "the cat runs fast"
```

**Limitations**:
- ❌ Still cannot handle a^n b^n c^n (context-sensitive)
- ❌ Long-range dependencies beyond CFG scope
- ❌ Semantic ambiguity ("saw" = past tense vs cutting tool)

### Tier 3: Neural Models (Optional)

**Capabilities**:
- Semantic disambiguation
- Discourse-level coherence (anaphora resolution)
- Pragmatic inference (implied meaning)
- Style transfer (formal ↔ informal)
- Complex long-range dependencies

**Algorithms**:
- **BERT masked language model**: Predict missing/incorrect words
- **Transformer sequence-to-sequence**: Full rewrite with context
- **Lattice rescoring**: Re-rank candidates from Tier 1+2 using neural scores

**Complexity**: O(n²) for transformer self-attention

**Example**:
```
Input lattice from CFG layer:
  [("I saw the movie", score=0.8),
   ("I seen the movie", score=0.6)]

BERT scoring:
  P("saw" | context) = 0.95
  P("seen" | context) = 0.05

Final ranking:
  "I saw the movie" (combined_score = 0.8 + 0.95 = 1.75) ← WINNER
  "I seen the movie" (combined_score = 0.6 + 0.05 = 0.65)
```

**Limitations**:
- ❌ Non-deterministic (same input may yield different outputs)
- ❌ Hallucination risk ("unrecoverable errors")
- ❌ Requires large training datasets
- ❌ Computationally expensive (latency 100-500ms)

---

## Six-Layer Pipeline Design

### Layer 1: Tokenization & Detection

**Purpose**: Identify regions requiring normalization

**Approach**:
1. Tokenize into words/subwords
2. Detect non-standard tokens:
   - Not in dictionary
   - Character-level anomalies (repeated letters "hellooo")
   - Phonetic patterns matching known errors
   - Grammar violations (POS tagging + shallow parsing)

**Output**: Annotated token stream with error spans

**Example**:
```
Input: "i seen a elephant yesterday"
Tokens: ["i", "seen", "a", "elephant", "yesterday"]
Detection:
  - "i" → capitalize (POS: pronoun, sentence-initial)
  - "seen" → potential tense error (past participle without auxiliary)
  - "a elephant" → article error (a + vowel-initial)
  - "yesterday" → OK (in dictionary)
Error spans: [(0, 1), (1, 2), (2, 4)]
```

### Layer 2: Phonetic Normalization (NFA-based)

**Purpose**: Expand phonetic variants using verified orthography rules

**Approach**:
1. Compile phonetic rules to NFA:
   ```
   ph → f
   c[aou] → k
   gh → ∅ / _#  (word-final)
   ough → uf
   ```
2. Apply NFA to each error span
3. Generate lattice of phonetic alternatives

**Formal Verification**: Rules proven in Coq (see `docs/verification/phonetic/`)

**Output**: Lattice with phonetic expansions

**Example**:
```
Input: "fone"
NFA regex: (ph|f)(o|oa)(n|ne)
Expanded: {"fone", "phone", "foane", "phne", ...}
Dictionary filter: {"fone", "phone"} ∩ dictionary = {"phone"}
Lattice: [("phone", phonetic_cost=0.5)]
```

### Layer 3: Levenshtein-based Spelling Correction (FST)

**Purpose**: Add edit distance corrections to phonetic candidates

**Approach**:
1. Build Levenshtein automaton for each candidate (distance ≤ n)
2. Intersect with dictionary trie
3. Merge with phonetic lattice

**Composition**: NFA(phonetic) ∩ FST(Levenshtein) ∩ Trie(dictionary)

**Output**: Extended lattice (phonetic + spelling corrections)

**Example**:
```
Input lattice: [("phone", cost=0.5)]
Levenshtein(distance ≤ 2):
  - "phone" (exact match, edit_cost=0)
  - "phones" (insertion, edit_cost=1)
  - "phoned" (insertion, edit_cost=1)
  - "hone" (deletion, edit_cost=1)

Combined lattice:
  [("phone", total_cost=0.5), ("phones", total_cost=1.5), ...]
```

### Layer 4: CFG-based Grammar Correction

**Purpose**: Apply syntactic error corrections requiring parsing

**Approach**:
1. Parse each lattice path with error grammar
2. Detect syntactic violations
3. Apply correction rules
4. Prune grammatically invalid paths

**Error Grammar**:
```coq
(* Article selection *)
DT[a] NP[+vowel_initial] → DT[an] NP  { cost = 0.1 }
DT[an] NP[-vowel_initial] → DT[a] NP  { cost = 0.1 }

(* Subject-verb agreement *)
NP[num=sg] VP[num=pl] → NP[num=sg] VP[num=sg]  { cost = 0.5 }
NP[num=pl] VP[num=sg] → NP[num=pl] VP[num=pl]  { cost = 0.5 }

(* Tense auxiliary *)
AUX[past] VP[present] → AUX[past] VP[past]  { cost = 0.3 }
```

**Parsing Algorithm**: Earley parser (handles arbitrary CFG, left-recursion)

**Output**: Grammatically corrected lattice

**Example**:
```
Input: "a elephant"
Parse:
  NP
  ├── DT: "a"
  └── N: "elephant" (+vowel_initial)

Grammar rule matches:
  DT[a] NP[+vowel_initial] → DT[an] NP

Correction applied:
  "a elephant" → "an elephant" (cost = 0.1)

Lattice: [("an elephant", cost=0.1)]
```

### Layer 5: Neural Language Model Disambiguation

**Purpose**: Rank candidates by contextual fluency (optional)

**Approach**:
1. For each lattice path, compute neural LM score
2. Combine with symbolic scores: `total = α·symbolic + β·neural`
3. Select top-k candidates

**Models**:
- **BERT masked LM**: P(word | context)
- **GPT-2/3 autoregressive**: P(sentence)
- **Fine-tuned on SMS/chat data**: Domain adaptation

**Output**: Re-ranked lattice with neural scores

**Example**:
```
Input lattice:
  [("I seen the movie", symbolic=0.6),
   ("I saw the movie", symbolic=0.8)]

BERT scores:
  P("seen" | "I ___ the movie") = 0.05
  P("saw" | "I ___ the movie") = 0.95

Combined (α=0.5, β=0.5):
  "I seen the movie" → 0.5·0.6 + 0.5·0.05 = 0.325
  "I saw the movie" → 0.5·0.8 + 0.5·0.95 = 0.875 ← WINNER
```

### Layer 6: Post-processing and Validation

**Purpose**: Final cleanup and safety checks

**Approach**:
1. Capitalization (sentence-initial, proper nouns)
2. Punctuation normalization
3. Whitespace cleanup
4. Safety checks:
   - Length ratio (output/input < threshold)
   - Character set validation
   - Profanity filter (if needed)

**Output**: Final corrected text

**Example**:
```
Input: "i saw an elephant yesterday"
Post-processing:
  - Capitalize "i" → "I" (sentence-initial pronoun)
  - Validate length: 28 chars → 29 chars (OK)

Output: "I saw an elephant yesterday"
```

---

## Composition Operators

### FST ∘ FST (Standard Transducer Composition)

**Definition**: Compose two finite-state transducers T1 and T2:
- T1: Input alphabet Σ → Intermediate alphabet Γ
- T2: Intermediate alphabet Γ → Output alphabet Δ
- T1 ∘ T2: Σ → Δ (direct composition)

**Application**: Chain spelling corrections
```
Input: "fone"
T1 (phonetic): "fone" → "phone"
T2 (capitalization): "phone" → "Phone"
T1 ∘ T2: "fone" → "Phone"
```

**Complexity**: O(|Q1| · |Q2|) states in worst case (product construction)

### NFA ∩ FST (Intersection)

**Definition**: Intersection of NFA language and FST input language:
- L(NFA ∩ FST) = {x : x ∈ L(NFA) ∧ x ∈ Domain(FST)}

**Application**: Phonetic patterns constrained by edit distance
```
NFA (phonetic regex): (ph|f)(o|oa)(n|ne)
FST (Levenshtein): edit_distance ≤ 2 from dictionary
Intersection: Phonetically plausible AND close to dictionary entry
```

**Implementation**: Product automaton construction
```rust
struct IntersectionState {
    nfa_state: StateId,
    fst_state: StateId,
    edit_count: usize,
}
```

### CFG × FST (Parse Tree Pruning)

**Definition**: Parse input with CFG, constrain by FST boundaries:
- CFG parses sentence structure
- FST limits which words can be substituted (edit distance constraint)

**Application**: Grammar correction on spelling-corrected lattice
```
FST lattice: {("seen", 0.6), ("saw", 0.8), ("scene", 0.3)}
CFG rule: S → NP VP[past]
Constraint: Only select from FST lattice
Parse: "I [seen/saw/scene] the movie"
Grammar: "saw" is past tense (matches VP[past])
Result: "I saw the movie"
```

**Complexity**: O(n³) CYK parsing × |lattice paths|

### Lattice → CFG Parser

**Definition**: Parse word lattice (directed acyclic graph) instead of string:
- Nodes: Word positions (0, 1, 2, ..., n)
- Edges: Word hypotheses with scores
- CFG spans edges, producing parse forest

**Application**: Handle ambiguous normalization candidates
```
Lattice:
  0 --("I", 1.0)--> 1 --("seen", 0.6)--> 2 --("the", 1.0)--> 3
                   |                    |
                   +--("saw", 0.8)-----+

CFG parses both paths:
  Path 1: "I seen the" → ERROR (seen requires auxiliary)
  Path 2: "I saw the" → OK

Select Path 2
```

**Algorithm**: Chart parsing extended to lattices (Earley or CYK)

---

## Weight Schemes and Scoring

### Tropical Semiring

**Definition**: (ℝ ∪ {∞}, ⊕ = min, ⊗ = +, 0̄ = ∞, 1̄ = 0)

**Operations**:
- **Addition**: a ⊕ b = min(a, b) (select best path)
- **Multiplication**: a ⊗ b = a + b (accumulate costs)
- **Identity**: 1̄ = 0 (no cost)
- **Annihilator**: 0̄ = ∞ (impossible path)

**Application**: Shortest path in weighted automaton
```
Path 1: cost = 0.5 + 0.3 + 0.2 = 1.0
Path 2: cost = 0.8 + 0.1 + 0.1 = 1.0
Path 3: cost = 0.4 + 0.4 + 0.4 = 1.2
Best path: min(1.0, 1.0, 1.2) = 1.0 (Paths 1 or 2)
```

### Cost Functions

**Edit Distance**: Character-level operation costs
```
Insertion: 1.0
Deletion: 1.0
Substitution: 1.0 (or 2.0 if conservative)
Transposition: 1.0 (Damerau-Levenshtein)
Match: 0.0
```

**Phonetic Similarity**: Sound-based costs
```
ph ↔ f: 0.1 (very similar)
c ↔ k: 0.2 (similar before a/o/u)
c ↔ s: 0.2 (similar before e/i)
gh → ∅: 0.3 (silent, common deletion)
Arbitrary substitution: 1.0
```

**Grammar Error**: Syntactic violation costs
```
Article error (a/an): 0.5
Subject-verb agreement: 1.0 (more severe)
Tense inconsistency: 0.8
Missing auxiliary: 1.2
```

**Language Model**: Contextual fluency
```
Log-probability: log P(word | context)
Normalized: -log P(...) (convert to cost, lower is better)
```

### Combined Scoring

**Linear interpolation**:
```
total_cost = α·edit + β·phonetic + γ·grammar + δ·LM

Typical weights:
α = 0.3  (edit distance)
β = 0.2  (phonetic similarity)
γ = 0.3  (grammar cost)
δ = 0.2  (language model)
```

**Example**:
```
Candidate: "I saw an elephant yesterday"
  edit = 2.0 (2 corrections: "seen"→"saw", "a"→"an")
  phonetic = 0.0 (no phonetic errors)
  grammar = 0.5 (article error corrected)
  LM = 1.2 (negative log probability)

total = 0.3·2.0 + 0.2·0.0 + 0.3·0.5 + 0.2·1.2
      = 0.6 + 0.0 + 0.15 + 0.24
      = 0.99
```

---

## Lattice Representation

### Data Structure

**Weighted Directed Acyclic Graph (DAG)**:

```rust
struct Lattice {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    start: NodeId,
    final_nodes: Vec<NodeId>,
}

struct Node {
    id: NodeId,
    position: usize,  // Position in original input (0..n)
}

struct Edge {
    source: NodeId,
    target: NodeId,
    label: String,    // Word or token
    weight: f64,      // Cost in tropical semiring
    features: Features,  // Phonetic, grammar, LM scores
}

struct Features {
    edit_distance: f64,
    phonetic_cost: f64,
    grammar_cost: f64,
    lm_score: f64,
}
```

### Example Lattice

**Input**: "i seen a elephant"

**After Layer 3** (Phonetic + Levenshtein):

```
   ┌─("I", 0.1)──────────────────────────────────┐
   │                                              │
   0                                              1
                                                  │
   ┌─("seen", 0.6)─┐                             │
   │               │                             │
   1               2                              │
   │               │                             │
   └─("saw", 0.8)──┘                             │
                                                  │
   ┌─("a", 0.0)────┐                             │
   │               │                             │
   2               3                              │
   │               │                             │
   └─("an", 0.5)───┘                             │
                                                  │
   ┌─("elephant", 0.0)────────────────────────────┤
   │                                              │
   3                                              4

Paths:
  1. "I seen a elephant" (cost = 0.1 + 0.6 + 0.0 + 0.0 = 0.7)
  2. "I seen an elephant" (cost = 0.1 + 0.6 + 0.5 + 0.0 = 1.2)
  3. "I saw a elephant" (cost = 0.1 + 0.8 + 0.0 + 0.0 = 0.9)
  4. "I saw an elephant" (cost = 0.1 + 0.8 + 0.5 + 0.0 = 1.4)
```

**After Layer 4** (CFG Grammar):

CFG applies article rule: `a + vowel_initial → an`

Prune paths violating grammar:
- Path 1: INVALID ("a elephant")
- Path 2: VALID
- Path 3: INVALID ("a elephant")
- Path 4: VALID

Grammar-corrected lattice:
```
Paths:
  2. "I seen an elephant" (cost = 1.2)
  4. "I saw an elephant" (cost = 1.4)
```

**After Layer 5** (Neural LM):

BERT scoring:
- "seen": P = 0.05 → cost = -log(0.05) = 3.0
- "saw": P = 0.95 → cost = -log(0.95) = 0.05

Combined (α=0.7 symbolic, β=0.3 neural):
- Path 2: 0.7·1.2 + 0.3·3.0 = 0.84 + 0.90 = 1.74
- Path 4: 0.7·1.4 + 0.3·0.05 = 0.98 + 0.015 = 0.995 ← WINNER

**Final Output**: "I saw an elephant"

### Lattice Parsing: Efficient CFG Integration

**Key Challenge**: The lattice from Tier 1 (FST/NFA) may contain hundreds of candidate paths. Parsing each path individually with a CFG (Tier 2) would cause exponential blowup.

**Solution**: **Lattice parsing** - parse the compact DAG representation directly, sharing computation across all paths.

#### Why Enumerate Paths is Inefficient

For the example lattice above with 4 paths, individual parsing requires:

```
Parse("I seen a elephant")  → 4 chart operations
Parse("I seen an elephant") → 4 chart operations
Parse("I saw a elephant")   → 4 chart operations
Parse("I saw an elephant")  → 4 chart operations
Total: 16 operations (4 paths × 4 words)
```

**Problem**: For K corrections per word over N words, this scales as **O(K^N × N³)** (exponential in N).

#### Lattice Parsing Algorithm

Instead of enumerating paths, parse the lattice directly:

```rust
// Modified Earley parser: index chart by (node_id, position) not just position
fn parse_lattice(grammar: &Grammar, lattice: &Lattice) -> ParseForest {
    let mut chart = EarleyChart::new();

    // Initialize with start state at start node
    chart.add_state(EarleyState {
        rule: grammar.start_rule(),
        dot_position: 0,
        start_node: lattice.start,
        current_node: lattice.start,
    });

    // Process nodes in topological order
    for node in lattice.topological_order() {
        for state in chart.states_at(node) {
            if state.is_complete() {
                // COMPLETER: backpropagate completed non-terminal
                chart.complete(state);
            } else {
                let next_symbol = state.next_symbol();

                if grammar.is_non_terminal(next_symbol) {
                    // PREDICTOR: add states for non-terminal expansion
                    chart.predict(state, next_symbol);
                } else {
                    // SCANNER: follow lattice edges with matching terminal
                    for edge in lattice.outgoing_edges(node) {
                        if lattice.edge_label(edge) == next_symbol {
                            chart.scan(state, edge);
                        }
                    }
                }
            }
        }
    }

    chart.extract_parse_forest()
}
```

**Key Differences from String Parsing**:

| String Parsing | Lattice Parsing |
|----------------|-----------------|
| Chart indexed by `position` | Chart indexed by `(node, position)` |
| Scanner checks `input[position]` | Scanner follows lattice edges |
| Advance by `position + 1` | Advance to `edge.target` node |
| O(N³) for single string | O(K×N × N²) for K branches |

#### Example: Shared Prefix Parsing

For our 4-path lattice:

```
Lattice parsing:
  Node 0→1: Parse "I" (1×)
  Node 1→2: Parse "seen" (1×), "saw" (1×)
  Node 2→3: Parse "a" (1×), "an" (1×)
  Node 3→4: Parse "elephant" (1×)

Total: 6 word parses (each word parsed once per unique occurrence)
```

**Speedup**: 16 operations → 6 operations = **2.67× faster** (for just 2 words!)

For realistic inputs:
- 5 words × 10 corrections: 100K paths → **~10,000× speedup**
- 10 words × 5 corrections: 9.7M paths → **>1M× speedup**

#### Parse Forest Output

Lattice parsing produces a **parse forest** - a compact DAG representing all grammatically valid parse trees:

```
Parse Forest for "I [seen|saw] an elephant":

                     S
                   /   \
                  /     \
                NP       VP
                |       / | \
                |      /  |  \
              "I"    V   NP
                     |   |  \
                [seen|saw] Det N
                          |   |
                        "an" "elephant"
```

The forest shares the common NP subtree `Det("an") + N("elephant")` across both parse trees.

**Extract best parse**:
```rust
let forest = parse_lattice(&grammar, &lattice);
let best_parse = forest.best_parse();  // Viterbi algorithm
println!("Best: {}", best_parse.sentence());
// Output: "I saw an elephant" (highest PCFG probability)
```

#### Complexity Analysis

| Representation | Parse Time | Memory |
|----------------|------------|--------|
| String list (K^N paths) | O(K^N × N³) | O(K^N × N) |
| Lattice (K×N edges) | O(K×N × N²) | O(K×N) |
| **Speedup** | **O(K^(N-1) × N)** | **O(K^(N-1))** |

**Practical measurements** (see [lattice_parsing.md](./lattice_parsing.md)):
- **3-10× speedup** on real-world queries
- **4× memory reduction** from shared structure
- Scales to 20+ word sentences (string enumeration fails at ~10 words)

#### Integration with Three-Tier Pipeline

```rust
// Tier 1: FST → Lattice
let transducer = Transducer::for_dictionary(dictionary)
    .algorithm(Algorithm::Transposition)
    .max_distance(2)
    .build();

let lattice = transducer
    .query("teh cat dont lik me")
    .to_lattice();  // Returns compact DAG, not path enumeration

// Tier 2: Lattice → Parse Forest (via lattice parsing)
let parser = EarleyParser::new(&grammar);
let forest = parser.parse_lattice(&lattice)?;

// Extract top-K grammatically valid candidates
let candidates = forest.k_best_parses(10);

// Tier 3 (optional): Neural reranking
let best = neural_reranker.rerank(&candidates);
```

**Key Advantage**: Tier 2 CFG parsing operates on the **compact lattice**, not the exponential path explosion, making grammatical error correction tractable for real-world queries.

**Further Reading**:
- [lattice_parsing.md](./lattice_parsing.md) - Complete pedagogical guide with worked examples
- [lattice_data_structures.md](./lattice_data_structures.md) - Technical reference for data structures
- [cfg_grammar_correction.md](./cfg_grammar_correction.md) - CFG formalism and grammar rules

---

## Integration with liblevenshtein-rust

### Current Capabilities

**Phonetic Rules** (`src/phonetic/`):
- 13 orthography rules (Zompist phonetic spelling)
- Formally verified in Coq (5 theorems proven)
- Rust implementation matches verified semantics
- 147 tests in test suite

**Levenshtein Automata** (`src/transducer/`):
- Standard edit distance (insertion, deletion, substitution)
- Damerau-Levenshtein (+ transposition)
- Dictionary backends: DoubleArrayTrie, PathMap

**Example Integration** (`examples/phonetic_fuzzy_matching.rs`):
- Demonstrates phonetic + Levenshtein combination
- 6 error correction scenarios

### Proposed Enhancements

#### 1. Weighted Levenshtein Automaton

**Current**: Uniform edit costs (all operations cost 1.0)

**Proposed**: Phonetic-aware transition weights

```rust
pub struct WeightedLevenshteinConfig {
    pub insertion_cost: f64,
    pub deletion_cost: f64,
    pub substitution_cost: Box<dyn Fn(char, char) -> f64>,
    pub transposition_cost: f64,
}

impl WeightedLevenshteinConfig {
    pub fn phonetic_aware() -> Self {
        Self {
            insertion_cost: 1.0,
            deletion_cost: 1.0,
            substitution_cost: Box::new(|a, b| {
                match (a, b) {
                    ('f', 'p') | ('p', 'f') if next_is('h') => 0.1,  // ph ↔ f
                    ('c', 'k') | ('k', 'c') => 0.2,  // c ↔ k
                    ('c', 's') | ('s', 'c') => 0.2,  // c ↔ s
                    _ => 1.0,
                }
            }),
            transposition_cost: 1.0,
        }
    }
}
```

#### 2. NFA Phonetic Regex Compiler

**Syntax**: Regular expressions for phonetic patterns

```rust
pub struct PhoneticRegex {
    pattern: String,
    nfa: NFA,
}

impl PhoneticRegex {
    pub fn compile(pattern: &str) -> Result<Self, ParseError> {
        // Parse: "(ph|f)(o|oa)(n|ne)"
        // Build NFA using Thompson's construction
        let nfa = thompson_construction(pattern)?;
        Ok(Self { pattern: pattern.to_string(), nfa })
    }

    pub fn intersect_with_levenshtein(
        &self,
        lev: &LevenshteinAutomaton
    ) -> ComposedAutomaton {
        // Product construction: NFA × Levenshtein
        product_automaton(&self.nfa, lev.as_nfa())
    }
}
```

**Example Usage**:
```rust
let phonetic = PhoneticRegex::compile("(ph|f)(o|oa)(n|ne)")?;
let lev = LevenshteinAutomaton::new("phone", 2)?;
let composed = phonetic.intersect_with_levenshtein(&lev);

let matches = composed.search(&dictionary);
// Returns: {"phone", "fone", "phones", ...}
```

#### 3. Lattice Output Format

**Current**: Iterator<Item = String> (single best matches)

**Proposed**: Iterator<Item = ScoredCandidate> (n-best with scores)

```rust
pub struct ScoredCandidate {
    pub text: String,
    pub edit_distance: usize,
    pub phonetic_cost: f64,
    pub total_cost: f64,
    pub path: Vec<Edge>,  // For debugging
}

pub struct Lattice {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub start: NodeId,
    pub finals: Vec<NodeId>,
}

impl Transducer {
    pub fn query_lattice(
        &self,
        term: &str,
        max_distance: usize,
    ) -> Lattice {
        // Returns full lattice instead of just strings
    }

    pub fn query_nbest(
        &self,
        term: &str,
        max_distance: usize,
        n: usize,
    ) -> Vec<ScoredCandidate> {
        // Returns top-n candidates with scores
    }
}
```

#### 4. Neural LM Integration API

**External Language Model Hook**:

```rust
pub trait LanguageModel {
    fn score(&self, tokens: &[&str]) -> f64;
    fn score_word_in_context(&self, word: &str, context: &[&str]) -> f64;
}

impl Transducer {
    pub fn query_with_lm<LM: LanguageModel>(
        &self,
        term: &str,
        max_distance: usize,
        lm: &LM,
        alpha: f64,  // Weight for edit distance
        beta: f64,   // Weight for LM score
    ) -> Vec<ScoredCandidate> {
        let lattice = self.query_lattice(term, max_distance);

        lattice.paths()
            .map(|path| {
                let edit_cost = path.total_edit_distance() as f64;
                let lm_cost = lm.score(&path.tokens());
                let total = alpha * edit_cost + beta * lm_cost;

                ScoredCandidate {
                    text: path.to_string(),
                    edit_distance: path.total_edit_distance(),
                    phonetic_cost: 0.0,  // TODO: Add phonetic scoring
                    total_cost: total,
                    path: path.edges,
                }
            })
            .sorted_by_key(|c| OrderedFloat(c.total_cost))
            .take(n)
            .collect()
    }
}
```

#### 5. CFG Grammar Correction (NEW)

**Error Grammar Definition**:

```rust
pub struct ErrorGrammar {
    productions: Vec<Production>,
    error_productions: Vec<ErrorProduction>,
}

pub struct Production {
    lhs: NonTerminal,
    rhs: Vec<Symbol>,
    weight: f64,
}

pub struct ErrorProduction {
    lhs: NonTerminal,
    rhs: Vec<Symbol>,
    error_type: ErrorType,
    correction: Correction,
    weight: f64,
}

pub enum ErrorType {
    ArticleError,
    SubjectVerbAgreement,
    TenseInconsistency,
    AuxiliaryError,
}

pub enum Correction {
    Replace(Symbol, Symbol),
    Insert(Symbol, usize),
    Delete(usize),
}
```

**Example Grammar**:

```rust
let mut grammar = ErrorGrammar::new();

// Well-formed rule
grammar.add_production(
    Production {
        lhs: NonTerminal::S,
        rhs: vec![Symbol::NP, Symbol::VP],
        weight: 0.0,
    }
);

// Error rule: Article selection
grammar.add_error_production(
    ErrorProduction {
        lhs: NonTerminal::NP,
        rhs: vec![
            Symbol::Terminal("a"),
            Symbol::NP { features: Features { vowel_initial: true } }
        ],
        error_type: ErrorType::ArticleError,
        correction: Correction::Replace(
            Symbol::Terminal("a"),
            Symbol::Terminal("an")
        ),
        weight: 0.5,
    }
);

// Error rule: Subject-verb agreement
grammar.add_error_production(
    ErrorProduction {
        lhs: NonTerminal::S,
        rhs: vec![
            Symbol::NP { number: Number::Singular },
            Symbol::VP { number: Number::Plural }
        ],
        error_type: ErrorType::SubjectVerbAgreement,
        correction: Correction::Replace(
            Symbol::VP { number: Number::Plural },
            Symbol::VP { number: Number::Singular }
        ),
        weight: 1.0,
    }
);
```

**Parser**:

```rust
pub struct EarleyParser {
    grammar: ErrorGrammar,
}

impl EarleyParser {
    pub fn parse(&self, tokens: &[&str]) -> ParseForest {
        // Earley algorithm with error productions
        // Returns all possible parses (including error corrections)
    }

    pub fn parse_lattice(&self, lattice: &Lattice) -> ParseForest {
        // Parse word lattice instead of single string
        // Combines CFG parsing with FST uncertainty
    }

    pub fn correct(&self, tokens: &[&str]) -> Vec<Correction> {
        let forest = self.parse(tokens);
        forest.extract_errors()
    }
}
```

**Integration Example**:

```rust
// Build pipeline
let phonetic_rules = orthography_rules();
let dict = DoubleArrayTrie::from_file("dictionary.txt")?;
let lev_transducer = Transducer::new(dict, Algorithm::Transposition);
let grammar = load_error_grammar("grammar.cfg")?;
let parser = EarleyParser::new(grammar);

// Process input
let input = "i seen a elephant yesterday";

// Layer 1-3: Phonetic + Levenshtein
let lattice = lev_transducer.query_lattice(input, 2);

// Layer 4: CFG Grammar
let parse_forest = parser.parse_lattice(&lattice);
let corrected_lattice = parse_forest.apply_corrections();

// Layer 5: Select best path
let best = corrected_lattice.shortest_path();
println!("{}", best);  // "I saw an elephant yesterday"
```

#### 6. Export to OpenFST FAR

**Interoperability with Production Systems**:

```rust
impl Transducer {
    pub fn export_openfst_far(&self, path: &Path) -> Result<(), Error> {
        // Export to OpenFST Archive File format
        // Compatible with Sparrowhawk, Thrax, Pynini
    }
}

impl PhoneticRegex {
    pub fn export_openfst(&self, path: &Path) -> Result<(), Error> {
        // Export NFA to OpenFST format
    }
}
```

### Implementation Roadmap

**Phase 1: Lattice Output (Foundation)**
- Modify transducer to return lattices instead of strings
- Implement n-best candidate extraction
- Add scoring framework (edit distance, phonetic, total)

**Phase 2: NFA Phonetic Regex**
- Implement Thompson's construction for regex → NFA
- Add intersection operator (NFA × FST)
- Integrate with existing phonetic rules

**Phase 3: Weighted Transitions**
- Add configurable cost functions
- Implement phonetic-aware substitution costs
- Benchmark performance impact

**Phase 4: CFG Grammar Correction**
- Implement Earley parser
- Define error grammar formalism
- Build example grammar for common errors

**Phase 5: Neural LM Integration**
- Define LanguageModel trait
- Add BERT integration (via ONNX or Python binding)
- Implement lattice rescoring

**Phase 6: Production Deployment**
- Export to OpenFST FAR format
- Optimize for latency (caching, lazy evaluation)
- Add deployment modes (Fast/Balanced/Accurate)

---

## Integration with MORK (MeTTa Optimal Reduction Kernel)

### Overview

MORK provides MeTTa pattern matching and query execution over PathMap-backed knowledge graphs. The WFST architecture integrates with MORK to enable **fuzzy pattern matching** in MeTTa queries.

### Architecture Alignment

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

### Key Integration Points

| liblevenshtein Component | MORK Integration | Purpose |
|--------------------------|------------------|---------|
| Transducer | FuzzySource | Fuzzy symbol matching in queries |
| Lattice | LatticeZipper | Ranked multi-candidate results |
| PhoneticNfa | WFST composition | Sound-alike pattern matching |
| PathMapDictionary | Shared storage | Single dictionary for both systems |

### MORK Pattern Matching Synergies

MORK's pattern matching capabilities directly support the WFST pipeline:

#### NFA Representation in MORK

```metta
; NFA state encoding as S-expressions
(state q0 [(trans a q0) (trans b q1)])
(state q1 [(trans b q1) (trans ε acc)])
(accepting acc)

; Pattern to find epsilon closure
Pattern: (state ?Q [(trans ε ?R) . ?rest])
Result: Bindings {?Q → q1, ?R → acc}
```

#### CFG Productions as Pattern/Template Pairs

```metta
; CFG Rule: NP → DT N
Pattern:  (np (dt ?D) (n ?N))
Template: (noun_phrase ?D ?N)

; Error Production: Article error
Pattern:  (np (dt "a") (n ?N))  ; where is_vowel_initial(?N)
Template: (np (dt "an") (n ?N))
Cost: 0.5
```

**Benefit**: MORK's `transform_multi_multi_()` (space.rs:1221) handles CFG-style transformations natively.

### Integration Phases

| Phase | Component | Deliverable |
|-------|-----------|-------------|
| **A** | FuzzySource | Basic fuzzy matching in MORK queries |
| **B** | Lattice | DAG output, n-best paths, weighted edges |
| **C** | Full WFST | Phonetic NFA, FST composition |
| **D** | Grammar | CFG via MORK patterns, structural correction |

### Example Usage

```metta
; Phase A: Basic fuzzy matching
!(match &space (fuzzy "colr" 2 $result) $result)
; Returns: color, colour, collar, ...

; Phase B: Ranked results
!(match &space (fuzzy-ranked "phone" 3 5) $results)
; Returns: [(phone 0.0) (fone 0.3) (phon 0.5) ...]

; Phase C: Phonetic pattern matching
!(match &space
    (wfst-query
        (pattern "(ph|f)(one|oan)")
        (max-dist 2)
        (phonetic english)
        (top-k 10))
    $results)
```

### Key MORK Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `match2()` | expr/src/lib.rs:921 | Recursive structural pattern matching |
| `unify()` | expr/src/lib.rs:1849 | Robinson's unification with variable binding |
| `query_multi_i()` | kernel/src/space.rs:992 | Multi-source query with lattice support |
| `transform_multi_multi_()` | kernel/src/space.rs:1221 | Pattern → template transformation |

### Three-Tier Architecture with MORK

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Lexical (liblevenshtein)                            │
│   FST/Levenshtein automata → Word lattice                   │
│   Files: src/transducer/, src/lattice/, src/wfst/           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Syntactic (MORK)                                    │
│   - CFG rules compiled to pattern/template pairs            │
│   - query_multi_i() matches against lattice                 │
│   - transform_multi_multi_() applies corrections            │
│   - Output: Valid parse forest + corrections                │
│   Files: kernel/src/sources.rs, kernel/src/space.rs         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Semantic (Type checker / LLM)                       │
│   Final ranking and validation                              │
└─────────────────────────────────────────────────────────────┘
```

### Lattice Processing Efficiency

MORK's `query_multi_i()` handles lattices efficiently:

```
FST Lattice (Tier 1)
    ↓ O(K×N) edges (not K^N paths)
MORK Pattern Matching (Tier 2)
    ↓ CFG productions as patterns
Parse Forest + Corrections
```

**Complexity**: O(K×N) edge processing instead of O(K^N) path enumeration.

### Documentation

For detailed implementation guides, see:

- [MORK Integration Overview](../integration/mork/README.md)
- [FuzzySource Implementation](../integration/mork/fuzzy_source.md) (Phase A)
- [Lattice Integration](../integration/mork/lattice_integration.md) (Phase B)
- [WFST Composition](../integration/mork/wfst_composition.md) (Phase C)
- [Grammar Correction](../integration/mork/grammar_correction.md) (Phase D)
- [Structural Repair](../integration/mork/structural_repair.md) (Future)
- [PathMap Infrastructure](../integration/pathmap/README.md)

---

## Integration with Large Language Models

### Why LLM Integration Matters

**Industry Trend (2023-2025)**: Large language models (GPT-4, Claude, Gemini, Llama) dominate conversational AI, but they have limitations:

1. **Hallucination**: Neural models generate plausible but incorrect text
2. **Grammar inconsistency**: Even large models make grammatical errors
3. **Token inefficiency**: Typos and grammar errors waste context window
4. **Non-determinism**: Same input can produce different outputs

**liblevenshtein-rust's Role**: Provide **deterministic symbolic correction** that complements LLMs:
- **Before LLM**: Clean user input (preprocessing)
- **After LLM**: Validate generated output (postprocessing)
- **With LLM**: Hybrid symbolic + neural pipelines

---

### A. Preprocessing User Input for LLMs

**Problem**: User input contains typos, grammar errors, and noise that degrade LLM performance.

**Solution**: Apply WFST correction before passing input to LLM.

#### Why Preprocess?

1. **Improved LLM Understanding**:
   ```
   User: "teh cat dont lik me"
   Without preprocessing: LLM confused by "teh", "dont", "lik"
   With preprocessing: "the cat doesn't like me" → Clear intent
   ```

2. **Better Embeddings**:
   - Corrected text produces more accurate semantic embeddings
   - Improves similarity search in RAG (Retrieval-Augmented Generation)

3. **Token Efficiency**:
   - Corrected text may be shorter/clearer
   - More efficient use of limited context window

4. **Consistent Prompts**:
   - Few-shot examples benefit from clean, grammatical input
   - Better in-context learning

#### Preprocessing Pipeline

```rust
use liblevenshtein::transducer::Transducer;
use liblevenshtein::cfg::{Grammar, EarleyParser};

async fn preprocess_for_llm(input: &str) -> Result<String, Error> {
    // Step 1: FST spelling correction (Tier 1)
    let transducer = Transducer::for_dictionary(dictionary)
        .algorithm(Algorithm::Transposition)
        .max_distance(2)
        .build();

    let lattice = transducer
        .query(input)
        .to_lattice();

    // Step 2: CFG grammar correction (Tier 2)
    let grammar = Grammar::from_file("grammar.cfg")?;
    let parser = EarleyParser::new(&grammar);

    let forest = parser.parse_lattice(&lattice)?;

    // Extract best grammatical candidate
    let corrected = forest.best_parse()
        .ok_or(Error::NoGrammaticalParse)?
        .sentence();

    Ok(corrected)
}

// LLM integration
let user_input = "teh cat dont lik me";
let cleaned_input = preprocess_for_llm(user_input).await?;

// Pass to LLM (GPT/Claude/etc.)
let llm_response = llm_client
    .messages(&cleaned_input)  // "the cat doesn't like me"
    .await?;
```

#### Latency Analysis

**Fast Mode** (FST + NFA only, skip CFG):
```rust
let config = PipelineConfig {
    tiers: vec![Tier::FST, Tier::NFA],
    max_distance: 2,
};

let corrected = correction_pipeline
    .with_config(config)
    .correct(input)?;
```

**Latency Breakdown**:
```
FST spelling correction:    5-20ms
NFA phonetic matching:      10-30ms
Total preprocessing:        15-50ms

LLM inference:              200-2000ms  (dominates)
Total pipeline:             215-2050ms
```

**Overhead**: <3% added latency → **negligible for user experience**

#### When to Preprocess

**Always preprocess**:
- User queries in customer service chatbots
- Search queries for semantic search/RAG
- Educational applications (student input)
- User-generated content (social media, reviews)

**Skip preprocessing**:
- High-quality input (professional writing, API calls)
- Real-time dictation (already processed by ASR)
- Privacy-sensitive contexts (avoid sending to correction service)

---

### B. Postprocessing LLM Output

**Problem**: LLMs generate plausible text with subtle errors:
- Grammatical inconsistencies
- Stylistic violations
- Structural mistakes

**Solution**: Validate LLM output with deterministic WFST/CFG rules.

#### Why Postprocess?

1. **Fix Hallucinated Errors**:
   ```
   LLM output: "The cats was sleeping on the couches"
   Grammar issue: Subject-verb disagreement ("cats" + "was")
   Corrected: "The cats were sleeping on the couches"
   ```

2. **Enforce Style Guides**:
   - Ensure professional tone (no contractions, slang)
   - Enforce formatting rules (dates, citations)
   - Validate domain-specific constraints

3. **Structural Validation**:
   - Check JSON/XML structure
   - Validate code syntax
   - Ensure well-formed HTML

4. **Factual Consistency** (partial):
   - Can't fix semantic hallucinations
   - But can catch grammatical inconsistencies that signal errors

#### Postprocessing Pipeline

```rust
async fn postprocess_llm_output(llm_output: &str) -> Result<String, Error> {
    // Step 1: CFG grammar validation (Tier 2)
    let grammar = Grammar::from_file("strict_grammar.cfg")?;
    let parser = EarleyParser::new(&grammar);

    let tokens = tokenize(llm_output);
    let parse_result = parser.parse(&tokens)?;

    if !parse_result.is_grammatical() {
        // Attempt correction
        let lattice = generate_correction_lattice(llm_output)?;
        let forest = parser.parse_lattice(&lattice)?;

        return Ok(forest.best_parse()?.sentence());
    }

    // Step 2: FST spelling check (Tier 1 - optional)
    let spell_checked = spelling_checker.correct(llm_output)?;

    Ok(spell_checked)
}

// LLM generation
let llm_output = llm.generate("Write a story about a cat").await?;
// "teh cat runed fast and jump over a fence"

// Validate and correct
let validated = postprocess_llm_output(&llm_output).await?;
// "the cat ran fast and jumped over a fence"
```

#### Latency Analysis

```
LLM generation:             200-2000ms  (varies by model)
CFG grammar validation:     50-150ms
FST spelling check:         10-30ms
Total postprocessing:       60-180ms

Total pipeline:             260-2180ms
```

**Overhead**: 3-9% added latency → **acceptable trade-off for quality**

#### When to Postprocess

**Always postprocess**:
- Customer-facing text (emails, responses)
- Educational feedback (tutoring systems)
- Code generation (validate syntax)
- Structured output (JSON, SQL queries)

**Skip postprocessing**:
- Creative writing (might overcorrect intentional style)
- Brainstorming/draft mode (speed over correctness)
- When LLM output is already validated (e.g., via constrained decoding)

---

### C. Hybrid Workflows

Combine symbolic (WFST/CFG) and neural (LLM) approaches for optimal results.

#### Workflow Pattern 1: Cascading Corrections

**Full Pipeline**: User → WFST → LLM → CFG → Response

```rust
async fn process_user_query(input: &str) -> Result<String, Error> {
    // Stage 1: Preprocess user input (symbolic)
    let cleaned_input = preprocess_for_llm(input).await?;

    // Stage 2: LLM understanding + generation (neural)
    let llm_response = llm_client
        .chat(&cleaned_input)
        .await?;

    // Stage 3: Postprocess LLM output (symbolic)
    let validated_response = postprocess_llm_output(&llm_response).await?;

    Ok(validated_response)
}

// Example
let user_input = "wat is teh whether in ny today";
let response = process_user_query(user_input).await?;

// Flow:
// 1. Preprocessing: "wat is teh whether in ny today"
//                → "what is the weather in ny today"
// 2. LLM: Understands query, fetches weather, generates:
//         "the weather in New York is 72°F and sunny today"
// 3. Postprocessing: (already grammatical, unchanged)
// 4. Final: "The weather in New York is 72°F and sunny today"
```

**Benefits**:
- **Clean input** → better LLM understanding
- **Validated output** → guaranteed quality
- **Deterministic bookends** → predictable behavior

#### Workflow Pattern 2: Symbolic-First with Neural Fallback

**Strategy**: Try deterministic correction first, use LLM only if needed.

```rust
async fn correct_with_fallback(input: &str) -> Result<String, Error> {
    // Try symbolic correction (fast, deterministic)
    let symbolic_result = correction_pipeline
        .correct_with_confidence(input)?;

    if symbolic_result.confidence > 0.8 {
        // High confidence → use symbolic result
        return Ok(symbolic_result.text);
    }

    // Low confidence → use LLM (slow, but handles edge cases)
    let llm_result = llm_client
        .correct(input)
        .await?;

    Ok(llm_result)
}
```

**Benefits**:
- **95% fast path**: Most inputs handled symbolically (<50ms)
- **5% slow path**: Only ambiguous cases need LLM (200ms+)
- **Cost savings**: Reduce LLM API calls

#### Workflow Pattern 3: Neural Explanation of Symbolic Corrections

**Strategy**: WFST/CFG identifies errors, LLM generates explanations.

```rust
async fn explain_corrections(input: &str) -> Result<CorrectionReport, Error> {
    // Symbolic: Identify errors
    let corrections = correction_pipeline
        .correct_with_details(input)?;

    // Neural: Generate explanations
    let explanations = Vec::new();
    for error in corrections.errors {
        let prompt = format!(
            "Explain why '{}' should be '{}'. Focus on grammar rule.",
            error.original, error.corrected
        );

        let explanation = llm_client
            .generate(&prompt)
            .await?;

        explanations.push(explanation);
    }

    Ok(CorrectionReport {
        corrected_text: corrections.text,
        explanations,
    })
}

// Example: Educational assistant
let student_input = "the cat dont like me";
let report = explain_corrections(student_input).await?;

// Output:
// Corrected: "the cat doesn't like me"
// Explanation: "The verb 'like' requires 'doesn't' (does not) in negative
//               form with third-person singular subject 'cat'. 'Dont' is
//               not a word; the contraction 'don't' (do not) is used with
//               plural subjects or 'I/you'."
```

**Benefits**:
- **Precise error detection** (symbolic)
- **Human-friendly explanations** (neural)
- **Best of both worlds**

---

### D. Practical Application Scenarios

#### Scenario 1: Customer Service Chatbot

**Problem**: Users type quickly with typos, chatbot must understand intent.

```rust
struct CustomerServiceBot {
    correction: CorrectionPipeline,
    llm: LLMClient,
}

impl CustomerServiceBot {
    async fn handle_query(&self, user_input: &str) -> Result<String, Error> {
        // Preprocess: Clean user input
        let cleaned = self.correction
            .preprocess(user_input)
            .await?;

        // LLM: Understand intent + generate response
        let context = format!(
            "Customer query: {}\nProvide helpful assistance.",
            cleaned
        );

        let response = self.llm
            .chat(&context)
            .await?;

        // Postprocess: Ensure professional tone
        let validated = self.correction
            .postprocess_with_style(&response, StyleGuide::Professional)
            .await?;

        Ok(validated)
    }
}

// Example interaction
let bot = CustomerServiceBot::new();

let user_input = "helo, my order didnt arive yet. can u help me pls?";
let response = bot.handle_query(user_input).await?;

// Internal flow:
// 1. Preprocessed: "hello, my order didn't arrive yet. can you help me please?"
// 2. LLM: Understands order issue, generates support response
// 3. Validated: Ensures response is grammatical and professional
// 4. Output: "I apologize for the delay. Let me help you track your order..."
```

**Benefits**:
- **Robust to typos**: Understands intent despite errors
- **Professional output**: Guaranteed grammatical responses
- **Fast**: <300ms total (acceptable for chat)

#### Scenario 2: Educational Writing Assistant

**Problem**: Students need feedback on grammar errors with explanations.

```rust
async fn provide_writing_feedback(essay: &str) -> Result<Feedback, Error> {
    // Tier 1-2: Detect all grammatical errors (symbolic)
    let corrections = correction_pipeline
        .identify_all_errors(essay)?;

    // Tier 3: Generate pedagogical explanations (neural)
    let feedback_items = Vec::new();
    for error in corrections.errors {
        let prompt = format!(
            "You are a grammar tutor. Explain this error to a student:\n\
             Incorrect: '{}'\n\
             Correct: '{}'\n\
             Provide a clear, concise explanation with the grammar rule.",
            error.original, error.corrected
        );

        let explanation = educational_llm
            .generate(&prompt)
            .await?;

        feedback_items.push(FeedbackItem {
            location: error.position,
            severity: error.severity,
            correction: error.corrected,
            explanation,
        });
    }

    Ok(Feedback {
        original: essay.to_string(),
        corrected: corrections.text,
        items: feedback_items,
    })
}
```

**Benefits**:
- **Precise error detection** (CFG rules catch all violations)
- **Pedagogical explanations** (LLM adapts to student level)
- **Trust**: Students can verify symbolic rules

#### Scenario 3: Semantic Search / RAG Systems

**Problem**: Query typos reduce retrieval accuracy in vector databases.

```rust
async fn semantic_search(query: &str, index: &VectorIndex) -> Vec<Document> {
    // Preprocess: Correct query before embedding
    let corrected_query = correction_pipeline
        .preprocess(query)
        .await?;

    // Embed corrected query
    let query_embedding = embedding_model
        .embed(&corrected_query)
        .await?;

    // Retrieve top-K documents
    let results = index
        .search(query_embedding, k=10)
        .await?;

    results
}

// Example
let user_query = "hw to bild a websit with pythn";

// Without correction:
// Embedding of "hw to bild a websit with pythn" → poor matches

// With correction:
// "how to build a website with python" → accurate retrieval
```

**Benefits**:
- **Better recall**: Corrected query matches more documents
- **Better precision**: Semantic similarity is more accurate
- **User experience**: Users don't need to correct themselves

#### Scenario 4: Code Generation with Validation

**Problem**: LLMs generate code with syntax errors or style violations.

```rust
async fn generate_validated_code(spec: &str) -> Result<String, Error> {
    // LLM: Generate code
    let generated_code = code_llm
        .generate(spec)
        .await?;

    // CFG: Validate syntax (using programming language grammar)
    let syntax_validator = CFGValidator::for_language(Language::Python);

    let validation_result = syntax_validator
        .validate(&generated_code)?;

    if !validation_result.is_valid {
        // Attempt repair
        let repaired = syntax_validator
            .repair(&generated_code)?;

        return Ok(repaired);
    }

    Ok(generated_code)
}
```

**Benefits**:
- **Syntax guarantee**: Output compiles/runs
- **Style enforcement**: Follows project conventions
- **Fewer iterations**: User doesn't need to debug syntax

---

### E. Latency and Performance Considerations

#### Latency Breakdown (Typical Values)

| Component | Fast Mode | Balanced Mode | Accurate Mode |
|-----------|-----------|---------------|---------------|
| **FST (Tier 1)** | 5-20ms | 10-30ms | 20-50ms |
| **NFA (Tier 1)** | 10-30ms | 20-50ms | 50-100ms |
| **CFG (Tier 2)** | - | 50-150ms | 100-200ms |
| **Symbolic Total** | 15-50ms | 80-230ms | 170-350ms |
| **LLM (Tier 3)** | 200-500ms | 500-1000ms | 1000-2000ms |
| **Full Pipeline** | 215-550ms | 580-1230ms | 1170-2350ms |

**Key Insight**: Symbolic correction (Tiers 1-2) adds <15% overhead vs. LLM latency.

#### Async Pipeline Design

**Concurrent Preprocessing + LLM Warmup**:

```rust
async fn optimized_pipeline(input: &str) -> Result<String, Error> {
    // Start correction and LLM warmup concurrently
    let correction_future = preprocess_for_llm(input);
    let llm_warmup_future = llm_client.warmup();  // Establish connection

    let (corrected, _) = tokio::join!(correction_future, llm_warmup_future);

    // LLM already warmed up → faster inference
    let response = llm_client
        .chat(&corrected?)
        .await?;

    Ok(response)
}
```

**Latency Savings**: ~50-100ms from concurrent warmup

#### When Preprocessing is Worth the Overhead

**Always worth it** (latency negligible vs. benefit):
- User queries (correctness >> speed)
- Batch processing (throughput matters, not latency)
- Async systems (pipelining hides latency)

**Trade-off zone** (consider caching):
- Real-time chat (but 50ms is acceptable)
- Mobile apps (but improves accuracy)

**Skip if**:
- Sub-50ms requirement (use streaming instead)
- Input guaranteed clean (API calls from code)

---

### F. Complementary Architecture Benefits

#### Why Combine Symbolic + Neural?

**Symbolic (WFST/CFG) Strengths**:
- ✅ **Deterministic**: Same input → same output
- ✅ **Verifiable**: Rules can be inspected and tested
- ✅ **Fast**: O(n) to O(n³) algorithms
- ✅ **No training data**: Hand-written rules
- ✅ **Interpretable**: Clear error explanations

**Symbolic (WFST/CFG) Limitations**:
- ❌ **Semantic blindness**: Can't understand meaning
- ❌ **Rigid**: Can't handle creative language
- ❌ **Coverage**: Limited to coded rules

**Neural (LLM) Strengths**:
- ✅ **Semantic understanding**: Grasps context and meaning
- ✅ **Flexible**: Handles novel inputs gracefully
- ✅ **General**: Trained on broad distribution

**Neural (LLM) Limitations**:
- ❌ **Hallucination**: Generates plausible but wrong text
- ❌ **Non-deterministic**: Sampling introduces variance
- ❌ **Expensive**: High computational cost
- ❌ **Black-box**: Hard to debug or explain

#### Complementary Design Principles

**Principle 1: Symbolic Preprocessing**
- Use WFST/CFG for structured, rule-based corrections
- Ensures clean input for LLM semantic understanding

**Principle 2: Neural Semantic Layer**
- Use LLM for tasks requiring world knowledge
- Handle ambiguity and context-dependent decisions

**Principle 3: Symbolic Validation**
- Use CFG to validate LLM outputs
- Catch structural errors before user sees them

**Principle 4: Graceful Degradation**
- If symbolic correction fails, fall back to LLM
- If LLM fails, return symbolic result (better than nothing)

#### Quote from NVIDIA NeMo (arXiv:2104.05055)

> "Low tolerance towards unrecoverable errors is the main reason why most ITN systems in production are still largely rule-based using WFSTs"

**Interpretation for LLM Integration**:
- LLMs can hallucinate → unrecoverable errors
- WFST/CFG provides **deterministic safety net**
- Hybrid approach gets **best of both worlds**

---

### Summary: LLM Integration Patterns

| Use Case | Pattern | Tiers Used | Latency | Benefit |
|----------|---------|------------|---------|---------|
| **Customer chatbot** | Preprocess → LLM | 1+2 → 3 | +80-230ms | Robust to typos |
| **Writing assistant** | Symbolic detect + LLM explain | 1+2+3 | +170-350ms | Precise + pedagogical |
| **Search/RAG** | Preprocess query | 1+2 | +80-230ms | Better retrieval |
| **Code generation** | LLM → CFG validate | 3 → 2 | +100-200ms | Syntax guarantee |
| **Content moderation** | LLM → CFG validate | 3 → 2 | +100-200ms | Structural checks |

**Key Takeaway**: liblevenshtein-rust's WFST+CFG layers provide **deterministic symbolic correction** that complements (not replaces) LLMs, enabling production-grade conversational AI systems with guaranteed quality.

---

### Extended Architecture for LLM Agents

The three-tier WFST core is extended by the **MeTTaIL correction architecture** for full LLM agent support:

```
┌─────────────────────────────────────────────────────────────────┐
│          EXTENDED CORRECTION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│  Dialogue Context Layer                                         │
│    Turn History │ Entity Registry │ Topic Graph                 │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           THREE-TIER WFST CORE (This Document)            │  │
│  │  Tier 1: Lexical → Tier 2: Syntactic → Tier 3: Semantic   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  LLM Integration Layer                                          │
│    Preprocessing → LLM API → Postprocessing                     │
│                              ↓                                   │
│  Agent Learning Layer                                           │
│    Feedback Collection │ Pattern Learning │ Online Updates      │
└─────────────────────────────────────────────────────────────────┘
```

**Extended Layer Documentation**:
- [Dialogue Context Layer](../mettail/dialogue/README.md) - Coreference resolution and topic tracking
- [LLM Integration Layer](../mettail/llm-integration/README.md) - Detailed preprocessing/postprocessing
- [Agent Learning Layer](../mettail/agent-learning/README.md) - Feedback integration and personalization
- [Full Architecture Overview](../mettail/correction-wfst/01-architecture-overview.md) - Complete 6-layer design

---

## Comparison with Industry Systems

### NVIDIA NeMo

**Architecture**: Hybrid WFST + Neural

**Strengths**:
- ✅ Production-ready (daily use by millions)
- ✅ Open source (Python + C++ runtime)
- ✅ Multi-language support (19+ languages)
- ✅ Extensive documentation

**Limitations**:
- ❌ FST-only for symbolic layer (no CFG)
- ❌ Must use neural models for grammar
- ⚠️ Python overhead (mitigated by Sparrowhawk export)

**Comparison to liblevenshtein-rust**:

| Feature | NVIDIA NeMo | liblevenshtein-rust |
|---------|-------------|---------------------|
| Spelling correction | ✅ FST | ✅ FST (Levenshtein) |
| Phonetic normalization | ✅ FST rules | ✅ NFA regex + verified rules |
| Grammar correction | ⚠️ Neural only | ✅ CFG + Neural |
| Formal verification | ❌ None | ✅ Coq proofs (phonetic) |
| Runtime performance | ⚠️ Python/C++ | ✅ Rust (native) |
| Deterministic output | ⚠️ Neural layer | ✅ Tier 1+2 symbolic |

### Google Sparrowhawk

**Architecture**: Pure WFST (text normalization for TTS)

**Strengths**:
- ✅ Production-grade C++ implementation
- ✅ Open source (Apache 2.0)
- ✅ Extremely fast (<10ms latency)
- ✅ Robust to edge cases

**Limitations**:
- ❌ Text normalization only (not grammar correction)
- ❌ FST-only (no CFG or neural)
- ❌ Requires Pynini for grammar development

**Comparison to liblevenshtein-rust**:

| Feature | Google Sparrowhawk | liblevenshtein-rust |
|---------|-------------------|---------------------|
| Text normalization | ✅ FST | ✅ FST + NFA |
| Grammar correction | ❌ Not supported | ✅ CFG |
| Development language | Python (Pynini) → C++ | Rust (native) |
| Deployment | C++ runtime | Rust binary |
| Phonetic rules | ⚠️ Manual FST | ✅ Verified + NFA regex |

### MoNoise (SOTA Lexical Normalization)

**Architecture**: Neural sequence-to-sequence

**Strengths**:
- ✅ State-of-the-art on LexNorm benchmark
- ✅ Handles creative spellings well
- ✅ End-to-end trainable

**Limitations**:
- ❌ Requires large labeled training data
- ❌ Prone to hallucination errors
- ❌ Latency: 100-500ms per sentence
- ❌ Non-deterministic output

**Comparison to liblevenshtein-rust**:

| Feature | MoNoise (Neural) | liblevenshtein-rust |
|---------|-----------------|---------------------|
| Accuracy (LexNorm) | ✅ SOTA | ⚠️ TBD (not yet benchmarked) |
| Latency | ❌ 100-500ms | ✅ <50ms (symbolic layers) |
| Training data needed | ❌ 10,000+ examples | ✅ None (rules-based) |
| Deterministic | ❌ No | ✅ Tiers 1-2 yes |
| Unrecoverable errors | ❌ Prone to hallucination | ✅ FST+CFG constrained |

### Summary: liblevenshtein-rust's Niche

**Unique Value Proposition**:

1. **Only system with FST + CFG + Neural three-tier architecture**
   - FST for speed (O(n))
   - CFG for syntax (O(n³))
   - Neural for semantics (optional)

2. **Formally verified phonetic rules** (Coq proofs)
   - Industry systems use manual FST (error-prone)
   - liblevenshtein-rust: Proven correct by construction

3. **Deterministic symbolic layers**
   - Tiers 1-2 reproducible, no hallucination
   - Neural only for disambiguation (optional)

4. **Rust native performance**
   - Zero-cost abstractions
   - Memory safety without GC overhead
   - Competitive with C++ Sparrowhawk

5. **Composable architecture**
   - NFA ∩ FST ∩ CFG composition
   - Modular: Each layer independently testable

**Target Users**:
- Applications requiring **deterministic** corrections (medical, legal)
- **Low-latency** systems (mobile, embedded)
- **Resource-constrained** environments (no GPU for neural)
- **Safety-critical** systems (formal verification)

---

## Performance Characteristics

### Latency Analysis

**Tier 1: Regular (FST/NFA)**

| Operation | Complexity | Latency (estimate) | Notes |
|-----------|------------|-------------------|-------|
| Levenshtein automaton | O(n) | <5ms | Pre-compiled dictionary |
| NFA phonetic regex | O(n·m) | <10ms | m = NFA states (small) |
| FST composition | O(n) | <5ms | Lazy evaluation |
| **Total Tier 1** | **O(n)** | **<20ms** | Deterministic |

**Tier 2: Context-Free (CFG)**

| Operation | Complexity | Latency (estimate) | Notes |
|-----------|------------|-------------------|-------|
| Earley parsing | O(n³) worst, O(n²) avg | <100ms | Depends on grammar size |
| CYK parsing (CNF) | O(n³·\|G\|) | <150ms | Requires CNF conversion |
| Error detection | O(n³) | <100ms | Same as parsing |
| Correction application | O(n) | <5ms | Tree transformation |
| **Total Tier 2** | **O(n³)** | **<200ms** | Deterministic |

**Tier 3: Neural (Optional)**

| Operation | Complexity | Latency (estimate) | Notes |
|-----------|------------|-------------------|-------|
| BERT masked LM | O(n²) | 50-200ms | Depends on batch size |
| GPT autoregressive | O(n²) | 100-500ms | Sequential generation |
| Lattice rescoring | O(k·n²) | 50-300ms | k = lattice size |
| **Total Tier 3** | **O(n²)** | **50-500ms** | Non-deterministic |

### Deployment Mode Latencies

**Fast Mode** (FST + NFA only):
- Latency: <20ms per sentence
- Accuracy: ~85% (spelling + phonetic)
- Use case: Mobile, embedded, real-time chat

**Balanced Mode** (FST + NFA + CFG):
- Latency: <200ms per sentence
- Accuracy: ~90% (+ grammar)
- Use case: Desktop applications, batch processing

**Accurate Mode** (FST + NFA + CFG + Neural):
- Latency: <500ms per sentence
- Accuracy: ~95% (+ semantic disambiguation)
- Use case: High-quality document processing

### Memory Footprint

**Tier 1: Regular**
- Dictionary trie: 10-100 MB (depends on vocabulary)
- Levenshtein automaton: 1-10 KB (per query)
- NFA phonetic: 10-100 KB (compiled rules)
- **Total**: <100 MB

**Tier 2: Context-Free**
- Error grammar: 1-5 MB (production rules)
- Chart: O(n²·|G|) = 1-10 MB (depends on sentence length)
- **Total**: <20 MB (per sentence)

**Tier 3: Neural**
- BERT model: 400 MB (base), 1.3 GB (large)
- Inference memory: 100-500 MB
- **Total**: 0.5-2 GB

**Total System**: 0.6-2.2 GB (with all tiers)

### Scalability

**Horizontal Scaling**:
- FST/CFG layers: Stateless, trivially parallelizable
- Neural layer: Batch multiple sentences for GPU efficiency

**Vertical Scaling**:
- Dictionary size: O(|V|) lookup (constant with trie)
- Grammar size: O(|G|) parsing (linear in grammar size)

---

## Deployment Modes

### 1. Fast Mode (Real-time)

**Configuration**:
```rust
let config = PipelineConfig {
    tiers: vec![Tier::FST, Tier::NFA],
    max_edit_distance: 2,
    phonetic_regex: Some("(ph|f)(ough|uff)..."),
    grammar: None,
    neural_lm: None,
};
```

**Performance**:
- Latency: <20ms
- Throughput: >1000 sentences/sec (single core)
- Memory: <100 MB

**Use Cases**:
- Mobile keyboards
- Real-time chat normalization
- Embedded devices
- High-throughput batch processing

**Trade-offs**:
- ✅ Very fast
- ✅ Deterministic
- ❌ No grammar correction
- ❌ No semantic disambiguation

### 2. Balanced Mode (Production)

**Configuration**:
```rust
let config = PipelineConfig {
    tiers: vec![Tier::FST, Tier::NFA, Tier::CFG],
    max_edit_distance: 2,
    phonetic_regex: Some("..."),
    grammar: Some(load_error_grammar("grammar.cfg")?),
    neural_lm: None,
};
```

**Performance**:
- Latency: <200ms
- Throughput: >100 sentences/sec
- Memory: <200 MB

**Use Cases**:
- Desktop applications
- Server-side normalization
- Document processing
- Email/SMS cleanup

**Trade-offs**:
- ✅ Grammar correction
- ✅ Still deterministic
- ⚠️ Moderate latency
- ❌ No semantic disambiguation

### 3. Accurate Mode (High-quality)

**Configuration**:
```rust
let config = PipelineConfig {
    tiers: vec![Tier::FST, Tier::NFA, Tier::CFG, Tier::Neural],
    max_edit_distance: 2,
    phonetic_regex: Some("..."),
    grammar: Some(load_error_grammar("grammar.cfg")?),
    neural_lm: Some(BertLanguageModel::load("bert-base-uncased")?),
    neural_weight: 0.3,
};
```

**Performance**:
- Latency: <500ms
- Throughput: >10 sentences/sec (CPU), >100 (GPU)
- Memory: 0.5-2 GB

**Use Cases**:
- High-quality document editing
- Professional writing tools
- Academic paper correction
- Offline batch processing

**Trade-offs**:
- ✅ Best accuracy
- ✅ Semantic disambiguation
- ❌ Higher latency
- ❌ Non-deterministic (neural layer)

---

## References

### Key Papers

1. **Shallow Fusion of WFST and Language Model for Text Normalization**
   - Authors: NVIDIA NeMo team
   - arXiv: 2203.15917
   - Year: 2022
   - Key idea: Non-deterministic WFST + neural LM disambiguation

2. **NeMo Inverse Text Normalization: From Development To Production**
   - arXiv: 2104.05055
   - Year: 2021
   - Production system: Hybrid FST + Neural

3. **Neural Grammatical Error Correction with Finite State Transducers**
   - Authors: Stahlberg, Bryant, Byrne
   - arXiv: 1903.10625
   - Venue: NAACL 2019
   - Key finding: FST + neural outperforms pure neural for GEC

4. **Soft-Masked BERT for Spelling Error Correction**
   - Authors: Zhang, Huang, et al.
   - arXiv: 2005.07421
   - Venue: ACL 2020
   - Architecture: Detection network + correction network

5. **The Kestrel TTS Text Normalization System**
   - Venue: Natural Language Engineering (Cambridge)
   - System: Google's production FST system (open-sourced as Sparrowhawk)

### Tools and Frameworks

- **OpenFST**: http://www.openfst.org/
- **Pynini**: https://www.opengrm.org/twiki/bin/view/GRM/Pynini
- **Sparrowhawk**: https://github.com/google/sparrowhawk
- **NVIDIA NeMo**: https://github.com/NVIDIA/NeMo-text-processing

### Benchmarks

- **LexNorm**: Lexical normalization benchmark
- **W-NUT 2015**: Twitter normalization (2,577 tweets)
- **W-NUT 2021**: Multilingual (12 languages)
- **CoNLL-2014**: Grammatical error correction
- **BEA-2019**: GEC shared task

### Further Reading

See `literature_review.md` for detailed paper summaries and `references/` directory for comprehensive bibliography.
