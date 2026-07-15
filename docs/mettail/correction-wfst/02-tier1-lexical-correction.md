# Tier 1: Lexical Correction

This document describes Tier 1 of the correction WFST architecture: lexical
correction via liblevenshtein.

**Source**: `/home/dylon/Workspace/f1r3fly.io/liblevenshtein-rust/`

**Related Integration Docs**:
- [FuzzySource Implementation](../../integration/mork/fuzzy_source.md) - Phase A details
- [Lattice Integration](../../integration/mork/lattice_integration.md) - Phase B details
- [WFST Composition](../../integration/mork/wfst_composition.md) - Phase C details
- [MORK Integration Overview](../../integration/mork/README.md) - Complete phase documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Edit Distance Automata](#edit-distance-automata)
3. [Phonetic Rules](#phonetic-rules)
4. [Dictionary Backends](#dictionary-backends)
5. [WFST Semirings](#wfst-semirings)
6. [Candidate Lattice Generation](#candidate-lattice-generation)
7. [FuzzySource Trait](#fuzzysource-trait)

---

## Overview

Tier 1 generates lexical correction candidates using edit distance and phonetic
similarity, producing a candidate lattice for downstream tiers.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tier 1: Lexical Correction                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: "teh"                                                   │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Edit Distance Automata                      ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ││
│  │  │  Levenshtein  │  │   Damerau-    │  │    Custom     │  ││
│  │  │   Distance    │  │  Levenshtein  │  │   Weights     │  ││
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  ││
│  │          │                  │                  │           ││
│  │          └──────────────────┼──────────────────┘           ││
│  │                             ▼                              ││
│  │              Edit-based candidates: the, tea, ten, ...    ││
│  └─────────────────────────────┬───────────────────────────────┘│
│                                │                                 │
│  ┌─────────────────────────────┼───────────────────────────────┐│
│  │                 Phonetic Rules                              ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ││
│  │  │   Metaphone   │  │    Soundex    │  │  Caverphone2  │  ││
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  ││
│  │          │                  │                  │           ││
│  │          └──────────────────┼──────────────────┘           ││
│  │                             ▼                              ││
│  │            Phonetic candidates: tea, tee, the, ...        ││
│  └─────────────────────────────┬───────────────────────────────┘│
│                                │                                 │
│                                ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Candidate Lattice                          ││
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐              ││
│  │  │ the │──│ tea │──│ ten │──│ tee │──│tech │              ││
│  │  │ 1   │  │ 1   │  │ 1   │  │ 2   │  │ 2  │              ││
│  │  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘              ││
│  │  (distance weights shown)                                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Edit Distance Automata

liblevenshtein provides multiple edit distance algorithms:

### Levenshtein Distance

Classic edit distance with insertions, deletions, and substitutions:

```rust
/// Levenshtein automaton for fuzzy matching
pub struct LevenshteinAutomaton {
    /// Query term
    query: Vec<u8>,
    /// Maximum edit distance
    max_distance: u8,
    /// Character position states
    states: Vec<LevenshteinState>,
}

impl LevenshteinAutomaton {
    /// Create automaton for query with max distance
    pub fn new(query: &[u8], max_distance: u8) -> Self {
        let states = build_parametric_states(query.len(), max_distance);
        Self { query: query.to_vec(), max_distance, states }
    }

    /// Check if candidate matches within distance
    pub fn matches(&self, candidate: &[u8]) -> Option<u8> {
        let mut state = self.initial_state();
        for &byte in candidate {
            state = self.transition(state, byte);
            if self.is_dead(state) {
                return None;
            }
        }
        self.distance(state)
    }
}
```

**Supported Operations**:
| Operation | Cost | Example |
|-----------|------|---------|
| Insertion | 1 | cat → ca**r**t |
| Deletion | 1 | cart → cat |
| Substitution | 1 | cat → c**o**t |

### Damerau-Levenshtein Distance

Extends Levenshtein with transposition:

```rust
/// Damerau-Levenshtein extends Levenshtein with transposition
pub struct DamerauLevenshteinAutomaton {
    /// Base Levenshtein automaton
    base: LevenshteinAutomaton,
    /// Track previous character for transposition detection
    transposition_states: Vec<TranspositionState>,
}

impl DamerauLevenshteinAutomaton {
    /// Transition with transposition check
    fn transition(&self, state: State, byte: u8) -> State {
        let base_state = self.base.transition(state.base, byte);

        // Check for transposition opportunity
        if state.prev_char == byte && self.can_transpose(state) {
            let transposed = self.apply_transposition(state);
            return self.min_distance_state(base_state, transposed);
        }

        State { base: base_state, prev_char: byte, .. }
    }
}
```

**Additional Operation**:
| Operation | Cost | Example |
|-----------|------|---------|
| Transposition | 1 | teh → t**he** |

### Custom Edit Costs

Support for weighted operations:

```rust
/// Custom edit costs for domain-specific correction
pub struct EditCosts {
    /// Cost of insertion
    pub insert: f32,
    /// Cost of deletion
    pub delete: f32,
    /// Cost matrix for substitution (256 × 256 for bytes)
    pub substitute: [[f32; 256]; 256],
    /// Cost of transposition
    pub transpose: f32,
}

impl EditCosts {
    /// QWERTY keyboard distance weights
    pub fn qwerty() -> Self {
        let mut substitute = [[1.0; 256]; 256];
        // Adjacent keys have lower cost
        substitute[b'q' as usize][b'w' as usize] = 0.5;
        substitute[b'w' as usize][b'e' as usize] = 0.5;
        // ... etc
        Self { insert: 1.0, delete: 1.0, substitute, transpose: 0.8 }
    }
}
```

---

## Phonetic Rules

Phonetic algorithms encode words by their pronunciation:

### Metaphone / Double Metaphone

Encodes English pronunciation patterns:

```rust
/// Double Metaphone produces primary and alternate encodings
pub struct DoubleMetaphone;

impl DoubleMetaphone {
    /// Encode word to phonetic codes
    pub fn encode(word: &str) -> (String, Option<String>) {
        let mut primary = String::new();
        let mut alternate = String::new();

        // Apply phonetic transformation rules
        // "knight" → ("NT", None)
        // "cesar" → ("SSR", Some("TSR"))

        (primary, if alternate.is_empty() { None } else { Some(alternate) })
    }
}
```

**Example Encodings**:
| Word | Primary | Alternate |
|------|---------|-----------|
| knight | NT | - |
| night | NT | - |
| caesar | SSR | TSR |

### Soundex

Classic phonetic algorithm (4 characters):

```rust
/// Soundex: 4-character phonetic code
pub fn soundex(word: &str) -> String {
    let mut code = String::with_capacity(4);
    let chars: Vec<char> = word.to_uppercase().chars().collect();

    // Keep first letter
    if let Some(&first) = chars.first() {
        code.push(first);
    }

    // Map remaining to digits
    for &c in &chars[1..] {
        let digit = match c {
            'B' | 'F' | 'P' | 'V' => '1',
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
            'D' | 'T' => '3',
            'L' => '4',
            'M' | 'N' => '5',
            'R' => '6',
            _ => continue,
        };
        if code.chars().last() != Some(digit) {
            code.push(digit);
            if code.len() == 4 { break; }
        }
    }

    // Pad to 4 characters
    while code.len() < 4 { code.push('0'); }
    code
}
```

### Caverphone2

Optimized for New Zealand/Australian English:

```rust
/// Caverphone2: 10-character phonetic code
pub fn caverphone2(word: &str) -> String {
    let mut result = word.to_lowercase();

    // Apply transformation rules
    result = result.replace("cough", "cof2");
    result = result.replace("rough", "rof2");
    result = result.replace("tough", "tof2");
    // ... more rules

    // Pad/truncate to 10 characters
    format!("{:10}", result)[..10].to_string()
}
```

---

## Dictionary Backends

liblevenshtein provides multiple dictionary implementations:

### DynamicDawg / DynamicDawgChar

Best for general purpose, updateable dictionaries:

```rust
/// DynamicDawg: DAWG with SIMD and Bloom filter optimizations
pub struct DynamicDawg {
    /// Root node
    root: DawgNode,
    /// Bloom filter for fast negative lookups
    bloom: BloomFilter,
    /// SIMD-optimized transition table
    transitions: SIMDTransitionTable,
}

impl DynamicDawg {
    /// Insert word into dictionary
    pub fn insert(&mut self, word: &[u8]) -> bool;

    /// Remove word from dictionary
    pub fn remove(&mut self, word: &[u8]) -> bool;

    /// Fuzzy search with max edit distance
    pub fn fuzzy_search(&self, query: &[u8], max_dist: u8)
        -> impl Iterator<Item = (&[u8], u8)>;
}
```

**Features**:
- SIMD-accelerated character comparison
- Bloom filter for fast negative lookups
- Dynamic updates (insert/remove)
- Both ASCII (`DynamicDawg`) and UTF-8 (`DynamicDawgChar`)

### DoubleArrayTrie / DoubleArrayTrieChar

Fastest for static, read-only dictionaries:

```rust
/// DoubleArrayTrie: Compact, fast static trie
pub struct DoubleArrayTrie {
    /// Base array (transitions)
    base: Vec<i32>,
    /// Check array (validation)
    check: Vec<i32>,
    /// Terminal markers
    terminal: BitVec,
}

impl DoubleArrayTrie {
    /// Build from sorted word list
    pub fn build(words: &[&[u8]]) -> Self;

    /// Exact lookup
    pub fn contains(&self, word: &[u8]) -> bool;

    /// Prefix search
    pub fn prefix_search(&self, prefix: &[u8])
        -> impl Iterator<Item = &[u8]>;
}
```

**Features**:
- Compact memory representation
- Cache-friendly access patterns
- Fastest exact lookups
- Best for static dictionaries (e.g., English words)

### SuffixAutomaton / SuffixAutomatonChar

For substring searching:

```rust
/// SuffixAutomaton: Efficient substring matching
pub struct SuffixAutomaton {
    /// States in the automaton
    states: Vec<SuffixState>,
    /// Character transitions
    transitions: Vec<HashMap<u8, usize>>,
}

impl SuffixAutomaton {
    /// Build from text corpus
    pub fn build(text: &[u8]) -> Self;

    /// Find all occurrences of pattern
    pub fn find_all(&self, pattern: &[u8])
        -> impl Iterator<Item = usize>;

    /// Check if pattern is substring
    pub fn contains_substring(&self, pattern: &[u8]) -> bool;
}
```

### Backend Comparison

| Backend | Insert | Lookup | Memory | Best For |
|---------|--------|--------|--------|----------|
| DynamicDawg | $`\mathcal{O}(n)`$ | $`\mathcal{O}(n)`$ | Medium | General purpose |
| DoubleArrayTrie | N/A | $`\mathcal{O}(n)`$ | Low | Static dictionaries |
| SuffixAutomaton | $`\mathcal{O}(n)`$ | $`\mathcal{O}(n)`$ | High | Substring search |

---

## WFST Semirings

Weights in the correction transducer use semiring algebras:

### Tropical Semiring

For shortest path (Viterbi) decoding:

```rust
/// Tropical semiring: (min, +)
#[derive(Clone, Copy)]
pub struct Tropical(pub f32);

impl Semiring for Tropical {
    fn zero() -> Self { Tropical(f32::INFINITY) }
    fn one() -> Self { Tropical(0.0) }

    /// Combine paths: take minimum
    fn add(&self, other: &Self) -> Self {
        Tropical(self.0.min(other.0))
    }

    /// Extend path: add costs
    fn mul(&self, other: &Self) -> Self {
        Tropical(self.0 + other.0)
    }
}
```

**Use**: Finding single best correction.

### Log Semiring

For probabilistic (forward-backward) computation:

```rust
/// Log semiring: (log-sum-exp, +)
#[derive(Clone, Copy)]
pub struct Log(pub f32);

impl Semiring for Log {
    fn zero() -> Self { Log(f32::NEG_INFINITY) }
    fn one() -> Self { Log(0.0) }

    /// Combine: log-sum-exp for numerical stability
    fn add(&self, other: &Self) -> Self {
        if self.0 == f32::NEG_INFINITY { return *other; }
        if other.0 == f32::NEG_INFINITY { return *self; }
        let max = self.0.max(other.0);
        Log(max + ((self.0 - max).exp() + (other.0 - max).exp()).ln())
    }

    /// Extend: add log probabilities
    fn mul(&self, other: &Self) -> Self {
        Log(self.0 + other.0)
    }
}
```

**Use**: Computing total probability over all paths.

### Probability Semiring

For raw probability computation:

```rust
/// Probability semiring: (+, ×)
#[derive(Clone, Copy)]
pub struct Probability(pub f64);

impl Semiring for Probability {
    fn zero() -> Self { Probability(0.0) }
    fn one() -> Self { Probability(1.0) }

    fn add(&self, other: &Self) -> Self {
        Probability(self.0 + other.0)
    }

    fn mul(&self, other: &Self) -> Self {
        Probability(self.0 * other.0)
    }
}
```

### Boolean Semiring

For acceptance checking:

```rust
/// Boolean semiring: (OR, AND)
#[derive(Clone, Copy)]
pub struct Boolean(pub bool);

impl Semiring for Boolean {
    fn zero() -> Self { Boolean(false) }
    fn one() -> Self { Boolean(true) }

    fn add(&self, other: &Self) -> Self {
        Boolean(self.0 || other.0)
    }

    fn mul(&self, other: &Self) -> Self {
        Boolean(self.0 && other.0)
    }
}
```

---

## Candidate Lattice Generation

The output of Tier 1 is a candidate lattice:

```rust
/// Candidate lattice for downstream processing
pub struct CandidateLattice<W: Semiring> {
    /// Nodes represent positions in input
    nodes: Vec<LatticeNode>,
    /// Edges represent candidate corrections
    edges: Vec<LatticeEdge<W>>,
}

/// Edge in candidate lattice
pub struct LatticeEdge<W: Semiring> {
    /// Source node
    from: usize,
    /// Target node
    to: usize,
    /// Candidate string
    candidate: Vec<u8>,
    /// Weight (edit distance, probability, etc.)
    weight: W,
}

impl<W: Semiring> CandidateLattice<W> {
    /// Build lattice from fuzzy search results
    pub fn from_candidates(
        candidates: impl Iterator<Item = (Vec<u8>, W)>,
    ) -> Self {
        let mut lattice = Self::empty();
        for (candidate, weight) in candidates {
            lattice.add_edge(0, 1, candidate, weight);
        }
        lattice
    }

    /// Compose with another lattice (e.g., language model)
    pub fn compose(&self, other: &CandidateLattice<W>) -> CandidateLattice<W> {
        // WFST composition algorithm
        // ...
    }
}
```

---

## FuzzySource Trait

Integration point for PathMap and other backends. For complete implementation
details, see [Phase A: FuzzySource Implementation](../../integration/mork/fuzzy_source.md).

The FuzzySource trait enables MORK's pattern matching to use liblevenshtein's
fuzzy dictionaries as data sources:

```rust
/// Trait for fuzzy dictionary lookup
pub trait FuzzySource {
    /// Fuzzy lookup with maximum edit distance
    fn fuzzy_lookup(&self, query: &[u8], max_distance: u8)
        -> Box<dyn Iterator<Item = (Vec<u8>, u8)> + '_>;

    /// Prefix search for completion
    fn prefix_search(&self, prefix: &[u8])
        -> Box<dyn Iterator<Item = Vec<u8>> + '_>;
}

/// PathMap implements FuzzySource
impl FuzzySource for PathMap<Vec<u8>, ()> {
    fn fuzzy_lookup(&self, query: &[u8], max_distance: u8)
        -> Box<dyn Iterator<Item = (Vec<u8>, u8)> + '_>
    {
        let automaton = LevenshteinAutomaton::new(query, max_distance);

        Box::new(self.keys().filter_map(move |key| {
            automaton.matches(&key).map(|dist| (key.clone(), dist))
        }))
    }

    fn prefix_search(&self, prefix: &[u8])
        -> Box<dyn Iterator<Item = Vec<u8>> + '_>
    {
        Box::new(self.prefix_iter(prefix).map(|(k, _)| k.clone()))
    }
}
```

---

## Summary

Tier 1 provides:

1. **Edit Distance Automata**: Levenshtein, Damerau-Levenshtein, custom costs
2. **Phonetic Rules**: Metaphone, Soundex, Caverphone2
3. **Dictionary Backends**: DynamicDawg, DoubleArrayTrie, SuffixAutomaton
4. **WFST Semirings**: Tropical, Log, Probability, Boolean
5. **Candidate Lattice**: Structured output for downstream tiers
6. **FuzzySource Trait**: PathMap integration

The candidate lattice feeds into Tier 2 for syntactic validation.

---

## References

- liblevenshtein: `/home/dylon/Workspace/f1r3fly.io/liblevenshtein-rust/`
- See [03-tier2-syntactic-validation.md](./03-tier2-syntactic-validation.md) for next tier
- See [bibliography.md](../reference/bibliography.md) for complete references
