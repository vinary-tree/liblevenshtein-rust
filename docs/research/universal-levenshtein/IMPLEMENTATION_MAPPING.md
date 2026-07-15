[← Documentation Index](../../README.md)

# Universal Levenshtein Automata - Implementation Mapping

**Purpose**: Bridge theoretical definitions from Mitankin's thesis to Rust implementation in liblevenshtein-rust

**Audience**: Developers implementing Universal Levenshtein Automata in Rust

**Prerequisites**:
- Read [ALGORITHMS.md](./ALGORITHMS.md) for pseudocode
- Read [THEORETICAL_FOUNDATIONS.md](./THEORETICAL_FOUNDATIONS.md) for definitions
- Familiarity with liblevenshtein-rust codebase structure

---

## Table of Contents

1. [Overview](#overview)
2. [Type System Mapping](#type-system-mapping)
3. [Core Data Structures](#core-data-structures)
4. [Function Mapping](#function-mapping)
5. [Module Organization](#module-organization)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Testing Strategy](#testing-strategy)

---

## Overview

### Translation Philosophy

**Theoretical → Rust Mapping Principles**:

1. **Type Safety**: Use Rust's type system to enforce theoretical invariants
2. **Zero-Cost Abstractions**: Theoretical constructs should compile to efficient code
3. **Ownership**: Leverage Rust ownership to manage automaton lifecycle
4. **Generics**: Use generics for distance variants $`(\chi  \in`$ $`\{\varepsilon , t, \text{ms}\}`$)
5. **Const Generics**: Use const n for maximum edit distance at compile time where possible

### Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│  Universal Automaton Layer (New)        │
│  - UniversalAutomaton<V, const N: usize> │
│  - UniversalPosition<V>                  │
│  - DiagonalCrossing, Subsumption         │
└─────────────────────────────────────────┘
              ↓ compiles to
┌─────────────────────────────────────────┐
│  Deterministic Automaton Layer           │
│  - State (set of positions)              │
│  - ElementaryTransition (δ^D,χ_e)        │
│  - BitVectorEncoding (h_n, β)            │
└─────────────────────────────────────────┘
              ↓ simulates
┌─────────────────────────────────────────┐
│  Nondeterministic Conceptual Layer       │
│  - Position (i#e, i#e_t, i#e_s)         │
│  - Edit operations (ins, del, sub, etc.) │
└─────────────────────────────────────────┘
```

---

## Type System Mapping

### Theory → Rust Type Correspondence

| Theoretical Construct | Rust Type | File Location |
|----------------------|-----------|---------------|
| **Position i#e** | `Position<V>` | `src/transducer/universal/position.rs` (new) |
| **Universal I + i#e** | `UniversalPosition<V>::INonFinal(i, e)` | `src/transducer/universal/position.rs` (new) |
| **Universal M + i#e** | `UniversalPosition<V>::MFinal(i, e)` | `src/transducer/universal/position.rs` (new) |
| **State $`q \subseteq`$ positions** | `UniversalState<V>` | `src/transducer/universal/state.rs` (new) |
| **Variant $`\chi`$** |  `enum Variant { Standard, Transposition, MergeAndSplit }` | `src/transducer/algorithm.rs` (existing) |
| **Bit vector b** | `BitVec` or `Vec<bool>` | `src/transducer/universal/bitvector.rs` (new) |
| **Automaton $`A^{\forall ,\chi }_n`$** | `UniversalAutomaton<V, const N: usize>` | `src/transducer/universal/automaton.rs` (new) |

---

## Core Data Structures

### 1. Universal Position

**Theory** (Def. 15, Page 30):
```
I^ε_s = {I + t#k | |t| ≤ k ∧ -n ≤ t ≤ n ∧ 0 ≤ k ≤ n}
M^ε_s = {M + t#k | k ≥ -t - n ∧ -2n ≤ t ≤ 0 ∧ 0 ≤ k ≤ n}
```

**Rust Implementation**:

```rust
/// Universal position with parameter (I or M)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UniversalPosition<V: PositionVariant> {
    /// I-type (non-final): I + offset#errors
    /// Represents position relative to start of word
    INonFinal {
        offset: i32,  // t in I + t#k  (-n ≤ t ≤ n)
        errors: u8,   // k in I + t#k  (0 ≤ k ≤ n)
        variant_state: V::State,
    },

    /// M-type (final): M + offset#errors
    /// Represents position relative to end of word
    MFinal {
        offset: i32,  // t in M + t#k  (-2n ≤ t ≤ 0)
        errors: u8,   // k in M + t#k  (0 ≤ k ≤ n)
        variant_state: V::State,
    },
}

/// Position variant (usual, transposition, split)
pub trait PositionVariant: Clone + Debug + Eq + Hash {
    type State: Clone + Debug + Eq + Hash + Default;

    fn variant_name() -> &'static str;

    fn compute_i_successors(
        offset: i32,
        errors: u8,
        variant_state: &Self::State,
        bit_vector: &CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>>;

    fn compute_m_successors(
        offset: i32,
        errors: u8,
        variant_state: &Self::State,
        bit_vector: &CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>>;
}

/// Standard positions (usual type only)
#[derive(Clone, Debug)]
pub struct Standard;

impl PositionVariant for Standard {
    type State = ();

    fn variant_name() -> &'static str {
        "Standard"
    }

    fn compute_i_successors(
        offset: i32,
        errors: u8,
        _variant_state: &Self::State,
        bit_vector: &CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>> {
        UniversalPosition::<Self>::successors_i_type_standard(
            offset,
            errors,
            bit_vector,
            max_distance,
        )
    }

    fn compute_m_successors(
        offset: i32,
        errors: u8,
        _variant_state: &Self::State,
        bit_vector: &CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>> {
        UniversalPosition::<Self>::successors_m_type_standard(
            offset,
            errors,
            bit_vector,
            max_distance,
        )
    }
}

/// Transposition positions (usual + transposition state)
#[derive(Clone, Debug)]
pub enum Transposition {
    Usual,
    TranspositionState,
}

/// Merge/Split positions (usual + split state)
#[derive(Clone, Debug)]
pub enum MergeAndSplit {
    Usual,
    SplitState,
}
```

**Invariants to Enforce**:
```rust
impl<V: PositionVariant> UniversalPosition<V> {
    /// Constructor that enforces invariants from Def. 15
    pub fn new_i(offset: i32, errors: u8, n: u8) -> Result<Self, PositionError> {
        if offset.abs() as u8 > errors || offset.abs() as u8 > n || errors > n {
            return Err(PositionError::InvalidIPosition);
        }
        Ok(Self::INonFinal { offset, errors, variant: PhantomData })
    }

    pub fn new_m(offset: i32, errors: u8, n: u8) -> Result<Self, PositionError> {
        if errors < (-offset - n as i32) as u8
           || offset > 0 || offset < -(2 * n as i32)
           || errors > n {
            return Err(PositionError::InvalidMPosition);
        }
        Ok(Self::MFinal { offset, errors, variant: PhantomData })
    }
}
```

---

### 2. Universal State

**Theory** (Def. 15):
States are **anti-chains** under subsumption relation $`\le ^\chi _s.`$

**Rust Implementation**:

```rust
use std::collections::HashSet;

/// Universal automaton state (anti-chain of positions)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniversalState<V: PositionVariant> {
    /// Positions in this state (maintains anti-chain invariant)
    positions: HashSet<UniversalPosition<V>>,

    /// Maximum edit distance n (for validation)
    max_distance: u8,
}

impl<V: PositionVariant> UniversalState<V> {
    /// Create new state, maintaining subsumption closure ⊔
    pub fn new(positions: impl IntoIterator<Item = UniversalPosition<V>>, n: u8) -> Self {
        let mut state = Self {
            positions: HashSet::new(),
            max_distance: n,
        };

        for pos in positions {
            state.add_position_with_subsumption(pos);
        }

        state
    }

    /// Add position, maintaining anti-chain property (⊔ operator)
    fn add_position_with_subsumption(&mut self, pos: UniversalPosition<V>) {
        // Remove any positions subsumed by new position
        self.positions.retain(|p| !subsumes(&pos, p, self.max_distance));

        // Add new position if not subsumed by existing positions
        if !self.positions.iter().any(|p| subsumes(p, &pos, self.max_distance)) {
            self.positions.insert(pos);
        }
    }

    /// Check if state is final (contains position subsuming M#n)
    pub fn is_final(&self) -> bool {
        self.positions.iter().any(|pos| match pos {
            UniversalPosition::MFinal { offset, errors, .. } => {
                *offset == 0 && *errors <= self.max_distance
            }
            _ => false,
        })
    }
}
```

---

### 3. Subsumption Relation

**Theory** (Def. 11, Page 18):

```
For ε (standard):
  i#e ≤^ε_s j#f  ⇔  f > e ∧ |j - i| ≤ f - e
```

**Rust Implementation**:

```rust
/// Check if pos1 <^χ_s pos2 (strict subsumption)
pub fn subsumes<V: PositionVariant>(
    pos1: &UniversalPosition<V>,
    pos2: &UniversalPosition<V>,
    n: u8,
) -> bool {
    use UniversalPosition::*;

    match (pos1, pos2) {
        // I-type subsumption
        (INonFinal { offset: i, errors: e, .. },
         INonFinal { offset: j, errors: f, .. }) => {
            f > e && (j - i).abs() <= (*f - *e) as i32
        }

        // M-type subsumption
        (MFinal { offset: i, errors: e, .. },
         MFinal { offset: j, errors: f, .. }) => {
            f > e && (j - i).abs() <= (*f - *e) as i32
        }

        // Different parameter types don't subsume
        _ => false,
    }
}
```

**For transposition** (extend with transposition state logic):
```rust
// Extends PositionVariant trait
impl PositionVariant for Transposition {
    // Additional subsumption rules for It + t#k positions
    // See Def. 11 for full rules
}
```

---

### 4. Bit Vector Encoding

**Theory** (Def. 7, Page 17; Def. 16, Page 40):

```
β(x, w) = b_1b_2...b_{|w|}  where b_i = 1 iff w_i = x
h_n(w, x) = β(x_1, s_n(w, 1))...β(x_{|x|}, s_n(w, |x|))
```

**Rust Implementation**:

```rust
/// Characteristic vector β(x, w)
pub struct CharacteristicVector {
    bits: BitVec,  // or Vec<bool> for simplicity
}

impl CharacteristicVector {
    /// Compute β(x, w) - characteristic vector for character x in word w
    pub fn new(ch: char, word: &str) -> Self {
        let bits = word.chars().map(|c| c == ch).collect();
        Self { bits }
    }

    /// Get bit at position i
    pub fn get(&self, i: usize) -> bool {
        self.bits.get(i).copied().unwrap_or(false)
    }

    /// Length of bit vector
    pub fn len(&self) -> usize {
        self.bits.len()
    }
}

/// Bit vector sequence h_n(w, x)
pub struct BitVectorSequence {
    vectors: Vec<CharacteristicVector>,
}

impl BitVectorSequence {
    /// Compute h_n(w, x) for word w and input x with max distance n
    pub fn encode(word: &str, input: &str, n: usize) -> Self {
        let vectors: Vec<_> = input
            .char_indices()
            .map(|(i, ch)| {
                let window = sliding_window(word, i, n);
                CharacteristicVector::new(ch, window)
            })
            .collect();

        Self { vectors }
    }
}

/// Extract sliding window s_n(w, i) from Def. 16
fn sliding_window(word: &str, position: usize, n: usize) -> &str {
    let start = position.saturating_sub(n);
    let end = (position + n + 1).min(word.len());
    &word[start..end]
}
```

---

### 5. Elementary Transition

**Theory** (Def. 7, Page 14-16; ALGORITHMS.md pg. 55-57):

```
δ^{D,ε}_e: (ℤ × ℕ) × {0,1}^+ → P(ℤ × ℕ)
```

**Rust Implementation**:

```rust
/// Elementary transition function δ^{D,χ}_e
pub struct ElementaryTransition<V: PositionVariant> {
    max_distance: u8,
    _variant: PhantomData<V>,
}

impl ElementaryTransition<Standard> {
    /// Compute δ^{D,ε}_e(position, bit_vector)
    pub fn apply(
        &self,
        pos: (i32, u8),  // (offset, errors)
        h: &CharacteristicVector,
    ) -> HashSet<(i32, u8)> {
        let (x, y) = pos;
        let mut result = HashSet::new();

        if h.len() == 0 {
            // Empty bit vector: only substitution
            if y < self.max_distance {
                result.insert((x, y + 1));
            }
            return result;
        }

        if h.get(0) {
            // First bit is 1: match
            result.insert((x + 1, y));
            return result;
        }

        // First bit is 0: mismatch
        if h.len() == 1 {
            if y < self.max_distance {
                result.insert((x, y + 1));      // substitution
                result.insert((x + 1, y + 1));  // insertion
            }
            return result;
        }

        // h has length ≥ 2, find first match
        let first_match = (1..h.len()).find(|&i| h.get(i));

        if let Some(j) = first_match {
            // Match found at position j
            if y < self.max_distance {
                result.insert((x, y + 1));          // substitution
                result.insert((x + 1, y + 1));      // insertion
                result.insert((x + j as i32, y + j as u8 - 1));  // deletions + match
            }
        } else {
            // No match found
            if y < self.max_distance {
                result.insert((x, y + 1));
                result.insert((x + 1, y + 1));
            }
        }

        result
    }
}
```

---

### 6. Diagonal Crossing

**Theory** (Def. 17, Page 42):

```
f_n(I + t#e, k) = (k ≤ 2n+1) ∧ (e ≤ t + 2n + 1 - k)
m_n(I + t#e, k) = M + (t + n + 1 - k)#e
```

**Rust Implementation**:

```rust
/// Diagonal crossing functions f_n and m_n
pub struct DiagonalCrossing {
    max_distance: u8,
}

impl DiagonalCrossing {
    /// Check if position should cross diagonal: f_n(pos, k)
    pub fn should_cross<V: PositionVariant>(
        &self,
        pos: &UniversalPosition<V>,
        bit_vector_len: usize,
    ) -> bool {
        let n = self.max_distance as i32;

        match pos {
            UniversalPosition::INonFinal { offset: t, errors: e, .. } => {
                let k = bit_vector_len as i32;
                k <= 2 * n + 1 && (*e as i32) <= t + 2 * n + 1 - k
            }
            UniversalPosition::MFinal { offset: t, errors: e, .. } => {
                (*e as i32) > t + n
            }
        }
    }

    /// Convert position across diagonal: m_n(pos, k)
    pub fn convert<V: PositionVariant>(
        &self,
        pos: UniversalPosition<V>,
        bit_vector_len: usize,
    ) -> UniversalPosition<V> {
        let k = bit_vector_len as i32;
        let n = self.max_distance as i32;

        match pos {
            UniversalPosition::INonFinal { offset: t, errors: e, .. } => {
                UniversalPosition::MFinal {
                    offset: t + n + 1 - k,
                    errors: e,
                    variant: PhantomData,
                }
            }
            UniversalPosition::MFinal { offset: t, errors: e, .. } => {
                UniversalPosition::INonFinal {
                    offset: t - n - 1 + k,
                    errors: e,
                    variant: PhantomData,
                }
            }
        }
    }
}
```

---

## Function Mapping

### Theoretical → Rust Function Table

| Theory Function | Pseudocode (ALGORITHMS.md) | Rust Implementation | Module |
|-----------------|----------------------------|---------------------|--------|
| $`\delta ^{\forall ,\chi }_n(q, b)`$ | `Delta(n, st, b)` | `UniversalState::transition(&self, b: &BitVec) -> Self` | `state.rs` |
| $`\delta ^{\forall ,\chi }_e(q, b)`$ | `Delta_E(n, q, b)` | `UniversalPosition::elementary_transition(&self, b: &BitVec) -> HashSet<Self>` | `position.rs` |
| $`\delta ^{D,\chi }_e(\pi , h)`$ | `Delta_E_D(n, pt, h)` | `ElementaryTransition::apply(&self, pos, h) -> HashSet<Pos>` | `transition.rs` |
| $`\le ^\chi _s`$ | `Less_Than_Subsume(q1, q2)` | `subsumes(pos1, pos2, n) -> bool` | `subsumption.rs` |
| $`\sqcup A`$ | Implicit in `Delta` | `UniversalState::add_position_with_subsumption(&mut self, pos)` | `state.rs` |
| f_n(q, k) | `F(n, pos, k)` | `DiagonalCrossing::should_cross(&self, pos, k) -> bool` | `diagonal.rs` |
| m_n(q, k) | `M(n, st, k)` | `DiagonalCrossing::convert(&self, pos, k) -> Pos` | `diagonal.rs` |
| r_n(q, b) | `R(n, pos, b)` | `extract_substring(&self, pos, b) -> &BitVec` | `bitvector.rs` |
| rm(A) | `RM(st)` | `UniversalState::rightmost(&self) -> Option<&Pos>` | `state.rs` |
| h_n(w, x) | N/A (computed) | `BitVectorSequence::encode(word, input, n) -> Self` | `bitvector.rs` |
| $`\beta (x, w)`$ | N/A (computed) | `CharacteristicVector::new(ch, word) -> Self` | `bitvector.rs` |

---

## Module Organization

### Proposed Directory Structure

```
src/transducer/universal/
├── mod.rs                    // Public API, re-exports
├── automaton.rs              // UniversalAutomaton<V, const N: usize>
├── position.rs               // UniversalPosition<V>, PositionVariant
├── state.rs                  // UniversalState<V>
├── transition.rs             // ElementaryTransition, transition logic
├── subsumption.rs            // Subsumption checking (≤^χ_s, ⊔)
├── diagonal.rs               // DiagonalCrossing (f_n, m_n)
├── bitvector.rs              // BitVectorSequence, CharacteristicVector
├── builder.rs                // Construction algorithm (Build_Automaton)
└── variants/
    ├── standard.rs           // Standard variant (ε)
    ├── transposition.rs      // Transposition variant (t)
    └── merge_split.rs        // MergeAndSplit variant (ms)
```

### Integration with Existing Code

**Existing Modules**:
- `src/transducer/algorithm.rs` - Add `Universal` variant to Algorithm enum
- `src/transducer/builder.rs` - Extend with `.universal(n)` method
- `src/transducer/transducer.rs` - Support UniversalAutomaton backend

**Example Integration**:

```rust
// In src/transducer/algorithm.rs
pub enum Algorithm {
    Standard,
    Transposition,
    MergeAndSplit,
    Universal { max_distance: u8 },  // NEW
}

// In src/transducer/builder.rs
impl TransducerBuilder {
    pub fn universal(mut self, n: u8) -> Self {
        self.algorithm = Algorithm::Universal { max_distance: n };
        self
    }
}
```

---

## Implementation Roadmap

### Phase 1: Core Types (Week 1)

**Goal**: Implement basic type system without full automation logic.

**Tasks**:
1. ✅ Create module structure (`src/transducer/universal/`)
2. ✅ Implement `UniversalPosition<V>` with invariant checks
3. ✅ Implement `PositionVariant` trait + Standard variant
4. ✅ Implement `UniversalState<V>` with anti-chain property
5. ✅ Implement subsumption checking (`subsumes`)
6. ✅ Write unit tests for position creation, subsumption

**Deliverables**:
- `position.rs` (~200 lines)
- `state.rs` (~150 lines)
- `subsumption.rs` (~100 lines)
- Tests (~200 lines)

---

### Phase 2: Bit Vector Encoding (Week 2)

**Goal**: Implement bit vector computation h_n(w, x).

**Tasks**:
1. ✅ Implement `CharacteristicVector` $`(\beta`$ function)
2. ✅ Implement `BitVectorSequence` (h_n function)
3. ✅ Implement sliding window extraction (s_n function)
4. ✅ Implement substring extraction (r_n function)
5. ✅ Optimize with SIMD (optional)
6. ✅ Write unit tests for all bit vector operations

**Deliverables**:
- `bitvector.rs` (~300 lines)
- Tests (~150 lines)

---

### Phase 3: Transitions (Week 3)

**Goal**: Implement transition functions $`\delta ^{D,\chi }_e`$, $`\delta ^{\forall ,\chi }_e`$, $`\delta ^{\forall ,\chi }_n`$.

**Tasks**:
1. ✅ Implement `ElementaryTransition` for Standard variant
2. ✅ Implement `UniversalPosition::elementary_transition`
3. ✅ Implement `UniversalState::transition` (full $`\delta ^{\forall ,\chi }_n`$)
4. ✅ Implement `DiagonalCrossing` (f_n, m_n functions)
5. ✅ Write unit tests for all transitions
6. ✅ Property-based tests (QuickCheck)

**Deliverables**:
- `transition.rs` (~400 lines)
- `diagonal.rs` (~100 lines)
- Tests (~300 lines)

---

### Phase 4: Automaton Construction (Week 4)

**Goal**: Implement BFS construction algorithm (Build_Automaton).

**Tasks**:
1. ✅ Implement `UniversalAutomaton<V, const N: usize>`
2. ✅ Implement `Builder::build()` using BFS
3. ✅ Implement state deduplication (HAS_NEVER_BEEN_PUSHED)
4. ✅ Implement query interface (given word w, traverse automaton)
5. ✅ Serialize/deserialize automaton
6. ✅ Write integration tests

**Deliverables**:
- `automaton.rs` (~300 lines)
- `builder.rs` (~250 lines)
- Tests (~200 lines)

---

### Phase 5: Variants (Week 5)

**Goal**: Extend to transposition and merge/split variants.

**Tasks**:
1. ✅ Implement `Transposition` variant
2. ✅ Implement `MergeAndSplit` variant
3. ✅ Extend subsumption logic for variants
4. ✅ Extend transition logic for variants
5. ✅ Write variant-specific tests

**Deliverables**:
- `variants/transposition.rs` (~200 lines)
- `variants/merge_split.rs` (~200 lines)
- Tests (~250 lines)

---

### Phase 6: Optimization & Integration (Week 6)

**Goal**: Optimize and integrate with existing liblevenshtein-rust.

**Tasks**:
1. ✅ Profile and optimize hot paths
2. ✅ SIMD optimizations for bit vectors
3. ✅ Integrate with `TransducerBuilder`
4. ✅ Add benchmarks (compare with existing Levenshtein impl)
5. ✅ Documentation and examples
6. ✅ CI/CD integration

**Deliverables**:
- Benchmarks
- Documentation
- Examples

---

## Testing Strategy

### Unit Tests

**Per Module**:
1. **Position Tests**:
   - Valid position creation
   - Invalid position rejection
   - Subsumption checks

2. **State Tests**:
   - Anti-chain property maintenance
   - State equality
   - Final state detection

3. **Bit Vector Tests**:
   - Characteristic vector correctness
   - Sliding window extraction
   - Sequence encoding

4. **Transition Tests**:
   - Elementary transition correctness
   - Diagonal crossing logic
   - Subsumption closure

### Property-Based Tests (QuickCheck)

```rust
#[quickcheck]
fn subsumption_is_transitive(pos1: UniversalPosition<Standard>,
                               pos2: UniversalPosition<Standard>,
                               pos3: UniversalPosition<Standard>) -> bool {
    if subsumes(&pos1, &pos2, 3) && subsumes(&pos2, &pos3, 3) {
        subsumes(&pos1, &pos3, 3)
    } else {
        true  // Property only holds when both premises true
    }
}

#[quickcheck]
fn diagonal_crossing_is_invertible(pos: UniversalPosition<Standard>, k: usize) -> bool {
    let diagonal = DiagonalCrossing { max_distance: 3 };
    let converted = diagonal.convert(pos.clone(), k);
    let reverted = diagonal.convert(converted, k);
    pos == reverted
}
```

### Integration Tests

**Against Reference DFA**:
```rust
#[test]
fn universal_matches_dfa_for_concrete_word() {
    let word = "test";
    let n = 2;

    // Build universal automaton
    let universal = UniversalAutomaton::<Standard, 2>::build();

    // Build reference DFA for specific word
    let dfa = build_reference_dfa(word, n);

    // Test equivalence on many inputs
    for input in test_inputs() {
        let universal_accepts = universal.accepts(word, input);
        let dfa_accepts = dfa.accepts(input);
        assert_eq!(universal_accepts, dfa_accepts);
    }
}
```

### Correctness Verification

**Proposition 19 Verification**:
```rust
#[test]
fn verify_proposition_19() {
    // For all words w and inputs x:
    // Universal automaton accepts x ⇔ d_L(x, w) ≤ n

    for word in test_words() {
        for input in test_inputs() {
            let universal_accepts = universal.accepts(word, input);
            let distance = levenshtein_distance(input, word);
            assert_eq!(universal_accepts, distance <= N);
        }
    }
}
```

---

## Performance Considerations

### Hot Paths to Optimize

1. **Subsumption Checking** (called $`\mathcal{O}(\text{states}^{2})`$ times per transition)
   - Cache subsumption results
   - Use SIMD for batch checking

2. **Bit Vector Encoding** (computed for every query)
   - Precompute characteristic vectors for common characters
   - Use bit-packed representations

3. **State Storage** (HashSet operations)
   - Consider perfect hashing for states
   - Pool allocator for positions

### Memory Layout

```rust
// Optimize position layout for cache efficiency
#[repr(C)]
pub struct UniversalPosition<V> {
    offset: i16,     // 2 bytes (sufficient for reasonable n)
    errors: u8,      // 1 byte
    param: u8,       // 1 byte (0=I, 1=M)
    variant: u8,     // 1 byte (0=usual, 1=t, 2=s)
    _padding: [u8; 3], // 3 bytes padding to align to 8 bytes
}
// Total: 8 bytes (fits in single cache line with other fields)
```

---

## Summary

This implementation mapping provides:

✅ **Type-safe** representation of theoretical constructs
✅ **Efficient** data structures leveraging Rust ownership
✅ **Testable** architecture with clear module boundaries
✅ **Extensible** design supporting all three variants
✅ **Documented** correspondence between theory and practice

**Next Steps**:
1. Create module skeleton
2. Implement Phase 1 (core types)
3. Add unit tests
4. Iterate through phases

**References**:
- [ALGORITHMS.md](./ALGORITHMS.md) - Pseudocode reference
- [THEORETICAL_FOUNDATIONS.md](./THEORETICAL_FOUNDATIONS.md) - Mathematical definitions
- [PAPER_SUMMARY.md](./PAPER_SUMMARY.md) - Complete thesis analysis

---

**Document Status**: ✅ Complete implementation mapping
**Last Updated**: 2025-11-11
**Ready For**: Implementation Phase 1
