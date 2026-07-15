# Phase C: WFST Composition Guide

**Last Updated**: 2025-12-21
**Version**: 0.9.1
**Status**: PROPOSED

> ⚠️ **PROPOSAL NOTICE**: This document describes a **proposed** `src/wfst/` module for full WFST support. The proposed structure is a design specification for future implementation.
>
> **Current Implementation**: liblevenshtein 0.9.1 provides `ProductAutomatonChar` (NFA × Levenshtein composition) in `src/phonetic/nfa/`. See [Current Capabilities](#current-capabilities) for what's available today.

This document provides detailed implementation guidance for Phase C of the MORK integration: full Weighted Finite State Transducer (WFST) implementation with phonetic NFA composition.

## Overview

**Goal**: Implement complete WFST infrastructure including weighted transitions, phonetic NFA compilation via Thompson's construction, and FST composition operators.

**Result**: Queries can compose phonetic patterns with Levenshtein automata, enabling sound-alike matching with configurable costs.

---

## Current Capabilities

Before implementing the proposed WFST module, liblevenshtein 0.9.1 provides these building blocks:

### What's Available Now

| Feature | Location | Description |
|---------|----------|-------------|
| `ProductAutomatonChar` | `src/phonetic/nfa/product.rs` | NFA × Levenshtein composition |
| `NFAChar` | `src/phonetic/nfa/nfa.rs` | Character-level NFA implementation |
| `ThompsonBuilderChar` | `src/phonetic/nfa/thompson.rs` | Thompson's construction for NFA |
| `llre!` macro | `liblevenshtein-macros` | Compile-time regex → NFA |
| `llev!` macro | `liblevenshtein-macros` | Compile-time phonetic rules |
| `english::base()` | `src/phonetic/rules/english.rs` | 62 pre-compiled orthographic rules |
| `english::homophones()` | `src/phonetic/rules/english.rs` | Homophone pairs |
| `english::text_speak()` | `src/phonetic/rules/english.rs` | Text-speak expansions |
| `RuleSetChar` | `src/phonetic/rules/mod.rs` | Combine multiple rule sets |
| `rules_to_nfa_char()` | `src/phonetic/verified/mod.rs` | Convert rules to NFA |

### Current API Examples

```rust
use liblevenshtein::phonetic::nfa::ProductAutomatonChar;
use liblevenshtein::phonetic::verified::rules_to_nfa_char;
use liblevenshtein::phonetic::rules::english;
use liblevenshtein::{llre, llev};

// Compile-time pattern embedding
let phone_pattern = llre!(r"(ph|f)one");

// Use pre-compiled English rules
let rules = english::base();
let nfa = rules_to_nfa_char(&rules.rules);
let product = ProductAutomatonChar::new(nfa, 2);

// Check acceptance (phonetic + edit distance)
assert!(product.accepts("phone"));
assert!(product.accepts("fone"));
```

### Current Module Structure

```
src/phonetic/           # CURRENT - Available in 0.9.1
├── nfa/
│   ├── mod.rs
│   ├── nfa.rs          # NFAChar implementation
│   ├── thompson.rs     # Thompson's construction
│   ├── product.rs      # ProductAutomatonChar (NFA × Levenshtein)
│   ├── compiler.rs     # Pattern compilation
│   └── types.rs        # StateId, Transition, CharClass
├── llev/               # LLEV rule parsing
├── llre/               # LLRE pattern compilation
├── rules/              # english::base(), homophones(), text_speak()
└── verified/           # rules_to_nfa_char()
```

---

## Proposed WFST Module

> **Status**: PROPOSED - Not yet implemented

The proposed `src/wfst/` module extends the current capabilities with:
- Arbitrary semiring weights (not just integer edit distance)
- General WFST × WFST composition (not just NFA × Levenshtein)
- Configurable cost functions
- Lattice output structures

---

## Architecture

### WFST Pipeline

```
Query Pattern: "(ph|f)(o|oa)(n|ne)"
    |
    v
PhoneticNfa::compile()
    |
    | Thompson's construction
    v
PhoneticNfa
    |
    v
ComposedAutomaton::new(phonetic_nfa, levenshtein, dictionary)
    |
    | FST ∘ FST ∘ Trie composition
    v
Lattice with phonetic-weighted edges
    |
    v
MORK query pipeline
```

### Component Locations (PROPOSED)

> **Note**: This is the **proposed** module structure. It does not currently exist.

```
liblevenshtein-rust/src/
├── wfst/                           # PROPOSED - To be implemented
│   ├── mod.rs                      # Module root and exports
│   ├── weight.rs                   # Semiring types (Tropical, Log)
│   ├── semiring.rs                 # Semiring trait definition
│   ├── transition.rs               # Weighted transition type
│   ├── nfa.rs                      # WeightedNFA (extends PhoneticNfa)
│   ├── composition.rs              # General WFST × WFST composition
│   └── phonetic_integration.rs     # Integration with verified rules
```

**Relationship to current implementation**:
- `wfst/nfa.rs` would extend `phonetic/nfa/nfa.rs` with arbitrary weights
- `wfst/composition.rs` would generalize `phonetic/nfa/product.rs`
- Thompson's construction already exists in `phonetic/nfa/thompson.rs`

---

## Core Concepts

### Semirings

WFSTs operate over algebraic semirings that define how weights combine:

```rust
/// Semiring trait for WFST weights.
///
/// A semiring (K, ⊕, ⊗, 0, 1) satisfies:
/// - (K, ⊕, 0) is a commutative monoid
/// - (K, ⊗, 1) is a monoid
/// - ⊗ distributes over ⊕
/// - 0 ⊗ a = a ⊗ 0 = 0
pub trait Semiring: Clone + Copy + PartialEq + Sized {
    /// Additive identity (worst path)
    const ZERO: Self;

    /// Multiplicative identity (no cost)
    const ONE: Self;

    /// Addition: combines parallel paths (e.g., min for shortest path)
    fn plus(self, other: Self) -> Self;

    /// Multiplication: combines sequential costs (e.g., + for path length)
    fn times(self, other: Self) -> Self;
}
```

### Tropical Semiring

The **tropical semiring** $`(\mathbb{R} \cup \{\infty\}, \min, +, \infty, 0)`$ is used for shortest-path problems:

```rust
/// Tropical semiring weight (min, +).
///
/// Used for finding minimum-cost paths:
/// - ⊕ = min (choose best parallel path)
/// - ⊗ = + (sum costs along path)
/// - 0 = ∞ (no path)
/// - 1 = 0 (zero cost)
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct TropicalWeight(pub f32);

impl TropicalWeight {
    pub const INFINITY: Self = Self(f32::INFINITY);

    pub fn new(value: f32) -> Self {
        Self(value)
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Semiring for TropicalWeight {
    const ZERO: Self = Self(f32::INFINITY);
    const ONE: Self = Self(0.0);

    fn plus(self, other: Self) -> Self {
        TropicalWeight(self.0.min(other.0))
    }

    fn times(self, other: Self) -> Self {
        TropicalWeight(self.0 + other.0)
    }
}
```

---

## Weighted Transitions

```rust
/// Weighted transition in a WFST.
///
/// Represents an arc from source to target with input/output labels and weight.
#[derive(Clone, Debug)]
pub struct WeightedTransition<W: Semiring = TropicalWeight> {
    /// Source state ID
    pub source: usize,

    /// Target state ID
    pub target: usize,

    /// Input label (None = epsilon)
    pub input: Option<u8>,

    /// Output label (None = epsilon)
    pub output: Option<u8>,

    /// Transition weight
    pub weight: W,
}

impl<W: Semiring> WeightedTransition<W> {
    /// Create a new weighted transition.
    pub fn new(
        source: usize,
        target: usize,
        input: Option<u8>,
        output: Option<u8>,
        weight: W,
    ) -> Self {
        Self {
            source,
            target,
            input,
            output,
            weight,
        }
    }

    /// Create an epsilon transition (no input/output).
    pub fn epsilon(source: usize, target: usize, weight: W) -> Self {
        Self::new(source, target, None, None, weight)
    }

    /// Create an identity transition (same input/output).
    pub fn identity(source: usize, target: usize, label: u8, weight: W) -> Self {
        Self::new(source, target, Some(label), Some(label), weight)
    }
}
```

---

## Cost Functions

```rust
/// Configurable cost function for edit operations.
pub trait CostFunction: Clone + Send + Sync {
    /// Cost of inserting a character.
    fn insertion_cost(&self, char: u8) -> f32;

    /// Cost of deleting a character.
    fn deletion_cost(&self, char: u8) -> f32;

    /// Cost of substituting one character for another.
    fn substitution_cost(&self, from: u8, to: u8) -> f32;

    /// Cost of transposing adjacent characters.
    fn transposition_cost(&self, c1: u8, c2: u8) -> f32;
}

/// Uniform cost function (standard Levenshtein).
#[derive(Clone, Default)]
pub struct UniformCost;

impl CostFunction for UniformCost {
    fn insertion_cost(&self, _: u8) -> f32 { 1.0 }
    fn deletion_cost(&self, _: u8) -> f32 { 1.0 }
    fn substitution_cost(&self, _: u8, _: u8) -> f32 { 1.0 }
    fn transposition_cost(&self, _: u8, _: u8) -> f32 { 1.0 }
}

/// Phonetic-aware cost function.
///
/// Assigns lower costs to phonetically similar character substitutions.
#[derive(Clone)]
pub struct PhoneticCost {
    /// Set of phonetically equivalent character pairs
    phonetic_pairs: HashSet<(u8, u8)>,

    /// Cost for phonetic substitutions (typically 0.1-0.5)
    phonetic_weight: f32,
}

impl PhoneticCost {
    /// Create a new phonetic cost function.
    pub fn new(phonetic_weight: f32) -> Self {
        let mut pairs = HashSet::new();

        // Add phonetic equivalences from verified rules
        // See: src/transducer/phonetic.rs
        pairs.insert((b'f', b'p'));   // ph ↔ f (as in "phone")
        pairs.insert((b'c', b'k'));   // c ↔ k (as in "cat")
        pairs.insert((b'c', b's'));   // c ↔ s (before e, i)
        pairs.insert((b'g', b'j'));   // g ↔ j (before e, i)
        pairs.insert((b's', b'z'));   // s ↔ z (voicing)
        pairs.insert((b't', b'd'));   // t ↔ d (voicing)
        pairs.insert((b'p', b'b'));   // p ↔ b (voicing)
        pairs.insert((b'k', b'g'));   // k ↔ g (voicing)
        pairs.insert((b'i', b'y'));   // i ↔ y (vowel/glide)
        pairs.insert((b'u', b'w'));   // u ↔ w (vowel/glide)

        Self {
            phonetic_pairs: pairs,
            phonetic_weight,
        }
    }

    /// Check if two characters are phonetically equivalent.
    pub fn is_phonetic_pair(&self, a: u8, b: u8) -> bool {
        self.phonetic_pairs.contains(&(a, b)) ||
        self.phonetic_pairs.contains(&(b, a))
    }
}

impl CostFunction for PhoneticCost {
    fn insertion_cost(&self, _: u8) -> f32 { 1.0 }
    fn deletion_cost(&self, _: u8) -> f32 { 1.0 }

    fn substitution_cost(&self, from: u8, to: u8) -> f32 {
        if self.is_phonetic_pair(from, to) {
            self.phonetic_weight
        } else {
            1.0
        }
    }

    fn transposition_cost(&self, _: u8, _: u8) -> f32 { 1.0 }
}
```

---

## Phonetic NFA

### Thompson's Construction

**File**: `liblevenshtein-rust/src/wfst/nfa.rs`

```rust
//! Phonetic NFA construction via Thompson's algorithm.

use super::*;
use std::collections::HashSet;

/// State in the phonetic NFA.
#[derive(Clone, Debug)]
pub struct NfaState {
    /// State ID
    pub id: usize,

    /// Whether this is a final/accepting state
    pub is_final: bool,

    /// Labeled transitions (byte → target states)
    pub transitions: Vec<NfaTransition>,

    /// Epsilon transitions (no label)
    pub epsilon_transitions: Vec<usize>,
}

/// Labeled transition in the NFA.
#[derive(Clone, Debug)]
pub struct NfaTransition {
    /// Input byte
    pub input: u8,

    /// Target state ID
    pub target: usize,

    /// Weight for this transition
    pub weight: TropicalWeight,
}

/// Phonetic NFA for pattern matching.
///
/// Represents a non-deterministic finite automaton that accepts
/// phonetic variants of a pattern.
///
/// # Example
///
/// ```rust
/// let nfa = PhoneticNfa::compile("(ph|f)(o|oa)(n|ne)")?;
/// // Accepts: "phone", "fone", "phoan", "foane", "phonne", etc.
///
/// for accepted in nfa.accepted_strings(6) {
///     println!("{}", accepted);
/// }
/// ```
pub struct PhoneticNfa {
    /// All states in the NFA
    pub states: Vec<NfaState>,

    /// Start state ID
    pub start: usize,

    /// Final state IDs
    pub finals: Vec<usize>,
}

impl PhoneticNfa {
    /// Compile a phonetic regex pattern into an NFA.
    ///
    /// # Pattern Syntax
    ///
    /// - `(a|b)` - Alternation: matches "a" OR "b"
    /// - `ab` - Concatenation: matches "a" followed by "b"
    /// - Parentheses for grouping
    ///
    /// # Examples
    ///
    /// - `"(ph|f)one"` - Matches "phone" or "fone"
    /// - `"col(o|ou)r"` - Matches "color" or "colour"
    /// - `"(c|k)at"` - Matches "cat" or "kat"
    pub fn compile(pattern: &str) -> Result<Self, NfaError> {
        let mut builder = NfaBuilder::new();
        builder.parse_pattern(pattern)?;
        Ok(builder.build())
    }

    /// Create an NFA from liblevenshtein's verified phonetic rules.
    pub fn from_phonetic_rules(rules: &[PhoneticRule]) -> Self {
        let mut builder = NfaBuilder::new();
        for rule in rules {
            builder.add_rule(rule);
        }
        builder.build()
    }

    /// Create an epsilon NFA (accepts empty string only).
    pub fn epsilon() -> Self {
        let start_state = NfaState {
            id: 0,
            is_final: true,
            transitions: Vec::new(),
            epsilon_transitions: Vec::new(),
        };

        Self {
            states: vec![start_state],
            start: 0,
            finals: vec![0],
        }
    }

    /// Check if the NFA accepts a string.
    pub fn accepts(&self, input: &str) -> bool {
        let mut current_states = self.epsilon_closure(&[self.start]);

        for byte in input.bytes() {
            let mut next_states = HashSet::new();

            for &state_id in &current_states {
                let state = &self.states[state_id];
                for trans in &state.transitions {
                    if trans.input == byte {
                        next_states.insert(trans.target);
                    }
                }
            }

            current_states = self.epsilon_closure(&next_states.into_iter().collect::<Vec<_>>());

            if current_states.is_empty() {
                return false;
            }
        }

        current_states.iter().any(|&s| self.finals.contains(&s))
    }

    /// Compute epsilon closure of a set of states.
    fn epsilon_closure(&self, states: &[usize]) -> HashSet<usize> {
        let mut closure = HashSet::new();
        let mut stack: Vec<usize> = states.to_vec();

        while let Some(state_id) = stack.pop() {
            if closure.insert(state_id) {
                let state = &self.states[state_id];
                for &eps_target in &state.epsilon_transitions {
                    stack.push(eps_target);
                }
            }
        }

        closure
    }

    /// Generate all strings accepted by the NFA up to a maximum length.
    ///
    /// **Warning**: May produce exponentially many strings.
    pub fn accepted_strings(&self, max_length: usize) -> Vec<String> {
        let mut results = Vec::new();
        let mut stack = vec![(self.start, String::new())];

        while let Some((state_id, current)) = stack.pop() {
            if current.len() > max_length {
                continue;
            }

            // Check for epsilon transitions
            for &eps_target in &self.states[state_id].epsilon_transitions {
                stack.push((eps_target, current.clone()));
            }

            // If final state, add current string
            if self.finals.contains(&state_id) {
                results.push(current.clone());
            }

            // Explore labeled transitions
            if current.len() < max_length {
                for trans in &self.states[state_id].transitions {
                    let mut next = current.clone();
                    next.push(trans.input as char);
                    stack.push((trans.target, next));
                }
            }
        }

        results.sort();
        results.dedup();
        results
    }

    /// Compose this NFA with a Levenshtein automaton.
    pub fn compose_with_levenshtein<'a>(
        &'a self,
        levenshtein: &'a impl LevenshteinAutomaton,
    ) -> ComposedAutomaton<'a> {
        ComposedAutomaton::new(self, levenshtein)
    }
}

/// Builder for NFA construction via Thompson's algorithm.
struct NfaBuilder {
    states: Vec<NfaState>,
    next_id: usize,
}

impl NfaBuilder {
    fn new() -> Self {
        Self {
            states: Vec::new(),
            next_id: 0,
        }
    }

    fn new_state(&mut self, is_final: bool) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.states.push(NfaState {
            id,
            is_final,
            transitions: Vec::new(),
            epsilon_transitions: Vec::new(),
        });
        id
    }

    fn add_transition(&mut self, from: usize, to: usize, input: u8) {
        self.states[from].transitions.push(NfaTransition {
            input,
            target: to,
            weight: TropicalWeight::ONE,
        });
    }

    fn add_epsilon(&mut self, from: usize, to: usize) {
        self.states[from].epsilon_transitions.push(to);
    }

    /// Parse pattern using recursive descent.
    fn parse_pattern(&mut self, pattern: &str) -> Result<(usize, usize), NfaError> {
        let chars: Vec<char> = pattern.chars().collect();
        let (start, end, _) = self.parse_alternation(&chars, 0)?;
        self.states[end].is_final = true;
        Ok((start, end))
    }

    /// Parse alternation: expr ('|' expr)*
    fn parse_alternation(
        &mut self,
        chars: &[char],
        pos: usize,
    ) -> Result<(usize, usize, usize), NfaError> {
        let (first_start, first_end, mut pos) = self.parse_concatenation(chars, pos)?;

        if pos < chars.len() && chars[pos] == '|' {
            // Create new start and end for alternation
            let alt_start = self.new_state(false);
            let alt_end = self.new_state(false);

            self.add_epsilon(alt_start, first_start);
            self.add_epsilon(first_end, alt_end);

            while pos < chars.len() && chars[pos] == '|' {
                pos += 1;  // Skip '|'
                let (next_start, next_end, new_pos) = self.parse_concatenation(chars, pos)?;
                pos = new_pos;

                self.add_epsilon(alt_start, next_start);
                self.add_epsilon(next_end, alt_end);
            }

            Ok((alt_start, alt_end, pos))
        } else {
            Ok((first_start, first_end, pos))
        }
    }

    /// Parse concatenation: atom+
    fn parse_concatenation(
        &mut self,
        chars: &[char],
        mut pos: usize,
    ) -> Result<(usize, usize, usize), NfaError> {
        let (mut start, mut end, new_pos) = self.parse_atom(chars, pos)?;
        pos = new_pos;

        while pos < chars.len() && chars[pos] != ')' && chars[pos] != '|' {
            let (next_start, next_end, new_pos) = self.parse_atom(chars, pos)?;
            pos = new_pos;

            self.add_epsilon(end, next_start);
            end = next_end;
        }

        Ok((start, end, pos))
    }

    /// Parse atom: '(' alternation ')' | char
    fn parse_atom(
        &mut self,
        chars: &[char],
        mut pos: usize,
    ) -> Result<(usize, usize, usize), NfaError> {
        if pos >= chars.len() {
            return Err(NfaError::UnexpectedEnd);
        }

        if chars[pos] == '(' {
            pos += 1;  // Skip '('
            let (start, end, new_pos) = self.parse_alternation(chars, pos)?;
            pos = new_pos;

            if pos >= chars.len() || chars[pos] != ')' {
                return Err(NfaError::UnmatchedParen);
            }
            pos += 1;  // Skip ')'

            Ok((start, end, pos))
        } else {
            // Single character
            let start = self.new_state(false);
            let end = self.new_state(false);
            self.add_transition(start, end, chars[pos] as u8);
            Ok((start, end, pos + 1))
        }
    }

    fn add_rule(&mut self, rule: &PhoneticRule) {
        // Implementation depends on PhoneticRule structure
        // See: src/transducer/phonetic.rs
    }

    fn build(self) -> PhoneticNfa {
        let finals: Vec<usize> = self.states
            .iter()
            .filter(|s| s.is_final)
            .map(|s| s.id)
            .collect();

        PhoneticNfa {
            states: self.states,
            start: 0,
            finals,
        }
    }
}

/// Errors during NFA construction.
#[derive(Debug, Clone)]
pub enum NfaError {
    UnexpectedEnd,
    UnmatchedParen,
    InvalidPattern(String),
}

impl std::fmt::Display for NfaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NfaError::UnexpectedEnd => write!(f, "Unexpected end of pattern"),
            NfaError::UnmatchedParen => write!(f, "Unmatched parenthesis"),
            NfaError::InvalidPattern(s) => write!(f, "Invalid pattern: {}", s),
        }
    }
}

impl std::error::Error for NfaError {}
```

---

## FST Composition

**File**: `liblevenshtein-rust/src/wfst/composition.rs`

```rust
//! FST composition for combining phonetic NFA with Levenshtein automaton.

use super::*;
use crate::lattice::{Lattice, LatticeBuilder};

/// Composed automaton: NFA(phonetic) ∘ FST(Levenshtein) ∘ Trie(dictionary).
///
/// This structure enables phonetic-aware fuzzy matching by:
/// 1. Expanding query through phonetic NFA to generate variants
/// 2. Querying each variant through Levenshtein transducer
/// 3. Intersecting results with dictionary
/// 4. Combining weights from all stages
pub struct ComposedAutomaton<'a> {
    /// Phonetic pattern NFA
    phonetic_nfa: &'a PhoneticNfa,

    /// Levenshtein automaton
    levenshtein: Box<dyn LevenshteinAutomaton + 'a>,

    /// Optional dictionary for intersection
    dictionary: Option<Box<dyn Dictionary + 'a>>,
}

impl<'a> ComposedAutomaton<'a> {
    /// Create a new composed automaton.
    pub fn new(
        phonetic_nfa: &'a PhoneticNfa,
        levenshtein: impl LevenshteinAutomaton + 'a,
    ) -> Self {
        Self {
            phonetic_nfa,
            levenshtein: Box::new(levenshtein),
            dictionary: None,
        }
    }

    /// Add a dictionary for intersection.
    pub fn with_dictionary(mut self, dictionary: impl Dictionary + 'a) -> Self {
        self.dictionary = Some(Box::new(dictionary));
        self
    }

    /// Query with phonetic expansion.
    ///
    /// # Algorithm
    ///
    /// 1. Generate all phonetic variants of query (up to max_length)
    /// 2. For each variant, query Levenshtein automaton
    /// 3. Combine results with phonetic weight adjustment
    /// 4. Optionally filter through dictionary
    pub fn query(
        &self,
        term: &str,
        max_distance: usize,
    ) -> impl Iterator<Item = ComposedCandidate> + '_ {
        // Generate phonetic variants
        let max_length = term.len() + 2;  // Allow slight expansion
        let variants = self.phonetic_nfa.accepted_strings(max_length);

        // Query each variant and collect results
        let mut results = Vec::new();

        for variant in variants {
            let is_phonetic = variant != term;
            let phonetic_cost = if is_phonetic { 0.5 } else { 0.0 };

            for candidate in self.levenshtein.query(&variant, max_distance) {
                // Combine costs
                let total_cost = candidate.distance as f32 + phonetic_cost;

                results.push(ComposedCandidate {
                    term: candidate.term.clone(),
                    edit_distance: candidate.distance,
                    phonetic_cost,
                    total_cost,
                    via_variant: if is_phonetic { Some(variant.clone()) } else { None },
                });
            }
        }

        // Deduplicate and sort by total cost
        results.sort_by(|a, b| {
            a.total_cost
                .partial_cmp(&b.total_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Deduplicate by term (keep lowest cost)
        let mut seen = std::collections::HashSet::new();
        results.retain(|c| seen.insert(c.term.clone()));

        results.into_iter()
    }

    /// Query returning a lattice with phonetic edges.
    pub fn query_lattice(
        &self,
        tokens: &[&str],
        max_distance: usize,
    ) -> Lattice {
        let mut builder = LatticeBuilder::new()
            .with_input(tokens.join(" "));

        for (pos, token) in tokens.iter().enumerate() {
            let next_pos = pos + 1;

            // Get all candidates including phonetic variants
            for candidate in self.query(token, max_distance) {
                builder.add_correction(
                    pos,
                    next_pos,
                    &candidate.term,
                    candidate.total_cost,
                    candidate.edit_distance as u8,
                    candidate.phonetic_cost > 0.0,
                );
            }
        }

        builder.build(tokens.len())
    }
}

/// Result from composed automaton query.
#[derive(Clone, Debug)]
pub struct ComposedCandidate {
    /// The matched term
    pub term: String,

    /// Edit distance from nearest variant
    pub edit_distance: usize,

    /// Phonetic transformation cost
    pub phonetic_cost: f32,

    /// Total combined cost
    pub total_cost: f32,

    /// Phonetic variant used (if any)
    pub via_variant: Option<String>,
}
```

---

## Integration with Verified Phonetic Rules

**File**: `liblevenshtein-rust/src/wfst/phonetic_integration.rs`

```rust
//! Integration with liblevenshtein's Coq-verified phonetic rules.

use super::*;
use crate::transducer::phonetic::{PhoneticRule, ENGLISH_RULES};

impl PhoneticNfa {
    /// Build NFA from liblevenshtein's verified English phonetic rules.
    ///
    /// These rules are formally verified in Coq and include:
    /// - Zompist English spelling rules (13 rules)
    /// - Vowel alternations
    /// - Consonant equivalences
    ///
    /// See: docs/verification/phonetic/ for proofs
    pub fn from_english_basic() -> Self {
        Self::from_phonetic_rules(&ENGLISH_RULES[..])
    }

    /// Build NFA for a specific phonetic pattern with custom costs.
    pub fn from_rules_with_costs<C: CostFunction>(
        rules: &[PhoneticRule],
        cost_function: &C,
    ) -> WeightedPhoneticNfa {
        let mut builder = WeightedNfaBuilder::new();

        for rule in rules {
            let cost = match rule {
                PhoneticRule::Substitution { from, to, .. } => {
                    cost_function.substitution_cost(*from, *to)
                }
                // Handle other rule types...
                _ => 1.0,
            };
            builder.add_rule_with_cost(rule, cost);
        }

        builder.build()
    }
}

/// Weighted phonetic NFA with transition costs.
pub struct WeightedPhoneticNfa {
    nfa: PhoneticNfa,
    transition_costs: HashMap<(usize, usize), TropicalWeight>,
}

impl WeightedPhoneticNfa {
    /// Get the cost of traversing from state `from` to state `to`.
    pub fn transition_cost(&self, from: usize, to: usize) -> TropicalWeight {
        self.transition_costs
            .get(&(from, to))
            .copied()
            .unwrap_or(TropicalWeight::ONE)
    }
}
```

---

## MORK Integration

### WfstFuzzySource

**Extension to**: `MORK/kernel/src/fuzzy_source.rs`

```rust
/// Extended fuzzy source with full WFST support.
///
/// Recognizes expressions of the form:
/// ```metta
/// (WFST max_dist pattern phonetic?)
/// ```
pub struct WfstFuzzySource {
    e: Expr,
    config: WfstConfig,
    composed: Option<ComposedAutomaton<'static>>,
}

/// Configuration for WFST-based fuzzy matching.
pub struct WfstConfig {
    /// Maximum edit distance
    pub max_distance: usize,

    /// Base algorithm (Standard, Transposition, MergeAndSplit)
    pub algorithm: Algorithm,

    /// Enable phonetic pattern expansion
    pub phonetic_enabled: bool,

    /// Cost for phonetic transformations
    pub phonetic_weight: f32,

    /// Weight combination scheme
    pub weight_scheme: WeightScheme,

    /// Return results as lattice (vs flat list)
    pub lattice_mode: bool,
}

/// Weight combination scheme.
pub enum WeightScheme {
    /// Weight = edit_distance + phonetic_cost
    Additive,

    /// Weight = edit_distance * phonetic_multiplier
    Multiplicative,

    /// Custom weighting function
    Custom(Box<dyn Fn(usize, f32) -> f32>),
}
```

---

## Usage Examples

### Phonetic NFA

```rust
use liblevenshtein::wfst::PhoneticNfa;

// Compile pattern for "phone" variants
let nfa = PhoneticNfa::compile("(ph|f)(o|oa)(n|ne)")?;

// Check acceptance
assert!(nfa.accepts("phone"));
assert!(nfa.accepts("fone"));
assert!(nfa.accepts("phoane"));
assert!(!nfa.accepts("cat"));

// Generate all accepted strings
let variants = nfa.accepted_strings(6);
// ["foan", "foane", "fon", "fone", "phoan", "phoane", "phon", "phone"]
```

### Composed Query

```rust
use liblevenshtein::wfst::{PhoneticNfa, ComposedAutomaton};
use liblevenshtein::transducer::Transducer;

let phonetic_nfa = PhoneticNfa::compile("(ph|f)(o|oa)(n|ne)")?;
let transducer = Transducer::new(dictionary, Algorithm::Standard);

let composed = phonetic_nfa.compose_with_levenshtein(&transducer);

for candidate in composed.query("phone", 2) {
    println!(
        "{}: edit={}, phonetic={:.1}, total={:.1}",
        candidate.term,
        candidate.edit_distance,
        candidate.phonetic_cost,
        candidate.total_cost
    );
}
// phone: edit=0, phonetic=0.0, total=0.0
// fone: edit=0, phonetic=0.5, total=0.5
// phones: edit=1, phonetic=0.0, total=1.0
// phon: edit=1, phonetic=0.0, total=1.0
```

### MORK Query with WFST

```metta
; Complex phonetic pattern matching
!(match &space
    (wfst-query
        (pattern "(ph|f)(one|oan)")
        (max-dist 2)
        (phonetic english)
        (top-k 10))
    $results)

; Returns candidates with combined scores:
; [(phone 0.0) (fone 0.5) (foam 1.5) ...]
```

---

## Performance Considerations

### NFA State Explosion

- Alternations multiply states: `(a|b)(c|d)` = 4 paths
- Limit pattern complexity or accepted string length
- Use lazy evaluation for large pattern spaces

### Composition Complexity

- NFA × Levenshtein: $`\mathcal{O}(\lvert\mathrm{NFA}\rvert \times \lvert\text{LEV states}\rvert)`$
- Practical limit: ~1000 NFA states, ~10 Levenshtein states
- Memoization helps for repeated queries

### Caching Strategies

1. **Cache compiled NFAs**: Reuse for same pattern
2. **Cache composed automata**: Reuse for same pattern + algorithm
3. **Cache accepted strings**: Expensive to regenerate

---

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_semiring() {
        let a = TropicalWeight(2.0);
        let b = TropicalWeight(3.0);

        assert_eq!(a.plus(b), TropicalWeight(2.0));  // min
        assert_eq!(a.times(b), TropicalWeight(5.0)); // +
    }

    #[test]
    fn test_phonetic_nfa_compile() {
        let nfa = PhoneticNfa::compile("(a|b)c").unwrap();
        assert!(nfa.accepts("ac"));
        assert!(nfa.accepts("bc"));
        assert!(!nfa.accepts("cc"));
    }

    #[test]
    fn test_accepted_strings() {
        let nfa = PhoneticNfa::compile("(a|b)(c|d)").unwrap();
        let accepted = nfa.accepted_strings(2);
        assert_eq!(accepted, vec!["ac", "ad", "bc", "bd"]);
    }
}
```

---

## References

- [Mohri et al., "Weighted Finite-State Transducers in Speech Recognition"](http://www.cs.nyu.edu/~mohri/pub/hbka.pdf)
- [Phase B: Lattice Integration](./lattice_integration.md)
- [Phonetic Rules Verification](../../verification/phonetic/README.md)
- WFST Architecture
