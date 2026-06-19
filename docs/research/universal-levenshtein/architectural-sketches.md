# Universal Levenshtein Automata - Architectural Sketches

**Date**: 2025-11-06
**Status**: Design Phase

---

## Overview

This document provides detailed code designs, trait definitions, struct layouts, and implementation sketches for Universal Levenshtein Automata features.

All designs follow **Option B (Configuration-Based)** approach, adding `substitution_set` as an optional configuration field.

---

## Table of Contents

1. [SubstitutionSet Structure](#substitutionset-structure)
2. [Builder API Extensions](#builder-api-extensions)
3. [Transition Function Modifications](#transition-function-modifications)
4. [Query Pipeline Integration](#query-pipeline-integration)
5. [Preset Substitution Sets](#preset-substitution-sets)
6. [Serialization Support](#serialization-support)
7. [Error Handling](#error-handling)
8. [Performance Optimizations](#performance-optimizations)

---

## SubstitutionSet Structure

### Basic Implementation

**File**: `/src/transducer/substitution.rs` (NEW)

```rust
use std::collections::HashSet;
use serde::{Deserialize, Serialize};

/// Represents a set of allowed character substitutions for restricted
/// Levenshtein distance computation.
///
/// In standard Levenshtein distance, any character can be substituted for
/// any other character. A `SubstitutionSet` restricts which character pairs
/// are allowed to substitute for each other.
///
/// # Mathematical Definition
///
/// Let S ⊆ Σ × Σ be a set of allowed character pairs. The generalized
/// distance d_L^S allows substituting character `a` for `b` only if
/// (a, b) ∈ S.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::SubstitutionSet;
///
/// let mut set = SubstitutionSet::new();
/// set.add_bidirectional('a', 's');  // Keyboard adjacent
/// set.add_bidirectional('a', 'd');
///
/// assert!(set.is_allowed('a', 's'));
/// assert!(set.is_allowed('s', 'a'));
/// assert!(!set.is_allowed('a', 'z'));  // Not adjacent
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubstitutionSet {
    /// Set of allowed (source, target) character pairs.
    /// (a, b) ∈ allowed ⟺ character a can substitute for character b.
    allowed: HashSet<(char, char)>,
}

impl SubstitutionSet {
    /// Creates a new empty substitution set.
    ///
    /// # Note
    /// An empty set means NO substitutions are allowed (only insert/delete).
    /// For unrestricted substitutions (standard Levenshtein), use
    /// `None` when building a transducer.
    pub fn new() -> Self {
        Self {
            allowed: HashSet::new(),
        }
    }

    /// Creates a new substitution set with the specified capacity.
    ///
    /// Useful when you know approximately how many substitution pairs
    /// you'll be adding, to avoid reallocations.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            allowed: HashSet::with_capacity(capacity),
        }
    }

    /// Adds an allowed substitution from `source` to `target`.
    ///
    /// This is a unidirectional operation. If you want bidirectional
    /// substitutions, use `add_bidirectional` or call this method twice.
    ///
    /// # Example
    /// ```
    /// let mut set = SubstitutionSet::new();
    /// set.add('a', 'b');
    ///
    /// assert!(set.is_allowed('a', 'b'));
    /// assert!(!set.is_allowed('b', 'a'));  // Not bidirectional
    /// ```
    pub fn add(&mut self, source: char, target: char) {
        self.allowed.insert((source, target));
    }

    /// Adds an allowed substitution in both directions.
    ///
    /// Equivalent to calling `add(a, b)` and `add(b, a)`.
    ///
    /// # Example
    /// ```
    /// let mut set = SubstitutionSet::new();
    /// set.add_bidirectional('x', 'y');
    ///
    /// assert!(set.is_allowed('x', 'y'));
    /// assert!(set.is_allowed('y', 'x'));  // Bidirectional
    /// ```
    pub fn add_bidirectional(&mut self, a: char, b: char) {
        self.allowed.insert((a, b));
        self.allowed.insert((b, a));
    }

    /// Checks if substituting `source` for `target` is allowed.
    ///
    /// # Returns
    /// `true` if (source, target) ∈ S, `false` otherwise.
    ///
    /// # Performance
    /// O(1) average case (HashSet lookup).
    #[inline]
    pub fn is_allowed(&self, source: char, target: char) -> bool {
        self.allowed.contains(&(source, target))
    }

    /// Removes an allowed substitution.
    ///
    /// Returns `true` if the pair was present, `false` otherwise.
    pub fn remove(&mut self, source: char, target: char) -> bool {
        self.allowed.remove(&(source, target))
    }

    /// Returns the number of allowed substitution pairs.
    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    /// Checks if the substitution set is empty.
    ///
    /// An empty set means no substitutions are allowed.
    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }

    /// Clears all allowed substitutions.
    pub fn clear(&mut self) {
        self.allowed.clear();
    }

    /// Returns an iterator over all allowed substitution pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(char, char)> {
        self.allowed.iter()
    }
}

impl Default for SubstitutionSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<(char, char)> for SubstitutionSet {
    fn from_iter<I: IntoIterator<Item = (char, char)>>(iter: I) -> Self {
        Self {
            allowed: iter.into_iter().collect(),
        }
    }
}
```

---

## Builder API Extensions

### TransducerBuilder Modifications

**File**: `/src/transducer/builder.rs`

```rust
use crate::transducer::substitution::SubstitutionSet;

pub struct TransducerBuilder<D> {
    algorithm: Algorithm,
    substitution_set: Option<SubstitutionSet>,  // NEW FIELD
    // ... other existing fields (max_distance, etc.)
}

impl<D> TransducerBuilder<D> {
    pub fn new() -> Self {
        Self {
            algorithm: Algorithm::Standard,
            substitution_set: None,  // Default: unrestricted
            // ... other defaults
        }
    }

    /// Configures the transducer to use restricted substitutions.
    ///
    /// When set, only substitutions in the provided set are allowed during
    /// fuzzy search. This is useful for:
    /// - Keyboard-proximity spell checking (QWERTY, AZERTY, Dvorak)
    /// - OCR error correction (visual confusions: 0↔O, 1↔I↔l)
    /// - Phonetic matching (sound-alike substitutions: f↔v, s↔c)
    /// - Handwriting recognition (shape similarities: a↔o, n↔m)
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::prelude::*;
    ///
    /// // Create QWERTY keyboard adjacency set
    /// let qwerty = SubstitutionSet::qwerty();
    ///
    /// let dict = TransducerBuilder::new()
    ///     .algorithm(Algorithm::Standard)
    ///     .with_substitution_set(qwerty)
    ///     .build_from_iter(words);
    ///
    /// // "tesy" → "test" (y↔t keyboard-adjacent, distance=1)
    /// let results: Vec<_> = dict.fuzzy_search("tesy", 1).collect();
    /// assert!(results.contains(&"test"));
    ///
    /// // "texz" → not "test" (x↔s not keyboard-adjacent, distance>1)
    /// let results: Vec<_> = dict.fuzzy_search("texz", 1).collect();
    /// assert!(!results.contains(&"test"));
    /// ```
    ///
    /// # Compatibility
    /// Works with all algorithms: Standard, Transposition, MergeAndSplit.
    ///
    /// # Performance
    /// Adds ~10-20% overhead due to substitution validity checks.
    pub fn with_substitution_set(mut self, set: SubstitutionSet) -> Self {
        self.substitution_set = Some(set);
        self
    }

    /// Configures QWERTY keyboard proximity constraints.
    ///
    /// Shorthand for `.with_substitution_set(SubstitutionSet::qwerty())`.
    ///
    /// Allows substitutions between horizontally and diagonally adjacent
    /// keys on a standard QWERTY keyboard layout.
    ///
    /// # Example
    /// ```
    /// let dict = TransducerBuilder::new()
    ///     .with_qwerty_substitutions()
    ///     .build_from_iter(words);
    /// ```
    pub fn with_qwerty_substitutions(self) -> Self {
        self.with_substitution_set(SubstitutionSet::qwerty())
    }

    /// Configures AZERTY keyboard proximity constraints.
    ///
    /// Shorthand for `.with_substitution_set(SubstitutionSet::azerty())`.
    pub fn with_azerty_substitutions(self) -> Self {
        self.with_substitution_set(SubstitutionSet::azerty())
    }

    /// Configures Dvorak keyboard proximity constraints.
    ///
    /// Shorthand for `.with_substitution_set(SubstitutionSet::dvorak())`.
    pub fn with_dvorak_substitutions(self) -> Self {
        self.with_substitution_set(SubstitutionSet::dvorak())
    }

    /// Configures OCR visual confusion constraints.
    ///
    /// Allows substitutions between visually similar characters commonly
    /// confused in optical character recognition:
    /// - 0 ↔ O (digit zero vs letter O)
    /// - 1 ↔ I ↔ l (digit one, capital I, lowercase L)
    /// - 5 ↔ S, 8 ↔ B, etc.
    ///
    /// # Example
    /// ```
    /// let dict = TransducerBuilder::new()
    ///     .with_ocr_confusions()
    ///     .build_from_iter(scanned_text_dictionary);
    /// ```
    pub fn with_ocr_confusions(self) -> Self {
        self.with_substitution_set(SubstitutionSet::ocr_confusions())
    }

    /// Configures English phonetic similarity constraints.
    ///
    /// Allows substitutions between characters with similar pronunciations
    /// in English: f↔v, s↔c↔k, etc.
    ///
    /// Useful for "sounds-like" fuzzy matching.
    pub fn with_phonetic_english(self) -> Self {
        self.with_substitution_set(SubstitutionSet::phonetic_english())
    }

    // ... rest of builder methods ...

    pub fn build_from_iter<I>(self, terms: I) -> Transducer<D>
    where
        I: IntoIterator<Item = impl AsRef<str>>,
    {
        // Pass substitution_set to Transducer construction
        Transducer {
            algorithm: self.algorithm,
            substitution_set: self.substitution_set,  // NEW
            // ... other fields
        }
    }
}
```

---

## Transition Function Modifications

### Standard Algorithm Transitions

**File**: `/src/transducer/transition.rs`

```rust
/// Computes the next state for the Standard algorithm with optional
/// restricted substitutions.
///
/// # Parameters
/// - `curr_state`: Current set of positions
/// - `dict_char`: Character from dictionary term being matched
/// - `query_chars`: Characters of the query string
/// - `max_distance`: Maximum allowed edit distance
/// - `substitution_set`: Optional set of allowed substitutions
///   - `None` → all substitutions allowed (standard Levenshtein)
///   - `Some(set)` → only substitutions in set allowed (Universal LA)
pub(crate) fn standard_transition(
    curr_state: &State,
    dict_char: char,
    query_chars: &[char],
    max_distance: usize,
    substitution_set: Option<&SubstitutionSet>,  // NEW PARAMETER
) -> State {
    let mut next_state = State::new();
    let query_len = query_chars.len();

    for &Position { term_index: i, num_errors: e, .. } in curr_state.iter() {
        // Process positions with query characters remaining
        if i < query_len {
            let query_char = query_chars[i];

            // ═══════════════════════════════════════════════════
            // MATCH: Same character, no error
            // ═══════════════════════════════════════════════════
            if query_char == dict_char {
                next_state.insert(Position::new(i + 1, e, false));
            }

            // ═══════════════════════════════════════════════════
            // SUBSTITUTION: Different characters
            // NEW: Check if substitution is allowed
            // ═══════════════════════════════════════════════════
            if query_char != dict_char && e < max_distance {
                // Determine if substitution is allowed
                let substitution_allowed = match substitution_set {
                    // No restrictions → all substitutions allowed
                    None => true,

                    // Check if (query_char, dict_char) ∈ S
                    Some(set) => set.is_allowed(query_char, dict_char),
                };

                if substitution_allowed {
                    next_state.insert(Position::new(i + 1, e + 1, false));
                }
            }

            // ═══════════════════════════════════════════════════
            // DELETION: Remove character from query
            // Always allowed (no restriction)
            // ═══════════════════════════════════════════════════
            if e < max_distance {
                next_state.insert(Position::new(i + 1, e + 1, false));
            }
        }

        // ═══════════════════════════════════════════════════
        // INSERTION: Add character to query
        // Always allowed (no restriction)
        // ═══════════════════════════════════════════════════
        if e < max_distance {
            next_state.insert(Position::new(i, e + 1, false));
        }
    }

    next_state
}
```

### Transposition Algorithm Transitions

**File**: `/src/transducer/transition.rs`

```rust
/// Computes the next state for the Transposition algorithm with optional
/// restricted substitutions.
///
/// Transposition allows swapping adjacent characters (ab → ba).
/// Substitution restrictions apply to non-transposition character matches.
pub(crate) fn transposition_transition(
    curr_state: &State,
    dict_char: char,
    query_chars: &[char],
    max_distance: usize,
    substitution_set: Option<&SubstitutionSet>,  // NEW PARAMETER
) -> State {
    let mut next_state = State::new();
    let query_len = query_chars.len();

    for &Position { term_index: i, num_errors: e, is_special } in curr_state.iter() {
        if i < query_len {
            let query_char = query_chars[i];

            // ═══════════════════════════════════════════════════
            // MATCH: Same character
            // ═══════════════════════════════════════════════════
            if query_char == dict_char {
                next_state.insert(Position::new(i + 1, e, false));
            }

            // ═══════════════════════════════════════════════════
            // SUBSTITUTION: Check if allowed
            // ═══════════════════════════════════════════════════
            if query_char != dict_char && e < max_distance {
                let substitution_allowed = match substitution_set {
                    None => true,
                    Some(set) => set.is_allowed(query_char, dict_char),
                };

                if substitution_allowed {
                    next_state.insert(Position::new(i + 1, e + 1, false));
                }
            }

            // ═══════════════════════════════════════════════════
            // TRANSPOSITION: Swap adjacent characters
            // Transposition operation itself is ALWAYS allowed
            // (not restricted by substitution set)
            // ═══════════════════════════════════════════════════
            if i + 1 < query_len && e < max_distance {
                let next_query_char = query_chars[i + 1];

                // Check for transposition pattern: query[i] != dict[j] but query[i+1] == dict[j]
                if query_char != dict_char && next_query_char == dict_char && !is_special {
                    // Mark as special to indicate transposition in progress
                    next_state.insert(Position::new(i, e + 1, true));
                }
            }

            // Complete transposition (special state)
            if is_special && query_char == dict_char {
                next_state.insert(Position::new(i + 1, e, false));
            }

            // ═══════════════════════════════════════════════════
            // DELETION: Always allowed
            // ═══════════════════════════════════════════════════
            if e < max_distance {
                next_state.insert(Position::new(i + 1, e + 1, false));
            }
        }

        // ═══════════════════════════════════════════════════
        // INSERTION: Always allowed
        // ═══════════════════════════════════════════════════
        if e < max_distance {
            next_state.insert(Position::new(i, e + 1, false));
        }
    }

    next_state
}
```

### MergeAndSplit Algorithm Transitions

**File**: `/src/transducer/transition.rs`

```rust
/// Computes the next state for the MergeAndSplit algorithm with optional
/// restricted substitutions.
///
/// - Merge: Two characters → one (e.g., "rn" → "m")
/// - Split: One character → two (e.g., "m" → "rn")
///
/// Merge/split operations are ALWAYS allowed (not restricted).
/// Substitution restrictions apply to regular character matches.
pub(crate) fn merge_split_transition(
    curr_state: &State,
    dict_char: char,
    query_chars: &[char],
    max_distance: usize,
    substitution_set: Option<&SubstitutionSet>,  // NEW PARAMETER
) -> State {
    let mut next_state = State::new();
    let query_len = query_chars.len();

    for &Position { term_index: i, num_errors: e, is_special } in curr_state.iter() {
        if i < query_len {
            let query_char = query_chars[i];

            // ═══════════════════════════════════════════════════
            // MATCH
            // ═══════════════════════════════════════════════════
            if query_char == dict_char {
                next_state.insert(Position::new(i + 1, e, false));
            }

            // ═══════════════════════════════════════════════════
            // SUBSTITUTION: Check if allowed
            // ═══════════════════════════════════════════════════
            if query_char != dict_char && e < max_distance {
                let substitution_allowed = match substitution_set {
                    None => true,
                    Some(set) => set.is_allowed(query_char, dict_char),
                };

                if substitution_allowed {
                    next_state.insert(Position::new(i + 1, e + 1, false));
                }
            }

            // ═══════════════════════════════════════════════════
            // MERGE: Two query characters → one dict character
            // ALWAYS allowed (not restricted)
            // ═══════════════════════════════════════════════════
            if i + 1 < query_len && e < max_distance && !is_special {
                // Merge operation: consume two query chars for one dict char
                next_state.insert(Position::new(i + 2, e + 1, true));
            }

            // ═══════════════════════════════════════════════════
            // DELETION
            // ═══════════════════════════════════════════════════
            if e < max_distance {
                next_state.insert(Position::new(i + 1, e + 1, false));
            }
        }

        // ═══════════════════════════════════════════════════
        // INSERTION & SPLIT
        // Both ALWAYS allowed (not restricted)
        // ═══════════════════════════════════════════════════
        if e < max_distance {
            next_state.insert(Position::new(i, e + 1, !is_special));
        }
    }

    next_state
}
```

---

## Query Pipeline Integration

### Transducer Structure

**File**: `/src/transducer/mod.rs`

```rust
pub struct Transducer<D> {
    dictionary: D,
    algorithm: Algorithm,
    substitution_set: Option<SubstitutionSet>,  // NEW FIELD
}

impl<D> Transducer<D>
where
    D: Dictionary,
{
    pub fn fuzzy_search(&self, query: &str, max_distance: usize) -> QueryIterator<D> {
        QueryIterator::new(
            &self.dictionary,
            query,
            self.algorithm,
            max_distance,
            self.substitution_set.as_ref(),  // Pass reference to query iterator
        )
    }
}
```

### QueryIterator Modifications

**File**: `/src/transducer/query.rs`

```rust
pub struct QueryIterator<'a, D> {
    dictionary: &'a D,
    query_chars: Vec<char>,
    algorithm: Algorithm,
    max_distance: usize,
    substitution_set: Option<&'a SubstitutionSet>,  // NEW FIELD
    // ... other fields (stack, current_state, etc.)
}

impl<'a, D> QueryIterator<'a, D>
where
    D: Dictionary,
{
    pub(crate) fn new(
        dictionary: &'a D,
        query: &str,
        algorithm: Algorithm,
        max_distance: usize,
        substitution_set: Option<&'a SubstitutionSet>,  // NEW PARAMETER
    ) -> Self {
        Self {
            dictionary,
            query_chars: query.chars().collect(),
            algorithm,
            max_distance,
            substitution_set,  // Store reference
            // ... initialize other fields
        }
    }

    fn next_state(&mut self, dict_char: char) -> State {
        use crate::transducer::transition::*;

        match self.algorithm {
            Algorithm::Standard => {
                standard_transition(
                    &self.current_state,
                    dict_char,
                    &self.query_chars,
                    self.max_distance,
                    self.substitution_set,  // Pass through
                )
            }
            Algorithm::Transposition => {
                transposition_transition(
                    &self.current_state,
                    dict_char,
                    &self.query_chars,
                    self.max_distance,
                    self.substitution_set,  // Pass through
                )
            }
            Algorithm::MergeAndSplit => {
                merge_split_transition(
                    &self.current_state,
                    dict_char,
                    &self.query_chars,
                    self.max_distance,
                    self.substitution_set,  // Pass through
                )
            }
        }
    }
}
```

---

## Preset Substitution Sets

### QWERTY Keyboard

**File**: `/src/transducer/substitution.rs`

```rust
impl SubstitutionSet {
    /// Creates a substitution set for QWERTY keyboard proximity.
    ///
    /// Includes:
    /// - Horizontal adjacencies (q↔w, w↔e, etc.)
    /// - Diagonal adjacencies (q↔a, w↔s, etc.)
    /// - Lowercase letters only
    ///
    /// For uppercase, call `.add_uppercase_variants()` after construction.
    pub fn qwerty() -> Self {
        let mut set = SubstitutionSet::with_capacity(150);

        // Row 1: q w e r t y u i o p
        let row1 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'];
        for window in row1.windows(2) {
            set.add_bidirectional(window[0], window[1]);
        }

        // Row 2: a s d f g h j k l
        let row2 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'];
        for window in row2.windows(2) {
            set.add_bidirectional(window[0], window[1]);
        }

        // Row 3: z x c v b n m
        let row3 = ['z', 'x', 'c', 'v', 'b', 'n', 'm'];
        for window in row3.windows(2) {
            set.add_bidirectional(window[0], window[1]);
        }

        // Diagonal adjacencies: row 1 ↔ row 2
        let diagonals_1_2 = [
            ('q', 'a'), ('w', 'a'), ('w', 's'), ('e', 's'), ('e', 'd'),
            ('r', 'd'), ('r', 'f'), ('t', 'f'), ('t', 'g'), ('y', 'g'),
            ('y', 'h'), ('u', 'h'), ('u', 'j'), ('i', 'j'), ('i', 'k'),
            ('o', 'k'), ('o', 'l'), ('p', 'l'),
        ];
        for &(a, b) in &diagonals_1_2 {
            set.add_bidirectional(a, b);
        }

        // Diagonal adjacencies: row 2 ↔ row 3
        let diagonals_2_3 = [
            ('a', 'z'), ('s', 'z'), ('s', 'x'), ('d', 'x'), ('d', 'c'),
            ('f', 'c'), ('f', 'v'), ('g', 'v'), ('g', 'b'), ('h', 'b'),
            ('h', 'n'), ('j', 'n'), ('j', 'm'), ('k', 'm'),
        ];
        for &(a, b) in &diagonals_2_3 {
            set.add_bidirectional(a, b);
        }

        set
    }

    /// Adds uppercase variants for all lowercase pairs in the set.
    ///
    /// For each (a, b) where both are lowercase, adds:
    /// - (A, B) - both uppercase
    /// - (a, B) - mixed case
    /// - (A, b) - mixed case
    pub fn add_uppercase_variants(&mut self) -> &mut Self {
        let lowercase_pairs: Vec<_> = self.allowed
            .iter()
            .filter(|(a, b)| a.is_lowercase() && b.is_lowercase())
            .copied()
            .collect();

        for (a, b) in lowercase_pairs {
            let a_upper = a.to_uppercase().next().unwrap();
            let b_upper = b.to_uppercase().next().unwrap();

            self.add(a_upper, b_upper);  // A ↔ B
            self.add(a, b_upper);        // a ↔ B
            self.add(a_upper, b);        // A ↔ b
        }

        self
    }
}
```

### OCR Confusions

```rust
impl SubstitutionSet {
    /// Creates a substitution set for common OCR visual confusions.
    ///
    /// Includes digit-letter and letter-letter confusions commonly
    /// seen in optical character recognition systems.
    pub fn ocr_confusions() -> Self {
        let mut set = SubstitutionSet::with_capacity(50);

        // Digit ↔ Letter confusions
        set.add_bidirectional('0', 'O');
        set.add_bidirectional('0', 'o');
        set.add_bidirectional('1', 'I');
        set.add_bidirectional('1', 'l');
        set.add_bidirectional('1', 'i');
        set.add_bidirectional('I', 'l');
        set.add_bidirectional('2', 'Z');
        set.add_bidirectional('2', 'z');
        set.add_bidirectional('5', 'S');
        set.add_bidirectional('5', 's');
        set.add_bidirectional('6', 'G');
        set.add_bidirectional('6', 'b');
        set.add_bidirectional('8', 'B');
        set.add_bidirectional('9', 'g');
        set.add_bidirectional('9', 'q');

        // Letter ↔ Letter confusions
        set.add_bidirectional('c', 'e');
        set.add_bidirectional('n', 'm');
        set.add_bidirectional('u', 'v');
        set.add_bidirectional('r', 'n');
        set.add_bidirectional('cl', 'd');  // Multi-char (if supported)

        set
    }
}
```

### Phonetic English

```rust
impl SubstitutionSet {
    /// Creates a substitution set for English phonetic similarities.
    ///
    /// Includes consonant and vowel confusions based on similar
    /// pronunciations in English.
    pub fn phonetic_english() -> Self {
        let mut set = SubstitutionSet::with_capacity(40);

        // Consonant confusions
        set.add_bidirectional('c', 'k');  // cat/kat
        set.add_bidirectional('c', 's');  // city/sity
        set.add_bidirectional('s', 'z');  // hose/hoze
        set.add_bidirectional('f', 'v');  // leaf/leav
        set.add_bidirectional('g', 'j');  // giraffe/jiraffe
        set.add_bidirectional('b', 'p');  // bad/pad
        set.add_bidirectional('d', 't');  // bed/bet

        // Vowel confusions
        set.add_bidirectional('a', 'e');
        set.add_bidirectional('i', 'y');
        set.add_bidirectional('o', 'u');

        // Uppercase variants
        set.add_uppercase_variants();

        set
    }
}
```

---

## Serialization Support

### Serde Integration

**File**: `/src/transducer/substitution.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubstitutionSet {
    allowed: HashSet<(char, char)>,
}

// SubstitutionSet can be serialized/deserialized automatically
// via serde's derive macros.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let mut set = SubstitutionSet::new();
        set.add_bidirectional('a', 'b');
        set.add_bidirectional('c', 'd');

        // Serialize to JSON
        let json = serde_json::to_string(&set).unwrap();

        // Deserialize from JSON
        let deserialized: SubstitutionSet = serde_json::from_str(&json).unwrap();

        assert_eq!(set, deserialized);
    }
}
```

---

## Error Handling

### No New Errors Expected

**Rationale**: Substitution restrictions don't introduce new error conditions.

- Empty SubstitutionSet is valid (means no substitutions allowed)
- None substitution_set is valid (means all substitutions allowed)
- Invalid character pairs are simply not allowed (no error, just not in set)

**Result**: No new error types needed.

---

## Performance Optimizations

### Phase 4 Optimizations (Optional)

#### 1. Perfect Hashing for Static Sets

**Current implementation decision**: `SubstitutionSet::keyboard_qwerty()`
uses a const pair table plus the measured hybrid storage strategy:

- `Small(Vec<(u8, u8)>)` for tiny sets where linear scan wins on cache and
  hashing overhead.
- `Large(FxHashSet<(u8, u8)>)` once the set exceeds the empirically measured
  crossover.

**Alternative to evaluate**: a dedicated static-membership backend could use a
compile-time perfect hash table for read-only presets, but it should only be
accepted if a Criterion comparison shows a real latency improvement without
increasing code size or complicating the generic substitution-set API.

#### 2. Bit Vector for ASCII

```rust
/// Optimized substitution set for ASCII-only alphabets.
///
/// Uses 256×256 bit array (8 KB) for O(1) lookup with excellent cache locality.
pub struct AsciiSubstitutionSet {
    /// Bit array: 256 chars × 256 chars / 64 bits/word = 1024 u64s
    allowed: [u64; 1024],
}

impl AsciiSubstitutionSet {
    pub fn new() -> Self {
        Self {
            allowed: [0; 1024],
        }
    }

    #[inline]
    pub fn add(&mut self, source: char, target: char) {
        if let (Some(src), Some(tgt)) = (Self::to_ascii(source), Self::to_ascii(target)) {
            let index = ((src as usize) << 8) | (tgt as usize);
            let word = index / 64;
            let bit = index % 64;
            self.allowed[word] |= 1u64 << bit;
        }
    }

    #[inline]
    pub fn is_allowed(&self, source: char, target: char) -> bool {
        if let (Some(src), Some(tgt)) = (Self::to_ascii(source), Self::to_ascii(target)) {
            let index = ((src as usize) << 8) | (tgt as usize);
            let word = index / 64;
            let bit = index % 64;
            (self.allowed[word] & (1u64 << bit)) != 0
        } else {
            false
        }
    }

    fn to_ascii(c: char) -> Option<u8> {
        if c.is_ascii() {
            Some(c as u8)
        } else {
            None
        }
    }
}
```

---

## Testing Infrastructure

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitution_allowed() {
        let mut set = SubstitutionSet::new();
        set.add('a', 'b');

        assert!(set.is_allowed('a', 'b'));
        assert!(!set.is_allowed('b', 'a'));  // Unidirectional
        assert!(!set.is_allowed('a', 'c'));  // Not in set
    }

    #[test]
    fn test_bidirectional() {
        let mut set = SubstitutionSet::new();
        set.add_bidirectional('x', 'y');

        assert!(set.is_allowed('x', 'y'));
        assert!(set.is_allowed('y', 'x'));
    }

    #[test]
    fn test_query_with_restrictions() {
        let mut set = SubstitutionSet::new();
        set.add_bidirectional('t', 'y');  // Adjacent on QWERTY

        let words = vec!["test"];
        let dict = TransducerBuilder::new()
            .with_substitution_set(set)
            .build_from_iter(words);

        // "tesy" → "test" (y↔t allowed)
        let results: Vec<_> = dict.fuzzy_search("tesy", 1).collect();
        assert!(results.contains(&"test"));

        // "texz" → not "test" (x↔s not allowed, distance>1)
        let results: Vec<_> = dict.fuzzy_search("texz", 1).collect();
        assert!(!results.contains(&"test"));
    }
}
```

---

## Module Organization

```
src/
├── transducer/
│   ├── mod.rs                   # Transducer struct (add substitution_set field)
│   ├── algorithm.rs             # Algorithm enum (UNCHANGED)
│   ├── builder.rs               # TransducerBuilder (add with_substitution_set)
│   ├── substitution.rs          # NEW: SubstitutionSet struct + presets
│   ├── transition.rs            # Modify transition functions
│   ├── position.rs              # Position struct (likely UNCHANGED)
│   └── query.rs                 # QueryIterator (add substitution_set field)
```

---

**Last Updated**: 2025-11-06
**Status**: Design Complete, Ready for Implementation
**Next**: Begin Phase 1, Task 1.1 (Create SubstitutionSet structure)
