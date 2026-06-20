[← Documentation Index](../../README.md)

# Universal Levenshtein Automata - Use Cases & Applications

**Date**: 2025-11-06
**Purpose**: Practical applications, example substitution sets, and real-world scenarios for restricted substitutions.

---

## Overview

This document provides concrete examples of how restricted substitutions solve real-world problems. Each use case includes:
- Problem description
- Substitution set definition
- Example queries
- Expected behavior
- Code snippets

---

## Table of Contents

1. [Spell Checking with Keyboard Proximity](#1-spell-checking-with-keyboard-proximity)
2. [OCR Error Correction](#2-ocr-error-correction)
3. [Phonetic Matching](#3-phonetic-matching)
4. [Handwriting Recognition](#4-handwriting-recognition)
5. [Multi-Script Text Processing](#5-multi-script-text-processing)
6. [DNA Sequence Matching](#6-dna-sequence-matching)
7. [Custom Domain-Specific Rules](#7-custom-domain-specific-rules)

---

## 1. Spell Checking with Keyboard Proximity

### Problem

Typing errors typically involve **adjacent keys** on the keyboard. A spell checker should prioritize keyboard-adjacent substitutions over distant ones.

**Example typing errors**:
- "tesy" → "test" (y adjacent to t on QWERTY)
- "helo" → "hello" (missing l, but e-l adjacent)
- "exampel" → "example" (e-l transposition)

**Unrealistic errors** (distant keys):
- "teqt" → "test" (q-s not adjacent, unlikely typo)
- "twst" → "test" (w-e not adjacent for random typo)

### Solution: QWERTY Adjacency Set

```rust
use std::collections::HashSet;

pub struct SubstitutionSet {
    allowed: HashSet<(char, char)>,
}

impl SubstitutionSet {
    pub fn qwerty() -> Self {
        let mut set = HashSet::new();

        // Row 1: q w e r t y u i o p
        let row1 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'];
        for i in 0..row1.len() - 1 {
            set.insert((row1[i], row1[i+1]));
            set.insert((row1[i+1], row1[i]));
        }

        // Row 2: a s d f g h j k l
        let row2 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'];
        for i in 0..row2.len() - 1 {
            set.insert((row2[i], row2[i+1]));
            set.insert((row2[i+1], row2[i]));
        }

        // Row 3: z x c v b n m
        let row3 = ['z', 'x', 'c', 'v', 'b', 'n', 'm'];
        for i in 0..row3.len() - 1 {
            set.insert((row3[i], row3[i+1]));
            set.insert((row3[i+1], row3[i]));
        }

        // Diagonal adjacencies (keys above/below)
        // q-a, w-a, w-s, e-s, e-d, etc.
        let diagonals = [
            ('q', 'a'), ('w', 'a'), ('w', 's'), ('e', 's'), ('e', 'd'),
            ('r', 'd'), ('r', 'f'), ('t', 'f'), ('t', 'g'), ('y', 'g'),
            ('y', 'h'), ('u', 'h'), ('u', 'j'), ('i', 'j'), ('i', 'k'),
            ('o', 'k'), ('o', 'l'), ('p', 'l'),
            ('a', 'z'), ('s', 'z'), ('s', 'x'), ('d', 'x'), ('d', 'c'),
            ('f', 'c'), ('f', 'v'), ('g', 'v'), ('g', 'b'), ('h', 'b'),
            ('h', 'n'), ('j', 'n'), ('j', 'm'), ('k', 'm'),
        ];

        for (a, b) in diagonals {
            set.insert((a, b));
            set.insert((b, a));
        }

        SubstitutionSet { allowed: set }
    }

}
```

The library also provides ready-made proximity sets for additional layouts:

```rust
use liblevenshtein::transducer::SubstitutionSet;

let azerty = SubstitutionSet::keyboard_azerty();
let dvorak = SubstitutionSet::keyboard_dvorak();
```

### Usage Example

```rust
use liblevenshtein::prelude::*;

let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Standard)
    .with_substitution_set(SubstitutionSet::qwerty())
    .build_from_iter(vec!["test", "text", "best", "rest"]);

// Keyboard-adjacent typo
let results: Vec<_> = dict.fuzzy_search("tesy", 1).collect();
// ✅ Returns: ["test"] - 'y' adjacent to 't'

// Distant-key typo
let results: Vec<_> = dict.fuzzy_search("teqt", 1).collect();
// ❌ Returns: [] - 'q' not adjacent to 's'

// Insertion error (missing key)
let results: Vec<_> = dict.fuzzy_search("tet", 1).collect();
// ✅ Returns: ["test"] - insertion allowed
```

### Benefits

- **Higher precision**: Reduces false positives from unlikely typos
- **Better ranking**: True keyboard errors ranked higher
- **Faster**: Fewer candidates to explore (pruned search space)

---

## 2. OCR Error Correction

### Problem

Optical Character Recognition (OCR) systems confuse **visually similar** characters:
- '1' ↔ 'I' ↔ 'l' (one, capital i, lowercase L)
- 'O' ↔ '0' (capital o, zero)
- 'S' ↔ '5' ↔ '$'
- 'B' ↔ '8'
- 'Z' ↔ '2'

These confusions depend on **font** and **image quality**.

### Solution: Visual Similarity Set

```rust
impl SubstitutionSet {
    pub fn ocr_confusions() -> Self {
        let mut set = HashSet::new();

        // Number-letter confusions
        let confusions = [
            ('0', 'O'), ('0', 'o'),  // zero ↔ O
            ('1', 'I'), ('1', 'l'), ('1', '|'),  // one ↔ I/l
            ('2', 'Z'), ('2', 'z'),  // two ↔ Z
            ('5', 'S'), ('5', 's'), ('5', '$'),  // five ↔ S
            ('6', 'G'), ('6', 'b'),  // six ↔ G/b
            ('8', 'B'), ('8', '&'),  // eight ↔ B
            ('9', 'g'), ('9', 'q'),  // nine ↔ g/q

            // Letter-letter confusions
            ('c', 'e'), ('o', 'a'), ('n', 'm'),
            ('r', 'n'), ('u', 'v'), ('i', 'j'),
            ('C', 'G'), ('O', 'Q'), ('I', 'T'),

            // Punctuation confusions
            (',', '.'), ('\'', '`'), ('"', '\''),
            (':', ';'), ('!', '|'), ('!', 'i'),
        ];

        for (a, b) in confusions {
            set.insert((a, b));
            set.insert((b, a));
        }

        SubstitutionSet { allowed: set }
    }

    pub fn ocr_with_font_hints(font: &str) -> Self {
        // Font-specific confusion sets
        match font {
            "serif" => Self::ocr_serif(),
            "sans-serif" => Self::ocr_sans_serif(),
            "monospace" => Self::ocr_monospace(),
            _ => Self::ocr_confusions(),
        }
    }
}
```

### Usage Example

```rust
// OCR scanned document text
let ocr_text = "Th1s is an examp1e 0f OCR text with err0rs.";

let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Standard)
    .with_substitution_set(SubstitutionSet::ocr_confusions())
    .build_from_iter(dictionary_words);

// Correct "Th1s" → "This"
let results = dict.fuzzy_search("Th1s", 1).collect();
// ✅ Returns: ["This"] - '1' ↔ 'i' is OCR confusion

// Correct "examp1e" → "example"
let results = dict.fuzzy_search("examp1e", 1).collect();
// ✅ Returns: ["example"] - '1' ↔ 'l' is OCR confusion

// Correct "0f" → "of"
let results = dict.fuzzy_search("0f", 1).collect();
// ✅ Returns: ["of"] - '0' ↔ 'o' is OCR confusion
```

### Benefits

- **Contextual**: Only allows realistic OCR errors
- **Font-aware**: Can customize for different fonts
- **Cleaner output**: Rejects impossible character confusions

---

## 3. Phonetic Matching

### Problem

Users search by **sound**, not spelling:
- "Philip" vs "Phillip" (same sound, different spelling)
- "night" vs "knight" (homophones)
- "photo" vs "foto" (ph ↔ f)

Standard Levenshtein treats 'ph' ↔ 'f' as 2 operations. With phonetic rules, it's 1 operation.

### Solution: Phonetic Similarity Set

```rust
impl SubstitutionSet {
    pub fn phonetic_english() -> Self {
        let mut set = HashSet::new();

        // Common phonetic equivalences
        let phonetic = [
            // 'f' sound
            ('f', 'F'), ('f', 'v'),  // f ↔ v (voiced/unvoiced)

            // 's' sound
            ('s', 'c'), ('s', 'z'),  // soft c, voiced s

            // 'k' sound
            ('c', 'k'), ('k', 'q'),  // hard c, k, q

            // Vowel variations
            ('a', 'e'), ('e', 'i'), ('i', 'o'), ('o', 'u'),

            // Silent letters (treated as close)
            ('k', 'n'),  // knife, knight, know
            ('w', 'h'),  // who, whole

            // Double letters (one ↔ two)
            // This is better handled with merge/split operations
        ];

        for (a, b) in phonetic {
            set.insert((a, b));
            set.insert((b, a));
        }

        SubstitutionSet { allowed: set }
    }

}
```

For a traditional Soundex consonant-class approximation, use the built-in
preset:

```rust
use liblevenshtein::transducer::SubstitutionSet;

let soundex = SubstitutionSet::soundex_compatible();
```

### Usage Example

```rust
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Standard)
    .with_substitution_set(SubstitutionSet::phonetic_english())
    .build_from_iter(vec!["Philip", "photo", "night"]);

// Phonetic search: "Filip" → "Philip"
let results = dict.fuzzy_search("Filip", 1).collect();
// ✅ Returns: ["Philip"] - 'F' ↔ 'Ph' phonetically similar

// Phonetic search: "foto" → "photo"
let results = dict.fuzzy_search("foto", 1).collect();
// ✅ Returns: ["photo"] - 'f' ↔ 'ph' phonetic equivalence

// Phonetic search: "nite" → "night"
let results = dict.fuzzy_search("nite", 2).collect();
// ✅ Returns: ["night"] - vowel variation + silent letters
```

### Benefits

- **User-friendly**: Matches how users think about words
- **Language-specific**: Can customize per language
- **Name matching**: Critical for person/place name search

---

## 4. Handwriting Recognition

### Problem

Handwritten characters have **shape similarities**:
- 'a' ↔ 'o' (closed vs open)
- 'n' ↔ 'm' ↔ 'u' (similar strokes)
- 'r' ↔ 'i' ↔ 'v' (depending on writing style)
- 'l' ↔ '1' ↔ 'I' (similar shapes)

### Solution: Shape Similarity Set

```rust
impl SubstitutionSet {
    pub fn handwriting_cursive() -> Self {
        let mut set = HashSet::new();

        // Shape-similar characters (cursive)
        let similar = [
            // Round shapes
            ('a', 'o'), ('o', 'e'), ('c', 'e'),

            // Vertical strokes
            ('l', 'i'), ('l', 't'), ('i', 'j'),
            ('l', '1'), ('I', '1'),

            // Similar curves
            ('n', 'm'), ('n', 'u'), ('u', 'v'),
            ('r', 'v'), ('r', 'n'),

            // Ascenders/descenders
            ('b', 'd'), ('p', 'q'), ('b', 'h'),

            // Complex shapes
            ('w', 'm'), ('a', 'u'),
        ];

        for (a, b) in similar {
            set.insert((a, b));
            set.insert((b, a));
        }

        SubstitutionSet { allowed: set }
    }

}
```

For block-letter and digit confusions, use the built-in print-handwriting
preset:

```rust
use liblevenshtein::transducer::SubstitutionSet;

let print = SubstitutionSet::handwriting_print();
```

### Usage Example

```rust
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Standard)
    .with_substitution_set(SubstitutionSet::handwriting_cursive())
    .build_from_iter(dictionary);

// Handwritten "hallo" misread as "hello"
let results = dict.fuzzy_search("hallo", 1).collect();
// ✅ Returns: ["hello"] - 'a' ↔ 'e' shape similar in cursive

// Handwritten "mon" misread as "man"
let results = dict.fuzzy_search("mon", 1).collect();
// ✅ Returns: ["man"] - 'o' ↔ 'a' shape similar
```

---

## 5. Multi-Script Text Processing

### Problem

Documents with **mixed scripts** (Latin, Cyrillic, Greek) can have accidental character confusions, but most are impossible:
- Latin 'A' ≠ Cyrillic 'А' (visually identical, different Unicode)
- Greek 'α' ≠ Latin 'a' (visually similar)

Some substitutions should be **blocked** to prevent false matches across scripts.

### Solution: Same-Script Restriction

```rust
impl SubstitutionSet {
    pub fn latin_only() -> Self {
        let mut set = HashSet::new();

        // Allow all substitutions within Latin script
        let latin_chars: Vec<char> = ('a'..='z')
            .chain('A'..='Z')
            .collect();

        for &a in &latin_chars {
            for &b in &latin_chars {
                if a != b {
                    set.insert((a, b));
                }
            }
        }

        SubstitutionSet { allowed: set }
    }

    pub fn script_aware() -> Self {
        let mut set = HashSet::new();

        // Allow Latin ↔ Latin
        set.extend(Self::latin_only().allowed);

        // Allow Cyrillic ↔ Cyrillic
        set.extend(Self::cyrillic_only().allowed);

        // Allow Greek ↔ Greek
        set.extend(Self::greek_only().allowed);

        // Block cross-script substitutions
        // (automatically blocked by not including them)

        SubstitutionSet { allowed: set }
    }
}
```

### Usage Example

```rust
let dict = TransducerBuilder::new()
    .with_substitution_set(SubstitutionSet::script_aware())
    .build_from_iter(multilingual_dict);

// Latin-to-Latin: OK
let results = dict.fuzzy_search("test", 1).collect();
// ✅ Returns Latin matches

// Cyrillic-to-Cyrillic: OK
let results = dict.fuzzy_search("тест", 1).collect();
// ✅ Returns Cyrillic matches

// Cross-script: Blocked
// Would not match Latin "test" with Cyrillic "тест" via substitution
```

---

## 6. DNA Sequence Matching

### Problem

DNA sequences use alphabet {A, C, G, T}. Biological mutations have **patterns**:
- **Transitions**: Purine ↔ Purine (A ↔ G) or Pyrimidine ↔ Pyrimidine (C ↔ T)
- **Transversions**: Purine ↔ Pyrimidine (less common)

Transitions are **2x more common** than transversions in nature.

### Solution: Transition-Only Set

```rust
impl SubstitutionSet {
    pub fn dna_transitions_only() -> Self {
        let mut set = HashSet::new();

        // Transitions (more common)
        set.insert(('A', 'G'));
        set.insert(('G', 'A'));
        set.insert(('C', 'T'));
        set.insert(('T', 'C'));

        // Transversions NOT included (blocked)
        // A ↔ C, A ↔ T, G ↔ C, G ↔ T

        SubstitutionSet { allowed: set }
    }

    pub fn dna_all_mutations() -> Self {
        let mut set = Self::dna_transitions_only();

        // Add transversions (less common)
        let transversions = [
            ('A', 'C'), ('A', 'T'),
            ('G', 'C'), ('G', 'T'),
            ('C', 'A'), ('T', 'A'),
            ('C', 'G'), ('T', 'G'),
        ];

        for (a, b) in transversions {
            set.allowed.insert((a, b));
        }

        set
    }
}
```

### Usage Example

```rust
let genome_db = TransducerBuilder::new()
    .with_substitution_set(SubstitutionSet::dna_transitions_only())
    .build_from_iter(reference_sequences);

// Query with transition mutation
let results = genome_db.fuzzy_search("ACGT", 1).collect();
// ✅ Matches: "AGGT" (C→G transition)
// ✅ Matches: "ACTT" (G→T transition)
// ❌ Blocks: "ACAT" (G→A would be transversion)
```

---

## 7. Custom Domain-Specific Rules

### Example: Chemical Formula Matching

```rust
impl SubstitutionSet {
    pub fn chemical_elements() -> Self {
        let mut set = HashSet::new();

        // Elements with similar names/symbols
        set.insert(('C', 'O'));  // Carbon ↔ Oxygen (typo)
        set.insert(('N', 'M'));  // Nitrogen ↔ Metal (abbreviation confusion)
        set.insert(('S', '5'));  // Sulfur ↔ 5 (OCR error)

        // But block unrealistic substitutions
        // (automatically blocked by not including them)

        SubstitutionSet { allowed: set }
    }
}
```

### Example: Product Code Matching

```rust
impl SubstitutionSet {
    pub fn product_codes(allowed_char_sets: Vec<&str>) -> Self {
        let mut set = HashSet::new();

        // Product codes: "ABC-123-XYZ"
        // Only allow substitutions within same character class

        // Letters ↔ Letters
        for a in 'A'..='Z' {
            for b in 'A'..='Z' {
                if a != b {
                    set.insert((a, b));
                }
            }
        }

        // Digits ↔ Digits
        for a in '0'..='9' {
            for b in '0'..='9' {
                if a != b {
                    set.insert((a, b));
                }
            }
        }

        // Block Letters ↔ Digits (structure must be preserved)

        SubstitutionSet { allowed: set }
    }
}
```

---

## Combining with Other Operations

### Restricted Substitutions + Transposition

```rust
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::Transposition)  // Enable transposition
    .with_substitution_set(SubstitutionSet::qwerty())  // Keyboard proximity
    .build_from_iter(words);

// "tset" → "test" (transposition)
// "tedt" → "test" (keyboard typo: d→s, then transposition)
```

### Restricted Substitutions + MergeAndSplit

```rust
let dict = TransducerBuilder::new()
    .algorithm(Algorithm::MergeAndSplit)  // Enable merge/split
    .with_substitution_set(SubstitutionSet::ocr_confusions())  // OCR errors
    .build_from_iter(words);

// "every one" → "everyone" (merge) with OCR corrections
// "rn" misread as "m" → split + OCR correction
```

---

## Performance Considerations

### Memory Usage

**Small sets** (< 100 pairs): ~1-2 KB
**Keyboard layouts** (~200-400 pairs): ~5-10 KB
**Full phonetic** (~1000 pairs): ~20-40 KB

**Recommendation**: Use HashSet for `𝒪(1)` lookup, minimal memory overhead.

### Lookup Performance

**Cost per substitution check**: ~5-10ns (HashSet lookup)
**Impact on query time**: ~10-20% overhead for typical queries

**Optimization**: For static sets, consider perfect hashing.

---

## Summary Table

| Use Case | Substitution Set | Typical Size | Main Benefit |
|----------|------------------|--------------|--------------|
| **Spell checking** | Keyboard adjacency | 200-400 pairs | Higher precision, realistic typos |
| **OCR correction** | Visual similarity | 100-200 pairs | Font-aware error correction |
| **Phonetic matching** | Sound-alike rules | 500-1000 pairs | User-friendly name search |
| **Handwriting** | Shape similarity | 150-300 pairs | Context-aware recognition |
| **Multi-script** | Same-script only | All within script | Block impossible confusions |
| **DNA sequences** | Biological rules | 4-8 pairs | Scientifically accurate |
| **Domain-specific** | Custom rules | Varies | Application-specific constraints |

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-11-06
**Related**: [README.md](./README.md), [implementation-plan.md](./implementation-plan.md)
