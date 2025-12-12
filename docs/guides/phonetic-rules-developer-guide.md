# Phonetic Rules Developer Guide

This guide explains how to write phonetic rewrite rules using LLev grammar and fuzzy regular expressions, how they integrate together, and how they work with Levenshtein automata for approximate string matching.

## Table of Contents

1. [Overview](#overview)
2. [Part 1: LLev Grammar](#part-1-llev-grammar)
3. [Part 2: Phonetic Regex](#part-2-phonetic-regex)
4. [Part 3: Integration](#part-3-integration)
5. [Part 4: Levenshtein Integration](#part-4-levenshtein-integration)
6. [Quick Reference](#quick-reference)

---

## Overview

The phonetic rules system provides a pipeline for transforming text based on phonetic patterns:

```
User Input
    ↓
┌─────────────────────────────────┐
│  Layer 1: Phonetic Normalization │ ← LLev Rules
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Layer 2: Fuzzy Dictionary Index │ ← Levenshtein Transducer
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Layer 3: Result Mapping         │ ← Normalized → Original
└─────────────────────────────────┘
    ↓
Output (matches with original forms)
```

**Key Benefits:**
- Phonetically similar words match even with different spellings
- Supports complex context-dependent transformations
- Formally verified for termination and bounded expansion
- Both ASCII (byte-level) and Unicode (character-level) support

---

## Part 1: LLev Grammar

LLev (Levenshtein Language) is a domain-specific language for defining phonetic rewrite rules. Files use the `.llev` extension.

### File Structure

```llev
// File metadata (optional)
@name "English Phonetic Rules"
@version "1.0"
@author "Your Name"
@description "Rules for English phonetic normalization"

// Symbol definitions
@define VOWEL = [aeiou]
@define FRONT_VOWEL = [ei]
@define CONSONANT = [bcdfghjklmnpqrstvwxyz]

// Rules with optional metadata
[id: 1, name: "ph to f", weight: 0.0, group: orthography]
ph -> f;

[id: 2, name: "silent final e", weight: 0.0]
e -> / _#;
```

### Directives

| Directive | Description | Example |
|-----------|-------------|---------|
| `@name` | Rule set name | `@name "English Rules"` |
| `@version` | Version string | `@version "1.0.0"` |
| `@author` | Author name | `@author "Jane Doe"` |
| `@description` | Description | `@description "..."` |
| `@include` | Include another file | `@include "common.llev"` |
| `@define` | Define a symbol | `@define VOWEL = [aeiou]` |

### Symbol Definitions

Symbols allow reuse of patterns across rules:

```llev
// Define character classes
@define VOWEL = [aeiou]
@define FRONT_VOWEL = [ei]
@define CONSONANT = [bcdfghjklmnpqrstvwxyz]

// Define patterns
@define CLUSTER = (st|sp|sk|str|spr)

// Use symbols with $ sigil
c -> s / _$FRONT_VOWEL;     // c → s before e or i
$CLUSTER -> / #_;           // delete initial clusters (disabled by pattern)
```

**Symbol Naming Convention:**
- User-defined symbols: `UPPERCASE` (e.g., `$VOWEL`, `$MY_CLASS`)
- Built-in classes: `lowercase` (e.g., `[:vowel:]`, `[:consonant:]`)

### Rewrite Rules

Basic syntax: `pattern -> replacement;`

```llev
// Simple substitution
ph -> f;           // ph → f (phone → fone)
ch -> ts;          // ch → ts

// Deletion (empty replacement)
gh -> ;            // delete gh (night → nit)
e -> / _#;         // delete final e (make → mak)

// Expansion
x -> ks;           // x → ks (box → boks)
```

### Context Expressions

Contexts specify where rules apply using `/` followed by position markers:

```llev
// Right context only (lookahead)
c -> s / _[ei];          // c → s before e or i (city → sity)

// Left context only (lookbehind)
s -> z / [aeiou]_;       // s → z after vowel

// Both contexts
t -> d / [aeiou]_[aeiou]; // t → d between vowels

// Word boundaries
e -> / _#;               // delete e at word end
wr -> r / #_;            // wr → r at word start
#cat# -> dog;            // replace whole word "cat"
```

**Context Operators:**

| Operator | Precedence | Description |
|----------|------------|-------------|
| `\|` | Lowest | OR - either context matches |
| `&` | Medium | AND - both contexts must match |
| `!` | Highest | NOT - context must not match |

```llev
// OR: before 'e' or 'i'
c -> s / _([e]|[i]);

// AND: after vowel AND before consonant
x -> z / [aeiou]&_[bcdfg];

// NOT: before anything except 'e' or 'i'
c -> k / _![ei];

// Complex: after vowel AND (before consonant OR word end)
t -> d / [aeiou]_([bcdfg]|#);
```

### Syllable Conditions

Add syllable-based constraints with `if`:

```llev
// Only in monosyllabic words
y -> i / _# if monosyllable;

// Only in polysyllabic words, final syllable
y -> i / _# if polysyllable & final_syllable;

// Complex condition
a -> aa / _ if open_syllable | initial_syllable;
```

**Available Conditions:**
- `monosyllable` - word has exactly one syllable
- `polysyllable` - word has more than one syllable
- `open_syllable` - current syllable ends with vowel
- `closed_syllable` - current syllable ends with consonant
- `final_syllable` - match is in the last syllable
- `initial_syllable` - match is in the first syllable

### Metadata Blocks

Add metadata to rules for documentation and control:

```llev
[id: 1, name: "ph to f", weight: 0.0, group: orthography, enabled: true]
ph -> f;

[id: 2, name: "disabled rule", enabled: false]
x -> y;  // This rule won't be applied
```

**Metadata Fields:**
- `id` - Unique rule identifier (integer)
- `name` - Human-readable name (string)
- `weight` - Priority weight (float, 0.0 = highest)
- `group` - Grouping category (string)
- `enabled` - Whether rule is active (boolean)

### Comments

```llev
// Line comment (preferred)
ph -> f;  // inline comment

/* Block comment
   can span multiple lines */

/* Nested /* block comments */ are supported */
```

**Note:** `#` is NOT a comment character—it's the word boundary marker.

### Escape Shortcuts

Escape sequences that expand to character classes. Both LLev and phonetic regex support these shortcuts.

**Convention:** Lowercase = positive match, Uppercase = negated match

#### Standard Regex Shortcuts

| Shortcut | Class | Negated | Class |
|----------|-------|---------|-------|
| `\d` | digit (0-9) | `\D` | non-digit |
| `\w` | word (a-z, A-Z, 0-9, _) | `\W` | non-word |
| `\s` | whitespace | `\S` | non-whitespace |

#### Phonetic Class Shortcuts

| Shortcut | Class | Negated | Class |
|----------|-------|---------|-------|
| `\v` | vowel | `\V` | non-vowel |
| `\c` | consonant | `\C` | non-consonant |
| `\f` | front_vowel | `\F` | non-front_vowel |
| `\k` | back_vowel | `\K` | non-back_vowel |
| `\h` | high_vowel | `\H` | non-high_vowel |
| `\l` | low_vowel | `\L` | non-low_vowel |
| `\m` | mid_vowel | `\M` | non-mid_vowel |
| `\o` | voiced | `\O` | voiceless |
| `\e` | fricative | `\E` | non-fricative |
| `\a` | affricate | `\A` | non-affricate |
| `\p` | stop/plosive | `\P` | non-stop |
| `\g` | glide | `\G` | non-glide |
| `\z` | nasal | `\Z` | non-nasal |
| `\q` | liquid | `\Q` | non-liquid |

**Note:** Uppercase letters used as shortcuts (`A`, `C`, `D`, `E`, `F`, `G`, `H`, `K`, `L`, `M`, `O`, `P`, `Q`, `S`, `V`, `W`, `Z`) cannot be escaped for literal use. Use quoted strings `"A"` or non-shortcut letters (`\B`, `\I`, `\J`, `\N`, `\R`, `\T`, `\X`, `\Y`) for literal uppercase characters.

```llev
// Using shortcuts
x -> gz / \v_\v;     // x → gz between vowels
t -> d / \v_\v;      // t → d between vowels
c -> s / _\f;        // c → s before front vowel
\o -> \O / _#;       // devoice voiced consonants word-finally
\d+ -> NUM;          // normalize digit sequences
```

---

## Part 2: Phonetic Regex

Phonetic regex extends standard regular expressions with features for phonetic pattern matching.

### Basic Operators

| Operator | Description | Example | Matches |
|----------|-------------|---------|---------|
| `\|` | Alternation | `cat\|dog` | "cat" or "dog" |
| `*` | Zero or more | `ab*c` | "ac", "abc", "abbc", ... |
| `+` | One or more | `ab+c` | "abc", "abbc", "abbbc", ... |
| `?` | Optional | `colou?r` | "color" or "colour" |
| `.` | Any character | `c.t` | "cat", "cot", "cut", ... |
| `(...)` | Grouping | `(ab)+` | "ab", "abab", ... |

### Quantifiers

| Syntax | Description | Example |
|--------|-------------|---------|
| `{n}` | Exactly n | `a{3}` matches "aaa" |
| `{n,}` | At least n | `a{2,}` matches "aa", "aaa", ... |
| `{,m}` | At most m | `a{,3}` matches "", "a", "aa", "aaa" |
| `{n,m}` | Between n and m | `a{2,4}` matches "aa", "aaa", "aaaa" |

### Operator Precedence

From lowest to highest:
1. **Alternation** (`|`) - lowest
2. **Concatenation** (implicit)
3. **Quantifiers** (`*`, `+`, `?`, `{n,m}`) - highest

```
a|bc*   = a | (b(c*))       // alternation binds loosest
ab+c    = a(b+)c            // quantifier binds to 'b'
(ab)+c  = ((ab)+)c          // use groups to change precedence
```

### Character Classes

```regex
[abc]       // matches 'a', 'b', or 'c'
[a-z]       // matches any lowercase letter
[A-Za-z]    // matches any letter
[^abc]      // matches anything except 'a', 'b', 'c'
[a-z0-9]    // matches letter or digit
[-abc]      // literal dash at start
[abc-]      // literal dash at end
```

### Named Character Classes

POSIX-style named classes for common character sets:

```regex
[:vowel:]       // vowels (a, e, i, o, u + IPA)
[:consonant:]   // consonants
[:alpha:]       // alphabetic characters
[:digit:]       // digits 0-9
[:alnum:]       // alphanumeric

// Usage in character class
[[:vowel:]y]    // vowels plus 'y'
[[:digit:].-]   // digits, dot, or dash
[^[:vowel:]]    // anything except vowels
```

**Available Classes:**

| Category | Classes |
|----------|---------|
| POSIX | `alpha`, `digit`, `alnum`, `space`, `punct`, `ascii`, `print`, `lower`, `upper`, `xdigit`, `word`, `blank` |
| Vowels | `vowel`, `front_vowel`, `back_vowel`, `high_vowel`, `low_vowel`, `mid_vowel`, `central_vowel`, `ascii_vowel` |
| Consonants | `consonant`, `stop`/`plosive`, `fricative`, `nasal`, `liquid`, `glide`/`semivowel`, `affricate`, `approximant` |
| Manner | `voiced`, `voiceless` |
| Place | `labial`, `dental`, `alveolar`, `palatal`, `velar`, `glottal` |

### Feature Bundles

Feature bundles allow you to specify the **intersection** of multiple phonetic features using space-separated syntax. This is essential for targeting specific sound classes like "voiced stops" or "high front vowels."

**Syntax:** `[:feature1 feature2 ...]:`

```llev
// Intersection: voiced AND stop → b, d, g
[:voiced stop:]

// Negation + intersection: (NOT nasal) AND stop → p, t, k, b, d, g
[:!nasal stop:]

// Multiple features: high AND front AND vowel → i, ɪ, y, ʏ
[:high front vowel:]

// Negation alone: anything NOT nasal
[:!nasal:]
```

**Key Semantics:**

| Syntax | Meaning | Result |
|--------|---------|--------|
| `[:a b:]` | a ∩ b | Characters with BOTH features |
| `[:!a:]` | ¬a | Characters WITHOUT feature a |
| `[:!a b:]` | (¬a) ∩ b | Characters with b but NOT a |
| `[[:a:][:b:]]` | a ∪ b | Characters with EITHER feature (union) |

**Why Feature Bundles Matter:**

The nested bracket syntax `[[^[:nasal:]][:stop:]]` creates a **union**: (not nasals) ∪ (stops) = almost everything. Feature bundles with space separation create an **intersection**: `[:!nasal stop:]` = (not nasal) ∩ (stop) = only oral stops.

**Practical Examples:**

```llev
// Phonological rules using feature bundles
[:voiced stop:] -> [:voiceless stop:] / _#;  // final devoicing: b→p, d→t, g→k
[:!nasal stop:] -> [:nasal:] / _[:nasal:];   // nasal assimilation
[:high front vowel:] -> [:mid front vowel:] / _[:consonant:][:consonant:];  // vowel lowering
```

### Symbol References

Reference user-defined symbols from LLev:

```regex
$VOWEL          // simple form
${FRONT_VOWEL}  // braced form (for clarity)

// In patterns
c -> s / _$FRONT_VOWEL;
[$VOWEL$CONSONANT]   // union of both classes
```

### Word Boundary

The `#` marker indicates word boundaries:

```regex
#abc        // "abc" at word start
abc#        // "abc" at word end
#abc#       // "abc" as complete word

// In context
/ #_        // at word start
/ _#        // at word end
```

### Escape Sequences

| Escape | Description |
|--------|-------------|
| `\n` | Newline |
| `\r` | Carriage return |
| `\t` | Tab |
| `\0` | Null |
| `\\` | Literal backslash |
| `\[`, `\]`, etc. | Literal special characters |
| `\xNN` | Hex byte (2 digits) |
| `\uNNNN` | Unicode (4 digits) |
| `\UNNNNNNNN` | Unicode (8 digits) |

---

## Part 3: Integration

### Loading and Parsing Rules

```rust
use liblevenshtein::phonetic::llev::{parse_str, RuleSetChar};

// Parse from string
let file = parse_str(r#"
    @define VOWEL = [aeiou]
    ph -> f;
    c -> s / _[ei];
"#)?;

// Convert to executable rules
let ruleset = RuleSetChar::from_llev(&file)?;

// Apply rules to text
let normalized = ruleset.apply("phone");  // → "fone"
```

### Byte-Level vs Character-Level

**`RuleSet` (Byte-Level):**
- ASCII only (0-127)
- ~5% faster, 4× less memory
- Best for pure ASCII text

**`RuleSetChar` (Character-Level):**
- Full Unicode support
- Required for IPA symbols, accented characters
- Use for internationalized text

```rust
// Byte-level (ASCII only)
let ruleset_byte = RuleSet::from_llev(&file)?;

// Character-level (Unicode)
let ruleset_char = RuleSetChar::from_llev(&file)?;

// IPA example (requires character-level)
let file = parse_str(r#"
    th -> θ;    // theta
    sh -> ʃ;    // esh
    ng -> ŋ;    // eng
"#)?;
let ruleset = RuleSetChar::from_llev(&file)?;
```

### Rule Application Pipeline

Rules are applied sequentially with "fuel" to ensure termination:

```rust
// Sequential application (default)
let result = ruleset.apply("enough");

// With custom fuel limit (max iterations)
let result = ruleset.apply_with_fuel("enough", 100);
```

**Verified Properties:**
1. **Termination** - Always reaches a fixed point
2. **Bounded Expansion** - Output ≤ input + 20 characters
3. **Idempotence** - Reapplication produces same result

### Loading from Files

```rust
use liblevenshtein::phonetic::llev::{parse_file, RuleSetChar};
use std::path::Path;

// Load single file
let file = parse_file(Path::new("rules/english.llev"))?;
let ruleset = RuleSetChar::from_llev(&file)?;

// Load directory of rules
fn load_rules_from_dir(dir: &Path) -> Result<RuleSetChar, Box<dyn Error>> {
    let mut combined = RuleSetChar::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension() == Some(OsStr::new("llev")) {
            let file = parse_file(&path)?;
            let ruleset = RuleSetChar::from_llev(&file)?;
            combined.merge(ruleset);
        }
    }
    Ok(combined)
}
```

### NFA Compilation for Complex Patterns

Simple patterns (literal sequences) use direct matching. Complex patterns with quantifiers or alternation compile to NFAs:

```rust
use liblevenshtein::phonetic::nfa::compile_rule;

// Simple pattern - direct matching
// ph -> f

// Complex pattern - NFA compiled
// (ph|f)one -> fon
// [aeiou]+ -> V

// The conversion happens automatically in RuleSet::from_llev()
```

---

## Part 4: Levenshtein Integration

### Architecture Overview

The system combines phonetic normalization with Levenshtein distance for robust fuzzy matching:

```
Query: "enuf"
    ↓
┌──────────────────────────────┐
│ Normalize query with rules   │  "enuf" → "enuf"
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│ Search normalized dictionary │  Find "enuf" (distance 0)
│ using Levenshtein transducer │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│ Map back to original forms   │  "enuf" → "enough"
└──────────────────────────────┘
    ↓
Result: "enough" (distance 0)
```

### Building a Phonetic Index

```rust
use liblevenshtein::phonetic::llev::{parse_file, RuleSetChar};
use liblevenshtein::transducer::Transducer;
use std::collections::HashMap;

fn build_phonetic_index(
    dictionary: &[String],
    rules: &RuleSetChar,
) -> (HashMap<String, Vec<String>>, Transducer) {
    // Build normalized → original mapping
    let mut normalized_to_original: HashMap<String, Vec<String>> = HashMap::new();
    let mut normalized_terms = Vec::new();

    for term in dictionary {
        let normalized = rules.apply(term);
        normalized_to_original
            .entry(normalized.clone())
            .or_default()
            .push(term.clone());
        normalized_terms.push(normalized);
    }

    // Build transducer on normalized forms
    let transducer = Transducer::new(&normalized_terms, Algorithm::Transposition);

    (normalized_to_original, transducer)
}
```

### Querying with Phonetic Normalization

```rust
fn phonetic_search(
    query: &str,
    rules: &RuleSetChar,
    transducer: &Transducer,
    normalized_to_original: &HashMap<String, Vec<String>>,
    max_distance: usize,
) -> Vec<(String, usize)> {
    // Normalize the query
    let normalized_query = rules.apply(query);

    // Search the transducer
    let mut results = Vec::new();
    for candidate in transducer.query(&normalized_query, max_distance) {
        // Map back to original forms
        if let Some(originals) = normalized_to_original.get(&candidate.term) {
            for original in originals {
                results.push((original.clone(), candidate.distance));
            }
        }
    }

    results.sort_by_key(|(_, d)| *d);
    results
}
```

### Complete Example

See `examples/phonetic_spellcheck/` for a complete working example:

```rust
// From examples/phonetic_spellcheck/src/main.rs

fn main() -> Result<(), Box<dyn Error>> {
    // Load dictionary
    let dictionary = load_dictionary("./data/english_words.txt")?;

    // Load phonetic rules
    let rules = load_rules("./rules")?;

    // Build index
    let (normalized_to_original, transducer) = build_index(&dictionary, &rules)?;

    // Interactive query loop
    loop {
        print!("query> ");
        let query = read_line()?;

        let normalized = rules.apply(&query);
        let candidates = transducer.query(&normalized, MAX_DISTANCE);

        for candidate in candidates {
            if let Some(originals) = normalized_to_original.get(&candidate.term) {
                for original in originals {
                    println!("  {} (distance: {})", original, candidate.distance);
                }
            }
        }
    }
}
```

### Phonetic Operation Types

The transducer supports specialized phonetic operations beyond standard Levenshtein:

```rust
use liblevenshtein::transducer::phonetic::PhoneticOperations;

// Create transducer with phonetic operations
let ops = PhoneticOperations::english();
let transducer = Transducer::with_operations(&dictionary, ops);
```

**Built-in Operations:**

| Category | Operations | Cost |
|----------|------------|------|
| Consonant Digraphs | ch↔k, sh↔s, ph↔f, th↔t | 0.15 |
| Initial Clusters | wr↔r, wh↔w, kn↔n, ps↔s | 0.20 |
| Phonetic Confusions | c↔k, c↔s, s↔z, g↔j, f↔v | 0.25 |
| Double Consonants | bb↔b, dd↔d, ff↔f, etc. | 0.10 |

### Performance Characteristics

| Aspect | Byte-Level | Character-Level |
|--------|------------|-----------------|
| Speed | ~5% faster | Baseline |
| Memory | 4× less | Baseline |
| Unicode | No | Yes |
| IPA | No | Yes |

**Recommendations:**
- Use byte-level for ASCII-only dictionaries
- Use character-level for Unicode/IPA support
- Pre-normalize dictionary at build time
- Cache normalized forms for repeated queries

---

## Quick Reference

### LLev Syntax Summary

```llev
// Directives
@name "Name"
@version "1.0"
@define SYMBOL = [pattern]

// Rules
pattern -> replacement;
pattern -> replacement / context;
pattern -> replacement / context if syllable;

// Metadata
[id: 1, name: "rule", weight: 0.0, group: "cat", enabled: true]

// Contexts
/ _X        // before X
/ X_        // after X
/ X_Y       // between X and Y
/ #_        // word start
/ _#        // word end
/ X|Y_      // X or Y before
/ X&Y_      // X and Y before (both must match)
/ !X_       // not X before

// Syllables
if monosyllable
if polysyllable
if final_syllable
if initial_syllable
if open_syllable
if closed_syllable

// Feature bundles (intersection of features)
[:voiced stop:]       voiced AND stop → b, d, g
[:!nasal stop:]       NOT nasal AND stop → p, t, k, b, d, g
[:high front vowel:]  high AND front AND vowel
[[:stop:][:fricative:]] stop OR fricative (union)
```

### Regex Syntax Summary

```regex
// Basic
abc         literal sequence
a|b         alternation
(ab)        grouping
.           any character
#           word boundary

// Quantifiers
a*          zero or more
a+          one or more
a?          optional
a{3}        exactly 3
a{2,}       2 or more
a{,3}       at most 3
a{2,4}      between 2 and 4

// Character classes
[abc]       any of a, b, c
[a-z]       range
[^abc]      negation
[:vowel:]   named class
$SYMBOL     user symbol

// Feature bundles (intersection)
[:voiced stop:]       voiced AND stop
[:!nasal stop:]       NOT nasal AND stop
[:high front vowel:]  high AND front AND vowel
[[:a:][:b:]]          union (a OR b)

// Escapes
\n \r \t    whitespace
\xNN        hex byte
\uNNNN      unicode
\\          literal backslash
```

### Common Phonetic Rules

```llev
// Digraphs
ph -> f;
ch -> ts;
sh -> s;
th -> t;

// Initial clusters
wr -> r / #_;
wh -> w / #_;
kn -> n / #_;
gn -> n / #_;

// Contextual
c -> s / _[ei];   // soft c
c -> k;           // hard c (default)
g -> j / _[ei];   // soft g

// Silent letters
e -> / _#;        // silent final e
gh -> ;           // silent gh

// Double consonants
bb -> b;
dd -> d;
ff -> f;
// etc.
```

---

## See Also

- [EBNF Grammar: LLev](../grammar/llev.ebnf)
- [EBNF Grammar: Regex](../grammar/regex.ebnf)
- [Example: Phonetic Spellcheck](../../examples/phonetic_spellcheck/)
- [Zompist Rules](../../examples/phonetic_spellcheck/rules/zompist.llev)
