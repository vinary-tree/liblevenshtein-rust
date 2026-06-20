# Phonetic Rules Developer Guide

This guide explains how to write phonetic rewrite rules using LLev grammar and fuzzy regular expressions, how they integrate together, and how they work with Levenshtein automata for approximate string matching.

## Table of Contents

1. [Overview](#overview)
2. [Part 1: LLev Grammar](#part-1-llev-grammar)
3. [Part 2: Phonetic Regex](#part-2-phonetic-regex)
4. [Part 3: Integration](#part-3-integration)
5. [Part 4: Levenshtein Integration](#part-4-levenshtein-integration)
6. [Part 5: Advanced Pattern Matching](#part-5-advanced-pattern-matching)
7. [Part 6: LLRE File Format](#part-6-llre-file-format)
8. [Part 7: Serialization & AOT](#part-7-serialization--aot)
9. [Part 8: Advanced Topics](#part-8-advanced-topics)
10. [Quick Reference](#quick-reference)

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

> **Terminology.** A **phoneme** is a contrastive unit of sound. A **distinctive (articulatory) feature** is one dimension of a sound — its *place of articulation* (where the vocal tract is constricted), its *manner of articulation* (how airflow is shaped, e.g. *stop*, *fricative*, *nasal*), or its *voicing* (whether the vocal folds vibrate). The **International Phonetic Alphabet (IPA)** assigns one symbol per sound; symbols such as `ʃ` ("sh"), `θ` ("th" in *thin*), `ŋ` ("ng"), `ɲ` (palatal nasal), and `ʎ` (palatal lateral) appear in the rules below. The empty string is written `ε`. See [`../../docs/GLOSSARY.md`](../GLOSSARY.md) for fuller definitions.

The LLev → NFA compilation pipeline these rules drive is shown below:

![LLev compilation pipeline: an LLev rewrite rule is parsed to an AST, lowered through Thompson construction into an NFA, optimized (epsilon-elimination and dead-state removal), and emitted as a matcher used during phonetic normalization.](../diagrams/phonetic/llev-compilation.svg)

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

**Note:** Standard regex word boundary escapes `\b` and `\B` are not supported. Use `#` for word boundaries instead.

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
| `[:a b:]` | `a ∩ b` | Characters with BOTH features |
| `[:!a:]` | `¬a` | Characters WITHOUT feature a |
| `[:!a b:]` | `(¬a) ∩ b` | Characters with b but NOT a |
| `[[:a:][:b:]]` | `a ∪ b` | Characters with EITHER feature (union) |

**Why Feature Bundles Matter:**

The nested bracket syntax `[[^[:nasal:]][:stop:]]` creates a **union**: `(¬nasal) ∪ stop` = almost everything. Feature bundles with space separation create an **intersection**: `[:!nasal stop:]` = `(¬nasal) ∩ stop` = only oral stops.

**Practical Examples:**

```llev
// Phonological rules using feature bundles
[:voiced stop:] -> [:voiceless stop:] / _#;  // final devoicing: b→p, d→t, g→k
[:!nasal stop:] -> [:nasal:] / _[:nasal:];   // nasal assimilation
[:high front vowel:] -> [:mid front vowel:] / _[:consonant:][:consonant:];  // vowel lowering
```

### Symbol References

Reference user-defined symbols from LLev (outside character classes only):

```regex
$VOWEL          // simple form - expands to vowel chars
${FRONT_VOWEL}  // braced form (for clarity)

// In context patterns
c -> s / _$FRONT_VOWEL;

// Inside character classes, use POSIX named class syntax:
[[:vowel:][:consonant:]]  // union of built-in classes

// $ is a LITERAL character inside character classes:
[$abc]  // matches literal '$', 'a', 'b', 'c'
```

### Character Class Negation

Negation follows De Morgan's laws for properly nested classes:

| Pattern | Equivalent | Description |
|---------|------------|-------------|
| `[^[:vowel:]]` | all non-vowels | Simple negation |
| `[^[:vowel:][:voiced:]]` | `¬(vowel ∪ voiced)` | Negate the union |
| `[^[^[:vowel:]]]` | `[:vowel:]` | Double negation cancels out |
| `[:!nasal stop:]` | `(¬nasal) ∩ stop` | Inner negation with intersection |

**De Morgan's Laws:**
- `¬(A ∪ B) = ¬A ∩ ¬B`
- `¬(A ∩ B) = ¬A ∪ ¬B`

For feature bundles, `!` inside the bundle negates individual features before intersection:
- `[:!nasal stop:]` = (NOT nasal) AND (stop) = oral stops only

### Word Boundary

The `#` marker indicates word boundaries:

```regex
#abc        // "abc" at word start
abc#        // "abc" at word end
#abc#       // "abc" as complete word

// In context
/ #_        // at word start
/ _#        // at word end
/ #_#       // complete word (both boundaries)
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

### NFA Optimization

Compiled NFAs are automatically optimized to reduce size and improve matching
performance. Optimization removes epsilon transitions, unreachable states, and
dead states.

```rust
use liblevenshtein::phonetic::nfa::{compile, OptimizationConfig, NFACompilerChar};
use liblevenshtein::phonetic::regex::parse;

// Optimization is enabled by default
let regex = parse("(ph|f)one")?;
let nfa = compile(&regex)?;

// View optimization statistics
let mut compiler = NFACompilerChar::new().without_optimization();
let unoptimized = compiler.compile(&regex)?;
let (optimized, stats) = unoptimized.optimize_with(OptimizationConfig::full());
println!("States: {} → {}", stats.original_states, stats.final_states);
println!("Epsilon transitions eliminated: {}", stats.epsilon_transitions_eliminated);

// Configuration presets:
// - OptimizationConfig::full()  - All passes (default)
// - OptimizationConfig::quick() - Remove unreachable/dead only
// - OptimizationConfig::none()  - No optimization

// Disable optimization for debugging
let mut compiler = NFACompilerChar::new().without_optimization();
let unoptimized = compiler.compile(&regex)?;
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

## Part 5: Advanced Pattern Matching

Beyond basic NFA matching, the phonetic module provides specialized matchers for different use cases: lazy DFA construction, fuzzy regex matching, cached matching, and streaming input.

### 5.1 Lazy DFA

The `LazyDFAChar` provides on-demand DFA state construction. Instead of computing the full powerset construction upfront, states are created only when needed during matching.

```rust
use liblevenshtein::phonetic::nfa::{compile, LazyDFAChar};
use liblevenshtein::phonetic::regex::parse;

// Parse and compile the pattern
let regex = parse("(ph|f)one")?;
let nfa = compile(&regex)?;

// Create lazy DFA - no upfront computation
let mut lazy_dfa = LazyDFAChar::new(nfa);

// States are constructed on-demand during matching
assert!(lazy_dfa.accepts("phone"));
assert!(lazy_dfa.accepts("fone"));
assert!(!lazy_dfa.accepts("bone"));

// View cache statistics
let stats = lazy_dfa.cache_stats();
println!("Cache size: {}", lazy_dfa.cache_size());
println!("Cache hits: {}, misses: {}", stats.hits, stats.misses);

// Clear cache if memory is a concern
lazy_dfa.clear_cache();
```

**When to use Lazy DFA:**
- Large patterns where full DFA would be memory-prohibitive
- Patterns with many branches that are rarely all exercised
- Applications with diverse inputs (cache benefits from locality)
- Startup-time sensitive applications (no upfront computation)

### 5.2 Product Automaton

The `ProductAutomatonChar` combines an NFA with a Levenshtein automaton, enabling fuzzy regex matching—find strings that match a pattern within a specified edit distance.

```rust
use liblevenshtein::phonetic::nfa::{compile, ProductAutomatonChar};
use liblevenshtein::phonetic::regex::parse;

let regex = parse("(ph|f)one")?;
let nfa = compile(&regex)?;

// Create product automaton with max edit distance 2
let product = ProductAutomatonChar::new(nfa, 2);

// Exact match - distance 0
assert!(product.accepts("phone"));
assert_eq!(product.min_distance("phone"), Some(0));

// Fuzzy match - distance 1
assert!(product.accepts("phon"));   // deletion
assert!(product.accepts("pphone")); // insertion
assert_eq!(product.min_distance("phon"), Some(1));

// Fuzzy match - distance 2
assert!(product.accepts("ph"));
assert_eq!(product.min_distance("ph"), Some(2));

// Beyond threshold - rejected
assert!(!product.accepts("x"));
assert_eq!(product.min_distance("xyz"), None);
```

**Phonetic Weighting:**

```rust
// Phonetically-weighted product automaton
let product = ProductAutomatonChar::new(nfa, 2)
    .with_phonetic_weight(0.5);  // Phonetic substitutions cost 0.5

// ph→f is phonetically similar, costs less than arbitrary substitution
```

**Use cases:**
- Spell checking with pattern constraints
- Fuzzy search in structured data (e.g., phone numbers, codes)
- OCR error correction with format validation

### 5.3 Memoized Matching

The `MemoizedMatcherChar` wraps a product automaton with an LRU cache, ideal for repeated queries over the same inputs.

```rust
use liblevenshtein::phonetic::nfa::{
    compile, ProductAutomatonChar, MemoizedMatcherChar
};
use liblevenshtein::phonetic::regex::parse;

let regex = parse("(ph|f)one")?;
let nfa = compile(&regex)?;
let product = ProductAutomatonChar::new(nfa, 2);

// Create memoized matcher with LRU cache (1000 entries)
let mut matcher = MemoizedMatcherChar::new(product, 1000);

// First query - computed and cached
assert!(matcher.accepts("phone"));

// Repeated query - cache hit
assert!(matcher.accepts("phone"));

// Distance queries are also cached
assert_eq!(matcher.min_distance("phone"), Some(0));

// View cache statistics
let stats = matcher.stats();
println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);

// Clear cache if needed
matcher.clear();
```

**When to use memoization:**
- Spell checking with common misspellings
- Autocomplete with frequently typed prefixes
- Batch processing with duplicate inputs
- Interactive applications with repeated queries

### 5.4 Incremental/Streaming Matching

The `IncrementalMatcherChar` processes input character-by-character, enabling real-time feedback during typing.

```rust
use liblevenshtein::phonetic::nfa::{compile, IncrementalMatcherChar};
use liblevenshtein::phonetic::regex::parse;

let regex = parse("(ph|f)one")?;
let nfa = compile(&regex)?;

let mut matcher = IncrementalMatcherChar::new(nfa);

// Feed characters one at a time
matcher.feed('p');
assert!(!matcher.is_accepting());  // "p" doesn't match yet
assert!(!matcher.is_dead());       // But matching could still succeed

matcher.feed('h');
matcher.feed('o');
matcher.feed('n');
matcher.feed('e');
assert!(matcher.is_accepting());   // "phone" matches!

// Feed a string all at once
matcher.reset();
matcher.feed_str("fone");
assert!(matcher.is_accepting());

// Snapshot for backtracking
let snapshot = matcher.snapshot();

matcher.feed('s');
assert!(!matcher.is_accepting());  // "fones" doesn't match

// Restore to previous state
matcher.restore(snapshot);
assert!(matcher.is_accepting());   // Back to "fone"
```

**Use cases:**
- Real-time validation in text editors
- Autocomplete with live feedback
- Streaming input from networks or pipes
- Undo/redo with snapshot/restore

### 5.5 Choosing the Right Matcher

| Matcher | Memory | Startup | Per-Query | Best For |
|---------|--------|---------|-----------|----------|
| `NFAChar` (direct) | Low | Fast | Slow | Simple patterns, one-off matching |
| `LazyDFAChar` | Medium | Fast | Fast* | Large patterns, diverse inputs |
| `ProductAutomatonChar` | High | Medium | Medium | Fuzzy regex matching |
| `MemoizedMatcherChar` | High | Medium | Very Fast** | Repeated queries |
| `IncrementalMatcherChar` | Low | Fast | Very Fast | Streaming, real-time feedback |

\* After cache warmup
\** For cached inputs

---

## Part 6: LLRE File Format

LLRE (Levenshtein Regular Expression) files define single regex patterns with metadata and imports. They complement LLev files which define rewrite rules.

### 6.1 Format Overview

```llre
@name "Phone Pattern"
@version "1.0"
@import "phonetic-symbols.llev"  # Import symbols from LLev file
@flags case_insensitive

# The regex pattern (only one per file)
(ph|f)one
```

### 6.2 Directives

| Directive | Description | Example |
|-----------|-------------|---------|
| `@name` | Pattern name for documentation | `@name "US Phone"` |
| `@version` | Semantic version | `@version "1.0.0"` |
| `@import` | Import symbols from LLev file | `@import "symbols.llev"` |
| `@flags` | Space-separated flags | `@flags multiline dotall` |

### 6.3 Flags

| Flag | Effect |
|------|--------|
| `multiline` | `^` and `$` match line boundaries, not just string boundaries |
| `dotall` | `.` matches newlines (normally excluded) |
| `case_insensitive` | Case-insensitive matching |
| `unicode` | Full Unicode character class support |
| `greedy` | Quantifiers are greedy by default (standard) |
| `lazy` | Quantifiers are lazy by default |

### 6.4 Loading LLRE Files

```rust
use liblevenshtein::phonetic::llre::{load_llre_file, compile_llre};
use liblevenshtein::phonetic::nfa::LazyDFAChar;

// Load and parse LLRE file
let file = load_llre_file("patterns/phone.llre")?;

// Access metadata
println!("Pattern: {}", file.name.unwrap_or("unnamed"));
println!("Version: {}", file.version.unwrap_or("0.0.0"));

// Compile to NFA
let nfa = compile_llre(&file)?;

// Use with any matcher
let mut dfa = LazyDFAChar::new(nfa);
assert!(dfa.accepts("phone"));
```

### 6.5 Symbol Imports

LLRE files can import symbols defined in LLev files:

```llev
// phonetic-symbols.llev
@define VOWEL = [aeiou]
@define CONSONANT = [bcdfghjklmnpqrstvwxyz]
@define DIGIT = [0-9]
```

```llre
// pattern.llre
@import "phonetic-symbols.llev"

# Use imported symbols
$CONSONANT+$VOWEL+$CONSONANT*
```

---

## Part 7: Serialization & AOT

For production deployments, pre-compile rules and patterns to binary format for instant loading.

> **Note:** Requires the `serialization` feature flag.

### 7.1 LLev Rule Serialization

```rust
use liblevenshtein::phonetic::llev::{
    RuleSetChar, save_char, load_char
};

// Compile rules from LLev file (done at build time)
let ruleset = RuleSetChar::from_file("rules/english.llev")?;

// Save to binary format
save_char(&ruleset, "rules/english.bin")?;

// At runtime: load instantly (no parsing)
let ruleset = load_char("rules/english.bin")?;

// Use normally
let normalized = ruleset.apply("knight");
```

### 7.2 NFA Serialization

```rust
use liblevenshtein::phonetic::llre::{
    save_compiled_llre, load_compiled_llre
};
use liblevenshtein::phonetic::nfa::{compile, LazyDFAChar};
use liblevenshtein::phonetic::regex::parse;

// Compile pattern (done at build time)
let regex = parse("(ph|f)one")?;
let nfa = compile(&regex)?;

// Save compiled NFA
save_compiled_llre(&nfa, "patterns/phone.nfa.bin")?;

// At runtime: load pre-compiled NFA
let nfa = load_compiled_llre("patterns/phone.nfa.bin")?;
let mut dfa = LazyDFAChar::new(nfa);
```

### 7.3 Build Script Integration

```rust
// build.rs
use liblevenshtein::phonetic::llev::{RuleSetChar, save_char};
use liblevenshtein::phonetic::llre::{load_llre_file, compile_llre, save_compiled_llre};

fn main() {
    // Compile LLev rules
    let ruleset = RuleSetChar::from_file("rules/english.llev")
        .expect("Failed to parse rules");
    save_char(&ruleset, "target/english.bin")
        .expect("Failed to save rules");

    // Compile LLRE patterns
    for entry in std::fs::read_dir("patterns").unwrap() {
        let path = entry.unwrap().path();
        if path.extension() == Some("llre".as_ref()) {
            let file = load_llre_file(&path).expect("Failed to parse");
            let nfa = compile_llre(&file).expect("Failed to compile");
            let out_path = path.with_extension("nfa.bin");
            save_compiled_llre(&nfa, &out_path).expect("Failed to save");
        }
    }

    println!("cargo:rerun-if-changed=rules/");
    println!("cargo:rerun-if-changed=patterns/");
}
```

### 7.4 Performance Comparison

| Operation | Parse + Compile | Load Binary |
|-----------|-----------------|-------------|
| 50-rule LLev | ~2ms | ~50μs |
| Complex regex | ~500μs | ~20μs |
| Large NFA (1000 states) | ~5ms | ~100μs |

Binary loading is 20-50× faster than parsing.

---

## Part 8: Advanced Topics

### 8.1 Context Patterns

Context patterns enable position-aware matching—ensuring patterns only match at specific positions like word boundaries.

```rust
use liblevenshtein::phonetic::nfa::{
    compile, ContextMatcherChar, ContextPatternChar, BoundaryKind
};
use liblevenshtein::phonetic::regex::parse;

let regex = parse("the")?;
let nfa = compile(&regex)?;

// Match only at word start
let pattern = ContextPatternChar::new()
    .with_left_boundary(BoundaryKind::WordStart)
    .with_pattern(nfa);

let matcher = ContextMatcherChar::new(pattern);

// "the" at position 0 - word start
assert!(matcher.matches_at("the quick fox", 0));

// "the" at position 10 - mid-word in "other"
assert!(!matcher.matches_at("another thing", 3));
```

**Boundary Types:**

| Boundary | Description |
|----------|-------------|
| `WordStart` | After whitespace or string start |
| `WordEnd` | Before whitespace or string end |
| `LineStart` | After newline or string start |
| `LineEnd` | Before newline or string end |
| `StringStart` | Only at position 0 |
| `StringEnd` | Only at final position |

### 8.2 Cycle Detection

When applying rules iteratively, infinite loops can occur if rules form a cycle. Use cycle detection to handle this safely.

```rust
use liblevenshtein::phonetic::{
    apply_rules_with_cycle_detection, NormalizationResult
};

match apply_rules_with_cycle_detection(&rules, input, 100) {
    NormalizationResult::FixedPoint(output) => {
        // Converged to stable result
        println!("Normalized: {}", output);
    }
    NormalizationResult::Cycle { value, cycle_start } => {
        // Detected infinite loop
        eprintln!("Cycle at iteration {}: {}", cycle_start, value);
    }
    NormalizationResult::MaxIterations(output) => {
        // Hit limit without converging
        println!("Result after max iterations: {}", output);
    }
}
```

### 8.3 Performance Optimization

#### Position Skipping

For very long strings with repetitive patterns (100+ characters), use the optimized applier:

```rust
use liblevenshtein::phonetic::apply_rules_seq_optimized;

// Up to 26× faster for synthetic strings with repetitive patterns
let result = apply_rules_seq_optimized(&rules, very_long_input);
```

**Recommendations:**
- `apply_rules_seq` - Default, best for typical words (<100 chars)
- `apply_rules_seq_optimized` - Only for very long strings with repetition

#### Thompson Builder (Low-level NFA)

For programmatic NFA construction without parsing:

```rust
use liblevenshtein::phonetic::nfa::ThompsonBuilderChar;

let builder = ThompsonBuilderChar::new();

// Build NFA directly
let a = builder.single_char('a');
let b = builder.single_char('b');
let a_or_b = builder.alternation(a, b);
let pattern = builder.kleene_star(a_or_b);  // (a|b)*

assert!(pattern.accepts(""));
assert!(pattern.accepts("abab"));
assert!(pattern.accepts("aaaa"));
```

**Thompson Builder Operations:**

| Method | Description | Regex Equivalent |
|--------|-------------|------------------|
| `single_char(c)` | Single character | `c` |
| `literal(s)` | Literal string | `abc` |
| `char_class(cc)` | Character class | `[abc]` |
| `epsilon()` | Empty match | ε |
| `concatenate(a, b)` | Sequence | `ab` |
| `alternation(a, b)` | Choice | `a\|b` |
| `kleene_star(a)` | Zero or more | `a*` |
| `kleene_plus(a)` | One or more | `a+` |
| `optional(a)` | Zero or one | `a?` |

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
- [Example: Phonetic Spellcheck](../../examples/phonetic_spellcheck/README.md)
- [Example LLev Rule Sets](../../examples/phonetic_spellcheck/rules/) (`base.llev`, `homophones.llev`, `text_speak.llev`)
- [Built-in Zompist Rules (source)](../../src/phonetic/rules/zompist_char.rs)

---

[← Documentation Index](../README.md)
