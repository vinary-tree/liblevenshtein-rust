# Phonetic Spellcheck Demo

Demonstrates the `PhoneticNormalizedDictionary` API with combined English phonetic
rules for robust fuzzy string matching.

## Features

- **PhoneticNormalizedDictionary**: Dual-index architecture with BK-tree optimization
- **Combined English Rules**: Base (62) + homophones (24) + text speak (31) = 117 rules
- **Fuzzy Matching**: Edit distance queries with automatic BK-tree acceleration
- **Regex Queries**: Pattern matching against normalized dictionary forms
- **Phonetic Pattern Expansion**: Automatic generation of phonetic alternations
- **Formally Verified**: All phonetic rules proven correct in Coq/Rocq

## Quick Start

```bash
cd examples/phonetic_spellcheck
cargo run --release
```

### From Project Root

```bash
cargo run --example phonetic_spellcheck --features "phonetic-rules,pathmap-backend,embedded-rules" --release
```

## Example Output

```
=== Phonetic Spellcheck Demo ===

Loading dictionary from data/english_words.txt...
  Loaded 123985 words in 45.2ms

Combining English phonetic rules...
  Combined 117 rules (base + homophones + text_speak)

Building PhoneticNormalizedDictionary...
  Built dictionary in 1.23s
  Original terms: 123985
  Normalized forms: 89234

--- Fuzzy Query Demo ---

Query: "fone" (distance: 2, 12 results in 0.8ms)
  Normalized query: "fon"
  1. phone (distance: 0, normalized: "fon")
  2. cone (distance: 1, normalized: "kon")
  3. done (distance: 1, normalized: "don")
  4. foe (distance: 1, normalized: "fo")
  5. one (distance: 1, normalized: "on")
  ... and 7 more

Query: "filosofy" (distance: 2, 1 results in 0.5ms)
  Normalized query: "filosofi"
  1. philosophy (distance: 0, normalized: "filosofi")

Query: "enuf" (distance: 1, 3 results in 0.3ms)
  Normalized query: "enuf"
  1. enough (distance: 0, normalized: "enuf")
  2. en (distance: 2, normalized: "en")
  3. ens (distance: 2, normalized: "ens")

--- Normalization Demo ---

"phone" -> "fon"
"elephant" -> "elefant"
"knight" -> "nit"
"through" -> "tru"

--- Regex Query Demo ---

Regex: "(ph|f)one" (distance: 0, 2 matches in 1.2ms)
  1. phone (distance: 0, normalized: "fon")
  2. fone (distance: 0, normalized: "fon")

--- Phonetic Pattern Expansion Demo ---

Input: "fone"
  Expanded pattern: "(f|ph)o(n|ne)"
  Matches: ["phone", "fone"]

Input: "nite"
  Expanded pattern: "(n|kn)i(t|te|ght)"
  Matches: ["night", "knight", "nite"]

=== Demo Complete ===
```

## How It Works

1. **Load Dictionary**: Reads `data/english_words.txt`
2. **Combine Rules**: Merges base + homophones + text_speak rule sets
3. **Build Dictionary**: Creates `PhoneticNormalizedDictionary` with:
   - HashMap for O(1) exact lookups
   - BK-tree for O(k log n) fuzzy queries
   - Phonetic normalization index
4. **Run Demos**: Demonstrates key API features:
   - `query(term, distance)` - Fuzzy matching
   - `normalize(term)` - String normalization
   - `query_regex(pattern, distance)` - Regex matching
   - `expand_to_phonetic_pattern(term)` - Pattern expansion

## Combined English Rules

The demo combines three phonetic rule sets:

### Base Rules (62 rules)
Based on Mark Rosenfelder's (Zompist) English spelling normalization:
- Affrication: tion -> shun, sion -> zhun
- GH patterns: ough -> o, aught -> ot
- Digraphs: ch -> ts, sh -> s, ph -> f, th -> t
- Initial clusters: wr -> r, kn -> n, gn -> n
- Double consonants: bb -> b, cc -> c, etc.
- Vowel digraphs: ea -> e, ee -> e, ai -> a, etc.

### Homophones (24 rules)
Common homophone equivalences:
- their/there/they're
- to/too/two
- your/you're
- its/it's

### Text Speak (31 rules)
Common abbreviations and informal spellings:
- u -> you, ur -> your
- 2 -> to/too, 4 -> for
- thru -> through, nite -> night

## API Reference

```rust
use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
use liblevenshtein::phonetic::rules::english;

// Combine rule sets
let mut rules = english::base().clone();
rules.merge(english::homophones().clone());
rules.merge(english::text_speak().clone());

// Build dictionary
let dict = PhoneticNormalizedDictionary::<()>::from_terms_with_rules(
    &words,
    rules.rules,
);

// Fuzzy query
let results = dict.query("fone", 2);
for candidate in results {
    println!("{} (distance: {})", candidate.term, candidate.distance);
}

// Normalize a string
let normalized = dict.normalize("elephant"); // -> "elefant"

// Regex query
let matches = dict.query_regex("(ph|f)one", 0)?;

// Phonetic pattern expansion
let pattern = dict.expand_to_phonetic_pattern("nite")?;
```

## Files

```
phonetic_spellcheck/
├── Cargo.toml           # Package manifest
├── README.md            # This file
├── data/
│   └── english_words.txt  # Dictionary (~124k words)
└── src/
    └── main.rs          # Demo source
```

## Formal Verification

The phonetic rules are formally verified in Coq/Rocq with five theorems:

1. **Well-Formedness**: All rules satisfy structural constraints
2. **Bounded Expansion**: Output <= input + 20 characters
3. **Non-Confluence**: Rule order matters (proven constructively)
4. **Termination**: Sequential application always terminates
5. **Idempotence**: Fixed points remain unchanged

See `docs/verification/phonetic/` for the complete proofs.

## Dependencies

```toml
[dependencies]
liblevenshtein = { path = "../..", features = ["phonetic-rules", "pathmap-backend", "embedded-rules"] }
```

## See Also

- `examples/phonetic_fuzzy_matching.rs` - Comprehensive phonetic matching demo
- `examples/phonetic_rewrite.rs` - Rule application demo
- `docs/verification/phonetic/` - Coq/Rocq formal proofs
