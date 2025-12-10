# Phonetic Spellcheck Demo

An interactive spell checker combining **phonetic normalization** with
**Damerau-Levenshtein distance** for robust fuzzy string matching.

## Features

- **62 Zompist Phonetic Rules**: Based on Mark Rosenfelder's English spelling rules
- **Transposition Support**: Catches adjacent character swaps ("teh" → "the")
- **Max Edit Distance 2**: Balances accuracy with result relevance
- **Interactive REPL**: Type queries and see matches in real-time
- **Formally Verified**: All phonetic rules proven correct in Coq/Rocq

## Quick Start

This is a self-contained example with its own `Cargo.toml` that depends on the
parent `liblevenshtein` crate via a path dependency.

```bash
cd examples/phonetic_spellcheck
make run
```

Or using cargo directly:

```bash
cd examples/phonetic_spellcheck
cargo run --release
```

### Alternative: From Project Root

You can also run as a cargo example from the project root:

```bash
cargo run --example phonetic_spellcheck --features "phonetic-rules,serialization" --release
```

## Example Session

```
╔══════════════════════════════════════════════════════════════════╗
║           Phonetic Spellcheck Demo                               ║
║   Combining phonetic normalization with Levenshtein distance     ║
╚══════════════════════════════════════════════════════════════════╝

Loading dictionary from data/english_words.txt...
  Loaded 123985 terms
Loading phonetic rules from rules/zompist.llev...
  Loaded 62 rules
Normalizing dictionary with phonetic rules...
  Built transducer with 98234 normalized forms

Using Damerau-Levenshtein (transposition) with max distance 2

Type a misspelled word to find matches. Commands:
  exit, quit, q - Exit the program
  help, ?       - Show this help

query> fone
Matches for "fone" (normalized: "fon"):
  1. phone (distance: 0) (exact phonetic match)
  2. cone (distance: 1)
  3. done (distance: 1)
  4. foe (distance: 1)
  5. one (distance: 1)

query> teh
Matches for "teh" (normalized: "te"):
  1. the (distance: 1)
  2. tea (distance: 1)
  3. ten (distance: 1)
  4. bet (distance: 2)

query> filosofy
Matches for "filosofy" (normalized: "filosofi"):
  1. philosophy (distance: 0) (exact phonetic match)

query> enuf
Matches for "enuf" (normalized: "enuf"):
  1. enough (distance: 0) (exact phonetic match)
  2. en (distance: 2)
```

## The 62 Zompist Rules

The rules in `rules/zompist.llev` normalize English spelling to a phonetic form.
They are organized into 11 phases by priority:

### Phase 1: Affrication (4 rules)
- tion → shun, sion → zhun, cious → shus, tious → shus

### Phase 2: GH Patterns (3 rules)
- ough → o, aught → ot, ought → ot

### Phase 3: GH Before Vowel (1 rule)
- gh → g before vowels

### Phase 4: Digraph Conversions (4 rules)
- ch → ts, sh → s, ph → f, th → t

### Phase 5: Initial Clusters (8 rules)
- wr → r, wh → w, gn → n, kn → n, mn → n, pt → t, ps → s, tm → m

### Phase 6: X Pronunciation (2 rules)
- x → gz (between vowels), x → ks (elsewhere)

### Phase 7: Contextual (4 rules)
- c → s before [ei], c → k elsewhere, g → j before [ei], qu → kw

### Phase 8: Additional Orthographic (4 rules)
- ck → k, mb → m (final), bt → t, mn → m (final)

### Phase 9: Double Consonants (13 rules)
- bb → b, cc → c, dd → d, ff → f, gg → g, ll → l, mm → m, nn → n, pp → p, rr → r, ss → s, tt → t, zz → z

### Phase 10: Vowel Digraphs (12 rules)
- ea → e, ee → e, ai → a, ay → a, oa → o, oe → o, ou → ow, oi → oy, ey → e, ie → i, oo → u, ue → u

### Phase 11: Fallback (2 rules)
- Final e → silent, gh → silent

## How It Works

1. **Load Dictionary**: Reads `data/english_words.txt` (copy of `/usr/share/dict/words`)
2. **Parse Rules**: Loads `rules/zompist.llev` using the `.llev` parser
3. **Normalize Dictionary**: Applies rules to each term (e.g., "phone" → "fon")
4. **Build Index**: Creates a mapping from normalized forms back to originals
5. **Build Transducer**: Constructs a Levenshtein automaton with transposition support
6. **Query Loop**: For each query:
   - Normalize the query using the same rules
   - Search the transducer with max distance 2
   - Map results back to original dictionary terms
   - Display with edit distances

## Make Targets

```bash
make build       # Build the binary
make run         # Build and run interactively
make setup-data  # Copy dictionary file if needed
make check       # Quick syntax/type check
make clean       # Remove build artifacts
make help        # Show help
```

## Files

```
phonetic_spellcheck/
├── Cargo.toml               # Package manifest with path dependency
├── Makefile                 # Build orchestration
├── README.md                # This file
├── data/
│   └── english_words.txt    # Dictionary (auto-copied from /usr/share/dict/words)
├── rules/
│   └── zompist.llev         # 62 Zompist phonetic rules
└── src/
    └── main.rs              # Interactive demo source
```

## Formal Verification

The phonetic rules are formally verified in Coq/Rocq with five theorems:

1. **Well-Formedness**: All rules satisfy structural constraints
2. **Bounded Expansion**: Output ≤ input + 20 characters
3. **Non-Confluence**: Rule order matters (proven constructively)
4. **Termination**: Sequential application always terminates
5. **Idempotence**: Fixed points remain unchanged

See `docs/verification/phonetic/` for the complete proofs.

## Dependencies

This example depends on the parent `liblevenshtein` crate via path dependency:

```toml
[dependencies]
liblevenshtein = { path = "../..", features = ["phonetic-rules", "serialization"] }
```

## See Also

- `examples/phonetic_fuzzy_matching.rs` - Comprehensive phonetic matching demo
- `examples/phonetic_rewrite.rs` - Rule application demo
- `docs/verification/phonetic/` - Coq/Rocq formal proofs
