# Phonetic Spellcheck Demo

An interactive spell checker combining **phonetic normalization** with
**Damerau-Levenshtein distance** for robust fuzzy string matching.

## Features

- **62 Zompist Phonetic Rules**: Based on Mark Rosenfelder's English spelling rules
- **Transposition Support**: Catches adjacent character swaps ("teh" → "the")
- **Max Edit Distance 2**: Balances accuracy with result relevance
- **Interactive REPL**: Type queries and see matches in real-time
- **Formally Verified**: All phonetic rules proven correct in Coq/Rocq

### Advanced Features

- **AOT Compilation**: Pre-compile rules to binary cache for ~50x faster startup
- **Query Memoization**: LRU cache for instant repeated query responses
- **Cycle Detection**: Safely handle pathological rule cycles
- **Performance Statistics**: Track cache hits, query counts, and timing
- **Verbose Mode**: Step-by-step normalization visualization

## Quick Start

This is a self-contained example with its own `Cargo.toml` that depends on the
parent `liblevenshtein` crate via a path dependency.

```bash
cd examples/phonetic_spellcheck
make run
```

### Fast Start (Recommended)

Pre-compile rules once, then enjoy ~50x faster startup:

```bash
cd examples/phonetic_spellcheck
make aot       # Pre-compile rules (one-time, ~2-3s)
make run-fast  # Fast startup (~50-100ms)
```

### Using Cargo Directly

```bash
cd examples/phonetic_spellcheck
cargo run --release

# With options:
cargo run --release -- --aot              # Pre-compile rules
cargo run --release -- --use-cache        # Use pre-compiled cache
cargo run --release -- --verbose --stats  # Show details and stats
```

### From Project Root

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
  stats         - Show performance statistics

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

query> stats

=== Statistics ===
Queries: 4
Cache hits: 0 (0.0%)
Cache misses: 4
Cycles detected: 0

Startup timing:
  Rule loading: 45ms
  Dictionary normalization: 1.2s
  Transducer build: 320ms
  Total: 1.57s
```

### Verbose Mode Example

With `--verbose`, you can see normalization steps:

```
query> filosofy
Normalization: "filosofy"
  Input phones: [f, i, l, o, s, o, f, y]
  Rule applied: y -> i (final position)
  Result: filosofi

Matches for "filosofy" (normalized: "filosofi"):
  1. philosophy (distance: 0) (exact phonetic match)
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
make aot         # Pre-compile rules to binary cache (~50x faster startup)
make run-fast    # Run with pre-compiled cache (requires: make aot)
make run-verbose # Run with verbose normalization output
make run-stats   # Run with statistics display on exit
make run-full    # Run with all features enabled
make setup-data  # Copy dictionary file if needed
make check       # Quick syntax/type check
make clean       # Remove build artifacts and cache
make help        # Show all options
```

## CLI Options

```
phonetic_spellcheck [OPTIONS]

Options:
  --aot             Pre-compile rules to binary cache
  --use-cache       Load from pre-compiled cache
  --verbose, -v     Show normalization steps for each query
  --stats           Show statistics on exit
  --detect-cycles   Enable cycle detection in rule application
  --cache-size N    Query cache size (default: 1000)
  --help, -h        Show help
```

## Files

```
phonetic_spellcheck/
├── Cargo.toml               # Package manifest with path dependency
├── Makefile                 # Build orchestration
├── README.md                # This file
├── cache/                   # Pre-compiled rule cache (after 'make aot')
│   └── rules.bin            # Binary serialized rules
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
