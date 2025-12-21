# Phonetic Spellcheck WASM Demo

A WebAssembly demo of the phonetic spellchecker, running entirely in the browser.

## Features

- **Phonetic normalization**: Uses Zompist rules to normalize spelling variations
- **Fuzzy matching**: Levenshtein distance with transposition support
- **Embedded data**: Dictionary (124k words) and rules compiled into WASM
- **Fast queries**: Sub-millisecond response times with result caching
- **Interactive UI**: Live search with example queries

## Quick Start

### Prerequisites

```bash
# Install wasm-pack
cargo install wasm-pack

# Optional: Install binaryen for wasm-opt
# On Arch: pacman -S binaryen
# On macOS: brew install binaryen
```

### Build and Run

```bash
# Build the WASM module
make build

# Start local server and open browser
make serve
# Visit http://localhost:8080
```

## Build Targets

| Command | Description |
|---------|-------------|
| `make build` | Build optimized WASM for browser |
| `make build-debug` | Build debug WASM (faster compile) |
| `make build-wasi` | Build for WASI runtime |
| `make optimize` | Further optimize with wasm-opt |
| `make serve` | Build and start dev server |
| `make test` | Run Rust tests |
| `make clean` | Remove build artifacts |

## Project Structure

```
phonetic_spellcheck_wasm/
├── Cargo.toml          # WASM dependencies
├── Makefile            # Build automation
├── src/
│   ├── lib.rs          # WASM entry points
│   ├── core.rs         # Spellcheck logic
│   └── embedded.rs     # Embedded dictionary/rules
├── www/                # Browser demo
│   ├── index.html
│   ├── app.js
│   └── style.css
└── pkg/                # wasm-pack output (gitignored)
```

## API

The WASM module exports these functions:

```javascript
import init, { init as initSpellchecker, query, get_stats, clear_cache } from './pkg/phonetic_spellcheck_wasm.js';

// Load WASM module
await init();

// Initialize spellchecker (builds index, ~500ms)
await initSpellchecker();

// Query for suggestions
const result = query("fone");
// {
//   original: "fone",
//   normalized: "fon",
//   matches: [{ word: "phone", distance: 0 }, ...],
//   from_cache: false
// }

// Get statistics
const stats = get_stats();
// { dictionary_size: 123985, rules_count: 165, cache_size: 0 }

// Clear query cache
clear_cache();
```

## Size

| Component | Size (raw) | Size (gzip) |
|-----------|-----------|-------------|
| WASM binary | ~400 KB | ~150 KB |
| Embedded dictionary | 1.19 MB | ~350 KB |
| Embedded rules | 21.7 KB | ~6 KB |
| **Total transfer** | - | **~500 KB** |

## Example Queries

| Query | Normalized | Top Match | Explanation |
|-------|------------|-----------|-------------|
| fone | fon | phone | ph→f normalization |
| teh | teh | the | Transposition e↔h |
| filosofy | filosofy | philosophy | ph→f + silent letters |
| enuf | enuf | enough | gh→silent |
| nite | nit | night | Silent gh |

## How It Works

1. **Embedded Data**: Dictionary and rules are compiled into the WASM binary using `include_str!`
2. **Phonetic Normalization**: Input is normalized using Zompist phonetic rules
3. **Levenshtein Search**: Transducer finds words within edit distance 2
4. **Result Mapping**: Normalized matches are mapped back to original spellings
5. **Caching**: Recent queries are cached for instant repeat lookups

## Development

```bash
# Run Rust tests
cargo test

# Watch for changes (requires cargo-watch)
cargo watch -x test

# Run browser tests
wasm-pack test --headless --chrome
```

## License

MIT
