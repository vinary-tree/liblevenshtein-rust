# Dictionary Layer — Implementations

**Per-backend implementation guides for the dictionary layer (Layer 1).**

The dictionary layer provides efficient storage and traversal of term
collections. Each document here covers one concrete backend: its node
representation, construction, query characteristics, memory profile, and the
workloads it is best suited for. Use this index to pick a backend; see the
[layer overview](../) for cross-cutting theory and the
[performance comparison](../performance/benchmarks.md) for head-to-head numbers.

## Implementations

| Document | Purpose |
|----------|---------|
| [double-array-trie.md](double-array-trie.md) | `DoubleArrayTrie` (ASCII, `u8`) — recommended general-purpose, read-optimized static dictionary; ~6–8 bytes/char, fast queries. |
| [double-array-trie-char.md](double-array-trie-char.md) | `DoubleArrayTrieChar` (UTF-8, `u32`) — character-level double-array trie for correct Unicode (CJK, emoji, combining marks). |
| [dynamic-dawg.md](dynamic-dawg.md) | `DynamicDawg` (ASCII, `u8`) — thread-safe insert/remove DAWG with SIMD and bloom-filter optimizations for run-time-mutable dictionaries. |
| [dynamic-dawg-char.md](dynamic-dawg-char.md) | `DynamicDawgChar` (UTF-8, `u32`) — character-level dynamic DAWG variant for mutable Unicode dictionaries. |
| [suffix-automaton.md](suffix-automaton.md) | `SuffixAutomaton` — substring/infix matching for full-text search. |
| [pathmap-dictionary.md](pathmap-dictionary.md) | `PathMapDictionary` — PathMap-backed dictionary adapter with persistent, structurally-shared tries. |

### Choosing a backend

```
Need run-time insert/remove?
├─ YES → DynamicDawg / DynamicDawgChar
└─ NO  (static / read-mostly)
    ├─ Unicode text? → DoubleArrayTrieChar
    ├─ Substring search? → SuffixAutomaton
    └─ Otherwise → DoubleArrayTrie  ⭐ recommended
```

**Status: Living reference.**

[← Documentation Index](../../../README.md)
