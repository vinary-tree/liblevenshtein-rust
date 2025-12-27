# Symmetric Compact Directed Acyclic Word Graph (SCDAWG)

This documentation provides a comprehensive, pedagogical treatment of the **Symmetric Compact DAWG** (SCDAWG), a data structure that enables O(|pattern|) substring searching with bidirectional navigation capabilities.

## Overview

The SCDAWG, also known as **C2S** (Compact Symmetric), is the most space-efficient index structure that supports:

1. **Substring search** in O(|pattern|) time
2. **Right extension**: given a pattern V, navigate to V followed by character σ
3. **Left extension**: given a pattern V, navigate to character σ followed by V
4. **Occurrence enumeration**: find all positions where a pattern occurs

These capabilities make the SCDAWG ideal for applications like the **WallBreaker** algorithm (Gerdjikov et al., 2013), which requires bidirectional pattern growth during dictionary-based fuzzy string matching.

## Document Structure

This documentation builds concepts progressively from fundamental to advanced:

| Document | Topic |
|----------|-------|
| [01-introduction](01-introduction.md) | Problem motivation: why we need substring indices |
| [02-suffix-automaton](02-suffix-automaton.md) | Foundation: equivalence classes, suffix links, end-positions |
| [03-cdawg](03-cdawg.md) | Compact DAWG: compaction and primary/secondary edges |
| [04-scdawg](04-scdawg.md) | Symmetric Compact DAWG: left extensions and prime subwords |
| [05-construction](05-construction.md) | On-line construction algorithm with sext links |
| [06-operations](06-operations.md) | Substring search, bidirectional extension, IS features |
| [07-references](07-references.md) | Annotated bibliography of source papers |

## Running Example

Throughout this documentation, we use the string **w = "abcabcab"** as a running example. This string is traced through each data structure:

```
String: a b c a b c a b
Index:  0 1 2 3 4 5 6 7
```

Key properties of this example:
- Length |w| = 8
- Alphabet Σ = {a, b, c}
- Contains repeated patterns: "ab" (3x), "abc" (2x), "bc" (2x), "cab" (2x)
- No unique end marker in raw form (added during construction)

## Complexity Summary

| Structure | States | Transitions | Space | Query Time |
|-----------|--------|-------------|-------|------------|
| Suffix Trie | O(n²) | O(n²) | O(n²) | O(m) |
| Suffix Tree | O(n) | O(n) | O(n) | O(m) |
| Suffix Automaton (DAWG) | ≤ 2n-1 | ≤ 3n-4 | O(n) | O(m) |
| CDAWG | ≤ n+1 | ≤ 2n-2 | O(n) | O(m) |
| **SCDAWG** | ≤ n+1 | ≤ 4n-4 | O(n) | O(m) |

Where n = |w| (text length) and m = |pattern| (query length).

## Key Concepts at a Glance

### Equivalence Classes

Strings belong to the same equivalence class if they share the same **end-position set**:

```
end-pos("ab") = {2, 5, 8}  (positions after "ab")
end-pos("cab") = {5, 8}    (positions after "cab")
```

Since "ab" and "cab" have different end-positions, they are in different classes.

### Suffix Links

Suffix links connect each state to its **longest proper suffix** that forms a distinct equivalence class:

```
State "abc" --suffix-link--> State "bc" --suffix-link--> State "c"
```

### Left Extension Edges (SCDAWG-specific)

While right extension edges (standard edges) navigate by **appending** characters:

```
"ab" --'c'--> "abc"   (append 'c' to "ab")
```

Left extension edges navigate by **prepending** characters:

```
"ab" --'c'--> "cab"   (prepend 'c' to "ab")
```

This bidirectional capability is what makes the SCDAWG "symmetric."

### Prime Subwords

A **prime subword** is a maximal representative of an equivalence class:

```
If every occurrence of "ab" is preceded by 'c' and followed by 'c',
then "cabcc" is the implication (prime subword) of "ab".
```

The SCDAWG contains only prime subwords as nodes, making it maximally compact.

## Prerequisites

This documentation assumes familiarity with:
- Basic automata theory (states, transitions, acceptance)
- Graph terminology (nodes, edges, DAG)
- Asymptotic complexity notation (O-notation)

No prior knowledge of suffix structures is required.

## References

The key papers that define and construct the SCDAWG are:

1. **Blumer et al. (1987)** - "Complete Inverted Files for Efficient Text Retrieval and Analysis"
   - Defines the SCDAWG structure (C2S)
   - Introduces IS (Inverted File) features: freq(), locations()

2. **Inenaga et al. (2001)** - "On-Line Construction of Symmetric Compact Directed Acyclic Word Graphs"
   - On-line O(n) construction algorithm
   - Key insight: sext links = edges of CDAWG(w^rev)

3. **Inenaga et al. (2005)** - "On-line construction of compact directed acyclic word graphs"
   - On-line O(n) CDAWG construction
   - Multi-string support with unique end markers

See [07-references](07-references.md) for the complete annotated bibliography.
