# Disk-based tries (PersistentARTrie) — integration with liblevenshtein

> **Source of truth for the theory:** the complete treatment of disk-based tries — B-tries,
> the Adaptive Radix Tree (ART), pointer swizzling, buffer management, and the hybrid
> **Persistent Adaptive Radix Trie (PART)** design — lives in the
> **[libdictenstein](https://github.com/f1r3fly-io/libdictenstein)** crate that owns the
> structure: [`libdictenstein/docs/theory/disk-tries/`](../../../../libdictenstein/docs/theory/disk-tries/README.md).
> This page covers only **how liblevenshtein uses it**. The earlier in-repo deep-dive chapters
> duplicated that treatment and are preserved under
> [`docs/archive/theory/disk-tries/`](../../archive/theory/README.md).

## What it is (in one paragraph)

When a dictionary exceeds available RAM, liblevenshtein reaches for a **disk-persisted** backend.
The **PersistentARTrie** (`libdictenstein`) is a hybrid of the **Adaptive Radix Tree** (ART) of
Leis et al. [[2]](#references) — whose adaptive `Node4/16/48/256` layout keeps exact lookup at
$`\mathcal{O}(m)`$ for a term of length $`m`$ — with **B-trie**-style bucket storage on disk
(Askitis & Zobel [[1]](#references)), memory-mapped and reached through a **lock-free CAS overlay**.
"Persistent" here means *non-volatile* (durable on SSD/HDD), **not** immutable: the
`Persistent*` family is fully **dynamic**, supporting atomic concurrent insert/remove.

## Why liblevenshtein needs it

liblevenshtein's transducer walks *any* `Dictionary` in lock-step with a simulated Levenshtein
automaton. When the backend is a `PersistentARTrie` (or the disk-persisted `PersistentScdawg`,
`PersistentSuffixAutomaton`, `PersistentSuffixTree`, `PersistentVocabARTrie`), that lock-step walk
runs directly over the **memory-mapped, lock-free** structure — so a dictionary far larger than RAM
is fuzzy-searched with the same $`\mathcal{O}(\lvert W\rvert)`$ per-query setup and $`\mathcal{O}(k)`$
per-transition cost as an in-memory backend, the difference being page-cache-bounded disk I/O rather
than resident memory. The ART node handles are the disk analogue of the in-memory `TrieRef` snapshot
that makes traversal $`\mathcal{O}(1)`$ per byte from the focus (see
[`docs/design/pathmap-trieref-rework.md`](../../design/pathmap-trieref-rework.md)).

| Operation (PersistentARTrie) | Time | Disk I/Os (typical) |
|---|---|---|
| Exact lookup | $`\mathcal{O}(m)`$ | 2–4 |
| Insert | $`\mathcal{O}(m + \log B)`$ amortized | 2–4 + 1 write |
| Prefix search | $`\mathcal{O}(m + r)`$ | depends on $`r`$ |
| Levenshtein ($`k = 1, 2`$) | $`\mathcal{O}(n \cdot m \cdot k^{2})`$ before pruning | varies with pruning |

where $`m`$ = term length, $`B`$ = bucket size (~100–500), $`r`$ = result count, $`n`$ = dictionary size.

## Further reading

- **Full disk-trie / PART theory (canonical source):** [`libdictenstein/docs/theory/disk-tries/`](../../../../libdictenstein/docs/theory/disk-tries/README.md).
- Backend selection in liblevenshtein: [`user-guide/backends.md`](../../user-guide/backends.md).
- Archived in-repo deep-dive chapters (superseded by libdictenstein): [`docs/archive/theory/disk-tries/`](../../archive/theory/README.md).

## References

1. N. Askitis and J. Zobel. "B-tries for disk-based string management." *The VLDB Journal*, 18(1):157–179, 2009. [doi:10.1007/s00778-008-0094-1](https://doi.org/10.1007/s00778-008-0094-1)
2. V. Leis, A. Kemper, and T. Neumann. "The adaptive radix tree: ARTful indexing for main-memory databases." *IEEE ICDE 2013*, pp. 38–49. [doi:10.1109/ICDE.2013.6544812](https://doi.org/10.1109/ICDE.2013.6544812)
