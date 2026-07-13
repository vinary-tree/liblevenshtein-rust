# Archived: disk-trie deep-dive chapters (superseded by libdictenstein)

**Status: Archived — historical reference, not maintained.**

The canonical, maintained disk-trie / PersistentARTrie theory now lives in the **libdictenstein**
crate. For how the disk-persisted backends integrate with liblevenshtein, see the live pointer
[`docs/theory/disk-tries/README.md`](../../../theory/disk-tries/README.md). These chapters are
preserved verbatim from before the 2026-07-12 documentation trim (they use the project's earlier
notation and are not re-conformed):

1. [01-foundations](01-foundations.md) — trie basics and disk-I/O fundamentals
2. [02-b-trie](02-b-trie.md) — B-trie architecture (Askitis & Zobel 2009)
3. [03-adaptive-radix-tree](03-adaptive-radix-tree.md) — Adaptive Radix Tree theory (Leis et al. 2013)
4. [04-persistent-art](04-persistent-art.md) — disk persistence with pointer swizzling
5. [05-buffer-management](05-buffer-management.md) — page cache, WAL, crash recovery
6. [06-persistent-artrie-design](06-persistent-artrie-design.md) — the hybrid PART design
7. [07-benchmark-results](07-benchmark-results.md) — benchmark results

See [`../README.md`](../README.md) for the archival rationale.
