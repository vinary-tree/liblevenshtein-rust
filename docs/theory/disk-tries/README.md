# Disk-Based Trie Data Structures

This documentation series provides a comprehensive, pedagogical introduction to disk-based trie data structures, culminating in the design of the **Persistent Adaptive Radix Trie (PART)** - a hybrid structure combining the Adaptive Radix Tree (ART) with B-trie-style bucket storage.

## Motivation

When dictionaries exceed available RAM, we need data structures that efficiently manage data on secondary storage (SSD/HDD). Traditional in-memory tries waste space and incur excessive I/O when naively persisted to disk. This series explores specialized techniques for building tries that minimize disk I/O while maintaining fast lookup and update operations.

## Document Organization

The documents are numbered to indicate the recommended reading order:

| Document | Topic | Prerequisites |
|----------|-------|---------------|
| [01-foundations](01-foundations.md) | Trie basics and disk I/O fundamentals | None |
| [02-b-trie](02-b-trie.md) | B-trie architecture (Askitis & Zobel 2009) | 01 |
| [03-adaptive-radix-tree](03-adaptive-radix-tree.md) | Adaptive Radix Tree theory (Leis et al. 2013) | 01 |
| [04-persistent-art](04-persistent-art.md) | Disk persistence with pointer swizzling | 01, 03 |
| [05-buffer-management](05-buffer-management.md) | Page cache, WAL, and crash recovery | 01 |
| [06-persistent-artrie-design](06-persistent-artrie-design.md) | Our hybrid PART design | All previous |

## Reading Paths

Depending on your background and goals, consider these reading paths:

### For Newcomers to Disk-Based Data Structures
Read all documents in order: 01 → 02 → 03 → 04 → 05 → 06

### For Those Familiar with B-trees but New to Tries
Start with 01 (trie foundations), then 02 (B-trie), then 06 (design summary)

### For Implementers
Focus on 04 (persistence techniques), 05 (buffer management), and 06 (final design)

### Quick Reference
Jump directly to 06 for a summary of the final design with references back to detailed explanations

## Key Concepts Quick Reference

### Data Structures

| Structure | Description | Best For |
|-----------|-------------|----------|
| **B-trie** | Disk-based burst trie with buckets | Balanced read/write, space efficiency |
| **ART** | Adaptive Radix Tree with Node4/16/48/256 | Low-latency lookups, SIMD acceleration |
| **PART** | Persistent ART + B-trie buckets | Our hybrid combining both strengths |

### Storage Techniques

| Technique | Purpose |
|-----------|---------|
| **Pointer Swizzling** | Dual memory/disk addressing in single 64-bit pointer |
| **Buffer Manager** | Page cache with LRU eviction and pinning |
| **Write-Ahead Log (WAL)** | Crash recovery through operation logging |
| **Path Compression** | Reduce tree height by collapsing single-child chains |

### Complexity Summary

For the Persistent ARTrie design:

| Operation | Time Complexity | Disk I/Os |
|-----------|-----------------|-----------|
| Exact lookup | O(m) | 2-4 (typical) |
| Insert | O(m + log B) amortized | 2-4 + 1 write |
| Prefix search | O(m + k) | Depends on k |
| Levenshtein (d=1,2) | O(n·m·d²) | Varies with pruning |

Where: m = term length, B = bucket size (~100-500), k = result count

## References

Primary sources underlying this documentation:

1. **B-tries for disk-based string management**
   Askitis, N. & Zobel, J. (2009). *The VLDB Journal*, 18(1), 157-179.
   [DOI: 10.1007/s00778-008-0094-1](https://doi.org/10.1007/s00778-008-0094-1)

2. **The Adaptive Radix Tree: ARTful Indexing for Main-Memory Databases**
   Leis, V., Kemper, A., & Neumann, T. (2013). *ICDE*.
   [PDF](https://db.in.tum.de/~leis/papers/ART.pdf)

3. **Persistent Storage of Adaptive Radix Trees in DuckDB**
   DuckDB Team (2022).
   [Blog Post](https://duckdb.org/2022/07/27/art-storage)

4. **SMART: A High-Performance Adaptive Radix Tree for Disaggregated Memory**
   Luo, X. et al. (2023). *OSDI*.
   [PDF](https://www.usenix.org/system/files/osdi23-luo.pdf)

5. **HOT: A Height Optimized Trie Index for Main-Memory Database Systems**
   Binna, R. et al. (2018). *SIGMOD*.
   [PDF](https://15721.courses.cs.cmu.edu/spring2019/papers/08-oltpindexes2/p521-binna.pdf)

## Related Documentation

- [SCDAWG Theory](../scdawg/) - Symmetric Compact Directed Acyclic Word Graph
- [Levenshtein Automata](../../algorithms/) - Fuzzy string matching algorithms
