# Foundations: Tries and Disk I/O

This document introduces the foundational concepts needed to understand disk-based trie data structures: the trie data structure itself, the external memory model for disk I/O, and the challenges of adapting in-memory data structures for secondary storage.

## Table of Contents

1. [The Trie Data Structure](#the-trie-data-structure)
2. [Trie Variants](#trie-variants)
3. [The External Memory Model](#the-external-memory-model)
4. [Adapting Tries for Disk](#adapting-tries-for-disk)
5. [Performance Metrics](#performance-metrics)

---

## The Trie Data Structure

### Definition

A **trie** (from "retrieval," pronounced either "try" or "tree") is a tree-based data structure for storing strings where each edge is labeled with a character (or more generally, a symbol from an alphabet). The path from the root to any node spells out a prefix of the stored strings.

```
              root
            /   |   \
           a    b    c
          /|\   |    |
         p n t  e    a
        /  |  \  \    \
       p   d   e  a    r
      /    |      |     \
     l     y     r*     t*
    /            |
   e*           s*

   Stores: {apple, andy, ate, bear, bears, cart}
   * marks word endings (final states)
```

### Formal Definition

A trie T over alphabet Σ is a rooted tree where:

1. Each edge is labeled with a symbol σ ∈ Σ
2. No two edges from the same node have the same label
3. Each node may be marked as "final" to indicate a stored string ends there
4. The string represented by a node is the concatenation of edge labels on the path from root to that node

### Properties

**Time Complexity:**
- Lookup: O(m) where m is the length of the query string
- Insert: O(m)
- Delete: O(m)
- Prefix search: O(m + k) where k is the number of matches

**Space Complexity:**
- Worst case: O(n × m × |Σ|) where n is the number of strings and |Σ| is alphabet size
- In practice: Much better due to prefix sharing

### Comparison with Other String Structures

| Structure | Lookup | Insert | Space | Notes |
|-----------|--------|--------|-------|-------|
| Sorted Array | O(m log n) | O(n) | O(N) | Binary search, N = total chars |
| Hash Table | O(m) expected | O(m) | O(N) | No prefix queries, hash collisions |
| Trie | O(m) | O(m) | Variable | Prefix queries, deterministic |
| BST of strings | O(m log n) | O(m log n) | O(N) | Balanced variants |

The key advantage of tries is that lookup time depends only on query length, not the number of stored strings.

---

## Trie Variants

Several trie variants optimize for specific use cases:

### Standard Trie

The basic trie as described above. Each node has up to |Σ| children, stored in some collection (array, hash map, linked list).

**Alphabet array representation:**
```
struct TrieNode {
    children: [Option<Box<TrieNode>>; 256],  // For ASCII
    is_final: bool,
}
```

This wastes space when nodes have few children.

### Patricia Trie (Radix Tree)

A **Patricia trie** (Practical Algorithm to Retrieve Information Coded in Alphanumeric) compresses chains of single-child nodes by storing edge labels as strings rather than single characters.

```
Standard Trie:          Patricia Trie:
      root                   root
       |                    /    \
       t                 "test"  "toast"
       |                   |
       e                  "er"
      / \                /    \
     s   o             "s"   "ing"
     |   |
     t   a
     |   |
     e   s
     |   |
     r   t
```

**Benefits:**
- Reduced tree height
- Fewer nodes to allocate and traverse
- Better cache locality

**Trade-off:**
- More complex node structure
- String comparisons at each node

### Burst Trie

A **burst trie** (Heinz et al. 2002) adapts its structure based on access patterns. It starts with simple bucket containers and "bursts" them into trie structure when they become too large or too frequently accessed.

```
Initial (bucket):       After burst:
    [cat, car, cup]         c
                           / \
                          a   u
                         / \   \
                        t  r    p
```

### DAWG (Directed Acyclic Word Graph)

A **DAWG** shares suffixes in addition to prefixes, creating a DAG rather than a tree. This minimizes space but complicates some operations.

### Succinct Tries

**Succinct** data structures use space close to the information-theoretic minimum. Examples include:

- **LOUDS** (Level-Order Unary Degree Sequence): Encodes tree structure in ~2n bits
- **FST** (Fast Succinct Trie): Combines LOUDS encoding with bit-parallel operations

These are typically read-only after construction.

---

## The External Memory Model

When data exceeds RAM, we must consider the cost of disk I/O. The **external memory model** (also called the I/O model or disk access model) quantifies this.

### Model Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| M | Main memory size (bytes) | 16-256 GB |
| B | Block size (bytes) | 4 KB - 256 KB |
| N | Problem size (data items or bytes) | > M |

### Cost Measure

In this model, we count the number of **I/O operations** (block transfers between disk and memory), not CPU operations. Each I/O transfers one block of B bytes.

**Key insight:** Reading 1 byte costs the same as reading B bytes, because disk access has high latency but good throughput. We must read/write in blocks.

### Disk I/O Characteristics

**Hard Disk Drives (HDD):**
- Seek time: 5-15 ms (mechanical head movement)
- Rotational latency: 2-8 ms
- Transfer rate: 100-200 MB/s sequential
- Random read: ~100 IOPS

**Solid State Drives (SSD):**
- No seek time (electronic)
- Latency: 0.05-0.2 ms
- Transfer rate: 500-7000 MB/s (NVMe)
- Random read: 10,000-500,000 IOPS

### Implications for Data Structure Design

1. **Maximize useful data per I/O**: Pack related data into blocks
2. **Minimize I/O count**: Trade CPU work for fewer disk accesses
3. **Prefer sequential access**: Sequential reads are 10-100× faster than random
4. **Consider block size**: Optimal block size depends on workload and device

### B-tree: The Classic Disk-Optimized Structure

B-trees exemplify disk-optimized design:

```
                    [M]
                   /   \
           [D, H]      [T, X]
          /  |  \      /  |  \
        [A-C][E-G][I-L][N-S][U-W][Y-Z]
```

- **High fanout**: Each node has O(B) children, reducing tree height
- **Height**: O(log_B N), so only O(log_B N) I/Os per operation
- **Node size = block size**: Each node read/write is one I/O

For N = 1 billion items and B = 4KB pages holding 400 keys:
- B-tree height: log₄₀₀(10⁹) ≈ 4 levels
- Binary tree height: log₂(10⁹) ≈ 30 levels

---

## Adapting Tries for Disk

Naively persisting a standard trie to disk performs poorly:

### Problems with Naive Persistence

1. **Pointer overhead**: In-memory pointers (8 bytes) become disk offsets
2. **Small nodes**: Nodes with few children waste space in fixed-size blocks
3. **Random access**: Tree traversal causes random I/Os
4. **Height**: Long strings require many levels = many I/Os

### Example: Naive Trie Persistence

Consider looking up "international" (13 characters) in a naive disk trie:
- Standard trie: 13 pointer traversals
- If each node is a separate disk read: 13 I/Os!

Compare to a B-tree with fanout 256:
- Height ≈ 3-4 for most dictionaries
- Only 3-4 I/Os per lookup

### Design Goals for Disk Tries

1. **Minimize tree height**: Use path compression, high fanout
2. **Pack nodes efficiently**: Adaptive node sizes, avoid wasted space
3. **Batch related data**: Store strings with common prefixes together
4. **Support prefix queries**: Maintain trie structure for efficient prefix matching
5. **Enable updates**: Unlike read-only succinct tries

### Approaches to Disk-Efficient Tries

| Approach | Description | Examples |
|----------|-------------|----------|
| **Burst/Bucket** | Store leaves in buckets, burst when full | B-trie, HAT-trie |
| **Adaptive Nodes** | Use different node types based on fanout | ART, HOT |
| **Block Packing** | Pack multiple nodes per disk block | String B-tree |
| **Serialized DAG** | Serialize DAWG with offset-based pointers | FST on disk |

---

## Performance Metrics

When evaluating disk-based tries, consider these metrics:

### I/O Complexity

- **Lookup I/Os**: Number of disk reads to find a key
- **Insert I/Os**: Reads + writes for insertion (including splits/rebalancing)
- **Space amplification**: (Disk space used) / (Raw data size)

### Throughput Metrics

- **Ops/second**: Lookup or insert operations per second
- **Sequential throughput**: For bulk loading or range scans
- **Random throughput**: For point queries with non-local access

### Cache Behavior

- **Working set size**: Frequently accessed data that should stay in RAM
- **Cache hit ratio**: Fraction of accesses served from buffer cache
- **Compulsory misses**: Cold-start misses unavoidable by any cache policy

### Practical Benchmarks

When comparing disk tries, measure:

1. **Construction time**: Time to build index from sorted/unsorted input
2. **Point lookup latency**: p50, p99, p99.9 for random lookups
3. **Prefix query throughput**: Queries per second for prefix patterns
4. **Update throughput**: Mixed read/write workloads
5. **Space efficiency**: Bytes per string stored

---

## Summary

This foundation establishes the core concepts:

1. **Tries** provide O(m) lookup independent of dictionary size
2. **External memory model** counts I/Os, not CPU operations
3. **Disk-efficient design** requires high fanout, block packing, and minimizing height
4. **B-trees** show how to achieve O(log_B N) I/Os; our goal is similar for tries

The following documents explore specific solutions:
- [02-b-trie](02-b-trie.md): Bucket-based approach from the B-tree world
- [03-adaptive-radix-tree](03-adaptive-radix-tree.md): Adaptive node types from in-memory optimization

## References

1. Fredkin, E. (1960). "Trie Memory". *Communications of the ACM*.
2. Morrison, D. R. (1968). "PATRICIA—Practical Algorithm To Retrieve Information Coded in Alphanumeric". *JACM*.
3. Heinz, S., Zobel, J., & Williams, H. E. (2002). "Burst Tries: A Fast, Efficient Data Structure for String Keys". *ACM TOIS*.
4. Aggarwal, A. & Vitter, J. S. (1988). "The Input/Output Complexity of Sorting and Related Problems". *CACM*.
5. Bayer, R. & McCreight, E. (1972). "Organization and Maintenance of Large Ordered Indexes". *Acta Informatica*.
