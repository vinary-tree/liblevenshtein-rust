# Persistent ARTrie Design

This document presents our hybrid design for the **Persistent Adaptive Radix Trie (PART)**, combining Adaptive Radix Tree (ART) nodes with B-trie-style buckets for leaf storage. This design is optimized for Levenshtein automata traversal in the liblevenshtein-rust library.

## Table of Contents

1. [Design Goals](#design-goals)
2. [Architecture Overview](#architecture-overview)
3. [Swizzled Pointer Design](#swizzled-pointer-design)
4. [Adaptive Node Types](#adaptive-node-types)
5. [Leaf Bucket Design](#leaf-bucket-design)
6. [Trait Integration](#trait-integration)
7. [Levenshtein Traversal Optimization](#levenshtein-traversal-optimization)
8. [File Format](#file-format)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Design Goals

### Primary Requirements

| Requirement | Priority | Notes |
|-------------|----------|-------|
| Levenshtein automata support | Critical | Must implement `DictionaryNode` for traversal |
| Large dictionary support | Critical | 10GB-1TB datasets exceeding RAM |
| Balanced read/write | High | Not write-only or read-only |
| Crash recovery | High | Durability without data loss |
| Low latency lookups | High | 2-4 disk I/Os for exact match |

### Why Hybrid ART + B-trie?

We combine the best features of both structures:

**From ART:**
- Adaptive node types (Node4/16/48/256) minimize space
- Trie structure enables efficient prefix/fuzzy queries
- SIMD-accelerated Node16 lookup
- Path compression reduces tree height

**From B-trie:**
- Leaf buckets store multiple strings per page
- Amortized I/O cost for dense regions
- Efficient bulk loading
- Proven disk performance

### Alternative Approaches Rejected

| Alternative | Reason for Rejection |
|-------------|---------------------|
| Pure B-trie | ART's adaptive nodes better match automata traversal |
| Pure persistent ART | Single-string leaves waste I/O for dense regions |
| LSM-trie | Write-optimized, poor for balanced workloads |
| FST/LOUDS | Read-only, no updates |
| HAT-trie (in-memory) | Not designed for disk |

---

## Architecture Overview

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        API Layer                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │ Dictionary  │  │ MappedDict   │  │ MutableMappedDictionary    │  │
│  │ trait       │  │ trait        │  │ trait (insert/remove)      │  │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────────┐
│                        Index Layer (ART)                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Adaptive Nodes with Swizzled Pointers                        │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐               │   │
│  │  │ Node4  │ │ Node16 │ │ Node48 │ │ Node256  │               │   │
│  │  │(linear)│ │ (SIMD) │ │(index) │ │ (direct) │               │   │
│  │  └────────┘ └────────┘ └────────┘ └──────────┘               │   │
│  │                                                               │   │
│  │  Path Compression: Collapse single-child chains               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────────┐
│                        Leaf Layer (B-trie Buckets)                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  String Buckets (8KB pages)                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │   │
│  │  │  Bucket A   │  │  Bucket B   │  │  Bucket C   │           │   │
│  │  │ ~100 strs   │  │ ~100 strs   │  │ ~100 strs   │           │   │
│  │  │ sorted      │  │ sorted      │  │ sorted      │           │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────────┐
│                        Storage Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐    │
│  │  Buffer Manager │  │  WAL            │  │  Disk Manager    │    │
│  │  (256KB blocks) │  │  (redo logging) │  │  (file I/O)      │    │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### When to Use Buckets vs. ART Nodes

The transition between ART nodes and buckets follows these rules:

```
                   Dense subtree?
                        │
          ┌─────────────┴─────────────┐
          ↓ Yes                       ↓ No
    Use ART nodes              Check fanout
    (many branches)                   │
                         ┌────────────┴────────────┐
                         ↓ Few children            ↓ Many children
                    Use bucket                 Use ART nodes
                  (store strings)            (continue trie)
```

**Heuristic:** Use buckets when:
- Subtree has < 100 strings
- All strings share a long common prefix
- Expected growth is low

---

## Swizzled Pointer Design

### 64-bit Layout

```
┌────────────────────────────────────────────────────────────────────┐
│ Bit 63 (MSB): Swizzle Flag                                         │
│   1 = Swizzled (memory pointer)                                    │
│   0 = Unswizzled (disk reference)                                  │
├────────────────────────────────────────────────────────────────────┤
│ When swizzled (memory pointer):                                    │
│   Bits 62-0: Memory address (mask off MSB)                         │
│   Valid because user-space addresses have bit 63 = 0               │
├────────────────────────────────────────────────────────────────────┤
│ When unswizzled (disk reference):                                  │
│   Bits 62-40: Block ID (23 bits = 8M blocks)                       │
│   Bits 39-18: Offset in block (22 bits = 4MB offset)               │
│   Bits 17-0:  Node type + flags (18 bits)                          │
│                                                                     │
│   With 256KB blocks: 8M × 256KB = 2TB addressable                  │
└────────────────────────────────────────────────────────────────────┘
```

### Rust Implementation

```rust
use std::sync::atomic::{AtomicU64, Ordering};

const SWIZZLE_FLAG: u64 = 1 << 63;
const BLOCK_ID_SHIFT: u64 = 40;
const OFFSET_SHIFT: u64 = 18;
const BLOCK_ID_MASK: u64 = 0x7FFFFF;  // 23 bits
const OFFSET_MASK: u64 = 0x3FFFFF;     // 22 bits
const FLAGS_MASK: u64 = 0x3FFFF;       // 18 bits

#[derive(Debug)]
pub struct SwizzledPtr(AtomicU64);

impl SwizzledPtr {
    /// Create unswizzled pointer to disk location
    pub fn disk(block_id: u32, offset: u32, node_type: NodeType) -> Self {
        debug_assert!(block_id <= BLOCK_ID_MASK as u32);
        debug_assert!(offset <= OFFSET_MASK as u32);

        let encoded = ((block_id as u64 & BLOCK_ID_MASK) << BLOCK_ID_SHIFT)
                    | ((offset as u64 & OFFSET_MASK) << OFFSET_SHIFT)
                    | (node_type as u64);
        Self(AtomicU64::new(encoded))
    }

    /// Create swizzled pointer to memory
    pub fn memory(ptr: *const Node) -> Self {
        let addr = ptr as u64;
        debug_assert!(addr & SWIZZLE_FLAG == 0);
        Self(AtomicU64::new(addr | SWIZZLE_FLAG))
    }

    /// Check if pointer is in memory
    #[inline]
    pub fn is_swizzled(&self) -> bool {
        self.0.load(Ordering::Acquire) & SWIZZLE_FLAG != 0
    }

    /// Get memory pointer (fast path)
    #[inline]
    pub unsafe fn as_ptr_unchecked(&self) -> *const Node {
        let val = self.0.load(Ordering::Acquire);
        (val & !SWIZZLE_FLAG) as *const Node
    }

    /// Decode disk location (slow path)
    pub fn disk_location(&self) -> Option<(u32, u32, NodeType)> {
        let val = self.0.load(Ordering::Acquire);
        if val & SWIZZLE_FLAG != 0 {
            return None;
        }
        let block_id = ((val >> BLOCK_ID_SHIFT) & BLOCK_ID_MASK) as u32;
        let offset = ((val >> OFFSET_SHIFT) & OFFSET_MASK) as u32;
        let node_type = NodeType::from_u8((val & FLAGS_MASK) as u8)?;
        Some((block_id, offset, node_type))
    }

    /// Atomically swizzle disk reference to memory pointer
    pub fn swizzle(&self, ptr: *const Node) -> Result<(), SwizzleError> {
        let old = self.0.load(Ordering::Acquire);
        if old & SWIZZLE_FLAG != 0 {
            return Err(SwizzleError::AlreadySwizzled);
        }

        let new = (ptr as u64) | SWIZZLE_FLAG;
        self.0.compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| SwizzleError::RaceCondition)
    }
}
```

---

## Adaptive Node Types

### Common Header

All node types share a header for polymorphic handling:

```rust
#[repr(C)]
pub struct NodeHeader {
    node_type: u8,        // Node4=4, Node16=16, Node48=48, Node256=0
    num_children: u8,     // Current child count
    partial_len: u8,      // Compressed path length
    _padding: u8,         // Alignment
    partial: [u8; 12],    // Compressed path bytes (up to 12)
}
```

### Node4 (1-4 children)

```rust
#[repr(C, align(64))]  // Cache line aligned
pub struct Node4 {
    header: NodeHeader,          // 16 bytes
    keys: [u8; 4],               // 4 bytes (unsorted)
    _key_padding: [u8; 12],      // Alignment
    children: [SwizzledPtr; 4],  // 32 bytes
}
// Total: 64 bytes (1 cache line)
```

**Lookup:** Linear scan of 4 keys.

### Node16 (5-16 children)

```rust
#[repr(C, align(64))]
pub struct Node16 {
    header: NodeHeader,           // 16 bytes
    keys: [u8; 16],               // 16 bytes (sorted, 16-byte aligned for SIMD)
    children: [SwizzledPtr; 16],  // 128 bytes
}
// Total: 160 bytes (2.5 cache lines)
```

**Lookup:** SIMD parallel comparison (SSE4.1).

```rust
#[cfg(target_arch = "x86_64")]
pub fn find_child_simd(&self, key: u8) -> Option<usize> {
    use std::arch::x86_64::*;
    unsafe {
        let search = _mm_set1_epi8(key as i8);
        let keys = _mm_load_si128(self.keys.as_ptr() as *const __m128i);
        let cmp = _mm_cmpeq_epi8(search, keys);
        let mask = _mm_movemask_epi8(cmp) & ((1 << self.header.num_children) - 1);
        if mask != 0 {
            Some(mask.trailing_zeros() as usize)
        } else {
            None
        }
    }
}
```

### Node48 (17-48 children)

```rust
#[repr(C)]
pub struct Node48 {
    header: NodeHeader,            // 16 bytes
    child_index: [u8; 256],        // 256 bytes (key → slot, 255 = empty)
    children: [SwizzledPtr; 48],   // 384 bytes
}
// Total: 656 bytes
```

**Lookup:** Two array accesses: `index[key]` → `children[slot]`.

### Node256 (49-256 children)

```rust
#[repr(C)]
pub struct Node256 {
    header: NodeHeader,             // 16 bytes
    children: [SwizzledPtr; 256],   // 2048 bytes
}
// Total: 2064 bytes
```

**Lookup:** Direct array access: `children[key]`.

### Node Type Summary

| Type | Capacity | Lookup | Size | Best For |
|------|----------|--------|------|----------|
| Node4 | 1-4 | O(4) linear | 64 B | Sparse leaf-adjacent |
| Node16 | 5-16 | O(1) SIMD | 160 B | Common inner nodes |
| Node48 | 17-48 | O(1) indexed | 656 B | Moderately dense |
| Node256 | 49-256 | O(1) direct | 2064 B | Dense (rare) |

---

## Leaf Bucket Design

### Bucket Structure

Buckets store multiple strings with a shared prefix (determined by their position in the trie):

```rust
#[repr(C)]
pub struct LeafBucket {
    header: BucketHeader,           // 32 bytes
    directory: [StringEntry; 256],  // 2048 bytes (sorted)
    data: [u8; 5952],               // Remaining space in 8KB
}

#[repr(C)]
pub struct BucketHeader {
    magic: u32,                     // Validation
    num_entries: u16,               // Current string count
    free_offset: u16,               // Next free byte in data
    min_key: u8,                    // First byte of smallest suffix
    max_key: u8,                    // First byte of largest suffix
    flags: u16,                     // Is_pure, is_sorted, etc.
    prefix_len: u16,                // Shared prefix length (implicit from path)
    lsn: u64,                       // Last modification LSN
    checksum: u64,                  // CRC of contents
}

#[repr(C)]
pub struct StringEntry {
    offset: u16,                    // Offset in data section
    length: u8,                     // Suffix length
    flags: u8,                      // Is_final, has_value, etc.
    value_offset: u16,              // Offset to value (if any)
    value_length: u16,              // Value length
}
```

### Bucket Operations

**Search (binary search within bucket):**

```rust
impl LeafBucket {
    pub fn search(&self, suffix: &[u8]) -> Option<&StringEntry> {
        let entries = &self.directory[..self.header.num_entries as usize];

        entries.binary_search_by(|entry| {
            let stored = self.get_suffix(entry);
            stored.cmp(suffix)
        }).ok().map(|i| &entries[i])
    }

    fn get_suffix(&self, entry: &StringEntry) -> &[u8] {
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        &self.data[start..end]
    }
}
```

**Insert:**

```rust
impl LeafBucket {
    pub fn insert(&mut self, suffix: &[u8], value: &[u8]) -> Result<(), BucketError> {
        if !self.has_space(suffix.len() + value.len()) {
            return Err(BucketError::Full);
        }

        // Find insertion position
        let pos = self.directory[..self.header.num_entries as usize]
            .binary_search_by(|e| self.get_suffix(e).cmp(suffix))
            .unwrap_or_else(|i| i);

        // Shift entries to make room
        let n = self.header.num_entries as usize;
        self.directory.copy_within(pos..n, pos + 1);

        // Write suffix to data section
        let suffix_offset = self.header.free_offset;
        self.data[suffix_offset as usize..][..suffix.len()]
            .copy_from_slice(suffix);

        // Write value
        let value_offset = suffix_offset + suffix.len() as u16;
        self.data[value_offset as usize..][..value.len()]
            .copy_from_slice(value);

        // Create entry
        self.directory[pos] = StringEntry {
            offset: suffix_offset,
            length: suffix.len() as u8,
            flags: StringEntryFlags::IS_FINAL,
            value_offset,
            value_length: value.len() as u16,
        };

        self.header.num_entries += 1;
        self.header.free_offset = value_offset + value.len() as u16;

        Ok(())
    }
}
```

### Bucket Split

When a bucket is full, split it:

```rust
pub fn split_bucket(bucket: &LeafBucket) -> (LeafBucket, LeafBucket, u8) {
    // Find median to balance split
    let mid = bucket.header.num_entries / 2;
    let split_key = bucket.directory[mid as usize].first_byte();

    let mut left = LeafBucket::new();
    let mut right = LeafBucket::new();

    for entry in bucket.entries() {
        if entry.first_byte() < split_key {
            left.insert_entry(entry);
        } else {
            right.insert_entry(entry);
        }
    }

    (left, right, split_key)
}
```

After split, the parent ART node gets a new child pointer for the right bucket.

---

## Trait Integration

### Dictionary Trait Implementation

```rust
impl<V: DictionaryValue> Dictionary for PersistentARTrie<V> {
    type Node = PersistentARTrieNode<V>;

    fn root(&self) -> Self::Node {
        PersistentARTrieNode {
            ptr: self.root_ptr.clone(),
            buffer_mgr: self.buffer_mgr.clone(),
            depth: 0,
        }
    }

    fn contains(&self, term: &str) -> bool {
        self.get(term.as_bytes()).is_some()
    }

    fn len(&self) -> Option<usize> {
        Some(self.entry_count.load(Ordering::Acquire))
    }

    fn sync_strategy(&self) -> SyncStrategy {
        SyncStrategy::InternalSync  // Thread-safe via atomic swizzling
    }
}
```

### DictionaryNode Implementation

This is critical for Levenshtein automata traversal:

```rust
impl<V: DictionaryValue> DictionaryNode for PersistentARTrieNode<V> {
    type Unit = u8;

    fn is_final(&self) -> bool {
        match self.load_node() {
            LoadedNode::Inner(node) => node.header().is_final,
            LoadedNode::Bucket(bucket) => {
                // Check if empty suffix exists in bucket
                bucket.contains_suffix(&[])
            }
        }
    }

    fn transition(&self, label: u8) -> Option<Self> {
        match self.load_node() {
            LoadedNode::Inner(node) => {
                node.find_child(label).map(|child_ptr| {
                    PersistentARTrieNode {
                        ptr: child_ptr,
                        buffer_mgr: self.buffer_mgr.clone(),
                        depth: self.depth + 1,
                    }
                })
            }
            LoadedNode::Bucket(bucket) => {
                // In a bucket, transition means moving to next suffix byte
                if bucket.has_suffix_starting_with(label) {
                    Some(self.bucket_child_node(label))
                } else {
                    None
                }
            }
        }
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> {
        match self.load_node() {
            LoadedNode::Inner(node) => {
                Box::new(node.children().map(move |(label, child_ptr)| {
                    (label, PersistentARTrieNode {
                        ptr: child_ptr,
                        buffer_mgr: self.buffer_mgr.clone(),
                        depth: self.depth + 1,
                    })
                }))
            }
            LoadedNode::Bucket(bucket) => {
                // Iterate unique first bytes of suffixes
                Box::new(bucket.first_bytes().map(move |b| {
                    (b, self.bucket_child_node(b))
                }))
            }
        }
    }
}
```

### MappedDictionary Implementation

```rust
impl<V: DictionaryValue> MappedDictionary for PersistentARTrie<V> {
    type Value = V;

    fn get_value(&self, term: &str) -> Option<V> {
        self.get(term.as_bytes())
    }
}

impl<V: DictionaryValue> MutableMappedDictionary for PersistentARTrie<V> {
    fn insert(&mut self, term: &str, value: V) -> Result<bool, Error> {
        // ... insert implementation with WAL logging
    }

    fn remove(&mut self, term: &str) -> Result<Option<V>, Error> {
        // ... remove implementation with WAL logging
    }
}
```

---

## Levenshtein Traversal Optimization

### Access Pattern Analysis

Levenshtein automata traversal has predictable patterns:

1. **DFS traversal**: Visit children depth-first
2. **Pruning**: Skip subtrees that exceed edit distance
3. **Multiple paths**: Many nodes visited per query

### Prefetching Strategy

Prefetch children while processing current node:

```rust
impl<V: DictionaryValue> PersistentARTrieNode<V> {
    pub fn prefetch_children(&self) {
        if let LoadedNode::Inner(node) = self.load_node() {
            for child_ptr in node.child_pointers() {
                if !child_ptr.is_swizzled() {
                    let (block_id, _, _) = child_ptr.disk_location().unwrap();
                    self.buffer_mgr.prefetch_async(block_id);
                }
            }
        }
    }
}

// Integration with transducer
impl<D: Dictionary> QueryIterator<D> {
    fn advance_with_prefetch(&mut self) {
        if let Some(node) = self.current_node() {
            // Prefetch before processing
            if let Some(art_node) = node.as_persistent_art() {
                art_node.prefetch_children();
            }
        }
        // ... normal advance logic
    }
}
```

### Bucket-Aware Traversal

For buckets, we can batch-check Levenshtein candidates:

```rust
impl LeafBucket {
    /// Check all strings in bucket against Levenshtein automaton
    pub fn levenshtein_matches<A: LevenshteinAutomaton>(
        &self,
        automaton: &A,
        prefix: &[u8],  // Path to this bucket
    ) -> Vec<(String, usize)> {
        let mut matches = Vec::new();

        for entry in self.entries() {
            let suffix = self.get_suffix(entry);
            let full_term: Vec<u8> = prefix.iter()
                .chain(suffix.iter())
                .copied()
                .collect();

            if let Some(distance) = automaton.eval(&full_term) {
                matches.push((
                    String::from_utf8_lossy(&full_term).into_owned(),
                    distance,
                ));
            }
        }

        matches
    }
}
```

### Cache Pinning for Hot Paths

Pin frequently-accessed nodes:

```rust
pub struct PersistentARTrie<V> {
    // ...
    hot_nodes: RwLock<Vec<PageId>>,  // Permanently pinned pages
}

impl<V: DictionaryValue> PersistentARTrie<V> {
    pub fn warm_cache(&self, depth: usize) {
        // Pin root and first `depth` levels
        let mut hot = Vec::new();
        self.collect_hot_pages(&self.root, depth, &mut hot);

        for page_id in &hot {
            self.buffer_mgr.pin_permanent(*page_id);
        }

        *self.hot_nodes.write() = hot;
    }
}
```

---

## File Format

### File Organization

```
persistent_artrie.db
├── Header (4KB)
│   ├── Magic number
│   ├── Version
│   ├── Root pointer
│   ├── Entry count
│   └── Metadata
├── Index blocks (ART nodes)
│   └── 256KB blocks packed with nodes
├── Leaf blocks (Buckets)
│   └── 8KB buckets
└── Free list
    └── Available block IDs
```

### Header Block

```rust
#[repr(C)]
pub struct FileHeader {
    magic: [u8; 8],           // "PART_v01"
    version: u32,             // Format version
    flags: u32,               // Compression, etc.
    root_block: u32,          // Root node block ID
    root_offset: u32,         // Root node offset in block
    entry_count: u64,         // Total strings stored
    block_count: u64,         // Total blocks allocated
    index_block_count: u64,   // ART node blocks
    leaf_block_count: u64,    // Bucket blocks
    checksum: u64,            // Header checksum
    created_at: u64,          // Unix timestamp
    modified_at: u64,         // Unix timestamp
    reserved: [u8; 4008],     // Pad to 4KB
}
```

### Block Allocation

```
Block ID allocation:
  0          : Header
  1..1M      : Index blocks (ART nodes)
  1M..8M     : Leaf blocks (Buckets)
  8M+        : Overflow (if needed)
```

---

## Implementation Roadmap

### Phase 1: Storage Foundation

**Files to create:**
- `src/dictionary/persistent_artrie/mod.rs`
- `src/dictionary/persistent_artrie/swizzled_ptr.rs`
- `src/dictionary/persistent_artrie/buffer_manager.rs`

**Tasks:**
- [ ] SwizzledPtr with atomic operations
- [ ] BufferManager with LRU cache
- [ ] Memory-mapped file I/O
- [ ] Basic WAL for crash recovery

### Phase 2: ART Node Layer

**Files to create:**
- `src/dictionary/persistent_artrie/nodes.rs`
- `src/dictionary/persistent_artrie/node_ops.rs`

**Tasks:**
- [ ] Node4, Node16, Node48, Node256 structures
- [ ] Path compression handling
- [ ] Node growth transitions (4→16→48→256)
- [ ] SIMD optimization for Node16
- [ ] Serialization/deserialization

### Phase 3: Bucket Layer

**Files to create:**
- `src/dictionary/persistent_artrie/bucket.rs`

**Tasks:**
- [ ] LeafBucket structure (8KB)
- [ ] Binary search insert/lookup
- [ ] Split algorithm
- [ ] Bucket ↔ ART transition

### Phase 4: Dictionary Trait Integration

**Files to create:**
- `src/dictionary/persistent_artrie/dict_impl.rs`
- `src/dictionary/persistent_artrie/node_impl.rs`

**Tasks:**
- [ ] `PersistentARTrieNode` implementing `DictionaryNode`
- [ ] `Dictionary`, `MappedDictionary` implementations
- [ ] `MutableMappedDictionary` for insert/remove
- [ ] Integration tests with transducers

### Phase 5: Optimization

**Tasks:**
- [ ] Prefetching for Levenshtein traversal
- [ ] Cache pinning for hot paths
- [ ] Bulk loading optimization
- [ ] Benchmark suite

### Phase 6: UTF-8 Variant

**Files to create:**
- `src/dictionary/persistent_artrie_char/`

**Tasks:**
- [ ] `PersistentARTrieChar` for `char` units
- [ ] 4-byte key handling in nodes

---

## Complexity Summary

| Operation | Time Complexity | Expected Disk I/Os |
|-----------|-----------------|-------------------|
| Exact lookup | O(m) | 2-4 |
| Insert | O(m + log B) amortized | 2-4 + 1 write |
| Delete | O(m + log B) | 2-4 + 1 write |
| Prefix search | O(m + k) | O(m/fanout + k/B) |
| Levenshtein (d=1) | O(n·m) | Varies with pruning |
| Levenshtein (d=2) | O(n·m·d) | Varies with pruning |

Where:
- m = term length
- B = bucket size (~100-500 strings)
- k = result count
- n = dictionary size
- d = edit distance

---

## Summary

The Persistent ARTrie design combines:

1. **Adaptive ART nodes** for the index layer
2. **B-trie-style buckets** for leaf storage
3. **Swizzled pointers** for transparent memory/disk addressing
4. **Buffer management** with LRU eviction
5. **WAL** for crash recovery
6. **DictionaryNode** trait for Levenshtein automata integration

This hybrid approach provides:
- Low-latency lookups (2-4 I/Os typical)
- Efficient space usage (adaptive nodes + packed buckets)
- Native Levenshtein automata support
- Scalability to TB-scale dictionaries

---

## References

1. Askitis, N. & Zobel, J. (2009). "B-tries for disk-based string management." *VLDB Journal*.

2. Leis, V., Kemper, A., & Neumann, T. (2013). "The Adaptive Radix Tree." *ICDE*.

3. DuckDB Team. (2022). "Persistent Storage of Adaptive Radix Trees in DuckDB."

4. Luo, X. et al. (2023). "SMART: A High-Performance Adaptive Radix Tree for Disaggregated Memory." *OSDI*.

5. Binna, R. et al. (2018). "HOT: A Height Optimized Trie Index." *SIGMOD*.
