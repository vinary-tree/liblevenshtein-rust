# Persistent ART: Disk Storage Techniques

This document examines techniques for persisting Adaptive Radix Trees to disk storage. We focus on pointer swizzling, serialization strategies, and on-demand loading—techniques developed by DuckDB and refined by subsequent research.

## Table of Contents

1. [The Persistence Challenge](#the-persistence-challenge)
2. [Pointer Swizzling](#pointer-swizzling)
3. [Serialization Strategy](#serialization-strategy)
4. [On-Demand Loading](#on-demand-loading)
5. [Block Layout](#block-layout)
6. [Concurrency Considerations](#concurrency-considerations)
7. [Lessons for Persistent ARTrie](#lessons-for-persistent-artrie)

---

## The Persistence Challenge

### Memory Pointers vs. Disk Offsets

In-memory ART nodes contain raw pointers to child nodes:

```rust
struct Node16 {
    // ... header fields ...
    children: [*mut Node; 16],  // Raw memory pointers
}
```

These pointers are:
- **Process-specific**: Valid only within one process's address space
- **Session-specific**: Invalid after process restart
- **Non-portable**: Different across machines

To persist ART to disk, we need to:
1. Convert memory pointers to disk locations
2. Reconstruct pointers when loading from disk
3. Handle partial loading (not everything fits in RAM)

### Naive Approaches and Their Problems

**Approach 1: Full serialization**
```
Write entire tree to disk → Read entire tree on startup
```
Problems:
- Startup time proportional to tree size
- RAM must hold entire tree
- Changes require rewriting entire structure

**Approach 2: Pointer-to-offset translation**
```
Write nodes with disk offsets → Translate on every access
```
Problems:
- Extra indirection on every child access
- Cannot use native memory operations
- Poor cache behavior

**Approach 3: Address space persistence (mmap)**
```
mmap file at fixed address → Use raw pointers
```
Problems:
- Requires ASLR disabled (security risk)
- Cannot share files across processes
- Fragile across OS updates

---

## Pointer Swizzling

Pointer swizzling provides an elegant solution: use a single 64-bit value that can represent either a memory pointer or a disk location.

### The Swizzled Pointer Design

```
┌────────────────────────────────────────────────────────────────┐
│ SwizzledPtr (64 bits)                                          │
├────────────────────────────────────────────────────────────────┤
│ Bit 63 (MSB): Swizzle flag                                     │
│   1 = Memory pointer (remaining 63 bits are address)           │
│   0 = Disk reference (page_id + offset encoding)               │
├────────────────────────────────────────────────────────────────┤
│ When MSB = 1 (in-memory):                                      │
│   Bits 62-0: Memory pointer (mask off MSB to get address)      │
├────────────────────────────────────────────────────────────────┤
│ When MSB = 0 (on-disk):                                        │
│   Option A: Bits 62-0 = raw file offset                        │
│   Option B: Bits 62-40 = page_id, Bits 39-0 = offset in page   │
│   Option C: Bits 62-24 = block_id, Bits 23-0 = offset in block │
└────────────────────────────────────────────────────────────────┘
```

### Why the MSB Works

On modern 64-bit systems:
- Virtual addresses use at most 48 bits (AMD64) or 52 bits (Intel 5-level)
- User-space addresses typically have bit 63 = 0
- Kernel addresses have bit 63 = 1, but we don't store kernel pointers

Thus, setting bit 63 = 1 for valid heap pointers creates a distinguishable encoding.

### Rust Implementation

```rust
use std::sync::atomic::{AtomicU64, Ordering};

const SWIZZLE_FLAG: u64 = 1 << 63;
const PTR_MASK: u64 = !SWIZZLE_FLAG;

#[repr(transparent)]
pub struct SwizzledPtr(AtomicU64);

impl SwizzledPtr {
    /// Create a new unswizzled (on-disk) pointer
    pub fn on_disk(block_id: u32, offset: u32) -> Self {
        let encoded = ((block_id as u64) << 24) | (offset as u64);
        debug_assert!(encoded & SWIZZLE_FLAG == 0);
        Self(AtomicU64::new(encoded))
    }

    /// Create a new swizzled (in-memory) pointer
    pub fn in_memory(ptr: *mut Node) -> Self {
        let addr = ptr as u64;
        debug_assert!(addr & SWIZZLE_FLAG == 0, "High bit must be clear");
        Self(AtomicU64::new(addr | SWIZZLE_FLAG))
    }

    /// Check if pointer is swizzled (in memory)
    pub fn is_swizzled(&self) -> bool {
        self.0.load(Ordering::Acquire) & SWIZZLE_FLAG != 0
    }

    /// Get memory pointer (panics if not swizzled)
    pub fn as_ptr(&self) -> *mut Node {
        let val = self.0.load(Ordering::Acquire);
        assert!(val & SWIZZLE_FLAG != 0, "Pointer not swizzled");
        (val & PTR_MASK) as *mut Node
    }

    /// Get disk location (panics if swizzled)
    pub fn disk_location(&self) -> (u32, u32) {
        let val = self.0.load(Ordering::Acquire);
        assert!(val & SWIZZLE_FLAG == 0, "Pointer is swizzled");
        let block_id = (val >> 24) as u32;
        let offset = (val & 0xFFFFFF) as u32;
        (block_id, offset)
    }

    /// Atomically swizzle: replace disk ref with memory pointer
    pub fn swizzle(&self, ptr: *mut Node) -> bool {
        let old = self.0.load(Ordering::Acquire);
        if old & SWIZZLE_FLAG != 0 {
            return false;  // Already swizzled
        }
        let new = (ptr as u64) | SWIZZLE_FLAG;
        self.0.compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    /// Atomically unswizzle: replace memory pointer with disk ref
    pub fn unswizzle(&self, block_id: u32, offset: u32) -> Option<*mut Node> {
        let old = self.0.load(Ordering::Acquire);
        if old & SWIZZLE_FLAG == 0 {
            return None;  // Already unswizzled
        }
        let new = ((block_id as u64) << 24) | (offset as u64);
        if self.0.compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            Some((old & PTR_MASK) as *mut Node)
        } else {
            None
        }
    }
}
```

### Atomic Swizzling for Concurrency

The `compare_exchange` ensures only one thread successfully swizzles a pointer:

```
Thread A                        Thread B
────────                        ────────
load ptr (sees disk ref)
                                load ptr (sees disk ref)
read from disk
allocate node
                                read from disk
                                allocate node
CAS(disk→memory) SUCCESS
                                CAS(disk→memory) FAILS
return node
                                sees swizzled ptr
                                frees allocated node
                                return existing node
```

Both threads get the same node; the losing thread just does redundant work.

---

## Serialization Strategy

### Post-Order Traversal

To serialize an ART, we use post-order traversal: children are written before their parents. This ensures that when writing a parent, all child offsets are known.

```rust
fn serialize_tree(root: &Node, writer: &mut BlockWriter) -> DiskRef {
    match root {
        Node::Leaf(leaf) => {
            writer.write_leaf(leaf)
        }
        Node::Inner(inner) => {
            // First, serialize all children
            let child_refs: Vec<DiskRef> = inner.children()
                .map(|child| serialize_tree(child, writer))
                .collect();

            // Then write this node with child references
            writer.write_inner_node(inner, &child_refs)
        }
    }
}
```

### Block Allocation Strategies

**Strategy 1: Sequential allocation**
```
┌──────────────────────────────────────────────────────────────┐
│ Block 0                                                       │
│ ┌─────────┬─────────┬─────────┬─────────┬──────────────────┐ │
│ │ Node A  │ Node B  │ Node C  │ Node D  │ (free space)     │ │
│ └─────────┴─────────┴─────────┴─────────┴──────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ Block 1                                                       │
│ ┌─────────┬─────────┬──────────────────────────────────────┐ │
│ │ Node E  │ Node F  │ (free space)                         │ │
│ └─────────┴─────────┴──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

Simple but leads to fragmentation over time.

**Strategy 2: Locality-aware allocation**

Place parent near children for better cache/prefetch behavior:

```
┌──────────────────────────────────────────────────────────────┐
│ Block 0: Subtree rooted at A                                  │
│ ┌─────────┬─────────┬─────────┬─────────┬──────────────────┐ │
│ │ Node A  │ Child 1 │ Child 2 │ Child 3 │ grandchildren... │ │
│ └─────────┴─────────┴─────────┴─────────┴──────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ Block 1: Subtree rooted at B                                  │
│ ┌─────────┬─────────────────────────────────────────────────┐ │
│ │ Node B  │ B's subtree...                                  │ │
│ └─────────┴─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Variable-Size Node Serialization

ART nodes have different sizes. We serialize with type tags:

```rust
fn serialize_node(node: &Node, buffer: &mut Vec<u8>) -> usize {
    let start = buffer.len();

    // Write type tag
    buffer.push(node.node_type() as u8);

    // Write common header
    buffer.push(node.partial_len());
    buffer.extend_from_slice(&node.partial()[..node.partial_len()]);

    // Write type-specific data
    match node {
        Node::Node4(n) => {
            buffer.push(n.num_children);
            buffer.extend_from_slice(&n.keys[..n.num_children]);
            for i in 0..n.num_children {
                serialize_swizzled_ptr(&n.children[i], buffer);
            }
        }
        Node::Node16(n) => {
            buffer.push(n.num_children);
            buffer.extend_from_slice(&n.keys[..16]);  // Full 16 for alignment
            for i in 0..n.num_children {
                serialize_swizzled_ptr(&n.children[i], buffer);
            }
        }
        // ... Node48, Node256 ...
    }

    buffer.len() - start
}
```

---

## On-Demand Loading

### Lazy Swizzling

The key insight: don't load the entire tree. Load nodes on-demand during traversal.

```rust
fn get_child(&self, key: u8, buffer_mgr: &BufferManager) -> Option<&Node> {
    let child_ptr = self.find_child_ptr(key)?;

    if child_ptr.is_swizzled() {
        // Fast path: already in memory
        Some(unsafe { &*child_ptr.as_ptr() })
    } else {
        // Slow path: load from disk
        let (block_id, offset) = child_ptr.disk_location();
        let node = buffer_mgr.load_node(block_id, offset);
        child_ptr.swizzle(node);  // Atomic; might fail if another thread swizzled
        Some(unsafe { &*child_ptr.as_ptr() })
    }
}
```

### Pinning During Traversal

When traversing, pin pages to prevent eviction:

```rust
fn lookup(&self, key: &[u8]) -> Option<&Value> {
    let mut pins: Vec<PagePin> = Vec::new();
    let mut node = &self.root;
    let mut depth = 0;

    while depth < key.len() {
        // Pin current page
        if !node.is_in_root_page() {
            pins.push(self.buffer_mgr.pin(node.page_id()));
        }

        // Navigate to child
        match node.get_child(key[depth], &self.buffer_mgr) {
            Some(child) => {
                node = child;
                depth += 1;
            }
            None => return None,
        }

        // Optionally release old pins to limit memory
        if pins.len() > MAX_PIN_DEPTH {
            pins.remove(0);  // Unpin oldest
        }
    }

    node.value()
    // Pins released when `pins` drops
}
```

### Prefetching

For predictable access patterns (e.g., DFS for Levenshtein automata), prefetch children:

```rust
fn prefetch_children(&self, buffer_mgr: &BufferManager) {
    for child_ptr in self.child_pointers() {
        if !child_ptr.is_swizzled() {
            let (block_id, offset) = child_ptr.disk_location();
            buffer_mgr.prefetch_async(block_id);
        }
    }
}

// During Levenshtein traversal
fn traverse_with_prefetch(&self, ...) {
    // Prefetch children of current node while processing
    self.prefetch_children(buffer_mgr);

    for (label, child) in self.edges() {
        if automaton.can_match(label) {
            traverse_with_prefetch(child, ...);
        }
    }
}
```

---

## Block Layout

### Block Size Selection

| Block Size | Pros | Cons |
|------------|------|------|
| 4 KB | Matches OS page size, fine-grained | More blocks, more metadata |
| 16 KB | Good for SSDs | Moderate overhead |
| 64 KB | Reduced metadata | May waste space |
| 256 KB | Matches NVMe optimal I/O | Large minimum allocation |

For NVMe SSDs with 128KB-256KB optimal I/O size, larger blocks amortize the per-I/O overhead.

### Block Header

```
┌─────────────────────────────────────────────────────────────────┐
│ Block Header (64 bytes)                                          │
├─────────────────────────────────────────────────────────────────┤
│ magic: u32           │ Identifies valid block                   │
│ version: u16         │ Format version                           │
│ block_type: u8       │ 0=nodes, 1=buckets, 2=metadata           │
│ flags: u8            │ Compression, etc.                        │
│ block_id: u32        │ This block's ID                          │
│ checksum: u64        │ CRC64 of contents                        │
│ num_entries: u16     │ Number of nodes/entries                  │
│ free_offset: u16     │ Start of free space                      │
│ prev_block: u32      │ Previous block in chain (or 0)           │
│ next_block: u32      │ Next block in chain (or 0)               │
│ padding: [u8; 28]    │ Reserved for future use                  │
└─────────────────────────────────────────────────────────────────┘
```

### Node Packing Within Blocks

```
┌───────────────────────────────────────────────────────────────────┐
│ Block (256 KB)                                                     │
├───────────────────────────────────────────────────────────────────┤
│ Header (64 B)                                                      │
├───────────────────────────────────────────────────────────────────┤
│ Node Directory (variable)                                          │
│ ┌────────────┬────────────┬────────────┬────────────────────────┐ │
│ │ entry[0]   │ entry[1]   │ entry[2]   │ ...                    │ │
│ │ off:64,    │ off:112,   │ off:272,   │                        │ │
│ │ len:48     │ len:160    │ len:656    │                        │ │
│ └────────────┴────────────┴────────────┴────────────────────────┘ │
├───────────────────────────────────────────────────────────────────┤
│ Node Data                                                          │
│ ┌────────────────────────────────────────────────────────────────┐│
│ │ [Node4 @ 64] [Node16 @ 112] [Node48 @ 272] [Node4 @ 928] ...  ││
│ └────────────────────────────────────────────────────────────────┘│
├───────────────────────────────────────────────────────────────────┤
│ Free Space                                                         │
└───────────────────────────────────────────────────────────────────┘
```

### Alignment Considerations

For SIMD operations (Node16), ensure 16-byte alignment:

```rust
fn allocate_in_block(block: &mut Block, size: usize, align: usize) -> Option<u32> {
    let current = block.free_offset as usize;
    let aligned = (current + align - 1) & !(align - 1);
    let end = aligned + size;

    if end > block.capacity() {
        return None;
    }

    block.free_offset = end as u16;
    Some(aligned as u32)
}

// For Node16, request 16-byte alignment
let offset = allocate_in_block(&mut block, size_of::<Node16>(), 16)?;
```

---

## Concurrency Considerations

### Read-Only Swizzling

Multiple readers can safely swizzle simultaneously:

```rust
// Safe: multiple threads may race to swizzle the same pointer
// Worst case: some threads load redundantly, but all get correct result
fn concurrent_lookup(&self, key: &[u8]) -> Option<&Value> {
    let node = self.get_child_swizzling(key[0])?;  // May race
    // ...
}
```

### Writes Require Coordination

For insert/delete with concurrent readers:

**Option 1: Copy-on-write**
```
1. Create modified copy of node
2. Atomically swap parent's child pointer
3. Old node becomes garbage (collect later)
```

**Option 2: Optimistic lock coupling**
```
1. Acquire version lock on parent
2. Modify child pointer
3. Increment version, release lock
4. Readers retry if version changed mid-read
```

**Option 3: Epoch-based reclamation**
```
1. Readers register in current epoch
2. Writers defer frees to "safe" epoch
3. Reclaim when no readers in old epochs
```

### DuckDB's Approach

DuckDB uses copy-on-write for its ART:

```rust
fn insert_cow(&mut self, key: &[u8], value: Value) -> Result<()> {
    let mut path: Vec<(*mut Node, usize)> = Vec::new();

    // Traverse, recording path
    let mut node = &mut self.root;
    let mut depth = 0;
    while depth < key.len() {
        path.push((node as *mut _, depth));
        node = node.get_child_mut(key[depth])?;
        depth += 1;
    }

    // Modify leaf, propagate copies upward
    let mut new_node = node.clone_with_modification(...);
    for (parent, d) in path.into_iter().rev() {
        let parent = unsafe { &mut *parent };
        let new_parent = parent.clone_with_child_replaced(key[d], new_node);
        new_node = new_parent;
    }

    self.root = new_node;
    Ok(())
}
```

---

## Lessons for Persistent ARTrie

### 1. Swizzled Pointers Enable Lazy Loading

The MSB-flag technique gives us:
- Native pointer performance when swizzled
- Compact on-disk representation
- Atomic swizzle operations for concurrent readers

### 2. Block Size Matters for I/O Efficiency

For SSDs:
- 256 KB blocks match optimal NVMe I/O
- Larger blocks amortize header overhead
- Pack multiple small nodes per block

### 3. Locality-Aware Allocation Improves Prefetching

When serializing:
- Keep subtrees together in blocks
- Parent nodes near children
- Enables effective prefetching during traversal

### 4. Copy-on-Write Simplifies Concurrency

For our use case:
- Levenshtein traversal is mostly read-only
- Inserts can use COW without blocking readers
- Epoch-based reclamation handles deferred frees

### 5. Checksum Everything

For crash recovery:
- Block-level checksums detect corruption
- Log checksums validate WAL entries
- Enables safe recovery after crash

### 6. Separate Index and Leaf Storage

Following B-trie lessons:
- ART nodes for index (inner) layer
- B-trie-style buckets for leaves
- Amortize leaf I/O across multiple strings

---

## Summary

Persisting ART to disk requires:

1. **Swizzled pointers**: Single 64-bit value for memory or disk reference
2. **Post-order serialization**: Write children before parents
3. **On-demand loading**: Lazy swizzle during traversal
4. **Block-based storage**: Pack nodes into large I/O units
5. **Concurrency handling**: Atomic swizzle, COW for writes

The next document covers buffer management: the page cache, LRU eviction, and crash recovery mechanisms that complete our storage layer.

---

## References

1. DuckDB Team. (2022). "Persistent Storage of Adaptive Radix Trees in DuckDB." [Blog Post](https://duckdb.org/2022/07/27/art-storage)

2. Luo, X., Luo, L., Zheng, W., & Kuo, T. W. (2023). "SMART: A High-Performance Adaptive Radix Tree for Disaggregated Memory." *OSDI*. [PDF](https://www.usenix.org/system/files/osdi23-luo.pdf)

3. Graefe, G. (2011). "Modern B-Tree Techniques." *Foundations and Trends in Databases*.

4. Leis, V., Haubenschild, M., Kemper, A., & Neumann, T. (2018). "LeanStore: In-Memory Data Management Beyond Main Memory." *ICDE*.

5. Neumann, T. & Leis, V. (2020). "Umbra: A Disk-Based System with In-Memory Performance." *CIDR*.
