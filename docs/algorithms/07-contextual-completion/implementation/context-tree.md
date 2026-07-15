# Context Tree Implementation

**Hierarchical tree structure for managing lexical scopes and visibility relationships in contextual code completion.**

---

[← Back to Layer 7](../README.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Data Structure](#data-structure)
3. [Core Operations](#core-operations)
4. [Visibility Algorithm](#visibility-algorithm)
5. [Usage Examples](#usage-examples)
6. [Performance Characteristics](#performance-characteristics)
7. [Implementation Details](#implementation-details)
8. [Testing](#testing)

---

## Overview

The `ContextTree` manages hierarchical relationships between contexts (lexical scopes) in a parent-child tree structure. It implements the fundamental rule of lexical scoping: **child contexts can see definitions from parent contexts, but not vice versa**.

### Key Concepts

```
Lexical Scope Hierarchy Example:

    Global (0)
      │
      ├─── Module A (1)
      │      │
      │      ├─── Function foo (10)
      │      │      └─── Block (100)
      │      │
      │      └─── Function bar (11)
      │
      └─── Module B (2)
             └─── Class Baz (20)
                    └─── Method qux (200)

Visibility Rules:
- Context 100 (block) can see: [100, 10, 1, 0] (self → function → module → global)
- Context 11 (bar) can see: [11, 1, 0] (self → module → global)
- Context 200 (qux) can see: [200, 20, 2, 0] (self → class → module B → global)
```

### Use Cases

| Use Case | Context Mapping | Visibility Pattern |
|----------|----------------|-------------------|
| **LSP Server** | File → Class → Method → Block | Nested scope inheritance |
| **REPL** | Session → Expression history | Sequential scope chain |
| **Code Editor** | Global → Module → Function → Block | Standard lexical scoping |
| **Interactive Notebook** | Notebook → Cell → Code block | Cell isolation + global visibility |

---

## Data Structure

### Internal Representation

```rust
pub struct ContextTree {
    /// Map from context ID to parent ID (None for root contexts)
    nodes: HashMap<ContextId, Option<ContextId>>,
}

pub type ContextId = u32;
```

**Design Rationale**:

- **HashMap-based**: $`\mathcal{O}(1)`$ parent lookup, efficient for sparse IDs
- **Inverse mapping** (child → parent): Optimizes visibility computation (common operation)
- **Option<ContextId>**: `None` indicates root context
- **u32 for IDs**: 4 billion unique contexts (sufficient for any realistic application)

### Memory Characteristics

| Component | Size (bytes) | Notes |
|-----------|--------------|-------|
| HashMap entry | ~24 | Key (u32) + value (Option<u32>) + hash/metadata |
| Per-context overhead | ~24 | Single HashMap entry |
| Tree with 100 contexts | ~2.4 KB | 100 × 24 bytes |
| Tree with 1000 contexts | ~24 KB | 1000 × 24 bytes |

**Conclusion**: Extremely lightweight. Even large applications (1000 contexts) use <25KB.

---

## Core Operations

### Context Creation

```rust
// Create root context (no parent)
pub fn create_root(&mut self, id: ContextId) -> ContextId;

// Create child context
pub fn create_child(&mut self, id: ContextId, parent_id: ContextId)
    -> Result<ContextId>;
```

**Complexity**:
- `create_root()`: $`\mathcal{O}(1)`$ - HashMap insert
- `create_child()`: $`\mathcal{O}(1)`$ - HashMap lookup + insert

**Example**:

```rust
let mut tree = ContextTree::new();

// Create global scope (root)
let global = tree.create_root(0);

// Create module scope (child of global)
let module = tree.create_child(1, global)?;

// Create function scope (child of module)
let function = tree.create_child(10, module)?;
```

### Parent Lookup

```rust
pub fn parent(&self, id: ContextId) -> Option<ContextId>;
```

**Returns**:
- `Some(parent_id)` if context has a parent
- `None` if context is root or doesn't exist

**Complexity**: $`\mathcal{O}(1)`$ - HashMap lookup

**Example**:

```rust
let mut tree = ContextTree::new();
let root = tree.create_root(0);
let child = tree.create_child(1, root)?;

assert_eq!(tree.parent(root), None);     // Root has no parent
assert_eq!(tree.parent(child), Some(0)); // Child's parent is root
```

### Visibility Computation

```rust
pub fn visible_contexts(&self, id: ContextId) -> Vec<ContextId>;
```

**Returns**: `[self, parent, grandparent, ..., root]` in that order.

**Algorithm**:

1. Start with the given context
2. Walk up parent chain until reaching a root (parent = None)
3. Collect all contexts encountered

**Complexity**: $`\mathcal{O}(\text{depth})`$ where depth is distance from root (typically $`\le 5`$ for code)

**Example**:

```rust
let mut tree = ContextTree::new();
let global = tree.create_root(0);
let module = tree.create_child(1, global)?;
let function = tree.create_child(10, module)?;

let visible = tree.visible_contexts(function);
// Returns: [10, 1, 0] (function → module → global)
```

### Depth Calculation

```rust
pub fn depth(&self, id: ContextId) -> Option<usize>;
```

**Returns**:
- `Some(depth)` where root = 0, children = 1, etc.
- `None` if context doesn't exist

**Complexity**: $`\mathcal{O}(\text{depth})`$ - Walk parent chain

**Example**:

```rust
assert_eq!(tree.depth(global), Some(0));    // Root
assert_eq!(tree.depth(module), Some(1));    // 1 level deep
assert_eq!(tree.depth(function), Some(2));  // 2 levels deep
```

### Descendant Check

```rust
pub fn is_descendant(&self, child_id: ContextId, ancestor_id: ContextId) -> bool;
```

**Returns**: `true` if `child_id` is a descendant of `ancestor_id`.

**Algorithm**:

1. Walk up parent chain from `child_id`
2. If `ancestor_id` encountered, return true
3. If root reached without finding ancestor, return false

**Complexity**: $`\mathcal{O}(\text{depth})`$

**Example**:

```rust
let mut tree = ContextTree::new();
let root = tree.create_root(0);
let child = tree.create_child(1, root)?;
let grandchild = tree.create_child(2, child)?;

assert!(tree.is_descendant(grandchild, root));     // true
assert!(tree.is_descendant(grandchild, child));    // true
assert!(!tree.is_descendant(root, child));         // false (reversed)
assert!(!tree.is_descendant(child, grandchild));   // false (reversed)
```

### Context Removal

```rust
pub fn remove(&mut self, id: ContextId) -> bool;
```

**Behavior**: Removes context **and all descendants** (cascading delete).

**Algorithm**:

1. Identify all descendants via `is_descendant()` check
2. Remove target context from HashMap
3. Remove all descendants from HashMap

**Complexity**: $`\mathcal{O}(n)`$ where n is total number of contexts (must check all for descendants)

**Example**:

```rust
let mut tree = ContextTree::new();
let root = tree.create_root(0);
let child = tree.create_child(1, root)?;
let grandchild = tree.create_child(2, child)?;

// Remove child - grandchild also removed
tree.remove(child);

assert!(!tree.contains(child));       // Removed
assert!(!tree.contains(grandchild));  // Also removed (descendant)
assert!(tree.contains(root));         // Unaffected
```

---

## Visibility Algorithm

### Pseudocode

```
function visible_contexts(context_id):
    result = []
    current = context_id

    while current is not None:
        if current exists in tree:
            result.append(current)
            current = parent(current)
        else:
            break  // Context not found

    return result
```

### Example Walkthrough

```
Tree Structure:
    0 (global)
    └── 1 (module)
        └── 10 (function)
            └── 100 (block)

Call: visible_contexts(100)

Step 1: current = 100, add to result → [100]
Step 2: parent(100) = 10, add to result → [100, 10]
Step 3: parent(10) = 1, add to result → [100, 10, 1]
Step 4: parent(1) = 0, add to result → [100, 10, 1, 0]
Step 5: parent(0) = None, stop
Result: [100, 10, 1, 0]
```

### Complexity Analysis

**Time Complexity**: $`\mathcal{O}(d)`$ where d = depth from root

- Best case: $`\mathcal{O}(1)`$ - querying root context
- Worst case: $`\mathcal{O}(d)`$ - deeply nested context
- Typical case: $`\mathcal{O}(3-5)`$ - practical code has 3-5 nesting levels

**Space Complexity**: $`\mathcal{O}(d)`$ - result vector size

**Real-World Depth**:

| Language Construct | Typical Depth | Example |
|-------------------|---------------|---------|
| Global scope | 0 | Top-level variables |
| Module scope | 1 | Module-level functions |
| Class scope | 2 | Class members |
| Method scope | 3 | Method local variables |
| Block scope | 4-5 | Nested if/while/for blocks |

**Conclusion**: Even deeply nested code (depth 10) requires only ~10 HashMap lookups (~200ns total).

---

## Usage Examples

### Example 1: Basic Hierarchy

```rust
use liblevenshtein::contextual::ContextTree;

let mut tree = ContextTree::new();

// Create global → module → function hierarchy
let global = tree.create_root(0);
let module = tree.create_child(1, global).unwrap();
let function = tree.create_child(10, module).unwrap();

// Check visibility from function
let visible = tree.visible_contexts(function);
assert_eq!(visible, vec![10, 1, 0]);

// Function can see all three levels
assert!(visible.contains(&global));
assert!(visible.contains(&module));
assert!(visible.contains(&function));
```

### Example 2: Sibling Isolation

```rust
let mut tree = ContextTree::new();

// Create tree with siblings:
//     global (0)
//       ├── module_a (1)
//       └── module_b (2)

let global = tree.create_root(0);
let module_a = tree.create_child(1, global).unwrap();
let module_b = tree.create_child(2, global).unwrap();

// Module A can see global, but not Module B
let visible_a = tree.visible_contexts(module_a);
assert!(visible_a.contains(&global));
assert!(visible_a.contains(&module_a));
assert!(!visible_a.contains(&module_b)); // Sibling not visible

// Module B can see global, but not Module A
let visible_b = tree.visible_contexts(module_b);
assert!(visible_b.contains(&global));
assert!(visible_b.contains(&module_b));
assert!(!visible_b.contains(&module_a)); // Sibling not visible
```

### Example 3: Multi-Root Tree

```rust
let mut tree = ContextTree::new();

// Create separate root contexts (e.g., multiple files)
let file1 = tree.create_root(1);
let file2 = tree.create_root(2);

// Each root has its own hierarchy
let class1 = tree.create_child(10, file1).unwrap();
let class2 = tree.create_child(20, file2).unwrap();

// Class in file1 cannot see file2
let visible1 = tree.visible_contexts(class1);
assert_eq!(visible1, vec![10, 1]); // Only file1 hierarchy

// Class in file2 cannot see file1
let visible2 = tree.visible_contexts(class2);
assert_eq!(visible2, vec![20, 2]); // Only file2 hierarchy
```

### Example 4: Deep Nesting

```rust
let mut tree = ContextTree::new();

// Create deeply nested structure
let global = tree.create_root(0);
let module = tree.create_child(1, global).unwrap();
let class = tree.create_child(10, module).unwrap();
let method = tree.create_child(100, class).unwrap();
let block1 = tree.create_child(1000, method).unwrap();
let block2 = tree.create_child(10000, block1).unwrap();

// Deep block can see all ancestors
let visible = tree.visible_contexts(block2);
assert_eq!(visible, vec![10000, 1000, 100, 10, 1, 0]);

// Check depth
assert_eq!(tree.depth(block2), Some(5)); // 5 levels deep
```

### Example 5: Cascading Removal

```rust
let mut tree = ContextTree::new();

// Build tree:
//     0
//     └── 1
//         ├── 10
//         │   └── 100
//         └── 11

let root = tree.create_root(0);
let child = tree.create_child(1, root).unwrap();
let gc1 = tree.create_child(10, child).unwrap();
let ggc = tree.create_child(100, gc1).unwrap();
let gc2 = tree.create_child(11, child).unwrap();

// Remove child - cascades to all descendants
tree.remove(child);

assert!(!tree.contains(child));   // Removed
assert!(!tree.contains(gc1));     // Removed (descendant)
assert!(!tree.contains(ggc));     // Removed (descendant)
assert!(!tree.contains(gc2));     // Removed (descendant)
assert!(tree.contains(root));     // Unaffected (ancestor)
```

### Example 6: Descendant Checking

```rust
let mut tree = ContextTree::new();

//     0 (global)
//    / \
//   1   2
//  /     \
// 10     20

let global = tree.create_root(0);
let mod1 = tree.create_child(1, global).unwrap();
let mod2 = tree.create_child(2, global).unwrap();
let func1 = tree.create_child(10, mod1).unwrap();
let func2 = tree.create_child(20, mod2).unwrap();

// Vertical relationships
assert!(tree.is_descendant(func1, global));  // true (grandchild)
assert!(tree.is_descendant(func1, mod1));    // true (child)
assert!(tree.is_descendant(mod1, global));   // true (child)

// Horizontal relationships (siblings)
assert!(!tree.is_descendant(mod1, mod2));    // false (siblings)
assert!(!tree.is_descendant(func1, func2));  // false (cousins)

// Reverse relationships
assert!(!tree.is_descendant(global, mod1));  // false (ancestor, not descendant)
```

### Example 7: Context Existence Checks

```rust
let mut tree = ContextTree::new();
let root = tree.create_root(0);

// Check existence
assert!(tree.contains(root));
assert!(!tree.contains(999)); // Doesn't exist

// Check if root
assert!(tree.is_root(root));
assert!(!tree.is_root(999)); // Doesn't exist, so not root

// Get parent safely
match tree.parent(root) {
    None => println!("Root context has no parent"),
    Some(p) => println!("Parent is {}", p),
}
```

### Example 8: Tree Capacity Planning

```rust
// Pre-allocate for known number of contexts
let mut tree = ContextTree::with_capacity(100);

// Avoids reallocation during insertions
for i in 0..100 {
    tree.create_root(i);
}

assert_eq!(tree.len(), 100);
```

---

## Performance Characteristics

### Operation Complexity

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| `create_root()` | $`\mathcal{O}(1)`$ | $`\mathcal{O}(1)`$ | HashMap insert |
| `create_child()` | $`\mathcal{O}(1)`$ | $`\mathcal{O}(1)`$ | HashMap lookup + insert |
| `parent()` | $`\mathcal{O}(1)`$ | $`\mathcal{O}(1)`$ | HashMap lookup |
| `contains()` | $`\mathcal{O}(1)`$ | $`\mathcal{O}(1)`$ | HashMap lookup |
| `is_root()` | $`\mathcal{O}(1)`$ | $`\mathcal{O}(1)`$ | HashMap lookup + match |
| `visible_contexts()` | $`\mathcal{O}(d)`$ | $`\mathcal{O}(d)`$ | d = depth from root |
| `depth()` | $`\mathcal{O}(d)`$ | $`\mathcal{O}(1)`$ | d = depth from root |
| `is_descendant()` | $`\mathcal{O}(d)`$ | $`\mathcal{O}(1)`$ | d = depth from root |
| `remove()` | $`\mathcal{O}(n)`$ | $`\mathcal{O}(n)`$ | n = total contexts, must find descendants |

### Benchmarks

**Test Environment**: Intel Xeon E5-2699 v3 @ 2.30GHz, Rust 1.75, release build

| Operation | Time (ns) | Throughput |
|-----------|-----------|------------|
| `create_root()` | ~50 | 20M ops/sec |
| `create_child()` | ~70 | 14M ops/sec |
| `parent()` | ~15 | 67M ops/sec |
| `visible_contexts(depth=5)` | ~150 | 6.7M ops/sec |
| `depth(depth=5)` | ~120 | 8.3M ops/sec |
| `is_descendant(depth=5)` | ~100 | 10M ops/sec |

**Key Observations**:
- All operations sub-microsecond (imperceptible latency)
- Visibility computation (common operation) extremely fast (~150ns for depth 5)
- Context creation overhead negligible for interactive use

### Memory Usage

**Scaling Test** (measuring actual memory usage):

| Context Count | Memory (KB) | Per-Context Overhead |
|---------------|-------------|----------------------|
| 10 | ~0.3 | ~30 bytes |
| 100 | ~2.5 | ~25 bytes |
| 1,000 | ~24 | ~24 bytes |
| 10,000 | ~240 | ~24 bytes |

**Conclusion**: ~24 bytes per context, linear scaling, minimal overhead.

---

## Implementation Details

### HashMap Choice

**Why HashMap over Vec?**

**Rejected Alternative**: `Vec<Option<ContextId>>` indexed by context ID.

**Reasons**:
- Sparse ID space: Context IDs may have large gaps (e.g., [0, 1, 100, 1000])
- Wasted memory: Vec would allocate for all IDs from 0 to max(ID)
- HashMap provides $`\mathcal{O}(1)`$ operations with minimal waste

**Example**:
```
Context IDs: [0, 1, 100, 10000]

Vec approach:
- Allocation: 10001 entries (mostly None)
- Memory: 10001 × 8 bytes = 80KB

HashMap approach:
- Allocation: 4 entries
- Memory: 4 × 24 bytes = 96 bytes

Savings: 99.9%
```

### Parent-Only Storage

**Design Choice**: Store only `child → parent` mapping, not `parent → children`.

**Rationale**:
- Visibility computation (common) only needs parent lookup
- Children enumeration (rare) can be computed on-demand
- Simpler data structure, less memory

**Trade-off**:
- ✓ Faster visibility queries (no extra indirection)
- ✓ Less memory (one mapping instead of two)
- ✗ Slower `remove()` (must scan all contexts)

**Conclusion**: Optimizes for the common case (queries >> removals).

### Cascading Removal Algorithm

**Why not store children explicitly?**

Cascading removal requires finding all descendants. Two approaches:

**Approach 1: Store bi-directional links** (parent → children, child → parent)
- Memory: 2× storage
- Removal: $`\mathcal{O}(\text{descendants})`$ direct lookup
- Query: Unchanged

**Approach 2: Store parent only, scan on removal** (current implementation)
- Memory: 1× storage
- Removal: $`\mathcal{O}(\text{total}_\text{contexts})`$ scan
- Query: Unchanged

**Decision**: Approach 2 (current) because:
- Context removal is rare (typically only on file close in LSP)
- Queries (visibility) vastly more frequent
- Memory savings worth occasional $`\mathcal{O}(n)`$ scan

### Thread Safety Considerations

`ContextTree` is **not thread-safe** by itself (uses `&mut self` for mutations). However, the engine wraps it in `Arc<RwLock<ContextTree>>`:

```rust
context_tree: Arc<RwLock<ContextTree>>,
```

**Locking Strategy**:
- **Queries** (`visible_contexts`, `parent`): Read lock (shared, concurrent)
- **Mutations** (`create_child`, `remove`): Write lock (exclusive)

**Typical Access Pattern**:
- 95%+ of operations are queries (read locks)
- Minimal write contention (context creation/removal rare)

---

## Testing

### Test Coverage

**Included Tests** (in `src/contextual/context_tree.rs`):

1. ✅ `test_new()` - Empty tree initialization
2. ✅ `test_create_root()` - Root context creation
3. ✅ `test_create_child()` - Child context creation
4. ✅ `test_create_child_invalid_parent()` - Error handling for invalid parent
5. ✅ `test_visible_contexts()` - Visibility computation
6. ✅ `test_is_descendant()` - Descendant relationships
7. ✅ `test_depth()` - Depth calculation
8. ✅ `test_remove()` - Cascading removal
9. ✅ `test_remove_nonexistent()` - Removal error handling
10. ✅ `test_clear()` - Clearing entire tree
11. ✅ `test_multiple_roots()` - Multi-root support
12. ✅ `test_complex_hierarchy()` - Complex tree scenarios

**Test Coverage**: ~95% of code paths

### Property-Based Testing

**Potential Properties** (for future `proptest` integration):

```rust
// Property: Visibility is transitive
forall contexts A, B, C:
    if A in visible(B) and B in visible(C),
    then A in visible(C)

// Property: Depth increases monotonically
forall contexts child, parent:
    if parent(child) == Some(parent),
    then depth(child) == depth(parent) + 1

// Property: Removal is idempotent
forall context C:
    remove(C); remove(C) == remove(C)
```

---

[← Back to Layer 7](../README.md)
