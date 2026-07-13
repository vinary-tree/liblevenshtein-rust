# Subsumption Optimization: Java vs Rust Implementation Comparison

## Executive Summary

**Answer**: Yes, the Rust pre-sorting optimization is **functionally equivalent** to the Java implementation, but with **different trade-offs**:

- **Java**: Uses linked lists + merge sort + nested iteration with early termination
- **Rust**: Uses BTreeSet with custom Ord + automatic sorting + early termination via `take_while()`

Both achieve `$\mathcal{O}(k)$` average-case subsumption checks where k << n, but through different mechanisms.

---

## Java Implementation Analysis

### Data Structure (State.java)

```java
public class State implements Iterable<Position>, Serializable {
    private Position head = null;  // Linked list of positions
}
```

**Key characteristics**:
- Singly-linked list of `Position` nodes
- Head pointer maintains list start
- Dynamic structure allows efficient insertion/removal during iteration

### Sorting Strategy (State.java, lines 106-191)

```java
public State sort(final Comparator<Position> comparator) {
    head = mergeSort(comparator, head);
    return this;
}

private Position mergeSort(final Comparator<Position> comparator, final Position lhsHead) {
    if (null == lhsHead || null == lhsHead.next()) {
        return lhsHead;
    }
    final Position middle = middle(lhsHead);
    final Position rhsHead = middle.next();
    middle.next(null);
    return merge(comparator, mergeSort(comparator, lhsHead), mergeSort(comparator, rhsHead));
}
```

**Algorithm**: Classic merge sort on linked list
- **Time**: `$\mathcal{O}(n \log  n)$`
- **Space**: `$\mathcal{O}(\log  n)$` for recursion stack (in-place on linked list)
- **Stability**: Yes (maintains relative order of equal elements)

### Unsubsumption Logic (UnsubsumeFunction.java, lines 44-67)

```java
public void at(final State state, final int queryLength) {
    final StateIterator outerIter = state.iterator();
    while (outerIter.hasNext()) {
        final Position outer = outerIter.next();
        final int outerErrors = outer.numErrors();

        final StateIterator innerIter = outerIter.copy();

        // EARLY TERMINATION: Skip positions with errors <= outer.errors
        while (innerIter.hasNext()) {
            final Position inner = innerIter.peek();
            if (outerErrors < inner.numErrors()) {
                break;  // Early termination!
            }
            innerIter.next();
        }

        // Check remaining positions for subsumption
        while (innerIter.hasNext()) {
            final Position inner = innerIter.next();
            if (subsumes.at(outer, inner, queryLength)) {
                innerIter.remove();  // Remove subsumed position
            }
        }
    }
}
```

**Key insights**:
1. **Nested iteration**: `$\mathcal{O}(n^{2})$` worst case, but with early termination
2. **Early termination** (lines 52-58): Skips positions until `errors > outerErrors`
   - Relies on positions being **sorted by errors ascending**
   - Breaks inner loop when it reaches positions that could be subsumed
3. **In-place removal**: Uses iterator's `remove()` to modify during iteration
4. **Two-phase inner loop**:
   - Phase 1: Skip to positions with more errors (early termination)
   - Phase 2: Check subsumption and remove if subsumed

### Subsumption Check (SubsumesFunction.java, lines 36-42)

```java
public boolean at(final Position lhs, final Position rhs, final int n) {
    final int i = lhs.termIndex();
    final int e = lhs.numErrors();
    final int j = rhs.termIndex();
    final int f = rhs.numErrors();
    return (i < j ? j - i : i - j) <= (f - e);
}
```

**Formula**: `$\lvert j - i\rvert \le (f - e)$`
**Note**: Missing explicit check for `f > e`, relies on caller to ensure this via sorting

---

## Rust Implementation Analysis

### Data Structure (state.rs, line 83)

```rust
pub struct UniversalState<V: PositionVariant> {
    positions: BTreeSet<UniversalPosition<V>>,
    max_distance: u8,
}
```

**Key characteristics**:
- Red-black tree (BTreeSet) maintains sorted order automatically
- Custom `Ord` implementation sorts by `(errors, offset)`
- No manual sorting required - happens during insertion

### Custom Ordering (position.rs, lines 207-228)

```rust
impl<V: PositionVariant> Ord for UniversalPosition<V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        use UniversalPosition::*;

        match (self, other) {
            (INonFinal { errors: e1, offset: o1, .. }, INonFinal { errors: e2, offset: o2, .. }) |
            (MFinal { errors: e1, offset: o1, .. }, MFinal { errors: e2, offset: o2, .. }) => {
                match e1.cmp(e2) {
                    Ordering::Equal => o1.cmp(o2),  // Secondary sort
                    other => other,                  // Primary sort by errors
                }
            }
            (INonFinal { .. }, MFinal { .. }) => Ordering::Less,
            (MFinal { .. }, INonFinal { .. }) => Ordering::Greater,
        }
    }
}
```

**Sorting**: Primary by errors (ascending), secondary by offset (ascending)
- **Automatic**: No explicit sort call needed
- **Always sorted**: Invariant maintained by BTreeSet

### Unsubsumption Logic (state.rs, lines 155-182)

```rust
pub fn add_position(&mut self, pos: UniversalPosition<V>) {
    let pos_errors = pos.errors();

    // Step 1: Remove positions subsumed by new position
    // EARLY TERMINATION: Only positions with MORE errors can be subsumed
    self.positions.retain(|p| {
        if p.errors() <= pos_errors {
            true  // Keep - cannot be subsumed by pos
        } else {
            !subsumes(&pos, p, self.max_distance)
        }
    });

    // Step 2: Check if new position is subsumed by existing
    // EARLY TERMINATION: Only positions with FEWER errors can subsume pos
    let is_subsumed = self.positions.iter()
        .take_while(|p| p.errors() < pos_errors)  // Early termination!
        .any(|p| subsumes(p, &pos, self.max_distance));

    if !is_subsumed {
        self.positions.insert(pos);  // O(log n) insertion
    }
}
```

**Key insights**:
1. **Online subsumption**: Applied during insertion, not as batch operation
2. **Early termination** (line 165): `if p.errors() <= pos_errors { true }`
   - Positions with fewer/equal errors cannot be subsumed by pos
3. **Early termination** (line 176): `.take_while(|p| p.errors() < pos_errors)`
   - Only positions with fewer errors can subsume pos
   - Iterator stops at first position with errors `$\ge$` pos.errors
4. **Automatic ordering**: BTreeSet maintains sort during insert/remove

### Subsumption Check (subsumption.rs, lines 113-142)

```rust
pub fn subsumes<V: PositionVariant>(
    lhs: &UniversalPosition<V>,
    rhs: &UniversalPosition<V>,
    max_distance: u8,
) -> bool {
    let i = lhs.offset();
    let e = lhs.errors();
    let j = rhs.offset();
    let f = rhs.errors();

    // Explicit check: rhs must have MORE errors
    if f <= e {
        return false;
    }

    // Distance check: |j - i| ≤ (f - e)
    let distance = if j >= i { j - i } else { i - j };
    distance <= (f - e) as i32
}
```

**Formula**: `$f > e \;\text{and}\; \lvert j - i\rvert \le (f - e)$`
**Note**: Explicit check for `f > e`, more defensive than Java version

---

## Detailed Comparison

### 1. Data Structure Trade-offs

| Aspect | Java (Linked List) | Rust (BTreeSet) |
|--------|-------------------|-----------------|
| **Insert** | `$\mathcal{O}(1)$` at head/tail, `$\mathcal{O}(n)$` middle | `$\mathcal{O}(\log  n)$` anywhere |
| **Remove** | `$\mathcal{O}(1)$` with iterator | `$\mathcal{O}(\log  n)$` |
| **Iteration** | `$\mathcal{O}(n)$` sequential | `$\mathcal{O}(n)$` in-order |
| **Memory** | 1-2 pointers per node | Tree nodes + color bits |
| **Cache locality** | Poor (scattered nodes) | Better (nodes in cache lines) |
| **Sorting** | Explicit `$\mathcal{O}(n \log  n)$` | Automatic during insert |

### 2. Algorithmic Equivalence

**Java approach**:
```
1. Build state with unsorted positions
2. Sort once: O(n log n) merge sort
3. Iterate with early termination: O(k) where k << n
4. Total per state: O(n log n + k)
```

**Rust approach**:
```
1. Insert positions one-by-one with subsumption: O(log n) per insert
2. Each insert checks early termination: O(k) where k << n
3. Automatic sorting maintained
4. Total per state: O(n * (log n + k))
```

**Asymptotic comparison**:
- Java: `$\mathcal{O}(n \log  n)$` + `$\mathcal{O}(n * k)$` for batch unsubsume
- Rust: `$\mathcal{O}(n * (\log  n + k)$`) for online unsubsume during inserts

**When k << log n** (sparse error distribution):
- Java: ~`$\mathcal{O}(n \log  n)$` dominated by sort
- Rust: ~`$\mathcal{O}(n \log  n)$` dominated by tree operations

**When `$k \approx  n$`** (dense error distribution):
- Java: ~`$\mathcal{O}(n^{2})$`
- Rust: ~`$\mathcal{O}(n^{2})$`

**Conclusion**: Asymptotically equivalent for typical cases!

### 3. Early Termination Mechanism

**Java** (explicit loop with break):
```java
while (innerIter.hasNext()) {
    final Position inner = innerIter.peek();
    if (outerErrors < inner.numErrors()) {
        break;  // Stop when errors increase
    }
    innerIter.next();
}
```

**Rust** (iterator combinator):
```rust
self.positions.iter()
    .take_while(|p| p.errors() < pos_errors)  // Stop when errors >= pos.errors
    .any(|p| subsumes(p, &pos, self.max_distance))
```

**Both achieve**: Skip positions that cannot participate in subsumption

### 4. Subsumption Formula Differences

**Java**:
```java
return (i < j ? j - i : i - j) <= (f - e);
```
- Computes `|j - i|` manually
- **Missing**: Explicit `f > e` check (relies on caller)

**Rust**:
```rust
if f <= e {
    return false;  // Explicit guard
}
let distance = if j >= i { j - i } else { i - j };
distance <= (f - e) as i32
```
- Explicit `f > e` check
- More defensive programming

**Equivalence**: Yes, when Java caller ensures proper sorting

---

## Key Differences

### 1. **Batch vs Online**

**Java**: Batch unsubsumption
- Build complete state, then sort, then unsubsume
- More cache-friendly (sequential access after sort)
- Better for large batches of positions

**Rust**: Online unsubsumption
- Unsubsume during each position insertion
- Maintains anti-chain invariant continuously
- Better for incremental construction

### 2. **Mutability Model**

**Java**: Mutable iteration with removal
```java
while (innerIter.hasNext()) {
    if (subsumes.at(outer, inner, queryLength)) {
        innerIter.remove();  // Modify during iteration
    }
}
```

**Rust**: Functional filtering
```rust
self.positions.retain(|p| {
    if p.errors() <= pos_errors {
        true  // Keep
    } else {
        !subsumes(&pos, p, self.max_distance)  // Remove if subsumed
    }
});
```

### 3. **Iterator Sophistication**

**Java**: Custom StateIterator with lookahead/lookbehind
- `peek()` without advancing
- `copy()` for nested iteration
- `insert()` and `remove()` during iteration
- Powerful but complex

**Rust**: Standard library iterators
- `take_while()` for early termination
- `any()` for existence check
- `retain()` for filtering
- Simple but composable

---

## Performance Implications

### Java Advantages:
1. **Single sort**: `$\mathcal{O}(n \log  n)$` once, then `$\mathcal{O}(k)$` checks
2. **In-place**: Linked list mutations don't reallocate
3. **Cache-friendly**: Sequential access after sort
4. **Batch-optimized**: Amortizes sort cost over many subsumption checks

### Rust Advantages:
1. **No separate sort**: Sorting integrated into insertion
2. **Memory safety**: No iterator invalidation issues
3. **Always sorted**: Can query/iterate at any time
4. **Incremental**: No need to build full state before unsubsuming

### Empirical Expectations:

For **typical Levenshtein states** (small, sparse error distributions):
- Java: **Slightly faster** for batch operations (better cache locality)
- Rust: **Comparable** with simpler code (no manual sort call)

For **large states** (many positions):
- Java: **Better** (batch sort amortizes better)
- Rust: **Better** (tree balancing prevents worst-case linked list behavior)

---

## Lessons for Rust Implementation

### 1. ✅ Already Implemented Correctly

The Rust BTreeSet approach **is equivalent** to Java's linked list + merge sort approach:
- Both sort by errors ascending
- Both use early termination
- Both achieve `$\mathcal{O}(k)$` subsumption checks where k << n

### 2. ⚠️ Potential Optimization: Batch Mode

The Java implementation suggests a **batch unsubsumption** approach could be beneficial:

```rust
// Current: Online (during each insert)
state.add_position(pos);  // Subsumes immediately

// Alternative: Batch (defer subsumption)
state.add_position_unchecked(pos);  // Skip subsumption
// ... add many positions ...
state.unsubsume();  // Single pass to remove subsumed
```

**When beneficial**:
- Building states with many positions at once
- Generating successors for multiple positions
- Better cache locality from sequential access

**Trade-off**:
- More complex API (two-phase: insert + unsubsume)
- Temporary violations of anti-chain invariant
- **Current approach is simpler and correct**

### 3. ✅ Early Termination is Optimal

Both implementations use the same early termination strategy:
- Java: `if (outerErrors < inner.numErrors()) break;`
- Rust: `.take_while(|p| p.errors() < pos_errors)`

**No improvements needed** - Rust's approach is idiomatic and efficient.

### 4. ✅ Subsumption Formula is Correct

Rust includes explicit `f > e` guard, which is **more defensive** than Java:
```rust
if f <= e {
    return false;  // Explicit check
}
```

Java relies on sorting to ensure this, which works but is less obvious.

### 5. 🔄 Consider Linked List for Parameterized (Optional)

For the **parameterized transducers** (Schulz & Mihov 2002), which use `SmallVec<[Position; 8]>`:

**Current**:
```rust
positions: SmallVec<[Position; 8]>  // Stack allocation for ≤8 positions
```

**Alternative** (Java-style):
```rust
positions: Option<Box<Position>>  // Linked list like Java
```

**Analysis**:
- SmallVec is **better for small states** (stack allocation, cache-friendly)
- Linked list is **better for large states** (no reallocation, `$\mathcal{O}(1)$` remove during iteration)
- **Recommendation**: Keep SmallVec (most states are small)

---

## Conclusion

### Question 1: Is the Rust pre-sorting equivalent to Java?

**Yes!** The Rust BTreeSet with custom Ord is **functionally equivalent** to Java's linked list + merge sort:

| Feature | Java | Rust |
|---------|------|------|
| **Sorting strategy** | Explicit merge sort | Automatic via BTreeSet |
| **Sort order** | By errors ascending | By (errors, offset) ascending |
| **Early termination** | Explicit break in loop | `.take_while()` iterator |
| **Subsumption checks** | `$\mathcal{O}(k)$` where k << n | `$\mathcal{O}(k)$` where k << n |
| **Asymptotic complexity** | `$\mathcal{O}(n \log  n + n*k)$` | `$\mathcal{O}(n \log  n + n*k)$` |

### Question 2: What can be gleaned from Java version?

**Key insights**:

1. ✅ **Sorting by errors first is optimal** - Both implementations do this
2. ✅ **Early termination is essential** - Both implementations have it
3. ✅ **Small k << n in practice** - Error distributions are sparse
4. 🔄 **Batch mode could help** - Java's two-phase (sort + unsubsume) might be faster for bulk operations
5. ✅ **Current Rust approach is simpler** - No manual sorting, cleaner code

**Recommendation**:
- **Keep current Rust implementation** for Universal transducers (BTreeSet is elegant and correct)
- **Keep SmallVec** for parameterized transducers (optimized for small states)
- **Optional**: Add batch unsubsumption mode if profiling shows it's beneficial
- **Document**: Add this comparison to show design rationale

The Rust implementation achieves the same algorithmic efficiency as Java with simpler, more idiomatic code!
