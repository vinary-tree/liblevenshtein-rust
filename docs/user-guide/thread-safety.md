# PathMap Thread Safety Analysis

The diagram below summarizes the concurrency model: how reads, writes, and snapshots
are coordinated across the backend wrappers (`RwLock`-guarded, `ArcSwap` copy-on-write,
and persistent lock-free variants).

![Concurrency model: how reader threads, writer threads, and snapshot readers coordinate through RwLock guards, ArcSwap copy-on-write, and persistent structural sharing](../diagrams/concurrency/concurrency-model.svg)

*Concurrency model: the coordination strategies available to dictionary backends.*

## Current Understanding

### PathMap's Internal Structure

```rust
pub struct PathMap<V: Clone + Send + Sync, A: Allocator = GlobalAlloc> {
    pub(crate) root: UnsafeCell<Option<TrieNodeODRc<V, A>>>,
    pub(crate) root_val: UnsafeCell<Option<V>>,
    pub(crate) alloc: A,
}

unsafe impl<V: Clone + Send + Sync, A: Allocator> Send for PathMap<V, A> {}
unsafe impl<V: Clone + Send + Sync, A: Allocator> Sync for PathMap<V, A> {}
```

### Key Observations

1. **Uses `UnsafeCell`**: Interior mutability without built-in synchronization
2. **Manual `Send + Sync`**: Authors explicitly marked it thread-safe
3. **README states**: "optimized for large data sets and can be used efficiently in a multi-threaded environment"
4. **Structural sharing**: Clone is cheap (shares nodes via reference counting)

## Thread Safety Guarantees (Current Knowledge)

### What We Know
- PathMap implements `Send` and `Sync`
- Can be shared across threads
- Uses reference-counted nodes (`TrieNodeODRc` - "On-Demand Reference Counted")
- Designed for "multi-threaded environment" per README

### What's Unclear
- **Concurrent writes**: Not documented whether `insert()`/`remove()` are thread-safe
- **Read-write concurrency**: Not documented whether reads can happen during writes
- **Internal synchronization**: No visible locks/atomics in the PathMap struct itself
- **Persistence semantics**: Whether mutations create new versions or modify in-place

## Our Current Implementation: RwLock Wrapper

### Why We Use `Arc<RwLock<PathMap<()>>>`

**Conservative approach** until we have more information:

```rust
pub struct PathMapDictionary {
    map: Arc<RwLock<PathMap<()>>>,
    term_count: Arc<RwLock<usize>>,
}
```

### Benefits of RwLock Approach

✅ **Proven thread-safe**: Rust's type system guarantees safety
✅ **Multiple concurrent reads**: Verified via tests (3.82x parallelism)
✅ **Safe writes**: Exclusive access prevents data races
✅ **No undefined behavior**: Even if PathMap has internal UnsafeCell issues

### Performance Characteristics (Measured)

From `tests/concurrency_test.rs`:
- **Parallelism ratio**: 3.82x (8 threads)
- **Read contention**: Minimal (readers don't block each other)
- **Write blocking**: 50 queries completed during 10 writes
- **Lock overhead**: Acceptable for most use cases

## Alternative Approaches

### Option 1: Arc-Swap (Copy-on-Write)

If PathMap is truly persistent/immutable internally:

```rust
use arc_swap::ArcSwap;

pub struct PathMapDictionary {
    map: Arc<ArcSwap<PathMap<()>>>,
    term_count: AtomicUsize,
}

impl PathMapDictionary {
    pub fn insert(&self, term: &str) -> bool {
        loop {
            let current = self.map.load();
            let mut new_map = (**current).clone(); // Structural sharing

            if new_map.insert(term.as_bytes(), ()).is_none() {
                // Atomic swap - readers see old or new version atomically
                self.map.store(Arc::new(new_map));
                self.term_count.fetch_add(1, Ordering::SeqCst);
                return true;
            } else {
                return false;
            }
        }
    }
}
```

**Pros**:
- Lock-free reads (zero overhead)
- Writers don't block readers
- Readers see consistent snapshots

**Cons**:
- Clone creates new root (even with structural sharing)
- Potential ABA problem without careful design
- Memory overhead (old versions retained until all readers release)

### Option 2: Direct PathMap Usage (If Truly Thread-Safe)

If PathMap's UnsafeCell usage is internally synchronized:

```rust
pub struct PathMapDictionary {
    map: Arc<PathMap<()>>,
    term_count: AtomicUsize,
}

// Direct calls without locks
impl PathMapDictionary {
    pub fn insert(&self, term: &str) -> bool {
        // If PathMap is thread-safe internally
        let bytes = term.as_bytes();
        if self.map.insert(bytes, ()).is_none() {
            self.term_count.fetch_add(1, Ordering::SeqCst);
            true
        } else {
            false
        }
    }
}
```

**Pros**:
- Zero locking overhead
- Maximum concurrency

**Cons**:
- **Requires confirmation** that PathMap is internally thread-safe
- Undefined behavior if assumption is wrong

## Recommended Path Forward

### Near Term: Keep RwLock (Current Implementation)

**Rationale**:
- Proven safe and working
- Acceptable performance (3.82x parallelism)
- No risk of undefined behavior
- Easy to understand and maintain

### Medium Term: Investigate PathMap Internals

**Next steps**:
1. Contact PathMap authors about thread safety guarantees
2. Review PathMap's reference counting implementation
3. Check if `TrieNodeODRc` uses `Arc` (thread-safe) or `Rc` (not thread-safe)
4. Test concurrent mutations without locks (in isolated test)

### Long Term: Optimize Based on Findings

**If PathMap is internally thread-safe**:
- Switch to `Arc<PathMap<()>>` (no locks)
- Or use `Arc-Swap` for lock-free snapshots

**If PathMap is NOT thread-safe**:
- Keep RwLock (current approach)
- Or propose/contribute thread-safe version to PathMap

## Testing Strategy

### Current Tests ✅

- `test_parallel_reads`: Verified concurrent read parallelism
- `test_read_during_write`: Verified reads during writes
- `test_pathmap_dictionary_concurrent_operations`: Basic multi-threading

### Additional Tests Needed

1. **Concurrent writes** without locks (to test PathMap's guarantees)
2. **Stress test**: Many threads reading + writing simultaneously
3. **Memory leak test**: Verify structural sharing doesn't leak
4. **Correctness test**: Linearizability of operations

## Conclusion

**Current implementation is correct and performant** using RwLock.

Future optimization possibilities exist if we can verify PathMap's internal thread-safety guarantees, but the current approach is:
- ✅ Safe
- ✅ Fast enough (proven via benchmarks)
- ✅ Maintainable
- ✅ Well-documented

**Recommendation**: Ship with RwLock, optimize later with data.

## Related Documentation

- [Backends](backends.md) - Dictionary backend comparison
- [Features](features.md) - Full feature overview
- [Getting Started](getting-started.md) - Basic usage

---

[← Documentation Index](../README.md)
