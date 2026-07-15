# LRU (Least Recently Used) Eviction Policy

## Overview

The LRU eviction policy tracks the last access time for each entry and identifies candidates for eviction based on recency. Entries that haven't been accessed recently are considered "cold" and prioritized for removal.

**Formula**: Eviction score = `last_accessed.elapsed()`

**Eviction Strategy**: Highest score (longest time since access) evicted first.

## Architecture

### Metadata Structure

```rust
struct EntryMetadata {
    last_accessed: Instant,
}
```

**Size**: 16 bytes (Instant = 2×u64 on most platforms)

### Wrapper Structure

```rust
pub struct Lru<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}
```

**Thread-Safety**: RwLock allows concurrent reads, serialized writes.

## Core Operations

### Access Tracking

```rust
impl<D: MappedDictionary<V>, V> MappedDictionary<V> for Lru<D> {
    fn get_value(&self, term: &str) -> Option<V> {
        // Update metadata
        {
            let mut metadata = self.metadata.write().unwrap();
            metadata
                .entry(term.to_string())
                .and_modify(|m| m.update_access())
                .or_insert_with(EntryMetadata::new);
        }

        // Forward to inner dictionary
        self.inner.get_value(term)
    }
}
```

**Complexity**: $`\mathcal{O}(1)`$ metadata update + $`\mathcal{O}(d)`$ inner lookup

### Finding LRU Entry

```rust
impl<D> Lru<D> {
    pub fn find_lru(&self, candidates: &[&str]) -> Option<String> {
        let metadata = self.metadata.read().unwrap();

        candidates
            .iter()
            .filter_map(|term| {
                metadata.get(*term).map(|m| (*term, m.recency()))
            })
            .max_by_key(|(_, recency)| *recency)
            .map(|(term, _)| term.to_string())
    }
}
```

**Complexity**: $`\mathcal{O}(n)`$ where n = number of candidates

## Usage Examples

### Example 1: Basic LRU Tracking

```rust
use liblevenshtein::cache::eviction::Lru;
use liblevenshtein::prelude::*;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::from_terms_with_values([
        ("active", 1),
        ("dormant", 2),
        ("cold", 3),
    ]);

    let lru = Lru::new(dict);

    // Access pattern: active (3×), dormant (1×), cold (0×)
    lru.get_value("active");
    thread::sleep(Duration::from_millis(10));

    lru.get_value("active");
    thread::sleep(Duration::from_millis(10));

    lru.get_value("dormant");
    thread::sleep(Duration::from_millis(10));

    lru.get_value("active");

    // Find LRU: "cold" (never accessed)
    let lru_entry = lru.find_lru(&["active", "dormant", "cold"]);
    assert_eq!(lru_entry, Some("cold".to_string()));

    Ok(())
}
```

### Example 2: Code Completion with LRU

```rust
use liblevenshtein::cache::eviction::Lru;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Identifiers in current scope
    let identifiers = PathMapDictionary::from_terms_with_values([
        ("count", "i32"),
        ("scratch", "String"),
        ("result", "Vec<u8>"),
        ("old_variable", "bool"),
    ]);

    let lru = Lru::new(identifiers);

    // User types code, referencing some identifiers
    lru.get_value("count");   // Used in loop
    lru.get_value("result");  // Used for accumulation
    lru.get_value("count");   // Used again

    // "old_variable" never accessed

    // When memory pressure high, evict LRU
    let candidates = vec!["count", "scratch", "result", "old_variable"];
    let evict = lru.find_lru(&candidates);
    assert_eq!(evict, Some("old_variable".to_string()));

    println!("Evict: {:?} (least recently used)", evict);

    Ok(())
}
```

### Example 3: Web Session Management

```rust
use liblevenshtein::cache::eviction::Lru;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Session tokens mapped to user IDs
    let sessions = PathMapDictionary::from_terms_with_values([
        ("session_abc123", "user_42"),
        ("session_def456", "user_99"),
        ("session_ghi789", "user_12"),
    ]);

    let lru_sessions = Lru::new(sessions);

    // Simulate user activity
    lru_sessions.get_value("session_abc123");  // User 42 active
    lru_sessions.get_value("session_def456");  // User 99 active
    // User 12 inactive (no access)

    // When cache full, evict least recently used session
    let candidates = vec!["session_abc123", "session_def456", "session_ghi789"];
    let evict_session = lru_sessions.find_lru(&candidates);

    assert_eq!(evict_session, Some("session_ghi789".to_string()));
    println!("Evict session: {:?}", evict_session);

    Ok(())
}
```

### Example 4: Composition with TTL

```rust
use liblevenshtein::cache::eviction::{Lru, Ttl};
use liblevenshtein::prelude::*;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::from_terms_with_values([
        ("foo", 1),
        ("bar", 2),
    ]);

    // Compose TTL + LRU
    let ttl = Ttl::new(dict, Duration::from_secs(300));
    let lru = Lru::new(ttl);

    // Access "foo" (recent)
    lru.get_value("foo");

    // "bar" not accessed (LRU)

    // Find LRU among non-expired entries
    let lru_entry = lru.find_lru(&["foo", "bar"]);
    assert_eq!(lru_entry, Some("bar".to_string()));

    Ok(())
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| get_value | $`\mathcal{O}(d)`$ + $`\mathcal{O}(1)`$ | Dictionary lookup + metadata update |
| contains | $`\mathcal{O}(d)`$ + $`\mathcal{O}(1)`$ | Dictionary check + metadata update |
| find_lru | $`\mathcal{O}(n)`$ | Linear scan of candidates |

Where:
- `d` = inner dictionary operation complexity
- `n` = number of candidates

### Space Complexity

**Per-Entry Overhead**: 16 bytes (Instant)

**Total Metadata**: 16n bytes for n tracked entries

**Comparison**:
- LRU: 16 bytes/entry (Instant)
- LFU: 8 bytes/entry (usize)
- CostAware: 32 bytes/entry (Instant + 2×usize)

### Lock Contention

**RwLock Behavior**:
- **Reads** (get_value): Shared lock for metadata read
- **Writes** (update_access): Exclusive lock for metadata write

**High-Concurrency Impact**:
- Read-heavy workloads: Minimal contention (concurrent reads)
- Write-heavy workloads: Serialized metadata updates

**Mitigation**:
- Shard cache by key prefix
- Use Noop wrapper in production if LRU not needed
- Profile lock wait times under load

## Comparison with Other Policies

| Aspect | LRU | LFU | Age | TTL |
|--------|-----|-----|-----|-----|
| **Metric** | Recency | Frequency | Insertion order | Time-to-live |
| **Overhead** | 16 bytes | 8 bytes | 16 bytes | 16 bytes |
| **Best for** | Access patterns | Hot content | FIFO queues | Expiration |
| **Eviction** | Longest elapsed | Lowest count | Oldest insert | Expired only |

### When to Use LRU

✅ **Good For**:
- Code completion (recent identifiers)
- General-purpose caching
- Temporal locality (recent $`\approx`$ likely to reuse)
- Session management

❌ **Not Ideal For**:
- Frequency-based (use LFU)
- Fixed expiration (use TTL)
- Sequential access patterns (all entries equally recent)

## Related Documentation

- **[Layer 8 README](../README.md)** - Caching layer overview
- **[TTL Policy](./ttl.md)** - Time-to-live expiration
- **[MemoryPressure Policy](./memory-pressure.md)** - Memory-aware eviction

## References

1. Belady, L. A. (1966). "A Study of Replacement Algorithms for Virtual-Storage Computer"
2. O'Neil, E. J., et al. (1993). "The LRU-K Page Replacement Algorithm For Database Disk Buffering"
3. Source code: `src/cache/eviction/lru.rs`

## Quick Reference

```rust
use liblevenshtein::cache::eviction::Lru;
use liblevenshtein::prelude::*;

// Basic usage
let dict = PathMapDictionary::from_terms_with_values([("key", "value")]);
let lru = Lru::new(dict);
lru.get_value("key");

// Find LRU entry
let lru_entry = lru.find_lru(&["key1", "key2"]);

// Composition with TTL
use std::time::Duration;
let ttl = Ttl::new(dict, Duration::from_secs(300));
let lru_ttl = Lru::new(ttl);
```
