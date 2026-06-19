# LFU (Least Frequently Used) Eviction Policy

## Overview

The LFU eviction policy tracks access frequency for each entry and identifies candidates for eviction based on how often they're accessed. Entries accessed infrequently are considered "unpopular" and prioritized for removal.

**Formula**: Eviction score = `access_count`

**Eviction Strategy**: Lowest score (least accesses) evicted first.

## Architecture

### Metadata Structure

```rust
struct EntryMetadata {
    access_count: u32,
}
```

**Size**: 8 bytes (u32 + padding)

### Wrapper Structure

```rust
pub struct Lfu<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}
```

## Core Operations

### Access Tracking

```rust
impl<D: MappedDictionary<V>, V> MappedDictionary<V> for Lfu<D> {
    fn get_value(&self, term: &str) -> Option<V> {
        // Update access count
        {
            let mut metadata = self.metadata.write().unwrap();
            metadata
                .entry(term.to_string())
                .and_modify(|m| m.increment())
                .or_insert_with(EntryMetadata::new);
        }

        // Forward to inner dictionary
        self.inner.get_value(term)
    }
}
```

**Complexity**: O(1) metadata update + O(d) inner lookup

### Finding LFU Entry

```rust
impl<D> Lfu<D> {
    pub fn find_lfu(&self, candidates: &[&str]) -> Option<String> {
        let metadata = self.metadata.read().unwrap();

        candidates
            .iter()
            .filter_map(|term| {
                metadata.get(*term).map(|m| (*term, m.access_count))
            })
            .min_by_key(|(_, count)| *count)
            .map(|(term, _)| term.to_string())
    }
}
```

**Complexity**: O(n) where n = number of candidates

## Usage Examples

### Example 1: Popular Content Caching

```rust
use liblevenshtein::cache::eviction::Lfu;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docs = PathMapDictionary::from_terms_with_values([
        ("getting-started", "Tutorial for beginners..."),
        ("advanced-guide", "Advanced techniques..."),
        ("changelog", "Release notes..."),
    ]);

    let lfu = Lfu::new(docs);

    // Simulate user access patterns
    // Getting started is very popular
    for _ in 0..100 {
        lfu.get_value("getting-started");
    }

    // Advanced guide moderately popular
    for _ in 0..50 {
        lfu.get_value("advanced-guide");
    }

    // Changelog rarely accessed
    for _ in 0..5 {
        lfu.get_value("changelog");
    }

    // Find least frequently used
    let candidates = vec!["getting-started", "advanced-guide", "changelog"];
    let lfu_entry = lfu.find_lfu(&candidates);

    // "changelog" has lowest access count (5)
    assert_eq!(lfu_entry, Some("changelog".to_string()));

    Ok(())
}
```

### Example 2: Code Completion Frequency

```rust
use liblevenshtein::cache::eviction::Lfu;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Track identifier usage in a codebase
    let identifiers = PathMapDictionary::from_terms_with_values([
        ("count", "i32"),      // Common loop variable
        ("result", "Vec<T>"),  // Common accumulator
        ("temp", "String"),    // Common temporary
        ("obscure_var", "f64"), // Rarely used
    ]);

    let lfu = Lfu::new(identifiers);

    // Simulate coding session
    for _ in 0..50 {
        lfu.get_value("count");   // Very common
    }
    for _ in 0..30 {
        lfu.get_value("result");  // Common
    }
    for _ in 0..10 {
        lfu.get_value("temp");    // Occasional
    }
    lfu.get_value("obscure_var"); // Rare

    // When memory constrained, evict LFU
    let candidates = vec!["count", "result", "temp", "obscure_var"];
    let evict = lfu.find_lfu(&candidates);

    assert_eq!(evict, Some("obscure_var".to_string()));
    println!("Evict: {} (least frequently used)", evict.unwrap());

    Ok(())
}
```

### Example 3: Long-Lived Cache with Stable Patterns

```rust
use liblevenshtein::cache::eviction::Lfu;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cache API responses with stable popularity
    let api_cache = PathMapDictionary::from_terms_with_values([
        ("/api/users", r#"{"users": [...]}"#),
        ("/api/posts", r#"{"posts": [...]}"#),
        ("/api/admin", r#"{"admin": [...]}"#),
    ]);

    let lfu = Lfu::new(api_cache);

    // Simulate API request patterns over time
    // Users endpoint: very popular
    for _ in 0..1000 {
        lfu.get_value("/api/users");
    }

    // Posts endpoint: moderately popular
    for _ in 0..500 {
        lfu.get_value("/api/posts");
    }

    // Admin endpoint: rarely accessed
    for _ in 0..10 {
        lfu.get_value("/api/admin");
    }

    // Evict least frequently used endpoint
    let endpoints = vec!["/api/users", "/api/posts", "/api/admin"];
    let evict = lfu.find_lfu(&endpoints);

    assert_eq!(evict, Some("/api/admin".to_string()));

    Ok(())
}
```

### Example 4: Composition with TTL

```rust
use liblevenshtein::cache::eviction::{Lfu, Ttl};
use liblevenshtein::prelude::*;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::from_terms_with_values([
        ("key1", 100),
        ("key2", 200),
        ("key3", 300),
    ]);

    // TTL filters expired entries, LFU tracks frequency among valid entries
    let ttl = Ttl::new(dict, Duration::from_secs(300));
    let lfu_ttl = Lfu::new(ttl);

    // Access pattern
    for _ in 0..100 {
        lfu_ttl.get_value("key1");
    }
    for _ in 0..10 {
        lfu_ttl.get_value("key2");
    }
    lfu_ttl.get_value("key3");

    // Among non-expired entries, find LFU
    let candidates = vec!["key1", "key2", "key3"];
    let lfu_entry = lfu_ttl.find_lfu(&candidates);

    assert_eq!(lfu_entry, Some("key3".to_string()));

    Ok(())
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| get_value | O(1) + O(d) | Counter increment + inner lookup |
| contains | O(1) + O(d) | Counter increment + inner check |
| find_lfu | O(n) | Linear scan of candidates |

### Space Complexity

**Per-Entry Overhead**: 8 bytes (u32 + padding)

**Total Metadata**: 8n bytes for n tracked entries

**Counter Overflow**:
- Uses `saturating_add()` to prevent overflow
- Max value: 4,294,967,295 (2^32 - 1)
- Saturates at max instead of wrapping

### Lock Contention

**RwLock Behavior**:
- Each access requires write lock (increment counter)
- More contention than LRU (which only updates timestamp)

**Mitigation**:
- Possible variant: use atomic counters for lower contention
- Shard cache by key prefix

## Comparison with Other Policies

| Aspect | LFU | LRU | Age |
|--------|-----|-----|-----|
| **Metric** | Frequency | Recency | Insertion order |
| **Best for** | Stable patterns | Temporal locality | FIFO queues |
| **Overhead** | 8 bytes | 16 bytes | 16 bytes |
| **Cold start** | Poor (new = LFU) | Good | N/A |
| **Eviction** | Lowest count | Oldest access | Oldest insert |

### When to Use LFU

✅ **Good For**:
- Long-lived caches with stable access patterns
- Popular content identification (top-k queries)
- Frequency-based ranking
- Scientific/analytical workloads

❌ **Not Ideal For**:
- Shifting access patterns (LRU better)
- Cold start scenarios (new entries immediately LFU)
- Time-sensitive eviction (use TTL)
- Bursty workloads (LRU + LFU hybrid better)

## Design Considerations

### Cold Start Problem

**Issue**: Newly inserted entries have count=1, making them immediately eligible for eviction.

**Mitigation Strategies**:

1. **Initial Count Boost**:
```rust
fn new_with_initial_count(count: u32) -> Self {
    Self { access_count: count }
}
```

2. **Aging Window**: Only consider entries inserted > N seconds ago
3. **Hybrid LFU+LRU**: Use recency as tiebreaker

### Counter Saturation

**Behavior**: Counter saturates at `u32::MAX` instead of wrapping.

```rust
fn increment(&mut self) {
    self.access_count = self.access_count.saturating_add(1);
}
```

**Why**: Prevents overflow causing hot entries to appear cold.

### Frequency Decay

**Not Implemented** (but could be added):
- Periodically halve all counters to favor recent frequency
- Implement sliding window frequency counting
- Use exponentially decaying average

## LFU vs LRU Tradeoffs

| Scenario | Better Policy | Rationale |
|----------|---------------|-----------|
| **Web cache** | LRU | Temporal locality strong |
| **Popular docs** | LFU | Stable popularity |
| **Code completion** | LRU | Recent identifiers likely reused |
| **API endpoints** | LFU | Stable usage patterns |
| **Bursty traffic** | LRU | Temporal spikes |
| **Long sessions** | LFU | Frequency accumulates |

### Hybrid Approach

**LRU-K**: Track last K accesses, combine recency + frequency. This is a
separate hybrid policy from the LFU policy described here.

## Related Documentation

- **[Layer 8 README](../README.md)** - Caching layer overview
- **[LRU Policy](./lru.md)** - Least Recently Used
- **[TTL Policy](./ttl.md)** - Time-to-live expiration
- **[MemoryPressure Policy](./memory-pressure.md)** - Memory-aware eviction

## References

1. Source code: `src/cache/eviction/lfu.rs`
2. O'Neil, E. J., et al. (1993). "The LRU-K Page Replacement Algorithm For Database Disk Buffering"
3. Lee, D., et al. (2001). "LRFU: A Spectrum of Policies that Subsumes the Least Recently Used and Least Frequently Used Policies"

## Quick Reference

```rust
use liblevenshtein::cache::eviction::Lfu;
use liblevenshtein::prelude::*;

// Basic usage
let dict = PathMapDictionary::from_terms_with_values([("key", "value")]);
let lfu = Lfu::new(dict);
lfu.get_value("key");

// Find LFU entry
let lfu_entry = lfu.find_lfu(&["key1", "key2"]);

// Composition with TTL
use std::time::Duration;
let ttl = Ttl::new(dict, Duration::from_secs(300));
let lfu_ttl = Lfu::new(ttl);
```
