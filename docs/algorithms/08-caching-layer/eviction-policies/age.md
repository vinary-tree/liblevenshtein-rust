# Age (FIFO) Eviction Policy

## Overview

The Age eviction policy tracks insertion time for each entry and identifies candidates for eviction based on how long they've been in the cache. This implements a FIFO (First In, First Out) strategy where the oldest entries are evicted first, regardless of access patterns.

**Formula**: Eviction score = `inserted_at.elapsed()`

**Eviction Strategy**: Highest score (oldest insertion) evicted first.

## Architecture

### Metadata Structure

```rust
struct EntryMetadata {
    inserted_at: Instant,
}
```

**Size**: 16 bytes (Instant = 2×u64 on most platforms)

### Wrapper Structure

```rust
pub struct Age<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}
```

**Thread-Safety**: RwLock allows concurrent reads, serialized writes.

## Core Operations

### Tracking Insertion Time

```rust
impl<D: MappedDictionary<V>, V: DictionaryValue> MappedDictionary<V> for Age<D> {
    fn get_value(&self, term: &str) -> Option<V> {
        // Record insertion time (first access only)
        self.record_access(term);

        // Forward to inner dictionary
        self.inner.get_value(term)
    }
}

fn record_access(&self, term: &str) {
    let mut metadata = self.metadata.write().unwrap();
    metadata
        .entry(term.to_string())
        .or_insert_with(EntryMetadata::new);  // Only sets on first access
}
```

**Key Insight**: Unlike LRU, insertion time is **never updated** on subsequent accesses.

**Complexity**: $`\mathcal{O}(1)`$ metadata check + $`\mathcal{O}(d)`$ inner lookup

### Finding Oldest Entry

```rust
impl<D> Age<D> {
    pub fn find_oldest(&self, terms: &[&str]) -> Option<String> {
        let metadata = self.metadata.read().unwrap();
        terms
            .iter()
            .filter_map(|&term| metadata.get(term).map(|m| (term, m.age())))
            .max_by_key(|(_, age)| *age)
            .map(|(term, _)| term.to_string())
    }
}
```

**Complexity**: $`\mathcal{O}(n)`$ where n = number of candidates

### Eviction with Cleanup

```rust
impl<D> Age<D> {
    pub fn evict_oldest(&self, terms: &[&str]) -> Option<String> {
        if let Some(oldest_term) = self.find_oldest(terms) {
            let mut metadata = self.metadata.write().unwrap();
            metadata.remove(&oldest_term);
            Some(oldest_term)
        } else {
            None
        }
    }
}
```

**Note**: Removes metadata entry on eviction.

## Usage Examples

### Example 1: Basic FIFO Queue

```rust
use liblevenshtein::cache::eviction::Age;
use liblevenshtein::prelude::*;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let queue = PathMapDictionary::from_terms_with_values([
        ("task1", "Process data"),
        ("task2", "Send email"),
        ("task3", "Update logs"),
    ]);

    let age = Age::new(queue);

    // Insert tasks in order
    age.get_value("task1");  // Oldest
    thread::sleep(Duration::from_millis(10));
    age.get_value("task2");
    thread::sleep(Duration::from_millis(10));
    age.get_value("task3");  // Newest

    // Find oldest task
    let candidates = vec!["task1", "task2", "task3"];
    let oldest = age.find_oldest(&candidates);

    // task1 is oldest (FIFO)
    assert_eq!(oldest, Some("task1".to_string()));

    Ok(())
}
```

### Example 2: Age vs LRU Behavior

```rust
use liblevenshtein::cache::eviction::{Age, Lru};
use liblevenshtein::prelude::*;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::from_terms_with_values([
        ("old_entry", 1),
        ("new_entry", 2),
    ]);

    let age_cache = Age::new(dict.clone());
    let lru_cache = Lru::new(dict);

    // Insert old_entry first
    age_cache.get_value("old_entry");
    lru_cache.get_value("old_entry");
    thread::sleep(Duration::from_millis(20));

    // Insert new_entry
    age_cache.get_value("new_entry");
    lru_cache.get_value("new_entry");
    thread::sleep(Duration::from_millis(20));

    // Access old_entry again (frequently used)
    for _ in 0..10 {
        age_cache.get_value("old_entry");
        lru_cache.get_value("old_entry");
    }

    // Age: old_entry is still oldest (FIFO ignores access patterns)
    assert_eq!(
        age_cache.find_oldest(&["old_entry", "new_entry"]),
        Some("old_entry".to_string())
    );

    // LRU: new_entry is LRU (old_entry was recently accessed)
    assert_eq!(
        lru_cache.find_lru(&["old_entry", "new_entry"]),
        Some("new_entry".to_string())
    );

    Ok(())
}
```

### Example 3: Fair Cache Eviction

```rust
use liblevenshtein::cache::eviction::Age;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cache where all entries should get equal time
    let cache = PathMapDictionary::from_terms_with_values([
        ("user_session_1", "data1"),
        ("user_session_2", "data2"),
        ("user_session_3", "data3"),
    ]);

    let age = Age::new(cache);

    // Sessions created in order
    age.get_value("user_session_1");  // Oldest
    age.get_value("user_session_2");
    age.get_value("user_session_3");  // Newest

    // Even if session_1 is very active:
    for _ in 0..100 {
        age.get_value("user_session_1");
    }

    // Fair eviction: session_1 still oldest (FIFO fairness)
    let candidates = vec!["user_session_1", "user_session_2", "user_session_3"];
    let evict = age.find_oldest(&candidates);

    assert_eq!(evict, Some("user_session_1".to_string()));
    println!("Evict oldest session fairly: {:?}", evict);

    Ok(())
}
```

### Example 4: Predictable Testing Behavior

```rust
use liblevenshtein::cache::eviction::Age;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test cache behavior predictably
    let cache = PathMapDictionary::from_terms_with_values([
        ("entry_a", vec![1, 2, 3]),
        ("entry_b", vec![4, 5, 6]),
        ("entry_c", vec![7, 8, 9]),
    ]);

    let age = Age::new(cache);

    // Insert in specific order
    age.get_value("entry_a");
    age.get_value("entry_b");
    age.get_value("entry_c");

    // Deterministic eviction order (FIFO)
    let candidates = vec!["entry_a", "entry_b", "entry_c"];

    // First eviction: entry_a (oldest)
    let evict1 = age.evict_oldest(&candidates);
    assert_eq!(evict1, Some("entry_a".to_string()));

    // Second eviction: entry_b (next oldest)
    let evict2 = age.evict_oldest(&candidates);
    assert_eq!(evict2, Some("entry_b".to_string()));

    // Third eviction: entry_c (newest)
    let evict3 = age.evict_oldest(&candidates);
    assert_eq!(evict3, Some("entry_c".to_string()));

    println!("Predictable FIFO eviction: a → b → c");

    Ok(())
}
```

### Example 5: Cache Rotation

```rust
use liblevenshtein::cache::eviction::Age;
use liblevenshtein::prelude::*;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rotating log cache with fixed size
    let logs = PathMapDictionary::from_terms_with_values([
        ("log_00:00", "System started"),
        ("log_00:01", "User login"),
        ("log_00:02", "Data processed"),
        ("log_00:03", "Task completed"),
    ]);

    let age = Age::new(logs);
    let max_logs = 3;

    // Access all logs
    age.get_value("log_00:00");
    thread::sleep(Duration::from_millis(10));
    age.get_value("log_00:01");
    thread::sleep(Duration::from_millis(10));
    age.get_value("log_00:02");
    thread::sleep(Duration::from_millis(10));
    age.get_value("log_00:03");

    let candidates = vec!["log_00:00", "log_00:01", "log_00:02", "log_00:03"];

    // Evict oldest to maintain max_logs size
    let evict = age.find_oldest(&candidates);
    assert_eq!(evict, Some("log_00:00".to_string()));

    println!("Rotate out oldest log: {:?}", evict);

    Ok(())
}
```

### Example 6: Composition with TTL

```rust
use liblevenshtein::cache::eviction::{Age, Ttl};
use liblevenshtein::prelude::*;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::from_terms_with_values([
        ("key1", "value1"),
        ("key2", "value2"),
        ("key3", "value3"),
    ]);

    // TTL filters expired entries, Age tracks insertion order among valid entries
    let ttl = Ttl::new(dict, Duration::from_secs(300));
    let age_ttl = Age::new(ttl);

    // Insert in order
    age_ttl.get_value("key1");
    age_ttl.get_value("key2");
    age_ttl.get_value("key3");

    // Among non-expired entries, find oldest (FIFO)
    let candidates = vec!["key1", "key2", "key3"];
    let oldest = age_ttl.find_oldest(&candidates);

    assert_eq!(oldest, Some("key1".to_string()));

    Ok(())
}
```

### Example 7: Request Queue Management

```rust
use liblevenshtein::cache::eviction::Age;
use liblevenshtein::prelude::*;

#[derive(Clone, Debug)]
struct Request {
    id: u64,
    priority: u8,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Request queue with FIFO processing
    let requests = PathMapDictionary::from_terms_with_values([
        ("req_001", Request { id: 1, priority: 5 }),
        ("req_002", Request { id: 2, priority: 8 }),
        ("req_003", Request { id: 3, priority: 3 }),
    ]);

    let age = Age::new(requests);

    // Enqueue requests
    age.get_value("req_001");
    age.get_value("req_002");
    age.get_value("req_003");

    // Process oldest request first (fair queue)
    let candidates = vec!["req_001", "req_002", "req_003"];
    let next = age.find_oldest(&candidates);

    assert_eq!(next, Some("req_001".to_string()));
    println!("Process next request: {:?}", next);

    Ok(())
}
```

### Example 8: Benchmark Result Caching

```rust
use liblevenshtein::cache::eviction::Age;
use liblevenshtein::prelude::*;

#[derive(Clone)]
struct BenchmarkResult {
    test_name: String,
    duration_ms: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cache benchmark results, evict oldest when full
    let benchmarks = PathMapDictionary::from_terms_with_values([
        ("test_sort", BenchmarkResult { test_name: "sort".into(), duration_ms: 42 }),
        ("test_hash", BenchmarkResult { test_name: "hash".into(), duration_ms: 15 }),
        ("test_search", BenchmarkResult { test_name: "search".into(), duration_ms: 28 }),
    ]);

    let age = Age::new(benchmarks);

    // Run benchmarks in order
    age.get_value("test_sort");
    age.get_value("test_hash");
    age.get_value("test_search");

    // Cache full - evict oldest result
    let candidates = vec!["test_sort", "test_hash", "test_search"];
    let evict = age.find_oldest(&candidates);

    assert_eq!(evict, Some("test_sort".to_string()));

    Ok(())
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| get_value | $`\mathcal{O}(1)`$ + $`\mathcal{O}(d)`$ | Metadata check + inner lookup |
| find_oldest | $`\mathcal{O}(n)`$ | Linear scan of candidates |
| evict_oldest | $`\mathcal{O}(n)`$ + $`\mathcal{O}(1)`$ | Find oldest + remove metadata |
| age | $`\mathcal{O}(1)`$ | Read metadata |

Where:
- `d` = inner dictionary operation complexity
- `n` = number of candidates

### Space Complexity

**Per-Entry Overhead**: 16 bytes (Instant)

**Total Metadata**: 16n bytes for n tracked entries

**Comparison**:
- Age: 16 bytes/entry (Instant)
- LRU: 16 bytes/entry (Instant)
- LFU: 8 bytes/entry (u32)
- MemoryPressure: 24 bytes/entry (usize + 2×u32)

### Lock Contention

**RwLock Behavior**:
- **First access**: Exclusive lock (insert metadata)
- **Subsequent accesses**: No lock (metadata unchanged)
- **Query operations**: Shared read lock

**Advantage**: Less contention than LRU (which updates on every access)

## Comparison with Other Policies

| Aspect | Age | LRU | LFU | TTL |
|--------|-----|-----|-----|-----|
| **Metric** | Insertion time | Last access | Access count | Expiration |
| **Access updates** | No | Yes | Yes | No |
| **Overhead** | 16 bytes | 16 bytes | 8 bytes | 16 bytes |
| **Best for** | FIFO fairness | Temporal locality | Hot content | Time-sensitive |
| **Eviction** | Oldest inserted | Oldest accessed | Least accessed | Expired only |
| **Deterministic** | ✓ | ✗ | ✗ | ✓ |

### When to Use Age

✅ **Good For**:
- Fair eviction regardless of popularity
- Predictable testing scenarios
- Simple FIFO queues
- Request/task processing
- Log rotation
- Fixed-size circular buffers

❌ **Not Ideal For**:
- Optimizing cache hit rate (use LRU/LFU)
- Hot content retention (use LFU)
- Time-sensitive expiration (use TTL)
- Memory-constrained systems (use MemoryPressure)

## Design Considerations

### Age vs LRU: Key Difference

**Age (FIFO)**:
```rust
// Insertion time NEVER changes
age.get_value("entry");  // Sets inserted_at (first time)
age.get_value("entry");  // No metadata update
age.get_value("entry");  // No metadata update
// Result: Age keeps increasing → eventually evicted
```

**LRU**:
```rust
// Last access time ALWAYS updated
lru.get_value("entry");  // Sets last_accessed
lru.get_value("entry");  // Updates last_accessed (recent!)
lru.get_value("entry");  // Updates last_accessed (recent!)
// Result: Always recent → never evicted
```

### Fairness Guarantee

**Age Policy**: Every entry gets equal lifespan in cache, regardless of:
- Access frequency
- Access patterns
- Entry size
- Entry value

**Use Case**: Multi-user systems where fairness is more important than efficiency.

### Determinism for Testing

**Predictable Behavior**:
```rust
// Test setup
let age = Age::new(dict);
age.get_value("a");  // Inserted at T0
age.get_value("b");  // Inserted at T1
age.get_value("c");  // Inserted at T2

// Deterministic eviction order: a → b → c (always)
```

**Contrast with LRU/LFU**: Eviction order depends on access patterns (non-deterministic in tests).

### Metadata Cleanup

**Eviction with Cleanup**:
```rust
pub fn evict_oldest(&self, terms: &[&str]) -> Option<String> {
    if let Some(oldest_term) = self.find_oldest(terms) {
        let mut metadata = self.metadata.write().unwrap();
        metadata.remove(&oldest_term);  // Clean up metadata
        Some(oldest_term)
    } else {
        None
    }
}
```

**Manual Cleanup**:
```rust
pub fn clear_metadata(&self) {
    let mut metadata = self.metadata.write().unwrap();
    metadata.clear();
}
```

## Age Policy Scenarios

### Scenario 1: Multi-User Session Cache

```
User A: Logs in at T0, very active (100 requests/min)
User B: Logs in at T1, moderately active (10 requests/min)
User C: Logs in at T2, inactive (1 request/min)

Cache full → Evict User A (oldest session, despite being most active)
Result: Fair treatment of all users
```

### Scenario 2: Log Rotation

```
log_00:00: 1000 lines, frequently queried
log_00:01: 500 lines, occasionally queried
log_00:02: 100 lines, rarely queried

Cache full → Evict log_00:00 (oldest log, despite being popular)
Result: Natural log rotation (FIFO)
```

### Scenario 3: Request Queue

```
Priority requests:
- High priority at T0
- Medium priority at T1
- Low priority at T2

Process oldest first (T0) regardless of priority
Result: Fair queue processing (no starvation)
```

## Age Score Analysis

### Formula

```
age_score = inserted_at.elapsed()
```

**Components**:
- **inserted_at**: Timestamp of first access (never updated)
- **elapsed()**: Duration since insertion

**Monotonicity**: Age score always increases over time.

### Example Scores

Given entries inserted at different times:

| Entry | Insertion Time | Current Age | Score | Priority |
|-------|----------------|-------------|-------|----------|
| entry_a | T0 (60s ago) | 60s | 60,000ms | Evict first |
| entry_b | T30 (30s ago) | 30s | 30,000ms | Evict second |
| entry_c | T50 (10s ago) | 10s | 10,000ms | Keep |

**Access patterns irrelevant**: Even if entry_c is never accessed and entry_a is accessed 1000×, entry_a is still evicted first.

## Related Documentation

- **[Layer 8 README](../README.md)** - Caching layer overview
- **[LRU Policy](./lru.md)** - Least Recently Used (access-based)
- **[LFU Policy](./lfu.md)** - Least Frequently Used
- **[TTL Policy](./ttl.md)** - Time-to-live expiration

## References

1. Source code: `src/cache/eviction/age.rs`
2. Belady, L. A. (1966). "A Study of Replacement Algorithms for Virtual-Storage Computer"
3. Podlipnig, S., & Böszörmenyi, L. (2003). "A Survey of Web Cache Replacement Strategies"

## Quick Reference

```rust
use liblevenshtein::cache::eviction::Age;
use liblevenshtein::prelude::*;

// Basic usage
let dict = PathMapDictionary::from_terms_with_values([("key", "value")]);
let age = Age::new(dict);
age.get_value("key");

// Find oldest entry
let oldest = age.find_oldest(&["key1", "key2"]);

// Evict oldest entry (with metadata cleanup)
let evicted = age.evict_oldest(&["key1", "key2"]);

// Get age for entry
let entry_age = age.age("key1");

// Clear all metadata
age.clear_metadata();

// Composition with TTL
use std::time::Duration;
let ttl = Ttl::new(dict, Duration::from_secs(300));
let age_ttl = Age::new(ttl);
```
