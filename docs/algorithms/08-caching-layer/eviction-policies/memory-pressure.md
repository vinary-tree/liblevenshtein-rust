# MemoryPressure Eviction Policy

## Overview

The MemoryPressure eviction policy balances memory consumption with cache effectiveness. It tracks entry size and hit rate to identify large, underutilized entries for eviction.

**Formula**: Memory pressure score = $`\text{size} / (\text{hit\_rate} + 0.1)`$

**Eviction Strategy**: Highest score evicted first (large size + low hit rate).

## Architecture

### Metadata Structure

```rust
struct EntryMetadata {
    size_bytes: usize,
    hit_count: u32,
    access_count: u32,
}
```

**Size**: 24 bytes (3×u64)

**Hit Rate**: `hit_count / access_count`

### Wrapper Structure

```rust
pub struct MemoryPressure<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}
```

## Core Operations

### Tracking Size and Hits

```rust
impl<D: MappedDictionary<V>, V: DictionaryValue> MappedDictionary<V> for MemoryPressure<D> {
    fn get_value(&self, term: &str) -> Option<V> {
        let value = self.inner.get_value(term)?;

        // Track size and hits
        {
            let mut metadata = self.metadata.write().unwrap();
            metadata
                .entry(term.to_string())
                .and_modify(|m| m.increment())
                .or_insert_with(|| EntryMetadata::new(value.size_in_bytes()));
        }

        Some(value)
    }
}
```

### Finding High-Pressure Entries

```rust
impl<D> MemoryPressure<D> {
    pub fn find_highest_pressure(&self, candidates: &[&str]) -> Option<String> {
        let metadata = self.metadata.read().unwrap();

        candidates
            .iter()
            .filter_map(|term| {
                metadata.get(*term).map(|m| (*term, m.memory_pressure_score()))
            })
            .max_by(|(_, score_a), (_, score_b)| {
                score_a.partial_cmp(score_b).unwrap()
            })
            .map(|(term, _)| term.to_string())
    }
}
```

**Complexity**: $`\mathcal{O}(n)`$ where n = number of candidates

## Usage Examples

### Example 1: Large Value Caching

```rust
use liblevenshtein::cache::eviction::MemoryPressure;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cache embeddings/vectors of different sizes
    let embeddings = PathMapDictionary::from_terms_with_values([
        ("small_vec", vec![1.0, 2.0, 3.0]),               // 24 bytes
        ("medium_vec", vec![1.0; 100]),                   // 800 bytes
        ("large_vec", vec![1.0; 1000]),                   // 8000 bytes
    ]);

    let memory = MemoryPressure::new(embeddings);

    // Access patterns: small (100×), medium (50×), large (1×)
    for _ in 0..100 {
        memory.get_value("small_vec");
    }
    for _ in 0..50 {
        memory.get_value("medium_vec");
    }
    memory.get_value("large_vec");  // Only once

    // Find highest memory pressure
    let candidates = vec!["small_vec", "medium_vec", "large_vec"];
    let high_pressure = memory.find_highest_pressure(&candidates);

    // large_vec has highest pressure: 8000 / (1/1 + 0.1) = ~7273
    // medium_vec: 800 / (50/50 + 0.1) = ~727
    // small_vec: 24 / (100/100 + 0.1) = ~22
    assert_eq!(high_pressure, Some("large_vec".to_string()));

    Ok(())
}
```

### Example 2: AST Caching

```rust
use liblevenshtein::cache::eviction::MemoryPressure;
use liblevenshtein::prelude::*;

#[derive(Clone)]
struct Ast {
    nodes: Vec<String>,
}

impl Ast {
    fn size(&self) -> usize {
        self.nodes.iter().map(|s| s.len()).sum::<usize>()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cache parsed ASTs
    let asts = PathMapDictionary::from_terms_with_values([
        ("small.rs", Ast { nodes: vec!["fn".to_string()] }),
        ("huge.rs", Ast { nodes: vec!["fn".to_string(); 10000] }),
    ]);

    let memory = MemoryPressure::new(asts);

    // small.rs accessed frequently (hot file)
    for _ in 0..100 {
        memory.get_value("small.rs");
    }

    // huge.rs accessed once (cold file)
    memory.get_value("huge.rs");

    // Evict huge.rs (high memory pressure)
    let candidates = vec!["small.rs", "huge.rs"];
    let evict = memory.find_highest_pressure(&candidates);
    assert_eq!(evict, Some("huge.rs".to_string()));

    Ok(())
}
```

### Example 3: Adaptive Memory Management

```rust
use liblevenshtein::cache::eviction::MemoryPressure;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = PathMapDictionary::from_terms_with_values([
        ("efficient", vec![1; 10]),      // Small, high hit rate
        ("wasteful", vec![1; 1000]),     // Large, low hit rate
        ("balanced", vec![1; 100]),      // Medium size, medium hits
    ]);

    let memory = MemoryPressure::new(cache);

    // Simulate access patterns
    for _ in 0..100 {
        memory.get_value("efficient");
    }
    for _ in 0..10 {
        memory.get_value("balanced");
    }
    memory.get_value("wasteful");

    // Memory pressure scores:
    // efficient: 40 / (1.0 + 0.1) = ~36
    // balanced: 400 / (0.1 + 0.1) = ~2000
    // wasteful: 4000 / (1/1 + 0.1) = ~3636

    let candidates = vec!["efficient", "balanced", "wasteful"];
    let evict = memory.find_highest_pressure(&candidates);
    assert_eq!(evict, Some("wasteful".to_string()));

    Ok(())
}
```

### Example 4: Composition with LRU

```rust
use liblevenshtein::cache::eviction::{MemoryPressure, Lru};
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::from_terms_with_values([
        ("key1", vec![1; 1000]),
        ("key2", vec![1; 10]),
    ]);

    // MemoryPressure tracks size/hits, LRU tracks recency
    let memory = MemoryPressure::new(dict);
    let lru_memory = Lru::new(memory);

    lru_memory.get_value("key1");
    lru_memory.get_value("key2");

    // Can query both dimensions:
    // - LRU: find least recently used
    // - MemoryPressure: find highest memory pressure

    Ok(())
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| get_value | $`\mathcal{O}(1)`$ + $`\mathcal{O}(d)`$ | Metadata update + inner lookup |
| find_highest_pressure | $`\mathcal{O}(n)`$ | Linear scan of candidates |

### Space Complexity

**Per-Entry Overhead**: 24 bytes (3×u64)

**Metadata**:
- `size_bytes`: usize (8 bytes)
- `hit_count`: u32 (4 bytes)
- `access_count`: u32 (4 bytes)
- **Total**: 16 bytes data + 8 bytes padding = 24 bytes

### Hit Rate Accuracy

**Hit Rate Calculation**: `hit_count / access_count`

**Behavior**:
- First access: hit_rate = 1.0 (100%)
- Cache miss: access_count++, hit_count unchanged → hit_rate decreases
- Cache hit: both increment → hit_rate increases

## Comparison with Other Policies

| Aspect | MemoryPressure | CostAware | LRU |
|--------|----------------|-----------|-----|
| **Metric** | $`\text{size} / \text{hit\_rate}`$ | $`(\text{age} \times \text{size}) / \text{hits}`$ | Recency |
| **Size-aware** | ✓ | ✓ | ✗ |
| **Time-aware** | ✗ | ✓ | ✓ |
| **Hit-aware** | ✓ | ✓ | ✗ |
| **Overhead** | 24 bytes | 32 bytes | 16 bytes |

### When to Use MemoryPressure

✅ **Good For**:
- Memory-constrained systems
- Variable-size values (embeddings, ASTs, images)
- Maximizing cache effectiveness per byte
- Adaptive memory management

❌ **Not Ideal For**:
- Uniform-size values (LRU simpler)
- Time-sensitive eviction (use TTL)
- Frequency-only (use LFU)

## Design Considerations

### Size Estimation

**DictionaryValue::size_in_bytes()**:
- Primitive types: `std::mem::size_of::<T>()`
- Vec/String: `capacity() * item_size`
- Custom types: implement trait

**Example**:
```rust
impl DictionaryValue for MyStruct {
    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.capacity()
    }
}
```

### Pressure Score Tuning

**Formula**: $`\text{size} / (\text{hit\_rate} + \varepsilon)`$

**Epsilon (0.1)**:
- Prevents division by zero
- Dampens impact of very low hit rates
- Adjustable based on use case

**Alternative Formulas**:
```rust
// Emphasize size more
size / (hit_rate + 0.01)

// Emphasize hit rate more
size / (hit_rate + 0.5)

// Logarithmic size impact
log(size) / (hit_rate + 0.1)
```

## Related Documentation

- **[Layer 8 README](../README.md)** - Caching layer overview
- **[LRU Policy](./lru.md)** - Least Recently Used
- **[TTL Policy](./ttl.md)** - Time-to-live expiration

## References

1. Source code: `src/cache/eviction/memory_pressure.rs`
2. Megiddo, N., & Modha, D. S. (2003). "ARC: A Self-Tuning, Low Overhead Replacement Cache"
3. Kim, H., et al. (2001). "LRFU: A Spectrum of Policies that Subsumes the Least Recently Used and Least Frequently Used Policies"

## Quick Reference

```rust
use liblevenshtein::cache::eviction::MemoryPressure;
use liblevenshtein::prelude::*;

// Basic usage
let dict = PathMapDictionary::from_terms_with_values([
    ("key", vec![1, 2, 3])
]);
let memory = MemoryPressure::new(dict);
memory.get_value("key");

// Find highest pressure
let high_pressure = memory.find_highest_pressure(&["key1", "key2"]);

// Composition with LRU
let lru_memory = Lru::new(MemoryPressure::new(dict));
```
