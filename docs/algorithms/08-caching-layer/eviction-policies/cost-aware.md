# CostAware Eviction Policy

## Overview

The CostAware eviction policy balances age, size, and hit count to make intelligent cost-benefit eviction decisions. It computes a cost score that identifies entries with poor value-to-cost ratios.

**Formula**: Cost score = `$(\text{age} \times \text{size}) / (\text{hits} + 1)$`

**Eviction Strategy**: Highest score evicted first (old + large + unpopular).

## Architecture

### Metadata Structure

```rust
struct EntryMetadata {
    inserted_at: Instant,
    hit_count: u32,
    size_bytes: usize,
}
```

**Size**: 32 bytes (Instant=16 + u32=4 + usize=8 + padding=4)

### Wrapper Structure

```rust
pub struct CostAware<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}
```

## Core Operations

### Tracking Age, Size, and Hits

```rust
impl<D: MappedDictionary<V>, V: DictionaryValue> MappedDictionary<V> for CostAware<D> {
    fn get_value(&self, term: &str) -> Option<V> {
        let value = self.inner.get_value(term)?;

        // Track age, size, hits
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

### Finding High-Cost Entry

```rust
impl<D> CostAware<D> {
    pub fn find_highest_cost(&self, candidates: &[&str]) -> Option<String> {
        let metadata = self.metadata.read().unwrap();

        candidates
            .iter()
            .filter_map(|term| {
                metadata.get(*term).map(|m| (*term, m.cost_score()))
            })
            .max_by(|(_, score_a), (_, score_b)| {
                score_a.partial_cmp(score_b).unwrap()
            })
            .map(|(term, _)| term.to_string())
    }
}
```

**Complexity**: `$\mathcal{O}(n)$` where n = number of candidates

## Usage Examples

### Example 1: LLM Response Caching

```rust
use liblevenshtein::cache::eviction::CostAware;
use liblevenshtein::prelude::*;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cache expensive LLM responses
    let llm_cache = PathMapDictionary::from_terms_with_values([
        ("simple_query", "Short answer".to_string()),          // Small, cheap
        ("complex_query", "A".repeat(10000)),                  // Large, expensive
        ("popular_query", "Frequently accessed".to_string()), // Popular
    ]);

    let cost = CostAware::new(llm_cache);

    // Access pattern: popular query accessed frequently
    for _ in 0..100 {
        cost.get_value("popular_query");
    }

    // Simple query accessed occasionally
    for _ in 0..10 {
        cost.get_value("simple_query");
    }

    // Complex query accessed once, then forgotten
    cost.get_value("complex_query");

    // Wait for age to accumulate
    thread::sleep(Duration::from_secs(10));

    // Find highest cost entry
    let candidates = vec!["simple_query", "complex_query", "popular_query"];
    let high_cost = cost.find_highest_cost(&candidates);

    // complex_query: (10s × 10000 bytes) / (1 + 1) = ~50,000
    // simple_query: (10s × 12 bytes) / (10 + 1) = ~11
    // popular_query: (10s × 20 bytes) / (100 + 1) = ~2
    assert_eq!(high_cost, Some("complex_query".to_string()));

    Ok(())
}
```

### Example 2: Documentation Search

```rust
use liblevenshtein::cache::eviction::CostAware;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cache search results of varying sizes
    let search_cache = PathMapDictionary::from_terms_with_values([
        ("api_docs", "A".repeat(5000)),     // Large result set
        ("tutorial", "B".repeat(2000)),     // Medium result set
        ("faq", "C".repeat(500)),           // Small result set
    ]);

    let cost = CostAware::new(search_cache);

    // FAQ accessed very frequently (100×)
    for _ in 0..100 {
        cost.get_value("faq");
    }

    // Tutorial accessed moderately (20×)
    for _ in 0..20 {
        cost.get_value("tutorial");
    }

    // API docs accessed rarely (2×)
    for _ in 0..2 {
        cost.get_value("api_docs");
    }

    // Cost scores (assuming 60s age):
    // api_docs: (60 × 5000) / (2 + 1) = 100,000
    // tutorial: (60 × 2000) / (20 + 1) = ~5,714
    // faq: (60 × 500) / (100 + 1) = ~297

    let candidates = vec!["api_docs", "tutorial", "faq"];
    let evict = cost.find_highest_cost(&candidates);

    assert_eq!(evict, Some("api_docs".to_string()));

    Ok(())
}
```

### Example 3: Balancing Cost and Value

```rust
use liblevenshtein::cache::eviction::CostAware;
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = PathMapDictionary::from_terms_with_values([
        ("small_hot", vec![1, 2, 3]),           // 24 bytes, popular
        ("large_hot", vec![1; 1000]),           // 8000 bytes, popular
        ("small_cold", vec![4, 5]),             // 16 bytes, unpopular
        ("large_cold", vec![1; 2000]),          // 16000 bytes, unpopular
    ]);

    let cost = CostAware::new(cache);

    // Hot entries: accessed 100 times
    for _ in 0..100 {
        cost.get_value("small_hot");
        cost.get_value("large_hot");
    }

    // Cold entries: accessed once
    cost.get_value("small_cold");
    cost.get_value("large_cold");

    // Cost scores (assuming 30s age):
    // small_hot: (30 × 24) / (100 + 1) ≈ 7
    // large_hot: (30 × 8000) / (100 + 1) ≈ 2,376
    // small_cold: (30 × 16) / (1 + 1) = 240
    // large_cold: (30 × 16000) / (1 + 1) = 240,000 ← highest!

    let candidates = vec!["small_hot", "large_hot", "small_cold", "large_cold"];
    let evict = cost.find_highest_cost(&candidates);

    assert_eq!(evict, Some("large_cold".to_string()));

    Ok(())
}
```

### Example 4: Composition with LRU

```rust
use liblevenshtein::cache::eviction::{CostAware, Lru};
use liblevenshtein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::from_terms_with_values([
        ("key1", vec![1; 1000]),
        ("key2", vec![1; 100]),
    ]);

    // CostAware tracks age/size/hits, LRU tracks recency
    let cost = CostAware::new(dict);
    let lru_cost = Lru::new(cost);

    lru_cost.get_value("key1");
    lru_cost.get_value("key2");

    // Can query both dimensions:
    // - LRU: find least recently used
    // - CostAware: find highest cost

    Ok(())
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| get_value | `$\mathcal{O}(1)$` + `$\mathcal{O}(d)$` | Metadata update + inner lookup |
| find_highest_cost | `$\mathcal{O}(n)$` | Linear scan of candidates |

### Space Complexity

**Per-Entry Overhead**: 32 bytes
- `inserted_at`: 16 bytes (Instant)
- `hit_count`: 4 bytes (u32)
- `size_bytes`: 8 bytes (usize)
- Padding: 4 bytes

**Total Metadata**: 32n bytes for n entries

## Cost Score Analysis

### Formula Breakdown

```math
\text{cost\_score} = (\text{age} \times \text{size}) / (\text{hits} + 1)
```

**Components**:
- **age**: Time since insertion (seconds)
- **size**: Entry size in bytes
- **hits**: Number of successful accesses
- **+1**: Prevents division by zero

### Intuition

| Factor | Effect on Score | Eviction Likelihood |
|--------|-----------------|---------------------|
| ↑ age | ↑ score | More likely (stale) |
| ↑ size | ↑ score | More likely (costly) |
| ↑ hits | ↓ score | Less likely (valuable) |

### Example Scores

Given 60-second age:

| Entry | Size | Hits | Score | Priority |
|-------|------|------|-------|----------|
| Large, cold | 10KB | 1 | 300,000 | Evict first |
| Large, hot | 10KB | 100 | 5,940 | Evict third |
| Small, cold | 100B | 1 | 3,000 | Evict second |
| Small, hot | 100B | 100 | 59 | Keep |

## Comparison with Other Policies

| Aspect | CostAware | MemoryPressure | LRU | LFU |
|--------|-----------|----------------|-----|-----|
| **Factors** | Age, size, hits | Size, hit rate | Recency | Frequency |
| **Time-aware** | ✓ | ✗ | ✓ | ✗ |
| **Size-aware** | ✓ | ✓ | ✗ | ✗ |
| **Hit-aware** | ✓ | ✓ | ✗ | ✓ |
| **Overhead** | 32 bytes | 24 bytes | 16 bytes | 8 bytes |

### When to Use CostAware

✅ **Good For**:
- Expensive-to-regenerate entries (LLM, API calls)
- Variable-size values with varying utility
- Balancing freshness, size, and popularity
- Multi-factor optimization

❌ **Not Ideal For**:
- Uniform-size values (use LRU or LFU)
- Pure time-based eviction (use TTL)
- Simple recency (use LRU)
- Memory-only concerns (use MemoryPressure)

## Design Considerations

### Tuning the Formula

**Standard Formula**:
```rust
(age * size) / (hits + 1)
```

**Alternative Formulas**:

1. **Emphasize Age More**:
```rust
(age.powi(2) * size) / (hits + 1)
```

2. **Emphasize Size More**:
```rust
(age * size.powi(2)) / (hits + 1)
```

3. **Logarithmic Size** (diminishing returns):
```rust
(age * log(size)) / (hits + 1)
```

4. **Weighted Formula**:
```rust
(age * W_age + size * W_size) / (hits * W_hits + 1)
```

### Age Measurement

**Current**: Absolute age since insertion

**Alternatives**:
- **Last access age**: Time since last access (like LRU)
- **Exponential decay**: Recent hits weighted more
- **Sliding window**: Only count hits in last N seconds

## Cost-Benefit Scenarios

### Scenario 1: LLM API Responses

```
Entry A: 100 bytes, 1 hit, 10s old → score = 500
Entry B: 10KB, 50 hits, 60s old → score = 11,764
Entry C: 1KB, 100 hits, 120s old → score = 1,188

Evict B (large, moderately popular, getting old)
```

### Scenario 2: Compilation Cache

```
Entry A: Small .o file, frequently recompiled → Low score
Entry B: Large .o file, rarely recompiled → High score
Entry C: Medium .o file, old + unpopular → High score

Evict B or C (size + low utility)
```

### Scenario 3: Database Query Cache

```
Query A: SELECT * (large result), popular → Moderate score
Query B: SELECT id (small result), unpopular → Moderate score
Query C: JOIN 10 tables (huge result), one-time → High score

Evict C (large + never reused)
```

## Related Documentation

- **[Layer 8 README](../README.md)** - Caching layer overview
- **[MemoryPressure Policy](./memory-pressure.md)** - Similar multi-factor approach
- **[LRU Policy](./lru.md)** - Recency-only eviction
- **[LFU Policy](./lfu.md)** - Frequency-only eviction

## References

1. Source code: `src/cache/eviction/cost_aware.rs`
2. Megiddo, N., & Modha, D. S. (2003). "ARC: A Self-Tuning, Low Overhead Replacement Cache"
3. Jiang, S., & Zhang, X. (2002). "LIRS: An Efficient Low Inter-reference Recency Set Replacement Policy"

## Quick Reference

```rust
use liblevenshtein::cache::eviction::CostAware;
use liblevenshtein::prelude::*;

// Basic usage
let dict = PathMapDictionary::from_terms_with_values([
    ("key", vec![1, 2, 3])
]);
let cost = CostAware::new(dict);
cost.get_value("key");

// Find highest cost
let high_cost = cost.find_highest_cost(&["key1", "key2"]);

// Composition with LRU
let lru_cost = Lru::new(CostAware::new(dict));
```
