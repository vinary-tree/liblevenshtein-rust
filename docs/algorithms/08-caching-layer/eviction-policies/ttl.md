# TTL (Time-to-Live) Eviction Policy

## Overview

The TTL eviction policy filters entries based on expiration time. Entries older than the specified duration are automatically excluded from results, effectively "evicting" them without physical removal.

**Formula**: Entry valid if `inserted_at.elapsed() <= duration`

**Eviction Strategy**: Hard expiration - entries inaccessible after TTL expires.

## Architecture

### Metadata Structure

```rust
struct EntryMetadata {
    inserted_at: Instant,
}
```

**Size**: 16 bytes (Instant)

### Wrapper Structure

```rust
pub struct Ttl<D> {
    inner: D,
    duration: Duration,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}
```

## Core Operations

### Expiration Check

```rust
impl<D: MappedDictionary<V>, V> MappedDictionary<V> for Ttl<D> {
    fn get_value(&self, term: &str) -> Option<V> {
        // Check if entry expired
        {
            let metadata = self.metadata.read().unwrap();
            if let Some(meta) = metadata.get(term) {
                if meta.inserted_at.elapsed() > self.duration {
                    return None;  // Expired!
                }
            }
        }

        // Update/create metadata
        {
            let mut metadata = self.metadata.write().unwrap();
            metadata
                .entry(term.to_string())
                .or_insert_with(EntryMetadata::new);
        }

        self.inner.get_value(term)
    }
}
```

**Complexity**: `$\mathcal{O}(1)$` expiration check + `$\mathcal{O}(d)$` inner lookup

## Usage Examples

### Example 1: Session Token Expiration

```rust
use liblevenshtein::cache::eviction::Ttl;
use liblevenshtein::prelude::*;
use std::time::Duration;
use std::thread;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sessions = PathMapDictionary::from_terms_with_values([
        ("session_abc", "user_42"),
        ("session_def", "user_99"),
    ]);

    // Sessions expire after 5 seconds
    let ttl = Ttl::new(sessions, Duration::from_secs(5));

    // Access before expiration
    assert_eq!(ttl.get_value("session_abc"), Some("user_42"));

    // Wait for expiration
    thread::sleep(Duration::from_secs(6));

    // Now expired
    assert_eq!(ttl.get_value("session_abc"), None);

    Ok(())
}
```

### Example 2: Cache Invalidation

```rust
use liblevenshtein::cache::eviction::Ttl;
use liblevenshtein::prelude::*;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // API response cache with 1-minute TTL
    let cache = PathMapDictionary::from_terms_with_values([
        ("/api/users", r#"{"users": [...]}"#),
        ("/api/posts", r#"{"posts": [...]}"#),
    ]);

    let ttl_cache = Ttl::new(cache, Duration::from_secs(60));

    // Fresh cache hit
    let response = ttl_cache.get_value("/api/users");
    assert!(response.is_some());

    // After 1 minute: automatic invalidation
    // (In real system, would refetch from API)

    Ok(())
}
```

### Example 3: Temporary Data Storage

```rust
use liblevenshtein::cache::eviction::Ttl;
use liblevenshtein::prelude::*;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One-time password (OTP) codes with 5-minute expiration
    let otps = PathMapDictionary::from_terms_with_values([
        ("user_42_otp", "123456"),
        ("user_99_otp", "789012"),
    ]);

    let ttl_otps = Ttl::new(otps, Duration::from_secs(300));

    // OTP valid within 5 minutes
    assert_eq!(ttl_otps.get_value("user_42_otp"), Some("123456"));

    // After 5 minutes: OTP invalid
    Ok(())
}
```

### Example 4: Composition with LRU

```rust
use liblevenshtein::cache::eviction::{Ttl, Lru};
use liblevenshtein::prelude::*;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dict = PathMapDictionary::from_terms_with_values([
        ("key1", "value1"),
        ("key2", "value2"),
    ]);

    // TTL filters expired, LRU tracks recency among valid entries
    let ttl = Ttl::new(dict, Duration::from_secs(300));
    let lru_ttl = Lru::new(ttl);

    lru_ttl.get_value("key1");
    lru_ttl.get_value("key2");

    // Among non-expired entries, find LRU
    let lru_entry = lru_ttl.find_lru(&["key1", "key2"]);
    println!("LRU (non-expired): {:?}", lru_entry);

    Ok(())
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| get_value | `$\mathcal{O}(1)$` + `$\mathcal{O}(d)$` | Expiration check + inner lookup |
| contains | `$\mathcal{O}(1)$` + `$\mathcal{O}(d)$` | Expiration check + inner contains |

### Space Complexity

**Per-Entry Overhead**: 16 bytes (Instant)

**Expired Entries**: Metadata remains until manually cleaned (memory leak potential)

**Cleanup Strategy** (optional):
```rust
impl<D> Ttl<D> {
    pub fn cleanup_expired(&self) {
        let mut metadata = self.metadata.write().unwrap();
        metadata.retain(|_, meta| meta.inserted_at.elapsed() <= self.duration);
    }
}
```

## Comparison with Other Policies

| Aspect | TTL | LRU | Age |
|--------|-----|-----|-----|
| **Metric** | Expiration time | Last access | Insertion time |
| **Eviction** | Hard cutoff | Soft (LRU scan) | Soft (oldest scan) |
| **Use Case** | Time-sensitive data | Access patterns | FIFO ordering |
| **Metadata** | 16 bytes | 16 bytes | 16 bytes |

### When to Use TTL

✅ **Good For**:
- Session management (hard expiration)
- Cache invalidation (fixed refresh interval)
- Temporary tokens (OTPs, API keys)
- Time-sensitive data (stock quotes, weather)

❌ **Not Ideal For**:
- Recency-based eviction (use LRU)
- Frequency-based eviction (use LFU)
- No natural expiration time

## Design Considerations

### Automatic vs Manual Cleanup

**Automatic Filtering** (current implementation):
- Expired entries filtered on access
- Metadata lingers until explicit cleanup
- Zero-cost for expired entries (just filtered)

**Manual Cleanup** (optional):
```rust
// Periodic cleanup to free metadata memory
ttl.cleanup_expired();
```

### TTL Duration Selection

| Use Case | Typical TTL | Rationale |
|----------|-------------|-----------|
| Session tokens | 15-60 minutes | Balance security vs UX |
| API responses | 1-5 minutes | Data freshness vs load |
| OTP codes | 5-10 minutes | Security requirement |
| Cache entries | 5-60 minutes | Depends on volatility |

## Related Documentation

- **[Layer 8 README](../README.md)** - Caching layer overview
- **[LRU Policy](./lru.md)** - Least Recently Used
- **[MemoryPressure Policy](./memory-pressure.md)** - Memory-aware eviction

## References

1. Source code: `src/cache/eviction/ttl.rs`
2. HTTP Cache-Control: RFC 7234 (TTL in web caching)

## Quick Reference

```rust
use liblevenshtein::cache::eviction::Ttl;
use liblevenshtein::prelude::*;
use std::time::Duration;

// Basic usage
let dict = PathMapDictionary::from_terms_with_values([("key", "value")]);
let ttl = Ttl::new(dict, Duration::from_secs(300));  // 5 minutes
ttl.get_value("key");

// Composition with LRU
let lru_ttl = Lru::new(Ttl::new(dict, Duration::from_secs(300)));
```
