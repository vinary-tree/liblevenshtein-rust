# FuzzyMultiMap Performance Analysis

## Baseline Benchmarks

| Benchmark | Time (µs) | Notes |
|-----------|-----------|-------|
| `fuzzy_multimap_query` | 38.55 | Standard query with 5 values per set |
| `fuzzy_multimap_high_aggregation` | 394.96 | High aggregation (10 values/set, distance=3) |
| `fuzzy_multimap_vec_concat` | 38.13 | Vec concatenation instead of HashSet |
| `transducer_query_baseline` | 46.69 | Baseline without aggregation |
| `dict_get_value_hashset` | 1.20 | Dictionary value access (10 lookups) |
| `hashset_aggregation` | 11.53 | Pure HashSet aggregation (50 sets) |
| `fuzzy_multimap_complete` | 211.45 | Complete workflow (5 queries) |

## Identified Bottlenecks

### 1. **Double Collection in `query()` method** (CRITICAL)

**Location**: `src/cache/multimap.rs:258-279`

**Problem**:
```rust
pub fn query(&self, query_term: &str, max_distance: usize) -> Option<C> {
    // Step 1: Collect candidates into Vec
    let candidates: Vec<_> = self.transducer
        .query(query_term, max_distance)
        .collect();  // ← FIRST ALLOCATION

    if candidates.is_empty() {
        return None;
    }

    // Step 2: Collect values into ANOTHER Vec
    let values: Vec<C> = candidates
        .into_iter()
        .filter_map(|term| self.dictionary.get_value(&term))
        .collect();  // ← SECOND ALLOCATION

    if values.is_empty() {
        return None;
    }

    // Step 3: Aggregate
    Some(C::aggregate(values.into_iter()))
}
```

**Impact**:
- Two separate `Vec` allocations
- First Vec holds `String` candidates (expensive clones from transducer)
- Second Vec holds collection values (cloned again)
- Total: 3 allocations minimum (candidates, values, aggregated result)

**Evidence**:
- `fuzzy_multimap_query` (38.55 µs) vs `trans ducer_query_baseline` (46.69 µs)
- FuzzyMultiMap is actually FASTER than baseline, suggesting the double-collect isn't the main issue
- BUT: `fuzzy_multimap_high_aggregation` (394.96 µs) shows quadratic behavior with more matches

### 2. **Excessive String Cloning**

**Problem**: Trans ducer returns owned `String` values which must be cloned during iteration.

**Evidence**:
- Transducer yields owned strings
- Each candidate requires dictionary lookup by string reference
- Could use string slices or references to reduce allocations

### 3. **HashSet Aggregation Overhead**

**Location**: `src/cache/multimap.rs:83-96`

**Problem**:
```rust
fn aggregate<I>(values: I) -> Self
where
    I: Iterator<Item = Self>,
{
    values.fold(HashSet::new(), |mut acc, set| {
        acc.extend(set);  // ← Repeated rehashing for each set
        acc
    })
}
```

**Impact**:
- `hashset_aggregation`: 11.53 µs for 50 sets
- Each `extend()` may trigger rehashing
- No pre-allocation based on estimated size

**Potential Optimization**:
- Pre-allocate with estimated capacity
- Use `reserve()` to reduce rehashing

### 4. **Vec Concatenation is Equally Expensive**

**Evidence**: `fuzzy_multimap_vec_concat` (38.13 µs) ≈ `fuzzy_multimap_query` (38.55 µs)
- Both HashSet and Vec aggregation have similar cost
- The bottleneck is NOT the aggregation type, but the collection process

## Optimization Opportunities (Ranked by Impact)

### 🔴 HIGH IMPACT

#### Optimization 1: **Single-Pass Collection with Iterator Fusion**

Replace double-collect with single iterator chain:

```rust
pub fn query(&self, query_term: &str, max_distance: usize) -> Option<C> {
    let mut values = self.transducer
        .query(query_term, max_distance)
        .filter_map(|term| self.dictionary.get_value(&term))
        .peekable();

    if values.peek().is_none() {
        return None;
    }

    Some(C::aggregate(values))
}
```

**Expected Impact**: 15-25% reduction in allocation overhead

#### Optimization 2: **Pre-allocated Aggregation with Capacity Hints**

Add capacity hints to `CollectionAggregate` trait:

```rust
pub trait CollectionAggregate: Sized {
    fn aggregate<I>(values: I) -> Self
    where
        I: Iterator<Item = Self>;

    // NEW: Aggregate with size hint
    fn aggregate_with_capacity<I>(values: I, capacity_hint: usize) -> Self
    where
        I: Iterator<Item = Self>,
    {
        Self::aggregate(values)  // Default implementation
    }
}

// HashSet implementation
impl<T: Eq + Hash + Clone> CollectionAggregate for HashSet<T> {
    fn aggregate_with_capacity<I>(mut values: I, capacity_hint: usize) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut acc = HashSet::with_capacity(capacity_hint);
        for set in values {
            acc.extend(set);
        }
        acc
    }
}
```

**Expected Impact**: 10-20% reduction in aggregation time (especially for high aggregation)

### 🟡 MEDIUM IMPACT

#### Optimization 3: **SmallVec for Candidate Collection**

Use `SmallVec` to avoid heap allocation for small result sets:

```rust
use smallvec::SmallVec;

// For queries that typically return <10 results
let candidates: SmallVec<[String; 8]> = self.transducer
    .query(query_term, max_distance)
    .collect();
```

**Expected Impact**: 5-10% improvement for small result sets (<10 matches)

#### Optimization 4: **Lazy Aggregation**

Return an iterator-based view instead of eagerly collecting:

```rust
pub fn query_iter<'a>(&'a self, query_term: &str, max_distance: usize)
    -> impl Iterator<Item = &'a C> + 'a
{
    self.transducer
        .query(query_term, max_distance)
        .filter_map(move |term| self.dictionary.get_value(&term))
}
```

**Expected Impact**: Eliminates allocation for streaming use cases

### 🟢 LOW IMPACT

#### Optimization 5: **String Interning**

Cache frequently-queried terms to avoid repeated string allocations.

**Expected Impact**: <5% for typical workloads

## Recommended Implementation Order

1. **Single-Pass Collection** (Optimization 1) - Low risk, high impact
2. **Capacity Hints** (Optimization 2) - Medium complexity, medium-high impact
3. **SmallVec** (Optimization 3) - Low complexity, targeted impact
4. **Lazy Aggregation** (Optimization 4) - API addition, no breaking change

## Expected Performance Gains

| Scenario | Current | Optimized | Improvement |
|----------|---------|-----------|-------------|
| Small queries (10-20 matches) | 38.55 µs | ~29 µs | **25%** |
| High aggregation (100+ matches) | 394.96 µs | ~300 µs | **24%** |
| Vec concatenation | 38.13 µs | ~30 µs | **21%** |

## Action Plan

1. ✅ Profile and identify bottlenecks
2. ⏳ Implement Optimization 1 (Single-Pass Collection)
3. ⏳ Implement Optimization 2 (Capacity Hints)
4. ⏳ Benchmark and compare
5. ⏳ Implement remaining optimizations if needed
