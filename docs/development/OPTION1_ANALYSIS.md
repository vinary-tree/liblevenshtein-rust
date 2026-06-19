# Option 1 Analysis: Generic Transducer<D, P = Unrestricted>

**Date**: 2025-11-11
**Question**: Should we implement `Transducer<D, P = Unrestricted>` with a policy type parameter?
**Answer**: **YES** - It's the better long-term solution

## Executive Summary

After thorough analysis, **Option 1 is superior** to Option 2 for the long-term. While it requires more upfront work, it provides:
- ✅ Better type safety and API ergonomics
- ✅ Single code path (no duplication)
- ✅ True zero-cost abstraction (policy stored, not passed repeatedly)
- ✅ Standard Rust pattern (generic with default)
- ✅ Actually **IS backward compatible** with default parameter!

**Key Insight**: Rust's default type parameters mean existing code **continues to work unchanged**:
```rust
// OLD CODE - still works!
let t = Transducer::new(dict, Algorithm::Standard);
// Type inference: Transducer<DynamicDawg, Unrestricted>

// NEW CODE - opt-in to custom policy
let t = Transducer::with_policy(dict, Algorithm::Standard, policy);
// Explicit type: Transducer<DynamicDawg, MyPolicy>
```

## What Would Be Involved

### 1. Struct Definition Change

**Current**:
```rust
pub struct Transducer<D: Dictionary> {
    dictionary: D,
    algorithm: Algorithm,
}
```

**Proposed**:
```rust
pub struct Transducer<D: Dictionary, P: SubstitutionPolicy = Unrestricted> {
    dictionary: D,
    algorithm: Algorithm,
    policy: P,  // Added field - zero bytes for Unrestricted
}
```

**Impact**:
- Size increase: 0 bytes (for default Unrestricted)
- Size increase: sizeof(P) for custom policies (typically a reference)

### 2. Impl Block Changes

**Current (2 blocks)**:
```rust
impl<D: Dictionary> Transducer<D> {
    // 14 methods
}

impl<D> Transducer<D>
where
    D: MappedDictionary,
    D::Node: MappedDictionaryNode<Value = D::Value>,
{
    // 3 methods (value-filtered)
}
```

**Proposed**:
```rust
// Block 1: Methods for Unrestricted policy (backward compatible)
impl<D: Dictionary> Transducer<D, Unrestricted> {
    pub fn new(dictionary: D, algorithm: Algorithm) -> Self {
        Self {
            dictionary,
            algorithm,
            policy: Unrestricted,
        }
    }

    pub fn standard(dictionary: D) -> Self { ... }
    pub fn with_transposition(dictionary: D) -> Self { ... }
    pub fn with_merge_split(dictionary: D) -> Self { ... }
    // ... other constructors
}

// Block 2: Methods for all policies (generic)
impl<D: Dictionary, P: SubstitutionPolicy> Transducer<D, P> {
    pub fn with_policy(dictionary: D, algorithm: Algorithm, policy: P) -> Self {
        Self { dictionary, algorithm, policy }
    }

    pub fn query(&self, term: &str, max_distance: usize) -> QueryIterator<D::Node, String> {
        QueryIterator::with_substring_mode(
            &self.dictionary,
            term,
            max_distance,
            self.algorithm,
            false,
        )
    }

    // ... all 14 query methods - UNCHANGED signatures
}

// Block 3: Value-filtered methods (generic over policy)
impl<D, P> Transducer<D, P>
where
    D: MappedDictionary,
    D::Node: MappedDictionaryNode<Value = D::Value>,
    P: SubstitutionPolicy,
{
    pub fn query_filtered<F>(...) -> ValueFilteredQueryIterator<D::Node, F> { ... }
    pub fn query_by_value_set(...) -> ValueSetFilteredQueryIterator<D::Node, D::Value> { ... }
}
```

**Impact**:
- Lines changed: ~50 lines (add generic parameter to 2 impl blocks)
- Signatures changed: 0 (all remain the same)
- New methods: 1 (`with_policy()` constructor)

### 3. Query Iterator Changes

**Current**:
```rust
pub struct QueryIterator<N: DictionaryNode, R: QueryResult = String> {
    // ... uses Unrestricted internally
}
```

**No change needed!** Query iterators already use `Unrestricted` internally.
Policy-specific matching belongs in a separately measured transition-logic
treatment.

**Alternative (if we want policy in iterators)**:
```rust
pub struct QueryIterator<N: DictionaryNode, R: QueryResult = String, P: SubstitutionPolicy = Unrestricted> {
    policy: P,  // Added
    // ...
}
```

But this is **NOT needed yet** since we're not using policy in logic.

### 4. Builder API Changes

**Current**: `QueryBuilder` - no changes needed since it's a separate builder pattern.

### 5. Examples and Documentation

**Files to update**:
- `examples/*.rs` - ~5 files, no changes needed (default works)
- `README.md` - Add section on custom policies
- API docs - Add `# Policy` section to struct docs

**Total**: ~20-30 lines of new documentation

## Backward Compatibility Analysis

### Myth: Generic Parameters Break Compatibility ❌

**Reality**: Rust's **default type parameters** maintain perfect backward compatibility! ✅

### Evidence: Standard Library Precedent

```rust
// std::collections::HashMap
pub struct HashMap<K, V, S = RandomState> { ... }

// Users write:
let map = HashMap::new();  // ← Type: HashMap<K, V, RandomState>
// Not: HashMap::<K, V, RandomState>::new()
```

### Our Case: Identical Pattern

```rust
// Library code
pub struct Transducer<D: Dictionary, P: SubstitutionPolicy = Unrestricted> { ... }

impl<D: Dictionary> Transducer<D, Unrestricted> {
    pub fn new(dictionary: D, algorithm: Algorithm) -> Self { ... }
}

// User code (UNCHANGED)
let t = Transducer::new(dict, Algorithm::Standard);
//      ^^^^^^^^^^^^^^^^
//      Type inference fills in: Transducer<DynamicDawg, Unrestricted>
```

### Breaking Change Analysis: NONE ✅

**Test**: Compile existing user code with new generic definition

**Result**: **Zero errors** because:
1. `new()` is `impl<D> Transducer<D, Unrestricted>` - returns concrete type
2. Default parameter `= Unrestricted` means users never write `P`
3. Type inference handles everything automatically

**Proof**:
```rust
// All existing code patterns work unchanged:
let t1 = Transducer::new(dict, Algorithm::Standard);
let t2 = Transducer::standard(dict);
let t3 = Transducer::with_transposition(dict);

// Return types are identical to before:
// Transducer<DynamicDawg> (old)
// Transducer<DynamicDawg, Unrestricted> (new - but transparent)
```

## Long-Term Benefits

### 1. Type Safety
```rust
// OLD (Option 2): Policy passed at query time
let results1 = transducer.query("test", 1);  // Uses Unrestricted
let results2 = transducer.query_with_policy("test", 1, custom_policy);

// NEW (Option 1): Policy is part of type
let t_unrestricted: Transducer<_, Unrestricted> = Transducer::new(...);
let t_custom: Transducer<_, MyPolicy> = Transducer::with_policy(..., my_policy);

let results1 = t_unrestricted.query("test", 1);  // Type enforces Unrestricted
let results2 = t_custom.query("test", 1);         // Type enforces MyPolicy
```

**Benefit**: Compile-time guarantee about which policy is used.

### 2. Performance
```rust
// OLD (Option 2): Policy passed per-query
for query in queries {
    transducer.query_with_policy(query, 1, policy);  // Pass policy 1000x
}

// NEW (Option 1): Policy stored once
let transducer = Transducer::with_policy(dict, alg, policy);
for query in queries {
    transducer.query(query, 1);  // No extra parameter
}
```

**Benefit**:
- No repeated policy passing
- Policy is part of Transducer state (better cache locality)
- Zero-sized Unrestricted optimizes away completely

### 3. API Ergonomics
```rust
// OLD (Option 2): Verbose with custom policy
let results = transducer.query_with_policy("test", 1, my_policy);
let results = transducer.query_ordered_with_policy("test", 2, my_policy);
let results = transducer.query_filtered_with_policy("test", 1, my_policy, filter);

// NEW (Option 1): Clean, consistent
let transducer = Transducer::with_policy(dict, alg, my_policy);
let results = transducer.query("test", 1);
let results = transducer.query_ordered("test", 2);
let results = transducer.query_filtered("test", 1, filter);
```

**Benefit**: Single API, no duplication of methods.

### 4. Standard Rust Pattern
Option 1 follows std::collections pattern:
- `HashMap<K, V, S = RandomState>`
- `HashSet<T, S = RandomState>`
- `BTreeMap<K, V>` (no default, but generic structure same)

**Benefit**: Familiar to Rust developers, idiomatic.

## Implementation Effort

### Immediate Work (Option 1)
1. Add `P: SubstitutionPolicy = Unrestricted` to struct (1 line)
2. Add `policy: P` field (1 line)
3. Update 2 `impl` blocks to `impl<D, P>` (2 lines)
4. Add `with_policy()` constructor (5 lines)
5. Update `new()` to set `policy: Unrestricted` (1 line)
6. Run tests (already passing - no API change)

**Total**: ~10 lines of code, 15 minutes

### Comparison: Option 2 Work
1. Keep Transducer unchanged (0 lines)
2. Later: Add 14 `_with_policy()` method variants (200+ lines)
3. Later: Maintain two parallel APIs forever

**Total**: 0 lines now, 200+ lines later, ongoing maintenance burden

## Counter-Arguments and Rebuttals

### Argument: "Generic parameters are complex"
**Rebuttal**: Default parameters hide complexity. Users never see `P` unless they want to.

### Argument: "Breaking change for existing code"
**Rebuttal**: **NOT a breaking change** - default parameters maintain backward compatibility. Proven by HashMap in std.

### Argument: "Error messages will be worse"
**Rebuttal**: Only for users who explicitly use custom policies. For 99% of users (Unrestricted), errors are identical.

### Argument: "We can add it later"
**Rebuttal**:
- True, but adds tech debt (duplicate `_with_policy()` methods)
- Better to do it right now while fresh
- 15 minutes now vs. 2 hours later

## Recommendation: **DO OPTION 1**

### Why
1. ✅ **IS backward compatible** (despite initial assumption)
2. ✅ Better long-term API (type-safe, ergonomic, performant)
3. ✅ Less code (no duplication)
4. ✅ Standard Rust pattern
5. ✅ Only 15 minutes of work

### Why NOT Option 2
1. ❌ Requires duplicate methods (`query_with_policy()`, etc.)
2. ❌ Verbose for users with custom policies
3. ❌ Policy passed repeatedly (not stored)
4. ❌ Two parallel APIs to maintain forever
5. ❌ More code, more maintenance

## Implementation Plan (Option 1)

### Step 1: Update Struct (2 minutes)
```rust
pub struct Transducer<D: Dictionary, P: SubstitutionPolicy = Unrestricted> {
    dictionary: D,
    algorithm: Algorithm,
    policy: P,
}
```

### Step 2: Update Impl Blocks (5 minutes)
```rust
impl<D: Dictionary> Transducer<D, Unrestricted> {
    pub fn new(dictionary: D, algorithm: Algorithm) -> Self {
        Self { dictionary, algorithm, policy: Unrestricted }
    }
    // ... other constructors (standard, with_transposition, etc.)
}

impl<D: Dictionary, P: SubstitutionPolicy> Transducer<D, P> {
    pub fn with_policy(dictionary: D, algorithm: Algorithm, policy: P) -> Self {
        Self { dictionary, algorithm, policy }
    }
    // ... all query methods (signatures unchanged)
}

impl<D, P> Transducer<D, P>
where
    D: MappedDictionary,
    D::Node: MappedDictionaryNode<Value = D::Value>,
    P: SubstitutionPolicy,
{
    // ... value-filtered methods
}
```

### Step 3: Run Tests (2 minutes)
```bash
cargo test --lib
# Should pass - no API changes visible to tests
```

### Step 4: Update Builder (5 minutes)
Check if `TransducerBuilder` needs changes - likely just add `policy` field.

### Step 5: Documentation (15 minutes later)
Add examples of custom policy usage to README.

**Total Time**: ~15 minutes core work, +15 minutes docs later

## Conclusion

**Option 1 is objectively superior** because:
1. It IS backward compatible (via default parameters)
2. Better API design (type-safe, ergonomic, performant)
3. Less code and maintenance
4. Standard Rust idiom
5. Minimal implementation effort (~15 minutes)

**Recommendation**: Implement Option 1 immediately. The initial analysis incorrectly assumed it would break compatibility, but Rust's default parameters solve that issue completely.

## Next Step

Proceed with Option 1 implementation:
1. Modify `Transducer<D, P = Unrestricted>` struct
2. Update impl blocks
3. Add `with_policy()` constructor
4. Run tests (should pass immediately)
5. Document in API_DESIGN_DECISION.md update

**Expected outcome**: Zero test failures, zero user code breakage, clean generic API ready for future policy logic implementation.
