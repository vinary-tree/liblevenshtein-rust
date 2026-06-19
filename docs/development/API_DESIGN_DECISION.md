# API Design Decision: Substitution Policy Integration

**Date**: 2025-11-11 (Updated: 2025-11-12)
**Status**: ✅ **IMPLEMENTED** (Option 1)
**Scope**: Transducer API for substitution policy support

## Implementation Update (2025-11-12)

After thorough analysis in `OPTION1_ANALYSIS.md`, we **implemented Option 1** with excellent results:
- ✅ Zero breaking changes (491/491 tests pass)
- ✅ Backward compatible via default type parameters
- ✅ Completed in ~15 minutes as predicted
- ✅ Follows Rust stdlib patterns (HashMap)

See `OPTION1_IMPLEMENTATION.md` for complete implementation details.

## Problem Statement

After successfully threading the `SubstitutionPolicy` parameter through all transition functions, we need to decide how to expose this functionality to users through the `Transducer` API.

## Options Considered

### Option 1: Generic Transducer with Default Policy
```rust
pub struct Transducer<D: Dictionary, P: SubstitutionPolicy = Unrestricted> {
    dictionary: D,
    algorithm: Algorithm,
    policy: P,
}

impl<D: Dictionary> Transducer<D, Unrestricted> {
    pub fn new(dictionary: D, algorithm: Algorithm) -> Self { ... }
}

impl<D: Dictionary, P: SubstitutionPolicy> Transducer<D, P> {
    pub fn with_policy(dictionary: D, algorithm: Algorithm, policy: P) -> Self { ... }
    pub fn query(&self, term: &str, max_distance: usize) -> QueryIterator<...> { ... }
}
```

**Pros**:
- Policy is part of the Transducer state
- Single code path for all queries
- Natural API: `transducer.query(...)`

**Cons**:
- **BREAKING CHANGE**: All existing `impl<D: Dictionary> Transducer<D>` blocks become `impl<D: Dictionary, P: SubstitutionPolicy> Transducer<D, P>`
- Type complexity increases for users
- Current code has 2 impl blocks with 10+ methods each
- Would require updating all examples and documentation
- Migration burden on existing users

### Option 2: Policy-Aware Query Methods (CHOSEN ✅)
```rust
pub struct Transducer<D: Dictionary> {
    dictionary: D,
    algorithm: Algorithm,
}

impl<D: Dictionary> Transducer<D> {
    // Existing methods unchanged - use Unrestricted internally
    pub fn query(&self, term: &str, max_distance: usize) -> QueryIterator<...> { ... }

    // NEW: Policy-aware methods for future use
    // (Deferred to when we actually implement policy-based matching)
    // pub fn query_with_policy<P: SubstitutionPolicy>(..., policy: P) -> ... { ... }
}
```

**Pros**:
- ✅ **ZERO BREAKING CHANGES**: Existing API unchanged
- ✅ Simple implementation: internally uses `Unrestricted`
- ✅ Future-proof: can add `_with_policy()` methods later
- ✅ No migration burden on users
- ✅ Aligns with Rust API guidelines (convenience > generics)

**Cons**:
- Policy must be passed per-query (not stored in Transducer)
- Slightly more verbose for users with custom policies

### Option 3: Builder Pattern
```rust
pub struct Transducer<D: Dictionary> { ... }

pub struct PolicyTransducer<D: Dictionary, P: SubstitutionPolicy> {
    transducer: Transducer<D>,
    policy: P,
}

impl<D: Dictionary> Transducer<D> {
    pub fn with_policy<P: SubstitutionPolicy>(self, policy: P) -> PolicyTransducer<D, P> { ... }
}
```

**Pros**:
- No breaking changes
- Type-level distinction between unrestricted and policy-based

**Cons**:
- Added complexity with wrapper type
- Two parallel APIs to maintain

## Decision: Option 1 (Generic Transducer with Default Policy) ✅

**Chosen**: 2025-11-12

**Rationale**:
1. **IS Backward Compatible**: Default type parameters mean existing code unchanged
2. **Better Long-Term Design**: Type-safe, no API duplication, follows Rust patterns
3. **Zero-Cost Abstraction**: `Unrestricted` is a ZST (0 bytes)
4. **Minimal Implementation**: ~15 minutes as predicted
5. **Standard Library Pattern**: Same as `HashMap<K, V, S = RandomState>`

**Implementation**:
```rust
pub struct Transducer<D: Dictionary, P: SubstitutionPolicy = Unrestricted> {
    dictionary: D,
    algorithm: Algorithm,
    policy: P,
}

// Backward-compatible constructors
impl<D: Dictionary> Transducer<D, Unrestricted> {
    pub fn new(dictionary: D, algorithm: Algorithm) -> Self { ... }
    pub fn standard(dictionary: D) -> Self { ... }
    pub fn with_transposition(dictionary: D) -> Self { ... }
    pub fn with_merge_split(dictionary: D) -> Self { ... }
}

// Generic methods (work with any policy)
impl<D: Dictionary, P: SubstitutionPolicy> Transducer<D, P> {
    pub fn with_policy(dictionary: D, algorithm: Algorithm, policy: P) -> Self { ... }
    pub fn query(&self, term: &str, max_distance: usize) -> QueryIterator<...> { ... }
    // ... all query methods
}
```

**Results**:
- ✅ Compilation: Success (0 errors)
- ✅ Tests: 491/491 passing
- ✅ Breaking Changes: 0
- ✅ Implementation Time: ~15 minutes

## Option 2 (Policy-Aware Query Methods) - NOT CHOSEN

**Why NOT Chosen**:
1. Requires duplicate `_with_policy()` method variants (14+ methods)
2. Verbose for users with custom policies
3. Policy passed per-query (not stored in Transducer)
4. More code to maintain
5. Less type-safe (policy enforcement at runtime, not compile-time)

**Note**: This was initially chosen in the 2025-11-11 version of this document, but after analyzing Option 1's feasibility (see `OPTION1_ANALYSIS.md`), we discovered it IS backward compatible due to default type parameters.

## Technical Justification

### Why Policy Parameter Exists Now
The policy parameter is threaded through transition functions **in preparation** for future policy-based matching. Currently:
- The parameter exists but is **not used** in logic
- All code paths use `Unrestricted`
- `characteristic_vector` still does exact matching only
- Policy will be used when we implement **restricted substitution transitions**

### Why NOT Expose It Yet
1. **No functional change**: Policy doesn't affect behavior yet
2. **Incomplete implementation**: Need to modify transition logic to actually check policy
3. **User confusion**: Exposing an API that doesn't work would confuse users
4. **Scientific rigor**: Complete and test the feature before exposing it

## Future API (When Policy Logic is Implemented)

When we implement actual policy-based matching in transition logic:

```rust
impl<D: Dictionary> Transducer<D> {
    /// Query with custom substitution policy
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::prelude::*;
    ///
    /// let dict = DynamicDawg::from_terms(vec!["cat", "bat"]);
    /// // Custom policy: allow 'c' ↔ 'k' substitution
    /// let mut policy_set = SubstitutionSet::new();
    /// policy_set.allow('c', 'k');
    /// let policy = Restricted::new(&policy_set);
    /// let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);
    ///
    /// // "kat" will match "cat" with distance 1 (c↔k allowed)
    /// // but "bat" won't match (b↔k not allowed)
    /// let results: Vec<_> = transducer.query("kat", 1).collect();
    /// ```
    pub fn with_policy(dictionary: D, algorithm: Algorithm, policy: P) -> Self {
        Self {
            dictionary,
            algorithm,
            policy,
        }
    }

    pub fn query(&self, term: &str, max_distance: usize) -> QueryIterator<D::Node, String, P> {
        QueryIterator::with_policy(
            self.dictionary.root(),
            term.to_owned(),
            max_distance,
            self.algorithm,
            self.policy,
        )
    }
}
```

## Migration Path (If Needed Later)

If we decide users need Transducer-level policy storage:

**Version 0.7.0** (Current):
- `Transducer<D>` - no policy support

**Version 0.8.0** (Hypothetical):
- Add `query_with_policy()` methods
- Deprecate nothing (additive change)

**Version 0.9.0** (Hypothetical):
- Add `Transducer::with_policy()` builder
- Keep `Transducer<D>` for backward compatibility

**Version 1.0.0** (Hypothetical):
- Evaluate usage data
- Consider `Transducer<D, P = Unrestricted>` if widely used
- Breaking change acceptable in 1.0

## Conclusion

**Final Decision**: **Implement Option 1** - Generic `Transducer<D, P = Unrestricted>` ✅

**Reasoning**:
- ✅ IS backward compatible (default type parameters)
- ✅ Better long-term API design
- ✅ Follows Rust stdlib patterns
- ✅ Zero-cost abstraction ready
- ✅ Type-safe policy enforcement
- ✅ No API duplication

**Implementation Status**: ✅ **COMPLETE** (2025-11-12)
- Compilation: Success (0 errors)
- Tests: 491/491 passing
- Time: ~15 minutes
- Breaking Changes: 0

**Next Steps**:
1. ✅ Implement Option 1 (2025-11-12)
2. ✅ Document implementation (OPTION1_IMPLEMENTATION.md)
3. ➡️ Move to Phase 4: Add substitution support to eager automaton
4. ➡️ Implement actual policy checks in transition logic (future)
5. ➡️ Phase 7: Verify zero-cost abstraction

**Status**: Phase 3 complete. Infrastructure in place and API exposed for future policy-based matching.

## References

- **Rust API Guidelines**: [C-INTERMEDIATE](https://rust-lang.github.io/api-guidelines/flexibility.html#c-intermediate) - Convenience over generics
- **Semantic Versioning**: [SemVer 2.0.0](https://semver.org/) - Pre-1.0 can break, but minimize churn
- **Plan**: `docs/development/RESTRICTED_SUBSTITUTIONS_PLAN.md` - 30-day implementation timeline
