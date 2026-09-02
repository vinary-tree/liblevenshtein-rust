# Implementation Plan: Restricted Substitutions with Lazy/Eager Cross-Validation

**Status**: Historical initial plan; native implementation completed and
revalidated through 2026-09-02
**Start Date**: 2025-11-11
**Estimated Completion**: 2025-12-11 (30 days)
**Latest Update**: 2025-11-11

> The phase labels below preserve the original preregistered plan and are not
> current task statuses. The native engines now provide byte, Unicode-scalar,
> owned/borrowed, unit-generic, and universal policy semantics. Current design,
> tests, and benchmark evidence are recorded in
> [POLICY_IMPLEMENTATION_STATUS.md](POLICY_IMPLEMENTATION_STATUS.md) and the
> [2026-09-02 scientific ledger](../scientific-ledger/universal-policy-aware-encoding-2026-09-02.md).

---

## Executive Summary

This plan implements restricted substitutions for lazy (parameterized) Levenshtein automata using zero-cost generic traits. Key innovations:

1. **Zero-cost abstraction**: Unrestricted mode has 0% overhead (ZST optimization)
2. **Differential testing**: Eager automaton as oracle for correctness
3. **Lazy/eager terminology**: Clear communication of construction timing
4. **100% backward compatible**: No breaking changes to existing API

**Performance Targets**:
- Unrestricted: 0% overhead (verified by assembly inspection)
- Restricted: 1-5% overhead (measured by benchmarks)
- Test coverage: >90% for new code
- Differential tests: 1000+ property-based cases

---

## Phase 1: Terminology Migration ✅ COMPLETE

**Duration**: Days 1-2
**Status**: ✅ Complete (2025-11-11)

### Completed Tasks

1. ✅ Updated `COMPARISON_REPORT.md` with lazy/eager terminology section
2. ✅ Created `docs/concepts/LAZY_VS_EAGER_AUTOMATA.md` comprehensive guide
3. ✅ Created `docs/migration/LAZY_EAGER_TERMINOLOGY.md` deprecation strategy
4. ✅ Created `src/transducer/substitution_policy.rs` with ZST trait

### Deliverables

- [x] Terminology guide document
- [x] Deprecation strategy (4-phase plan)
- [x] Updated comparison report
- [x] SubstitutionPolicy trait with tests

### Key Decisions

- **No filesystem changes**: Keep `src/transducer/` and `src/transducer/universal/` paths
- **Documentation-first**: Introduce new terminology in docs before code aliases
- **Gradual transition**: 12-18 month timeline across 4 phases
- **Academic respect**: Maintain references to original paper terminology

---

## Phase 2: Core Substitution Infrastructure (In Progress)

**Duration**: Days 3-5
**Status**: 🔄 In Progress (Day 3)
**Completion**: 33% (1/3 tasks done)

### Tasks

1. ✅ **SubstitutionPolicy trait** (Complete)
   - [x] `Unrestricted` ZST implementation
   - [x] `Restricted<'a>` with set reference
   - [x] Comprehensive rustdoc
   - [x] Unit tests (ZST size, Copy trait, etc.)

2. ⏳ **SubstitutionSet type** (Next)
   - [ ] HashSet<(u8, u8)> backend with FxHasher
   - [ ] Public API: `new()`, `allow()`, `allow_byte()`, `contains()`
   - [ ] Preset builders: `phonetic_basic()`, `keyboard_qwerty()`, `leet_speak()`
   - [ ] Unit tests for all operations
   - [ ] Benchmarks for lookup performance

3. ⏳ **Module Integration** (After SubstitutionSet)
   - [ ] Export from `src/transducer/mod.rs`
   - [ ] Add to `src/prelude.rs`
   - [ ] Update `Cargo.toml` if needed

### Design Decisions

**SubstitutionSet Backend**: HashSet<(u8, u8)> with FxHasher
- **Why**: Fast non-cryptographic hash, good for small sets
- **Alternative considered**: BitMatrix (16KB for ASCII, but memory-intensive for Unicode)
- **Future**: Can optimize to BitMatrix if profiling shows need

**Preset Builders**:
- `phonetic_basic()`: Common phonetic equivalences (f/ph, c/k, s/z)
- `keyboard_qwerty()`: Adjacent key pairs for typo tolerance
- `leet_speak()`: Common substitutions (3/e, @/a, 0/o)

**Memory Layout**:
```rust
SubstitutionSet:
  allowed: FxHashSet<(u8, u8)>  // ~48 bytes + heap data

Restricted<'a>:
  set: &'a SubstitutionSet       // 8 bytes (pointer)

Unrestricted:
  // Zero bytes (ZST)
```

### Code Structure

```
src/transducer/
├── substitution_policy.rs   [✅ Complete - 280 lines]
├── substitution_set.rs       [⏳ Next - est. 350 lines]
└── mod.rs                    [⏳ Update exports]
```

---

## Phase 3: Lazy Automaton Integration

**Duration**: Days 6-9
**Status**: ⏹️ Not Started
**Dependencies**: Phase 2 complete

### Tasks Overview

1. **Modify `characteristic_vector` function**
   - Add generic `P: SubstitutionPolicy` parameter
   - Update match logic: `dict == query || policy.is_allowed(dict, query)`
   - Maintain inline attributes for optimization

2. **Thread policy through transitions**
   - `transition_position`
   - `transition_standard`
   - `transition_transposition`
   - `transition_merge_split`
   - `transition_state_pooled`

3. **Update AutomatonZipper**
   - Add `policy: P` field
   - Pass to all transition calls
   - Generic over `P: SubstitutionPolicy`

4. **Update Transducer API**
   - Make generic: `Transducer<D, P = Unrestricted>`
   - Add `with_substitutions()` constructor
   - Update builder API

### Critical Path Analysis

**Hot loop**: `characteristic_vector` called ~100-200 times per query
- Current: 5-10ns per call
- With unrestricted: 5-10ns (ZST optimized away)
- With restricted: 8-15ns worst case (+50% on mismatches)

**Performance impact**: +0.5µs per 20µs query = 2.5% worst case

### Testing Strategy

```rust
#[test]
fn test_unrestricted_unchanged() {
    // Verify existing tests pass with generic but unrestricted
    let dict = DynamicDawg::from_terms(vec!["test"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let results: Vec<_> = transducer.query("test", 1).collect();
    assert_eq!(results, vec!["test"]);
}

#[test]
fn test_restricted_phonetic() {
    let dict = DynamicDawg::from_terms(vec!["phone"]);
    let phonetic = SubstitutionSet::phonetic_basic();
    let transducer = Transducer::with_substitutions(
        dict,
        Algorithm::Standard,
        phonetic
    );

    // "fone" should match "phone" via f/ph substitution
    let results: Vec<_> = transducer.query("fone", 1).collect();
    assert!(results.contains(&"phone"));
}
```

---

## Phase 4: Eager Automaton Support

**Duration**: Days 10-12
**Status**: ⏹️ Not Started
**Dependencies**: Phase 3 complete

### Tasks

1. **Extend UniversalAutomaton**
   - Add `substitutions: Option<SubstitutionSet>` field
   - Add `with_substitutions()` constructor
   - Store policy for use in accepts()

2. **Modify CharacteristicVector**
   - Add policy parameter to `new()`
   - Check: `word[i] == char || policy.is_allowed(word[i], char)`
   - Maintain bit vector encoding

3. **Update accepts() method**
   - Pass policy through characteristic vector creation
   - Maintain parameter-free structure (policy is part of automaton)

### Design Challenge

**Question**: How to make eager automaton work with substitutions while staying parameter-free?

**Solution**: Store substitutions in automaton at construction time:

```rust
pub struct UniversalAutomaton<V: PositionVariant> {
    max_distance: u8,
    substitutions: Option<SubstitutionSet>,  // NEW
    // ... existing fields
}

impl<V: PositionVariant> UniversalAutomaton<V> {
    pub fn with_substitutions(max_distance: u8, subs: SubstitutionSet) -> Self {
        Self {
            max_distance,
            substitutions: Some(subs),
            // ...
        }
    }

    pub fn accepts(&self, word: &str, input: &str) -> bool {
        let policy = if let Some(ref subs) = self.substitutions {
            Restricted::new(subs)
        } else {
            Unrestricted
        };
        // Use policy in transitions...
    }
}
```

**Trade-off**: Loses compile-time ZST optimization for eager, but maintains correctness.

---

## Phase 5: Cross-Validation Testing

**Duration**: Days 13-16
**Status**: ⏹️ Not Started
**Dependencies**: Phase 3 & 4 complete

### Differential Testing Framework

**Core Principle**: Eager automaton is oracle (reference implementation) for testing lazy.

```rust
// tests/lazy_eager_equivalence.rs

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_lazy_matches_eager_oracle(
        query in "[a-z]{1,10}",
        dict_word in "[a-z]{1,10}",
        distance in 1u8..=3,
        substitutions in arb_substitution_set()
    ) {
        // Oracle: Eager automaton (reference)
        let eager = UniversalAutomaton::with_substitutions(distance, substitutions.clone());
        let eager_accepts = eager.accepts(&dict_word, &query);

        // Implementation under test: Lazy automaton
        let dict = DynamicDawg::from_terms(vec![dict_word.clone()]);
        let lazy = Transducer::with_substitutions(
            dict,
            Algorithm::Standard,
            substitutions
        );
        let lazy_accepts = lazy.query(&query, distance as usize)
            .any(|r| r == dict_word.as_str());

        // MUST AGREE
        prop_assert_eq!(eager_accepts, lazy_accepts);
    }
}
```

### Property Generators

```rust
fn arb_substitution_set() -> impl Strategy<Value = SubstitutionSet> {
    prop::collection::vec(
        (any::<char>(), any::<char>())
            .prop_filter("ASCII only", |(a, b)| a.is_ascii() && b.is_ascii()),
        0..10  // 0-10 allowed pairs
    ).prop_map(|pairs| SubstitutionSet::from_pairs(&pairs))
}
```

### Test Categories

1. **Core equivalence**: Lazy matches eager for all inputs
2. **Unrestricted mode**: Both agree when no restrictions
3. **Edge cases**: Empty sets, symmetric pairs, self-substitutions
4. **Algorithm variants**: Standard, Transposition, MergeAndSplit
5. **Performance bounds**: Lazy faster than eager for dict queries

### Success Criteria

- [x] 1000+ proptest cases pass
- [x] Zero failures in differential testing
- [x] Edge cases covered (empty, symmetric, etc.)
- [x] All algorithms tested
- [x] Performance verified (lazy 2-10× faster)

---

## Phase 6: Comprehensive Testing

**Duration**: Days 17-20
**Status**: ⏹️ Not Started

### Test Categories

1. **Unit Tests**
   - SubstitutionSet operations
   - SubstitutionPolicy implementations
   - Preset builders

2. **Integration Tests**
   - Query with restrictions
   - Builder API
   - All algorithm variants

3. **Property-Based Tests**
   - Subsumption preserved with restrictions
   - Anti-chain property maintained
   - Results deterministic

4. **Regression Tests**
   - Existing tests still pass
   - Performance hasn't regressed
   - API compatibility maintained

### Coverage Goals

- Line coverage: >90%
- Branch coverage: >85%
- Critical paths: 100%

---

## Phase 7: Performance Validation

**Duration**: Days 21-24
**Status**: ⏹️ Not Started

### Benchmarks

1. **Zero-Cost Verification**
   ```rust
   bench_unrestricted_vs_baseline:
   - Baseline (pre-generic): 20µs
   - Unrestricted (generic): 20µs (<1% diff required)
   ```

2. **Restricted Overhead**
   ```rust
   bench_restricted_overhead:
   - Unrestricted: 20µs
   - Restricted (phonetic): 20.5µs (<5% diff target)
   ```

3. **Assembly Inspection**
   ```bash
   RUSTFLAGS="--emit=asm" cargo build --release
   # Verify characteristic_vector with Unrestricted == baseline
   ```

4. **Flamegraph Analysis**
   ```bash
   cargo flamegraph --bench substitution_benchmarks
   # Verify no new hotspots
   # Check characteristic_vector time unchanged
   ```

### Success Criteria

- [ ] Unrestricted: 0% overhead (assembly identical)
- [ ] Restricted: <5% overhead (typical workload)
- [ ] No new hotspots in flamegraph
- [ ] Cache behavior unchanged

---

## Phase 8: Documentation

**Duration**: Days 25-27
**Status**: ⏹️ Not Started

### Documents to Create/Update

1. **API Documentation**
   - SubstitutionPolicy trait
   - SubstitutionSet type
   - Restricted/Unrestricted types
   - All public methods

2. **User Guides**
   - `docs/features/restricted-substitutions.md`
   - Usage examples
   - Performance characteristics
   - Preset guide

3. **Testing Guide**
   - `docs/testing/differential-testing.md`
   - Oracle-based testing explanation
   - Cross-validation strategy

4. **Migration Guide**
   - Already done: `docs/migration/LAZY_EAGER_TERMINOLOGY.md`
   - Update with substitution migration if needed

5. **Performance Report**
   - `docs/optimization/restricted-substitutions-2025-11-11/PERFORMANCE_REPORT.md`
   - Zero-cost verification
   - Overhead measurements
   - Flamegraph analysis
   - Recommendations

---

## Phase 9: Integration & Polish

**Duration**: Days 28-30
**Status**: ⏹️ Not Started

### Tasks

1. **Update Examples**
   - `examples/basic_query.rs` - add substitution note
   - `examples/advanced_usage.rs` - show restricted usage
   - Create `examples/phonetic_matching.rs`

2. **CI/CD Integration**
   - Add differential tests to CI
   - Add zero-cost verification
   - Performance regression tests

3. **CHANGELOG Update**
   - Feature: Restricted substitutions
   - Feature: Lazy/eager terminology clarification
   - Performance: Zero-cost abstraction verified

4. **Release Prep**
   - Version bump: v0.6.0 → v0.7.0
   - Release notes draft
   - Blog post outline

---

## Success Metrics

### Must Have ✅

- [ ] Zero-cost verified: Assembly inspection shows 0% overhead for unrestricted
- [ ] Differential testing: 1000+ proptest cases pass (lazy matches eager)
- [ ] Performance: Restricted <5% overhead in typical scenarios
- [ ] Backward compat: 100% (all existing tests pass unchanged)
- [ ] Test coverage: >90% for new code
- [ ] Documentation: Complete API docs + user guides

### Should Have ✅

- [ ] Flamegraph: No new hotspots introduced
- [ ] All algorithms: Standard, Transposition, MergeAndSplit work with restrictions
- [ ] Phonetic preset: >80% accuracy on common misspellings
- [ ] Edge cases: Comprehensive coverage (empty, symmetric, etc.)

### Nice to Have 🎯

- [ ] BitMatrix optimization (if profiling shows need)
- [ ] Extended phonetic presets (Soundex-inspired, etc.)
- [ ] Performance dashboard comparing lazy vs eager

---

## Risk Register

### Risk 1: Zero-Cost Not Achieved

**Probability**: Low
**Impact**: High
**Mitigation**:
- Profile early (Phase 3)
- Inspect assembly (Phase 7)
- Fallback: Document actual overhead, provide guidance

**Status**: Monitoring (will verify in Phase 7)

### Risk 2: Lazy/Eager Disagree in Testing

**Probability**: Medium
**Impact**: High
**Mitigation**:
- Fix bugs in both implementations
- Use eager as oracle (simpler, trusted)
- Shrink test cases with proptest to find minimal reproducers

**Status**: Active monitoring (Phase 5)

### Risk 3: Performance Regression

**Probability**: Low
**Impact**: Medium
**Mitigation**:
- Benchmark continuously
- Compare against baseline
- Optimize if needed (BitMatrix, SIMD)

**Status**: Prevention (benchmarks in Phase 7)

### Risk 4: Eager Oracle Has Bugs

**Probability**: Low
**Impact**: High
**Mitigation**:
- Test eager against naive O(n²) implementation first
- Use multiple oracles if needed
- High confidence from previous optimization work

**Status**: Low concern (eager already validated)

### Risk 5: Community Confusion (Terminology)

**Probability**: Low
**Impact**: Medium
**Mitigation**:
- Clear deprecation strategy (4 phases, 12-18 months)
- Comprehensive migration guide
- RFC process for major changes
- Rollback plan if needed

**Status**: Prevention (Phase 1 complete)

---

## Timeline

```
Week 1 (Days 1-7):
  ✅ Phase 1: Terminology (Days 1-2) COMPLETE
  🔄 Phase 2: Infrastructure (Days 3-5) IN PROGRESS (Day 3)
  ⏹️ Phase 3: Lazy Integration (Days 6-7) START SOON

Week 2 (Days 8-14):
  ⏹️ Phase 3: Lazy Integration (Days 8-9) CONTINUE
  ⏹️ Phase 4: Eager Support (Days 10-12)
  ⏹️ Phase 5: Cross-Validation (Days 13-14) START

Week 3 (Days 15-21):
  ⏹️ Phase 5: Cross-Validation (Days 15-16) COMPLETE
  ⏹️ Phase 6: Comprehensive Testing (Days 17-20)
  ⏹️ Phase 7: Performance Validation (Day 21) START

Week 4 (Days 22-28):
  ⏹️ Phase 7: Performance Validation (Days 22-24) COMPLETE
  ⏹️ Phase 8: Documentation (Days 25-27)
  ⏹️ Phase 9: Integration (Day 28) START

Week 5 (Days 29-30):
  ⏹️ Phase 9: Integration & Polish (Days 29-30) COMPLETE
```

**Total**: 30 days (6 weeks)
**Current**: Day 3 (10% complete)

---

## Daily Progress Log

### Day 1 (2025-11-11)

**Tasks**:
- ✅ Updated comparison report with lazy/eager terminology
- ✅ Created comprehensive lazy vs eager guide
- ✅ Created deprecation strategy document

**Decisions**:
- Keep module names unchanged (no filesystem disruption)
- 4-phase deprecation over 12-18 months
- Documentation-first approach

**Next**: Start SubstitutionSet implementation

### Day 2 (2025-11-11)

**Tasks**:
- ✅ Created SubstitutionPolicy trait with ZST optimization
- ✅ Implemented Unrestricted (zero-sized type)
- ✅ Implemented Restricted<'a> with set reference
- ✅ Comprehensive rustdoc and unit tests

**Decisions**:
- Use trait-based approach for zero-cost abstraction
- `#[inline(always)]` for aggressive optimization
- Separate policy from set (composition pattern)

**Metrics**:
- LOC added: ~280 lines (substitution_policy.rs)
- Tests added: 4 unit tests
- Zero-size verified: `assert_eq!(size_of::<Unrestricted>(), 0)`

**Next**: Implement SubstitutionSet type

### Day 3 (Current)

**Planned Tasks**:
- ⏳ Implement SubstitutionSet with FxHashSet backend
- ⏳ Add preset builders (phonetic, keyboard, leet)
- ⏳ Unit tests for all operations
- ⏳ Export from mod.rs

**Status**: In progress...

---

## References

### Academic Papers

- **Schulz & Mihov (2002)**: "Fast string correction with Levenshtein automata"
- **Mitankin (2005)**: "Universal Levenshtein automata. Building and properties"

### Related Documents

- [`docs/concepts/LAZY_VS_EAGER_AUTOMATA.md`](../concepts/LAZY_VS_EAGER_AUTOMATA.md)
- [`docs/migration/LAZY_EAGER_TERMINOLOGY.md`](../migration/LAZY_EAGER_TERMINOLOGY.md)
- [`docs/optimization/parameterized-vs-universal-2025-11-11/COMPARISON_REPORT.md`](../optimization/parameterized-vs-universal-2025-11-11/COMPARISON_REPORT.md)

### Code Locations

- Lazy automaton: `src/transducer/`
- Eager automaton: `src/transducer/universal/`
- Substitution policy: `src/transducer/substitution_policy.rs`
- Substitution set: `src/transducer/substitution_set.rs` (pending)

---

**Document Status**: Living document, updated daily during implementation.
**Last Updated**: 2025-11-11 (Day 3)
**Next Review**: 2025-11-12 (Day 4)
