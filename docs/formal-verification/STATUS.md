# Formal Verification Status Report

**Last Updated**: 2026-05-27 (weighted articulatory distance + value-yielding query verification)
**Session**: Phase 3 - I-type complete, M-type next; G4/G9 verification added (see 2026-05-27 update)
**Token Usage**: 84k/200k (42%)

---

## Executive Summary

**Completed**: Phases 1 & 2 (Foundation + Invariants), Phase 3 I-Type Operations ✅
**In Progress**: Phase 3 M-Type Operations
**Achievements**: 26 theorems proven across 4 files, 0 bugs found, 3 simplifications identified

---

## 2026-05-27 Update — Weighted Articulatory Distance (G4) + Value-Yielding Query (G9)

- **Articulatory distance is now proven for arbitrary non-negative per-dimension weights.**
  `docs/verification/articulatory/theories/FeatureDistance.v` was generalized to a `FeatureWeights`
  record (mirroring the Rust `FeatureDistanceWeights`): symmetry and identity hold for **all** weights;
  non-negativity, bound-by-weight-sum, and **per-dimension monotonicity** hold for non-negative
  weights; the standard-weight `0 <= d <= 1` bound is recovered as a corollary. A faithful
  7-dimension model (vowel path + `.min(1.0)` cap via `Qmin`) is added in `FeatureDistanceWeighted.v`.
  Both files are **admission-free** and build under the `light` profile.
- **Value-yielding query (`Transducer::query_values`)** is modeled in
  `docs/verification/tla/ValueYieldingQuery.tla` and TLC-checked (`ValueCorrectness`, `Soundness`,
  `NoValuelessYielded`, `DedupInv`, `CompletenessInv`, termination); evidence in
  `docs/verification/tla/states/tlc-results-2026-05-27.txt`. Cross-validated by Rust property tests.
- **Correction:** the articulatory module does **not** prove a triangle-inequality counterexample; the
  b/t/k triple is a *tight equality* case (`triangle_b_t_k_tight`). Stale "triangle failure" / "1
  Admitted" wording in `FINDINGS_LEDGER.md` and `VERIFICATION_IMPROVEMENTS.md` has been corrected.
- See `docs/verification/FINDINGS_LEDGER.md` Finding 16 for the full hypothesis→experiment→result record.

---

## Completed Work

### Phase 1: Foundational Subsumption Proofs ✅

**File**: `rocq/liblevenshtein/Core.v` (585 lines, 53,419 bytes compiled)

**Theorems Proven**:
1. `subsumes_irreflexive`: ∀p, ¬(p ⊑ p)
2. `subsumes_transitive`: p₁⊑p₂ ∧ p₂⊑p₃ → p₁⊑p₃
3. `subsumes_variant_restriction`: variant(p₁)≠variant(p₂) → ¬(p₁⊑p₂)
4. `subsumes_antisymmetric`: p₁⊑p₂ ∧ p₂⊑p₁ → False (derived)

**Documentation**: `docs/formal-verification/proofs/01_subsumption_properties.md` (600+ lines)

**Key Result**: Subsumption is a strict partial order, justifying anti-chain state minimization.

**Git Commit**: `5da8801` - "feat(formal-verification): Phase 1 - Foundational subsumption proofs"

---

### Phase 2: Position Constructor Correctness ✅

**File**: `rocq/liblevenshtein/Invariants.v` (600 lines, 41,700 bytes compiled)

**Theorems Proven** (13 total):
1. Constructor correctness (6): `new_i_correct`, `new_m_correct`, `new_i_transposing_correct`, `new_m_transposing_correct`, `new_i_splitting_correct`, `new_m_splitting_correct`
2. Decidability (2): `i_invariant_decidable`, `m_invariant_decidable`
3. Accessor safety (4): `valid_position_errors_bounded`, `valid_i_offset_bounded`, `valid_i_reachable`, `i_zero_errors_on_diagonal`
4. Relationships (1): `i_zero_errors_on_diagonal`

**Documentation**: `docs/formal-verification/proofs/02_position_invariants.md` (600+ lines)

**Key Result**: All position constructors maintain invariants; invariant checking is computable.

**Git Commit**: `49561bd` - "feat(formal-verification): Phase 2 - Position constructor correctness"

---

### Phase 3: Standard Operations ✅ (I-Type Complete)

**Completed**: Operations.v ✅, Transitions.v (I-type) ✅
**Next**: Transitions.v (M-type), Documentation, Property Tests

#### Operations.v ✅ (Complete)

**File**: `rocq/liblevenshtein/Operations.v` (~340 lines, 13,762 bytes compiled)

**Definitions**:
- `StandardOperation`: Inductive type (Match, Delete, Insert, Substitute)
- `operation_cost`: Cost semantics (0 for match, 1 for others)
- `consume_word`, `consume_query`: Consumption behavior
- `offset_change`: Offset delta per operation
- `CharacteristicVector`: Axiomatized (like Rholang approach)
- `has_match`: Match checking predicate

**Lemmas Proven** (6):
1. `match_is_free`: Only match costs 0
2. `non_match_costs_one`: Other ops cost 1
3. `only_delete_moves_left`: Only delete changes offset
4. `operation_consumes_something`: All ops consume ≥1 char
5. `cost_relates_to_match`: Cost 0 ⟺ Match
6. `error_op_needs_budget`: Error ops need budget

**Status**: ✅ Compiled successfully, ready for use in Transitions.v

**Git Commit**: `28f041b` - "feat(formal-verification): Phase 3 - Operations.v and status documentation"

---

#### Transitions.v ✅ (I-Type Complete)

**File**: `rocq/liblevenshtein/Transitions.v` (515 lines, 30,700 bytes compiled)

**Inductive Relations Defined**:
- `i_successor`: Successor relation for I-type positions with 4 standard operations
  * `ISucc_Match`: Requires `has_match`, preserves offset and errors
  * `ISucc_Delete`: Requires `errors < n` and `offset > -n`, offset decreases, errors increase
  * `ISucc_Insert`: Requires `errors < n`, preserves offset, errors increase
  * `ISucc_Substitute`: Requires `errors < n`, preserves offset, errors increase

**Theorems Proven** (7 for I-type):
1. `i_match_preserves_invariant`: Match transitions preserve i_invariant
2. `i_delete_preserves_invariant`: Delete transitions preserve i_invariant (complex proof with case analysis)
3. `i_insert_preserves_invariant`: Insert transitions preserve i_invariant
4. `i_substitute_preserves_invariant`: Substitute transitions preserve i_invariant
5. `i_successor_preserves_invariant`: ALL I-type operations preserve invariants (main theorem)
6. `i_successor_cost_correct`: Error accounting matches operation costs exactly

**Key Proof Techniques**:
- Boolean reflection (Z.leb_spec) for extracting integer inequality proofs
- Case analysis on offset sign using Z.abs_eq/Z.abs_neq
- Nat-to-Z conversion with `replace` tactic for S n = n + 1
- Linear integer arithmetic automation (lia) for numeric goals
- Manual conjunct splitting for clarity and debugging

**Findings** (see FINDINGS.md):
- ✅ NO BUGS FOUND - Rust implementation is mathematically correct
- 🔍 F1: Redundant check identified (errors < n ⟹ errors+1 ≤ n)
- 📋 F2: Delete boundary check is implicit in constructor (documented)
- 💡 F3: Offset change could be centralized (simplification opportunity)

**Status**: ✅ All I-type proofs complete, compiles with 0 Admitted

**Git Commit**: `c696c0a` - "feat(formal-verification): Phase 3 - I-type transitions with complete proofs"

---

## Next Steps: Complete Phase 3

### What Remains for Phase 3

#### 1. Add M-Type Successor Relations to Transitions.v

**Current**: I-type complete (515 lines)
**Target**: Add M-type relations and proofs (~400-500 more lines)

M-type successor relation to define (different offset semantics!):

```coq
Inductive m_successor : Position -> StandardOperation ->
                        CharacteristicVector -> Position -> Prop :=

  | MSucc_Match : forall offset errors n cv,
      has_match cv ... ->
      (errors <= n)%nat ->
      m_successor
        (mkPosition VarMFinal offset errors n None)
        OpMatch
        bv
        (mkPosition VarINonFinal offset errors n None)

  | ISucc_Delete : forall offset errors n bv,
      (errors < n)%nat ->
      offset > -Z.of_nat n ->  (* BUG CHECK: Need this? *)
      i_successor
        (mkPosition VarINonFinal offset errors n None)
        OpDelete
        bv
        (mkPosition VarINonFinal (offset - 1) (S errors) n None)

  | ISucc_Insert : (* Similar pattern *)
  | ISucc_Substitute : (* Similar pattern *)
```

#### 2. Prove Invariant Preservation (Main Goal)

```coq
Theorem i_successor_preserves_invariant :
  forall p op bv p',
    i_invariant p ->
    i_successor p op bv p' ->
    i_invariant p'.

(* Prove for each operation separately *)
Lemma i_match_preserves_invariant : ...
Lemma i_delete_preserves_invariant : ...
Lemma i_insert_preserves_invariant : ...
Lemma i_substitute_preserves_invariant : ...
```

#### 3. Prove Cost Correctness

```coq
Theorem i_successor_cost_correct :
  forall p op bv p',
    i_successor p op bv p' ->
    errors p' = (errors p + operation_cost op)%nat.
```

#### 4. M-Type Successors

Same structure for M-type positions:
- `m_successor` inductive relation
- `m_successor_preserves_invariant` theorem
- `m_successor_cost_correct` theorem

---

## Critical Findings & Discrepancies

### From Phase 1-2 Validation

#### ⚠️ Discrepancy 1: Relaxed Invariants

**Location**: `src/transducer/generalized/position.rs:273-276` (I-type)
**Issue**: Rust allows `offset > errors` when `errors == 0` for fractional-weight operations
**Status**: NOT proven in Coq (Phase 3b extension)
**Action**: Document as unproven extension, add warning comments

#### ⚠️ Discrepancy 2: Hybrid Splitting Invariants

**Location**: `src/transducer/generalized/position.rs:436-439` (I-splitting)
**Issue**: Uses M-type subsumption logic not in formal model
**Status**: Implementation more complex than proven spec
**Action**: Either prove necessity or simplify to match Coq

#### ⚠️ Discrepancy 3: Missing Property Tests

**Issue**: `subsumes_transitive` has no property test validating it
**Status**: Critical for anti-chain algorithm, needs test
**Action**: Add property test in Phase 3

---

## Expected Bug Discovery in Transitions.v

### High Probability Bugs (from proof strategy)

1. **Delete at left boundary**
   - **Check**: Does `successors_i_type` verify `offset > -n` before delete?
   - **Location**: `state.rs:297-314`
   - **Proof will require**: `offset > -Z.of_nat n` precondition
   - **If missing**: Bug found, needs fix

2. **Operations at error budget limit**
   - **Check**: When `errors = max_distance`, are error ops correctly blocked?
   - **Location**: All successor functions
   - **Proof will require**: `errors < n` for error ops
   - **If missing**: May create invalid positions

3. **Match index calculation**
   - **Check**: Is `(offset + n)` the correct index for characteristic vector?
   - **Location**: `state.rs:266` (I-type), `state.rs:570` (M-type)
   - **Proof will require**: Correct indexing formula
   - **If wrong**: Match won't work correctly

4. **Insert offset handling**
   - **Check**: Insert should NOT change offset (stays same)
   - **Location**: `state.rs:315-329`
   - **Proof will require**: `offset' = offset`
   - **If wrong**: Offset tracking breaks

---

## Simplification Opportunities Identified

### 1. Extraneous Parameters in Successor Functions

**Current signature** (`state.rs:238`):
```rust
fn successors_i_type(
    offset: i32,
    errors: u8,
    operations: &[OperationType],
    bit_vector: &CharacteristicVector,
    max_distance: u8,
    full_word: &str,        // ⚠️ Potentially extraneous
    word_slice: &str,       // ⚠️ Redundant with full_word?
    input_char: char,
) -> Vec<GeneralizedPosition>
```

**Questions for validation**:
- Is `full_word` actually used? Or just for debugging?
- Can `word_slice` be derived from `full_word + offset`?
- Should bundle into context struct?

**Action**: After proving minimal preconditions in Coq, compare with Rust params

### 2. Redundant Checks

**Pattern observed**:
```rust
if errors < max_distance {
    let new_errors = errors + 1;
    if new_errors <= max_distance {  // ⚠️ Redundant?
        // create successor
    }
}
```

**From arithmetic**: `errors < n ⟹ errors + 1 ≤ n`

**Action**: If invariant proves this, replace second check with `debug_assert!`

---

## Project Goals (from user clarification)

1. **Foundational Theory** - Support current + future work ✅ (Phases 1-2 done, 3 in progress)
2. **Find & Fix Bugs** - Use proofs to identify issues 🚧 (Expected in Transitions.v)
3. **Simplify Implementation** - Remove unnecessary complexity 🚧 (Analysis in progress)
4. **Solid Documentation** - Correctness + theory ✅ (1300+ lines docs so far)
5. **Property-Based Testing** - Validate proofs, find edge cases ⏳ (Planned for Phase 3)

---

## Validation Strategy

### Cross-Validation Process

For each proven theorem:

1. **Extract specification** from Coq proof
2. **Locate corresponding Rust code**
3. **Compare**: Do preconditions match? Do outputs match?
4. **Document**: Create entry in validation matrix
5. **Test**: Write property test validating theorem
6. **Simplify**: Identify unnecessary complexity

### Validation Matrix Structure

| Coq Theorem | Location | Rust Code | Location | Property Test | Status |
|-------------|----------|-----------|----------|---------------|--------|
| `new_i_correct` | Invariants.v:176 | `new_i()` | position.rs:266 | position.rs:566 | ⚠️ Relaxed |
| `subsumes_irreflexive` | Core.v:275 | `subsumes()` | subsumption.rs:62 | subsumption.rs:208 | ✅ Match |
| `subsumes_transitive` | Core.v:306 | (implied) | anti-chain logic | (missing) | ⚠️ Need test |
| *(more to come in Phase 3)* | | | | | |

---

## Files Created/Modified

### Coq Files (rocq/liblevenshtein/)
- ✅ `Core.v` (Phase 1) - 585 lines, compiled
- ✅ `Invariants.v` (Phase 2) - 600 lines, compiled
- ✅ `Operations.v` (Phase 3) - 340 lines, compiled
- ⏳ `Transitions.v` (Phase 3) - **NEXT TO CREATE**
- 📋 `_CoqProject` - Build configuration

### Documentation (docs/formal-verification/)
- ✅ `README.md` - Project overview (1300+ lines)
- ✅ `proofs/01_subsumption_properties.md` (600+ lines)
- ✅ `proofs/02_position_invariants.md` (600+ lines)
- ⏳ `proofs/03_standard_operations.md` - **NEXT TO CREATE**
- ⏳ `VALIDATION_MATRIX.md` - **TO CREATE**
- ⏳ `DISCREPANCIES.md` - **TO CREATE**
- ⏳ `SIMPLIFICATION_OPPORTUNITIES.md` - **TO CREATE**

---

## How to Resume

### 1. Start Fresh Session

Simply start a new conversation in Claude Code. The git repository state is preserved.

### 2. Context to Provide

```
I'm continuing formal verification of Levenshtein automata. We've completed
Phases 1-2 (subsumption + invariants). Phase 3 is in progress - Operations.v
is done, now need to create Transitions.v with successor relations and
invariant preservation proofs.

Please read: docs/formal-verification/STATUS.md for current state.

Key focus: As we prove successor functions correct, proof failures will
reveal bugs. Document all bugs found, identify simplification opportunities,
and create property tests for proven theorems.
```

### 3. First Task in New Session

```coq
(* Create Transitions.v starting with: *)

From Stdlib Require Import ZArith Arith Lia Bool.
Require Import Core.
Require Import Invariants.
Require Import Operations.

(* Define i_successor inductive relation *)
(* Start with Match operation, prove preservation *)
(* Then Delete, Insert, Substitute *)
(* Look for bugs in preconditions *)
```

### 4. Commands to Verify State

```bash
# Verify all previous work compiles
cd rocq/liblevenshtein
coqc -R . LevensteinAutomata Core.v
coqc -R . LevensteinAutomata Invariants.v
coqc -R . LevensteinAutomata Operations.v
# All should succeed

# Check git status
git log --oneline -5
# Should see Phase 1 and Phase 2 commits
```

---

## Compilation Commands

```bash
# Compile individual files
cd /home/dylon/Workspace/f1r3fly.io/liblevenshtein-rust/rocq/liblevenshtein
coqc -R . LevensteinAutomata Core.v
coqc -R . LevensteinAutomata Invariants.v
coqc -R . LevensteinAutomata Operations.v

# Compile Transitions.v (when created)
coqc -R . LevensteinAutomata Transitions.v
```

---

## Success Metrics for Phase 3

When Phase 3 is complete, we should have:

✅ Operations.v compiled (13,762 bytes) - **DONE**
⏳ Transitions.v compiled (~40,000+ bytes) - **NEXT**
⏳ 15+ theorems proven (successor preservation + cost correctness)
⏳ 3+ bugs discovered and documented
⏳ 3+ simplification opportunities identified
⏳ 10+ property tests added
⏳ Complete validation matrix
⏳ Documentation: 03_standard_operations.md (700+ lines)

---

## Token Budget Notes

- **This session**: Used 125k/200k tokens (62.5%)
- **Recommendation**: Start fresh for Transitions.v (complex proofs need budget)
- **Transitions.v estimate**: 50-70k tokens for proofs + documentation + validation

---

## References

### Theory
- `docs/research/weighted-levenshtein-automata/README.md` - Part I, II
- Mitankin thesis - Position types, subsumption

### Implementation
- `src/transducer/generalized/position.rs` - Position constructors
- `src/transducer/generalized/state.rs:238-890` - Successor functions
- `src/transducer/generalized/subsumption.rs` - Subsumption implementation
- `src/transducer/operation_type.rs` - Operation definitions

### Tests
- `tests/proptest_comprehensive.rs` - Property-based tests
- `tests/proptest_automaton_distance_cross_validation.rs` - Oracle testing
- `src/transducer/generalized/position.rs:562-784` - Constructor tests

---

**End of Status Report**

*This document should contain everything needed to resume Phase 3 with full context.*
