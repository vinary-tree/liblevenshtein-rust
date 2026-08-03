# Formal Verification Improvements

This document tracks the implementation of formal verification improvements for liblevenshtein-rust using TLA+ (model checking) and Rocq/Coq (theorem proving).

## Overview

The verification infrastructure combines two complementary approaches:
- **TLA+**: Model checking for state machine specifications and temporal properties
- **Rocq/Coq**: Theorem proving for mathematical properties and algorithmic correctness

## New Verification Modules Created

### TLA+ Specifications (`docs/verification/tla/`)

| File | Purpose | Key Properties |
|------|---------|----------------|
| `OnlineScanner.tla` | Multi-match tracking | ActiveMatchesBounded, NoMissedMatches, PositionMonotonicity |
| `ProductAutomaton.tla` | NFA × Levenshtein composition | ProductCorrectness, StateSpaceBounded, CostMonotonicity |
| `Subsumption.tla` | Executable classic-position coverage and antichain lifecycle | Standard/OSA reflexivity, MergeSplit irreflexivity, variant separation, strict-dominance laws, retained-cover preservation, termination |
| `PriorityQuery.tla` | A* search correctness | HeapInvariant, AdmissibleHeuristic, Optimality |

### Rocq/Coq Modules

#### Executable Subsumption Conformance (`docs/verification/core/theories/Conformance/`)

| File | Purpose | Key Theorems |
|------|---------|--------------|
| `RustSubsumption.v` | Exact branch model for `Position::subsumes`, its weighted carrier, and state insertion | cross-kind OSA separation, pending/pending exactness, normal/normal equivalence to Standard, arithmetic transitivity, equality-aware retention, shrinking-state witness |

`core/theories/Automaton/Subsumption.v` is a separate conservative research
relation used inside the partial Automaton proof tree. It intentionally prunes
less aggressively than Rust, including same-index-only normal OSA pruning and
extra finality guards. The formal manifest therefore classifies it as
`partial`; it must not be cited as executable conformance. Exact Rust claims
use `Conformance/RustSubsumption.v` together with `tla/Subsumption.tla`.

#### LLRE Module (`docs/verification/llre/theories/`)

| File | Purpose | Key Theorems |
|------|---------|--------------|
| `SymbolExpansion.v` | Symbol table expansion | `symbol_expansion_terminates`, `expansion_preserves_language` |
| `ThompsonConstruction.v` | Regex → NFA compilation | `thompson_soundness`, `thompson_completeness`, `thompson_correctness` |

#### Myers Module (`docs/verification/myers/theories/`)

| File | Purpose | Key Theorems |
|------|---------|--------------|
| `Equivalence.v` | Bit-parallel algorithm | `myers_equivalence` (for patterns ≤ 64 chars) |

#### Articulatory Module (`docs/verification/articulatory/theories/`)

| File | Purpose | Key Theorems |
|------|---------|--------------|
| `FeatureDistance.v` | Phoneme feature distance, parameterized by a `FeatureWeights` record | `articulatory_w_symmetric`, `articulatory_w_identity`, `articulatory_w_nonneg`, `articulatory_w_bounded_by_sum`, `articulatory_w_monotone`; standard-weight corollaries `articulatory_symmetric`/`_identity`/`articulatory_bounded` |
| `FeatureDistanceWeighted.v` | Faithful 7-dimension model (vowel path + `Qmin` cap, mirrors Rust `FeatureDistanceWeights`) | `fsd7_symmetric`, `fsd7_identity`, `fsd7_nonneg`, `fsd7_bounded`, `fsd7_monotone` |

Note: the module does **not** assert a metric-space triangle theorem for articulatory distance; the b/t/k triple is a *tight* (equality) triangle case, not a counterexample (see Key Insights below).

#### Product Module (`docs/verification/product/theories/`)

| File | Purpose | Key Theorems |
|------|---------|--------------|
| `ProductState.v` | NFA × Levenshtein product | `product_soundness`, `product_completeness_empty`, `subsumption_preserves_reachability`, `product_correctness` |

## Admitted Lemmas Completed

### Grammar NFA Types (`Types.v`)

The following lemmas were completed:

1. **`op_match_1_bounded`** - Match operation satisfies bounded diagonal
2. **`op_insert_1_bounded`** - Insert operation satisfies bounded diagonal
3. **`op_delete_1_bounded`** - Delete operation satisfies bounded diagonal
4. **`op_substitute_1_bounded`** - Substitute operation satisfies bounded diagonal
5. **`op_transpose_1_bounded`** - Transpose operation satisfies bounded diagonal
6. **`cv_set_test_neq`** - Characteristic vector bit independence

## Remaining Admitted Lemmas

### Critical (Blocks Other Proofs)

#### Automaton Completeness (`core/theories/Automaton/Completeness.v`)
- ~15 admitted lemmas for position state construction
- Key: `reachable_implies_contained_aux`, `automaton_run_not_dead_for_reachable`

#### NFA Soundness (`grammar/theories/NFA/Soundness.v`)
- ~12 admitted lemmas
- Key: `path_cost_matches_operations`, `nfa_soundness`

#### NFA Completeness (`grammar/theories/NFA/Completeness.v`)
- ~11 admitted lemmas
- Key: `edit_sequence_induces_path`, `nfa_completeness`

### Important (MSM Metric)

#### MSM Triangle Inequality (`msm/theories/Metric/TriangleInequality.v`)
- Main theorem: `msm_triangle`
- Strategy: DP composition with c-function context

#### MSM Distance (`msm/theories/Core/MsmDistance.v`)
- 4 admitted: `msm_nonneg`, `msm_reflexive`, `msm_zero_implies_equal`

### Lower Priority

#### Grammar Core (`grammar/theories/Core/Edit.v`)
- 10 admitted: symmetry, triangle inequality, bounds

#### Grammar Lattice (`grammar/theories/Core/Lattice.v`)
- 7 admitted: lattice properties

## Build Instructions

### TLA+ Model Checking

```bash
# Install TLA+ tools
wget https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar

# Check specifications
java -jar tla2tools.jar docs/verification/tla/OnlineScanner.tla
java -jar tla2tools.jar docs/verification/tla/ProductAutomaton.tla
java -jar tla2tools.jar docs/verification/tla/Subsumption.tla
java -jar tla2tools.jar docs/verification/tla/PriorityQuery.tla
```

### Rocq/Coq Compilation

```bash
# Compile with resource limits (per CLAUDE.md)
cd docs/verification

# Individual modules
cd llre/theories && make
cd myers/theories && make
cd articulatory/theories && make
cd product/theories && make

# Or compile all with resource limits
systemd-run --user --scope -p MemoryMax=126G -p CPUQuota=1800% \
  -p IOWeight=30 -p TasksMax=200 make -j1
```

## Verification Strategy

### For TLA+ Specifications
1. Define state variables and type invariants
2. Specify Init and Next relations
3. Add safety invariants (properties that must always hold)
4. Add liveness properties (properties that must eventually hold)
5. Model check with small bounds, then increase

### For Rocq Proofs
1. Mirror Rust types in Coq
2. Define semantic relations (matching, accepting, etc.)
3. Prove key lemmas incrementally
4. Build up to main theorems
5. Extract to OCaml for testing (optional)

## Correspondence to Rust Implementation

| Rocq Module | Rust File |
|-------------|-----------|
| `SymbolExpansion.v` | `src/phonetic/llre/symbol_expander.rs` |
| `ThompsonConstruction.v` | `src/phonetic/llre/nfa_compiler.rs` |
| `Equivalence.v` | `src/distance/myers.rs` |
| `FeatureDistance.v` | `src/phonetic/feature_distance.rs`, `src/transducer/articulatory_costs.rs` |
| `ProductState.v` | `src/phonetic/nfa/product.rs` |
| `RustSubsumption.v` | `src/cost/subsumption.rs`, `src/transducer/{position,state}.rs`, and their weighted twins |
| `Automaton/Subsumption.v` | conservative research relation only; no line-by-line executable correspondence |

| TLA+ Spec | Rust File |
|-----------|-----------|
| `OnlineScanner.tla` | `src/phonetic/online_scanner.rs` |
| `ProductAutomaton.tla` | `src/phonetic/nfa/product.rs` |
| `Subsumption.tla` | `src/cost/subsumption.rs`, `src/transducer/{position,state}.rs`, and their weighted twins |
| `PriorityQuery.tla` | `src/transducer/priority_query.rs` |

## Key Insights

### Triangle Inequality for Articulatory Distance

The articulatory feature distance is **not asserted to be a metric**: `FeatureDistance.v` proves
symmetry, identity, and boundedness, but no metric-space triangle theorem. The commonly-cited b/t/k
triple is a *tight* (equality) triangle case, **not** a counterexample:

```
phoneme_b = (Bilabial, Plosive, Voiced)      -- "b"
phoneme_k = (Velar, Plosive, Voiceless)      -- "k"
phoneme_t = (Alveolar, Plosive, Voiceless)   -- "t"

d(b, k) = 28/100   (place 0.4*0.6 + voice 0.2*0.2)
d(b, t) = 16/100   (place 0.4*0.3 + voice 0.2*0.2)
d(t, k) = 12/100   (place 0.4*0.3)

28/100 = 16/100 + 12/100   -- triangle holds with EQUALITY (theorem triangle_b_t_k_tight)
```

Because the module proves no triangle theorem, algorithms that *require* the triangle inequality
(e.g. A* with an admissible articulatory heuristic) still have no metric guarantee — but this rests
on the absence of a proof plus a tight example, not on a proven counterexample. (An earlier draft
mis-stated `d(b,k) = 32/100`; the correct value is `0.4*0.6 + 0.2*0.2 = 0.28`.)

### Myers Algorithm Constraint

The Myers bit-parallel algorithm requires pattern length ≤ 64 characters (word size). For longer patterns, a block-based approach is needed. The `Equivalence.v` proof is scoped to the single-word case.

### MSM Metric Properties (2026-01-20 Progress)

The Move-Split-Merge distance metric proofs have been advanced:

**Completed:**
- `c_func_triangle_helper` - C function triangle helper (2 admits → Qed)
- `msm_nonneg` - Non-negativity of MSM distance (fully proven)
- `msm_reflexive_singleton` - Reflexivity for single-element lists

**Partially Complete:**
- `msm_reflexive` - General reflexivity (1 admit for 2+ element lists)
- `msm_symmetric` - Symmetry (1 admit for main case; empty cases proven)
- `msm_triangle` - Triangle inequality (4 admits with proof strategies documented)

**Key Insight for Reflexivity:**
The original `msm_init_row_same_last` lemma was incorrectly stated. Only diagonal elements are 0 when X = Y; non-diagonal elements use c_func which adds cost ≥ c_const. The proof requires tracking diagonal positions through the DP computation.

### Grammar NFA Soundness (2026-01-20 Progress)

**Critical Discovery: Minimal Implementations**

The NFA verification had two minimal implementations that prevented certain
axioms from being converted to actual proofs:

1. **`extract_edit_sequence`** (Soundness.v): Always returns `[]` for any path
   - ~~Minimal extraction only~~ → **`extract_edit_sequence_full` NOW IMPLEMENTED (2026-01-20)**
2. **`apply_edit_sequence`** (Completeness.v): ~~Always returns the original string unchanged~~ - **NOW IMPLEMENTED**

**Implementation of `apply_edit_sequence` (2026-01-20):**

The real implementation was added to Completeness.v:

```coq
(** Helper: Convert list of ascii to string *)
Fixpoint string_of_list_ascii (l : list ascii) : string :=
  match l with
  | [] => EmptyString
  | c :: cs => String c (string_of_list_ascii cs)
  end.

Fixpoint apply_edit_sequence (s : string) (edits : list OperationType) : string :=
  match edits with
  | [] => s  (* No more edits: return remaining source unchanged *)
  | op :: rest =>
      let output := string_of_list_ascii (op_chars_y op) in
      let remaining := substring (op_consume_x op) (String.length s) s in
      append output (apply_edit_sequence remaining rest)
  end.
```

Semantics:
- For empty edits, returns the source string unchanged (identity transformation)
- For each operation: outputs `op_chars_y`, advances source by `op_consume_x`, continues
- Remaining source is appended when edits are exhausted

Examples:
- `apply_edit_sequence "abc" [] = "abc"` (identity)
- `apply_edit_sequence "abc" [op_delete 'a'] = "bc"` (delete first char)
- `apply_edit_sequence "ab" [op_insert 'x', op_match 'a', op_match 'b'] = "xab"`

**Implementation of `extract_edit_sequence_full` (2026-01-20):**

A real extraction function was added to Soundness.v that infers operations from position changes and string context:

```coq
(** Infer operation from position change and string context *)
Definition infer_operation
    (target input : string)
    (p1 p2 : Position)
    (input_pos : nat)
    : option (OperationType * bool) :=
  let di := pos_i p2 - pos_i p1 in
  let de := pos_e p2 - pos_e p1 in
  match di, de with
  | 1, 0 => Some (op_match target_char, true)   (* Match *)
  | 0, 1 => Some (op_insert input_char, true)   (* Insert *)
  | 1, 1 => (* Delete or Substitute based on char comparison *)
  | 2, 1 => Some (op_transpose c1 c2, true)     (* Transpose *)
  | _, _ => None
  end.

(** Extract edit sequence with full string context *)
Fixpoint extract_edit_sequence_full_aux
    (target input : string)
    (input_pos : nat)
    (path : AutomatonPath)
    : list OperationType := ...

Definition extract_edit_sequence_full
    (target input : string)
    (path : AutomatonPath)
    : list OperationType :=
  extract_edit_sequence_full_aux target input 0 path.
```

Key design decisions:
- Tracks input position separately (Position only tracks target position)
- Distinguishes Delete vs Substitute by checking if target[i] = input[j]
- Returns (operation, consumed) tuple where `consumed` indicates if input was consumed
- Handles the 5 standard operations: Match, Insert, Delete, Substitute, Transpose

Note: The original `extract_edit_sequence` is kept as a compatibility shim.
Use `extract_edit_sequence_full` for actual operation extraction.

With this implementation, the following axioms can now potentially be proven:

**Axioms Now Potentially Provable (Implementations Complete):**

| Axiom | Status | Notes |
|-------|--------|-------|
| `accepting_automaton_has_edit_sequence` | **CAN NOW BE PROVEN** | Use `extract_edit_sequence_full` + `apply_edit_sequence` |
| `phonetic_only_when_phonetic_ops_used` | Partially blocked | Needs operation filtering from extracted sequence |
| `edit_sequence_empty_output_zero_consume` | **CAN NOW BE PROVEN** | `apply_edit_sequence` semantics are well-defined |

**Documentation Added:**
- Detailed comments explaining WHY each axiom cannot be proven
- What implementation changes would be needed
- Proof strategies for when real implementations are added

**Bug Fixes:**
- Fixed `cv_set_test_neq` proof in Types.v (incorrect proof structure with N.shiftl_spec_low)
- Reorganized `phonetic_path_cheaper_ax` axiom in Operations.v (moved after `path_cost` definition)

**Proofs Completed:**
- `op_match_1_bounded`, `op_insert_1_bounded`, `op_delete_1_bounded`, `op_substitute_1_bounded`, `op_transpose_1_bounded` - All standard operations satisfy bounded diagonal
- `cv_set_test_neq` - Characteristic vector bit independence (complete proof)

**Key Insight:**
The original `valid_path` definition didn't constrain error bounds for intermediate positions in paths longer than one element.

**Fix Applied (2026-01-20):**
The `valid_path` definition in Completeness.v has been updated to include error bounds for all positions:

```coq
Fixpoint valid_path ... :=
  match path with
  | [] => True
  | [p] => pos_e p <= automaton_max_distance aut
  | p1 :: p2 :: rest =>
      pos_e p1 <= automaton_max_distance aut /\  (* NEW: was missing! *)
      (exists op, ...) /\
      valid_path aut target input (p2 :: rest)
  end.
```

A new lemma `valid_path_positions_bounded` proves that all positions in a valid path have bounded error counts. The `valid_path_bounded` in Soundness.v is now equivalent to the fixed `valid_path`.

## Future Work

1. ~~**Implement real `extract_edit_sequence`**~~ → **DONE** (`extract_edit_sequence_full` in Soundness.v)
2. ~~**Implement real `apply_edit_sequence`**~~ → **DONE** (in Completeness.v)
3. **Prove round-trip property**: Show `apply_edit_sequence target (extract_edit_sequence_full target input path) = input` for valid paths
4. **Convert `accepting_automaton_has_edit_sequence` from axiom to theorem** using the new implementations
5. Complete remaining admitted lemmas in NFA Soundness/Completeness
6. Add more TLA+ temporal properties
7. Extend Myers proof to block-based algorithm
8. Add extraction to test against Rust implementation
9. Formalize phonetic rule priority ordering
10. Complete MSM reflexivity proof by tracking diagonal elements through msm_compute_row
11. Prove MSM lower bound lemma: MSM(X, Y) >= ||X| - |Y|| * c for triangle inequality
12. Fix `valid_path` definition to include error bounds for all positions
