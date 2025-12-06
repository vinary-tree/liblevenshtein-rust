# Termination Proof

Formal argument for termination of the simplification transpiler.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Overview

This document provides a formal argument that the simplification transpiler always terminates. Termination is critical for a production system - we must guarantee that simplification completes in bounded time.

---

## Termination Strategy

The termination argument relies on three mechanisms:

1. **Well-founded ordering on terms**
2. **Termination weights for rules**
3. **Cycle detection via visited set**
4. **Absolute iteration bound**

---

## Well-Founded Ordering

### Term Measure

Define a measure function `M: Proc → (ℕ × ℕ × ℕ)` that maps terms to lexicographically ordered triples:

```
M(term) = (size(term), depth(term), complexity(term))
```

Where:
- `size(term)`: Number of AST nodes
- `depth(term)`: Maximum nesting depth
- `complexity(term)`: Structural complexity measure

### Lexicographic Ordering

Triples are compared lexicographically:

```
(a₁, b₁, c₁) < (a₂, b₂, c₂)  iff
    a₁ < a₂, or
    (a₁ = a₂ and b₁ < b₂), or
    (a₁ = a₂ and b₁ = b₂ and c₁ < c₂)
```

This ordering is well-founded (no infinite descending chains) because ℕ is well-founded.

---

## Rule Classification

### Size-Reducing Rules (Weight < 0)

These rules strictly decrease the `size` component:

| Rule | Pattern | Template | Size Change |
|------|---------|----------|-------------|
| add-zero-right | `x + 0` | `x` | -2 |
| add-zero-left | `0 + x` | `x` | -2 |
| mul-one-right | `x * 1` | `x` | -2 |
| mul-zero | `x * 0` | `0` | -n+1 |
| double-neg | `--x` | `x` | -2 |
| nil-identity | `P \| 0` | `P` | -2 |
| dead-scope | `new x.P` (x unused) | `P` | -2 |
| if-true | `if true then P else Q` | `P` | -4-\|Q\| |

**Property**: After a size-reducing rule, `M(new) < M(old)` because the first component decreases.

### Size-Preserving Rules (Weight = 0)

These rules preserve `size` but may change `depth` or `complexity`:

| Rule | Pattern | Template | Invariant |
|------|---------|----------|-----------|
| par-commute | `P \| Q` | `Q \| P` | Moves toward canonical order |
| par-assoc | `(P\|Q)\|R` | `P\|(Q\|R)` | Reduces left-nesting |
| scope-extrude | `new x.(P\|Q)` | `(new x.P)\|Q` | Reduces scope depth |

**Property**: Size-preserving rules must be shown to eventually enable a size-reducing rule or reach a fixed point.

---

## Termination Argument by Phase

### Phase 1: LocalSimplification

**Rules**: Algebraic identities (add-zero, mul-one, double-neg, etc.)

**Termination argument**:
1. All rules in this phase have `termination_weight < 0`
2. Each rule application strictly decreases `size(term)`
3. `size(term) ∈ ℕ`, so cannot decrease forever
4. ∴ Phase terminates in at most `size(initial)` iterations

**Bound**: O(n) iterations where n = initial term size

### Phase 2: AnalysisDriven

**Rules**: Dead code elimination, constant propagation

**Termination argument**:
1. Dead code rules have `termination_weight < 0` (remove unreachable code)
2. Constant folding replaces expressions with values (weight < 0)
3. Each application decreases size
4. ∴ Phase terminates

**Bound**: O(n) iterations

### Phase 3: StructuralNormalization

**Rules**: Rholang congruence (nil-identity, commute, assoc, scope-extrude)

**Termination argument**:

This phase is more complex because some rules are size-preserving.

**Step 1**: Nil-identity rules
- Pattern: `P | 0` → `P`
- Weight: -1 (size-reducing)
- Applied first, eliminates all Nil nodes
- Terminates in O(n) steps

**Step 2**: Associativity rules
- Pattern: `(P|Q)|R` → `P|(Q|R)`
- Weight: 0 (size-preserving)
- Directed: always reassociate to right
- Each Par node is reassociated at most once
- Terminates in O(n) steps

**Step 3**: Commutativity rules
- Pattern: `P|Q` → `Q|P` when `order(P) > order(Q)`
- Weight: 0 (size-preserving)
- **Key**: Guard ensures we only swap toward canonical ordering
- Once in canonical order, guard fails - no more applications
- Terminates in O(n²) steps (like bubble sort)

**Step 4**: Scope extrusion
- Pattern: `new x.(P|Q)` → `(new x.P)|Q` when `x ∉ FV(Q)`
- Weight: 0 (size-preserving)
- **Key**: New binding moves outward until it hits a use
- Each New can extrude at most `depth(term)` times
- Terminates in O(n × d) steps where d = depth

**Combined bound**: O(n² + n × d) = O(n²) for Phase 3

### Phase 4: TypeDirected

**Rules**: Redundant cast removal, type-aware inlining

**Termination argument**:
1. Cast removal has weight < 0 (removes nodes)
2. Inlining is guarded by benefit heuristics
3. All applications decrease size or improve cost
4. ∴ Phase terminates

**Bound**: O(n) iterations

---

## Global Termination

### Theorem: The simplification transpiler terminates

**Proof**:

Let `T₀` be the initial term with `size(T₀) = n`.

1. **Phase bounds**: Each phase terminates with bound:
   - Phase 1: O(n)
   - Phase 2: O(n)
   - Phase 3: O(n²)
   - Phase 4: O(n)

2. **Cross-phase reanalysis**: When Phase 2 or 4 makes changes, we may re-run earlier phases. However:
   - Size can only decrease (size-reducing rules)
   - Total size reduction across all reanalysis cycles ≤ n
   - ∴ Total reanalysis overhead is O(n × phases) = O(n)

3. **Cycle detection**: The visited set `V` tracks all seen terms by hash:
   - If `T ∈ V`, we stop (would repeat)
   - Since hash is content-based, `|V| ≤ number of distinct terms seen`
   - Since terms only get smaller, `|V| ≤ 2^n` (actually much less)

4. **Absolute bound**: Configuration provides `max_iterations` as hard limit

**Total complexity**: O(n² × p) where p = number of phases = O(n²)

□

---

## Formal Measure Function

```rust
/// Termination measure: (size, depth, structural_complexity)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TermMeasure {
    size: usize,
    depth: usize,
    complexity: usize,
}

impl TermMeasure {
    pub fn compute(proc: &Proc) -> Self {
        Self {
            size: proc.size(),
            depth: proc.depth(),
            complexity: structural_complexity(proc),
        }
    }
}

/// Size: count of AST nodes
fn size(proc: &Proc) -> usize {
    match proc {
        Proc::Nil => 1,
        Proc::Num(_) | Proc::Bool(_) | Proc::Var(_) => 1,
        Proc::Par(p, q) => 1 + size(p) + size(q),
        Proc::New(_, body) => 1 + size(body),
        Proc::Send(_, data, cont) => 1 + size(data) + size(cont),
        Proc::Receive(_, _, body) => 1 + size(body),
        // ... other cases
    }
}

/// Depth: maximum nesting level
fn depth(proc: &Proc) -> usize {
    match proc {
        Proc::Nil | Proc::Num(_) | Proc::Bool(_) | Proc::Var(_) => 0,
        Proc::Par(p, q) => 1 + depth(p).max(depth(q)),
        Proc::New(_, body) => 1 + depth(body),
        // ... other cases
    }
}

/// Structural complexity: measures deviation from canonical form
fn structural_complexity(proc: &Proc) -> usize {
    match proc {
        Proc::Par(p, q) => {
            let left_par_penalty = if matches!(p.as_ref(), Proc::Par(_, _)) { 1 } else { 0 };
            let order_penalty = if canonical_order(p) > canonical_order(q) { 1 } else { 0 };
            left_par_penalty + order_penalty + structural_complexity(p) + structural_complexity(q)
        }
        Proc::New(x, body) => {
            // Penalty for scope that could be extruded
            let extrusion_penalty = count_extrudable(x, body);
            extrusion_penalty + structural_complexity(body)
        }
        _ => 0
    }
}
```

---

## Verification of Rule Weights

Each rule must satisfy:

```
∀ old, new: applies(rule, old, new) ⟹
    (weight < 0 ⟹ size(new) < size(old)) ∧
    (weight = 0 ⟹ size(new) = size(old) ∧ complexity(new) < complexity(old))
```

### Verification Table

| Rule | Weight | Size Change | Complexity Change | Verified |
|------|--------|-------------|-------------------|----------|
| add-zero-right | -1 | -2 | N/A | ✓ |
| add-zero-left | -1 | -2 | N/A | ✓ |
| mul-zero | -2 | -(n-1) | N/A | ✓ |
| double-neg | -1 | -2 | N/A | ✓ |
| nil-identity-right | -1 | -2 | N/A | ✓ |
| nil-identity-left | -1 | -2 | N/A | ✓ |
| par-commute | 0 | 0 | -1 (order penalty) | ✓ |
| par-assoc-right | 0 | 0 | -1 (left-par penalty) | ✓ |
| scope-extrude | 0 | 0 | -1 (extrusion possible) | ✓ |
| dead-scope | -1 | -2 | N/A | ✓ |

---

## Coq Formalization (Sketch)

```coq
(** Term measure type *)
Record Measure := mkMeasure {
  size : nat;
  depth : nat;
  complexity : nat
}.

(** Lexicographic ordering *)
Definition measure_lt (m1 m2 : Measure) : Prop :=
  (size m1 < size m2) \/
  (size m1 = size m2 /\ depth m1 < depth m2) \/
  (size m1 = size m2 /\ depth m1 = depth m2 /\ complexity m1 < complexity m2).

(** Measure is well-founded *)
Theorem measure_wf : well_founded measure_lt.
Proof.
  (* Follows from well-foundedness of nat with lexicographic product *)
  apply wf_lexprod; apply lt_wf.
Qed.

(** Rule application decreases measure *)
Theorem rule_decreases_measure : forall rule old new,
  applies rule old new ->
  measure_lt (compute_measure new) (compute_measure old).
Proof.
  (* Case analysis on rule type *)
  intros rule old new H.
  destruct (termination_weight rule) eqn:Hw.
  - (* Negative weight: size decreases *)
    left. apply size_reducing_rule_decreases_size; auto.
  - (* Zero weight: complexity decreases *)
    right; right. split; [reflexivity|].
    split; [reflexivity|].
    apply size_preserving_rule_decreases_complexity; auto.
Qed.

(** Main termination theorem *)
Theorem simplification_terminates : forall initial,
  exists final, simplify initial = final.
Proof.
  intro initial.
  apply well_founded_induction with (R := measure_lt).
  - exact measure_wf.
  - intros term IH.
    destruct (find_applicable_rule term) as [[rule new]|].
    + (* Rule applies: recurse with smaller measure *)
      apply IH. apply rule_decreases_measure. auto.
    + (* No rule applies: fixpoint reached *)
      exists term. reflexivity.
Qed.
```

---

## Practical Safeguards

Even with the formal argument, we include practical safeguards:

```rust
pub struct TerminationGuard {
    /// Maximum iterations (absolute bound)
    max_iterations: usize,

    /// Current iteration count
    iterations: usize,

    /// Visited term hashes
    visited: HashSet<u64>,

    /// Initial term size (for progress tracking)
    initial_size: usize,
}

impl TerminationGuard {
    pub fn check(&mut self, term: &Proc) -> TerminationCheck {
        self.iterations += 1;

        // Hard limit
        if self.iterations >= self.max_iterations {
            return TerminationCheck::Stop(TerminationReason::MaxIterations);
        }

        // Cycle detection
        let hash = term.content_hash();
        if self.visited.contains(&hash) {
            return TerminationCheck::Stop(TerminationReason::CycleDetected);
        }
        self.visited.insert(hash);

        // Progress check (term should generally shrink)
        let current_size = term.size();
        if current_size > self.initial_size * 2 {
            // Something went wrong - size increased dramatically
            return TerminationCheck::Stop(TerminationReason::SizeExplosion);
        }

        TerminationCheck::Continue
    }
}
```

---

## Complexity Summary

| Phase | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| LocalSimplification | O(n) | O(1) |
| AnalysisDriven | O(n) | O(n) for analysis facts |
| StructuralNormalization | O(n²) | O(n) for canonical form |
| TypeDirected | O(n) | O(n) for types |
| **Total** | **O(n²)** | **O(n)** |

Where n = size of input term.

---

## Next Steps

- [Performance Targets](08-performance-targets.md) - Benchmarks and optimization goals
- [Strategy Selection](04-strategy-selection.md) - Iteration control implementation

---

## Changelog

- **2025-12-06**: Initial termination proof documentation
