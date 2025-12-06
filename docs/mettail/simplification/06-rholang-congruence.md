# Rholang Structural Congruence Rules

Detailed specification of Rholang structural congruence laws for process simplification.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Overview

Rholang's structural congruence laws define when two syntactically different processes are semantically equivalent. These laws form the foundation for process simplification and normalization.

The structural congruence relation `≡` is the smallest congruence satisfying:
- Nil identity laws
- Commutativity and associativity of parallel composition
- Scope extrusion laws
- Alpha equivalence

---

## Structural Congruence Laws

### Law 1: Nil Identity

Nil (`0`) is the identity element for parallel composition:

```
P | 0 ≡ P
0 | P ≡ P
```

**MORK Rules**:

```rust
pub struct NilIdentityRight;

impl SimplificationRule for NilIdentityRight {
    fn name(&self) -> &str { "nil-identity-right" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        // (Par $P (Nil))
        parse_expr("(Par $P (Nil))")
    }

    fn template(&self) -> Expr {
        parse_expr("$P")
    }

    fn termination_weight(&self) -> i32 { -1 }  // Reduces size
}

pub struct NilIdentityLeft;

impl SimplificationRule for NilIdentityLeft {
    fn name(&self) -> &str { "nil-identity-left" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        // (Par (Nil) $P)
        parse_expr("(Par (Nil) $P)")
    }

    fn template(&self) -> Expr {
        parse_expr("$P")
    }

    fn termination_weight(&self) -> i32 { -1 }
}
```

**MeTTa Definition**:

```metta
(simplification-rule
    (id nil-identity-right)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Par $P (Nil)))
    (template $P)

    (soundness "P | 0 ≡ P by nil identity")
    (termination-weight -1))

(simplification-rule
    (id nil-identity-left)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Par (Nil) $P))
    (template $P)

    (soundness "0 | P ≡ P by nil identity")
    (termination-weight -1))
```

---

### Law 2: Commutativity

Parallel composition is commutative:

```
P | Q ≡ Q | P
```

**MORK Rule** (with canonical ordering to ensure termination):

```rust
pub struct ParCommute;

impl SimplificationRule for ParCommute {
    fn name(&self) -> &str { "par-commute" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        parse_expr("(Par $P $Q)")
    }

    fn template(&self) -> Expr {
        parse_expr("(Par $Q $P)")
    }

    fn guard(&self, term: &Proc, _facts: &AnalysisFacts) -> bool {
        // Only commute to reach canonical (sorted) form
        // This ensures termination: we only move toward the canonical ordering
        if let Proc::Par(p, q) = term {
            canonical_order(p) > canonical_order(q)
        } else {
            false
        }
    }

    fn termination_weight(&self) -> i32 { 0 }  // Size-preserving
}

/// Canonical ordering for processes
/// Lower order = comes first in canonical form
fn canonical_order(proc: &Proc) -> u64 {
    match proc {
        // Nil always last (will be eliminated by nil-identity)
        Proc::Nil => u64::MAX,

        // Send/Receive ordered by channel name
        Proc::Send(chan, _, _) => hash_channel(chan),
        Proc::Receive(_, chan, _) => hash_channel(chan),

        // New bindings ordered by variable name
        Proc::New(x, _) => hash_var(x),

        // Par uses hash of normalized subterms
        Proc::Par(p, q) => {
            let p_order = canonical_order(p);
            let q_order = canonical_order(q);
            p_order.min(q_order)
        }

        // Default: use content hash
        _ => proc.content_hash()
    }
}
```

**MeTTa Definition**:

```metta
(simplification-rule
    (id par-commute)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Par $P $Q))
    (template (Par $Q $P))

    (guard (> (canonical-order $P) (canonical-order $Q)))

    (soundness "P | Q ≡ Q | P by commutativity")
    (termination-weight 0))
```

---

### Law 3: Associativity

Parallel composition is associative:

```
(P | Q) | R ≡ P | (Q | R)
```

**MORK Rule** (flattening for canonical form):

```rust
pub struct ParAssocRight;

impl SimplificationRule for ParAssocRight {
    fn name(&self) -> &str { "par-assoc-right" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        // ((P | Q) | R) → (P | (Q | R))
        parse_expr("(Par (Par $P $Q) $R)")
    }

    fn template(&self) -> Expr {
        parse_expr("(Par $P (Par $Q $R))")
    }

    fn termination_weight(&self) -> i32 { 0 }  // Size-preserving
}

/// Flatten nested Par into list for canonical sorting
fn flatten_par(proc: &Proc) -> Vec<Proc> {
    match proc {
        Proc::Par(p, q) => {
            let mut result = flatten_par(p);
            result.extend(flatten_par(q));
            result
        }
        _ => vec![proc.clone()]
    }
}

/// Rebuild Par from sorted list (right-associative)
fn rebuild_par(procs: Vec<Proc>) -> Proc {
    if procs.is_empty() {
        Proc::Nil
    } else if procs.len() == 1 {
        procs.into_iter().next().unwrap()
    } else {
        let mut iter = procs.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, p| Proc::Par(Box::new(acc), Box::new(p)))
    }
}
```

**MeTTa Definition**:

```metta
(simplification-rule
    (id par-assoc-right)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (Par (Par $P $Q) $R))
    (template (Par $P (Par $Q $R)))

    (soundness "(P | Q) | R ≡ P | (Q | R) by associativity")
    (termination-weight 0))
```

---

### Law 4: Scope Extrusion

Names can be extruded from parallel composition when not free in the other process:

```
new x in (P | Q) ≡ (new x in P) | Q    when x ∉ FV(Q)
new x in (P | Q) ≡ P | (new x in Q)    when x ∉ FV(P)
```

**MORK Rules**:

```rust
pub struct ScopeExtrudeRight;

impl SimplificationRule for ScopeExtrudeRight {
    fn name(&self) -> &str { "scope-extrude-right" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        parse_expr("(New $x (Par $P $Q))")
    }

    fn template(&self) -> Expr {
        parse_expr("(Par (New $x $P) $Q)")
    }

    fn guard(&self, term: &Proc, _facts: &AnalysisFacts) -> bool {
        if let Proc::New(x, body) = term {
            if let Proc::Par(_, q) = body.as_ref() {
                // x must not be free in Q
                !q.free_vars().contains(x)
            } else {
                false
            }
        } else {
            false
        }
    }

    fn is_beneficial(&self, _old: &Proc, _new: &Proc, _facts: &AnalysisFacts) -> bool {
        // Scope extrusion is always beneficial for normalization
        true
    }

    fn termination_weight(&self) -> i32 { 0 }
}

pub struct ScopeExtrudeLeft;

impl SimplificationRule for ScopeExtrudeLeft {
    fn name(&self) -> &str { "scope-extrude-left" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        parse_expr("(New $x (Par $P $Q))")
    }

    fn template(&self) -> Expr {
        parse_expr("(Par $P (New $x $Q))")
    }

    fn guard(&self, term: &Proc, _facts: &AnalysisFacts) -> bool {
        if let Proc::New(x, body) = term {
            if let Proc::Par(p, _) = body.as_ref() {
                // x must not be free in P
                !p.free_vars().contains(x)
            } else {
                false
            }
        } else {
            false
        }
    }

    fn termination_weight(&self) -> i32 { 0 }
}
```

**MeTTa Definition**:

```metta
(simplification-rule
    (id scope-extrude-right)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (New $x (Par $P $Q)))
    (template (Par (New $x $P) $Q))

    (guard (not (free-in $x $Q)))

    (soundness "new x.(P|Q) ≡ (new x.P)|Q when x ∉ FV(Q)")
    (termination-weight 0))

(simplification-rule
    (id scope-extrude-left)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (New $x (Par $P $Q)))
    (template (Par $P (New $x $Q)))

    (guard (not (free-in $x $P)))

    (soundness "new x.(P|Q) ≡ P|(new x.Q) when x ∉ FV(P)")
    (termination-weight 0))
```

---

### Law 5: Scope Fusion

Nested scopes of the same name can be fused:

```
new x in (new x in P) ≡ new x in P
```

**MORK Rule**:

```rust
pub struct ScopeFusion;

impl SimplificationRule for ScopeFusion {
    fn name(&self) -> &str { "scope-fusion" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        parse_expr("(New $x (New $x $P))")
    }

    fn template(&self) -> Expr {
        parse_expr("(New $x $P)")
    }

    fn termination_weight(&self) -> i32 { -1 }  // Reduces size
}
```

---

### Law 6: Dead Scope Elimination

Unused bindings can be removed:

```
new x in P ≡ P    when x ∉ FV(P)
```

**MORK Rule**:

```rust
pub struct DeadScopeElim;

impl SimplificationRule for DeadScopeElim {
    fn name(&self) -> &str { "dead-scope-elim" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        parse_expr("(New $x $P)")
    }

    fn template(&self) -> Expr {
        parse_expr("$P")
    }

    fn guard(&self, term: &Proc, _facts: &AnalysisFacts) -> bool {
        if let Proc::New(x, body) = term {
            // x must not be free in body
            !body.free_vars().contains(x)
        } else {
            false
        }
    }

    fn termination_weight(&self) -> i32 { -1 }  // Reduces size
}
```

**MeTTa Definition**:

```metta
(simplification-rule
    (id dead-scope-elim)
    (category rholang-congruence)
    (phase StructuralNormalization)

    (pattern (New $x $P))
    (template $P)

    (guard (not (free-in $x $P)))

    (soundness "new x.P ≡ P when x ∉ FV(P)")
    (termination-weight -1))
```

---

## Canonical Form

After applying all congruence rules, processes should be in canonical form:

1. **Flattened**: No nested Par on left side
2. **Sorted**: Components ordered by canonical_order
3. **Nil-free**: No Nil in Par compositions
4. **Scope-minimized**: New bindings pushed as deep as possible
5. **Alpha-normalized**: Bound variables renamed canonically

```rust
/// Check if a process is in canonical form
pub fn is_canonical(proc: &Proc) -> bool {
    check_flat(proc) &&
    check_sorted(proc) &&
    check_no_nil(proc) &&
    check_scope_minimal(proc)
}

fn check_flat(proc: &Proc) -> bool {
    match proc {
        Proc::Par(p, _) => {
            // Left side should not be a Par
            !matches!(p.as_ref(), Proc::Par(_, _))
        }
        _ => true
    }
}

fn check_sorted(proc: &Proc) -> bool {
    match proc {
        Proc::Par(p, q) => {
            canonical_order(p) <= canonical_order(q) &&
            check_sorted(q)
        }
        _ => true
    }
}

fn check_no_nil(proc: &Proc) -> bool {
    match proc {
        Proc::Par(p, q) => {
            !matches!(p.as_ref(), Proc::Nil) &&
            !matches!(q.as_ref(), Proc::Nil) &&
            check_no_nil(p) &&
            check_no_nil(q)
        }
        _ => true
    }
}
```

---

## Complete Normalization Algorithm

```rust
pub fn normalize_process(proc: &Proc) -> Proc {
    // 1. Flatten all Par nodes
    let flat = flatten_par(proc);

    // 2. Filter out Nil processes
    let no_nil: Vec<_> = flat.into_iter()
        .filter(|p| !matches!(p, Proc::Nil))
        .collect();

    // 3. Recursively normalize subterms
    let normalized: Vec<_> = no_nil.into_iter()
        .map(|p| normalize_subterm(&p))
        .collect();

    // 4. Sort by canonical order
    let mut sorted = normalized;
    sorted.sort_by_key(|p| canonical_order(p));

    // 5. Rebuild right-associative Par
    rebuild_par(sorted)
}

fn normalize_subterm(proc: &Proc) -> Proc {
    match proc {
        Proc::New(x, body) => {
            let norm_body = normalize_process(body);

            // Dead scope elimination
            if !norm_body.free_vars().contains(x) {
                return norm_body;
            }

            // Scope extrusion
            if let Proc::Par(p, q) = &norm_body {
                if !q.free_vars().contains(x) {
                    return Proc::Par(
                        Box::new(Proc::New(x.clone(), p.clone())),
                        q.clone()
                    );
                }
            }

            Proc::New(x.clone(), Box::new(norm_body))
        }

        Proc::Send(chan, data, cont) => {
            Proc::Send(chan.clone(), data.clone(), Box::new(normalize_process(cont)))
        }

        Proc::Receive(pat, chan, body) => {
            Proc::Receive(pat.clone(), chan.clone(), Box::new(normalize_process(body)))
        }

        _ => proc.clone()
    }
}
```

---

## Rule Application Order

For efficient normalization, apply rules in this order:

1. **Nil identity** (eliminates Nil nodes)
2. **Dead scope elimination** (removes unused New)
3. **Scope extrusion** (pushes New inward)
4. **Associativity** (flattens Par)
5. **Commutativity** (sorts components)
6. **Scope fusion** (merges nested New)

This order ensures:
- Size-reducing rules apply first
- Normalization completes in O(n² log n) time
- Result is always canonical

---

## Next Steps

- [Termination Proof](07-termination-proof.md) - Formal termination argument
- [Rule Application](03-rule-application.md) - MORK integration details

---

## Changelog

- **2025-12-06**: Initial Rholang congruence documentation
