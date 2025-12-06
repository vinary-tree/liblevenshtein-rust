# Layer 2: Rule Application Layer

The rule application layer applies simplification rules via MORK pattern/template matching.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Overview

This layer uses MORK's `transform_multi_multi_()` function to apply pattern/template rewriting rules. Rules are organized by category and checked against analysis facts before application.

```
Input: Proc + AnalysisFacts
         ↓
┌──────────────────────────────────────┐
│      RULE APPLICATION LAYER          │
│                                      │
│  ┌────────────────────────────────┐  │
│  │        Rule Registry           │  │
│  │  ┌──────────┐ ┌──────────┐    │  │
│  │  │Algebraic │ │ Control  │    │  │
│  │  │  Rules   │ │  Flow    │    │  │
│  │  └──────────┘ └──────────┘    │  │
│  │  ┌──────────┐ ┌──────────┐    │  │
│  │  │Type-Aware│ │ Rholang  │    │  │
│  │  │  Rules   │ │Congruence│    │  │
│  │  └──────────┘ └──────────┘    │  │
│  └────────────────────────────────┘  │
│                 ↓                    │
│  ┌────────────────────────────────┐  │
│  │   MORK transform_multi_multi_  │  │
│  │   Pattern → Template rewrite   │  │
│  └────────────────────────────────┘  │
│                                      │
└──────────────────────────────────────┘
         ↓
Output: Transformed Proc
```

---

## Module Structure

```
simplification/rules/
├── mod.rs              # Rule registry and dispatch
├── algebraic.rs        # x+0→x, x*1→x, double-neg
├── control_flow.rs     # if(true)→then, dead code
├── type_aware.rs       # Redundant cast removal
├── beta_reduction.rs   # Lambda application
├── inlining.rs         # Let binding simplification
└── rholang/
    ├── mod.rs
    ├── nil_identity.rs     # P|0 ≡ P
    ├── commutativity.rs    # P|Q ≡ Q|P (canonical)
    ├── associativity.rs    # (P|Q)|R ≡ P|(Q|R)
    └── scope_extrusion.rs  # new x in (P|Q) scope laws
```

---

## SimplificationRule Trait

```rust
/// Core trait for all simplification rules
pub trait SimplificationRule: Send + Sync {
    /// Unique identifier for this rule
    fn name(&self) -> &str;

    /// Category for grouping and phase selection
    fn category(&self) -> RuleCategory;

    /// Which phase(s) this rule applies in
    fn phase(&self) -> SimplificationPhase;

    /// MORK pattern expression (what to match)
    fn pattern(&self) -> Expr;

    /// MORK template expression (what to produce)
    fn template(&self) -> Expr;

    /// Guard predicate - returns true if rule can apply
    /// May use analysis facts for context-dependent rules
    fn guard(&self, term: &Proc, facts: &AnalysisFacts) -> bool {
        true  // Default: always applicable if pattern matches
    }

    /// Check if transformation is beneficial
    fn is_beneficial(&self, old: &Proc, new: &Proc, facts: &AnalysisFacts) -> bool {
        // Default: beneficial if new is smaller
        new.size() <= old.size()
    }

    /// Termination weight - must be negative for size-decreasing rules
    fn termination_weight(&self) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleCategory {
    Algebraic,
    ControlFlow,
    TypeAware,
    BetaReduction,
    Inlining,
    RholangCongruence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimplificationPhase {
    LocalSimplification,
    AnalysisDriven,
    StructuralNormalization,
    TypeDirected,
}
```

---

## MORK Integration

### Using transform_multi_multi_()

The core rewriting uses MORK's transformation function from `MORK/kernel/src/space.rs:1221`:

```rust
pub struct SimplificationEngine {
    space: Space,
    rules: RuleRegistry,
    analysis_facts: AnalysisFacts,
}

impl SimplificationEngine {
    /// Apply a single rule to a term
    pub fn apply_rule(
        &mut self,
        rule: &dyn SimplificationRule,
        term: &Proc,
    ) -> Option<Proc> {
        // 1. Check guard predicate
        if !rule.guard(term, &self.analysis_facts) {
            return None;
        }

        // 2. Convert term to MORK expression
        let term_expr = term.to_mork_expr();

        // 3. Use MORK's transform_multi_multi_ for pattern/template rewrite
        let (touched, changed) = self.space.transform_multi_multi_(
            rule.pattern(),
            rule.template(),
            term_expr,
        );

        if !changed {
            return None;
        }

        // 4. Extract result from space
        let new_term = self.extract_result();

        // 5. Check if transformation is beneficial
        if rule.is_beneficial(term, &new_term, &self.analysis_facts) {
            Some(new_term)
        } else {
            None
        }
    }

    /// Apply all applicable rules until fixpoint
    pub fn simplify(&mut self, term: Proc) -> Proc {
        let mut current = term;
        let mut visited = HashSet::new();

        loop {
            let hash = current.content_hash();
            if visited.contains(&hash) {
                break;  // Cycle detected - stop
            }
            visited.insert(hash);

            let mut changed = false;
            for rule in self.rules.all_rules() {
                if let Some(simplified) = self.apply_rule(rule, &current) {
                    current = simplified;
                    changed = true;
                    break;  // Restart from highest priority
                }
            }

            if !changed {
                break;  // Fixpoint reached
            }
        }

        current
    }
}
```

### Pattern/Template Format

Rules are defined using MORK S-expression patterns:

```rust
impl SimplificationRule for AddZeroRight {
    fn name(&self) -> &str { "add-zero-right" }
    fn category(&self) -> RuleCategory { RuleCategory::Algebraic }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::LocalSimplification }

    fn pattern(&self) -> Expr {
        // (Plus ?x (Num 0))
        parse_expr("(Plus $x (Num 0))")
    }

    fn template(&self) -> Expr {
        // ?x
        parse_expr("$x")
    }

    fn termination_weight(&self) -> i32 { -1 }
}
```

---

## Rule Categories

### Category 1: Algebraic Rules

```rust
// algebraic.rs

pub struct AddZeroRight;  // x + 0 → x
pub struct AddZeroLeft;   // 0 + x → x
pub struct MulOneRight;   // x * 1 → x
pub struct MulOneLeft;    // 1 * x → x
pub struct MulZeroRight;  // x * 0 → 0
pub struct MulZeroLeft;   // 0 * x → 0
pub struct DoubleNeg;     // --x → x
pub struct IdempotentAnd; // x && x → x
pub struct IdempotentOr;  // x || x → x

impl SimplificationRule for MulZeroRight {
    fn name(&self) -> &str { "mul-zero-right" }
    fn category(&self) -> RuleCategory { RuleCategory::Algebraic }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::LocalSimplification }

    fn pattern(&self) -> Expr {
        parse_expr("(Mult $x (Num 0))")
    }

    fn template(&self) -> Expr {
        parse_expr("(Num 0)")
    }

    fn termination_weight(&self) -> i32 { -2 }  // Eliminates entire subterm
}
```

### Category 2: Control Flow Rules

```rust
// control_flow.rs

pub struct IfTrue;        // if(true) P else Q → P
pub struct IfFalse;       // if(false) P else Q → Q
pub struct IfSameBranch;  // if(c) P else P → P
pub struct DeadCodeAfterReturn;  // return e; rest → return e

impl SimplificationRule for IfTrue {
    fn name(&self) -> &str { "if-true" }
    fn category(&self) -> RuleCategory { RuleCategory::ControlFlow }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::LocalSimplification }

    fn pattern(&self) -> Expr {
        parse_expr("(If (Bool true) $then $else)")
    }

    fn template(&self) -> Expr {
        parse_expr("$then")
    }

    fn termination_weight(&self) -> i32 { -3 }
}

impl SimplificationRule for DeadCodeAfterReturn {
    fn name(&self) -> &str { "dead-code-after-return" }
    fn category(&self) -> RuleCategory { RuleCategory::ControlFlow }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::AnalysisDriven }

    fn pattern(&self) -> Expr {
        parse_expr("(Seq (Return $e) $rest)")
    }

    fn template(&self) -> Expr {
        parse_expr("(Return $e)")
    }

    fn guard(&self, term: &Proc, facts: &AnalysisFacts) -> bool {
        // Only apply if rest is actually unreachable
        if let Proc::Seq(_, rest) = term {
            facts.is_dead(rest.id())
        } else {
            false
        }
    }

    fn termination_weight(&self) -> i32 { -1 }
}
```

### Category 3: Type-Aware Rules

```rust
// type_aware.rs

pub struct RedundantCast;  // (T)x where x : T → x
pub struct IdentityInto;   // x.into::<T>() where x : T → x

impl SimplificationRule for RedundantCast {
    fn name(&self) -> &str { "redundant-cast" }
    fn category(&self) -> RuleCategory { RuleCategory::TypeAware }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::TypeDirected }

    fn pattern(&self) -> Expr {
        parse_expr("(Cast $T $x)")
    }

    fn template(&self) -> Expr {
        parse_expr("$x")
    }

    fn guard(&self, term: &Proc, facts: &AnalysisFacts) -> bool {
        // Check if x already has type T
        if let Proc::Cast(target_type, inner) = term {
            facts.type_of(inner).map(|t| t == *target_type).unwrap_or(false)
        } else {
            false
        }
    }

    fn termination_weight(&self) -> i32 { -1 }
}
```

### Category 4: Rholang Congruence Rules

```rust
// rholang/nil_identity.rs

pub struct NilIdentityRight;  // P | 0 → P
pub struct NilIdentityLeft;   // 0 | P → P

impl SimplificationRule for NilIdentityRight {
    fn name(&self) -> &str { "nil-identity-right" }
    fn category(&self) -> RuleCategory { RuleCategory::RholangCongruence }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::StructuralNormalization }

    fn pattern(&self) -> Expr {
        parse_expr("(Par $P (Nil))")
    }

    fn template(&self) -> Expr {
        parse_expr("$P")
    }

    fn termination_weight(&self) -> i32 { -1 }
}
```

```rust
// rholang/commutativity.rs

pub struct ParCommute;  // P | Q → Q | P (when order(P) > order(Q))

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
        // Only apply to reach canonical form
        if let Proc::Par(p, q) = term {
            term_order(p) > term_order(q)
        } else {
            false
        }
    }

    fn termination_weight(&self) -> i32 { 0 }  // Size-preserving
}

/// Canonical term ordering for commutativity
fn term_order(proc: &Proc) -> u64 {
    proc.content_hash()  // Use hash as stable ordering
}
```

```rust
// rholang/scope_extrusion.rs

pub struct ScopeExtrusion;  // new x in (P | Q) → (new x in P) | Q when x ∉ FV(Q)

impl SimplificationRule for ScopeExtrusion {
    fn name(&self) -> &str { "scope-extrusion" }
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

    fn termination_weight(&self) -> i32 { 0 }  // Size-preserving
}
```

### Category 5: Beta/Inlining Rules

```rust
// beta_reduction.rs

pub struct BetaReduction;  // (λx.P)(Q) → P{Q/x}

impl SimplificationRule for BetaReduction {
    fn name(&self) -> &str { "beta-reduction" }
    fn category(&self) -> RuleCategory { RuleCategory::BetaReduction }
    fn phase(&self) -> SimplificationPhase { SimplificationPhase::LocalSimplification }

    fn pattern(&self) -> Expr {
        parse_expr("(App (Lambda $x $body) $arg)")
    }

    fn template(&self) -> Expr {
        // Template uses substitution
        parse_expr("(Subst $body $x $arg)")
    }

    fn guard(&self, term: &Proc, facts: &AnalysisFacts) -> bool {
        // Only inline if argument is small or used once
        if let Proc::App(Proc::Lambda(x, body), arg) = term {
            arg.size() <= 3 || body.use_count(x) <= 1
        } else {
            false
        }
    }

    fn termination_weight(&self) -> i32 { -1 }
}
```

---

## Rule Registry

```rust
pub struct RuleRegistry {
    rules: Vec<Box<dyn SimplificationRule>>,
}

impl RuleRegistry {
    pub fn new() -> Self {
        let mut registry = Self { rules: Vec::new() };

        // Register all rules
        registry.register(Box::new(AddZeroRight));
        registry.register(Box::new(AddZeroLeft));
        registry.register(Box::new(MulOneRight));
        registry.register(Box::new(MulZeroRight));
        registry.register(Box::new(DoubleNeg));
        registry.register(Box::new(IfTrue));
        registry.register(Box::new(IfFalse));
        registry.register(Box::new(DeadCodeAfterReturn));
        registry.register(Box::new(NilIdentityRight));
        registry.register(Box::new(NilIdentityLeft));
        registry.register(Box::new(ParCommute));
        registry.register(Box::new(ScopeExtrusion));
        registry.register(Box::new(BetaReduction));
        // ... more rules

        registry
    }

    pub fn register(&mut self, rule: Box<dyn SimplificationRule>) {
        self.rules.push(rule);
    }

    pub fn rules_for_phase(&self, phase: SimplificationPhase) -> Vec<&dyn SimplificationRule> {
        self.rules.iter()
            .filter(|r| r.phase() == phase)
            .map(|r| r.as_ref())
            .collect()
    }

    pub fn all_rules(&self) -> impl Iterator<Item = &dyn SimplificationRule> {
        self.rules.iter().map(|r| r.as_ref())
    }
}
```

---

## Next Steps

- [Strategy Selection](04-strategy-selection.md) - Phase ordering and conflict resolution
- [Rholang Congruence](06-rholang-congruence.md) - Detailed Rholang rules

---

## Changelog

- **2025-12-06**: Initial rule application layer documentation
