# Tier 3: Semantic Type Checking

This document describes Tier 3 of the correction WFST architecture: semantic
type checking via MeTTaIL, MeTTaTron, and Rholang.

**Sources**:
- MeTTaIL: `/home/dylon/Workspace/f1r3fly.io/MeTTaIL/`
- MeTTaTron: `/home/dylon/Workspace/f1r3fly.io/MeTTa-Compiler/`
- Rholang: `/home/dylon/Workspace/f1r3fly.io/f1r3node/`

---

## Table of Contents

1. [Overview](#overview)
2. [Semantic Type Checking Rationale](#semantic-type-checking-rationale)
3. [MeTTaIL Predicate Definitions](#mettail-predicate-definitions)
4. [MeTTaTron Execution](#mettatron-execution)
5. [Rholang Behavioral Verification](#rholang-behavioral-verification)
6. [Integration Flow](#integration-flow)
7. [Gradual Typing Support](#gradual-typing-support)
8. [Error Message Generation](#error-message-generation)

---

## Overview

Tier 3 validates semantic correctness using type predicates and behavioral
verification, ensuring corrections are not just syntactically valid but
semantically meaningful.

```
┌─────────────────────────────────────────────────────────────────┐
│                Tier 3: Semantic Type Checking                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Syntactically Valid Candidates from Tier 2              │
│  ┌─────┐  ┌─────┐                                              │
│  │ the │  │ tea │                                              │
│  └──┬──┘  └──┬──┘                                              │
│     │        │                                                   │
│     ▼        ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    MeTTaIL Predicates                       ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │ (: well-typed (-> Term Prop))                       │   ││
│  │  │ (: terminates (-> Process Prop))                    │   ││
│  │  │ (: valid-correction (-> Term Term Prop))            │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 MeTTaTron Type Checker                      ││
│  │  • Pattern matching via MORK Space                         ││
│  │  • Type predicate evaluation                               ││
│  │  • Behavioral property verification                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Rholang Bridge                             ││
│  │  PathMap <-> MeTTa State <-> Rholang Par                   ││
│  │  (Cross-language behavioral verification)                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  Output: Semantically Valid Corrections                         │
│  ┌─────┐                                                        │
│  │ the │  (type-checked as determiner in context)              │
│  └─────┘                                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Semantic Type Checking Rationale

### Beyond Syntax

Syntactic validity is necessary but not sufficient:

| Error Type | Syntactic | Semantic |
|------------|-----------|----------|
| `int x = "hello";` | Valid | Type mismatch |
| `null.method()` | Valid | Null dereference |
| `f(x, x) where f linear` | Valid | Linearity violation |
| `@channel!(data)` on wrong namespace | Valid | Namespace violation |

### OSLF Foundation

Semantic types are derived from OSLF (Operational Semantics as Logical Framework):

```
λ-theory →[P]→ presheaf topos →[L]→ type system
```

This construction provides:
1. **Structural predicates** on term constructors
2. **Behavioral predicates** on reduction sequences
3. **Combined reasoning** for semantic properties

---

## MeTTaIL Predicate Definitions

### Core Type Predicates

```metta
; Well-typedness predicate
(: well-typed (-> Term Prop))
(= (well-typed $t)
   (match &types
     (: $t $type)
     True))

; Type consistency predicate
(: consistent-types (-> Term Term Prop))
(= (consistent-types $t1 $t2)
   (and (well-typed $t1)
        (well-typed $t2)
        (unify-types (type-of $t1) (type-of $t2))))

; Valid correction predicate
(: valid-correction (-> Term Term Prop))
(= (valid-correction $original $corrected)
   (and (syntactically-valid $corrected)
        (consistent-types $original $corrected)
        (semantically-equivalent $original $corrected)))
```

### Behavioral Predicates (OSLF)

From Native Type Theory, behavioral predicates reason about computation:

```metta
; Termination predicate: eventually reaches normal form
; ◇(NormalForm) where ◇φ := μX. φ ∨ F!(X)
(: terminates (-> Process Prop))
(= (terminates $p)
   (eventually (normal-form $p)))

; Liveness predicate: can always eventually act
; □(◇(CanAct)) where □φ := νX. φ ∧ F*(X)
(: live (-> Process Prop))
(= (live $p)
   (always (eventually (can-act $p))))

; Namespace isolation: only communicates on allowed channels
(: namespace-isolated (-> Process Namespace Prop))
(= (namespace-isolated $p $ns)
   (always
     (forall $ch
       (implies (communicates $p $ch)
                (in-namespace $ch $ns)))))
```

### Modality Operators

The behavioral modalities from OSLF:

| Modality | Symbol | Meaning |
|----------|--------|---------|
| Eventually | ◇`$\varphi$` | `$\varphi$` holds at some reachable state |
| Always | □`$\varphi$` | `$\varphi$` holds at all reachable states |
| Can step | F!`$(\varphi )$` |  Exists transition to state where `$\varphi$` | 
| Must step  | `$F*(\varphi )$` |  All transitions lead to state where `$\varphi$` | 

```metta
; Eventually: least fixed point of possibility
(: eventually (-> Prop Prop))
(= (eventually $phi)
   (mu $X (or $phi (can-step $X))))

; Always: greatest fixed point of necessity
(: always (-> Prop Prop))
(= (always $phi)
   (nu $X (and $phi (must-step $X))))

; Possibility modality
(: can-step (-> Prop Prop))
(= (can-step $phi)
   (exists $state
     (and (transition current-state $state)
          (holds $phi $state))))

; Necessity modality
(: must-step (-> Prop Prop))
(= (must-step $phi)
   (forall $state
     (implies (transition current-state $state)
              (holds $phi $state))))
```

---

## MeTTaTron Execution

MeTTaTron evaluates type predicates via pattern matching in MORK Space:

### Type Checking Pipeline

```rust
/// Type checker for correction candidates
pub struct TypeChecker {
    /// MORK Space with type information
    space: Space,
    /// Predicate evaluator
    evaluator: PredicateEvaluator,
}

impl TypeChecker {
    /// Check if candidate is semantically valid correction
    pub fn check_correction(
        &self,
        original: &Term,
        candidate: &Term,
    ) -> Result<TypedCorrection, TypeError> {
        // Step 1: Check structural validity
        let struct_valid = self.check_structural(candidate)?;

        // Step 2: Check type consistency
        let type_consistent = self.check_type_consistency(original, candidate)?;

        // Step 3: Check behavioral predicates
        let behavior_valid = self.check_behavioral(candidate)?;

        Ok(TypedCorrection {
            candidate: candidate.clone(),
            confidence: self.compute_confidence(&struct_valid, &type_consistent, &behavior_valid),
        })
    }

    /// Check structural properties via MORK pattern matching
    fn check_structural(&self, term: &Term) -> Result<StructuralCheck, TypeError> {
        let pattern = term_to_pattern(term);
        let matches = self.space.match_pattern(&pattern)?;

        if matches.is_empty() {
            Err(TypeError::StructurallyInvalid(term.clone()))
        } else {
            Ok(StructuralCheck { matches })
        }
    }

    /// Check type consistency between original and corrected
    fn check_type_consistency(
        &self,
        original: &Term,
        candidate: &Term,
    ) -> Result<TypeCheck, TypeError> {
        let orig_type = self.infer_type(original)?;
        let cand_type = self.infer_type(candidate)?;

        if self.unifies(&orig_type, &cand_type) {
            Ok(TypeCheck { expected: orig_type, found: cand_type })
        } else {
            Err(TypeError::TypeMismatch {
                expected: orig_type,
                found: cand_type,
            })
        }
    }

    /// Check behavioral predicates
    fn check_behavioral(&self, term: &Term) -> Result<BehavioralCheck, TypeError> {
        // Evaluate behavioral predicates
        let terminates = self.evaluator.eval_predicate("terminates", term)?;
        let safe = self.evaluator.eval_predicate("safe", term)?;

        if terminates && safe {
            Ok(BehavioralCheck { terminates, safe })
        } else {
            Err(TypeError::BehavioralViolation {
                terminates,
                safe,
            })
        }
    }
}
```

### Predicate Evaluation

```rust
/// Evaluate OSLF predicates
pub struct PredicateEvaluator {
    /// Predicate definitions
    predicates: HashMap<String, Predicate>,
    /// Reduction engine for behavioral modalities
    reducer: ReductionEngine,
}

impl PredicateEvaluator {
    /// Evaluate predicate on term
    pub fn eval_predicate(&self, name: &str, term: &Term) -> Result<bool, EvalError> {
        let predicate = self.predicates.get(name)
            .ok_or(EvalError::UnknownPredicate(name.to_string()))?;

        self.eval(predicate, term)
    }

    /// Evaluate predicate expression
    fn eval(&self, pred: &Predicate, term: &Term) -> Result<bool, EvalError> {
        match pred {
            Predicate::And(a, b) => {
                Ok(self.eval(a, term)? && self.eval(b, term)?)
            }
            Predicate::Or(a, b) => {
                Ok(self.eval(a, term)? || self.eval(b, term)?)
            }
            Predicate::Eventually(inner) => {
                self.eval_eventually(inner, term)
            }
            Predicate::Always(inner) => {
                self.eval_always(inner, term)
            }
            Predicate::WellTyped => {
                self.check_well_typed(term)
            }
            Predicate::NormalForm => {
                Ok(!self.reducer.can_reduce(term))
            }
            // ... other cases
        }
    }

    /// Evaluate eventually modality (bounded exploration)
    fn eval_eventually(&self, inner: &Predicate, term: &Term) -> Result<bool, EvalError> {
        let mut visited = HashSet::new();
        let mut frontier = vec![term.clone()];

        while let Some(current) = frontier.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            // Check if inner predicate holds
            if self.eval(inner, &current)? {
                return Ok(true);
            }

            // Explore reductions
            for next in self.reducer.reduce_all(&current) {
                frontier.push(next);
            }
        }

        Ok(false)
    }
}
```

---

## Rholang Behavioral Verification

For Rholang code, additional behavioral properties can be verified:

### Namespace Isolation

```rholang
// Annotate contract with namespace restriction
@isolated(internal)
contract worker(internal: Namespace, external: Namespace) = {
  // Valid: communicating within allowed namespace
  new ch in {
    internal.ch!(42) | for(@x <- internal.ch) { ... }
  }

  // Invalid: would violate @isolated(internal)
  // external.leak!(secret)  // TYPE ERROR
}
```

### Behavioral Type Verification

```rust
/// Verify Rholang behavioral types
pub fn verify_rholang_behavior(
    proc: &Proc,
    annotations: &[Annotation],
) -> Result<(), BehaviorError> {
    for annotation in annotations {
        match annotation {
            Annotation::Isolated(namespace) => {
                check_isolation(proc, namespace)?;
            }
            Annotation::Terminating => {
                check_termination(proc)?;
            }
            Annotation::Safe => {
                check_safety(proc)?;
            }
            Annotation::Bisimilar(other) => {
                check_bisimilar(proc, other)?;
            }
        }
    }
    Ok(())
}

/// Check namespace isolation
fn check_isolation(proc: &Proc, namespace: &Namespace) -> Result<(), BehaviorError> {
    let free_names = proc.free_names();
    for name in free_names {
        if !namespace.contains(&name) {
            return Err(BehaviorError::NamespaceViolation {
                name,
                allowed: namespace.clone(),
            });
        }
    }
    Ok(())
}
```

### Bisimulation Checking

```rust
/// Check if two processes are bisimilar
fn check_bisimilar(p: &Proc, q: &Proc) -> Result<(), BehaviorError> {
    // Compute bisimulation relation via fixed point
    let relation = compute_bisimulation(p, q);

    if relation.contains(&(p.clone(), q.clone())) {
        Ok(())
    } else {
        // Find distinguishing context
        let context = find_distinguishing_context(p, q);
        Err(BehaviorError::NotBisimilar { context })
    }
}

/// Compute bisimulation as greatest fixed point
fn compute_bisimulation(p: &Proc, q: &Proc) -> Relation<Proc> {
    let mut relation = Relation::full(); // Start with all pairs

    loop {
        let refined = simulation_step(&relation);
        if refined == relation {
            break;
        }
        relation = refined;
    }

    relation
}
```

---

## Integration Flow

### PathMap ↔ MeTTa State ↔ Rholang Par

```rust
/// Convert syntactically valid candidates through type checking
pub fn type_check_candidates<W: Semiring>(
    candidates: &CandidateLattice<W>,
    type_checker: &TypeChecker,
) -> Result<CandidateLattice<W>, TypeError> {
    let mut valid_edges = Vec::new();

    for edge in candidates.edges() {
        // Convert candidate to MeTTa term
        let term = bytes_to_metta_term(&edge.candidate)?;

        // Type check via MeTTaTron
        match type_checker.check_correction(&original_term, &term) {
            Ok(typed) => {
                // Optionally verify in Rholang
                if let Some(rholang_check) = &type_checker.rholang_verifier {
                    let par = metta_state_to_pathmap_par(&term.to_state())?;
                    rholang_check.verify(&par)?;
                }

                // Weight by type confidence
                let typed_weight = edge.weight.mul(&W::from_f64(typed.confidence));
                valid_edges.push(LatticeEdge {
                    weight: typed_weight,
                    ..edge.clone()
                });
            }
            Err(_) => {
                // Filter out type-invalid candidates
                continue;
            }
        }
    }

    Ok(CandidateLattice::from_edges(valid_edges))
}
```

### Cross-Language Type Checking

```rust
/// Cross-language type checking flow
pub struct CrossLanguageChecker {
    metta_checker: MeTTaTronChecker,
    rholang_checker: RholangChecker,
    pathmap: PathMap,
}

impl CrossLanguageChecker {
    /// Check candidate in both MeTTa and Rholang contexts
    pub fn check_cross_language(
        &self,
        candidate: &Term,
    ) -> Result<CrossLanguageResult, TypeError> {
        // Check in MeTTa
        let metta_result = self.metta_checker.check(candidate)?;

        // Convert to Rholang via PathMap
        let metta_state = candidate.to_metta_state();
        let par = metta_state_to_pathmap_par(&metta_state)?;

        // Check in Rholang
        let rholang_result = self.rholang_checker.check(&par)?;

        Ok(CrossLanguageResult {
            metta: metta_result,
            rholang: rholang_result,
            consistent: self.check_consistency(&metta_result, &rholang_result),
        })
    }
}
```

---

## Gradual Typing Support

Support incremental typing for legacy codebases:

### Typing Modes

```rust
/// Typing strictness mode
pub enum TypingMode {
    /// No type checking
    Off,
    /// Check only annotated code
    Gradual,
    /// Infer types where possible
    Infer,
    /// Require all types
    Full,
}

impl TypeChecker {
    /// Check with specified typing mode
    pub fn check_with_mode(
        &self,
        term: &Term,
        mode: TypingMode,
    ) -> Result<TypeResult, TypeError> {
        match mode {
            TypingMode::Off => Ok(TypeResult::unchecked()),
            TypingMode::Gradual => self.check_annotated_only(term),
            TypingMode::Infer => self.check_with_inference(term),
            TypingMode::Full => self.check_strict(term),
        }
    }
}
```

### Dynamic Type Integration

```metta
; Dynamic type for untyped code
(: Dynamic Type)

; Gradual typing rules
(= (type-of $x) Dynamic)  ; Default when no annotation
(= (consistent Dynamic $t) True)  ; Dynamic consistent with any type
(= (consistent $t Dynamic) True)
```

---

## Error Message Generation

### Structured Error Messages

```rust
/// Type error with context for correction
pub struct TypeError {
    /// Error kind
    pub kind: TypeErrorKind,
    /// Source location
    pub span: Span,
    /// Expected type (if applicable)
    pub expected: Option<Type>,
    /// Found type
    pub found: Option<Type>,
    /// Suggested corrections
    pub suggestions: Vec<Suggestion>,
}

impl TypeError {
    /// Generate human-readable error message
    pub fn format(&self) -> String {
        match &self.kind {
            TypeErrorKind::Mismatch => {
                format!(
                    "Type mismatch at {}:\n  expected: {}\n  found: {}",
                    self.span,
                    self.expected.as_ref().map(|t| t.to_string()).unwrap_or_default(),
                    self.found.as_ref().map(|t| t.to_string()).unwrap_or_default(),
                )
            }
            TypeErrorKind::NamespaceViolation { name, namespace } => {
                format!(
                    "Namespace violation at {}:\n  {} not in namespace {}",
                    self.span, name, namespace
                )
            }
            TypeErrorKind::BehavioralViolation { property } => {
                format!(
                    "Behavioral property violation at {}:\n  {} does not hold",
                    self.span, property
                )
            }
        }
    }

    /// Generate correction suggestions
    pub fn suggest_corrections(&self, context: &Context) -> Vec<Suggestion> {
        // Use Tier 1 and 2 to find syntactically valid alternatives
        // that would satisfy the type requirements
        let candidates = context.find_alternatives(&self.span);
        candidates.into_iter()
            .filter(|c| self.would_fix(c))
            .take(3)
            .collect()
    }
}
```

---

## Summary

Tier 3 provides:

1. **MeTTaIL Predicates**: Structural and behavioral type specifications
2. **MeTTaTron Execution**: Pattern-based type checking via MORK
3. **Rholang Verification**: Behavioral property checking
4. **Integration Flow**: PathMap ↔ MeTTa ↔ Rholang conversion
5. **Gradual Typing**: Incremental adoption for legacy code
6. **Error Messages**: Structured errors with suggestions

Tier 3 ensures corrections are semantically meaningful, not just syntactically
valid, enabling high-precision correction for programming languages.

---

## References

- MeTTaIL: `/home/dylon/Workspace/f1r3fly.io/MeTTaIL/`
- MeTTaTron: `/home/dylon/Workspace/f1r3fly.io/MeTTa-Compiler/`
- f1r3node: `/home/dylon/Workspace/f1r3fly.io/f1r3node/`
- See [02-native-type-theory-oslf.md](../theoretical-foundations/02-native-type-theory-oslf.md) for OSLF theory
- See [05-data-flow.md](./05-data-flow.md) for complete pipeline
- See [bibliography.md](../reference/bibliography.md) for complete references
