# Layer 4: Verification Layer

The verification layer validates that simplifications preserve program semantics.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Overview

The verification layer ensures that simplified programs are semantically equivalent to their originals. This provides confidence that simplifications don't introduce bugs or change program behavior.

```
Input: Original Proc + Simplified Proc
         ↓
┌──────────────────────────────────────┐
│        VERIFICATION LAYER            │
│                                      │
│  ┌────────────────────────────────┐  │
│  │     Semantic Equivalence       │  │
│  │   Structural + operational     │  │
│  └────────────────────────────────┘  │
│                 ↓                    │
│  ┌────────────────────────────────┐  │
│  │     Type Preservation          │  │
│  │   infer(orig) = infer(simp)    │  │
│  └────────────────────────────────┘  │
│                 ↓                    │
│  ┌────────────────────────────────┐  │
│  │   Behavioral Equivalence       │  │
│  │   Bisimulation (for Rholang)   │  │
│  └────────────────────────────────┘  │
│                 ↓                    │
│  ┌────────────────────────────────┐  │
│  │     Regression Testing         │  │
│  │   Property-based + golden      │  │
│  └────────────────────────────────┘  │
│                                      │
└──────────────────────────────────────┘
         ↓
Output: VerificationResult (pass/fail + evidence)
```

---

## Module Structure

```
simplification/verification/
├── mod.rs              # Verification orchestration
├── semantic_equiv.rs   # Semantic equivalence checking
├── type_preserving.rs  # Type preservation validation
├── behavioral.rs       # Behavioral equivalence (Rholang)
└── regression.rs       # Regression testing integration
```

---

## Verification Result

```rust
pub struct VerificationResult {
    /// Overall verification status
    pub status: VerificationStatus,

    /// Detailed evidence for the result
    pub evidence: VerificationEvidence,

    /// Non-fatal warnings
    pub warnings: Vec<VerificationWarning>,

    /// Time taken for verification
    pub verification_time: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerificationStatus {
    /// All checks passed
    Valid,

    /// Verification failed with reason
    Invalid(String),

    /// Could not determine (timeout or complexity)
    Inconclusive(String),
}

#[derive(Clone, Debug)]
pub enum VerificationEvidence {
    /// Type preservation evidence
    TypePreserved {
        original_type: Type,
        simplified_type: Type,
    },

    /// Structural equivalence via rule application trace
    RuleTrace {
        rules_applied: Vec<RuleApplication>,
        all_rules_verified: bool,
    },

    /// Bisimulation witness for Rholang processes
    BisimilarProcesses {
        witness: BisimulationWitness,
    },

    /// Regression tests all passed
    RegressionPassed {
        test_count: usize,
        test_names: Vec<String>,
    },

    /// Combined evidence from multiple checks
    Combined(Vec<VerificationEvidence>),
}

#[derive(Clone, Debug)]
pub struct VerificationWarning {
    pub kind: WarningKind,
    pub message: String,
    pub location: Option<ProcId>,
}

#[derive(Clone, Debug)]
pub enum WarningKind {
    /// Performance regression possible
    PerformanceRegression,

    /// Precision loss in numeric operations
    PrecisionLoss,

    /// Evaluation order may differ
    EvaluationOrderChange,

    /// Side effect timing may differ
    SideEffectTiming,
}
```

---

## Semantic Equivalence

### Structural Equivalence

For pure expressions, structural equivalence modulo known identities:

```rust
pub struct StructuralEquivalenceChecker;

impl StructuralEquivalenceChecker {
    /// Check if two terms are structurally equivalent
    pub fn check(&self, original: &Proc, simplified: &Proc) -> EquivalenceResult {
        // Normalize both terms to canonical form
        let norm_orig = self.normalize(original);
        let norm_simp = self.normalize(simplified);

        if norm_orig.content_hash() == norm_simp.content_hash() {
            return EquivalenceResult::Equivalent;
        }

        // Check modulo known algebraic identities
        if self.equivalent_modulo_identities(&norm_orig, &norm_simp) {
            return EquivalenceResult::Equivalent;
        }

        EquivalenceResult::NotEquivalent {
            reason: "Structural difference after normalization".to_string(),
        }
    }

    fn normalize(&self, proc: &Proc) -> Proc {
        // Apply canonical ordering (e.g., sort commutative operands)
        let mut normalized = proc.clone();

        // Canonicalize associative operations (flatten)
        normalized = self.flatten_associative(&normalized);

        // Canonicalize commutative operations (sort)
        normalized = self.sort_commutative(&normalized);

        // Canonicalize variable names (alpha-equivalence)
        normalized = self.alpha_normalize(&normalized);

        normalized
    }

    fn equivalent_modulo_identities(&self, a: &Proc, b: &Proc) -> bool {
        // Check known equivalences:
        // x + 0 ≡ x, x * 1 ≡ x, etc.
        let identities = [
            (|p: &Proc| matches!(p, Proc::Plus(x, Proc::Num(0))) ),
            (|p: &Proc| matches!(p, Proc::Mult(x, Proc::Num(1))) ),
            // ... more identity patterns
        ];

        // Apply identities to both sides and compare
        let a_reduced = self.apply_identities(a);
        let b_reduced = self.apply_identities(b);

        a_reduced.content_hash() == b_reduced.content_hash()
    }
}
```

### Operational Equivalence

For effectful computations, check operational semantics:

```rust
pub struct OperationalEquivalenceChecker {
    /// Maximum reduction steps
    max_steps: usize,
}

impl OperationalEquivalenceChecker {
    /// Check operational equivalence by comparing reduction sequences
    pub fn check(&self, original: &Proc, simplified: &Proc) -> EquivalenceResult {
        // For terminating programs, compare final values
        match (self.evaluate(original), self.evaluate(simplified)) {
            (Some(v1), Some(v2)) if v1 == v2 => {
                EquivalenceResult::Equivalent
            }
            (Some(v1), Some(v2)) => {
                EquivalenceResult::NotEquivalent {
                    reason: format!("Different values: {:?} vs {:?}", v1, v2),
                }
            }
            (None, None) => {
                // Both diverge or timeout - inconclusive
                EquivalenceResult::Inconclusive {
                    reason: "Both programs timeout".to_string(),
                }
            }
            _ => {
                EquivalenceResult::NotEquivalent {
                    reason: "Termination behavior differs".to_string(),
                }
            }
        }
    }

    fn evaluate(&self, proc: &Proc) -> Option<Value> {
        let mut current = proc.clone();
        let mut steps = 0;

        while steps < self.max_steps {
            if current.is_value() {
                return Some(current.to_value());
            }
            current = self.step(&current)?;
            steps += 1;
        }

        None  // Timeout
    }
}
```

---

## Type Preservation

Verify that simplification doesn't change the inferred type:

```rust
pub struct TypePreservationChecker {
    type_checker: MeTTaILTypeChecker,
}

impl TypePreservationChecker {
    /// Check that types are preserved
    pub fn check(&self, original: &Proc, simplified: &Proc) -> TypeCheckResult {
        let orig_type = self.type_checker.infer(original);
        let simp_type = self.type_checker.infer(simplified);

        match (orig_type, simp_type) {
            (Ok(t1), Ok(t2)) if self.types_compatible(&t1, &t2) => {
                TypeCheckResult::Preserved {
                    original_type: t1,
                    simplified_type: t2,
                }
            }
            (Ok(t1), Ok(t2)) => {
                TypeCheckResult::TypeChanged {
                    original: t1,
                    simplified: t2,
                    reason: "Types not compatible".to_string(),
                }
            }
            (Err(e), _) => {
                TypeCheckResult::OriginalIllTyped { error: e }
            }
            (_, Err(e)) => {
                TypeCheckResult::SimplifiedIllTyped { error: e }
            }
        }
    }

    /// Check type compatibility (allowing subtyping)
    fn types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        // Exact equality
        if t1 == t2 {
            return true;
        }

        // Subtyping: simplified can have more specific type
        if self.type_checker.is_subtype(t2, t1) {
            return true;
        }

        false
    }
}
```

### MeTTaIL Predicate Integration

```metta
; Type preservation predicate
(: type-preserving (-> Proc Proc Prop))
(= (type-preserving $original $simplified)
   (let $orig_type (infer-type $original)
        $simp_type (infer-type $simplified)
     (or (= $orig_type $simp_type)
         (subtype $simp_type $orig_type))))

; Subtype relation
(: subtype (-> Type Type Prop))
(= (subtype Int Int) True)
(= (subtype Bool Bool) True)
(= (subtype (Process $A) (Process $B))
   (subtype $A $B))
(= (subtype $T (Union $T $U)) True)
```

---

## Behavioral Equivalence (Rholang)

For Rholang processes, verify bisimulation equivalence:

```rust
pub struct BehavioralEquivalenceChecker {
    /// Maximum simulation depth
    max_depth: usize,
}

impl BehavioralEquivalenceChecker {
    /// Check behavioral equivalence via bisimulation
    pub fn check(&self, original: &Proc, simplified: &Proc) -> BehavioralResult {
        // Build labeled transition systems
        let lts_orig = self.build_lts(original);
        let lts_simp = self.build_lts(simplified);

        // Check bisimulation
        match self.check_bisimulation(&lts_orig, &lts_simp) {
            Some(witness) => {
                BehavioralResult::Bisimilar { witness }
            }
            None => {
                BehavioralResult::NotBisimilar {
                    counterexample: self.find_counterexample(&lts_orig, &lts_simp),
                }
            }
        }
    }

    /// Build labeled transition system for a process
    fn build_lts(&self, proc: &Proc) -> LabeledTransitionSystem {
        let mut lts = LabeledTransitionSystem::new(proc.clone());
        let mut frontier = vec![proc.clone()];
        let mut visited = HashSet::new();

        while let Some(current) = frontier.pop() {
            if visited.contains(&current.content_hash()) {
                continue;
            }
            visited.insert(current.content_hash());

            // Find all possible transitions
            for (label, next) in self.transitions(&current) {
                lts.add_transition(current.clone(), label, next.clone());
                frontier.push(next);
            }
        }

        lts
    }

    /// Get all possible transitions from a state
    fn transitions(&self, proc: &Proc) -> Vec<(Label, Proc)> {
        let mut transitions = Vec::new();

        match proc {
            // Send action
            Proc::Send(channel, data, continuation) => {
                transitions.push((
                    Label::Send(channel.clone()),
                    continuation.as_ref().clone(),
                ));
            }

            // Receive action
            Proc::Receive(binding, channel, body) => {
                // For each possible input value (symbolic)
                transitions.push((
                    Label::Receive(channel.clone()),
                    body.as_ref().clone(),  // With binding substituted
                ));
            }

            // Parallel composition: interleaving semantics
            Proc::Par(p, q) => {
                // Left process can move
                for (label, p_next) in self.transitions(p) {
                    transitions.push((
                        label,
                        Proc::Par(Box::new(p_next), q.clone()),
                    ));
                }
                // Right process can move
                for (label, q_next) in self.transitions(q) {
                    transitions.push((
                        label,
                        Proc::Par(p.clone(), Box::new(q_next)),
                    ));
                }
                // Communication (tau transition)
                if let Some(tau) = self.find_comm(p, q) {
                    transitions.push((Label::Tau, tau));
                }
            }

            // Nil has no transitions
            Proc::Nil => {}

            // Other cases...
            _ => {}
        }

        transitions
    }

    /// Check if two LTS are bisimilar
    fn check_bisimulation(
        &self,
        lts1: &LabeledTransitionSystem,
        lts2: &LabeledTransitionSystem,
    ) -> Option<BisimulationWitness> {
        // Partition refinement algorithm
        let mut partition = self.initial_partition(lts1, lts2);

        loop {
            let refined = self.refine(&partition, lts1, lts2);
            if refined == partition {
                break;
            }
            partition = refined;
        }

        // Check if initial states are in same partition
        if self.same_partition(&partition, &lts1.initial, &lts2.initial) {
            Some(BisimulationWitness { partition })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct BisimulationWitness {
    /// Partition of states that witnesses bisimulation
    partition: Vec<HashSet<Proc>>,
}

#[derive(Clone, Debug)]
pub enum Label {
    /// Send on channel
    Send(Channel),
    /// Receive on channel
    Receive(Channel),
    /// Internal/silent action
    Tau,
}
```

### MeTTaIL Bisimulation Predicates

```metta
; Behavioral equivalence via bisimulation
(: behaviorally-equivalent (-> Process Process Prop))
(= (behaviorally-equivalent $P $Q)
   (and (weak-bisimilar $P $Q)
        (implies (terminates $P) (terminates $Q))
        (implies (terminates $Q) (terminates $P))))

; Weak bisimulation (ignores internal steps)
(: weak-bisimilar (-> Process Process Prop))
(= (weak-bisimilar $P $Q)
   (forall (action $A)
     (and (implies (can-do $P $A $P')
                   (exists ($Q') (and (can-do-weak $Q $A $Q')
                                      (weak-bisimilar $P' $Q'))))
          (implies (can-do $Q $A $Q')
                   (exists ($P') (and (can-do-weak $P $A $P')
                                      (weak-bisimilar $P' $Q')))))))

; Weak transition (zero or more tau, one visible, zero or more tau)
(: can-do-weak (-> Process Action Process Prop))
(= (can-do-weak $P $A $Q)
   (exists ($P1 $P2)
     (and (tau-star $P $P1)
          (can-do $P1 $A $P2)
          (tau-star $P2 $Q))))
```

---

## Regression Testing

Validate simplifications against test suites:

```rust
pub struct RegressionTester {
    /// Property-based test generator
    property_tester: PropertyTester,

    /// Golden test cases
    golden_tests: Vec<GoldenTest>,
}

impl RegressionTester {
    /// Run all regression tests
    pub fn run(&self, original: &Proc, simplified: &Proc) -> RegressionResult {
        let mut results = Vec::new();

        // Property-based tests
        for prop in &self.properties {
            let result = self.property_tester.test(prop, original, simplified);
            results.push(result);
        }

        // Golden tests
        for test in &self.golden_tests {
            let result = self.run_golden(test, original, simplified);
            results.push(result);
        }

        RegressionResult::combine(results)
    }

    fn run_golden(
        &self,
        test: &GoldenTest,
        original: &Proc,
        simplified: &Proc,
    ) -> TestResult {
        // Execute with test input
        let orig_output = self.execute(original, &test.input);
        let simp_output = self.execute(simplified, &test.input);

        // Compare outputs
        if orig_output == simp_output {
            TestResult::Pass {
                name: test.name.clone(),
            }
        } else {
            TestResult::Fail {
                name: test.name.clone(),
                expected: orig_output,
                actual: simp_output,
            }
        }
    }
}

/// Properties that should hold for all simplifications
pub struct SimplificationProperty {
    /// Property name
    name: String,

    /// Property predicate
    predicate: fn(&Proc, &Proc) -> bool,
}

impl SimplificationProperty {
    pub fn standard_properties() -> Vec<Self> {
        vec![
            Self {
                name: "termination_preserved".to_string(),
                predicate: |orig, simp| {
                    // If original terminates, simplified should terminate
                    // (converse not required - simplification may remove divergence)
                    !terminates(orig) || terminates(simp)
                },
            },
            Self {
                name: "determinism_preserved".to_string(),
                predicate: |orig, simp| {
                    // If original is deterministic, simplified should be
                    !is_deterministic(orig) || is_deterministic(simp)
                },
            },
            Self {
                name: "purity_preserved".to_string(),
                predicate: |orig, simp| {
                    // If original is pure, simplified should be pure
                    !is_pure(orig) || is_pure(simp)
                },
            },
        ]
    }
}
```

---

## Complete Verification Orchestration

```rust
pub struct VerificationLayer {
    structural: StructuralEquivalenceChecker,
    operational: OperationalEquivalenceChecker,
    type_checker: TypePreservationChecker,
    behavioral: BehavioralEquivalenceChecker,
    regression: RegressionTester,
}

impl VerificationLayer {
    /// Verify that simplification is valid
    pub fn verify(
        &self,
        original: &Proc,
        simplified: &Proc,
        config: &VerificationConfig,
    ) -> VerificationResult {
        let start = Instant::now();
        let mut evidence = Vec::new();
        let mut warnings = Vec::new();

        // 1. Type preservation (fast, always run)
        let type_result = self.type_checker.check(original, simplified);
        match type_result {
            TypeCheckResult::Preserved { original_type, simplified_type } => {
                evidence.push(VerificationEvidence::TypePreserved {
                    original_type,
                    simplified_type,
                });
            }
            TypeCheckResult::TypeChanged { .. } => {
                return VerificationResult {
                    status: VerificationStatus::Invalid("Type not preserved".to_string()),
                    evidence: VerificationEvidence::Combined(evidence),
                    warnings,
                    verification_time: start.elapsed(),
                };
            }
            _ => {}
        }

        // 2. Structural equivalence (for pure terms)
        if original.is_pure() {
            match self.structural.check(original, simplified) {
                EquivalenceResult::Equivalent => {
                    // Structural equivalence is strong evidence
                    return VerificationResult {
                        status: VerificationStatus::Valid,
                        evidence: VerificationEvidence::Combined(evidence),
                        warnings,
                        verification_time: start.elapsed(),
                    };
                }
                _ => {}
            }
        }

        // 3. Behavioral equivalence (for Rholang processes)
        if config.check_behavioral && original.is_process() {
            match self.behavioral.check(original, simplified) {
                BehavioralResult::Bisimilar { witness } => {
                    evidence.push(VerificationEvidence::BisimilarProcesses { witness });
                }
                BehavioralResult::NotBisimilar { counterexample } => {
                    return VerificationResult {
                        status: VerificationStatus::Invalid(
                            format!("Not bisimilar: {:?}", counterexample)
                        ),
                        evidence: VerificationEvidence::Combined(evidence),
                        warnings,
                        verification_time: start.elapsed(),
                    };
                }
            }
        }

        // 4. Regression testing
        if config.run_regression {
            let regression_result = self.regression.run(original, simplified);
            match regression_result {
                RegressionResult::AllPassed { count, names } => {
                    evidence.push(VerificationEvidence::RegressionPassed {
                        test_count: count,
                        test_names: names,
                    });
                }
                RegressionResult::SomeFailed { failures } => {
                    return VerificationResult {
                        status: VerificationStatus::Invalid(
                            format!("Regression failures: {:?}", failures)
                        ),
                        evidence: VerificationEvidence::Combined(evidence),
                        warnings,
                        verification_time: start.elapsed(),
                    };
                }
            }
        }

        // 5. Collect warnings
        warnings.extend(self.check_for_warnings(original, simplified));

        VerificationResult {
            status: VerificationStatus::Valid,
            evidence: VerificationEvidence::Combined(evidence),
            warnings,
            verification_time: start.elapsed(),
        }
    }

    fn check_for_warnings(&self, original: &Proc, simplified: &Proc) -> Vec<VerificationWarning> {
        let mut warnings = Vec::new();

        // Check for potential performance regressions
        if simplified.estimated_cost() > original.estimated_cost() * 1.1 {
            warnings.push(VerificationWarning {
                kind: WarningKind::PerformanceRegression,
                message: "Simplified code may be slower".to_string(),
                location: None,
            });
        }

        // Check for evaluation order changes in expressions with side effects
        if !original.is_pure() && !self.same_eval_order(original, simplified) {
            warnings.push(VerificationWarning {
                kind: WarningKind::EvaluationOrderChange,
                message: "Evaluation order may differ".to_string(),
                location: None,
            });
        }

        warnings
    }
}

pub struct VerificationConfig {
    /// Check behavioral equivalence (expensive)
    pub check_behavioral: bool,

    /// Run regression tests
    pub run_regression: bool,

    /// Timeout for verification
    pub timeout: Duration,

    /// Maximum LTS depth for bisimulation
    pub max_lts_depth: usize,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            check_behavioral: true,
            run_regression: true,
            timeout: Duration::from_secs(30),
            max_lts_depth: 100,
        }
    }
}
```

---

## MeTTaIL Complete Verification Predicate

```metta
; Complete simplification validity predicate
(: simplification-valid (-> Proc Proc Prop))
(= (simplification-valid $original $simplified)
   (and
     ; Must be well-typed
     (well-typed $original)

     ; Type must be preserved
     (type-preserving $original $simplified)

     ; Behavioral equivalence (for processes)
     (implies (is-process $original)
              (behaviorally-equivalent $original $simplified))

     ; Operational equivalence (for expressions)
     (implies (is-expression $original)
              (operationally-equivalent $original $simplified))))

; Well-typedness
(: well-typed (-> Proc Prop))
(= (well-typed $P)
   (match (infer-type $P)
     (Ok $_) True
     (Err $_) False))

; Operational equivalence for terminating expressions
(: operationally-equivalent (-> Expr Expr Prop))
(= (operationally-equivalent $e1 $e2)
   (= (eval $e1) (eval $e2)))
```

---

## Next Steps

- [Rholang Congruence](06-rholang-congruence.md) - Detailed Rholang structural laws
- [Termination Proof](07-termination-proof.md) - Formal termination argument

---

## Changelog

- **2025-12-06**: Initial verification layer documentation
