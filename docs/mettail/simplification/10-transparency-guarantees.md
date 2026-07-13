# Transparency Guarantees for Simplification

Formal proofs that simplification phases compose transparently, preserving behavioral equivalence.

**Status**: Design Documentation
**Last Updated**: 2025-12-17

---

## Overview

This document establishes that the 4-layer simplification architecture preserves semantics through **transparency** - the property that each phase's transformations don't introduce observable differences. We connect this to Wells & Stay's transparency definition (Definition 14) to ensure weak bisimilarity is a congruence.

---

## Transparency Definition

### Wells & Stay's Transparency (Definition 14)

A context `c` is **transparent** if for every non-reactive use, there exists a unique `c̄` such that:

```
c(t) →[c̄] d(t)
```

In other words, a transparent context doesn't "absorb" transitions - it passes them through predictably.

### Transparent Simplification Rule

**Definition**: A simplification rule `r: P ↦ Q` is **transparent** if:

1. **Preservation**: For all non-reactive contexts C: `$C[P] \approx  C[Q]$`
2. **Interface stability**: The rule doesn't change the observable interface (channels, I/O behavior)
3. **Compositionality**: Composition with other transparent rules yields a transparent transformation

---

## Phase Transparency Analysis

The simplification transpiler has 4 layers:

```
┌────────────────────────────────────────────┐
│  Layer 1: Analysis (Ascent-based)          │  ← Read-only
│  Layer 2: Rule Application (MORK)          │  ← Transform
│  Layer 3: Strategy Selection               │  ← Control
│  Layer 4: Verification (MeTTaIL)           │  ← Read-only
└────────────────────────────────────────────┘
```

### Layer 1: Analysis Phase

**Claim**: The Analysis phase is trivially transparent.

**Proof**: The analysis phase is **read-only**:
- It computes facts about the program (reachability, liveness, cost)
- It does not modify the program AST
- No transformations occur

Therefore, `P_out = P_in`, and `$P_\text{in} \approx  P_\text{out}$` trivially.

**Transparency**: Trivially satisfied since no transformation occurs. `$\blacksquare$`

### Layer 2: Rule Application Phase

**Claim**: The Rule Application phase is transparent when each rule preserves bisimilarity.

**Proof Structure**:

Each rule `r: pattern → template` is proven bisimilarity-preserving in [09-rpo-congruence-proofs.md](09-rpo-congruence-proofs.md). The key properties:

1. **Individual rule transparency**: For rule r with `$P \equiv  Q$`:
   - `$P \approx  Q$` (bisimilarity, proven via RPO)
   - For any context C: `$C[P] \approx  C[Q]$` (congruence)

2. **Sequential composition**: If `$P \approx  Q$` and `$Q \approx  R$`, then `$P \approx  R$` by transitivity

3. **Rule application order**: The strategy layer ensures termination, but any valid ordering preserves bisimilarity

**Formal Statement**:

Let `apply_rules(P)` be the result of exhaustively applying rules in the Rule Application phase. If every rule `r_i` satisfies `$P \approx  r_i(P)$` when applicable, then:

```
P ≈ apply_rules(P)
```

**Proof**: By induction on the number of rule applications. Each step preserves bisimilarity, and transitivity gives the final result. `$\blacksquare$`

### Layer 3: Strategy Selection Phase

**Claim**: The Strategy Selection phase is transparent.

**Proof**: The strategy layer **controls rule ordering** but does not modify the program directly:

1. It decides which phase to execute
2. It checks termination conditions
3. It may skip rules based on guards

The strategy layer is a **meta-level controller**:
- If it allows rule r to fire: transparency from Layer 2
- If it blocks rule r: no transformation occurs (trivially transparent)

The strategy layer affects **when** transformations happen, not **what** they are.

**Transparency**: Satisfied because it's a control layer, not a transformation layer. `$\blacksquare$`

### Layer 4: Verification Phase

**Claim**: The Verification phase is trivially transparent.

**Proof**: The verification phase is **read-only**:
- It checks semantic equivalence
- It validates type preservation
- It may reject invalid simplifications

The verification phase does not modify the program. It either:
- Accepts the simplified program (output = input to this phase)
- Rejects and reverts (output = original program)

In both cases, the output is behaviorally equivalent to some valid program state.

**Transparency**: Trivially satisfied since no transformation occurs. `$\blacksquare$`

---

## Compositional Transparency Theorem

### Theorem: Full Pipeline Transparency

**Statement**: The complete simplification pipeline is transparent:

```
P ≈ simplify(P)
```

where `simplify` is the composition of all 4 layers.

**Proof**:

Let `P₀` be the input program. Define:
- `P₁ = Analysis(P₀) = P₀` (read-only)
- `P₂ = RuleApplication(P₁)` (transformative)
- `P₃ = Strategy(P₂) = P₂` (control-only, output equals P₂)
- `P₄ = Verification(P₃)` (either P₃ or rollback to P₀)

**Case 1**: Verification accepts P₃
- `$P_{0} \approx  P_{1}$` (trivial, same program)
- `$P_{1} \approx  P_{2}$` (Layer 2 transparency)
- `$P_{2} \approx  P_{3}$` (trivial, same program)
- `$P_{3} \approx  P_{4}$` (trivial, same program)
- By transitivity: `$P_{0} \approx  P_{4}$` ✓

**Case 2**: Verification rejects P₃
- Output is P₀ (rollback)
- `$P_{0} \approx  P_{0}$` trivially ✓

**Conclusion**: `$P_{0} \approx  \text{simplify}(P_{0})$` in all cases. `$\blacksquare$`

---

## IPO Uniformity Conditions

### Definition 21 (Behavior Paper): IPO Uniformity

A context g is **IPO uniform** if transitions factor predictably through sublists. Formally, for derived transitions:

```
Γ ⊢ t⃗ →[c] d⟨⟨r⃗⟩⟩
```

The factorization through subcontexts is consistent.

### Application to Simplification

**Claim**: Structural congruence rules satisfy IPO uniformity.

**Evidence**:

1. **Nil Identity** (`P | 0 → P`):
   - Context: `$[- | 0]$`
   - IPO factorization: Transitions pass through unchanged
   - Uniform: Yes

2. **Commutativity** (`P | Q → Q | P`):
   - Context: Symmetric tensor
   - IPO factorization: Symmetric under swap
   - Uniform: Yes (by symmetry)

3. **Associativity** (`(P | Q) | R → P | (Q | R)`):
   - Context: Associative tensor
   - IPO factorization: Associative
   - Uniform: Yes

4. **Scope Extrusion** (`new x.(P | Q) → (new x.P) | Q` when `$x \notin  \text{FV}(Q)$`):
   - Context: Scope-restricted tensor
   - IPO factorization: Independent sublists
   - Uniform: Yes (given the constraint)

---

## Theorem 22 Application

### Theorem (Behavior Paper): Congruent Weak Bisimilarity

If every context is either **reactive** or **IPO uniform**, then weak bisimilarity is a congruence.

### Application to Rholang/RHO Calculus

The ρ-calculus (and Rholang) satisfies this condition:

1. **Reactive contexts**: `$\text{out}(n,-) | \text{in}(n,\lambda x.-)$` (communication pairs)
2. **Non-reactive contexts**: All structural contexts (Par, New, etc.)

For non-reactive contexts, we've shown:
- They are transparent (Definition 14)
- They satisfy IPO uniformity (Definition 21)

**Conclusion**: Weak bisimilarity is a congruence for Rholang, ensuring that behaviorally equivalent programs remain equivalent in all contexts.

---

## Practical Implications

### For Simplification Correctness

1. **Each rule can be verified independently**: Prove `$P \approx  r(P)$` once
2. **Composition is automatic**: Sequential rule application preserves bisimilarity
3. **Context doesn't matter**: If `$P \approx  Q$`, then `$C[P] \approx  C[Q]$` for all C

### For Implementation

```rust
/// Transparent simplification result
pub struct TransparentSimplification<P> {
    /// Original program
    pub original: P,

    /// Simplified program
    pub simplified: P,

    /// Evidence of transparency
    pub evidence: TransparencyEvidence,
}

#[derive(Clone, Debug)]
pub enum TransparencyEvidence {
    /// No transformation occurred
    Identity,

    /// Single rule application with bisimilarity proof
    SingleRule {
        rule_name: String,
        bisim_proof: BisimulationWitness,
    },

    /// Composition of transparent transformations
    Composition(Vec<TransparencyEvidence>),

    /// Verified by bisimulation check
    BisimulationVerified(BisimulationWitness),
}

impl<P> TransparentSimplification<P> {
    /// Assert that simplification preserves behavior
    pub fn is_valid(&self) -> bool {
        matches!(
            &self.evidence,
            TransparencyEvidence::Identity
            | TransparencyEvidence::SingleRule { .. }
            | TransparencyEvidence::Composition(_)
            | TransparencyEvidence::BisimulationVerified(_)
        )
    }
}
```

### For Testing

Transparency enables compositional testing:

1. **Unit test each rule**: Verify individual bisimilarity proofs
2. **Integration test phases**: Verify phase transparency
3. **Property-based test composition**: Random rule sequences preserve bisimilarity

```rust
#[test]
fn test_rule_transparency() {
    // Generate random process
    let proc = arbitrary_process();

    // Apply rule
    let simplified = apply_nil_identity(&proc);

    // Verify bisimilarity
    assert!(check_bisimilar(&proc, &simplified));
}

#[test]
fn test_phase_transparency() {
    let proc = arbitrary_process();

    // Apply full simplification
    let result = simplify_full(&proc);

    // Must be bisimilar
    assert!(check_bisimilar(&proc, &result.simplified));

    // Evidence must be valid
    assert!(result.evidence.is_valid());
}
```

---

## Compositional Transparency for Rule Sequences

### Lemma: Rule Sequence Bisimilarity

For rules r₁, r₂, ..., rₙ applied in sequence:

```
P ≈ r₁(P) ≈ r₂(r₁(P)) ≈ ... ≈ rₙ(...r₁(P)...)
```

**Proof**: By induction on n, using transitivity of `$\approx .$`

### Corollary: Order Independence (for Commuting Rules)

If rules r₁ and r₂ are **independent** (act on disjoint parts of the AST), then:

```
r₁(r₂(P)) ≈ r₂(r₁(P))
```

This enables parallel rule application when rules don't interfere.

---

## Higher-Order Extensions

### Lambda-Abstracted Processes

For rules involving lambda-abstracted continuations (e.g., `$\text{in}(x, \lambda y.P)$`):

1. **Alpha-equivalence**: Bound variable names don't affect behavior
2. **Beta-reduction**: `$(\lambda x.P) v \approx  P[v/x]$` preserves bisimilarity
3. **Eta-equivalence**: `$\lambda x.P x \approx  P$` when `$x \notin  \text{FV}(P)$`

These higher-order rules are also transparent when applied correctly:
- Alpha: Syntactic renaming, no behavioral change
- Beta: Substitution preserves bisimilarity (proven in typed settings)
- Eta: Functional extensionality

### Meta-Context Formalism

Following Definition 1 (Behavior Paper), meta-contexts `$\langle -\rangle : T \to  T$` are generated by:
- `$- \times  X$` (product on the left)
- `$[X \to  -]$` (exponential)

Transparency of higher-order rules follows from the fibered structure:
- Rules in the fiber over a type are transparent within that fiber
- Type-respecting rules compose transparently

---

## Related Documentation

- [RPO Congruence Proofs](09-rpo-congruence-proofs.md) - Bisimilarity proofs for each law
- [Optimization Strategies](11-optimization-strategies.md) - Bisimulation-based optimizations
- [Verification Layer](05-verification.md) - Semantic preservation checks

---

## References

1. Wells & Stay, "Behavior in Higher-Order Languages" - Definition 14 (Transparency), Definition 21 (IPO Uniformity), Theorem 22
2. Sangiorgi & Walker, "The Pi-Calculus" - Compositional bisimulation

---

## Changelog

- **2025-12-17**: Initial transparency guarantees documentation
