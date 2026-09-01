# MeTTaIL: Semantic Type Checking for MeTTa

This documentation provides a rigorous, pedagogical treatment of semantic type checking
for MeTTa using MeTTaIL. It synthesizes insights from four foundational papers and two
prototype implementations to answer the question:

> **Can MeTTaIL be used for semantic type checking?**

**Answer**: Yes, with qualifications. MeTTaIL provides foundational capabilities for
semantic type checking, but realizing the full vision requires implementing Native Type
Theory (OSLF) as described in the Williams & Stay paper.

---

## Executive Summary

MeTTa's operational semantics (Meta-MeTTa paper, 2023) explicitly recommends OSLF for
deriving "a type system for MeTTa that includes spatial and behavioral types." This
documentation series explains:

1. **What semantic type checking means** for a rewriting language like MeTTa
2. **How OSLF (Native Type Theory)** provides the mathematical foundation
3. **How Gph-enriched Lawvere theories** offer a simpler alternative path
4. **How RHO calculus reflection** enables nominal feature elimination
5. **What the current prototypes implement** and what remains to be built

### Three Levels of Semantic Type Checking

| Level | Approach | Capabilities | Status |
|-------|----------|--------------|--------|
| **Basic** | Current MeTTaIL | Sort validation, constructor checking | Done |
| **Operational** | Gph-enriched Lawvere | Reduction semantics, evaluation control, resources | Medium effort |
| **Full** | OSLF/Native Types | Behavioral predicates, bisimulation, refined binding | High effort |

---

## Documentation Structure

### [Theoretical Foundations](./theoretical-foundations/README.md)

These documents explain the mathematical foundations required to understand semantic
type checking for MeTTa.

| Document | Description |
|----------|-------------|
| [01-metta-operational-semantics.md](./theoretical-foundations/01-metta-operational-semantics.md) | MeTTa as a state machine with rewrite rules |
| [02-native-type-theory-oslf.md](./theoretical-foundations/02-native-type-theory-oslf.md) | The 2-functor construction from $`\lambda`$-theories to type systems |
| [03-gph-enriched-lawvere.md](./theoretical-foundations/03-gph-enriched-lawvere.md) | Simpler semantics when binding is eliminated via reflection |
| [04-rho-calculus.md](./theoretical-foundations/04-rho-calculus.md) | Rholang's theoretical foundation and reflection mechanism |
| [05-type-lifting.md](./theoretical-foundations/05-type-lifting.md) | Deriving types from operational semantics via T(-) transformation |
| [06-inference-rules.md](./theoretical-foundations/06-inference-rules.md) | Practical guide to reading and implementing inference rules |

**Reading Order**: Start with (01) for MeTTa background. For type derivation, read (05)
and (06) next. Then either (02) for full OSLF theory or (03) for the simpler Gph-enriched
approach. Document (04) explains how RHO calculus reflection bridges to Rholang.

### [Implementation](./implementation/README.md)

These documents describe the existing prototypes and the path forward.

| Document | Description |
|----------|-------------|
| [01-mettail-scala-prototype.md](./implementation/01-mettail-scala-prototype.md) | Theory definitions, hypercube transformation, BNFC generation |
| [02-mettail-rust-prototype.md](./implementation/02-mettail-rust-prototype.md) | Category-based checking, rewrite engine, Ascent Datalog |
| [03-rholang-integration.md](./implementation/03-rholang-integration.md) | Current MeTTa-Rholang bridge via mettatron |
| [04-implementation-roadmap.md](./implementation/04-implementation-roadmap.md) | Layered approach to full semantic type checking |

### [MeTTa Ecosystem](./metta-ecosystem/README.md)

These documents describe the broader MeTTa ecosystem and implementation architectures.

| Document | Description |
|----------|-------------|
| [01-opencog-hyperon.md](./metta-ecosystem/01-opencog-hyperon.md) | OpenCog Hyperon architecture, Atomspace, four meta-types |
| [02-hyperon-experimental.md](./metta-ecosystem/02-hyperon-experimental.md) | Official MeTTa implementation details |
| [03-mettatron.md](./metta-ecosystem/03-mettatron.md) | F1R3FLY.io's MeTTaTron compiler architecture |
| [04-mork-pathmap-integration.md](./metta-ecosystem/04-mork-pathmap-integration.md) | MORK and PathMap storage layer integration |

### [Correction WFST](./correction-wfst/README.md)

These documents describe the unified three-tier correction WFST architecture that
integrates liblevenshtein with MeTTaIL for semantic type-aware correction.

| Document | Description |
|----------|-------------|
| [01-architecture-overview.md](./correction-wfst/01-architecture-overview.md) | Three-tier WFST architecture overview |
| [02-tier1-lexical-correction.md](./correction-wfst/02-tier1-lexical-correction.md) | liblevenshtein edit distance and phonetic rules |
| [03-tier2-syntactic-validation.md](./correction-wfst/03-tier2-syntactic-validation.md) | CFG validation via MORK/PathMap |
| [04-tier3-semantic-type-checking.md](./correction-wfst/04-tier3-semantic-type-checking.md) | MeTTaIL/MeTTaTron/Rholang semantic checking |
| [05-data-flow.md](./correction-wfst/05-data-flow.md) | Complete data flow through the stack |
| [06-integration-possibilities.md](./correction-wfst/06-integration-possibilities.md) | Cross-language, ASR, IDE integrations |

### [Reference](./reference/README.md)

Quick-reference materials and supporting documentation.

| Document | Description |
|----------|-------------|
| [gap-analysis.md](./reference/gap-analysis.md) | What's missing for full OSLF implementation |
| [use-cases.md](./reference/use-cases.md) | Concrete examples of semantic type checking |
| [bibliography.md](./reference/bibliography.md) | Complete reference list with annotations |

---

## Key Insight: The Binding Decision

The choice between Gph-enriched Lawvere theories and full OSLF depends on one key
question:

> **Can MeTTa's unification patterns be given a combinator representation?**

- **If yes** → Gph-theories suffice (simpler path)
- **If no** → Full OSLF required (handles binding natively)

The RHO calculus paper demonstrates that reflection can eliminate nominal features,
suggesting the simpler Gph-theory path is viable for MeTTa's reflective capabilities.

---

## Recommended Reading Path

### For Implementers

1. [01-metta-operational-semantics](./theoretical-foundations/01-metta-operational-semantics.md) - Understand the target
2. [06-inference-rules](./theoretical-foundations/06-inference-rules.md) - Learn to read type notation
3. [05-type-lifting](./theoretical-foundations/05-type-lifting.md) - Derive types from semantics
4. [03-gph-enriched-lawvere](./theoretical-foundations/03-gph-enriched-lawvere.md) - Simplest semantic model
5. [04-implementation-roadmap](./implementation/04-implementation-roadmap.md) - Concrete steps
6. [gap-analysis](./reference/gap-analysis.md) - What to build

### For Theorists

1. [02-native-type-theory-oslf](./theoretical-foundations/02-native-type-theory-oslf.md) - Full mathematical foundation
2. [05-type-lifting](./theoretical-foundations/05-type-lifting.md) - Type lifting transformation
3. [04-rho-calculus](./theoretical-foundations/04-rho-calculus.md) - Reflection theory
4. [use-cases](./reference/use-cases.md) - Applications of behavioral types

### For Project Managers

1. This README - Overview
2. [04-implementation-roadmap](./implementation/04-implementation-roadmap.md) - Phases and dependencies
3. [gap-analysis](./reference/gap-analysis.md) - Scope of remaining work

### For Correction/WFST Integration

1. [01-architecture-overview](./correction-wfst/01-architecture-overview.md) - Three-tier architecture
2. [02-tier1-lexical-correction](./correction-wfst/02-tier1-lexical-correction.md) - liblevenshtein integration
3. [04-tier3-semantic-type-checking](./correction-wfst/04-tier3-semantic-type-checking.md) - MeTTaIL as semantic layer
4. [05-data-flow](./correction-wfst/05-data-flow.md) - Complete pipeline

### For MeTTa Ecosystem Understanding

1. [01-opencog-hyperon](./metta-ecosystem/01-opencog-hyperon.md) - Theoretical foundations
2. [02-hyperon-experimental](./metta-ecosystem/02-hyperon-experimental.md) - Reference implementation
3. [03-mettatron](./metta-ecosystem/03-mettatron.md) - F1R3FLY.io compiler
4. [04-mork-pathmap-integration](./metta-ecosystem/04-mork-pathmap-integration.md) - Storage layer

---

## Source Materials

This documentation synthesizes:

### Papers

1. **Meta-MeTTa: an operational semantics for MeTTa** (Meredith et al., 2023)
2. **Native Type Theory** (Williams & Stay, ACT 2021)
3. **Representing operational semantics with enriched Lawvere theories** (Stay & Meredith, 2017)
4. **A Reflective Higher-order Calculus** (Meredith & Radestock, ENTCS 2005)
5. **OpenCog Hyperon: A Framework for AGI** (Goertzel et al., 2023)

### Implementations

- MeTTaIL Scala prototype (`/home/dylon/Workspace/f1r3fly.io/MeTTaIL/`)
- mettail-rust prototype (`/home/dylon/Workspace/f1r3fly.io/mettail-rust/`)
- Rholang f1r3node (`/home/dylon/Workspace/f1r3fly.io/f1r3node/`)
- hyperon-experimental (`/home/dylon/Workspace/f1r3fly.io/hyperon-experimental/`)
- MeTTaTron (`/home/dylon/Workspace/f1r3fly.io/MeTTa-Compiler/`)
- MORK (`/home/dylon/Workspace/f1r3fly.io/MORK/`)
- PathMap (`/home/dylon/Workspace/f1r3fly.io/PathMap/`)
- liblevenshtein (`/home/dylon/Workspace/f1r3fly.io/liblevenshtein-rust/`)

---

## Quick Reference: The OSLF 2-Functor

The core mathematical construction underlying full semantic type checking:

```
λ-theory  ──[P]──>  presheaf topos  ──[L]──>  type system
   T                    P(T)                   LP(T)
```

Where:
- **T** = MeTTa formalized as $`a \lambda`$-theory with equality
- **P** = Presheaf construction (preserves products, equality, function types)
- **L** = Internal language functor (extracts the type theory)

This construction is **native** because types arise directly from the syntax, not
imposed externally.

---

## Notation Quick Reference

Common type-theoretic notation used throughout these documents:

| Symbol | Name | Meaning | Example |
|--------|------|---------|---------|
| $`\vdash`$ | Turnstile | "derives" or "proves" | $`\Gamma  \vdash  M : A`$ means "$`\Gamma`$ proves M has type A" |
| $`\Gamma`$ | Context | Type assumptions in scope | `x: Int, y: Bool` |
| `:` | Type ascription | "has type" | `M : A` means "M has type A" |
| `→` | Function type | Functions from A to B | `A → B` |
| `×` | Product type | Pairs of A and B | `A × B` |
| `◇` | Possibility | "possibly" (modal) | `◇A` = "can become A" |
| `T(-)` | Type lifting | Transformation function | `T(A → B) = T(A) × (T(A) → T(B))` |

For detailed explanations, see [06-inference-rules.md](./theoretical-foundations/06-inference-rules.md).
