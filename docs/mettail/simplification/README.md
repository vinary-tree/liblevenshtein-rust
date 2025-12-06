# Source-to-Source Simplification Transpiler

A multi-layer simplification/optimization transpiler that processes the output of the semantic corrector, following MeTTaIL's architectural patterns.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Overview

The simplification transpiler is a **post-correction source-to-source transformer** that takes corrected programs and produces simplified, more readable output while preserving semantics. It integrates with the MeTTaIL correction pipeline as a final optimization pass.

```
Semantic Corrector Output (Corrected Proc)
         ↓
┌────────────────────────────────────────────┐
│     SIMPLIFICATION TRANSPILER              │
│                                            │
│  Layer 1: Analysis Layer (Ascent-based)    │
│  Layer 2: Rule Application Layer (MORK)    │
│  Layer 3: Strategy Selection Layer         │
│  Layer 4: Verification Layer (MeTTaIL)     │
│                                            │
└────────────────────────────────────────────┘
         ↓
Simplified Source (Same Language)
```

---

## Key Features

- **Multi-layer architecture** following MeTTaIL patterns
- **Datalog-based analysis** via Ascent for reachability, liveness, and cost
- **Pattern/template rewriting** via MORK's `transform_multi_multi_()`
- **Semantic verification** via MeTTaIL predicates
- **Rholang congruence rules** for process calculus simplification
- **Termination guarantees** with formal proof sketch

---

## Technology Integration

| Technology | Role |
|------------|------|
| **MORK** | Pattern/template rule application |
| **PathMap** | Memoization cache for simplified terms |
| **Ascent** | Datalog-based program analysis |
| **MeTTaIL** | Semantic predicate evaluation |
| **MeTTaTron** | Guard predicate execution |
| **liblevenshtein** | Pre-normalization of symbols |
| **Rholang** | Structural congruence laws |

---

## Documentation Structure

### Core Architecture

1. [Architecture Overview](01-architecture.md) - 4-layer transpiler design
2. [Analysis Layer](02-analysis-layer.md) - Ascent-based program analysis
3. [Rule Application](03-rule-application.md) - MORK pattern matching integration
4. [Strategy Selection](04-strategy-selection.md) - Phase ordering and termination
5. [Verification](05-verification.md) - Semantic preservation checks

### Specialized Topics

6. [Rholang Congruence](06-rholang-congruence.md) - Process calculus structural laws
7. [Termination Proof](07-termination-proof.md) - Formal termination argument
8. [Performance Targets](08-performance-targets.md) - Benchmarks and optimization goals

### Rule Definitions

Located in [`rules/`](rules/):

- [`algebraic.metta`](rules/algebraic.metta) - Algebraic identity rules
- [`control-flow.metta`](rules/control-flow.metta) - Control flow simplification
- [`type-aware.metta`](rules/type-aware.metta) - Type-directed rules
- [`beta-reduction.metta`](rules/beta-reduction.metta) - Lambda calculus rules
- [`rholang-congruence.metta`](rules/rholang-congruence.metta) - Rholang structural laws

---

## Quick Start

### Simplification Pipeline

```rust
// 1. Receive corrected program from semantic corrector
let corrected_proc: Proc = semantic_corrector.correct(input)?;

// 2. Run analysis pass (Ascent-based)
let analysis = AnalysisLayer::new(&theory_def);
let facts = analysis.analyze(&corrected_proc);

// 3. Apply simplification rules (MORK-based)
let engine = SimplificationEngine::new(&facts);
let simplified = engine.simplify(corrected_proc);

// 4. Verify semantic preservation (MeTTaIL-based)
let verifier = VerificationLayer::new();
verifier.check(&corrected_proc, &simplified)?;

// 5. Return simplified source
Ok(simplified)
```

### Rule Definition Format

```metta
(simplification-rule
    (id add-zero-right)
    (category algebraic)
    (phase LocalSimplification)

    (pattern (Plus ?x (Num 0)))
    (template ?x)

    (termination-weight -1))
```

---

## Simplification Phases

Rules are organized into phases that execute in order:

| Phase | Purpose | Example Rules |
|-------|---------|---------------|
| `LocalSimplification` | Fast algebraic rewrites | x+0→x, x*1→x |
| `AnalysisDriven` | Requires analysis facts | Dead code elimination |
| `StructuralNormalization` | Canonical form | Rholang congruence |
| `TypeDirected` | Requires type info | Redundant cast removal |

---

## Related Documentation

### MeTTaIL Correction Pipeline

- [Correction WFST Architecture](../correction-wfst/01-architecture-overview.md) - Full correction pipeline
- [Semantic Type Checking](../correction-wfst/04-tier3-semantic-type-checking.md) - Input to simplifier
- [Dialogue Context](../dialogue/README.md) - Context-aware corrections

### Grammar Correction Design

- [Grammar Correction README](../../design/grammar-correction/README.md) - 5-layer architecture
- [MAIN_DESIGN.md](../../design/grammar-correction/MAIN_DESIGN.md) - Complete design document

### MORK/PathMap Integration

- [MORK Integration](../../integration/mork/README.md) - Pattern matching integration
- [PathMap Integration](../../integration/pathmap/README.md) - Shared storage layer

---

## Performance Targets

| Metric | Target |
|--------|--------|
| AST size reduction | 15-30% |
| Latency | <50ms for typical programs |
| Memory overhead | <10% additional |

See [Performance Targets](08-performance-targets.md) for detailed benchmarks.

---

## Changelog

- **2025-12-06**: Initial design documentation created
