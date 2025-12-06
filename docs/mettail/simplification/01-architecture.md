# Simplification Transpiler Architecture

This document describes the 4-layer architecture of the source-to-source simplification transpiler.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Architecture Overview

The simplification transpiler follows MeTTaIL's layered architecture pattern, with each layer consuming the output of the previous layer:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SIMPLIFICATION TRANSPILER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Corrected Proc + TheoryDef                                          │
│         ↓                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 1: ANALYSIS                                   │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │ Reachability │ │  Liveness    │ │  Dataflow    │ │    Cost      │  │ │
│  │  │  Analysis    │ │  Analysis    │ │  Analysis    │ │  Estimation  │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  │                                                                        │ │
│  │  Technology: Ascent datalog                                            │ │
│  │  Output: AnalysisFacts { reachable, live, constant_value, dead_code }  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│         ↓                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 2: RULE APPLICATION                           │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │  Algebraic   │ │ Control Flow │ │  Type-Aware  │ │   Rholang    │  │ │
│  │  │    Rules     │ │    Rules     │ │    Rules     │ │  Congruence  │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  │                                                                        │ │
│  │  Technology: MORK transform_multi_multi_()                             │ │
│  │  Output: Sequence of applicable transformations                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│         ↓                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 3: STRATEGY SELECTION                         │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │   Priority   │ │ Termination  │ │   Conflict   │ │   Metrics    │  │ │
│  │  │   Ordering   │ │   Control    │ │  Resolution  │ │   Tracking   │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  │                                                                        │ │
│  │  Technology: Custom strategy engine                                    │ │
│  │  Output: Ordered transformation schedule                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│         ↓                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 4: VERIFICATION                               │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │ │
│  │  │   Semantic   │ │     Type     │ │  Behavioral  │ │  Regression  │  │ │
│  │  │ Equivalence  │ │ Preservation │ │  Equivalence │ │   Testing    │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │ │
│  │                                                                        │ │
│  │  Technology: MeTTaIL predicates                                        │ │
│  │  Output: Verification result (pass/fail + evidence)                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│         ↓                                                                    │
│  Output: Simplified Proc                                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### Layer 1: Analysis Layer

**Purpose**: Compute program properties needed by simplification rules.

**Input**: Corrected `Proc` term + `TheoryDef` context
**Output**: `AnalysisFacts` structure containing derived facts

**Module structure** (following MeTTaIL's `ascent/` pattern):
```
simplification/analysis/
├── mod.rs              # Analysis orchestration
├── relations.rs        # Ascent relation declarations
├── reachability.rs     # What code is reachable from entry
├── liveness.rs         # What variables are live at each point
├── dataflow.rs         # Value propagation (constant folding)
└── cost.rs             # Static cost estimation for decisions
```

**Key interfaces**:
```rust
pub struct AnalysisLayer {
    theory: TheoryDef,
    ascent_program: AscentProgram,
}

pub struct AnalysisFacts {
    pub reachable: HashSet<ProcId>,
    pub live: HashMap<Var, HashSet<ProcId>>,
    pub constant_values: HashMap<Expr, Value>,
    pub dead_code: HashSet<ProcId>,
    pub cost_estimates: HashMap<ProcId, Cost>,
}

impl AnalysisLayer {
    pub fn analyze(&self, proc: &Proc) -> AnalysisFacts;
}
```

See [Analysis Layer](02-analysis-layer.md) for details.

---

### Layer 2: Rule Application Layer

**Purpose**: Apply simplification rules via pattern/template matching.

**Input**: `AnalysisFacts` + `Proc` term
**Output**: Sequence of applicable `Transformation`s

**Module structure** (following MeTTaIL's `codegen/` pattern):
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
    ├── commutativity.rs    # P|Q ≡ Q|P
    ├── associativity.rs    # (P|Q)|R ≡ P|(Q|R)
    └── scope_extrusion.rs  # new x in (P|Q) scope laws
```

**Key interfaces**:
```rust
pub trait SimplificationRule {
    fn name(&self) -> &str;
    fn category(&self) -> RuleCategory;
    fn phase(&self) -> SimplificationPhase;

    // MORK pattern/template expressions
    fn pattern(&self) -> Expr;
    fn template(&self) -> Expr;

    // Guard predicate (may use analysis facts)
    fn guard(&self, term: &Proc, facts: &AnalysisFacts) -> bool;

    // Cost/benefit check
    fn is_beneficial(&self, old: &Proc, new: &Proc, facts: &AnalysisFacts) -> bool;

    // Termination weight (must decrease for termination guarantee)
    fn termination_weight(&self) -> i32;
}

pub struct RuleRegistry {
    rules: Vec<Box<dyn SimplificationRule>>,
}

impl RuleRegistry {
    pub fn rules_for_phase(&self, phase: SimplificationPhase) -> Vec<&dyn SimplificationRule>;
}
```

See [Rule Application](03-rule-application.md) for details.

---

### Layer 3: Strategy Selection Layer

**Purpose**: Control rule application order, termination, and conflict resolution.

**Input**: Available rules + current term
**Output**: Ordered transformation schedule

**Module structure**:
```
simplification/strategies/
├── mod.rs              # Strategy orchestration
├── priority.rs         # Rule priority ordering
├── termination.rs      # Fixpoint detection, cycle prevention
├── conflict.rs         # Resolve multiple applicable rules
└── metrics.rs          # Track optimization effectiveness
```

**Key interfaces**:
```rust
pub enum SimplificationPhase {
    /// Fast local rewrites (algebraic identities)
    LocalSimplification,
    /// Requires analysis facts (dead code, constant folding)
    AnalysisDriven,
    /// Rholang structural normalization (canonical form)
    StructuralNormalization,
    /// Requires type information
    TypeDirected,
}

pub struct StrategyContext {
    phase: SimplificationPhase,
    iteration: usize,
    max_iterations: usize,
    visited: HashSet<u64>,  // Term hashes for cycle detection
}

pub struct TransformationSchedule {
    transforms: Vec<ScheduledTransform>,
}

impl StrategyContext {
    pub fn select_next(&mut self, term: &Proc, rules: &[&dyn SimplificationRule])
        -> Option<&dyn SimplificationRule>;
    pub fn record_application(&mut self, rule: &dyn SimplificationRule, old: &Proc, new: &Proc);
    pub fn should_continue(&self) -> bool;
}
```

See [Strategy Selection](04-strategy-selection.md) for details.

---

### Layer 4: Verification Layer

**Purpose**: Validate that simplifications preserve semantics.

**Input**: Original term + simplified term
**Output**: Verification result with evidence

**Module structure** (following MeTTaIL's `validator.rs` pattern):
```
simplification/verification/
├── mod.rs              # Verification orchestration
├── semantic_equiv.rs   # Semantic equivalence checking
├── type_preserving.rs  # Type preservation validation
├── behavioral.rs       # Behavioral equivalence (Rholang)
└── regression.rs       # Regression testing integration
```

**Key interfaces**:
```rust
pub struct VerificationResult {
    pub valid: bool,
    pub evidence: VerificationEvidence,
    pub warnings: Vec<VerificationWarning>,
}

pub enum VerificationEvidence {
    TypePreserved { original_type: Type, simplified_type: Type },
    BisimilarProcesses { witness: BisimulationWitness },
    RegressionPassed { test_count: usize },
}

pub struct VerificationLayer {
    type_checker: MeTTaILTypeChecker,
    behavioral_checker: BehavioralEquivalenceChecker,
}

impl VerificationLayer {
    pub fn verify(&self, original: &Proc, simplified: &Proc) -> VerificationResult;
}
```

See [Verification](05-verification.md) for details.

---

## Data Flow Summary

```
Proc (corrected)
    │
    ├──► Layer 1: Analysis
    │    ├── Input: Proc, TheoryDef
    │    ├── Process: Run Ascent datalog program
    │    └── Output: AnalysisFacts
    │
    ├──► Layer 2: Rule Application
    │    ├── Input: Proc, AnalysisFacts
    │    ├── Process: Match patterns, apply templates via MORK
    │    └── Output: List of applicable transformations
    │
    ├──► Layer 3: Strategy Selection
    │    ├── Input: Proc, transformations, AnalysisFacts
    │    ├── Process: Order by phase/priority, apply iteratively
    │    └── Output: Simplified Proc
    │
    └──► Layer 4: Verification
         ├── Input: Original Proc, Simplified Proc
         ├── Process: Check type/semantic/behavioral preservation
         └── Output: VerificationResult
                     │
                     ▼
              Simplified Proc (verified)
```

---

## Integration with MeTTaIL Pipeline

The simplification transpiler fits into the MeTTaIL correction pipeline as a post-processing stage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        METTAIL CORRECTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Text                                                                  │
│      ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Tier 1: Lexical Correction (liblevenshtein)                         │    │
│  │         Edit distance automata, phonetic rules                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Tier 2: Syntactic Correction (MORK/CFG)                             │    │
│  │         Grammar validation, pattern matching                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Tier 3: Semantic Correction (MeTTaIL)                               │    │
│  │         Type checking, behavioral verification                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      ↓                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ ★ SIMPLIFICATION TRANSPILER ★                                       │    │
│  │   Layer 1: Analysis → Layer 2: Rules → Layer 3: Strategy → Layer 4  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      ↓                                                                       │
│  Simplified Output                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

The transpiler accepts configuration for each layer:

```rust
pub struct SimplificationConfig {
    /// Which phases to run
    pub enabled_phases: Vec<SimplificationPhase>,

    /// Maximum iterations per phase
    pub max_iterations: usize,

    /// Whether to verify each transformation
    pub verify_each_step: bool,

    /// Cost threshold for beneficial transformations
    pub benefit_threshold: f64,

    /// Enable PathMap memoization
    pub enable_caching: bool,
}
```

---

## Next Steps

- [Analysis Layer](02-analysis-layer.md) - Detailed analysis design
- [Rule Application](03-rule-application.md) - MORK integration details
- [Strategy Selection](04-strategy-selection.md) - Phase ordering
- [Verification](05-verification.md) - Semantic preservation

---

## Changelog

- **2025-12-06**: Initial architecture documentation
