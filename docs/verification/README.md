# Formal Verification with Rocq

This directory contains the Rocq (formerly Coq), Dafny, Verus, SMT, and TLA+
verification material
for liblevenshtein-rust. It includes completed proof islands and older
in-progress proof trees; use `FORMAL_VERIFICATION_MANIFEST.tsv` and
`README_FORMAL_GATES.md` as the source of truth for which artifacts currently
support public correctness claims.

## Overview

Trusted files are gate-checked for active `Admitted.`, unallowlisted
assumptions, stale contracts, and evidence links. Debug, legacy, and partial
files are still audited, but they do not support library correctness claims
until promoted in the manifest.

## Exact Dyck correction and binary-persistence proof island

The multi-kind Dyck corrector is verified against the same four candidate
families used by the Rust interval table. Rocq proves that reconstruction is
kind-sensitive Dyck and that, once all strict subinterval minima are available,
the least recurrence candidate is equivalent to the global minimum over every
correction tree. This is the increasing-interval-length induction invariant,
not a bounded-language approximation.

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `core/theories/Conformance/DyckCorrection.v` | typed grammar, witness soundness, zero identity, total correction, first-pair decomposition, strict-subinterval dependencies, and global recurrence exactness |
| Rocq | `core/theories/Conformance/OperationSetSerialization.v` | explicit-applicability name independence; bincode exactness; protobuf preflight/exact-bit; gzip bounds |
| Dafny | `dafny/DyckSerialization.dfy` | replacement/pair arithmetic, recurrence minimum selection, and bincode/protobuf/gzip admission guards |
| Verus | `verus/dyck_serialization.rs` | Rust-facing typed-pair, candidate-minimum, zero-cost, bincode/protobuf preflight, exact-bit, and gzip obligations without assumptions |
| Z3 + cvc5 | `smt/dyck_serialization.smt2` | independent negated Dyck, bincode, protobuf-limit/version/exact-bit, and gzip-limit/trailing obligations are UNSAT in both solvers |
| TLA+ TLC | `tla/OperationSetDecode.tla`, `tla/OperationSetPortableDecode.tla` | exhaustive finite bincode lifecycle plus protobuf pre-allocation and single-member gzip admission |
| proptest | `tests/proptest_phase9_downstream.rs`, `tests/operation_set_serialization.rs`, `tests/operation_set_protobuf.rs`, `tests/operation_set_gzip.rs` | 2,000-case exhaustive-language differential and algebraic Dyck invariants; deterministic, canonical, execution-equivalent round trips; hostile decode and compression-correspondence cases |

The [exact-correction design](../design/grammar-correction/dyck-projection-lower-bound.md)
and [binary persistence guide](../user-guide/serialization.md) map the proof
relations to runtime branches, tests, and resource limits.

## Automaton-variant proof island

The Phase 5 automaton-variant seam is verified by the assumption-free Rocq
development `Conformance/PositionKindVariant.v`, the Rust-facing Verus model
`position_kind_variant.rs`, independent Z3/cvc5 counterexample checks, and the
TLA+ `VariantDispatch` trace-equivalence model. The corresponding Rust
properties turn representation, selector, and subsumption lemmas into 2,000-case
executable invariants.

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `core/theories/Conformance/PositionKindVariant.v` | full-key injectivity, selector equivalence, continuation separation, and error-order prerequisites |
| Verus | `verus/position_kind_variant.rs` | Rust-facing payload, ordering-key, selector, and built-in variant obligations |
| Z3 + cvc5 | `smt/position_kind_variant.smt2` | eight independent negated representation/dispatch/subsumption obligations are UNSAT in both solvers |
| TLA+ TLC | `tla/VariantDispatch.tla` | one edge-level selection is stable, processes each position once, and yields the legacy per-position trace |
| proptest | `tests/proptest_position_kind_variants.rs` | 2,000 reference subsumption and 2,000 deterministic typed-transition cases plus ordering/layout properties |

## Affine-gap proof island

The affine-gap development verifies the three-layer Gotoh refinement, exact
integer guards, same-index B-4 subsumption, layer-aware completion, and the
operation-derived characteristic-vector window. The proof boundary is explicit:
cross-index B-5 pruning is disabled until the transition kernel can fuse a
query skip with the current dictionary-edge consumption.

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `affine/theories/AffineGap.v` | layer separation, uniform switch penalty, arbitrary-trace B-4 soundness, trailing completion, window, and budget guards |
| Dafny | `dafny/AffineGap.dfy` | 14 imperative-style arithmetic and layer obligations |
| Verus | `verus/affine_gap.rs` | eight Rust-facing layer, B-4, completion, window, and checked-add obligations |
| Z3 + cvc5 | `smt/affine_gap.smt2` | ten independent negated obligations are UNSAT in both solvers |
| TLA+ TLC | `tla/AffineGap.tla` | bounded identical-action traces preserve B-4 and cost dominance |
| proptest | `src/transducer/variants/affine.rs`, `tests/affine_gap.rs` | 2,000-case invariant families plus exact complete-result-map differential tests against an independent Gotoh implementation |

See the [literate algorithm](../algorithms/10-affine-gap/README.md), [design](../design/affine-gap-automaton.md), and [Gotoh paper summary](../research/gotoh/PAPER_SUMMARY.md) for the implementation correspondence.

## Unrestricted Damerau streaming proof island

The Phase 6 proof island verifies the finite-history refinement used by
`Algorithm::DamerauLevenshtein`. A pending position stores the positive query
endpoint displacement `$`\delta`$`; entry prepays the transposition and skipped
query units, each extension charges one dictionary insertion, and resolution
advances to the exact opposite endpoint without changing cost.

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `damerau/theories/DamerauStreaming.v` | entry budget and payload validity; extension and resolution; no pending epsilon; Lowrance–Wagner macro-cost equivalence; kind-aware subsumption; quadratic frontier envelope |
| Verus | `verus/damerau_streaming.rs` | six Rust-facing entry, extension, resolution, cost, and pending-key obligations |
| Z3 + cvc5 | `smt/damerau_streaming.smt2` | five independent negated budget, macro-cost, endpoint, and pending-subsumption obligations are UNSAT in both solvers |
| TLA+ TLC | `tla/DamerauStreaming.tla` | all `$`k=3`$` entry/extend/resolve traces preserve charge, delta, endpoint, consumption, and terminal-cost invariants |
| proptest | `tests/proptest_true_damerau.rs`, `tests/proptest_true_damerau_metric.rs` | exact reference-map equivalence for budgets 0 through 3 and the four metric laws |

The central refinement equation is
`$`\delta+b=(\delta-1)+b+1`$`: the streaming charge equals the
Lowrance–Wagner query-interior deletions, dictionary-interior insertions, and
one transposition. See the [literate algorithm](../algorithms/11-true-damerau/README.md)
and [streaming design](../design/true-damerau-streaming.md) for the
implementation correspondence.

## Generic elastic-search telemetry proof island

The generic K1–K4 proof island now includes the accounting invariants exposed
by `ElasticSearchStats`. These counters are observational: the verified claim
is that each decision contributes to exactly one partition and that adding the
observation cannot affect search results.

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `core/theories/Conformance/ElasticKernel.v` | edge and candidate partitions over complete decision traces; filtered counters are subsets |
| Rocq | `elastic/theories/WalkerSoundness.v` | local K1 terminal bounds plus K2 child inflation imply recursive DFS completeness; every emitted result is a real in-cutoff terminal |
| Verus | `verus/elastic_kernel.rs` | prefix, column, candidate-bound, and exact steps preserve their partitions; subset bounds |
| Z3 + cvc5 | `smt/elastic_kernel.smt2` | five negated partition-preservation/subset obligations are UNSAT in each solver, alongside K1–K4 |
| proptest | `tests/elastic_kernel_contract.rs`, `tests/dtw_transducer_tests.rs` | 2,000 generic and 2,000 DTW result-transparency/accounting cases |

The counter proofs do not replace kernel-specific admissibility proofs. They
establish that reported work is internally consistent once K1–K4 decide each
branch. The [elastic-kernel design](../design/elastic-kernels.md) and
[shared UCR protocol](../scientific-ledger/elastic-ucr-harness-2026-08-01.md)
state the exact correspondence.

## Current language-product proof island

The generic language product is covered by independent tools whose obligations
map directly to property tests:

| Tool | Artifact | Checked invariant |
|---|---|---|
| Verus | `verus/language_product.rs` | safe level arithmetic and bit shifts; canonicalization and union laws |
| Rocq | `core/theories/Conformance/LanguageProduct.v` | relational-image distribution, acceptance preservation, disjoint levels, 256-level bound |
| Z3 + cvc5 | `smt/language_product.smt2` | four bounded counterexample queries are UNSAT in both solvers |
| proptest | `tests/proptest_language_product.rs` | reference equivalence, merge law, unit/backend cases, resource guards |

Run `scripts/verify-formal.sh verus`, `scripts/verify-formal.sh smt`, and
`scripts/verify-formal.sh coq-file light docs/verification/core/theories/Conformance/LanguageProduct.v`.
The [language-product design](../design/language-product.md) explains how these
statements correspond to the Rust frontier.

## Current cost-monoid proof island

The ordered cost algebra is checked at three abstraction levels. Its exact
fixed-point path and its floating-point boundary are intentionally not
conflated:

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `core/theories/Conformance/CostMonoid.v` | assumption-free additive-real and bottleneck L1/L2/L3/L4/L6/L7 laws |
| Verus | `verus/cost_monoid.rs` | integer arithmetic, maximum, monotonicity, and exact scale divisibility |
| Z3 + cvc5 | `smt/cost_monoid.smt2` | bounded saturation, monotonicity, maximum, `TOP`, and scale-overflow counterexample queries |
| proptest | `tests/cost_monoid_laws.rs` | all concrete carriers, exact dyadic addition, general `f64` error envelope, scale round trips, and invalid values |

Rocq's weighted theorem is a mathematical-real model. It does not prove
bitwise associativity of arbitrary IEEE-754 addition; the Rust contract instead
tests exact dyadic inputs and a documented forward-error envelope. See the
[cost-monoid design](../design/cost-monoid.md) for the correspondence.

Run `scripts/verify-formal.sh verus`, `scripts/verify-formal.sh smt`, and
`scripts/verify-formal.sh coq-file light docs/verification/core/theories/Conformance/CostMonoid.v`.

## Current banded-DTW proof island

Banded DTW is intentionally verified as an exact non-metric kernel. The ship
gate is admissible pruning, while a separate executable theorem pins the
triangle-inequality failure:

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `dtw/theories/Indexing/DtwProperties.v` | squared interval admissibility and point exactness; recurrence monotonicity; non-negative additive inflation; prefix LB_Keogh induction; first-gate pruning; endpoint band reachability; symmetry |
| Rocq | `dtw/theories/NotAMetric.v` | concrete band-one root-distance triangle counterexample, without axioms or admissions |
| Verus | `verus/dtw_kernel.rs` | 16 Rust-facing square, interval, recurrence, prefix, prune, reachability, symmetry/non-negativity, and counterexample-witness obligations |
| Z3 + cvc5 | `smt/dtw_kernel.smt2` | 10 independent negated real-arithmetic obligations, all required `UNSAT` in both solvers |
| TLA+ | `tla/ElasticTrieSearch.tla` | prefix pruning precedes column construction; prefix/column/candidate pruning are sound; exact emission, completeness, and termination |
| proptest | `src/time_series/kernels/{dtw,keogh}.rs`, `tests/dtw_transducer_tests.rs` | 8,000 kernel/reference/invariant cases and 4,000 public range/kNN databases |

Rocq reasons over mathematical reals and naturals. Verus cross-checks the
Rust-facing branch structure over unbounded integers. SMT uses nonlinear real
arithmetic so both Z3 and cvc5 independently reason about squares. Rust
properties instantiate the obligations with exactly represented integer-valued
finite `f64` samples, then separately cover non-finite and overflow boundaries.

TLC exhausts the nine-node model: 203 generated states, 69 distinct states,
and no violation. Its `PrefixGatePrecedesColumn` invariant corresponds to the
walker's evaluation order, not merely to prune soundness. Metric status is also
queryable as `ElasticKernel::IS_METRIC`; the compile-time
`MetricElasticKernel` gate excludes `DtwConfig` from triangle-dependent
generic structures.

See the [DTW/LB_Keogh source analysis](../research/dtw/PAPER_SUMMARY.md),
[literate algorithm](../algorithms/12-elastic-measures/README.md), and
[security controls](../security/resource-exhaustion.md).

## Current discrete Fréchet proof island

Discrete Fréchet is the first production kernel to exercise the generic
walker with `BottleneckCost` instead of additive `WeightedCost`:

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `frechet/theories/Metric/FrechetProperties.v` | interval/point recurrence; minimax inflation and local triangle composition; pinned endpoints; one-sided Hausdorff; zero-link identity |
| Verus | `verus/frechet_kernel.rs` | nine Rust-facing arithmetic, monotonicity, candidate-prune, triangle-step, and identity obligations |
| Z3 + cvc5 | `smt/frechet_kernel.smt2` | nine independent negated obligations, all required `UNSAT` in both solvers |
| TLA+ | `tla/ElasticTrieSearch.tla` | unchanged generic traversal, including bottleneck-compatible K1/K2 ordering, exact emission, root/full terminal completeness, and termination |
| proptest | `src/time_series/kernels/frechet.rs`, `tests/frechet_transducer_tests.rs` | reference-DP differential, interval/point columns, endpoint/Hausdorff bounds, run-collapse identity, triangle, and range/kNN equivalence |

Rocq uses mathematical reals, while Verus and SMT deliberately use unbounded
integers to cross-check branch structure without IEEE-754 ambiguity. Generated
Rust properties instantiate the same invariants over finite exactly represented
integer-valued `f64` samples. See the [paper analysis](../research/frechet/PAPER_SUMMARY.md)
and [elastic-kernel design](../design/elastic-kernels.md).

## Current ERP proof island

ERP's measure-specific arithmetic is checked separately from the already
verified generic walker state machine:

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `erp/theories/Metric/ErpProperties.v` | interval admissibility and point exactness; arbitrary-script gap-mass bound; quotient relation; zero-cost alignment implies quotient identity |
| Verus | `verus/erp_kernel.rs` | Rust-facing interval branches, reverse absolute inequality, zero-gap generator, K4 prune, row cutoff, and inflation |
| Z3 + cvc5 | `smt/erp_kernel.smt2` | eight independent bounded counterexample queries, all required `UNSAT` |
| TLA+ | `tla/ElasticTrieSearch.tla` | unchanged generic K1/K4 traversal, exact emission, terminal completeness, and termination |
| proptest | `src/time_series/kernels/erp.rs`, `tests/erp_transducer_tests.rs` | full-matrix differential, interval/point leaves, metric quotient, range/kNN, determinism, and saved counterexample |

Rocq uses mathematical real numbers and Verus/SMT use unbounded integers; Rust
properties connect those algebraic statements to finite `f64` behavior. The
[ERP research analysis](../research/erp/PAPER_SUMMARY.md) and
[elastic-kernel design](../design/elastic-kernels.md) document this trust
boundary and why raw-sequence identity is stated only modulo the fixed gap.

## Current TWED proof island

TWED's measure-specific proof island discharges the adjacent-bin arithmetic
premises consumed by the already verified generic walker:

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `twed/theories/Metric/TwedProperties.v` | arbitrary-real interval match/delete admissibility and point exactness; additive recurrence monotonicity; arbitrary-script length bound; strict stiffness; zero-parameter witness; script-cost composition |
| Verus | `verus/twed_kernel.rs` | 13 Rust-shaped interval, separability, recurrence, K4, metric-gate, degeneracy, and composition obligations |
| Z3 + cvc5 | `smt/twed_kernel.smt2` | 13 independent negated integer-arithmetic obligations, all required `UNSAT` in both solvers |
| TLA+ | `tla/ElasticTrieSearch.tla` | unchanged generic carry-aware K1/K4 traversal, exact emission, root/full terminal completeness, and termination |
| proptest | `src/time_series/kernels/twed.rs`, `tests/twed_transducer_tests.rs` | 8,000 kernel/reference/invariant/metric cases and 4,000 public range/kNN databases |

The Rocq file does not assume or admit the full Marteau metric theorem. It
proves the local and arbitrary-script obligations represented in the kernel;
generated Rust triples exercise the complete executable triangle recurrence.
The API independently makes the theorem's strict stiffness premise a type
invariant: only `MetricTwedConfig` implements `MetricElasticKernel`.

See the [TWED source analysis](../research/twed/PAPER_SUMMARY.md),
[literate recurrence](../algorithms/12-elastic-measures/README.md), and
[security controls](../security/resource-exhaustion.md).

## Current generalized-automaton repair proof island

The exact operation grid translates directly into path-fold and coordinate
invariants:

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `core/theories/Conformance/GeneralizedAutomatonRepair.v` | path consumption/cost folds, extension monotonicity, Hamming length, absent-deletion rejection, progress |
| Verus | `verus/generalized_automaton.rs` | positive scaled weights, budget monotonicity, coordinate progress, checked accumulation, fractional boundary |
| Z3 + cvc5 | `smt/generalized_automaton.smt2` | five bounded counterexample queries are UNSAT in both solvers |
| proptest | `tests/proptest_generalized_automaton_repair.rs` | standard, Hamming, indel/LCS, bounded-skip, budget, fractional, Unicode, and invalid-value correspondence |

See the [repair design](../design/generalized-automaton-repair.md) and [literate
algorithm](../algorithms/14-generalized-operation-grid/README.md).

## Class-A preset proof island

The preset layer specializes the generalized alignment semantics while proving
the independent references and validation boundary:

| Tool | Artifact | Checked invariant |
|---|---|---|
| Rocq | `core/theories/Conformance/ClassAPresets.v` | Hamming metric laws; reversible/composable indel scripts and length bounds; exact skip cost; aggregate-prefix validation |
| Dafny | `dafny/ClassAPresets.dfy` | 16 Hamming, indel parity, band, empty-side, skip, progress, and validation obligations |
| Verus | `verus/class_a_presets.rs` | 10 Rust-shaped accumulated-law, checked-aggregate, and boundary obligations |
| Z3 + cvc5 | `smt/class_a_presets.smt2` | 13 negated invariants, all `unsat` in both solvers |
| TLA+ | `tla/ClassAPresets.tla` | all three operation sets over the complete configured grid; 72 reachable states, no invariant violation |
| proptest | `tests/proptest_class_a_presets.rs` | 20,000 three-way, metric, threshold, ordering, parity, and resource cases |
| corpus | `tests/corpus_validation.rs` | 42,395 deterministic Birkbeck pair comparisons |

The bounded finite TLC model complements, rather than replaces, the
assumption-free arithmetic proofs. See the [Class-A design](../design/class-a-presets.md)
and [literate references](../algorithms/15-class-a-presets/README.md).

The remaining phase descriptions below are historical context for the older
phonetic proof tree; they do not override the manifest.

### Verification Workflow

```
┌─────────────────┐
│ 1. Formalize    │  Define algorithm in Rocq
│    in Rocq      │  Specify correctness properties
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Prove        │  Prove trusted theorems
│    Theorems     │  No Admitted in trusted scope
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Extract      │  Extract OCaml code
│    Reference    │  Reference implementation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Implement    │  Write Rust code
│    in Rust      │  Guided by proofs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Validate     │  QuickCheck tests
│    Properties   │  Mirror Rocq theorems
└─────────────────┘
```

## Directory Structure

```
docs/verification/
├── README.md                           # This file
├── phonetic/
│   ├── rewrite_rules.v                 # Phonetic rewrite system
│   ├── context.v                       # Context patterns
│   └── zompist.v                       # Zompist spelling rules
├── regex/
│   ├── nfa.v                           # NFA construction
│   ├── thompson.v                      # Thompson's algorithm
│   └── fuzzy_matching.v                # Fuzzy regex matching
├── phonetic_regex/
│   └── composition.v                   # Phonetic + Regex composition
└── cfg/
    ├── syntax.v                        # CFG definitions
    ├── operations.v                    # Structural operations
    ├── distance.v                      # Edit distance metric
    ├── earley.v                        # Earley parser
    └── soundness.v                     # Correctness proofs
```

## Phase 1: Phonetic Rewrite Rules

**Status**: In Progress ✅

### Files

- `phonetic/rewrite_rules.v` - Core formalization

### Theorems to Prove

| Theorem | Description | Status |
|---------|-------------|--------|
| `zompist_rules_wellformed` | All rules are well-formed | ⏳ To Do |
| `rule_application_bounded` | String expansion is bounded | ⏳ To Do |
| `some_rules_dont_commute` | Order matters for some rules | ⏳ To Do |
| `sequential_application_terminates` | Algorithm always terminates | ⏳ To Do |
| `rewrite_idempotent` | Fixed point property | ⏳ To Do |

### Definitions Complete

- ✅ `Phone` - Phonetic symbol type
- ✅ `Context` - Rule application contexts
- ✅ `RewriteRule` - Rule structure
- ✅ `apply_rule_at` - Single rule application
- ✅ `apply_rules_seq` - Sequential application
- ✅ Helper functions (`Phone_eqb`, `is_Some`, etc.)

### Next Steps

1. Define the 56 zompist rules as Rocq constants
2. Prove `zompist_rules_wellformed` by enumeration
3. Prove `rule_application_bounded` using rule analysis
4. Prove `sequential_application_terminates` using well-founded recursion
5. Prove `rewrite_idempotent` using fixed point argument

## Phase 2: Regex Automaton

**Status**: Not Started ⏳

### Planned Theorems

- `thompson_correctness` - Thompson construction preserves semantics
- `determinize_correct` - Determinization preserves language
- `fuzzy_accepts_generalizes` - Fuzzy matching generalizes exact matching

## Phase 3: Phonetic Fuzzy Regex

**Status**: Not Started ⏳

### Planned Theorems

- `composition_sound` - Combined system is sound
- `phonetic_regex_commutes` - Operations compose correctly

## Phase 4: Structural CFG

**Status**: Not Started ⏳

### Planned Theorems

- `transpose_type_safe` - Type-safe transposition
- `structural_ops_preserve_wf` - Well-formedness preservation
- `distance_identity` - Edit distance identity property
- `distance_symmetric` - Edit distance symmetry
- `distance_triangle` - Triangle inequality
- `earley_terminates` - Parser termination
- `earley_soundness` - Parser correctness

## Building Proofs

### Prerequisites

```bash
# Install Rocq (Coq 8.18+)
opam install coq

# Verify installation
coqc --version
```

### Compile Proofs

```bash
# Compile single file
coqc docs/verification/phonetic/rewrite_rules.v

# Generate documentation
coqdoc --html -d docs/verification/html docs/verification/phonetic/*.v
```

### Extract OCaml

```bash
# Extract OCaml code
coqc docs/verification/phonetic/rewrite_rules.v
# Produces: Phone.ml, Context.ml, rewrite_rules.ml
```

## Rust Integration

### Proof References in Code

Rust code includes inline references to Rocq proofs:

```rust
/// Apply phonetic rules sequentially
///
/// # Correctness (PROVEN):
/// - Terminates (Theorem sequential_application_terminates)
/// - Idempotent (Theorem rewrite_idempotent)
/// - Bounded expansion (Theorem rule_application_bounded)
///
/// Verification: docs/verification/phonetic/rewrite_rules.v:250-265
pub fn apply_rules_sequential(
    rules: &[RewriteRule],
    input: &[Phone],
) -> Vec<Phone> {
    // Implementation mirrors Rocq definition
}
```

### Property Tests

QuickCheck tests mirror Rocq theorems:

```rust
#[cfg(test)]
mod properties {
    /// Property: Sequential application terminates
    /// Corresponds to: Theorem sequential_application_terminates
    /// Proof: rewrite_rules.v:250
    #[quickcheck]
    fn sequential_application_terminates(input: Vec<Phone>) -> bool {
        let rules = zompist_rule_set();
        let _result = apply_rules_sequential(&rules, &input);
        true  // If we get here, it terminated (proven in Rocq)
    }

    /// Property: Rewriting is idempotent
    /// Corresponds to: Theorem rewrite_idempotent
    /// Proof: rewrite_rules.v:275
    #[quickcheck]
    fn rewrite_idempotent(input: Vec<Phone>) -> bool {
        let rules = zompist_rule_set();
        let once = apply_rules_sequential(&rules, &input);
        let twice = apply_rules_sequential(&rules, &once);
        once == twice
    }
}
```

## Verification Progress

### Overall Timeline

| Phase | Duration | Rocq | Rust | Total | Status |
|-------|----------|------|------|-------|--------|
| 1. Phonetic Rules | 6-8 weeks | 3-4 weeks | 3-4 weeks | 50% | 🟡 In Progress |
| 2. Regex NFA | 8-10 weeks | 4-5 weeks | 4-5 weeks | 0% | ⏳ Not Started |
| 3. Phonetic Regex | 6-8 weeks | 3-4 weeks | 3-4 weeks | 0% | ⏳ Not Started |
| 4. Structural CFG | 16-20 weeks | 8-10 weeks | 8-10 weeks | 0% | ⏳ Not Started |

**Total**: 36-46 weeks (8-11 months)

### Current Sprint: Phase 1, Week 1

**Goals**:
- ✅ Create directory structure
- ✅ Define core types (Phone, Context, RewriteRule)
- ✅ Define helper functions
- ⏳ Define 56 zompist rules
- ⏳ Prove well-formedness theorem

## References

### Rocq Resources

- [Rocq Documentation](https://rocq-prover.org/)
- [Software Foundations](https://softwarefoundations.cis.upenn.edu/)
- [Verified Software Toolchain](https://vst.cs.princeton.edu/)

### Phonetic Rules

- [Zompist Spelling Rules](https://zompist.com/spell.html)
- Original research on English orthography-to-phonology mapping

### Formal Verification

- **Verified Compilers**: CompCert
- **Verified OS**: seL4
- **Verified Crypto**: HACL*

## Contributing

When adding new features:

1. **Formalize first** in Rocq before coding
2. **Prove theorems** completely (no `Admitted`)
3. **Extract** OCaml reference implementation
4. **Implement** Rust version guided by proofs
5. **Write tests** that mirror Rocq theorems

## License

Same as parent project (see top-level LICENSE).
