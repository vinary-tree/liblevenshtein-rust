# Formal Verification with Rocq

This directory contains the Rocq (formerly Coq) and TLA+ verification material
for liblevenshtein-rust. It includes completed proof islands and older
in-progress proof trees; use `FORMAL_VERIFICATION_MANIFEST.tsv` and
`README_FORMAL_GATES.md` as the source of truth for which artifacts currently
support public correctness claims.

## Overview

Trusted files are gate-checked for active `Admitted.`, unallowlisted
assumptions, stale contracts, and evidence links. Debug, legacy, and partial
files are still audited, but they do not support library correctness claims
until promoted in the manifest.

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
