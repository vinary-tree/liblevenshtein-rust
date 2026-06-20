[← Documentation Index](../../README.md)

# Formal Verification Tools for Rust: Research Summary

**Date**: 2025-10-30
**Purpose**: Evaluate formal verification tools for proving correctness of Levenshtein distance functions

## Overview

Multiple formal verification tools are available for Rust, each with different strengths, limitations, and use cases. This document evaluates their suitability for verifying the correctness of our Levenshtein distance implementations.

## Available Tools

### 1. **Kani** (AWS/Model Checking)

**Type**: Bounded model checking
**Backing**: AWS-funded, actively maintained
**Repository**: https://github.com/model-checking/kani

**Strengths**:
- ✅ Can verify unsafe Rust
- ✅ Checks memory safety properties automatically
- ✅ Verifies user-specified assertions
- ✅ Good integration with existing Rust projects (Firecracker, s2n-quic)
- ✅ Part of AWS's effort to verify Rust standard library
- ✅ Fully automated

**Limitations**:
- ❌ Bounded verification (limited loop unrolling)
- ❌ Only verifies monomorphic code (no generics)
- ❌ May have scalability issues with complex recursive functions

**Suitability for Distance Functions**: ⭐⭐⭐⭐ (4/5)
- Good for verifying concrete test cases
- Bounded nature may limit verification of recursive depth
- Can verify our specific distance computations up to certain string lengths

### 2. **Prusti** (ETH Zürich)

**Type**: Deductive verification (uses Viper as backend)
**Backing**: Academic (ETH Zürich), well-documented
**Repository**: https://github.com/viperproject/prusti-dev

**Strengths**:
- ✅ Proves rich correctness properties
- ✅ Low annotation overhead
- ✅ Excellent VSCode integration (checks as you type)
- ✅ Can handle generic code
- ✅ Best user experience among verification tools
- ✅ Specification language integrated with Rust

**Limitations**:
- ❌ Cannot reason about unsafe Rust
- ❌ Not fully automated (requires annotations)
- ❌ May struggle with complex memoization patterns

**Suitability for Distance Functions**: ⭐⭐⭐⭐⭐ (5/5)
- **BEST CHOICE** for our use case
- Our distance functions are safe Rust
- Can specify and verify distance metric properties:
  - Non-negativity
  - Symmetry
  - Triangle inequality (for standard distance)
  - Identity of indiscernibles
- Good documentation and tooling support

### 3. **Creusot** (Inria/WhyML)

**Type**: Deductive verification (uses Why3 as backend)
**Backing**: French research institute (Inria)
**Repository**: https://github.com/creusot-rs/creusot

**Strengths**:
- ✅ State-of-the-art verifier for safe Rust
- ✅ Can prove functional correctness
- ✅ Handles generic code
- ✅ Leverages Rust's type system effectively
- ✅ Based on solid theoretical foundations (RustHornBelt)

**Limitations**:
- ❌ Cannot verify unsafe Rust
- ❌ Requires understanding of Why3/separation logic
- ❌ Steeper learning curve than Prusti
- ❌ Less polished tooling

**Suitability for Distance Functions**: ⭐⭐⭐⭐ (4/5)
- Can verify our safe Rust implementations
- More complex to set up than Prusti
- Good choice if Prusti fails on specific patterns

### 4. **coq-of-rust**

**Type**: Translation to Coq
**Backing**: Formal verification company (formal.land)
**Repository**: https://github.com/formal-land/coq-of-rust

**Strengths**:
- ✅ Translates Rust to Coq for theorem proving
- ✅ 100% verification of execution cases (in theory)
- ✅ Uses Cargo integration
- ✅ Based on Creusot's translation infrastructure

**Limitations**:
- ❌ Requires Coq expertise
- ❌ Manual proof construction in Coq
- ❌ Time-intensive process
- ❌ Less mature than other tools

**Suitability for Distance Functions**: ⭐⭐ (2/5)
- Very high effort for manual proofs
- Overkill for our use case
- Only consider if other tools fail

### 5. **Verus** (CMU)

**Type**: SMT-based verification
**Backing**: CMU research
**Website**: https://verus-lang.github.io/

**Strengths**:
- ✅ Designed for systems programming
- ✅ Linear types and ghost code support
- ✅ Good performance on verification tasks
- ✅ Growing ecosystem

**Limitations**:
- ❌ Requires learning Verus-specific syntax
- ❌ Not standard Rust (special Verus syntax)
- ❌ Less mature than Prusti
- ❌ Would require rewriting functions

**Suitability for Distance Functions**: ⭐⭐ (2/5)
- Would require significant code changes
- Not practical for existing codebase

## Recommendation

### Primary Choice: **Prusti** ⭐⭐⭐⭐⭐

**Reasons**:
1. Best fit for safe Rust verification
2. Low annotation overhead
3. Excellent tooling and documentation
4. Can express distance metric properties naturally
5. Active development and support

**Implementation Plan**:
1. Install Prusti: `cargo install prusti-cli`
2. Add specifications to distance functions using `#[requires]`, `#[ensures]`
3. Specify mathematical properties:
   ```rust
   #[ensures(result >= 0)]  // Non-negativity
   #[ensures(source == target ==> result == 0)]  // Identity
   #[ensures(result == old(standard_distance(target, source)))]  // Symmetry
   ```
4. Run verification: `cargo prusti`

### Fallback: **Kani**

If Prusti cannot handle recursive memoization patterns:
- Use Kani for bounded verification
- Verify specific string lengths (e.g., up to 20 characters)
- Complement property-based testing

### Example Prusti Specification

```rust
use prusti_contracts::*;

#[pure]
#[ensures(result >= 0)]
#[ensures(source == target ==> result == 0)]
#[ensures(result == standard_distance(target, source))]  // Symmetry
pub fn standard_distance(source: &str, target: &str) -> usize {
    // ... implementation
}

// Triangle inequality (more complex, may need helper lemmas)
#[ensures(
    standard_distance(a, c) <=
    standard_distance(a, b) + standard_distance(b, c)
)]
pub fn triangle_inequality_holds(a: &str, b: &str, c: &str) -> bool {
    true
}
```

## Next Steps

1. ✅ **Research formal verification tools** (COMPLETED)
2. ⏭️ Install Prusti and experiment with simple examples
3. ⏭️ Add specifications to iterative distance functions first
4. ⏭️ Extend to recursive functions if successful
5. ⏭️ Document any limitations or workarounds needed
6. ⏭️ If Prusti fails, try Kani with bounded verification

## References

- Rust Formal Methods Interest Group: https://rust-formal-methods.github.io/
- Prusti User Guide: https://viperproject.github.io/prusti-dev/user-guide/
- Kani Tutorial: https://model-checking.github.io/kani/
- AWS Rust Verification Blog: https://aws.amazon.com/blogs/opensource/verify-the-safety-of-the-rust-standard-library/
- High Assurance Rust Book: https://highassurance.rs/

## Conclusion

**Formal verification is feasible for our distance functions**. Prusti offers the best balance of power, usability, and suitability for our use case. Property-based testing with `proptest` provides excellent confidence already, but formal verification with Prusti could provide mathematical proofs of correctness for critical properties.

**Recommendation**: Start with Prusti for the iterative distance functions, as they are simpler and will help us learn the tool. If successful, extend to recursive implementations.
