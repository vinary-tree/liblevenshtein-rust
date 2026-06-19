//! Property-based tests for the phonetic × Levenshtein product automaton.
//!
//! These mirror the TLA+ `ProductAutomaton` model (`docs/verification/tla/`),
//! which models the product construction's correctness and cost monotonicity
//! over an abstract NFA transition relation. Here we exercise the real Rust
//! construction (`src/phonetic/nfa/product.rs`).
//!
//! Key idea for a brute-force oracle: when the phonetic NFA recognizes a single
//! literal pattern, the product automaton accepts an input iff that input is
//! within the edit-distance bound of the pattern. So plain Levenshtein distance
//! (`standard_distance`) is an exact oracle for acceptance and `min_distance`.
//!
//! The phonetic NFA lives behind the `phonetic-rules` feature, so this whole
//! test file is gated on it; run with `cargo test --features phonetic-rules`.
#![cfg(feature = "phonetic-rules")]

use liblevenshtein::distance::standard_distance;
use liblevenshtein::phonetic::nfa::product::ProductAutomatonChar;
use liblevenshtein::phonetic::nfa::thompson::ThompsonBuilderChar;
use proptest::prelude::*;

fn arb_pattern() -> impl Strategy<Value = String> {
    // Non-empty literal pattern over a small alphabet so near matches are common.
    prop::string::string_regex("[a-c]{1,6}").unwrap()
}

fn arb_input() -> impl Strategy<Value = String> {
    // Inputs may be empty to exercise the d(w, ε) = |w| boundary.
    prop::string::string_regex("[a-c]{0,6}").unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// ProductCorrectness: with a literal-pattern NFA, the product accepts
    /// `input` iff the standard edit distance from the pattern is within bound.
    #[test]
    fn product_accepts_iff_within_distance(
        pattern in arb_pattern(),
        input in arb_input(),
        max_distance in 0u8..=3,
    ) {
        let nfa = ThompsonBuilderChar::new().literal(&pattern);
        let product = ProductAutomatonChar::new(nfa, max_distance);
        let expected = standard_distance(&pattern, &input) <= max_distance as usize;
        prop_assert_eq!(
            product.accepts(&input), expected,
            "pattern={:?} input={:?} max_distance={}", pattern, input, max_distance
        );
    }

    /// `min_distance` returns the exact edit distance when within bound, else None.
    #[test]
    fn product_min_distance_matches_oracle(
        pattern in arb_pattern(),
        input in arb_input(),
        max_distance in 0u8..=3,
    ) {
        let nfa = ThompsonBuilderChar::new().literal(&pattern);
        let product = ProductAutomatonChar::new(nfa, max_distance);
        let d = standard_distance(&pattern, &input);
        let expected = if d <= max_distance as usize { Some(d as u8) } else { None };
        prop_assert_eq!(
            product.min_distance(&input), expected,
            "pattern={:?} input={:?} max_distance={} true_distance={}",
            pattern, input, max_distance, d
        );
    }

    /// CostMonotonicity: enlarging the budget never loses an accepted input.
    #[test]
    fn product_cost_monotonic(
        pattern in arb_pattern(),
        input in arb_input(),
        k1 in 0u8..=2,
    ) {
        let k2 = k1 + 1;
        let p1 = ProductAutomatonChar::new(ThompsonBuilderChar::new().literal(&pattern), k1);
        let p2 = ProductAutomatonChar::new(ThompsonBuilderChar::new().literal(&pattern), k2);
        if p1.accepts(&input) {
            prop_assert!(
                p2.accepts(&input),
                "monotonicity violated: pattern={:?} input={:?} k1={} k2={}",
                pattern, input, k1, k2
            );
        }
    }

    /// An exact occurrence of the pattern is always accepted (distance 0).
    #[test]
    fn product_accepts_exact(pattern in arb_pattern(), max_distance in 0u8..=3) {
        let nfa = ThompsonBuilderChar::new().literal(&pattern);
        let product = ProductAutomatonChar::new(nfa, max_distance);
        prop_assert!(product.accepts(&pattern));
        prop_assert_eq!(product.min_distance(&pattern), Some(0u8));
    }
}
