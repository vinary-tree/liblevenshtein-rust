//! Property-based tests mirroring the 5 Coq theorems.
//!
//! This module uses proptest to verify that the Rust implementation maintains
//! the formal properties proven in Coq/Rocq. Each property test corresponds
//! to one of the 5 theorems in `docs/verification/phonetic/zompist_rules.v`.
//!
//! # Theorems Tested
//!
//! 1. **Well-formedness** (Theorem 1, zompist_rules.v:285)
//! 2. **Bounded Expansion** (Theorem 2, zompist_rules.v:425)
//! 3. **Non-Confluence** (Theorem 3, zompist_rules.v:491)
//! 4. **Termination** (Theorem 4, zompist_rules.v:569)
//! 5. **Idempotence** (Theorem 5, zompist_rules.v:615)

#[cfg(test)]
mod tests {
    use super::super::application::{apply_rule_at, apply_rules_seq, MAX_EXPANSION_FACTOR};
    use super::super::rules::{orthography_rules, phonetic_rules, test_rules, zompist_rules};
    use super::super::types::{
        Context, ContextByte, Phone, PhoneByte, RewriteRule, RewriteRuleByte,
    };
    use proptest::prelude::*;

    // ========================================================================
    // Proptest Generators
    // ========================================================================

    /// Generate arbitrary Phone values
    fn arb_phone() -> impl Strategy<Value = PhoneByte> {
        prop_oneof![
            any::<u8>()
                .prop_filter("valid ASCII", |&c| c >= 32 && c < 127)
                .prop_map(PhoneByte::Vowel),
            any::<u8>()
                .prop_filter("valid ASCII", |&c| c >= 32 && c < 127)
                .prop_map(PhoneByte::Consonant),
            (
                any::<u8>().prop_filter("valid ASCII", |&c| c >= 32 && c < 127),
                any::<u8>().prop_filter("valid ASCII", |&c| c >= 32 && c < 127)
            )
                .prop_map(|(c1, c2)| PhoneByte::Digraph(c1, c2)),
            Just(PhoneByte::Silent),
        ]
    }

    /// Generate arbitrary Context values
    fn arb_context() -> impl Strategy<Value = ContextByte> {
        prop_oneof![
            Just(ContextByte::Initial),
            Just(ContextByte::Final),
            prop::collection::vec(any::<u8>().prop_filter("valid ASCII", |&c| c >= 32 && c < 127), 0..5)
                .prop_map(ContextByte::BeforeVowel),
            prop::collection::vec(any::<u8>().prop_filter("valid ASCII", |&c| c >= 32 && c < 127), 0..5)
                .prop_map(ContextByte::AfterConsonant),
            prop::collection::vec(any::<u8>().prop_filter("valid ASCII", |&c| c >= 32 && c < 127), 0..5)
                .prop_map(ContextByte::BeforeConsonant),
            prop::collection::vec(any::<u8>().prop_filter("valid ASCII", |&c| c >= 32 && c < 127), 0..5)
                .prop_map(ContextByte::AfterVowel),
            Just(ContextByte::Anywhere),
        ]
    }

    /// Generate arbitrary RewriteRule values
    fn arb_rewrite_rule() -> impl Strategy<Value = RewriteRuleByte> {
        (
            any::<usize>(),
            "[a-z]+",
            prop::collection::vec(arb_phone(), 1..5), // Non-empty pattern (well-formedness)
            prop::collection::vec(arb_phone(), 0..7), // Replacement can be empty
            arb_context(),
            prop::num::f64::NORMAL.prop_filter("non-negative", |&w| w >= 0.0),
        )
            .prop_map(
                |(rule_id, rule_name, pattern, replacement, context, weight)| RewriteRuleByte {
                    rule_id,
                    rule_name,
                    pattern,
                    replacement,
                    context,
                    weight,
                    syllable_condition: None,
                },
            )
    }

    /// Generate phonetic strings (sequences of phones)
    fn arb_phonetic_string() -> impl Strategy<Value = Vec<PhoneByte>> {
        prop::collection::vec(arb_phone(), 0..20)
    }

    // ========================================================================
    // Property Test 1: Well-formedness (Theorem 1)
    // ========================================================================

    /// **Theorem 1: Well-formedness** (zompist_rules.v:285)
    ///
    /// All rules in the zompist rule set satisfy well-formedness:
    /// - Pattern is non-empty
    /// - Weight is non-negative
    #[test]
    fn test_zompist_rules_wellformed() {
        let rules = zompist_rules();

        for rule in rules {
            // Pattern must be non-empty
            assert!(
                !rule.pattern.is_empty(),
                "Rule {} has empty pattern",
                rule.rule_name
            );

            // Weight must be non-negative
            assert!(
                rule.weight >= 0.0,
                "Rule {} has negative weight: {}",
                rule.rule_name,
                rule.weight
            );
        }
    }

    proptest! {
        /// Property: Generated rules satisfy well-formedness
        #[test]
        fn prop_generated_rules_wellformed(rule in arb_rewrite_rule()) {
            // Pattern is non-empty (enforced by generator)
            prop_assert!(!rule.pattern.is_empty());

            // Weight is non-negative (enforced by generator)
            prop_assert!(rule.weight >= 0.0);
        }
    }

    // ========================================================================
    // Property Test 2: Bounded Expansion (Theorem 2)
    // ========================================================================

    /// **Theorem 2: Bounded Expansion** (zompist_rules.v:425)
    ///
    /// For all rules in zompist_rule_set and all positions,
    /// if apply_rule_at succeeds, the output length is bounded:
    ///
    /// ```text
    /// length(output) ≤ length(input) + MAX_EXPANSION_FACTOR
    /// ```
    #[test]
    fn test_zompist_rules_bounded_expansion() {
        let rules = zompist_rules();

        // Test with various inputs
        let test_inputs = vec![
            vec![Phone::Consonant(b'x')],                              // Test rule x→yy
            vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],      // Test gh→∅
            vec![Phone::Consonant(b'p'), Phone::Consonant(b'h')],      // Test ph→f
            vec![Phone::Consonant(b'q'), Phone::Consonant(b'u')],      // Test qu→kw
        ];

        for rule in &rules {
            for input in &test_inputs {
                for pos in 0..=input.len() {
                    if let Some(output) = apply_rule_at(rule, input, pos) {
                        let expansion = output.len() as i64 - input.len() as i64;
                        assert!(
                            expansion <= MAX_EXPANSION_FACTOR as i64,
                            "Rule {} expanded by {} (max: {})\nInput: {:?}\nOutput: {:?}",
                            rule.rule_name,
                            expansion,
                            MAX_EXPANSION_FACTOR,
                            input,
                            output
                        );
                    }
                }
            }
        }
    }

    proptest! {
        /// Property: Any rule application is bounded
        #[test]
        fn prop_rule_application_bounded(
            rule in arb_rewrite_rule(),
            s in arb_phonetic_string(),
            pos in 0usize..20
        ) {
            if let Some(result) = apply_rule_at(&rule, &s, pos) {
                let expansion = result.len() as i64 - s.len() as i64;
                prop_assert!(
                    expansion <= MAX_EXPANSION_FACTOR as i64,
                    "Expansion {} exceeds maximum {}",
                    expansion,
                    MAX_EXPANSION_FACTOR
                );
            }
        }

        /// Property: Sequential application maintains bounded growth
        #[test]
        fn prop_sequential_application_bounded(
            s in arb_phonetic_string(),
            fuel in 1usize..100
        ) {
            let rules = orthography_rules();
            if let Some(result) = apply_rules_seq(&rules, &s, fuel) {
                // Total expansion bounded by number of applications × MAX_EXPANSION_FACTOR
                let max_total_expansion = fuel * MAX_EXPANSION_FACTOR;
                prop_assert!(
                    result.len() <= s.len() + max_total_expansion,
                    "Sequential application expanded beyond bound"
                );
            }
        }
    }

    // ========================================================================
    // Property Test 3: Non-Confluence (Theorem 3)
    // ========================================================================

    /// **Theorem 3: Non-Confluence** (zompist_rules.v:491)
    ///
    /// There exist rules r1, r2 in zompist_rule_set such that
    /// applying them in different orders produces different results.
    ///
    /// Proven with counterexample: x→yy and y→z on input "xy"
    #[test]
    fn test_non_confluence_counterexample() {
        let test_rules = test_rules();
        let rule_x_expand = &test_rules[0]; // x → yy
        let rule_y_to_z = &test_rules[1]; // y → z

        // Input: "xy" (as phones)
        let input = vec![Phone::Consonant(b'x'), Phone::Consonant(b'y')];

        // Order 1: Apply x→yy first, then y→z
        let temp1 = apply_rule_at(rule_x_expand, &input, 0).expect("x→yy should apply");
        // temp1 = [y, y, y] (from x→yy, keeping y)
        let result1 = apply_rules_seq(&[rule_y_to_z.clone()], &temp1, 10).expect("Should succeed");

        // Order 2: Apply y→z first, then x→yy
        let temp2 = apply_rule_at(rule_y_to_z, &input, 1).expect("y→z should apply");
        // temp2 = [x, z] (keeping x, y→z)
        let result2 = apply_rules_seq(&[rule_x_expand.clone()], &temp2, 10).expect("Should succeed");

        // Results should differ (non-confluence)
        assert_ne!(
            result1, result2,
            "Rules commute when they shouldn't!\nOrder 1: {:?}\nOrder 2: {:?}",
            result1, result2
        );
    }

    // ========================================================================
    // Property Test 4: Termination (Theorem 4)
    // ========================================================================

    /// **Theorem 4: Termination** (zompist_rules.v:569)
    ///
    /// For all well-formed rule sets and inputs,
    /// sequential application terminates with sufficient fuel.
    #[test]
    fn test_sequential_application_terminates() {
        let rules = zompist_rules();
        let test_inputs = vec![
            vec![],
            vec![Phone::Consonant(b'a')],
            vec![Phone::Consonant(b'x'), Phone::Consonant(b'y')],
            vec![Phone::Consonant(b'p'), Phone::Consonant(b'h'), Phone::Vowel(b'o')],
        ];

        for input in test_inputs {
            // Sufficient fuel: input.len() × rules.len() × MAX_EXPANSION_FACTOR
            let fuel = (input.len().max(1)) * rules.len() * MAX_EXPANSION_FACTOR;

            let result = apply_rules_seq(&rules, &input, fuel);
            assert!(
                result.is_some(),
                "Sequential application failed to terminate with fuel={}",
                fuel
            );
        }
    }

    proptest! {
        /// Property: Sequential application always terminates with sufficient fuel
        #[test]
        fn prop_sequential_terminates(s in arb_phonetic_string()) {
            let rules = orthography_rules();
            // Sufficient fuel as per theorem
            let fuel = (s.len().max(1)) * rules.len() * MAX_EXPANSION_FACTOR;

            let result = apply_rules_seq(&rules, &s, fuel);
            prop_assert!(result.is_some(), "Failed to terminate with sufficient fuel");
        }

        /// Property: Zero fuel always returns input unchanged
        #[test]
        fn prop_zero_fuel_identity(s in arb_phonetic_string()) {
            let rules = orthography_rules();
            let result = apply_rules_seq(&rules, &s, 0);
            prop_assert_eq!(result, Some(s.clone()), "Zero fuel should return input");
        }
    }

    // ========================================================================
    // Property Test 5: Idempotence (Theorem 5)
    // ========================================================================

    /// **Theorem 5: Idempotence** (zompist_rules.v:615)
    ///
    /// If apply_rules_seq reaches a fixed point (no more rules apply),
    /// then applying the rules again produces the same result.
    #[test]
    fn test_rewrite_idempotent() {
        let rules = zompist_rules();
        let test_inputs = vec![
            vec![Phone::Consonant(b'p'), Phone::Consonant(b'h')], // ph→f
            vec![Phone::Consonant(b'c'), Phone::Consonant(b'h')], // ch→ç
            vec![Phone::Vowel(b'e')],                             // Final e→silent (but not at final)
        ];

        for input in test_inputs {
            let fuel = input.len() * rules.len() * MAX_EXPANSION_FACTOR;

            // First application
            let result1 = apply_rules_seq(&rules, &input, fuel).expect("Should terminate");

            // Second application (should be idempotent)
            let result2 = apply_rules_seq(&rules, &result1, fuel).expect("Should terminate");

            assert_eq!(
                result1, result2,
                "Rewrite is not idempotent!\nFirst: {:?}\nSecond: {:?}",
                result1, result2
            );
        }
    }

    proptest! {
        /// Property: Applying rules twice gives same result (idempotence)
        #[test]
        fn prop_rewrite_idempotent(s in arb_phonetic_string()) {
            let rules = orthography_rules();
            let fuel = (s.len().max(1)) * rules.len() * MAX_EXPANSION_FACTOR;

            // First application
            if let Some(result1) = apply_rules_seq(&rules, &s, fuel) {
                // Second application on result
                if let Some(result2) = apply_rules_seq(&rules, &result1, fuel) {
                    // Should be idempotent (fixed point)
                    prop_assert_eq!(
                        result1, result2,
                        "Sequential application is not idempotent"
                    );
                }
            }
        }

        /// Property: Fixed point is stable
        #[test]
        fn prop_fixed_point_stable(s in arb_phonetic_string()) {
            let rules = phonetic_rules();
            let fuel = (s.len().max(1)) * rules.len() * MAX_EXPANSION_FACTOR;

            if let Some(result) = apply_rules_seq(&rules, &s, fuel) {
                // Apply again - should get same result
                let result2 = apply_rules_seq(&rules, &result, fuel);
                prop_assert_eq!(result2, Some(result.clone()));
            }
        }
    }

    // ========================================================================
    // Additional Properties
    // ========================================================================

    proptest! {
        /// Property: Empty rule set is identity
        #[test]
        fn prop_empty_rules_identity(s in arb_phonetic_string()) {
            let empty_rules: Vec<RewriteRuleByte> = vec![];
            let result = apply_rules_seq(&empty_rules, &s, 100);
            prop_assert_eq!(result, Some(s.clone()));
        }

        /// Property: Rule application is deterministic
        #[test]
        fn prop_deterministic(
            rule in arb_rewrite_rule(),
            s in arb_phonetic_string(),
            pos in 0usize..20
        ) {
            let result1 = apply_rule_at(&rule, &s, pos);
            let result2 = apply_rule_at(&rule, &s, pos);
            prop_assert_eq!(result1, result2, "Non-deterministic behavior detected");
        }
    }
}
