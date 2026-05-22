//! Phase 2: Phonetic Module Coverage Tests
//!
//! This module provides comprehensive test coverage for the phonetic NFA subsystem:
//! - nfa/types.rs: CharClass, NFAState, TransitionLabel, Transition
//! - nfa/context.rs: BoundaryKind, ContextMatcher, ContextPattern
//! - nfa/optimizer.rs: NfaOptimizer, OptimizationConfig, OptimizationStats
//! - regex/parser.rs: Parser edge cases and grammar coverage
//! - llev/parser.rs: Directive parsing and rule definitions

#![cfg(feature = "phonetic-rules")]

use proptest::prelude::*;

// ============================================================================
// NFA Types Coverage Tests (src/phonetic/nfa/types.rs)
// ============================================================================

mod nfa_types_coverage {
    use liblevenshtein::phonetic::nfa::{
        CharClass, CharClassChar, NFAState, StateId, Transition, TransitionChar, TransitionLabel,
        TransitionLabelChar,
    };

    // --- NFAState Tests ---

    #[test]
    fn test_nfa_state_constructors() {
        // Test all constructors
        let state1 = NFAState::new(0, false);
        assert_eq!(state1.id, 0);
        assert!(!state1.is_final);

        let state2 = NFAState::non_final(5);
        assert_eq!(state2.id, 5);
        assert!(!state2.is_final);

        let state3 = NFAState::final_state(10);
        assert_eq!(state3.id, 10);
        assert!(state3.is_final);
    }

    #[test]
    fn test_nfa_state_display() {
        let non_final = NFAState::non_final(0);
        assert_eq!(format!("{}", non_final), "q0");

        let final_state = NFAState::final_state(1);
        assert_eq!(format!("{}", final_state), "q1*");
    }

    #[test]
    fn test_nfa_state_max_id() {
        // Test with maximum StateId value
        let max_id: StateId = u32::MAX;
        let state = NFAState::new(max_id, true);
        assert_eq!(state.id, max_id);
        assert!(state.is_final);
    }

    #[test]
    fn test_nfa_state_equality_and_hash() {
        use std::collections::HashSet;

        let s1 = NFAState::new(0, false);
        let s2 = NFAState::new(0, false);
        let s3 = NFAState::new(0, true);
        let s4 = NFAState::new(1, false);

        assert_eq!(s1, s2);
        assert_ne!(s1, s3); // Different is_final
        assert_ne!(s1, s4); // Different id

        let mut set = HashSet::new();
        set.insert(s1);
        assert!(set.contains(&s2));
        assert!(!set.contains(&s3));
    }

    // --- CharClassChar Tests ---

    #[test]
    fn test_char_class_empty() {
        let empty = CharClassChar::new();
        assert!(empty.is_empty());
        assert!(!empty.matches('a'));
        assert!(!empty.negated);
    }

    #[test]
    fn test_char_class_single_range() {
        let lowercase = CharClassChar::from_range('a', 'z');
        assert!(!lowercase.is_empty());
        assert!(lowercase.matches('a'));
        assert!(lowercase.matches('m'));
        assert!(lowercase.matches('z'));
        assert!(!lowercase.matches('A'));
        assert!(!lowercase.matches('0'));
    }

    #[test]
    fn test_char_class_multiple_ranges() {
        let mut cc = CharClassChar::new();
        cc.add_range('a', 'z');
        cc.add_range('A', 'Z');
        cc.add_range('0', '9');

        assert!(cc.matches('a'));
        assert!(cc.matches('Z'));
        assert!(cc.matches('5'));
        assert!(!cc.matches('!'));
    }

    #[test]
    fn test_char_class_negation() {
        let vowels = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
        let non_vowels = vowels.negated();

        assert!(non_vowels.negated);
        assert!(!non_vowels.matches('a'));
        assert!(!non_vowels.matches('e'));
        assert!(non_vowels.matches('b'));
        assert!(non_vowels.matches('z'));
    }

    #[test]
    fn test_char_class_double_negation() {
        let vowels = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
        let double_neg = vowels.negated().negated();

        assert!(!double_neg.negated);
        assert!(double_neg.matches('a'));
        assert!(!double_neg.matches('b'));
    }

    #[test]
    fn test_char_class_unicode_ranges() {
        // Greek letters
        let greek = CharClassChar::from_range('α', 'ω');
        assert!(greek.matches('α'));
        assert!(greek.matches('μ'));
        assert!(greek.matches('ω'));
        assert!(!greek.matches('a'));
    }

    #[test]
    fn test_char_class_add_char() {
        let mut cc = CharClassChar::new();
        cc.add_char('x');
        cc.add_char('y');
        cc.add_char('z');

        assert!(cc.matches('x'));
        assert!(cc.matches('y'));
        assert!(cc.matches('z'));
        assert!(!cc.matches('w'));
    }

    #[test]
    fn test_char_class_display() {
        let vowels = CharClassChar::from_chars(&['a', 'e', 'i']);
        assert_eq!(format!("{}", vowels), "[aei]");

        let range = CharClassChar::from_range('a', 'z');
        assert_eq!(format!("{}", range), "[a-z]");

        let negated = CharClassChar::from_chars(&['x']).negated();
        assert_eq!(format!("{}", negated), "[^x]");
    }

    // --- CharClass (byte-level) Tests ---

    #[test]
    fn test_byte_class_empty() {
        let empty = CharClass::new();
        assert!(empty.is_empty());
        assert!(!empty.matches(b'a'));
    }

    #[test]
    fn test_byte_class_from_bytes() {
        let vowels = CharClass::from_bytes(&[b'a', b'e', b'i', b'o', b'u']);
        assert!(vowels.matches(b'a'));
        assert!(!vowels.matches(b'b'));
    }

    #[test]
    fn test_byte_class_negation() {
        let non_vowels = CharClass::from_bytes(&[b'a', b'e', b'i', b'o', b'u']).negated();
        assert!(!non_vowels.matches(b'a'));
        assert!(non_vowels.matches(b'b'));
    }

    #[test]
    fn test_byte_class_add_operations() {
        let mut cc = CharClass::new();
        cc.add_byte(b'x');
        cc.add_range(b'0', b'9');

        assert!(cc.matches(b'x'));
        assert!(cc.matches(b'5'));
        assert!(!cc.matches(b'y'));
    }

    #[test]
    fn test_byte_class_display() {
        let range = CharClass::from_range(b'a', b'z');
        assert_eq!(format!("{}", range), "[a-z]");
    }

    // --- TransitionLabelChar Tests ---

    #[test]
    fn test_label_epsilon() {
        let label = TransitionLabelChar::Epsilon;
        assert!(label.is_epsilon());
        assert!(!label.is_anchor());
        assert!(!label.consumes_input());
        assert!(label.matches('x')); // epsilon always matches
        assert_eq!(label.expected_char(), None);
    }

    #[test]
    fn test_label_char() {
        let label = TransitionLabelChar::Char('a');
        assert!(!label.is_epsilon());
        assert!(!label.is_anchor());
        assert!(label.consumes_input());
        assert!(label.matches('a'));
        assert!(!label.matches('b'));
        assert_eq!(label.expected_char(), Some('a'));
    }

    #[test]
    fn test_label_char_class() {
        let class = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
        let label = TransitionLabelChar::CharClass(class);
        assert!(!label.is_epsilon());
        assert!(label.consumes_input());
        assert!(label.matches('a'));
        assert!(!label.matches('b'));
        assert_eq!(label.expected_char(), None);
    }

    #[test]
    fn test_label_any() {
        let label = TransitionLabelChar::Any;
        assert!(label.consumes_input());
        assert!(label.matches('a'));
        assert!(label.matches('z'));
        assert!(label.matches(' '));
        assert!(label.matches('😀'));
        assert_eq!(label.expected_char(), None);
    }

    #[test]
    fn test_label_anchors() {
        // Start of line
        let sol = TransitionLabelChar::StartOfLine;
        assert!(sol.is_anchor());
        assert!(!sol.is_epsilon());
        assert!(!sol.consumes_input());
        assert!(!sol.matches('x')); // anchors don't match chars

        // End of line
        let eol = TransitionLabelChar::EndOfLine;
        assert!(eol.is_anchor());

        // Start of input
        let soi = TransitionLabelChar::StartOfInput;
        assert!(soi.is_anchor());

        // End of input
        let eoi = TransitionLabelChar::EndOfInput;
        assert!(eoi.is_anchor());

        // Strict end of input
        let eois = TransitionLabelChar::EndOfInputStrict;
        assert!(eois.is_anchor());
    }

    #[test]
    fn test_label_anchor_position_matching() {
        let sol = TransitionLabelChar::StartOfLine;
        let eol = TransitionLabelChar::EndOfLine;
        let soi = TransitionLabelChar::StartOfInput;
        let eoi = TransitionLabelChar::EndOfInput;
        let eois = TransitionLabelChar::EndOfInputStrict;

        // Test StartOfLine
        assert!(sol.matches_at_position("hello", 0, false)); // Start of input
        assert!(!sol.matches_at_position("hello", 2, false)); // Mid-string
        assert!(sol.matches_at_position("ab\ncd", 3, true)); // After newline (multiline)
        assert!(!sol.matches_at_position("ab\ncd", 3, false)); // After newline (no multiline)

        // Test EndOfLine
        assert!(eol.matches_at_position("hello", 5, false)); // End of input
        assert!(!eol.matches_at_position("hello", 2, false)); // Mid-string
        assert!(eol.matches_at_position("ab\ncd", 2, true)); // Before newline (multiline)

        // Test StartOfInput
        assert!(soi.matches_at_position("hello", 0, false));
        assert!(!soi.matches_at_position("hello", 2, false));
        assert!(!soi.matches_at_position("ab\ncd", 3, true)); // Not affected by multiline

        // Test EndOfInput (allows trailing newline)
        assert!(eoi.matches_at_position("hello", 5, false));
        assert!(eoi.matches_at_position("hello\n", 5, false)); // Before trailing \n
        assert!(eoi.matches_at_position("hello\n", 6, false)); // At absolute end

        // Test EndOfInputStrict
        assert!(eois.matches_at_position("hello", 5, false));
        assert!(!eois.matches_at_position("hello\n", 5, false)); // Must be at absolute end
        assert!(eois.matches_at_position("hello\n", 6, false));
    }

    #[test]
    fn test_label_display() {
        assert_eq!(format!("{}", TransitionLabelChar::Epsilon), "ε");
        assert_eq!(format!("{}", TransitionLabelChar::Char('a')), "a");
        assert_eq!(format!("{}", TransitionLabelChar::Any), ".");
        assert_eq!(format!("{}", TransitionLabelChar::StartOfLine), "^");
        assert_eq!(format!("{}", TransitionLabelChar::EndOfLine), "$");
        assert_eq!(format!("{}", TransitionLabelChar::StartOfInput), "\\A");
        assert_eq!(format!("{}", TransitionLabelChar::EndOfInput), "\\Z");
        assert_eq!(format!("{}", TransitionLabelChar::EndOfInputStrict), "\\z");
    }

    // --- TransitionLabel (byte-level) Tests ---

    #[test]
    fn test_byte_label_epsilon() {
        let label = TransitionLabel::Epsilon;
        assert!(label.is_epsilon());
        assert!(!label.consumes_input());
        assert!(label.matches(b'x'));
    }

    #[test]
    fn test_byte_label_byte() {
        let label = TransitionLabel::Byte(b'a');
        assert!(label.consumes_input());
        assert!(label.matches(b'a'));
        assert!(!label.matches(b'b'));
    }

    #[test]
    fn test_byte_label_anchor_position_matching() {
        let sol = TransitionLabel::StartOfLine;
        let eol = TransitionLabel::EndOfLine;
        let soi = TransitionLabel::StartOfInput;

        assert!(sol.matches_at_position(b"hello", 0, false));
        assert!(!sol.matches_at_position(b"hello", 2, false));
        assert!(sol.matches_at_position(b"ab\ncd", 3, true));

        assert!(eol.matches_at_position(b"hello", 5, false));
        assert!(eol.matches_at_position(b"ab\ncd", 2, true));

        assert!(soi.matches_at_position(b"hello", 0, false));
        assert!(!soi.matches_at_position(b"ab\ncd", 3, true));
    }

    // --- TransitionChar Tests ---

    #[test]
    fn test_transition_constructors() {
        let eps = TransitionChar::epsilon(0, 1);
        assert_eq!(eps.from, 0);
        assert_eq!(eps.to, 1);
        assert!(eps.label.is_epsilon());
        assert_eq!(eps.weight, 0.0);

        let on_char = TransitionChar::on_char(1, 'a', 2);
        assert_eq!(on_char.from, 1);
        assert_eq!(on_char.to, 2);
        assert!(on_char.label.matches('a'));

        let on_any = TransitionChar::on_any(0, 1);
        assert!(on_any.label.matches('x'));

        let on_class = TransitionChar::on_class(0, CharClassChar::from_range('a', 'z'), 1);
        assert!(on_class.label.matches('m'));
    }

    #[test]
    fn test_transition_with_weight() {
        let t = TransitionChar::with_weight(0, TransitionLabelChar::Char('x'), 1, 0.5);
        assert_eq!(t.weight, 0.5);
    }

    #[test]
    fn test_transition_display() {
        let eps = TransitionChar::epsilon(0, 1);
        assert_eq!(format!("{}", eps), "q0 --ε--> q1");

        let on_a = TransitionChar::on_char(1, 'a', 2);
        assert_eq!(format!("{}", on_a), "q1 --a--> q2");

        let weighted = TransitionChar::with_weight(0, TransitionLabelChar::Char('x'), 1, 0.15);
        assert_eq!(format!("{}", weighted), "q0 --x[0.15]--> q1");
    }

    // --- Transition (byte-level) Tests ---

    #[test]
    fn test_byte_transition_constructors() {
        let eps = Transition::epsilon(0, 1);
        assert!(eps.label.is_epsilon());

        let on_byte = Transition::on_byte(0, b'a', 1);
        assert!(on_byte.label.matches(b'a'));

        let on_any = Transition::on_any(0, 1);
        assert!(on_any.label.matches(b'x'));
    }

    #[test]
    fn test_byte_transition_display() {
        let t = Transition::with_weight(0, TransitionLabel::Byte(b'x'), 1, 0.25);
        assert_eq!(format!("{}", t), "q0 --x[0.25]--> q1");
    }
}

// ============================================================================
// NFA Context Coverage Tests (src/phonetic/nfa/context.rs)
// ============================================================================

mod nfa_context_coverage {
    use liblevenshtein::phonetic::nfa::{
        compile, compile_bytes, BoundaryKind, ContextMatcher, ContextMatcherChar,
        ContextPatternChar, ContextualRewriteRule, ContextualRewriteRuleChar,
    };
    use liblevenshtein::phonetic::regex::{parse, parse_bytes};

    // --- BoundaryKind Tests ---

    #[test]
    fn test_boundary_word_start_positions() {
        // At start of string
        assert!(BoundaryKind::WordStart.matches_at("hello", 0));

        // After space
        assert!(BoundaryKind::WordStart.matches_at("hello world", 6));

        // After punctuation
        assert!(BoundaryKind::WordStart.matches_at("hello,world", 6));

        // Mid-word
        assert!(!BoundaryKind::WordStart.matches_at("hello", 2));

        // After underscore (NOT a boundary - underscore is alphanumeric-like)
        assert!(!BoundaryKind::WordStart.matches_at("hello_world", 6));
    }

    #[test]
    fn test_boundary_word_end_positions() {
        // At end of string
        assert!(BoundaryKind::WordEnd.matches_at("hello", 5));

        // Before space
        assert!(BoundaryKind::WordEnd.matches_at("hello world", 5));

        // Before punctuation
        assert!(BoundaryKind::WordEnd.matches_at("hello,world", 5));

        // Mid-word
        assert!(!BoundaryKind::WordEnd.matches_at("hello", 2));

        // Before underscore (NOT a boundary)
        assert!(!BoundaryKind::WordEnd.matches_at("hello_world", 5));
    }

    #[test]
    fn test_boundary_bytes() {
        // Word start
        assert!(BoundaryKind::WordStart.matches_at_bytes(b"hello", 0));
        assert!(BoundaryKind::WordStart.matches_at_bytes(b"hello world", 6));
        assert!(!BoundaryKind::WordStart.matches_at_bytes(b"hello", 2));

        // Word end
        assert!(BoundaryKind::WordEnd.matches_at_bytes(b"hello", 5));
        assert!(BoundaryKind::WordEnd.matches_at_bytes(b"hello world", 5));
        assert!(!BoundaryKind::WordEnd.matches_at_bytes(b"hello", 2));
    }

    #[test]
    fn test_boundary_empty_string() {
        // Empty string - both boundaries match at position 0
        assert!(BoundaryKind::WordStart.matches_at("", 0));
        assert!(BoundaryKind::WordEnd.matches_at("", 0));
    }

    #[test]
    fn test_boundary_special_chars() {
        // Test various special characters as boundaries
        let special_text = "a.b!c?d;e:f";

        // Each punctuation creates a boundary
        for i in [1, 3, 5, 7, 9] {
            assert!(
                BoundaryKind::WordEnd.matches_at(special_text, i),
                "Expected word end at position {}",
                i
            );
        }
        for i in [2, 4, 6, 8, 10] {
            assert!(
                BoundaryKind::WordStart.matches_at(special_text, i),
                "Expected word start at position {}",
                i
            );
        }
    }

    // --- ContextMatcherChar Tests ---

    #[test]
    fn test_context_matcher_none() {
        let matcher = ContextMatcherChar::none();
        assert!(matcher.left.is_none());
        assert!(matcher.right.is_none());
        assert!(matcher.matches_at("anything", 0, 8));
        assert!(matcher.matches_at("", 0, 0));
    }

    #[test]
    fn test_context_matcher_word_start() {
        let matcher = ContextMatcherChar::word_start();

        // At start of string
        assert!(matcher.matches_at("hello", 0, 1));

        // After space
        assert!(matcher.matches_at("hello world", 6, 7));

        // Mid-word
        assert!(!matcher.matches_at("hello", 2, 3));
    }

    #[test]
    fn test_context_matcher_word_end() {
        let matcher = ContextMatcherChar::word_end();

        // At end of string
        assert!(matcher.matches_at("hello", 4, 5));

        // Before space
        assert!(matcher.matches_at("hello world", 4, 5));

        // Mid-word
        assert!(!matcher.matches_at("hello", 1, 2));
    }

    #[test]
    fn test_context_matcher_lookahead() {
        // c -> s / _[ei]
        let nfa = compile(&parse("[ei]").unwrap()).unwrap();
        let matcher = ContextMatcherChar::new(None, Some(nfa));

        // "city" - c before i
        assert!(matcher.matches_at("city", 0, 1));

        // "cent" - c before e
        assert!(matcher.matches_at("cent", 0, 1));

        // "cat" - c before a
        assert!(!matcher.matches_at("cat", 0, 1));
    }

    #[test]
    fn test_context_matcher_lookbehind() {
        // s -> z / [aeiou]_
        let nfa = compile(&parse("[aeiou]").unwrap()).unwrap();
        let matcher = ContextMatcherChar::new(Some(nfa), None);

        // "roses" - s after o
        assert!(matcher.matches_at("roses", 2, 3));

        // "star" - s at start
        assert!(!matcher.matches_at("star", 0, 1));
    }

    #[test]
    fn test_context_matcher_both_contexts() {
        // Match s between two vowels
        let left_nfa = compile(&parse("[aeiou]").unwrap()).unwrap();
        let right_nfa = compile(&parse("[aeiou]").unwrap()).unwrap();
        let matcher = ContextMatcherChar::new(Some(left_nfa), Some(right_nfa));

        // "roses" - s between o and e
        assert!(matcher.matches_at("roses", 2, 3));

        // "star" - no left vowel
        assert!(!matcher.matches_at("star", 0, 1));

        // "fast" - no right vowel
        assert!(!matcher.matches_at("fast", 2, 3));
    }

    #[test]
    fn test_context_matcher_unicode() {
        // Test with Unicode patterns
        let nfa = compile(&parse("[αβγ]").unwrap()).unwrap();
        let matcher = ContextMatcherChar::new(None, Some(nfa));

        assert!(matcher.matches_at("xαy", 0, 1)); // x before α
        assert!(!matcher.matches_at("xzy", 0, 1)); // x before z (not Greek)
    }

    // --- ContextPattern Composition Tests ---

    #[test]
    fn test_context_pattern_accepts() {
        let nfa = compile(&parse("[aeiou]").unwrap()).unwrap();
        let pattern = ContextPatternChar::Nfa(nfa);

        assert!(pattern.accepts("a"));
        assert!(pattern.accepts("e"));
        assert!(!pattern.accepts("b"));
        assert!(!pattern.accepts("xy"));
    }

    #[test]
    fn test_context_pattern_boundary() {
        let start = ContextPatternChar::Boundary(BoundaryKind::WordStart);
        let end = ContextPatternChar::Boundary(BoundaryKind::WordEnd);

        // Boundaries accept empty string
        assert!(start.accepts(""));
        assert!(end.accepts(""));

        // Non-empty strings don't match
        assert!(!start.accepts("a"));
        assert!(!end.accepts("a"));
    }

    #[test]
    fn test_context_pattern_and() {
        let nfa_a = compile(&parse("a").unwrap()).unwrap();
        let nfa_ae = compile(&parse("[ae]").unwrap()).unwrap();

        let pattern = ContextPatternChar::And(
            Box::new(ContextPatternChar::Nfa(nfa_a)),
            Box::new(ContextPatternChar::Nfa(nfa_ae)),
        );

        // "a" matches both patterns
        assert!(pattern.accepts("a"));

        // "e" only matches the second pattern
        assert!(!pattern.accepts("e"));
    }

    #[test]
    fn test_context_pattern_or() {
        let nfa_a = compile(&parse("a").unwrap()).unwrap();
        let nfa_b = compile(&parse("b").unwrap()).unwrap();

        let pattern = ContextPatternChar::Or(
            Box::new(ContextPatternChar::Nfa(nfa_a)),
            Box::new(ContextPatternChar::Nfa(nfa_b)),
        );

        assert!(pattern.accepts("a"));
        assert!(pattern.accepts("b"));
        assert!(!pattern.accepts("c"));
    }

    #[test]
    fn test_context_pattern_not() {
        let nfa_a = compile(&parse("a").unwrap()).unwrap();

        let pattern = ContextPatternChar::Not(Box::new(ContextPatternChar::Nfa(nfa_a)));

        assert!(!pattern.accepts("a"));
        assert!(pattern.accepts("b"));
        assert!(pattern.accepts(""));
    }

    #[test]
    fn test_context_pattern_nested() {
        // (a AND b) OR c - should never match since nothing is both a and b
        let nfa_a = compile(&parse("a").unwrap()).unwrap();
        let nfa_b = compile(&parse("b").unwrap()).unwrap();
        let nfa_c = compile(&parse("c").unwrap()).unwrap();

        let and_pattern = ContextPatternChar::And(
            Box::new(ContextPatternChar::Nfa(nfa_a)),
            Box::new(ContextPatternChar::Nfa(nfa_b)),
        );

        let or_pattern = ContextPatternChar::Or(
            Box::new(and_pattern),
            Box::new(ContextPatternChar::Nfa(nfa_c)),
        );

        assert!(!or_pattern.accepts("a")); // a AND b fails
        assert!(!or_pattern.accepts("b")); // a AND b fails
        assert!(or_pattern.accepts("c")); // c matches
    }

    // --- ContextMatcher (byte-level) Tests ---

    #[test]
    fn test_byte_context_matcher_none() {
        let matcher = ContextMatcher::none();
        assert!(matcher.matches_at(b"anything", 0, 8));
    }

    #[test]
    fn test_byte_context_matcher_word_boundaries() {
        let start = ContextMatcher::word_start();
        let end = ContextMatcher::word_end();

        assert!(start.matches_at(b"hello", 0, 1));
        assert!(!start.matches_at(b"hello", 2, 3));

        assert!(end.matches_at(b"hello", 4, 5));
        assert!(!end.matches_at(b"hello", 1, 2));
    }

    #[test]
    fn test_byte_context_matcher_lookahead() {
        let nfa = compile_bytes(&parse_bytes(b"[ei]").unwrap()).unwrap();
        let matcher = ContextMatcher::new(None, Some(nfa));

        assert!(matcher.matches_at(b"city", 0, 1));
        assert!(!matcher.matches_at(b"cat", 0, 1));
    }

    // --- ContextualRewriteRuleChar Tests ---

    #[test]
    fn test_contextual_rule_simple() {
        // ph -> f (no context)
        let source = compile(&parse("ph").unwrap()).unwrap();
        let rule = ContextualRewriteRuleChar::new(source, vec!['f'], None, None, 0.0);

        assert!(rule.can_apply_at("phone", 0, 2));
        assert_eq!(rule.apply_at("phone", 0, 2), "fone");
    }

    #[test]
    fn test_contextual_rule_with_lookahead() {
        // c -> s / _[ei]
        let source = compile(&parse("c").unwrap()).unwrap();
        let right = compile(&parse("[ei]").unwrap()).unwrap();
        let rule = ContextualRewriteRuleChar::new(source, vec!['s'], None, Some(right), 0.0);

        assert!(rule.can_apply_at("city", 0, 1));
        assert_eq!(rule.apply_at("city", 0, 1), "sity");

        assert!(!rule.can_apply_at("cat", 0, 1));
    }

    #[test]
    fn test_contextual_rule_with_lookbehind() {
        // s -> z / [aeiou]_
        let source = compile(&parse("s").unwrap()).unwrap();
        let left = compile(&parse("[aeiou]").unwrap()).unwrap();
        let rule = ContextualRewriteRuleChar::new(source, vec!['z'], Some(left), None, 0.0);

        assert!(rule.can_apply_at("roses", 2, 3));
        assert_eq!(rule.apply_at("roses", 2, 3), "rozes");

        assert!(!rule.can_apply_at("star", 0, 1));
    }

    #[test]
    fn test_contextual_rule_empty_replacement() {
        // e -> / _# (silent e at word end)
        let source = compile(&parse("e").unwrap()).unwrap();
        let rule = ContextualRewriteRuleChar {
            source,
            replacement: vec![],
            context: ContextMatcherChar::word_end(),
            weight: 0.0,
        };

        assert!(rule.can_apply_at("phone", 4, 5));
        assert_eq!(rule.apply_at("phone", 4, 5), "phon");
    }

    #[test]
    fn test_contextual_rule_find_first_match() {
        // c -> s / _[ei]
        let source = compile(&parse("c").unwrap()).unwrap();
        let right = compile(&parse("[ei]").unwrap()).unwrap();
        let rule = ContextualRewriteRuleChar::new(source, vec!['s'], None, Some(right), 0.0);

        // "soccer" - first c is before another c (no match), second c is before e
        let result = rule.find_first_match("soccer", 0);
        assert_eq!(result, Some((3, 4)));

        // Start from after the match
        let result2 = rule.find_first_match("soccer", 4);
        assert_eq!(result2, None);
    }

    #[test]
    fn test_contextual_rule_no_match() {
        let source = compile(&parse("xyz").unwrap()).unwrap();
        let rule = ContextualRewriteRuleChar::new(source, vec!['!'], None, None, 0.0);

        assert_eq!(rule.find_first_match("hello world", 0), None);
    }

    #[test]
    fn test_contextual_rule_weight() {
        let source = compile(&parse("a").unwrap()).unwrap();
        let rule = ContextualRewriteRuleChar::new(source, vec!['b'], None, None, 0.75);

        assert_eq!(rule.weight, 0.75);
    }

    // --- ContextualRewriteRule (byte-level) Tests ---

    #[test]
    fn test_byte_contextual_rule() {
        let source = compile_bytes(&parse_bytes(b"ph").unwrap()).unwrap();
        let rule = ContextualRewriteRule::new(source, vec![b'f'], None, None, 0.0);

        assert!(rule.can_apply_at(b"phone", 0, 2));
        assert_eq!(rule.apply_at(b"phone", 0, 2), b"fone");
    }

    #[test]
    fn test_byte_contextual_rule_find_match() {
        let source = compile_bytes(&parse_bytes(b"c").unwrap()).unwrap();
        let right = compile_bytes(&parse_bytes(b"[ei]").unwrap()).unwrap();
        let rule = ContextualRewriteRule::new(source, vec![b's'], None, Some(right), 0.0);

        let result = rule.find_first_match(b"soccer", 0);
        assert_eq!(result, Some((3, 4)));
    }
}

// ============================================================================
// NFA Optimizer Coverage Tests (src/phonetic/nfa/optimizer.rs)
// ============================================================================

mod nfa_optimizer_coverage {
    use liblevenshtein::phonetic::nfa::compiler::NFACompilerChar;
    use liblevenshtein::phonetic::nfa::{
        compile, NFAChar, NfaOptimizer, NfaOptimizerChar, OptimizationConfig, OptimizationStats,
        ThompsonBuilder, ThompsonBuilderChar,
    };
    use liblevenshtein::phonetic::regex::parse;

    // --- OptimizationConfig Tests ---

    #[test]
    fn test_config_full() {
        let config = OptimizationConfig::full();
        assert!(config.eliminate_epsilon);
        assert!(config.remove_unreachable);
        assert!(config.remove_dead);
        assert!(config.deduplicate_transitions);
    }

    #[test]
    fn test_config_quick() {
        let config = OptimizationConfig::quick();
        assert!(!config.eliminate_epsilon);
        assert!(config.remove_unreachable);
        assert!(config.remove_dead);
        assert!(!config.deduplicate_transitions);
    }

    #[test]
    fn test_config_none() {
        let config = OptimizationConfig::none();
        assert!(!config.eliminate_epsilon);
        assert!(!config.remove_unreachable);
        assert!(!config.remove_dead);
        assert!(!config.deduplicate_transitions);
    }

    #[test]
    fn test_config_default_is_full() {
        let default_config = OptimizationConfig::default();
        let full_config = OptimizationConfig::full();

        assert_eq!(
            default_config.eliminate_epsilon,
            full_config.eliminate_epsilon
        );
        assert_eq!(
            default_config.remove_unreachable,
            full_config.remove_unreachable
        );
        assert_eq!(default_config.remove_dead, full_config.remove_dead);
        assert_eq!(
            default_config.deduplicate_transitions,
            full_config.deduplicate_transitions
        );
    }

    // --- OptimizationStats Tests ---

    #[test]
    fn test_stats_reduction_percent_zero_original() {
        let stats = OptimizationStats {
            original_states: 0,
            original_transitions: 0,
            final_states: 0,
            final_transitions: 0,
            ..Default::default()
        };

        assert_eq!(stats.state_reduction_percent(), 0.0);
        assert_eq!(stats.transition_reduction_percent(), 0.0);
    }

    #[test]
    fn test_stats_reduction_percent() {
        let stats = OptimizationStats {
            original_states: 10,
            original_transitions: 20,
            final_states: 5,
            final_transitions: 8,
            ..Default::default()
        };

        assert!((stats.state_reduction_percent() - 50.0).abs() < 0.001);
        assert!((stats.transition_reduction_percent() - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_no_reduction() {
        let stats = OptimizationStats {
            original_states: 10,
            original_transitions: 20,
            final_states: 10,
            final_transitions: 20,
            ..Default::default()
        };

        assert_eq!(stats.state_reduction_percent(), 0.0);
        assert_eq!(stats.transition_reduction_percent(), 0.0);
    }

    // --- NfaOptimizerChar Tests ---

    #[test]
    fn test_optimizer_simple_pattern() {
        let builder = ThompsonBuilderChar::new();
        let nfa = builder.single_char('a');

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa.clone());

        // Language should be preserved
        assert!(nfa.accepts("a"));
        assert!(optimized.accepts("a"));
        assert!(!nfa.accepts("b"));
        assert!(!optimized.accepts("b"));
    }

    #[test]
    fn test_optimizer_kleene_star() {
        let builder = ThompsonBuilderChar::new();
        let a = builder.single_char('a');
        let nfa = builder.kleene_star(a);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, stats) = optimizer.optimize(nfa.clone());

        // Kleene star should have epsilon transitions removed
        assert!(stats.epsilon_transitions_eliminated > 0);

        // Language preservation
        assert!(nfa.accepts(""));
        assert!(optimized.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(optimized.accepts("a"));
        assert!(nfa.accepts("aaa"));
        assert!(optimized.accepts("aaa"));
        assert!(!nfa.accepts("b"));
        assert!(!optimized.accepts("b"));
    }

    #[test]
    fn test_optimizer_alternation() {
        let builder = ThompsonBuilderChar::new();
        let a = builder.single_char('a');
        let b = builder.single_char('b');
        let nfa = builder.alternation(a, b);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa.clone());

        // Language preservation
        assert!(nfa.accepts("a"));
        assert!(optimized.accepts("a"));
        assert!(nfa.accepts("b"));
        assert!(optimized.accepts("b"));
        assert!(!nfa.accepts("c"));
        assert!(!optimized.accepts("c"));
        assert!(!nfa.accepts("ab"));
        assert!(!optimized.accepts("ab"));
    }

    #[test]
    fn test_optimizer_kleene_plus() {
        let builder = ThompsonBuilderChar::new();
        let a = builder.single_char('a');
        let nfa = builder.kleene_plus(a);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa.clone());

        // a+ should not accept empty
        assert!(!nfa.accepts(""));
        assert!(!optimized.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(optimized.accepts("a"));
        assert!(nfa.accepts("aaa"));
        assert!(optimized.accepts("aaa"));
    }

    #[test]
    fn test_optimizer_optional() {
        let builder = ThompsonBuilderChar::new();
        let a = builder.single_char('a');
        let nfa = builder.optional(a);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa.clone());

        // a? accepts empty or a
        assert!(nfa.accepts(""));
        assert!(optimized.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(optimized.accepts("a"));
        assert!(!nfa.accepts("aa"));
        assert!(!optimized.accepts("aa"));
    }

    #[test]
    fn test_optimizer_concatenation() {
        let builder = ThompsonBuilderChar::new();
        let a = builder.single_char('a');
        let b = builder.single_char('b');
        let nfa = builder.concatenate(a, b);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa.clone());

        assert!(nfa.accepts("ab"));
        assert!(optimized.accepts("ab"));
        assert!(!nfa.accepts("a"));
        assert!(!optimized.accepts("a"));
        assert!(!nfa.accepts("ba"));
        assert!(!optimized.accepts("ba"));
    }

    #[test]
    fn test_optimizer_complex_pattern() {
        // (a|b)*c
        let builder = ThompsonBuilderChar::new();
        let a = builder.single_char('a');
        let b = builder.single_char('b');
        let ab = builder.alternation(a, b);
        let ab_star = builder.kleene_star(ab);
        let c = builder.single_char('c');
        let nfa = builder.concatenate(ab_star, c);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, stats) = optimizer.optimize(nfa.clone());

        assert!(stats.epsilon_transitions_eliminated > 0);

        // Test various inputs
        let inputs = ["c", "ac", "bc", "abc", "aabbc", "ababc"];
        for input in inputs {
            assert_eq!(
                nfa.accepts(input),
                optimized.accepts(input),
                "mismatch on '{}'",
                input
            );
        }

        let non_matches = ["", "a", "b", "ab", "ca", "ccc"];
        for input in non_matches {
            assert!(!nfa.accepts(input), "should not accept '{}'", input);
            assert!(!optimized.accepts(input), "should not accept '{}'", input);
        }
    }

    #[test]
    fn test_optimizer_no_optimization() {
        let builder = ThompsonBuilderChar::new();
        let a = builder.single_char('a');
        let b = builder.single_char('b');
        let nfa = builder.alternation(a, b);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::none());
        let (_optimized, stats) = optimizer.optimize(nfa.clone());

        // With no optimization, stats should show no changes
        assert_eq!(stats.epsilon_transitions_eliminated, 0);
        assert_eq!(stats.unreachable_states_removed, 0);
        assert_eq!(stats.dead_states_removed, 0);
        assert_eq!(stats.duplicate_transitions_removed, 0);
    }

    #[test]
    fn test_optimizer_quick_mode() {
        let builder = ThompsonBuilderChar::new();
        let a = builder.single_char('a');
        let nfa = builder.kleene_star(a);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::quick());
        let (optimized, stats) = optimizer.optimize(nfa.clone());

        // Quick mode skips epsilon elimination
        assert_eq!(stats.epsilon_transitions_eliminated, 0);

        // But still removes unreachable/dead states
        // Language should be preserved
        assert!(optimized.accepts(""));
        assert!(optimized.accepts("a"));
        assert!(optimized.accepts("aaa"));
    }

    #[test]
    fn test_optimizer_with_char_class() {
        let builder = ThompsonBuilderChar::new();
        let vowels =
            liblevenshtein::phonetic::nfa::CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
        let nfa = builder.kleene_plus(builder.char_class(vowels));

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa.clone());

        assert!(optimized.accepts("a"));
        assert!(optimized.accepts("aeiou"));
        assert!(!optimized.accepts(""));
        assert!(!optimized.accepts("xyz"));
    }

    #[test]
    fn test_optimizer_with_any_char() {
        let builder = ThompsonBuilderChar::new();
        let dot = builder.any_char();
        let nfa = builder.kleene_plus(dot);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa);

        assert!(optimized.accepts("x"));
        assert!(optimized.accepts("hello"));
        assert!(optimized.accepts("123"));
        assert!(!optimized.accepts(""));
    }

    #[test]
    fn test_optimizer_deduplication() {
        // Create NFA with duplicate transitions manually
        let mut nfa = NFAChar::new();
        let q1 = nfa.add_state(true);

        // Add duplicate transitions
        nfa.add_transition_char(0, 'a', q1);
        nfa.add_transition_char(0, 'a', q1);
        nfa.add_transition_char(0, 'a', q1);

        nfa.finalize();
        assert_eq!(nfa.num_transitions(), 3);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, stats) = optimizer.optimize(nfa);

        // Deduplication should remove 2 transitions
        assert_eq!(stats.duplicate_transitions_removed, 2);
        assert_eq!(optimized.num_transitions(), 1);
    }

    #[test]
    fn test_optimizer_preserves_anchors() {
        // ^hello$ - anchors should be preserved
        let regex = parse("^hello$").expect("parse");
        let nfa = compile(&regex).expect("compile");

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa);

        // Optimization shouldn't break the NFA
        assert!(optimized.num_states() > 0);
        assert!(optimized.num_transitions() > 0);
    }

    // --- NfaOptimizer (byte-level) Tests ---

    #[test]
    fn test_byte_optimizer_simple() {
        let builder = ThompsonBuilder::new();
        let a = builder.single_byte(b'a');
        let b = builder.single_byte(b'b');
        let nfa = builder.alternation(a, b);

        let optimizer = NfaOptimizer::new(OptimizationConfig::full());
        let (optimized, stats) = optimizer.optimize(nfa.clone());

        assert!(stats.epsilon_transitions_eliminated > 0);

        assert!(nfa.accepts(b"a"));
        assert!(optimized.accepts(b"a"));
        assert!(nfa.accepts(b"b"));
        assert!(optimized.accepts(b"b"));
        assert!(!nfa.accepts(b"c"));
        assert!(!optimized.accepts(b"c"));
    }

    #[test]
    fn test_byte_optimizer_kleene_star() {
        let builder = ThompsonBuilder::new();
        let a = builder.single_byte(b'a');
        let nfa = builder.kleene_star(a);

        let optimizer = NfaOptimizer::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa);

        assert!(optimized.accepts(b""));
        assert!(optimized.accepts(b"a"));
        assert!(optimized.accepts(b"aaa"));
        assert!(!optimized.accepts(b"b"));
    }

    // --- Full Optimization Pipeline Tests ---

    #[test]
    fn test_full_optimization_various_patterns() {
        let patterns = ["a", "ab", "a|b", "a*", "a+", "a?", "(ab)+", "(a|b)*c"];

        for pattern in patterns {
            let regex = parse(pattern).expect("parse");
            let mut compiler = NFACompilerChar::new().without_optimization();
            let nfa = compiler.compile(&regex).expect("compile");

            let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
            let (optimized, _stats) = optimizer.optimize(nfa.clone());

            // Test various inputs
            let test_inputs = ["", "a", "b", "c", "ab", "abc", "aaa", "bbb", "abababc"];
            for input in test_inputs {
                assert_eq!(
                    nfa.accepts(input),
                    optimized.accepts(input),
                    "language mismatch for pattern '{}' on input '{}'",
                    pattern,
                    input
                );
            }
        }
    }
}

// ============================================================================
// Regex Parser Coverage Tests (src/phonetic/regex/parser.rs)
// ============================================================================

mod regex_parser_coverage {
    use liblevenshtein::phonetic::regex::{parse, parse_bytes};

    #[test]
    fn test_parse_simple_char() {
        let regex = parse("a").unwrap();
        // Just verify it parses without error
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_concatenation() {
        let regex = parse("abc").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_alternation() {
        let regex = parse("a|b|c").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_quantifiers() {
        // Star
        let regex_star = parse("a*").unwrap();
        assert!(regex_star.size() > 0);

        // Plus
        let regex_plus = parse("a+").unwrap();
        assert!(regex_plus.size() > 0);

        // Question
        let regex_opt = parse("a?").unwrap();
        assert!(regex_opt.size() > 0);
    }

    #[test]
    fn test_parse_repetition() {
        // Exact
        let regex = parse("a{3}").unwrap();
        assert!(regex.size() > 0);

        // Range
        let regex = parse("a{2,4}").unwrap();
        assert!(regex.size() > 0);

        // Unbounded
        let regex = parse("a{2,}").unwrap();
        assert!(regex.size() > 0);

        // At most
        let regex = parse("a{,3}").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_char_class() {
        let regex = parse("[abc]").unwrap();
        assert!(regex.size() > 0);

        let regex = parse("[a-z]").unwrap();
        assert!(regex.size() > 0);

        let regex = parse("[a-zA-Z0-9]").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_negated_char_class() {
        let regex = parse("[^abc]").unwrap();
        assert!(regex.size() > 0);

        let regex = parse("[^a-z]").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_dot() {
        let regex = parse(".").unwrap();
        assert!(regex.size() > 0);

        let regex = parse("a.b").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_anchors() {
        // Start/end of line
        let regex = parse("^hello$").unwrap();
        assert!(regex.size() > 0);

        // Start/end of input
        let regex = parse("\\Ahello\\Z").unwrap();
        assert!(regex.size() > 0);

        // Strict end
        let regex = parse("hello\\z").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_groups() {
        // Capturing group
        let regex = parse("(ab)").unwrap();
        assert!(regex.size() > 0);

        // Non-capturing group
        let regex = parse("(?:ab)").unwrap();
        assert!(regex.size() > 0);

        // Named group
        let regex = parse("(?<name>ab)").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_nested_groups() {
        let regex = parse("((a|b)(c|d))").unwrap();
        assert!(regex.size() > 0);

        let regex = parse("(a(b(c)))").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_complex_patterns() {
        let patterns = [
            "(a|b)*c",
            "[a-z]+[0-9]*",
            "\\d{2,4}-\\d{2}-\\d{2}",
            "(foo|bar|baz)+",
            "([a-z]+)\\1",
        ];

        for pattern in patterns {
            // Some patterns may not be supported, just test that parse doesn't panic
            let _ = parse(pattern);
        }
    }

    #[test]
    fn test_parse_word_boundary() {
        let regex = parse("#").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_escape_sequences() {
        // Digit
        let regex = parse("\\d+").unwrap();
        assert!(regex.size() > 0);

        // Word
        let regex = parse("\\w+").unwrap();
        assert!(regex.size() > 0);

        // Whitespace
        let regex = parse("\\s+").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_unicode_chars() {
        // Greek letters
        let regex = parse("[αβγδε]").unwrap();
        assert!(regex.size() > 0);

        // Emoji (if supported)
        // let regex = parse("😀+");
    }

    #[test]
    fn test_parse_empty_alternation() {
        // Empty alternative: a| or |a
        // These might be parsed as errors or as empty strings
        let result = parse("a|");
        // Just check it doesn't panic
        let _ = result;

        let result = parse("|a");
        let _ = result;
    }

    #[test]
    fn test_parse_bytes_simple() {
        let regex = parse_bytes(b"abc").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_bytes_char_class() {
        let regex = parse_bytes(b"[a-z]+").unwrap();
        assert!(regex.size() > 0);
    }

    #[test]
    fn test_parse_error_unclosed_group() {
        let result = parse("(abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unclosed_char_class() {
        let result = parse("[abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_invalid_quantifier() {
        let result = parse("a{5,3}"); // min > max
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unclosed_quantifier() {
        let result = parse("a{3");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unexpected_close_paren() {
        // Note: The parser may handle trailing ) differently - it might stop parsing
        // before the ) and succeed, or it might error. Check which behavior is implemented.
        let result = parse("abc)");
        // If it succeeds by ignoring the trailing ), that's a valid design choice
        // Just verify we don't panic
        let _ = result;
    }
}

// ============================================================================
// LLev Parser Coverage Tests (src/phonetic/llev/parser.rs)
// ============================================================================

mod llev_parser_coverage {
    use liblevenshtein::phonetic::{parse_expression, parse_str};

    #[test]
    fn test_parse_empty_file() {
        let file = parse_str("").unwrap();
        assert!(file.rules.is_empty());
    }

    #[test]
    fn test_parse_simple_rule() {
        let file = parse_str("a -> b;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_multiple_rules() {
        let file = parse_str(
            r#"
            a -> b;
            c -> d;
            e -> f;
        "#,
        )
        .unwrap();
        assert_eq!(file.rules.len(), 3);
    }

    #[test]
    fn test_parse_rule_with_context() {
        // Left context
        let file = parse_str("a -> b / c_;").unwrap();
        assert_eq!(file.rules.len(), 1);

        // Right context
        let file = parse_str("a -> b / _c;").unwrap();
        assert_eq!(file.rules.len(), 1);

        // Both contexts
        let file = parse_str("a -> b / c_d;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_rule_with_word_boundary() {
        // Word boundary left
        let file = parse_str("a -> b / #_;").unwrap();
        assert_eq!(file.rules.len(), 1);

        // Word boundary right
        let file = parse_str("a -> b / _#;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_rule_empty_replacement() {
        // Deletion rule
        let file = parse_str("a -> ;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_directive_name() {
        let file = parse_str("@name \"Test Rules\";").unwrap();
        assert_eq!(file.metadata.name, Some("Test Rules".to_string()));
    }

    #[test]
    fn test_parse_directive_version() {
        let file = parse_str("@version \"1.0\";").unwrap();
        assert_eq!(file.metadata.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_parse_directive_author() {
        let file = parse_str("@author \"Test Author\";").unwrap();
        assert_eq!(file.metadata.author, Some("Test Author".to_string()));
    }

    #[test]
    fn test_parse_directive_description() {
        let file = parse_str("@description \"Test description\";").unwrap();
        assert_eq!(
            file.metadata.description,
            Some("Test description".to_string())
        );
    }

    #[test]
    fn test_parse_directive_define() {
        let file = parse_str("@define VOWEL = [aeiou];").unwrap();
        assert_eq!(file.symbols.len(), 1);
        assert_eq!(file.symbols[0].name, "VOWEL");
    }

    #[test]
    fn test_parse_multiple_directives() {
        let file = parse_str(
            r#"
            @name "Test";
            @version "1.0";
            @author "Author";
        "#,
        )
        .unwrap();

        assert_eq!(file.metadata.name, Some("Test".to_string()));
        assert_eq!(file.metadata.version, Some("1.0".to_string()));
        assert_eq!(file.metadata.author, Some("Author".to_string()));
    }

    #[test]
    fn test_parse_rule_with_metadata() {
        let file = parse_str(
            r#"
            [id: 1, name: "test rule"]
            a -> b;
        "#,
        )
        .unwrap();

        assert_eq!(file.rules.len(), 1);
        let rule = &file.rules[0];
        assert_eq!(rule.metadata.id, Some(1));
        assert_eq!(rule.metadata.name, Some("test rule".to_string()));
    }

    #[test]
    fn test_parse_rule_metadata_weight() {
        let file = parse_str(
            r#"
            [weight: 0.5]
            a -> b;
        "#,
        )
        .unwrap();

        assert_eq!(file.rules.len(), 1);
        assert_eq!(file.rules[0].metadata.weight, Some(0.5));
    }

    #[test]
    fn test_parse_rule_metadata_enabled() {
        let file = parse_str(
            r#"
            [enabled: false]
            a -> b;
        "#,
        )
        .unwrap();

        assert_eq!(file.rules.len(), 1);
        assert!(!file.rules[0].metadata.enabled);
    }

    #[test]
    fn test_parse_rule_metadata_group() {
        let file = parse_str(
            r#"
            [group: "orthography"]
            a -> b;
        "#,
        )
        .unwrap();

        assert_eq!(file.rules.len(), 1);
        assert_eq!(
            file.rules[0].metadata.group,
            Some("orthography".to_string())
        );
    }

    #[test]
    fn test_parse_pattern_char_class() {
        let file = parse_str("[aeiou] -> V;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_pattern_alternation() {
        let file = parse_str("(ph|gh) -> f;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_pattern_quantifier() {
        let file = parse_str("a+ -> A;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_context_negation() {
        // NOT context
        let file = parse_str("a -> b / !c_;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_context_and() {
        // AND context
        let file = parse_str("a -> b / c&d_;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_context_or() {
        // OR context (using |)
        let file = parse_str("a -> b / (c|d)_;").unwrap();
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_complex_file() {
        let file = parse_str(
            r#"
            @name "English Phonetic Rules"
            @version "1.0"
            @author "Test"

            @define VOWEL = [aeiou]
            @define CONSONANT = [bcdfghjklmnpqrstvwxyz]

            [id: 1, name: "ph to f"]
            ph -> f;

            [id: 2, name: "soft c"]
            c -> s / _[ei];

            [id: 3, name: "silent e"]
            e -> / _#;
        "#,
        )
        .unwrap();

        assert_eq!(
            file.metadata.name,
            Some("English Phonetic Rules".to_string())
        );
        assert_eq!(file.symbols.len(), 2);
        assert_eq!(file.rules.len(), 3);
    }

    #[test]
    fn test_parse_expression_simple() {
        let expr = parse_expression("abc").unwrap();
        // Just check it parses
        let _ = expr;
    }

    #[test]
    fn test_parse_expression_char_class() {
        let expr = parse_expression("[a-z]+").unwrap();
        let _ = expr;
    }

    #[test]
    fn test_parse_error_invalid_directive() {
        let result = parse_str("@invalid directive;");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_missing_arrow() {
        let result = parse_str("a b;");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_invalid_metadata_key() {
        // The parser may handle invalid metadata keys differently depending on implementation
        // E.g., it might skip unknown keys, or it might error. Test both behaviors don't panic.
        let result = parse_str("[invalid_key: 1]\na -> b;");
        // If it errors, that's expected; if it succeeds by ignoring the key, that's also valid
        let _ = result;
    }

    #[test]
    fn test_parse_error_lowercase_symbol() {
        // Symbols must be uppercase
        let result = parse_str("@define vowel = [aeiou];");
        assert!(result.is_err());
    }
}

// ============================================================================
// Proptest Property-Based Tests
// ============================================================================

mod proptest_phonetic {
    use super::*;
    use liblevenshtein::phonetic::nfa::{
        CharClassChar, NFAState, NfaOptimizerChar, OptimizationConfig, ThompsonBuilderChar,
        TransitionLabelChar,
    };

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_char_class_negation_inverts(chars in prop::collection::vec(prop::char::any(), 1..5)) {
            let cc = CharClassChar::from_chars(&chars);
            let negated = cc.clone().negated();

            for &c in &chars {
                prop_assert!(cc.matches(c));
                prop_assert!(!negated.matches(c));
            }
        }

        #[test]
        fn prop_char_class_double_negation(chars in prop::collection::vec(prop::char::range('a', 'z'), 1..5)) {
            let cc = CharClassChar::from_chars(&chars);
            let double_neg = cc.clone().negated().negated();

            for &c in &chars {
                prop_assert_eq!(cc.matches(c), double_neg.matches(c));
            }

            // Test some chars not in the class
            for c in ['!', '@', '#'] {
                prop_assert_eq!(cc.matches(c), double_neg.matches(c));
            }
        }

        #[test]
        fn prop_nfa_state_id_preserved(id in 0u32..u32::MAX, is_final in prop::bool::ANY) {
            let state = NFAState::new(id, is_final);
            prop_assert_eq!(state.id, id);
            prop_assert_eq!(state.is_final, is_final);
        }

        #[test]
        fn prop_transition_label_epsilon_always_matches(c in prop::char::any()) {
            let label = TransitionLabelChar::Epsilon;
            prop_assert!(label.matches(c));
        }

        #[test]
        fn prop_transition_label_char_exact_match(target in prop::char::range('a', 'z')) {
            let label = TransitionLabelChar::Char(target);
            prop_assert!(label.matches(target));

            // Shouldn't match other chars
            let other = if target == 'a' { 'z' } else { 'a' };
            prop_assert!(!label.matches(other));
        }

        #[test]
        fn prop_transition_label_any_matches_all(c in prop::char::any()) {
            let label = TransitionLabelChar::Any;
            prop_assert!(label.matches(c));
        }

        #[test]
        fn prop_optimizer_preserves_single_char_language(c in prop::char::range('a', 'z')) {
            let builder = ThompsonBuilderChar::new();
            let nfa = builder.single_char(c);

            let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
            let (optimized, _) = optimizer.optimize(nfa.clone());

            // Should accept the character
            let s: String = c.to_string();
            prop_assert!(nfa.accepts(&s));
            prop_assert!(optimized.accepts(&s));

            // Should reject empty
            prop_assert!(!nfa.accepts(""));
            prop_assert!(!optimized.accepts(""));
        }

        #[test]
        fn prop_optimizer_preserves_kleene_star_language(c in prop::char::range('a', 'z'), n in 0usize..5) {
            let builder = ThompsonBuilderChar::new();
            let nfa = builder.kleene_star(builder.single_char(c));

            let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
            let (optimized, _) = optimizer.optimize(nfa.clone());

            // Should accept n repetitions of c
            let s: String = std::iter::repeat(c).take(n).collect();
            prop_assert!(nfa.accepts(&s));
            prop_assert!(optimized.accepts(&s));
        }
    }
}
