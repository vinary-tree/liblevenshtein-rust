//! Tests for the `.llev` recursive descent parser.

use super::super::ast::{ContextExpr, Expression, SyllableCondition, SyllableExpr};
use super::super::error::LLevErrorKind;
use super::*;

#[test]
fn test_parse_simple_expression() {
    let expr = parse_expression("abc").expect("test: parse_expression abc");
    // Should be Concat(Concat(Char('a'), Char('b')), Char('c'))
    match expr {
        Expression::Concat(left, right) => match (*left, *right) {
            (Expression::Concat(ll, lr), Expression::Char('c')) => {
                assert!(matches!(*ll, Expression::Char('a')));
                assert!(matches!(*lr, Expression::Char('b')));
            }
            _ => panic!("unexpected structure"),
        },
        _ => panic!("expected Concat"),
    }
}

#[test]
fn test_parse_alternation() {
    let expr = parse_expression("a|b").expect("test: parse_expression a|b");
    match expr {
        Expression::Alt(left, right) => {
            assert!(matches!(*left, Expression::Char('a')));
            assert!(matches!(*right, Expression::Char('b')));
        }
        _ => panic!("expected Alt"),
    }
}

#[test]
fn test_parse_char_class() {
    let expr = parse_expression("[aeiou]").expect("test: parse_expression [aeiou]");
    match expr {
        Expression::CharClass { chars, negated } => {
            assert!(!negated);
            assert_eq!(chars, vec!['a', 'e', 'i', 'o', 'u']);
        }
        _ => panic!("expected CharClass"),
    }
}

#[test]
fn test_parse_char_class_negated() {
    let expr = parse_expression("[^aeiou]").expect("test: parse_expression [^aeiou]");
    match expr {
        Expression::CharClass { chars, negated } => {
            assert!(negated);
            assert_eq!(chars, vec!['a', 'e', 'i', 'o', 'u']);
        }
        _ => panic!("expected negated CharClass"),
    }
}

#[test]
fn test_parse_char_range() {
    let expr = parse_expression("[a-c]").expect("test: parse_expression [a-c]");
    match expr {
        Expression::CharClass { chars, negated } => {
            assert!(!negated);
            assert_eq!(chars, vec!['a', 'b', 'c']);
        }
        _ => panic!("expected CharClass with range"),
    }
}

#[test]
fn test_parse_quantifiers() {
    let star = parse_expression("a*").expect("test: parse_expression a*");
    assert!(matches!(star, Expression::Star(_)));

    let plus = parse_expression("a+").expect("test: parse_expression a+");
    assert!(matches!(plus, Expression::Plus(_)));

    let opt = parse_expression("a?").expect("test: parse_expression a?");
    assert!(matches!(opt, Expression::Optional(_)));
}

#[test]
fn test_parse_counted_quantifier() {
    let exact = parse_expression("a{3}").expect("test: parse_expression a{3}");
    match exact {
        Expression::RepeatExact(_, n) => {
            assert_eq!(n, 3);
        }
        _ => panic!("expected RepeatExact, got {:?}", exact),
    }

    let range = parse_expression("a{2,4}").expect("test: parse_expression a{2,4}");
    match range {
        Expression::RepeatRange { min, max, .. } => {
            assert_eq!(min, 2);
            assert_eq!(max, Some(4));
        }
        _ => panic!("expected RepeatRange, got {:?}", range),
    }

    let min_only = parse_expression("a{2,}").expect("test: parse_expression a{2,}");
    match min_only {
        Expression::RepeatRange { min, max, .. } => {
            assert_eq!(min, 2);
            assert_eq!(max, None);
        }
        _ => panic!("expected RepeatRange, got {:?}", min_only),
    }
}

#[test]
fn test_parse_group() {
    let expr = parse_expression("(ab)").expect("test: parse_expression (ab)");
    match expr {
        Expression::Concat(left, right) => {
            assert!(matches!(*left, Expression::Char('a')));
            assert!(matches!(*right, Expression::Char('b')));
        }
        _ => panic!("expected group contents"),
    }
}

#[test]
fn test_parse_any() {
    let expr = parse_expression(".").expect("test: parse_expression .");
    assert!(matches!(expr, Expression::Any));
}

#[test]
fn test_parse_word_boundary() {
    let expr = parse_expression("#").expect("test: parse_expression #");
    assert!(matches!(expr, Expression::WordBoundary));
}

#[test]
fn test_parse_symbol_ref() {
    // Test symbol reference with defined symbol - expands to the definition
    let input = r#"
@define VOWEL = [aeiou]
$VOWEL -> x
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    // The symbol ref is expanded to the character class at parse time
    let rule = &file.rules[0].rule;
    match &rule.pattern {
        Expression::CharClass { chars, negated } => {
            assert!(!negated);
            assert_eq!(chars.len(), 5);
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'e'));
            assert!(chars.contains(&'i'));
            assert!(chars.contains(&'o'));
            assert!(chars.contains(&'u'));
        }
        _ => panic!("expected CharClass, got {:?}", rule.pattern),
    }
}

#[test]
fn test_parse_empty_file() {
    let file = parse_str("").expect("test: parse_str empty input");
    assert!(file.rules.is_empty());
    assert!(file.symbols.is_empty());
    assert!(file.includes.is_empty());
}

#[test]
fn test_parse_file_metadata() {
    let input = r#"
@name "Test Rules"
@version "1.0"
@author "Test Author"
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.metadata.name, Some("Test Rules".to_string()));
    assert_eq!(file.metadata.version, Some("1.0".to_string()));
    assert_eq!(file.metadata.author, Some("Test Author".to_string()));
}

#[test]
fn test_parse_define_directive() {
    // Use MY_VOWEL to avoid conflict with built-in "vowel" class
    let input = "@define MY_VOWEL = [aeiou]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.symbols.len(), 1);
    assert_eq!(file.symbols[0].name, "MY_VOWEL");
}

#[test]
fn test_parse_include_directive() {
    let input = r#"@include "other.llev""#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.includes.len(), 1);
    assert_eq!(file.includes[0].path, "other.llev");
}

#[test]
fn test_parse_simple_rule() {
    let input = "ph -> f";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    // Pattern should be 'p' concat 'h'
    match &rule.pattern {
        Expression::Concat(left, right) => {
            assert!(matches!(**left, Expression::Char('p')));
            assert!(matches!(**right, Expression::Char('h')));
        }
        _ => panic!("expected Concat for pattern"),
    }
    // Replacement should be 'f'
    assert!(matches!(rule.replacement, Expression::Char('f')));
}

#[test]
fn test_parse_rule_with_context() {
    let input = "c -> s / _[ei]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    assert!(matches!(rule.pattern, Expression::Char('c')));
    assert!(matches!(rule.replacement, Expression::Char('s')));
    assert!(rule.context.is_some());
    let ctx = rule.context.as_ref().expect("should have context");
    assert!(ctx.left.is_none());
    assert!(ctx.right.is_some());
}

#[test]
fn test_parse_deletion_rule() {
    let input = "gh -> ;";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    assert!(matches!(rule.replacement, Expression::Empty));
}

#[test]
fn test_parse_rule_with_metadata() {
    let input = r#"[id: 1, name: "ph to f"]
ph -> f"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let def = &file.rules[0];
    assert_eq!(def.metadata.id, Some(1));
    assert_eq!(def.metadata.name, Some("ph to f".to_string()));
}

#[test]
fn test_parse_multiple_rules() {
    let input = r#"
ph -> f
gh ->
c -> k
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 3);
}

#[test]
fn test_parse_complex_pattern() {
    let expr = parse_expression("(a|b)*c+d?").expect("test: parse_expression (a|b)*c+d?");
    // Should parse without error
    assert!(matches!(expr, Expression::Concat(_, _)));
}

#[test]
fn test_parse_metadata_group_as_identifier() {
    let input = r#"[id: 1, name: "test", group: orthography]
ph -> f"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);
    let def = &file.rules[0];
    assert_eq!(def.metadata.group, Some("orthography".to_string()));
}

#[test]
fn test_parse_metadata_group_as_string() {
    // This test verifies that group can be specified as a string literal
    let input = r#"[id: 1, name: "test", group: "orthography"]
ph -> f"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);
    let def = &file.rules[0];
    assert_eq!(def.metadata.group, Some("orthography".to_string()));
}

#[test]
fn test_parse_metadata_ipa_phonemic() {
    // Test parsing IPA phonemic transcription (with slashes)
    let input = r#"[id: 1, name: "sch to sh", ipa: "/ʃ/"]
sch -> sh"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);
    let def = &file.rules[0];
    assert_eq!(def.metadata.ipa, Some("/ʃ/".to_string()));
}

#[test]
fn test_parse_metadata_ipa_phonetic() {
    // Test parsing IPA phonetic transcription (with brackets)
    let input = r#"[id: 2, name: "ch to x", ipa: "[x]"]
ch -> X / V_"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);
    let def = &file.rules[0];
    assert_eq!(def.metadata.ipa, Some("[x]".to_string()));
}

#[test]
fn test_parse_metadata_ipa_with_all_fields() {
    // Test parsing IPA with all other metadata fields
    let input = r#"[id: 100, name: "tsch affricate", weight: 0.5, group: german_consonants, ipa: "/t͡ʃ/"]
tsch -> tsh"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);
    let def = &file.rules[0];
    assert_eq!(def.metadata.id, Some(100));
    assert_eq!(def.metadata.name, Some("tsch affricate".to_string()));
    assert_eq!(def.metadata.weight, Some(0.5));
    assert_eq!(def.metadata.group, Some("german_consonants".to_string()));
    assert_eq!(def.metadata.ipa, Some("/t͡ʃ/".to_string()));
}

#[test]
fn test_parse_inline_weight_suffix() {
    let file = parse_str("c -> s [0.3];").expect("test: parse_str input");

    assert_eq!(file.rules.len(), 1);
    assert_eq!(file.rules[0].rule.weight, Some(0.3));
}

#[test]
fn test_parse_inline_weight_suffix_without_terminator() {
    let file = parse_str("c -> s [0.3]").expect("test: parse_str input");

    assert_eq!(file.rules.len(), 1);
    assert_eq!(file.rules[0].rule.weight, Some(0.3));
}

#[test]
fn test_parse_replacement_without_terminator_after_weight_lookahead() {
    let file = parse_str("ph -> f").expect("test: parse_str input");

    assert_eq!(file.rules.len(), 1);
    assert!(matches!(
        file.rules[0].rule.replacement,
        Expression::Char('f')
    ));
    assert!(file.rules[0].rule.weight.is_none());
}

#[test]
fn test_parse_inline_weight_suffix_after_context() {
    let file = parse_str("c -> s / _[ei] [0.25];").expect("test: parse_str input");

    assert_eq!(file.rules.len(), 1);
    assert_eq!(file.rules[0].rule.weight, Some(0.25));
    assert!(file.rules[0].rule.context.is_some());
}

#[test]
fn test_parse_inline_weight_suffix_after_empty_right_context() {
    let file = parse_str("e ->  / _ [0.125];").expect("test: parse_str input");

    assert_eq!(file.rules.len(), 1);
    assert_eq!(file.rules[0].rule.weight, Some(0.125));
    let context = file.rules[0]
        .rule
        .context
        .as_ref()
        .expect("rule should have context");
    assert!(context.left.is_none());
    assert!(context.right.is_none());
}

#[test]
fn test_parse_inline_weight_suffix_after_syllable_context() {
    let file = parse_str("e ->  / _ if final_syllable [0.2];").expect("test: parse_str input");

    assert_eq!(file.rules.len(), 1);
    assert_eq!(file.rules[0].rule.weight, Some(0.2));
    let context = file.rules[0]
        .rule
        .context
        .as_ref()
        .expect("rule should have context");
    assert!(matches!(
        context.syllable,
        Some(SyllableExpr::Cond(SyllableCondition::FinalSyllable))
    ));
}

#[test]
fn test_digit_leading_char_classes_are_not_weights() {
    let file = parse_str("d -> [0-9] / _[0-9];").expect("test: parse_str input");

    assert_eq!(file.rules.len(), 1);
    assert!(file.rules[0].rule.weight.is_none());
    assert!(matches!(
        file.rules[0].rule.replacement,
        Expression::CharClass { .. }
    ));

    let context = file.rules[0]
        .rule
        .context
        .as_ref()
        .expect("rule should have context");
    assert!(matches!(
        context.right.as_deref(),
        Some(ContextExpr::Pattern(Expression::CharClass { .. }))
    ));
}

#[test]
fn test_parse_inline_weight_keeps_metadata_weight_separate() {
    let file = parse_str(
        r#"
        [weight: 0.9]
        c -> s [0.3];
        "#,
    )
    .expect("test: parse_str input");

    assert_eq!(file.rules.len(), 1);
    assert_eq!(file.rules[0].metadata.weight, Some(0.9));
    assert_eq!(file.rules[0].rule.weight, Some(0.3));
}

#[test]
fn test_parse_escaped_uppercase_literal() {
    // Test that \B in a pattern becomes a literal 'B' character
    // Note: Using \B because \A is now the affricate shortcut
    let input = r#"\B -> b"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    // Pattern should be literal 'B'
    assert!(matches!(rule.pattern, Expression::Char('B')));
    // Replacement should be literal 'b'
    assert!(matches!(rule.replacement, Expression::Char('b')));
}

#[test]
fn test_parse_string_literal_for_uppercase() {
    // Test that "ABC" in a pattern works for literal uppercase strings
    // This is an alternative to using escape sequences
    let input = r#""ABC" -> abc"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    // Pattern should be Concat(Concat(Char('A'), Char('B')), Char('C'))
    match &rule.pattern {
        Expression::Concat(left, right) => {
            assert!(matches!(**right, Expression::Char('C')));
            match &**left {
                Expression::Concat(ll, lr) => {
                    assert!(matches!(**ll, Expression::Char('A')));
                    assert!(matches!(**lr, Expression::Char('B')));
                }
                _ => panic!("expected nested Concat"),
            }
        }
        _ => panic!("expected Concat for pattern"),
    }
}

#[test]
fn test_parse_mixed_escaped_and_regular() {
    // Test mixing escaped uppercase with regular patterns
    // Note: Using \B because \A is now the affricate shortcut
    let input = r#"\Bcd -> bcd"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    // Pattern should be Concat(Concat(Char('B'), Char('c')), Char('d'))
    match &rule.pattern {
        Expression::Concat(left, right) => {
            assert!(matches!(**right, Expression::Char('d')));
            match &**left {
                Expression::Concat(ll, lr) => {
                    assert!(matches!(**ll, Expression::Char('B')));
                    assert!(matches!(**lr, Expression::Char('c')));
                }
                _ => panic!("expected nested Concat"),
            }
        }
        _ => panic!("expected Concat for pattern"),
    }
}

// ==================== Compound Context Tests ====================

#[test]
fn test_parse_context_word_boundary_initial() {
    let input = "wr -> r / #_";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    // Left should be word boundary
    assert!(matches!(
        ctx.left.as_deref(),
        Some(ContextExpr::WordBoundary)
    ));
    assert!(ctx.right.is_none());
}

#[test]
fn test_parse_context_word_boundary_final() {
    let input = "e -> / _#";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    assert!(ctx.left.is_none());
    // Right should be word boundary
    assert!(matches!(
        ctx.right.as_deref(),
        Some(ContextExpr::WordBoundary)
    ));
}

#[test]
fn test_parse_context_not() {
    let input = "c -> k / _![ei]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    assert!(ctx.left.is_none());
    // Right should be NOT
    match ctx.right.as_deref() {
        Some(ContextExpr::Not(inner)) => {
            assert!(matches!(**inner, ContextExpr::Pattern(_)));
        }
        _ => panic!("expected Not context"),
    }
}

#[test]
fn test_parse_context_and() {
    let input = "x -> gz / [aeiou]&[aeiou]_";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    // Left should be AND
    match ctx.left.as_deref() {
        Some(ContextExpr::And(_, _)) => {}
        _ => panic!("expected And context"),
    }
    assert!(ctx.right.is_none());
}

#[test]
fn test_parse_context_or() {
    let input = "e -> / _([bcdfg]|#)";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    assert!(ctx.left.is_none());
    // Right should be OR (grouped)
    match ctx.right.as_deref() {
        Some(ContextExpr::Or(_, _)) => {}
        _ => panic!("expected Or context, got {:?}", ctx.right),
    }
}

#[test]
fn test_parse_context_complex_compound() {
    // NOT has higher precedence than AND, which is higher than OR
    let input = "t -> d / ![x]&[aeiou]|[ou]_";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    // Left should be OR at the top level
    match ctx.left.as_deref() {
        Some(ContextExpr::Or(_, _)) => {}
        _ => panic!("expected Or context at top level"),
    }
}

// ==================== Syllable Condition Tests ====================

#[test]
fn test_parse_syllable_monosyllable() {
    let input = "y -> i / _# if monosyllable";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    match &ctx.syllable {
        Some(SyllableExpr::Cond(SyllableCondition::Monosyllable)) => {}
        _ => panic!("expected monosyllable condition"),
    }
}

#[test]
fn test_parse_syllable_polysyllable() {
    let input = "y -> i / _# if polysyllable";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    match &ctx.syllable {
        Some(SyllableExpr::Cond(SyllableCondition::Polysyllable)) => {}
        _ => panic!("expected polysyllable condition"),
    }
}

#[test]
fn test_parse_syllable_and() {
    let input = "y -> i / _# if polysyllable & final_syllable";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    match &ctx.syllable {
        Some(SyllableExpr::And(_, _)) => {}
        _ => panic!("expected And syllable condition"),
    }
}

#[test]
fn test_parse_syllable_or() {
    let input = "a -> aa / _ if open_syllable | initial_syllable";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    match &ctx.syllable {
        Some(SyllableExpr::Or(_, _)) => {}
        _ => panic!("expected Or syllable condition"),
    }
}

#[test]
fn test_parse_syllable_not() {
    let input = "e -> i / _[bcdfg] if !final_syllable";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    match &ctx.syllable {
        Some(SyllableExpr::Not(_)) => {}
        _ => panic!("expected Not syllable condition"),
    }
}

#[test]
fn test_parse_syllable_complex() {
    let input = "t -> d / [aeiou]_[bcdfg] if !monosyllable & !final_syllable";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    // Should be And at top level with two Not conditions
    match &ctx.syllable {
        Some(SyllableExpr::And(left, right)) => {
            assert!(matches!(**left, SyllableExpr::Not(_)));
            assert!(matches!(**right, SyllableExpr::Not(_)));
        }
        _ => panic!("expected And of two Not conditions"),
    }
}

#[test]
fn test_parse_syllable_with_parens() {
    let input = "y -> i / _# if (monosyllable | polysyllable) & final_syllable";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    // Should be And at top level with Or on left side
    match &ctx.syllable {
        Some(SyllableExpr::And(left, _)) => {
            assert!(matches!(**left, SyllableExpr::Or(_, _)));
        }
        _ => panic!("expected And with Or on left"),
    }
}

#[test]
fn test_parse_all_syllable_keywords() {
    // Test all six syllable keywords parse correctly
    let keywords = [
        ("monosyllable", SyllableCondition::Monosyllable),
        ("polysyllable", SyllableCondition::Polysyllable),
        ("open_syllable", SyllableCondition::OpenSyllable),
        ("closed_syllable", SyllableCondition::ClosedSyllable),
        ("final_syllable", SyllableCondition::FinalSyllable),
        ("initial_syllable", SyllableCondition::InitialSyllable),
    ];

    for (kw, expected_cond) in keywords {
        let input = format!("a -> b / _ if {}", kw);
        let file = parse_str(&input).expect(&format!("failed to parse {}", kw));
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        match &ctx.syllable {
            Some(SyllableExpr::Cond(cond)) => {
                assert_eq!(
                    std::mem::discriminant(cond),
                    std::mem::discriminant(&expected_cond),
                    "keyword {} didn't produce expected condition",
                    kw
                );
            }
            _ => panic!("expected Cond for keyword {}", kw),
        }
    }
}

#[test]
fn test_parse_context_only_no_syllable() {
    // Verify that rules without syllable clause still work
    let input = "c -> s / _[ei]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");
    assert!(ctx.syllable.is_none());
}

#[test]
fn test_parse_no_context_no_syllable() {
    // Rules without context also shouldn't have syllable
    let input = "ph -> f";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    assert!(rule.context.is_none());
}

#[test]
fn test_parse_user_symbol_in_char_class() {
    // $ is now a LITERAL character inside character classes
    // [$MY_VOWEL] = literal chars: $, M, Y, _, V, O, W, E, L
    let input = r#"
@define MY_VOWEL = [aeiou]
x -> gz / [$abc]_[AB]
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    // Left context should be a char class with literal $, a, b, c
    if let Some(ref boxed) = ctx.left {
        if let ContextExpr::Pattern(ref expr) = **boxed {
            if let Expression::CharClass { ref chars, negated } = expr {
                assert!(!negated);
                // Should contain literal $
                assert!(chars.contains(&'$'), "Should contain literal $");
                // Should contain literal a, b, c
                assert!(chars.contains(&'a'), "Should contain literal a");
                assert!(chars.contains(&'b'), "Should contain literal b");
                assert!(chars.contains(&'c'), "Should contain literal c");
            } else {
                panic!("expected CharClass in left context");
            }
        } else {
            panic!("expected Pattern in left context");
        }
    } else {
        panic!("expected Some in left context");
    }
}

#[test]
fn test_parse_negated_user_symbol_in_char_class() {
    // Use \$ for literal $ in negated char class: [^\$ab] = negated class with $, a, b
    let input = r#"
c -> k / _[^\$ab]
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    // Right context should be a negated char class with $, a, b
    if let Some(ref boxed) = ctx.right {
        if let ContextExpr::Pattern(ref expr) = **boxed {
            if let Expression::CharClass { ref chars, negated } = expr {
                assert!(negated);
                assert!(chars.contains(&'$'));
                assert!(chars.contains(&'a'));
                assert!(chars.contains(&'b'));
            } else {
                panic!("expected CharClass in right context");
            }
        } else {
            panic!("expected Pattern in right context");
        }
    } else {
        panic!("expected Some in right context");
    }
}

#[test]
fn test_parse_mixed_symbol_and_chars_in_class() {
    // $ is now a LITERAL character inside character classes
    // [$xyz] contains literal $, x, y, z
    let input = r#"
@define M = [mn]
a -> b / _[$xyz]
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    // Right context should have literal $, x, y, z
    if let Some(ref boxed) = ctx.right {
        if let ContextExpr::Pattern(ref expr) = **boxed {
            if let Expression::CharClass { ref chars, negated } = expr {
                assert!(!negated);
                // Literal $
                assert!(chars.contains(&'$'), "Should contain literal $");
                // Literal chars
                assert!(chars.contains(&'x'));
                assert!(chars.contains(&'y'));
                assert!(chars.contains(&'z'));
            } else {
                panic!("expected CharClass in right context");
            }
        } else {
            panic!("expected Pattern in right context");
        }
    } else {
        panic!("expected Some in right context");
    }
}

#[test]
fn test_parse_dollar_as_literal_in_char_class() {
    // $ is a literal character inside character classes
    // This is standard regex convention
    let input = "a -> b / _[$]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    if let Some(ref boxed) = ctx.right {
        if let ContextExpr::Pattern(ref expr) = **boxed {
            if let Expression::CharClass { ref chars, negated } = expr {
                assert!(!negated);
                assert_eq!(chars.len(), 1);
                assert!(chars.contains(&'$'));
            } else {
                panic!("expected CharClass");
            }
        } else {
            panic!("expected Pattern");
        }
    } else {
        panic!("expected right context");
    }
}

#[test]
fn test_parse_empty_named_class_error() {
    // Test error for empty named class [::]
    // This tests POSIX syntax with empty name
    let input = "a -> b / _[[::]xyz]";
    let result = parse_str(input);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, LLevErrorKind::InvalidPattern(_)));
}

#[test]
fn test_parse_symbol_reference_outside_char_class() {
    // Symbol references ($NAME) work outside character classes
    // but inside char classes, $ is literal
    let input = r#"
@define MY_VOWEL = [aeiou]
$MY_VOWEL -> X
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    // Pattern should be the vowels from MY_VOWEL
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
        assert!(chars.contains(&'i'));
        assert!(chars.contains(&'o'));
        assert!(chars.contains(&'u'));
    } else {
        panic!("expected CharClass");
    }
}

// ==================== Built-in Named Class Tests ====================

#[test]
fn test_parse_builtin_vowel_class() {
    // Test built-in [:vowel:] class
    let input = "c -> s / _[[:vowel:]]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    if let Some(ref boxed) = ctx.right {
        if let ContextExpr::Pattern(ref expr) = **boxed {
            if let Expression::CharClass { ref chars, negated } = expr {
                assert!(!negated);
                // Should contain ASCII vowels
                assert!(chars.contains(&'a'));
                assert!(chars.contains(&'e'));
                assert!(chars.contains(&'i'));
                assert!(chars.contains(&'o'));
                assert!(chars.contains(&'u'));
                // Should contain uppercase
                assert!(chars.contains(&'A'));
                // Should contain IPA vowels
                assert!(chars.contains(&'ə'));
            } else {
                panic!("expected CharClass");
            }
        } else {
            panic!("expected Pattern");
        }
    } else {
        panic!("expected Some");
    }
}

#[test]
fn test_parse_builtin_full_word_alias() {
    // Test full-word alias [:plosive:] (alias for stop)
    let input = "[[:plosive:]] -> 0";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, negated } = rule.pattern {
        assert!(!negated);
        // plosive should be alias for stop
        assert!(chars.contains(&'p'));
        assert!(chars.contains(&'t'));
        assert!(chars.contains(&'k'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_builtin_case_insensitive() {
    // Test case-insensitive lookup [:VOWEL:] and [:Vowel:]
    let inputs = vec![
        "c -> s / _[[:VOWEL:]]",
        "c -> s / _[[:Vowel:]]",
        "c -> s / _[[:vowel:]]",
    ];

    for input in inputs {
        let file = parse_str(input).expect(&format!("should parse: {}", input));
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        if let Some(ref boxed) = ctx.right {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, .. } = expr {
                    assert!(chars.contains(&'a'));
                }
            }
        }
    }
}

#[test]
fn test_parse_builtin_consonant_class() {
    // Test built-in [:consonant:] class
    let input = "[[:consonant:]] -> 0";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, negated } = rule.pattern {
        assert!(!negated);
        // Should contain ASCII consonants
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'c'));
        assert!(chars.contains(&'d'));
        // Should NOT contain vowels
        assert!(!chars.contains(&'a'));
        assert!(!chars.contains(&'e'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_builtin_posix_alpha() {
    // Test POSIX [:alpha:] class
    let input = "[[:alpha:]] -> x";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, .. } = rule.pattern {
        // Should have all letters
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'z'));
        assert!(chars.contains(&'A'));
        assert!(chars.contains(&'Z'));
        // Should NOT have digits
        assert!(!chars.contains(&'0'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_builtin_mixed_with_chars() {
    // Test mixing built-in class with extra chars: [[:vowel:]y]
    let input = "c -> s / _[[:vowel:]y]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    if let Some(ref boxed) = ctx.right {
        if let ContextExpr::Pattern(ref expr) = **boxed {
            if let Expression::CharClass { ref chars, .. } = expr {
                // Should have vowels AND 'y'
                assert!(chars.contains(&'a'));
                assert!(chars.contains(&'e'));
                assert!(chars.contains(&'y'));
            } else {
                panic!("expected CharClass");
            }
        }
    }
}

#[test]
fn test_parse_define_lowercase_symbol_rejected() {
    // Test that lowercase symbol names are rejected
    let input = "@define vowel = [aeiou]";
    let result = parse_str(input);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err.kind,
        LLevErrorKind::SymbolNameMustBeUppercase { .. }
    ));
}

#[test]
fn test_parse_define_uppercase_required() {
    // Test that only UPPERCASE symbol names are accepted
    // These should fail (contain lowercase)
    let invalid_inputs = vec![
        "@define vowel = [aeiou]", // all lowercase
        "@define Vowel = [aeiou]", // mixed case
        "@define alpha = [abc]",   // all lowercase
    ];

    for input in invalid_inputs {
        let result = parse_str(input);
        assert!(result.is_err(), "should error on: {}", input);
        let err = result.unwrap_err();
        assert!(
            matches!(err.kind, LLevErrorKind::SymbolNameMustBeUppercase { .. }),
            "expected SymbolNameMustBeUppercase for: {}",
            input
        );
    }

    // These should succeed (all uppercase)
    let valid_inputs = vec![
        "@define VOWEL = [aeiou]",
        "@define V = [aeiou]",
        "@define MY_CLASS = [abc]",
    ];

    for input in valid_inputs {
        let result = parse_str(input);
        assert!(
            result.is_ok(),
            "should succeed for: {}, got: {:?}",
            input,
            result.err()
        );
    }
}

#[test]
fn test_parse_user_defined_non_conflicting() {
    // Test that user can define symbols that don't conflict with built-ins
    // User symbols use $ sigil
    let input = r#"
@define MY_VOWELS = [aeiou]
c -> s / _[$MY_VOWELS]
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.symbols.len(), 1);
    assert_eq!(file.rules.len(), 1);
}

#[test]
fn test_parse_builtin_front_vowel() {
    // Test front vowel class (full name)
    let input = "c -> s / _[[:front_vowel:]]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    if let Some(ref boxed) = ctx.right {
        if let ContextExpr::Pattern(ref expr) = **boxed {
            if let Expression::CharClass { ref chars, .. } = expr {
                // Front vowels: e, i
                assert!(chars.contains(&'e'));
                assert!(chars.contains(&'i'));
                // Should NOT have back vowels
                assert!(!chars.contains(&'o'));
                assert!(!chars.contains(&'u'));
            } else {
                panic!("expected CharClass");
            }
        }
    }
}

#[test]
fn test_parse_builtin_stop_consonant() {
    // Test stop consonant class (full name)
    let input = "[[:stop:]] -> 0";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, .. } = rule.pattern {
        // Stop consonants: p, b, t, d, k, g
        assert!(chars.contains(&'p'));
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'t'));
        assert!(chars.contains(&'d'));
        assert!(chars.contains(&'k'));
        assert!(chars.contains(&'g'));
        // Should NOT have fricatives
        assert!(!chars.contains(&'f'));
        assert!(!chars.contains(&'s'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_builtin_ascii_vowel() {
    // Test ASCII-only vowel subset
    let input = "[[:ascii_vowel:]] -> V";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, .. } = rule.pattern {
        // Should have ASCII vowels
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
        // Should NOT have IPA vowels
        assert!(!chars.contains(&'ə'));
        assert!(!chars.contains(&'ɪ'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_builtin_ipa_vowel() {
    // Test IPA-only vowel subset
    let input = "[[:ipa_vowel:]] -> V";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, .. } = rule.pattern {
        // Should have IPA vowels
        assert!(chars.contains(&'ə'));
        assert!(chars.contains(&'ɪ'));
        // Should NOT have ASCII vowels
        assert!(!chars.contains(&'a'));
        assert!(!chars.contains(&'e'));
    } else {
        panic!("expected CharClass");
    }
}

// ==================== Standalone Named Class Syntax Tests ====================

#[test]
fn test_parse_standalone_vowel_class() {
    // Test standalone [:vowel:] syntax (shorthand for [[:vowel:]])
    let input = "[:vowel:] -> V";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, negated } = rule.pattern {
        assert!(!negated);
        // Should contain vowels
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
        assert!(chars.contains(&'ə'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_standalone_vowel() {
    // Test standalone [:vowel:] syntax
    let input = "[:vowel:] -> V";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, .. } = rule.pattern {
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_standalone_in_context() {
    // Test standalone syntax in context
    let input = "c -> s / _[:front_vowel:]";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    if let Some(ref boxed) = ctx.right {
        if let ContextExpr::Pattern(ref expr) = **boxed {
            if let Expression::CharClass { ref chars, .. } = expr {
                // front_vowel = front vowels (e, i)
                assert!(chars.contains(&'e'));
                assert!(chars.contains(&'i'));
                assert!(!chars.contains(&'o'));
            } else {
                panic!("expected CharClass");
            }
        } else {
            panic!("expected Pattern");
        }
    } else {
        panic!("expected right context");
    }
}

#[test]
fn test_parse_standalone_consonant() {
    // Test standalone [:consonant:] in pattern
    let input = "[:consonant:][:vowel:] -> CV";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    // Pattern should be Concat(CharClass(consonants), CharClass(vowels))
    match &rule.pattern {
        Expression::Concat(left, right) => {
            if let Expression::CharClass { chars: l_chars, .. } = &**left {
                assert!(l_chars.contains(&'b'));
                assert!(l_chars.contains(&'c'));
                assert!(!l_chars.contains(&'a'));
            } else {
                panic!("expected left CharClass");
            }
            if let Expression::CharClass { chars: r_chars, .. } = &**right {
                assert!(r_chars.contains(&'a'));
                assert!(r_chars.contains(&'e'));
                assert!(!r_chars.contains(&'b'));
            } else {
                panic!("expected right CharClass");
            }
        }
        _ => panic!("expected Concat"),
    }
}

#[test]
fn test_parse_standalone_posix_digit() {
    // Test standalone [:digit:] POSIX class
    let input = "[:digit:] -> 0";
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::CharClass { ref chars, .. } = rule.pattern {
        assert!(chars.contains(&'0'));
        assert!(chars.contains(&'5'));
        assert!(chars.contains(&'9'));
        assert!(!chars.contains(&'a'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_standalone_user_symbol() {
    // Test user symbol with $ sigil as pattern
    let input = r#"
@define MY_CHARS = [xyz]
$MY_CHARS -> 0
"#;
    let file = parse_str(input).expect("test: parse_str input");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    // Pattern is expanded to CharClass at parse time
    if let Expression::CharClass { ref chars, .. } = rule.pattern {
        assert_eq!(chars.len(), 3);
        assert!(chars.contains(&'x'));
        assert!(chars.contains(&'y'));
        assert!(chars.contains(&'z'));
    } else {
        panic!("expected CharClass, got {:?}", rule.pattern);
    }
}

#[test]
fn test_parse_standalone_undefined_symbol_error() {
    // Test error for undefined symbol with $ sigil
    let input = "$UNDEFINED_CLASS -> x";
    let result = parse_str(input);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, LLevErrorKind::UndefinedSymbol(_)));
}

#[test]
fn test_parse_standalone_empty_error() {
    // Test error for empty standalone named class [::]
    let input = "[::]-> x";
    let result = parse_str(input);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err.kind, LLevErrorKind::InvalidPattern(_)));
}

#[test]
fn test_parse_standalone_vs_posix() {
    // Test that both syntaxes produce the same result
    let standalone = parse_str("[:vowel:] -> V").expect("test: parse [:vowel:] -> V");
    let posix = parse_str("[[:vowel:]] -> V").expect("test: parse [[:vowel:]] -> V");

    let standalone_rule = &standalone.rules[0].rule;
    let posix_rule = &posix.rules[0].rule;

    if let (
        Expression::CharClass { chars: s_chars, .. },
        Expression::CharClass { chars: p_chars, .. },
    ) = (&standalone_rule.pattern, &posix_rule.pattern)
    {
        assert_eq!(s_chars.len(), p_chars.len());
        for c in s_chars {
            assert!(p_chars.contains(c));
        }
    } else {
        panic!("expected CharClass for both");
    }
}

#[test]
fn test_parse_inline_named_class() {
    // [x:vowel:] is NOW parsed as literals: x, :, v, o, w, e, l, :
    // To union with named class, use: [x[[:vowel:]]]
    let file = parse_str("[x:vowel:] -> X").expect("test: parse [x:vowel:] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // All chars are literals
        assert!(chars.contains(&'x'));
        assert!(chars.contains(&':'));
        assert!(chars.contains(&'v'));
        assert!(chars.contains(&'o'));
        assert!(chars.contains(&'w'));
        assert!(chars.contains(&'e'));
        assert!(chars.contains(&'l'));
        // Should NOT contain 'a', 'i', 'u' (vowels not from named class)
        assert!(!chars.contains(&'a'));
        assert!(!chars.contains(&'i'));
        assert!(!chars.contains(&'u'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_inline_named_class_correct_syntax() {
    // [x[:vowel:]] is the correct syntax for x union with named class
    // Note: [x[:vowel:]] is equivalent to [x[[:vowel:]]]
    let file = parse_str("[x[:vowel:]] -> X").expect("test: parse [x[:vowel:]] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        assert!(chars.contains(&'x'));
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
        assert!(chars.contains(&'i'));
        assert!(chars.contains(&'o'));
        assert!(chars.contains(&'u'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_inline_named_class_bracket_equivalence() {
    // All these are equivalent:
    // [x[:vowel:]] = [x[[:vowel:]]] = [x[[[:vowel:]]]]
    let file1 = parse_str("[x[:vowel:]] -> X").expect("test: parse file1");
    let file2 = parse_str("[x[[:vowel:]]] -> X").expect("test: parse file2");
    let file3 = parse_str("[x[[[:vowel:]]]] -> X").expect("test: parse file3");
    if let (
        Expression::CharClass { chars: chars1, .. },
        Expression::CharClass { chars: chars2, .. },
        Expression::CharClass { chars: chars3, .. },
    ) = (
        &file1.rules[0].rule.pattern,
        &file2.rules[0].rule.pattern,
        &file3.rules[0].rule.pattern,
    ) {
        // All should have the same chars
        assert_eq!(chars1.len(), chars2.len());
        assert_eq!(chars1.len(), chars3.len());
        for c in chars1 {
            assert!(chars2.contains(c));
            assert!(chars3.contains(c));
        }
    } else {
        panic!("expected CharClass for all");
    }
}

#[test]
fn test_parse_inline_named_class_with_symbol() {
    // $ is now a LITERAL character inside character classes
    // [$abc:xyz:] = literal $, a, b, c, :, x, y, z, :
    let input = r#"
@define ABC = [abc]
[$abc:xyz:] -> X
"#;
    let file = parse_str(input).expect("test: parse_str input");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // Literal $
        assert!(chars.contains(&'$'), "Should contain literal $");
        // Literal a, b, c
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'c'));
        // : is literal
        assert!(chars.contains(&':'));
        // x, y, z from the middle part
        assert!(chars.contains(&'x'));
        assert!(chars.contains(&'y'));
        assert!(chars.contains(&'z'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_chars_union_named_class() {
    // [xyz[:consonant:]] unions literal chars with named class
    let file = parse_str("[xyz[:consonant:]] -> X").expect("test: parse [xyz[:consonant:]] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // Should have literal x, y, z
        assert!(chars.contains(&'x'));
        assert!(chars.contains(&'y'));
        assert!(chars.contains(&'z'));
        // Should have consonants from named class
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'c'));
        assert!(chars.contains(&'d'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_negated_nested_class() {
    // Test [^[:vowel:]] syntax - negated vowels = consonants + other chars
    let file = parse_str("[[^:vowel:]] -> C").expect("test: parse [[^:vowel:]] -> C");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        // The char class itself is not negated, but contains negated vowels
        assert!(!negated);
        // Should NOT contain vowels
        assert!(!chars.contains(&'a'));
        assert!(!chars.contains(&'e'));
        assert!(!chars.contains(&'i'));
        assert!(!chars.contains(&'o'));
        assert!(!chars.contains(&'u'));
        // Should contain consonants and other printable ASCII
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'c'));
        assert!(chars.contains(&'!'));
        assert!(chars.contains(&' '));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_nested_class_union() {
    // Test union of nested classes with literal chars + named class
    // Since $ is literal inside char classes, use [abc[:consonant:]] syntax
    let file = parse_str("[abc[:consonant:]] -> X").expect("test: parse [abc[:consonant:]] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // Should have literal a, b, c
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'c'));
        // Should have consonants from named class (includes b, c, d, etc.)
        assert!(chars.contains(&'d'));
        assert!(chars.contains(&'f'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_deeply_nested_class() {
    // Test arbitrary nesting: [[[[:vowel:]]]]
    let file = parse_str("[[[[:vowel:]]]] -> V").expect("test: parse [[[[:vowel:]]]] -> V");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
        assert!(chars.contains(&'i'));
        assert!(chars.contains(&'o'));
        assert!(chars.contains(&'u'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_colon_as_literal() {
    // Bare ':' without closing ':' should be treated as literal
    let file = parse_str("[:abc] -> X").expect("test: parse [:abc] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // ':' 'a' 'b' 'c' should all be literals
        assert!(chars.contains(&':'));
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'c'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_escaped_brackets_in_char_class() {
    // \[ and \] should be literal brackets inside character class
    let file = parse_str(r"[a\[\]b] -> X").expect("test: parse [a\\[\\]b] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'['));
        assert!(chars.contains(&']'));
        assert!(chars.contains(&'b'));
        assert_eq!(chars.len(), 4);
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_parse_escaped_vs_unescaped_bracket() {
    // [ starts nested class, \[ is literal
    let file = parse_str(r"[\[[[:vowel:]]] -> X").expect("test: parse [\\[[[:vowel:]]] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // Should have literal '[' from \[
        assert!(chars.contains(&'['));
        // Should have vowels from nested class
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
        assert!(chars.contains(&'i'));
        assert!(chars.contains(&'o'));
        assert!(chars.contains(&'u'));
    } else {
        panic!("expected CharClass");
    }
}

// =========================================================================
// Feature Bundle Tests
// =========================================================================

#[test]
fn test_feature_bundle_voiced_stop() {
    // [:voiced stop:] should give only voiced stops: b, d, g
    let file = parse_str("[:voiced stop:] -> V").expect("test: parse [:voiced stop:] -> V");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // Voiced stops
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'d'));
        assert!(chars.contains(&'g'));
        // NOT voiceless stops
        assert!(!chars.contains(&'p'));
        assert!(!chars.contains(&'t'));
        assert!(!chars.contains(&'k'));
        // NOT non-stops
        assert!(!chars.contains(&'v'));
        assert!(!chars.contains(&'z'));
        assert!(!chars.contains(&'a'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_feature_bundle_negated_nasal_stop() {
    // [:!nasal stop:] should give oral stops: p, t, k, b, d, g
    let file = parse_str("[:!nasal stop:] -> S").expect("test: parse [:!nasal stop:] -> S");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // All stops
        assert!(chars.contains(&'p'));
        assert!(chars.contains(&'t'));
        assert!(chars.contains(&'k'));
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'d'));
        assert!(chars.contains(&'g'));
        // NOT nasals
        assert!(!chars.contains(&'m'));
        assert!(!chars.contains(&'n'));
        // NOT fricatives
        assert!(!chars.contains(&'f'));
        assert!(!chars.contains(&'s'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_feature_bundle_single_negated() {
    // [:!nasal:] should give everything except nasals
    let file = parse_str("[:!nasal:] -> X").expect("test: parse [:!nasal:] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // NOT nasals
        assert!(!chars.contains(&'m'));
        assert!(!chars.contains(&'n'));
        // Other consonants should be included
        assert!(chars.contains(&'p'));
        assert!(chars.contains(&'b'));
        // Vowels should be included
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_feature_bundle_three_features() {
    // [:high front vowel:] should give high front vowels only
    let file = parse_str("[:high_vowel front_vowel vowel:] -> I")
        .expect("test: parse multi-feature class");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // High front vowels
        assert!(chars.contains(&'i'));
        assert!(chars.contains(&'I'));
        // NOT low vowels
        assert!(!chars.contains(&'a'));
        // NOT back vowels
        assert!(!chars.contains(&'u'));
        // NOT consonants
        assert!(!chars.contains(&'p'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_feature_bundle_nested_syntax() {
    // Nested syntax [[:voiced stop:]] should also work
    let file =
        parse_str("[x[[:voiced stop:]]] -> X").expect("test: parse [x[[:voiced stop:]]] -> X");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // Should have 'x'
        assert!(chars.contains(&'x'));
        // Voiced stops
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'d'));
        assert!(chars.contains(&'g'));
        // NOT voiceless stops
        assert!(!chars.contains(&'p'));
    } else {
        panic!("expected CharClass");
    }
}

#[test]
fn test_feature_bundle_backwards_compatible() {
    // Single term should still work: [:stop:] = all stops
    let file = parse_str("[:stop:] -> S").expect("test: parse [:stop:] -> S");
    let rule = &file.rules[0].rule;
    if let Expression::CharClass { chars, negated } = &rule.pattern {
        assert!(!negated);
        // All stops
        assert!(chars.contains(&'p'));
        assert!(chars.contains(&'t'));
        assert!(chars.contains(&'k'));
        assert!(chars.contains(&'b'));
        assert!(chars.contains(&'d'));
        assert!(chars.contains(&'g'));
    } else {
        panic!("expected CharClass");
    }
}

// ========================================================================
// Tests for scoped flags (case-sensitive opt-out)
// ========================================================================

#[test]
fn test_parse_scoped_flags_case_sensitive_c() {
    // (?c:...) - case-sensitive pattern
    let file = parse_str("(?c:ABC) -> abc").expect("test: parse (?c:ABC) -> abc");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::ScopedFlags { flags, inner } = &rule.pattern {
        // Should have case_insensitive = Some(false) (case-sensitive)
        assert_eq!(flags.case_insensitive, Some(false));
        // Inner should be concatenation of A, B, C
        if let Expression::Concat(_, _) = **inner {
            // Good - it's a concat
        } else if let Expression::Char(_) = **inner {
            // Single char also ok for short patterns
        } else {
            panic!(
                "expected Concat or Char inside ScopedFlags, got {:?}",
                inner
            );
        }
    } else {
        panic!("expected ScopedFlags, got {:?}", rule.pattern);
    }
}

#[test]
fn test_parse_scoped_flags_case_sensitive_minus_i() {
    // (?-i:...) - case-sensitive pattern (regex-style)
    let file = parse_str("(?-i:XYZ) -> xyz").expect("test: parse (?-i:XYZ) -> xyz");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::ScopedFlags { flags, inner: _ } = &rule.pattern {
        // Should have case_insensitive = Some(false) (case-sensitive)
        assert_eq!(flags.case_insensitive, Some(false));
    } else {
        panic!("expected ScopedFlags, got {:?}", rule.pattern);
    }
}

#[test]
fn test_parse_scoped_flags_in_context() {
    // Scoped flags in context position
    let file = parse_str("a -> b / (?c:ABC)_").expect("test: parse a -> b / (?c:ABC)_");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    let ctx = rule.context.as_ref().expect("should have context");

    // Check left context has ScopedFlags
    if let Some(ref left) = ctx.left {
        if let ContextExpr::Pattern(ref expr) = **left {
            assert!(
                matches!(expr, Expression::ScopedFlags { .. }),
                "expected ScopedFlags in left context"
            );
        } else {
            panic!("expected Pattern in left context");
        }
    } else {
        panic!("expected left context");
    }
}

#[test]
fn test_parse_scoped_flags_with_alternation() {
    // Scoped flags containing alternation
    let file = parse_str("(?c:A|B) -> x").expect("test: parse (?c:A|B) -> x");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    if let Expression::ScopedFlags { flags, inner } = &rule.pattern {
        assert_eq!(flags.case_insensitive, Some(false));
        // Inner should be alternation
        assert!(
            matches!(**inner, Expression::Alt(_, _)),
            "expected Alt inside ScopedFlags, got {:?}",
            inner
        );
    } else {
        panic!("expected ScopedFlags");
    }
}

#[test]
fn test_parse_regular_group_not_scoped_flags() {
    // Regular groups should NOT become ScopedFlags
    let file = parse_str("(abc) -> xyz").expect("test: parse (abc) -> xyz");
    assert_eq!(file.rules.len(), 1);

    let rule = &file.rules[0].rule;
    // Should be Concat of a, b, c (groups are transparent in this parser)
    // or a nested structure, but NOT ScopedFlags
    assert!(
        !matches!(rule.pattern, Expression::ScopedFlags { .. }),
        "regular group should not become ScopedFlags"
    );
}
