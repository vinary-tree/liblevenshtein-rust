//! Tests for Zompist English phonetic rules (byte- and char-level aggregators).

use super::*;

#[test]
fn test_orthography_rules_count() {
    assert_eq!(orthography_rules().len(), 45);
}

#[test]
fn test_vowel_digraph_rules_count() {
    assert_eq!(vowel_digraph_rules().len(), 12);
}

#[test]
fn test_phonetic_rules_count() {
    assert_eq!(phonetic_rules().len(), 3);
}

#[test]
fn test_test_rules_count() {
    assert_eq!(test_rules().len(), 2);
}

#[test]
fn test_zompist_rules_count() {
    assert_eq!(zompist_rules().len(), 62);
}

#[test]
fn test_rule_weights() {
    // Orthography rules should have weight 0.0
    for rule in orthography_rules().iter() {
        assert_eq!(
            rule.weight, 0.0,
            "Rule {} should have weight 0.0",
            rule.rule_name
        );
    }

    // Vowel digraph rules should have weight 0.1
    for rule in vowel_digraph_rules().iter() {
        assert!(
            (rule.weight - 0.1).abs() < f64::EPSILON,
            "Rule {} should have weight 0.1",
            rule.rule_name
        );
    }

    // Phonetic rules should have weight 0.15
    for rule in phonetic_rules().iter() {
        assert_eq!(
            rule.weight, 0.15,
            "Rule {} should have weight 0.15",
            rule.rule_name
        );
    }
}

#[test]
fn test_char_rules_count() {
    assert_eq!(orthography_rules_char().len(), 45);
    assert_eq!(vowel_digraph_rules_char().len(), 12);
    assert_eq!(phonetic_rules_char().len(), 3);
    assert_eq!(test_rules_char().len(), 2);
    assert_eq!(zompist_rules_char().len(), 62);
}
