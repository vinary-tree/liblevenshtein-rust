//! Test to verify if position skipping preserves correctness.
//!
//! This program tests whether starting rule search from the last modified
//! position produces the same results as always starting from position 0.

#[cfg(feature = "phonetic-rules")]
use liblevenshtein::phonetic::*;

#[cfg(not(feature = "phonetic-rules"))]
fn main() {
    println!("This example requires the 'phonetic-rules' feature.");
    println!("Run with: cargo run --example position_skip_test --features phonetic-rules");
}

#[cfg(feature = "phonetic-rules")]
fn find_first_match_baseline(rule: &RewriteRuleByte, s: &[PhoneByte]) -> Option<usize> {
    (0..=s.len()).find(|&pos| {
        context_matches(&rule.context, s, pos, rule.pattern.len())
            && pattern_matches_at(&rule.pattern, s, pos)
    })
}

#[cfg(feature = "phonetic-rules")]
fn apply_rules_seq_baseline(
    rules: &[RewriteRuleByte],
    s: &[PhoneByte],
    fuel: usize,
) -> Option<Vec<PhoneByte>> {
    let mut current = s.to_vec();
    let mut remaining_fuel = fuel;

    loop {
        if remaining_fuel == 0 {
            return Some(current);
        }

        let mut applied = false;

        for rule in rules {
            if let Some(pos) = find_first_match_baseline(rule, &current) {
                if let Some(new_s) = apply_rule_at(rule, &current, pos) {
                    current = new_s;
                    remaining_fuel -= 1;
                    applied = true;
                    break;
                }
            }
        }

        if !applied {
            return Some(current);
        }
    }
}

#[cfg(feature = "phonetic-rules")]
fn find_first_match_from(
    rule: &RewriteRuleByte,
    s: &[PhoneByte],
    start_pos: usize,
) -> Option<usize> {
    (start_pos..=s.len()).find(|&pos| {
        context_matches(&rule.context, s, pos, rule.pattern.len())
            && pattern_matches_at(&rule.pattern, s, pos)
    })
}

#[cfg(feature = "phonetic-rules")]
fn apply_rules_seq_optimized(
    rules: &[RewriteRuleByte],
    s: &[PhoneByte],
    fuel: usize,
) -> Option<Vec<PhoneByte>> {
    let mut current = s.to_vec();
    let mut remaining_fuel = fuel;
    let mut last_pos = 0;

    loop {
        if remaining_fuel == 0 {
            return Some(current);
        }

        let mut applied = false;

        for rule in rules {
            // Try from last_pos first
            if let Some(pos) = find_first_match_from(rule, &current, last_pos) {
                if let Some(new_s) = apply_rule_at(rule, &current, pos) {
                    last_pos = pos;
                    current = new_s;
                    remaining_fuel -= 1;
                    applied = true;
                    break;
                }
            }
        }

        if !applied {
            return Some(current);
        }
    }
}

#[cfg(feature = "phonetic-rules")]
fn main() {
    println!("=== Testing Position Skipping Optimization ===\n");

    let ortho_rules = orthography_rules();

    let test_words = ["phone", "phonetics", "phonograph", "telephone", "symphony"];

    let mut all_match = true;

    for word in &test_words {
        let phones: Vec<PhoneByte> = word
            .bytes()
            .map(|b| {
                if matches!(b, b'a' | b'e' | b'i' | b'o' | b'u') {
                    PhoneByte::Vowel(b)
                } else {
                    PhoneByte::Consonant(b)
                }
            })
            .collect();

        let fuel = phones.len() * ortho_rules.len() * 100;

        let baseline_result = apply_rules_seq_baseline(&ortho_rules, &phones, fuel);
        let optimized_result = apply_rules_seq_optimized(&ortho_rules, &phones, fuel);

        let match_status = if baseline_result == optimized_result {
            "✅ MATCH"
        } else {
            all_match = false;
            "❌ MISMATCH"
        };

        println!("{}: {}", match_status, word);

        if baseline_result != optimized_result {
            println!("  Baseline:  {:?}", baseline_result);
            println!("  Optimized: {:?}", optimized_result);
        }
    }

    println!();
    if all_match {
        println!("✅ All tests passed - optimization preserves correctness!");
    } else {
        println!("❌ Optimization changes behavior - NOT SAFE");
    }
}
