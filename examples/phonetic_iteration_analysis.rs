//! Analysis program to measure iteration counts in apply_rules_seq()
//!
//! This program instruments the phonetic rewrite system to count the number
//! of iterations required for different input sizes, testing hypothesis H5:
//! that iteration count increases superlinearly with input size.

use liblevenshtein::phonetic::*;

/// Instrumented version of apply_rules_seq that counts iterations
fn apply_rules_seq_instrumented(
    rules: &[RewriteRuleByte],
    s: &[PhoneByte],
    fuel: usize,
) -> (Option<Vec<PhoneByte>>, usize) {
    let mut current = s.to_vec();
    let mut remaining_fuel = fuel;
    let mut iteration_count = 0;

    loop {
        iteration_count += 1;

        if remaining_fuel == 0 {
            return (Some(current), iteration_count);
        }

        let mut applied = false;

        for rule in rules {
            // Manually inline find_first_match and apply_rule_at to avoid code duplication
            let mut match_pos = None;
            for pos in 0..=current.len() {
                if context_matches(&rule.context, &current, pos, rule.pattern.len())
                    && pattern_matches_at(&rule.pattern, &current, pos)
                {
                    match_pos = Some(pos);
                    break;
                }
            }

            if let Some(pos) = match_pos {
                // Apply the rule
                if context_matches(&rule.context, &current, pos, rule.pattern.len())
                    && pattern_matches_at(&rule.pattern, &current, pos)
                {
                    let mut result = Vec::with_capacity(current.len() + MAX_EXPANSION_FACTOR);
                    result.extend_from_slice(&current[..pos]);
                    result.extend_from_slice(&rule.replacement);
                    result.extend_from_slice(&current[(pos + rule.pattern.len())..]);

                    current = result;
                    remaining_fuel -= 1;
                    applied = true;
                    break;
                }
            }
        }

        if !applied {
            return (Some(current), iteration_count);
        }
    }
}

fn main() {
    println!("=== PhoneBytetic Rules Iteration Count Analysis ===\n");
    println!("Testing H5: Iteration count increases superlinearly with input size\n");

    let ortho_rules = orthography_rules();

    // Test different input sizes
    let test_cases = [
        (5, "phone"),   // 5 phones
        (10, "phonetics"), // ~10 phones
        (20, "phoneticsphonetics"), // ~20 phones
        (50, "phoneticsphoneticsphoneticsphoneticsphoneticsphone"), // ~50 phones
    ];

    println!("| Input Size | Actual Length | Iterations | Ratio vs 5-phone |");
    println!("|-----------:|-------------:|-----------:|-----------------:|");

    let mut baseline_iterations = 0;

    for (expected_size, word) in test_cases.iter() {
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

        let actual_size = phones.len();
        let fuel = actual_size * ortho_rules.len() * MAX_EXPANSION_FACTOR;

        let (_result, iterations) = apply_rules_seq_instrumented(&ortho_rules, &phones, fuel);

        if *expected_size == 5 {
            baseline_iterations = iterations;
        }

        let ratio = if baseline_iterations > 0 {
            iterations as f64 / baseline_iterations as f64
        } else {
            1.0
        };

        println!(
            "| {} | {} | {} | {:.2}× |",
            expected_size, actual_size, iterations, ratio
        );
    }

    println!("\n=== Analysis ===\n");
    println!("If H5 is correct:");
    println!("- Iteration ratio should increase faster than linear");
    println!("- 50-phone case should show >> 10× iterations vs 5-phone");
    println!("\nIf H5 is incorrect:");
    println!("- Iteration ratio should be approximately linear with size");
    println!("- 50-phone case should show ~10× iterations vs 5-phone");
}
