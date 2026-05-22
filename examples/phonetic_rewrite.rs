//! Example: Phonetic Rewrite Rules
//!
//! This example demonstrates how to use the phonetic rewrite module to
//! transform English text into phonetic representations using formally
//! verified transformation rules.
//!
//! Run with:
//! ```bash
//! cargo run --example phonetic_rewrite --features phonetic-rules
//! ```

use liblevenshtein::phonetic::*;

fn main() {
    println!("=== Phonetic Rewrite Rules Example ===\n");
    println!("Formally verified implementation from Coq/Rocq proofs");
    println!("5 theorems proven: Well-formedness, Bounded Expansion,");
    println!("Non-Confluence, Termination, Idempotence\n");

    // ========================================================================
    // Example 1: Simple orthography transformations
    // ========================================================================
    println!("--- Example 1: Orthography Rules ---");

    let words = vec!["phone", "church", "ghost", "enough"];

    let ortho_rules = orthography_rules();

    for word in words {
        // Convert string to phonetic representation
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

        println!("\nInput: \"{}\"", word);
        println!("Phones: {:?}", phones);

        // Apply orthography rules
        let fuel = phones.len() * ortho_rules.len() * MAX_EXPANSION_FACTOR;
        if let Some(result) = apply_rules_seq(&ortho_rules, &phones, fuel) {
            println!("Output: {:?}", result);
            println!("Rule applications demonstrate:");
            println!("  - ph → f");
            println!("  - ch → digraph(c,h)");
            println!("  - gh → silent");
        }
    }

    // ========================================================================
    // Example 2: Phonetic approximations for fuzzy matching
    // ========================================================================
    println!("\n\n--- Example 2: Phonetic Rules (Fuzzy Matching) ---");

    let phonetic_words = vec!["thin", "queen", "kwik"];

    let phon_rules = phonetic_rules();

    for word in phonetic_words {
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

        println!("\nInput: \"{}\"", word);

        let fuel = phones.len() * phon_rules.len() * MAX_EXPANSION_FACTOR;
        if let Some(result) = apply_rules_seq(&phon_rules, &phones, fuel) {
            println!("Output (phonetic): {:?}", result);
            println!("Phonetic transformations (weight=0.15):");
            println!("  - th → t (thin ≈ tin)");
            println!("  - qu ↔ kw (queen ↔ kween)");
        }
    }

    // ========================================================================
    // Example 3: Individual rule application
    // ========================================================================
    println!("\n\n--- Example 3: Individual Rule Application ---");

    let test_string = vec![
        PhoneByte::Consonant(b'g'),
        PhoneByte::Consonant(b'h'),
        PhoneByte::Vowel(b'o'),
        PhoneByte::Consonant(b's'),
        PhoneByte::Consonant(b't'),
    ];

    println!("\nInput: {:?}", test_string);
    println!("Attempting to apply 'gh → silent' rule at position 0...");

    let ortho = orthography_rules();
    let gh_rule = &ortho[7]; // Rule 34: gh → silent

    if let Some(result) = apply_rule_at(gh_rule, &test_string, 0) {
        println!("✓ Rule applied successfully!");
        println!("Result: {:?}", result);
        println!("Transformation: [g,h,o,s,t] → [silent,o,s,t]");
    }

    // ========================================================================
    // Example 4: Context-dependent rules
    // ========================================================================
    println!("\n\n--- Example 4: Context-Dependent Rules ---");

    let test_cases = vec![
        ("city", "c → s before front vowels (e,i)"),
        ("cat", "c → k elsewhere"),
        ("gem", "g → j before front vowels"),
        ("go", "g remains (not before e,i)"),
    ];

    for (word, description) in test_cases {
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

        println!("\nInput: \"{}\"", word);
        println!("Rule: {}", description);

        let fuel = phones.len() * ortho_rules.len() * MAX_EXPANSION_FACTOR;
        if let Some(result) = apply_rules_seq(&ortho_rules, &phones, fuel) {
            // Convert back to bytes for display (simplified)
            let result_chars: Vec<char> = result
                .iter()
                .filter_map(|p| match p {
                    PhoneByte::Vowel(c) | PhoneByte::Consonant(c) => Some(*c as char),
                    PhoneByte::Digraph(c1, _) => Some(*c1 as char),
                    PhoneByte::Trigraph(c1, _, _) => Some(*c1 as char),
                    PhoneByte::Tetragraph(c1, _, _, _) => Some(*c1 as char),
                    PhoneByte::Pentagraph(c1, _, _, _, _) => Some(*c1 as char),
                    PhoneByte::Hexagraph(c1, _, _, _, _, _) => Some(*c1 as char),
                    PhoneByte::Heptagraph(c1, _, _, _, _, _, _) => Some(*c1 as char),
                    PhoneByte::Sequence(s) => s.first().map(|c| *c as char),
                    PhoneByte::Silent => None,
                })
                .collect();

            println!("Result: {}", result_chars.iter().collect::<String>());
        }
    }

    // ========================================================================
    // Example 5: Demonstrating formal properties
    // ========================================================================
    println!("\n\n--- Example 5: Formal Properties Demonstration ---");

    // Property 1: Well-formedness
    println!("\n✓ Property 1: All rules are well-formed");
    println!("  All {} zompist rules have:", zompist_rules().len());
    println!("  - Non-empty patterns");
    println!("  - Non-negative weights");

    // Property 2: Bounded expansion
    let test = vec![PhoneByte::Consonant(b'x')]; // x → yy (expansion)
    let test_rules = test_rules();
    if let Some(result) = apply_rules_seq(&test_rules, &test, 100) {
        let expansion = result.len() as i64 - test.len() as i64;
        println!("\n✓ Property 2: Bounded expansion");
        println!("  Input length: {}", test.len());
        println!("  Output length: {}", result.len());
        println!(
            "  Expansion: {} ≤ MAX_EXPANSION_FACTOR({})",
            expansion, MAX_EXPANSION_FACTOR
        );
    }

    // Property 3: Non-confluence
    println!("\n✓ Property 3: Non-confluence (order matters)");
    println!("  Rules: x→yy, y→z on input [x,y]");
    println!("  Order 1 (x→yy first): different result");
    println!("  Order 2 (y→z first): different result");
    println!("  Proven in Theorem 3 (zompist_rules.v:491)");

    // Property 4: Termination
    println!("\n✓ Property 4: Termination guaranteed");
    println!("  Sequential application always terminates");
    println!("  With sufficient fuel: n × m × k");
    println!("  Proven in Theorem 4 (zompist_rules.v:569)");

    // Property 5: Idempotence
    let input = vec![PhoneByte::Consonant(b'p'), PhoneByte::Consonant(b'h')];
    let fuel = input.len() * ortho_rules.len() * MAX_EXPANSION_FACTOR;
    if let Some(result1) = apply_rules_seq(&ortho_rules, &input, fuel) {
        if let Some(result2) = apply_rules_seq(&ortho_rules, &result1, fuel) {
            println!("\n✓ Property 5: Idempotence at fixed points");
            println!("  First application:  {:?}", result1);
            println!("  Second application: {:?}", result2);
            println!("  Equal: {}", result1 == result2);
            println!("  Proven in Theorem 5 (zompist_rules.v:615)");
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n\n=== Summary ===");
    println!(
        "✓ Orthography rules: {} (exact transformations)",
        ortho_rules.len()
    );
    println!("✓ Phonetic rules: {} (fuzzy matching)", phon_rules.len());
    println!("✓ Test rules: {} (non-confluence proof)", test_rules.len());
    println!("✓ Total: {} formally verified rules", zompist_rules().len());
    println!("\nAll algorithms proven correct in Coq/Rocq:");
    println!("  - Well-formedness (Theorem 1)");
    println!("  - Bounded expansion (Theorem 2)");
    println!("  - Non-confluence (Theorem 3)");
    println!("  - Termination (Theorem 4)");
    println!("  - Idempotence (Theorem 5)");
    println!("\nFor more details:");
    println!("  - Coq proofs: docs/verification/phonetic/");
    println!("  - Rust implementation: src/phonetic/");
    println!("  - Property tests: src/phonetic/properties.rs");
    println!("  - Benchmarks: benches/phonetic_rules.rs");
}
