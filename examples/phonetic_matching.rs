//! Phonetic Matching Example
//!
//! Demonstrates how to use restricted substitutions for phonetic fuzzy matching.
//! This allows matching words that sound similar but are spelled differently.
//!
//! Run this example with:
//! ```bash
//! cargo run --example phonetic_matching
//! ```

use liblevenshtein::prelude::*;
use liblevenshtein::transducer::{Algorithm, SubstitutionSet};

fn main() {
    println!("=== Phonetic Matching with Restricted Substitutions ===\n");

    // Create a dictionary of correctly spelled words
    let terms = vec!["phone", "center", "cat", "dogs", "philosophy", "circle"];

    let dict = DoubleArrayTrie::from_terms(terms.clone());
    println!("Dictionary terms: {:?}\n", terms);

    // Create phonetic substitution set
    let phonetic = SubstitutionSet::phonetic_basic();
    println!("Phonetic equivalences enabled:");
    println!("  - f ↔ p (for 'ph' sounds)");
    println!("  - c ↔ k");
    println!("  - c ↔ s");
    println!("  - s ↔ z");
    println!();

    // Create transducer with phonetic substitutions
    let transducer = Transducer::with_substitutions(dict, Algorithm::Standard, phonetic);

    // Test phonetic queries
    let test_queries = vec![
        ("fone", "phone"),          // f↔p (ph sound)
        ("filosofy", "philosophy"), // f↔p, f↔p
        ("kat", "cat"),             // k↔c
        ("senter", "center"),       // s↔c
        ("dogz", "dogs"),           // z↔s
        ("sirkle", "circle"),       // s↔c, k↔c
    ];

    println!("Testing phonetic queries (max distance = 2):\n");

    for (query, expected) in test_queries {
        print!("Query: {:8} → ", query);

        let results: Vec<String> = transducer.query(query, 2).take(5).collect();

        if results.contains(&expected.to_string()) {
            println!("✓ Found '{}' (matches: {:?})", expected, results);
        } else {
            println!("✗ Expected '{}', got: {:?}", expected, results);
        }
    }

    println!("\n=== Comparison: Standard vs Phonetic ===\n");

    // Create transducer WITHOUT phonetic substitutions for comparison
    let dict_standard = DoubleArrayTrie::from_terms(vec!["phone", "cat"]);
    let transducer_standard = Transducer::new(dict_standard, Algorithm::Standard);

    let query = "fone";
    println!("Query: '{}'\n", query);

    // Standard Levenshtein (no substitutions)
    let standard_results: Vec<(String, u8)> = transducer_standard
        .query(query, 2)
        .map(|term| {
            // Calculate actual distance
            let dist = levenshtein_distance(query, &term);
            (term, dist)
        })
        .collect();

    println!("Standard Levenshtein (no substitutions):");
    for (term, dist) in &standard_results {
        println!("  - '{}' at distance {}", term, dist);
    }

    // Phonetic matching
    let dict_phonetic = DoubleArrayTrie::from_terms(vec!["phone", "cat"]);
    let phonetic_set = SubstitutionSet::phonetic_basic();
    let transducer_phonetic =
        Transducer::with_substitutions(dict_phonetic, Algorithm::Standard, phonetic_set);

    let phonetic_results: Vec<(String, u8)> = transducer_phonetic
        .query(query, 2)
        .map(|term| {
            // Calculate distance with phonetic equivalences
            let dist = phonetic_distance(query, &term);
            (term, dist)
        })
        .collect();

    println!("\nPhonetic matching (f↔p substitution):");
    for (term, dist) in &phonetic_results {
        println!("  - '{}' at effective distance {}", term, dist);
    }

    println!("\n📊 Result: 'fone' matches 'phone' more closely with phonetic substitutions!");
}

/// Calculate standard Levenshtein distance
fn levenshtein_distance(a: &str, b: &str) -> u8 {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut prev_row: Vec<u8> = (0..=b_bytes.len() as u8).collect();
    let mut curr_row = vec![0u8; b_bytes.len() + 1];

    for (i, &a_byte) in a_bytes.iter().enumerate() {
        curr_row[0] = (i + 1) as u8;

        for (j, &b_byte) in b_bytes.iter().enumerate() {
            let cost = if a_byte == b_byte { 0 } else { 1 };
            curr_row[j + 1] = std::cmp::min(
                std::cmp::min(prev_row[j + 1] + 1, curr_row[j] + 1),
                prev_row[j] + cost,
            );
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b_bytes.len()]
}

/// Calculate phonetic distance (simplified: treats f↔p as equivalent)
fn phonetic_distance(a: &str, b: &str) -> u8 {
    let a_normalized = a.replace('f', "p");
    let b_normalized = b.replace('f', "p");
    levenshtein_distance(&a_normalized, &b_normalized)
}
