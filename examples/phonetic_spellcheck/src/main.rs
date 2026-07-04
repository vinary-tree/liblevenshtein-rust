//! Phonetic Spellcheck Demo
//!
//! Demonstrates the `PhoneticNormalizedDictionary` API with combined English rules
//! (base + homophones + text_speak) and showcases fuzzy matching, regex queries,
//! and phonetic pattern expansion.
//!
//! # Features
//!
//! - **PhoneticNormalizedDictionary**: Dual-index architecture with BK-tree optimization
//! - **Combined English Rules**: Base phonetic rules + homophones + text speak
//! - **Fuzzy Matching**: Edit distance queries with automatic BK-tree acceleration
//! - **Regex Queries**: Pattern matching against normalized dictionary forms
//! - **Phonetic Pattern Expansion**: Automatic generation of phonetic alternations
//!
//! # Usage
//!
//! ```bash
//! cd examples/phonetic_spellcheck
//! cargo run --release
//! ```

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
use liblevenshtein::dictionary::Dictionary;
use liblevenshtein::phonetic::llev::RuleSetChar;
use liblevenshtein::phonetic::rules::english;

/// Load dictionary words from a file (one word per line).
fn load_dictionary<P: AsRef<Path>>(path: P) -> Vec<String> {
    let file = fs::File::open(path).expect("Failed to open dictionary file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// Combine all three English rule sets.
fn combined_english_rules() -> RuleSetChar {
    let mut combined = english::base().clone();
    combined.merge(english::homophones().clone());
    combined.merge(english::text_speak().clone());
    combined
}

fn main() {
    println!("=== Phonetic Spellcheck Demo ===\n");

    // Load dictionary
    let dict_path = "data/english_words.txt";
    println!("Loading dictionary from {}...", dict_path);
    let start = Instant::now();
    let words = load_dictionary(dict_path);
    let word_count = words.len();
    println!("  Loaded {} words in {:?}\n", word_count, start.elapsed());

    // Combine all English rules
    println!("Combining English phonetic rules...");
    let combined_rules = combined_english_rules();
    let rule_count = combined_rules.len();
    println!(
        "  Combined {} rules (base + homophones + text_speak)\n",
        rule_count
    );

    // Build PhoneticNormalizedDictionary with all optimizations
    println!("Building PhoneticNormalizedDictionary...");
    let start = Instant::now();
    let dict =
        PhoneticNormalizedDictionary::<()>::from_terms_with_rules(&words, combined_rules.rules);
    let build_time = start.elapsed();
    println!("  Built dictionary in {:?}", build_time);
    println!("  Original terms: {}", dict.len().unwrap_or(0));
    println!("  Normalized forms: {}\n", dict.normalized_count());

    // Demo: Fuzzy queries
    println!("--- Fuzzy Query Demo ---\n");
    demo_fuzzy_query(&dict, "fone", 2);
    demo_fuzzy_query(&dict, "filosofy", 2);
    demo_fuzzy_query(&dict, "enuf", 1);
    demo_fuzzy_query(&dict, "teh", 1);
    demo_fuzzy_query(&dict, "nite", 1);

    // Demo: Normalization
    println!("--- Normalization Demo ---\n");
    demo_normalization(&dict, "phone");
    demo_normalization(&dict, "elephant");
    demo_normalization(&dict, "knight");
    demo_normalization(&dict, "through");

    // Demo: Regex queries
    println!("--- Regex Query Demo ---\n");
    demo_regex_query(&dict, "(ph|f)one", 0);
    demo_regex_query(&dict, "kn.*t", 1);

    // Demo: Phonetic pattern expansion
    println!("--- Phonetic Pattern Expansion Demo ---\n");
    demo_pattern_expansion(&dict, "fone");
    demo_pattern_expansion(&dict, "nite");

    println!("=== Demo Complete ===");
}

/// Demonstrate fuzzy querying with edit distance.
fn demo_fuzzy_query(dict: &PhoneticNormalizedDictionary<()>, query: &str, max_distance: usize) {
    let start = Instant::now();
    let results = dict.query(query, max_distance);
    let elapsed = start.elapsed();

    println!(
        "Query: \"{}\" (distance: {}, {} results in {:?})",
        query,
        max_distance,
        results.len(),
        elapsed
    );
    println!("  Normalized query: \"{}\"", dict.normalize(query));

    for (i, candidate) in results.iter().take(5).enumerate() {
        println!(
            "  {}. {} (distance: {}, normalized: \"{}\")",
            i + 1,
            candidate.term,
            candidate.distance,
            candidate.normalized_form
        );
    }
    if results.len() > 5 {
        println!("  ... and {} more", results.len() - 5);
    }
    println!();
}

/// Demonstrate string normalization.
fn demo_normalization(dict: &PhoneticNormalizedDictionary<()>, term: &str) {
    let normalized = dict.normalize(term);
    println!("\"{}\" → \"{}\"", term, normalized);
}

/// Demonstrate regex queries.
fn demo_regex_query(dict: &PhoneticNormalizedDictionary<()>, pattern: &str, max_distance: u8) {
    let start = Instant::now();
    let results = dict.query_regex(pattern, max_distance);
    let elapsed = start.elapsed();

    match results {
        Ok(matches) => {
            println!(
                "Regex: \"{}\" (distance: {}, {} matches in {:?})",
                pattern,
                max_distance,
                matches.len(),
                elapsed
            );
            for (i, candidate) in matches.iter().take(5).enumerate() {
                println!(
                    "  {}. {} (distance: {}, normalized: \"{}\")",
                    i + 1,
                    candidate.term,
                    candidate.distance,
                    candidate.normalized_form
                );
            }
            if matches.len() > 5 {
                println!("  ... and {} more", matches.len() - 5);
            }
        }
        Err(e) => {
            println!("Regex error for \"{}\": {}", pattern, e);
        }
    }
    println!();
}

/// Demonstrate phonetic pattern expansion.
///
/// The expansion works by:
/// 1. Normalizing the query (e.g., "fone" → "fon")
/// 2. Expanding to a pattern matching original spellings (e.g., "(ph|f)on")
/// 3. Searching original dictionary terms with the pattern
fn demo_pattern_expansion(dict: &PhoneticNormalizedDictionary<()>, query: &str) {
    let normalized = dict.normalize(query);
    let pattern = dict.expand_to_phonetic_pattern(query);

    println!("Input: \"{}\"", query);
    println!("  Normalized: \"{}\"", normalized);
    println!("  Expanded pattern: \"{}\"", pattern);

    // Query ORIGINAL terms with the expanded pattern (not normalized forms)
    // Use distance 0 for exact pattern matches
    match dict.query_original_regex(&pattern, 0) {
        Ok(matches) => {
            let terms: Vec<_> = matches.iter().take(10).map(|c| c.term.as_str()).collect();
            println!("  Matches: {:?}", terms);
        }
        Err(e) => {
            println!("  Error: {}", e);
        }
    }
    println!();
}
