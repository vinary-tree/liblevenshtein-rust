//! Real-world dictionary benchmarks using /usr/share/dict/words
//!
//! This benchmark:
//! 1. Compares dictionary backends on real-world data
//! 2. Tests performance characteristics

use liblevenshtein::prelude::*;
use std::fs;

fn main() {
    println!("=== Real-World Dictionary Analysis ===\n");

    // Load real dictionary
    println!("Loading real dictionary...");
    let real_words = load_real_dictionary();
    println!(
        "Loaded {} words from English dictionary\n",
        real_words.len()
    );

    // Load synthetic dictionary for comparison
    println!("Generating synthetic dictionary...");
    let synthetic_words = generate_synthetic_dictionary(10_000);
    println!("Generated {} synthetic words\n", synthetic_words.len());

    // Build dictionaries
    println!("Building DoubleArrayTrie from real dictionary...");
    let start = std::time::Instant::now();
    let real_dat = DoubleArrayTrie::from_terms(real_words.clone());
    let real_build_time = start.elapsed();
    println!("Real DAT built in {:?}", real_build_time);
    println!("Real DAT terms: {}\n", real_dat.len().unwrap_or(0));

    println!("Building DoubleArrayTrie from synthetic dictionary...");
    let start = std::time::Instant::now();
    let synthetic_dat = DoubleArrayTrie::from_terms(synthetic_words.clone());
    let synthetic_build_time = start.elapsed();
    println!("Synthetic DAT built in {:?}", synthetic_build_time);
    println!("Synthetic DAT terms: {}\n", synthetic_dat.len().unwrap_or(0));

    // Performance benchmarks
    println!("\n=== Performance Comparison ===\n");

    // Contains benchmark
    println!("Testing contains() performance...");

    // Real dictionary
    let real_test_words: Vec<_> = real_words.iter().take(10_000).collect();
    let start = std::time::Instant::now();
    let mut real_found = 0;
    for _ in 0..100 {
        for word in &real_test_words {
            if real_dat.contains(word) {
                real_found += 1;
            }
        }
    }
    let real_contains_time = start.elapsed();
    println!(
        "Real dictionary: {} calls in {:?} ({:.2} µs/call)",
        real_test_words.len() * 100,
        real_contains_time,
        real_contains_time.as_micros() as f64 / (real_test_words.len() * 100) as f64
    );
    println!("Found: {}", real_found);

    // Synthetic dictionary
    let synthetic_test_words: Vec<_> = synthetic_words.iter().take(10_000).collect();
    let start = std::time::Instant::now();
    let mut synthetic_found = 0;
    for _ in 0..100 {
        for word in &synthetic_test_words {
            if synthetic_dat.contains(word) {
                synthetic_found += 1;
            }
        }
    }
    let synthetic_contains_time = start.elapsed();
    println!(
        "Synthetic dictionary: {} calls in {:?} ({:.2} µs/call)",
        synthetic_test_words.len() * 100,
        synthetic_contains_time,
        synthetic_contains_time.as_micros() as f64 / (synthetic_test_words.len() * 100) as f64
    );
    println!("Found: {}", synthetic_found);

    // Query benchmark
    println!("\nTesting fuzzy query performance...");

    // Real dictionary queries
    let real_query_words: Vec<_> = real_words
        .iter()
        .step_by(real_words.len() / 1000)
        .take(1000)
        .collect();

    let real_transducer = Transducer::new(real_dat, Algorithm::Standard);
    let start = std::time::Instant::now();
    let mut real_results = 0;
    for word in &real_query_words {
        for _ in real_transducer.query(word, 2) {
            real_results += 1;
        }
    }
    let real_query_time = start.elapsed();
    println!(
        "Real dictionary: {} queries in {:?} ({:.2} µs/query)",
        real_query_words.len(),
        real_query_time,
        real_query_time.as_micros() as f64 / real_query_words.len() as f64
    );
    println!("Total results: {}", real_results);

    // Synthetic dictionary queries
    let synthetic_query_words: Vec<_> = synthetic_words
        .iter()
        .step_by(synthetic_words.len() / 1000)
        .take(1000)
        .collect();

    let synthetic_transducer = Transducer::new(synthetic_dat, Algorithm::Standard);
    let start = std::time::Instant::now();
    let mut synthetic_results = 0;
    for word in &synthetic_query_words {
        for _ in synthetic_transducer.query(word, 2) {
            synthetic_results += 1;
        }
    }
    let synthetic_query_time = start.elapsed();
    println!(
        "Synthetic dictionary: {} queries in {:?} ({:.2} µs/query)",
        synthetic_query_words.len(),
        synthetic_query_time,
        synthetic_query_time.as_micros() as f64 / synthetic_query_words.len() as f64
    );
    println!("Total results: {}", synthetic_results);

    println!("\n=== Analysis Complete ===");
}

fn load_real_dictionary() -> Vec<String> {
    let contents = fs::read_to_string("data/english_words.txt")
        .expect("Failed to read data/english_words.txt");

    contents
        .lines()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic()))
        .collect()
}

fn generate_synthetic_dictionary(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("word{:06}", i)).collect()
}
