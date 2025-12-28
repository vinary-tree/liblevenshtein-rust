/// Profiling benchmark for realistic workload
/// This benchmark is designed to exercise hot paths for profiling and PGO
use liblevenshtein::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_dictionary() -> Vec<String> {
    // Try to load a real dictionary file if available
    if let Ok(file) = File::open("/usr/share/dict/words") {
        BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .take(10000) // Use 10k words for realistic profiling
            .collect()
    } else {
        // Generate synthetic dictionary
        let mut words = Vec::with_capacity(10000);
        for i in 0..10000 {
            words.push(format!("word{:06}", i));
        }
        words
    }
}

fn main() {
    println!("=== Profiling Benchmark: Realistic Workload ===\n");

    // Load dictionary
    println!("Loading dictionary...");
    let words = load_dictionary();
    println!("Loaded {} words\n", words.len());

    // Build DoubleArrayTrie (exercises construction)
    println!("Building DoubleArrayTrie...");
    let start = std::time::Instant::now();
    let dict = DoubleArrayTrie::from_terms(words.iter().map(|s| s.as_str()));
    println!("DoubleArrayTrie built in {:?}\n", start.elapsed());

    // Build transducer (exercises hot paths)
    println!("Building transducer...");
    let start = std::time::Instant::now();
    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
    println!("Transducer built in {:?}\n", start.elapsed());

    // Perform queries (exercises edge lookup, traversal, state management)
    println!("Performing queries...");
    // Use queries that will actually match synthetic dictionary (word000000, word000001, etc.)
    let test_queries = vec![
        ("wrd000050", 1),  // typo: word000050 (missing 'o')
        ("word00050", 1),  // typo: word000050 (missing '0')
        ("word000", 2),    // partial match: many words
        ("word001000", 2), // typo: word001000
        ("word005000", 2), // exact or near match
    ];

    let start = std::time::Instant::now();
    let mut total_results = 0;

    // Run each query multiple times to generate enough profile data
    for _ in 0..1000 {
        for (query, max_distance) in &test_queries {
            let results: Vec<_> = transducer.query(query, *max_distance).collect();
            total_results += results.len();
        }
    }

    let elapsed = start.elapsed();
    println!("Completed 5000 queries in {:?}", elapsed);
    println!("Average: {:?} per query", elapsed / 5000);
    println!("Total results: {}\n", total_results);

    // Dictionary operations (exercises contains, node traversal)
    println!("Performing dictionary operations...");
    let start = std::time::Instant::now();
    let mut found_count = 0;

    for _ in 0..10000 {
        for word in words.iter().take(100) {
            if dict.contains(word) {
                found_count += 1;
            }
        }
    }

    println!("Completed 1M contains() calls in {:?}", start.elapsed());
    println!("Found: {} words\n", found_count);

    println!("=== Profiling Complete ===");
}
