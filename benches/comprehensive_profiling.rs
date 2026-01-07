//! Comprehensive profiling benchmark for flame graph generation
//!
//! This benchmark compares:
//! - Exact matching (standard dictionaries)
//! - Prefix matching (standard dictionaries with .prefix())
//! - Substring matching (suffix automata)
//!
//! Run with: cargo flamegraph --bench comprehensive_profiling

use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::suffix_automaton::SuffixAutomaton;
use liblevenshtein::transducer::{Algorithm, Transducer};

fn load_dictionary() -> Vec<String> {
    std::fs::read_to_string("/usr/share/dict/words")
        .unwrap_or_else(|_| {
            (0..10000)
                .map(|i| format!("word{}", i))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .lines()
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= 3 && s.len() <= 15)
        .take(10000)
        .collect()
}

fn main() {
    eprintln!("Loading dictionary...");
    let words = load_dictionary();
    eprintln!("Loaded {} words", words.len());

    // Build dictionaries
    eprintln!("Building PathMap dictionary...");
    let pathmap_dict: PathMapDictionary<()> = PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));

    eprintln!("Building SuffixAutomaton dictionary...");
    let suffix_dict: SuffixAutomaton<()> = SuffixAutomaton::from_texts(words.iter().map(|s| s.as_str()));

    let queries = vec![
        "test", "get", "set", "value", "user", "data", "calc", "find", "prog", "proc", "func",
        "class", "method", "var", "const", "let",
    ];

    eprintln!("Warming up...");
    // Warm-up
    for _ in 0..10 {
        for query in &queries {
            // Exact matching
            let _: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query_ordered(query, 1)
                .take(10)
                .collect();

            // Prefix matching
            let _: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query_ordered(query, 1)
                .prefix()
                .take(10)
                .collect();

            // Substring matching
            let _: Vec<_> = Transducer::new(suffix_dict.clone(), Algorithm::Standard)
                .query_ordered(query, 1)
                .take(10)
                .collect();
        }
    }

    eprintln!("Starting profiling run...");

    // Main profiling loop
    for iteration in 0..500 {
        if iteration % 100 == 0 {
            eprintln!("Iteration {}/500", iteration);
        }

        for query in &queries {
            // Scenario 1: Exact matching (distance 0, 1, 2)
            for distance in [0, 1, 2] {
                let _: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                    .query_ordered(query, distance)
                    .take(10)
                    .collect();
            }

            // Scenario 2: Prefix matching (distance 0, 1, 2)
            for distance in [0, 1, 2] {
                let _: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                    .query_ordered(query, distance)
                    .prefix()
                    .take(10)
                    .collect();
            }

            // Scenario 3: Substring matching (distance 0, 1, 2)
            for distance in [0, 1, 2] {
                let _: Vec<_> = Transducer::new(suffix_dict.clone(), Algorithm::Standard)
                    .query_ordered(query, distance)
                    .take(10)
                    .collect();
            }
        }
    }

    eprintln!(
        "Profiling complete: processed {} total queries",
        500 * queries.len() * 9 // 500 iterations × 16 queries × 9 scenarios
    );
}
