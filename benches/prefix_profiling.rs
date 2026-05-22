// Profiling-optimized benchmark for flame graph generation
// Run with: cargo flamegraph --bench prefix_profiling

use liblevenshtein::prelude::*;

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
        .collect()
}

fn main() {
    let words = load_dictionary();
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));

    // Simulate realistic workload: many queries with prefix matching and filtering
    let queries = vec![
        "test", "get", "set", "value", "user", "data", "calc", "find",
    ];

    // Warm-up
    for _ in 0..10 {
        for query in &queries {
            let _: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(query, 1)
                .prefix()
                .take(10)
                .collect();
        }
    }

    // Main profiling loop
    for _ in 0..1000 {
        for query in &queries {
            // Scenario 1: Prefix matching
            let _: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(query, 1)
                .prefix()
                .take(10)
                .collect();

            // Scenario 2: Prefix matching with filter
            let _: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(query, 1)
                .prefix()
                .filter(|c| c.term.len() > 5)
                .take(10)
                .collect();

            // Scenario 3: Higher edit distance
            let _: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(query, 2)
                .prefix()
                .take(5)
                .collect();
        }
    }

    println!(
        "Profiling complete: processed {} iterations",
        1000 * queries.len() * 3
    );
}
