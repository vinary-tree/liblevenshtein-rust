//! Test for large edit distance queries

use liblevenshtein::prelude::*;

#[test]
fn test_query_with_large_distance() {
    // Regression case: querying with very large max_distance returned only 2 results
    let terms = vec!["foo", "bar", "baz", "quo", "qux"];
    let dict = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("quuo", 99).collect();

    eprintln!("Query 'quuo' with distance 99:");
    eprintln!("Found {} results (expected 5):", results.len());
    for candidate in &results {
        eprintln!("  {} (distance: {})", candidate.term, candidate.distance);
    }

    // All 5 terms should be found with a large enough distance
    assert_eq!(
        results.len(),
        5,
        "Should find all 5 terms with distance 99, but only found: {:?}",
        results.iter().map(|c| &c.term).collect::<Vec<_>>()
    );
}
