use libdictenstein::double_array_trie_char::DoubleArrayTrieChar;
use liblevenshtein::prelude::*;

#[test]
fn test_empty_query_with_unicode() {
    println!("\n=== Test Empty Query with Unicode ===\n");

    // Failing case from proptest - use DoubleArrayTrieChar for character-level semantics
    let dict = DoubleArrayTrieChar::from_terms(vec!["¡".to_string()]);
    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

    println!("Dictionary: [\"¡\"]");
    println!("Query: \"\" (empty)");
    println!("Max distance: 1\n");

    // With empty query and max_distance=1, we should be able to reach "¡"
    // by inserting one character
    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Results: {:?}", results);

    // Expected: ["¡"] (can reach by inserting one character)
    // Actual: []

    if results.contains(&"¡".to_string()) {
        println!("✓ Found \"¡\"");
    } else {
        println!("❌ Missing \"¡\"");
        println!("\nExpected behavior:");
        println!("  Empty query \"\" can match \"¡\" with 1 insertion");
        println!("  Distance = 1 (insert '¡')");
    }

    // Test with other algorithms
    println!("\n--- Testing other algorithms ---\n");

    for algo in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ] {
        let trans = Transducer::new(dict.clone(), algo);
        let results: Vec<_> = trans.query("", 1).collect();
        println!("{:?}: {:?}", algo, results);
    }

    // Also test with simple ASCII
    println!("\n--- Testing with ASCII ---\n");
    let dict_ascii = DoubleArrayTrie::from_terms(vec!["a".to_string()]);
    let trans_ascii = Transducer::new(dict_ascii, Algorithm::Standard);
    let results_ascii: Vec<_> = trans_ascii.query("", 1).collect();
    println!(
        "Dictionary: [\"a\"], Query: \"\", Results: {:?}",
        results_ascii
    );

    assert!(
        results.contains(&"¡".to_string()),
        "Empty query should match \"¡\" at distance 1"
    );
}

#[test]
fn test_empty_query_various_distances() {
    println!("\n=== Test Empty Query with Various Distances ===\n");

    let dict =
        DoubleArrayTrie::from_terms(vec!["a".to_string(), "ab".to_string(), "abc".to_string()]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    for max_dist in 0..=3 {
        let results: Vec<_> = transducer.query("", max_dist).collect();
        println!("Max distance {}: {:?}", max_dist, results);
    }

    // Expected:
    // Distance 0: [] (no exact matches with empty)
    // Distance 1: ["a"] (one insertion)
    // Distance 2: ["a", "ab"] (two insertions)
    // Distance 3: ["a", "ab", "abc"] (three insertions)
}

#[test]
fn test_empty_dictionary_with_query() {
    println!("\n=== Test Empty Dictionary with Query ===\n");

    let dict: DoubleArrayTrie = DoubleArrayTrie::from_terms(Vec::<String>::new());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("test", 5).collect();
    println!("Empty dict, query \"test\": {:?}", results);

    assert!(
        results.is_empty(),
        "Empty dictionary should return no results"
    );
}
