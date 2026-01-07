//! Integration tests for SuffixAutomaton with Transducer.
//!
//! These tests verify that the suffix automaton works correctly with
//! the existing Levenshtein transducer infrastructure.

use libdictenstein::suffix_automaton::SuffixAutomaton;
use liblevenshtein::transducer::{Algorithm, Transducer};

#[test]
fn test_exact_substring_match() {
    let text = "The quick brown fox jumps over the lazy dog";
    let dict = SuffixAutomaton::<()>::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Exact matches with distance 0
    let results: Vec<_> = transducer.query("quick", 0).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "quick"));

    let results: Vec<_> = transducer.query("fox", 0).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "fox"));
}

#[test]
fn test_approximate_substring_match() {
    let text = "testing approximate matching in suffix automaton";
    let dict = SuffixAutomaton::<()>::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Should find "approximate" with 1 typo
    let results: Vec<_> = transducer.query("aproximate", 1).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "approximate"));

    // Should find "matching" with 1 edit
    let results: Vec<_> = transducer.query("matchng", 1).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "matching"));
}

#[test]
fn test_ordered_query() {
    let text = "algorithm implementation test";
    let dict = SuffixAutomaton::<()>::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Query with ordered results
    let results: Vec<_> = transducer.query_ordered("algoritm", 2).collect();

    // Should find "algorithm" with distance 1 (missing 'h')
    assert!(!results.is_empty());

    // Verify results are ordered by distance
    let mut prev_distance = 0;
    for candidate in results {
        assert!(candidate.distance >= prev_distance);
        prev_distance = candidate.distance;
    }
}

#[test]
fn test_multiple_documents() {
    let docs = vec![
        "first document with test content",
        "second document also has test",
        "third one contains testing",
    ];
    let dict = SuffixAutomaton::<()>::from_texts(docs);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Search across all documents
    let results: Vec<_> = transducer.query("test", 0).collect();
    assert!(!results.is_empty());

    // Multiple occurrences of "test" should be found
    let test_count = results.iter().filter(|term| *term == "test").count();
    assert!(test_count > 0);
}

#[test]
fn test_code_search_example() {
    let code = r#"
    fn calculate_total(items: &[Item]) -> f64 {
        items.iter().map(|i| i.price).sum()
    }

    fn calculate_average(items: &[Item]) -> f64 {
        let total = calculate_total(items);
        total / items.len() as f64
    }
    "#;

    let dict = SuffixAutomaton::<()>::from_text(code);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Find function names with typos
    let results: Vec<_> = transducer.query("calculat", 1).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term.starts_with("calculate")));

    // Find variable names
    let results: Vec<_> = transducer.query("items", 0).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "items"));
}

#[test]
fn test_substring_vs_prefix_behavior() {
    // Compare suffix automaton (substring matching) behavior
    let text = "contestation";
    let dict = SuffixAutomaton::<()>::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Suffix automaton should find substrings anywhere
    let results: Vec<_> = transducer.query("test", 0).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "test")); // "test" appears in "contestation"

    let results: Vec<_> = transducer.query("station", 0).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "station")); // "station" is a suffix
}

#[test]
fn test_dynamic_updates_with_queries() {
    let dict = SuffixAutomaton::<()>::new();
    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

    // Initially empty
    let results: Vec<_> = transducer.query("test", 0).collect();
    assert!(results.is_empty());

    // Add content
    dict.insert("testing dynamic updates");

    // Should now find matches
    let results: Vec<_> = transducer.query("test", 0).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "test"));

    // Add more content
    dict.insert("another test case");

    // Should find matches from both strings
    let results: Vec<_> = transducer.query("test", 0).collect();
    assert!(!results.is_empty());
}

#[test]
fn test_empty_query() {
    let text = "sample text";
    let dict = SuffixAutomaton::<()>::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Empty query with suffix automaton returns all substrings (correct behavior)
    // An empty query matches everything at distance 0
    let results: Vec<_> = transducer.query("", 0).collect();
    // Should contain all possible substrings from "sample text"
    assert!(!results.is_empty());
    assert!(results.contains(&"sample text".to_string()));
    assert!(results.contains(&"sample".to_string()));
    assert!(results.contains(&"text".to_string()));
}

#[test]
fn test_distance_variations() {
    let text = "programming";
    let dict = SuffixAutomaton::<()>::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Distance 0: exact match only
    let results_d0: Vec<_> = transducer.query("program", 0).collect();
    assert!(results_d0.iter().any(|term| term == "program"));

    // Distance 1: should include more variations
    let results_d1: Vec<_> = transducer.query("program", 1).collect();
    assert!(results_d1.len() >= results_d0.len());

    // Distance 2: even more variations
    let results_d2: Vec<_> = transducer.query("program", 2).collect();
    assert!(results_d2.len() >= results_d1.len());
}

#[test]
fn test_special_characters() {
    let text = "hello-world test_case foo.bar";
    let dict = SuffixAutomaton::<()>::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Should handle special characters
    let results: Vec<_> = transducer.query("hello", 0).collect();
    assert!(!results.is_empty());

    let results: Vec<_> = transducer.query("world", 0).collect();
    assert!(!results.is_empty());

    let results: Vec<_> = transducer.query("test", 0).collect();
    assert!(!results.is_empty());
}

#[test]
fn test_unicode_text() {
    let text = "hello мир 世界 🌍";
    let dict = SuffixAutomaton::<()>::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // ASCII substring
    let results: Vec<_> = transducer.query("hello", 0).collect();
    assert!(!results.is_empty());
    assert!(results.iter().any(|term| term == "hello"));

    // Note: Unicode support is byte-based, so multi-byte characters
    // may not work as expected without grapheme cluster support
}
