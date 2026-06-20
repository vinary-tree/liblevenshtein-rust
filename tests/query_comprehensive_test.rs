//! Comprehensive tests for query iterators with varying distances and modifiers

use liblevenshtein::prelude::*;

// Test dictionary with predictable patterns
fn create_test_dict() -> DoubleArrayTrie {
    let terms = vec![
        "a", "ab", "abc", "abcd", "abcde", "b", "bc", "bcd", "bcde", "test", "testing", "tested",
        "tester", "tests", "best", "rest", "nest", "west", "quest", "foo", "food", "fool",
        "football", "bar", "bark", "barn", "barley",
    ];
    DoubleArrayTrie::from_terms(terms)
}

#[test]
fn test_ordered_query_distance_0() {
    // Distance 0 should return only exact matches
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("test", 0).collect();
    assert_eq!(results.len(), 1, "Distance 0 should match exactly 1 term");
    assert_eq!(results[0].term, "test");
    assert_eq!(results[0].distance, 0);
}

#[test]
fn test_ordered_query_distance_1() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("test", 1).collect();

    // Should find: test(0), tests(1), best/rest/nest/west (1 each)
    assert!(
        results.len() >= 5,
        "Distance 1 should find at least 5 matches, found {}",
        results.len()
    );

    // Verify ordering: distance 0 first, then distance 1 in lexicographic order
    assert_eq!(results[0].distance, 0);
    assert_eq!(results[0].term, "test");

    for r in &results[1..] {
        assert_eq!(r.distance, 1, "All remaining should be distance 1");
    }
}

#[test]
fn test_ordered_query_distance_2() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("test", 2).collect();

    // Should find many more matches
    assert!(
        results.len() >= 7,
        "Distance 2 should find at least 7 matches"
    );

    // Verify ordering by distance
    let mut prev_distance = 0;
    for r in &results {
        assert!(r.distance <= 2, "Distance should not exceed 2");
        assert!(
            r.distance >= prev_distance,
            "Results should be ordered by distance"
        );
        prev_distance = r.distance;
    }
}

#[test]
fn test_ordered_query_distance_10() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("test", 10).collect();

    // With distance 10, should match most/all terms
    assert!(results.len() >= 10, "Distance 10 should find many matches");

    // Verify ordering
    let mut prev_distance = 0;
    for r in &results {
        assert!(r.distance <= 10);
        assert!(
            r.distance >= prev_distance,
            "Results not ordered: {} came after {}",
            r.distance,
            prev_distance
        );
        prev_distance = r.distance;
    }
}

#[test]
fn test_ordered_query_large_distance() {
    // Regression test for the defect we just fixed
    let terms = vec!["foo", "bar", "baz", "quo", "qux"];
    let dict = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("quuo", 99).collect();

    assert_eq!(
        results.len(),
        5,
        "Should find all 5 terms with large distance"
    );

    // Verify all terms are present
    let term_set: std::collections::HashSet<_> = results.iter().map(|r| r.term.as_str()).collect();
    assert!(term_set.contains("foo"));
    assert!(term_set.contains("bar"));
    assert!(term_set.contains("baz"));
    assert!(term_set.contains("quo"));
    assert!(term_set.contains("qux"));
}

#[test]
fn test_unordered_query_distance_0() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("test", 0).collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], "test");
}

#[test]
fn test_unordered_query_distance_1() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("test", 1).collect();
    assert!(results.len() >= 5, "Should find at least 5 matches");
    assert!(results.contains(&"test".to_string()));
}

#[test]
fn test_unordered_query_distance_2() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("test", 2).collect();
    assert!(results.len() >= 7);
}

#[test]
fn test_unordered_query_large_distance() {
    let terms = vec!["foo", "bar", "baz", "quo", "qux"];
    let dict = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("quuo", 99).collect();

    assert_eq!(
        results.len(),
        5,
        "Unordered query should also find all 5 terms"
    );
}

#[test]
fn test_prefix_mode_distance_0() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("test", 0).prefix().collect();

    // Should find: test, testing, tested, tester, tests (all start with "test")
    assert_eq!(
        results.len(),
        5,
        "Prefix mode should find all terms starting with 'test'"
    );

    for r in &results {
        assert_eq!(r.distance, 0);
        assert!(
            r.term.starts_with("test"),
            "{} should start with 'test'",
            r.term
        );
    }
}

#[test]
fn test_prefix_mode_distance_1() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("tes", 1).prefix().collect();

    // Should find many terms approximately starting with "tes"
    assert!(
        results.len() >= 5,
        "Prefix mode with distance 1 should find multiple matches"
    );
}

#[test]
fn test_prefix_vs_standard_mode() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Standard mode with "test" and distance 0 finds only "test"
    let standard: Vec<_> = transducer.query_ordered("test", 0).collect();
    assert_eq!(standard.len(), 1);

    // Prefix mode with "test" and distance 0 finds all words starting with "test"
    let prefix: Vec<_> = transducer.query_ordered("test", 0).prefix().collect();
    assert!(prefix.len() >= 5);
}

#[test]
fn test_all_algorithms_distance_0() {
    let dict = create_test_dict();

    for algorithm in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ] {
        let transducer = Transducer::new(dict.clone(), algorithm);
        let results: Vec<_> = transducer.query_ordered("test", 0).collect();

        assert_eq!(
            results.len(),
            1,
            "Algorithm {:?} should find exact match",
            algorithm
        );
        assert_eq!(results[0].term, "test");
    }
}

#[test]
fn test_all_algorithms_distance_2() {
    let dict = create_test_dict();

    for algorithm in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ] {
        let transducer = Transducer::new(dict.clone(), algorithm);
        let results: Vec<_> = transducer.query_ordered("test", 2).collect();

        assert!(
            results.len() >= 5,
            "Algorithm {:?} should find matches",
            algorithm
        );

        // Verify ordering
        let mut prev_distance = 0;
        for r in &results {
            assert!(r.distance <= 2);
            assert!(r.distance >= prev_distance);
            prev_distance = r.distance;
        }
    }
}

#[test]
fn test_empty_query_distance_0() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("", 0).collect();

    // Empty query with distance 0 should match empty string only (if in dict)
    // Our test dict doesn't have empty string, so should be empty
    assert_eq!(
        results.len(),
        0,
        "Empty query with distance 0 should match nothing in our dict"
    );
}

#[test]
fn test_query_not_in_dict_distance_0() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("notindictionary", 0).collect();
    assert_eq!(results.len(), 0);
}

#[test]
fn test_query_not_in_dict_distance_1() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("xyz", 1).collect();

    // Might find some short terms like "a", "b" within distance 1
    // Verify all results are actually within distance 1
    for r in &results {
        assert!(r.distance <= 1);
    }
}

#[test]
fn test_ordered_query_returns_in_order() {
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query_ordered("test", 5).collect();

    // Verify strict ordering: first by distance, then lexicographically
    for i in 1..results.len() {
        let prev = &results[i - 1];
        let curr = &results[i];

        assert!(
            prev.distance < curr.distance
                || (prev.distance == curr.distance && prev.term <= curr.term),
            "Results not properly ordered: {:?} should come before {:?}",
            prev,
            curr
        );
    }
}

#[test]
fn test_distance_boundaries() {
    // Test edge cases around distance boundaries
    let dict = create_test_dict();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results_0: Vec<_> = transducer.query_ordered("test", 0).collect();
    let results_1: Vec<_> = transducer.query_ordered("test", 1).collect();

    // Results at distance 1 should be a superset of results at distance 0
    assert!(results_1.len() >= results_0.len());

    // All distance-0 results should also appear in distance-1 results
    for r0 in &results_0 {
        assert!(results_1.iter().any(|r1| r1.term == r0.term));
    }
}
