use libdictenstein::Dictionary;
use liblevenshtein::prelude::*;
use std::collections::HashSet;

#[test]
fn test_high_distance_returns_all_terms() {
    // Regression test where high max_distance didn't return all terms
    let terms = vec!["foo", "bar", "baz", "qux", "quo"];
    let dict = DoubleArrayTrie::from_terms(terms.clone());

    let transducer = Transducer::new(dict, Algorithm::Standard);
    let results: Vec<_> = transducer.query("quuo", 99).collect();

    eprintln!("Found {} results:", results.len());
    for term in &results {
        eprintln!("  {}", term);
    }

    let found_terms: HashSet<_> = results.iter().map(|t| t.as_str()).collect();
    let expected_terms: HashSet<_> = terms.iter().copied().collect();

    assert_eq!(
        found_terms,
        expected_terms,
        "With max_distance=99, all terms should be found. Missing: {:?}",
        expected_terms.difference(&found_terms).collect::<Vec<_>>()
    );
}

#[test]
fn test_query_longer_than_term() {
    // Compute actual Levenshtein distance
    fn naive_lev(s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }
        for (j, cell) in matrix[0].iter_mut().enumerate().take(len2 + 1) {
            *cell = j;
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1),
                    matrix[i - 1][j - 1] + cost,
                );
            }
        }

        matrix[len1][len2]
    }

    let query = "aahaara";
    let term = "hr";
    let actual_distance = naive_lev(query, term);
    eprintln!(
        "Actual Levenshtein distance from '{}' to '{}': {}",
        query, term, actual_distance
    );

    let dict = DoubleArrayTrie::from_terms(vec![term]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Test with different distances to find the boundary
    for d in 1..=10 {
        let results: Vec<_> = transducer.query(query, d).collect();
        eprintln!(
            "Distance {}: Found {} results: {:?}",
            d,
            results.len(),
            results
        );
    }

    // Should find with actual distance
    let results: Vec<_> = transducer.query(query, actual_distance).collect();
    assert!(
        results.contains(&term.to_string()),
        "Should find '{}' when searching '{}' with distance {}",
        term,
        query,
        actual_distance
    );
}

#[test]
fn test_basic_dictionary_ops() {
    let dict = DoubleArrayTrie::from_terms(vec!["test"]);
    let root = dict.root();

    println!("Root is_final: {}", root.is_final());

    let t = root.transition(b't');
    assert!(t.is_some(), "Should have 't' transition");
    let t_node = t.unwrap();
    println!("After 't', is_final: {}", t_node.is_final());

    let e = t_node.transition(b'e');
    assert!(e.is_some(), "Should have 'e' transition");
    let e_node = e.unwrap();
    println!("After 'te', is_final: {}", e_node.is_final());

    let s = e_node.transition(b's');
    assert!(s.is_some(), "Should have 's' transition");
    let s_node = s.unwrap();
    println!("After 'tes', is_final: {}", s_node.is_final());

    let t2 = s_node.transition(b't');
    assert!(t2.is_some(), "Should have second 't' transition");
    let t2_node = t2.unwrap();
    println!("After 'test', is_final: {}", t2_node.is_final());
    assert!(t2_node.is_final(), "'test' should be final!");
}

#[test]
fn test_simple_query() {
    let dict = DoubleArrayTrie::from_terms(vec!["test"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("test", 0).collect();
    println!("Results: {:?}", results);
    assert_eq!(results, vec!["test"]);
}

#[test]
fn test_distance_zero() {
    // Test that distance=0 only returns exact matches
    let terms = vec!["test", "rest", "best", "jest"];
    let dict = DoubleArrayTrie::from_terms(terms.clone());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Exact match should return exactly one result
    let results: Vec<_> = transducer.query("test", 0).collect();
    assert_eq!(
        results,
        vec!["test"],
        "Distance 0 should only return exact matches"
    );

    // Non-match should return nothing
    let results: Vec<_> = transducer.query("fest", 0).collect();
    assert_eq!(
        results.len(),
        0,
        "Distance 0 should return nothing for non-exact matches"
    );
}

#[test]
fn test_distance_max() {
    // Test that distance=usize::MAX returns all terms in the dictionary
    let terms = vec![
        "foo",
        "bar",
        "baz",
        "qux",
        "quo",
        "abcdefghijklmnopqrstuvwxyz",
    ];
    let dict = DoubleArrayTrie::from_terms(terms.clone());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // With max distance, should return all terms
    let results: Vec<_> = transducer.query("x", usize::MAX).collect();

    let found_terms: HashSet<_> = results.iter().map(|t| t.as_str()).collect();
    let expected_terms: HashSet<_> = terms.iter().copied().collect();

    assert_eq!(
        found_terms,
        expected_terms,
        "With max_distance=usize::MAX, all terms should be found. Missing: {:?}",
        expected_terms.difference(&found_terms).collect::<Vec<_>>()
    );
}

#[test]
fn test_all_algorithms_distance_zero() {
    // Test distance=0 for all algorithms
    let terms = vec!["test", "rest", "best"];
    let dict = DoubleArrayTrie::from_terms(terms.clone());

    for algorithm in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ] {
        let transducer = Transducer::new(dict.clone(), algorithm);
        let results: Vec<_> = transducer.query("test", 0).collect();
        assert_eq!(
            results,
            vec!["test"],
            "Algorithm {:?} with distance 0 should only return exact match",
            algorithm
        );
    }
}

#[test]
fn test_all_algorithms_distance_max() {
    // Test distance=usize::MAX for all algorithms
    let terms = vec!["foo", "bar", "baz"];
    let dict = DoubleArrayTrie::from_terms(terms.clone());

    for algorithm in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ] {
        let transducer = Transducer::new(dict.clone(), algorithm);
        let results: Vec<_> = transducer.query("x", usize::MAX).collect();

        let found_terms: HashSet<_> = results.iter().map(|t| t.as_str()).collect();
        let expected_terms: HashSet<_> = terms.iter().copied().collect();

        assert_eq!(
            found_terms,
            expected_terms,
            "Algorithm {:?} with max_distance=usize::MAX should return all terms. Missing: {:?}",
            algorithm,
            expected_terms.difference(&found_terms).collect::<Vec<_>>()
        );
    }
}
