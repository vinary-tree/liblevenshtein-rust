//! Test for deletion defect fixes

use liblevenshtein::prelude::*;

#[test]
fn test_aple_to_apple() {
    let dict = DoubleArrayTrie::from_terms(vec!["apple"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("aple", 1).collect();

    println!("Query 'aple' against 'apple' with distance 1:");
    for term in &results {
        println!("  Found: {}", term);
    }

    assert!(
        !results.is_empty(),
        "Should find 'apple' from query 'aple' with distance 1"
    );
    assert!(
        results.contains(&"apple".to_string()),
        "Results should contain 'apple'"
    );
}

#[test]
fn test_apple_to_aple() {
    let dict = DoubleArrayTrie::from_terms(vec!["aple"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("apple", 1).collect();

    println!("Query 'apple' against 'aple' with distance 1:");
    for term in &results {
        println!("  Found: {}", term);
    }

    assert!(
        !results.is_empty(),
        "Should find 'aple' from query 'apple' with distance 1"
    );
    assert!(
        results.contains(&"aple".to_string()),
        "Results should contain 'aple'"
    );
}

#[test]
fn test_deletion_operations() {
    // Test various deletion scenarios
    let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "apple", "world"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Query has extra character (deletion from query)
    let results: Vec<_> = transducer.query("testt", 1).collect();
    println!("Query 'testt' with distance 1: {:?}", results);
    assert!(
        results.contains(&"test".to_string()),
        "'testt' should match 'test'"
    );

    // Query missing character (insertion from dictionary)
    let results: Vec<_> = transducer.query("aple", 1).collect();
    println!("Query 'aple' with distance 1: {:?}", results);
    assert!(
        results.contains(&"apple".to_string()),
        "'aple' should match 'apple'"
    );
}

#[test]
fn test_exact_match_with_prefix() {
    let dict = DoubleArrayTrie::from_terms(vec!["z", "za"]);
    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

    // Verify dictionary contains both terms
    assert!(dict.contains("z"));
    assert!(dict.contains("za"));

    // Query for "z" should work
    let results_z: Vec<_> = transducer.query("z", 0).collect();
    assert!(
        results_z.contains(&"z".to_string()),
        "Should find exact match for 'z'"
    );

    // Query for "za" should work but currently fails
    let results_za: Vec<_> = transducer.query("za", 0).collect();
    assert!(
        results_za.contains(&"za".to_string()),
        "Should find exact match for 'za'"
    );
}

#[test]
fn test_dat_conflict_rh_qpo_ry() {
    // Regression test: proptest found this minimal case
    // Defect: DAT conflict resolution didn't preserve all children
    let dict = DoubleArrayTrie::from_terms(vec!["rh", "qpo", "ry"]);

    assert!(dict.contains("rh"), "DAT should contain 'rh'");
    assert!(dict.contains("qpo"), "DAT should contain 'qpo'");
    assert!(dict.contains("ry"), "DAT should contain 'ry'");

    let transducer = Transducer::new(dict, Algorithm::Standard);
    let results: Vec<_> = transducer.query("rh", 0).collect();
    assert!(results.contains(&"rh".to_string()), "Should find 'rh'");
}

#[test]
fn test_dat_grandchildren_gjzhkidoa_gl() {
    // Regression test: proptest found this minimal case
    // Defect: DAT relocation didn't update grandchildren's CHECK pointers
    let dict = DoubleArrayTrie::from_terms(vec!["gjzhkidoa", "gl"]);

    assert!(dict.contains("gjzhkidoa"), "DAT should contain 'gjzhkidoa'");
    assert!(dict.contains("gl"), "DAT should contain 'gl'");

    let transducer = Transducer::new(dict, Algorithm::Standard);
    let results: Vec<_> = transducer.query("gjzhkidoa", 0).collect();
    assert!(
        results.contains(&"gjzhkidoa".to_string()),
        "Should find 'gjzhkidoa'"
    );
}

#[test]
fn test_levenshtein_prefix_ve_v() {
    // Regression test: proptest found this case
    // Defect: query iteration didn't explore children of final nodes with distance > max
    let dict = DoubleArrayTrie::from_terms(vec!["ve", "v"]);

    let transducer = Transducer::new(dict, Algorithm::Standard);
    let results: Vec<_> = transducer.query("ae", 1).collect();

    // "ve" has distance 1 from "ae" (substitute 'a'->'v', match 'e')
    assert!(
        results.contains(&"ve".to_string()),
        "Should find 've' from query 'ae' with distance 1"
    );
}

#[test]
fn test_dat_base_calculation_n_a_ag() {
    // Regression test: original DAT defect from first proptest run
    // Defect: find_free_base was subtracting bytes[0], causing off-by-one errors
    let dict = DoubleArrayTrie::from_terms(vec!["n", "a", "ag"]);

    assert!(dict.contains("n"), "DAT should contain 'n'");
    assert!(dict.contains("a"), "DAT should contain 'a'");
    assert!(dict.contains("ag"), "DAT should contain 'ag'");

    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results_a: Vec<_> = transducer.query("a", 0).collect();
    assert!(results_a.contains(&"a".to_string()), "Should find 'a'");

    let results_ag: Vec<_> = transducer.query("ag", 0).collect();
    assert!(results_ag.contains(&"ag".to_string()), "Should find 'ag'");
}
