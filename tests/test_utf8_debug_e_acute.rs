use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::Dictionary;
use liblevenshtein::prelude::*;

#[test]
fn test_e_acute_dict_creation() {
    println!("\n=== Testing 'é' Dictionary Creation ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["é"]);

    println!("Created dictionary with: [\"é\"]");
    println!("Character 'é': {:?} (U+{:04X})", 'é', 'é' as u32);
    println!(
        "String \"é\": {} bytes, {} chars",
        "é".len(),
        "é".chars().count()
    );

    // Test contains
    let contains = dict.contains("é");
    println!("dict.contains(\"é\"): {}", contains);
    assert!(contains, "Dictionary should contain \"é\"");

    // Test root
    let root = dict.root();
    let edges: Vec<char> = root.edges().map(|(c, _)| c).collect();
    println!("Root edges: {:?}", edges);
    println!(
        "Root edges as U+: {:?}",
        edges
            .iter()
            .map(|&c| format!("U+{:04X}", c as u32))
            .collect::<Vec<_>>()
    );

    assert!(edges.contains(&'é'), "Root should have edge for 'é'");

    // Test transition
    let e_node = root.transition('é');
    assert!(e_node.is_some(), "Should be able to transition on 'é'");
    assert!(e_node.unwrap().is_final(), "'é' node should be final");

    println!("\n✓ Dictionary correctly stores 'é'");
}

#[test]
fn test_e_acute_empty_query() {
    println!("\n=== Testing Empty Query with 'é' ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["é"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"é\"]");
    println!("Query: \"\" (empty)");
    println!("Max distance: 1");

    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Results: {:?}", results);
    println!("Number of results: {}", results.len());

    if results.is_empty() {
        println!("\n❌ PROBLEM: Empty results when expecting [\"é\"]");
    } else {
        println!("\n✓ Found results");
    }

    assert!(
        results.contains(&"é".to_string()),
        "Empty query at distance 1 should find \"é\" (one insertion)"
    );
}

#[test]
fn test_e_acute_vs_plain_e() {
    println!("\n=== Testing 'é' vs 'e' ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["é", "e"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"é\", \"e\"]");

    // Empty query at distance 1
    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Empty query at distance 1: {:?}", results);

    // Should find both (each requires 1 insertion)
    assert!(results.len() >= 1, "Should find at least one result");
}

#[test]
fn test_multiple_e_acute() {
    println!("\n=== Testing Multiple 'é' Terms ===\n");

    let terms = vec!["é", "ée", "éée"];
    println!("Input terms: {:?}", terms);

    // Check what chars we're actually getting
    for term in &terms {
        let chars: Vec<char> = term.chars().collect();
        println!("  \"{}\": {} chars = {:?}", term, chars.len(), chars);
    }

    let dict = DoubleArrayTrieChar::from_terms(terms);

    // Test contains for each
    println!("\nTesting contains:");
    println!("  contains(\"é\"): {}", dict.contains("é"));
    println!("  contains(\"ée\"): {}", dict.contains("ée"));
    println!("  contains(\"éée\"): {}", dict.contains("éée"));

    assert!(dict.contains("é"));
    assert!(dict.contains("ée"));
    assert!(dict.contains("éée"));

    println!("\n✓ All terms stored correctly");
}
