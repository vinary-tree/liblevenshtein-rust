use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::Dictionary;
use liblevenshtein::prelude::*;

#[test]
fn test_simple_char_dict() {
    println!("\n=== Simple Character Dictionary Test ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["a"]);
    println!("Dictionary contains: [\"a\"]");

    // Test contains
    assert!(dict.contains("a"));
    println!("✓ Contains \"a\"");

    // Test root and edges
    let root = dict.root();
    let edges: Vec<_> = root.edges().map(|(c, _)| c).collect();
    println!("Root edges: {:?}", edges);
    assert!(edges.contains(&'a'));

    // Test transition
    let a_node = root.transition('a');
    assert!(a_node.is_some());
    let a_node = a_node.unwrap();
    assert!(a_node.is_final());
    println!("✓ Can transition to 'a' and it's final");
}

#[test]
fn test_empty_query_simple() {
    println!("\n=== Empty Query Simple Test ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["a"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"a\"]");
    println!("Query: \"\" (empty)");
    println!("Max distance: 1");

    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Results: {:?}", results);

    assert!(
        results.contains(&"a".to_string()),
        "Empty query at distance 1 should find \"a\" (one insertion)"
    );
}

#[test]
fn test_exact_match_simple() {
    println!("\n=== Exact Match Simple Test ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["hello"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"hello\"]");
    println!("Query: \"hello\"");
    println!("Max distance: 0");

    let results: Vec<_> = transducer.query("hello", 0).collect();
    println!("Results: {:?}", results);

    assert!(
        results.contains(&"hello".to_string()),
        "Exact match should find \"hello\""
    );
}

#[test]
fn test_one_edit_simple() {
    println!("\n=== One Edit Simple Test ===\n");

    let dict = DoubleArrayTrieChar::from_terms(vec!["hello"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"hello\"]");
    println!("Query: \"hallo\"");
    println!("Max distance: 1");

    let results: Vec<_> = transducer.query("hallo", 1).collect();
    println!("Results: {:?}", results);

    assert!(
        results.contains(&"hello".to_string()),
        "Query \"hallo\" at distance 1 should find \"hello\" (substitute a→e)"
    );
}
