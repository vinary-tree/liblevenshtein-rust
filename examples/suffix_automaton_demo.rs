//! Demonstration of suffix automaton for substring matching.

use liblevenshtein::dictionary::suffix_automaton::SuffixAutomaton;
use liblevenshtein::dictionary::{Dictionary, DictionaryNode};
use liblevenshtein::transducer::{Algorithm, Transducer};

fn main() {
    // Use simple text first for inspection
    let text = "ab";
    let dict = SuffixAutomaton::<()>::from_text(text);

    println!("=== Dictionary Info ===");
    println!("String count: {}", dict.string_count());
    println!("Dictionary length: {:?}", dict.len());
    println!("State count: {}", dict.state_count());

    println!("\n=== Automaton Structure ===");
    dict.debug_print();

    println!("\n=== Testing Dictionary::contains ===");
    println!("Dictionary contains 'ab': {}", dict.contains("ab"));
    println!("Dictionary contains 'a': {}", dict.contains("a"));
    println!("Dictionary contains 'b': {}", dict.contains("b"));

    // Test root node
    let root = dict.root();
    println!("\n=== Testing Root Node ===");
    println!("Root is_final: {}", root.is_final());
    println!("Root has edge 'a': {}", root.has_edge(b'a'));
    if let Some(a_node) = root.transition(b'a') {
        println!("After 'a' is_final: {}", a_node.is_final());
        if let Some(b_node) = a_node.transition(b'b') {
            println!("After 'ab' is_final: {}", b_node.is_final());
        }
    }

    println!("\n=== Testing with Transducer ===");
    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

    println!("\nQuery results for 'ab' with distance 0:");
    let results: Vec<_> = transducer.query("ab", 0).collect();
    println!("  Found {} results", results.len());
    for term in results.iter().take(10) {
        println!("  - {}", term);
    }

    println!("\nQuery results for 'a' with distance 0:");
    let results: Vec<_> = transducer.query("a", 0).collect();
    println!("  Found {} results", results.len());
    for term in results.iter().take(10) {
        println!("  - {}", term);
    }

    println!("\nQuery results for 'b' with distance 0:");
    let results: Vec<_> = transducer.query("b", 0).collect();
    println!("  Found {} results", results.len());
    for term in results.iter().take(10) {
        println!("  - {}", term);
    }
}
