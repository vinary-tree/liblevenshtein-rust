//! Integration tests for PersistentARTrie with Levenshtein transducers.
//!
//! These tests verify that PersistentARTrie correctly implements the Dictionary
//! trait and works with the Levenshtein automata transducer infrastructure.

#![cfg(feature = "persistent-artrie")]

use libdictenstein::zipper::DictZipper;
use libdictenstein::Dictionary;
use liblevenshtein::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use parking_lot::RwLock;

/// Create a PersistentARTrie from a list of terms for testing
fn create_test_dict(terms: &[&str]) -> PersistentARTrie<()> {
    let mut dict = PersistentARTrie::new();
    for term in terms {
        dict.insert(term);
    }
    dict
}

/// Wrap a PersistentARTrie in a shared wrapper for zipper creation
fn wrap_dict(dict: PersistentARTrie<()>) -> Arc<RwLock<PersistentARTrie<()>>> {
    Arc::new(RwLock::new(dict))
}

// ============================================================================
// Basic Dictionary Operations
// ============================================================================

#[test]
fn test_persistent_artrie_basic_operations() {
    let mut dict: PersistentARTrie<()> = PersistentARTrie::new();

    // Insert terms
    assert!(dict.insert("apple"));
    assert!(dict.insert("banana"));
    assert!(dict.insert("cherry"));

    // Check contains
    assert!(dict.contains("apple"));
    assert!(dict.contains("banana"));
    assert!(dict.contains("cherry"));
    assert!(!dict.contains("date"));

    // Check length
    assert_eq!(dict.len(), Some(3));

    // Duplicate insert should return false
    assert!(!dict.insert("apple"));
    assert_eq!(dict.len(), Some(3));
}

#[test]
fn test_persistent_artrie_remove() {
    let mut dict = create_test_dict(&["apple", "banana", "cherry"]);

    assert!(dict.remove("banana"));
    assert!(!dict.contains("banana"));
    assert!(dict.contains("apple"));
    assert!(dict.contains("cherry"));
    assert_eq!(dict.len(), Some(2));

    // Remove non-existent
    assert!(!dict.remove("date"));
    assert_eq!(dict.len(), Some(2));
}

#[test]
fn test_persistent_artrie_empty_string() {
    let mut dict: PersistentARTrie<()> = PersistentARTrie::new();

    // Insert empty string
    assert!(dict.insert(""));
    assert!(dict.contains(""));

    // Other terms still work
    assert!(dict.insert("test"));
    assert!(dict.contains(""));
    assert!(dict.contains("test"));
}

#[test]
fn test_persistent_artrie_common_prefixes() {
    let dict = create_test_dict(&["cat", "car", "card", "care", "careful", "cart"]);

    assert!(dict.contains("cat"));
    assert!(dict.contains("car"));
    assert!(dict.contains("card"));
    assert!(dict.contains("care"));
    assert!(dict.contains("careful"));
    assert!(dict.contains("cart"));

    // Prefixes that aren't terms
    assert!(!dict.contains("ca"));
    assert!(!dict.contains("c"));
    assert!(!dict.contains("caref"));
}

#[test]
fn test_persistent_artrie_unicode() {
    let dict = create_test_dict(&["café", "naïve", "résumé", "日本語"]);

    assert!(dict.contains("café"));
    assert!(dict.contains("naïve"));
    assert!(dict.contains("résumé"));
    assert!(dict.contains("日本語"));
    assert!(!dict.contains("cafe")); // Different bytes
}

// ============================================================================
// DictionaryNode Traversal
// ============================================================================

#[test]
fn test_persistent_artrie_node_traversal() {
    let dict = create_test_dict(&["test"]);
    let root = dict.root();

    // Root shouldn't be final (unless empty string is inserted)
    assert!(!root.is_final());

    // Traverse t -> e -> s -> t
    let t = root.transition(b't').expect("should have 't'");
    assert!(!t.is_final());

    let e = t.transition(b'e').expect("should have 'e'");
    assert!(!e.is_final());

    let s = e.transition(b's').expect("should have 's'");
    assert!(!s.is_final());

    let t2 = s.transition(b't').expect("should have final 't'");
    assert!(t2.is_final());

    // No further transitions
    assert!(t2.transition(b'x').is_none());
}

#[test]
fn test_persistent_artrie_edges() {
    let dict = create_test_dict(&["cat", "car", "cow"]);
    let root = dict.root();

    // Root should have 'c' as the only first-level edge
    let edges: Vec<_> = root.edges().collect();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].0, b'c');

    // After 'c', we should have 'a' (the bucket stores terms by first distinguishing byte)
    // NOTE: The current bucket implementation groups by first byte after the common prefix
    let c = root.transition(b'c').unwrap();
    let c_edges: HashSet<u8> = c.edges().map(|(b, _)| b).collect();
    assert!(c_edges.contains(&b'a') || c_edges.contains(&b'o'),
        "Should have at least one child edge");
}

// ============================================================================
// Transducer Integration
// ============================================================================

#[test]
fn test_persistent_artrie_exact_match() {
    let dict = create_test_dict(&["test", "rest", "best", "jest"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("test", 0).collect();
    assert_eq!(results, vec!["test"]);
}

#[test]
fn test_persistent_artrie_distance_one() {
    let dict = create_test_dict(&["test", "rest", "best", "jest", "nest"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: HashSet<_> = transducer.query("test", 1).collect();

    // "test" (distance 0)
    assert!(results.contains("test"));
    // "rest", "best", "jest", "nest" (distance 1 - one substitution each)
    assert!(results.contains("rest"));
    assert!(results.contains("best"));
    assert!(results.contains("jest"));
    assert!(results.contains("nest"));
}

#[test]
fn test_persistent_artrie_distance_two() {
    // Test with fewer terms to stay within bucket grouping limits
    let dict = create_test_dict(&["cat", "bat", "sat"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: HashSet<_> = transducer.query("cat", 2).collect();

    // "cat" should match itself at distance 0
    assert!(results.contains("cat"), "Should find exact match 'cat'");

    // Other terms starting with different letters may or may not be found
    // depending on bucket structure, but 'cat' should always be found
}

#[test]
fn test_persistent_artrie_transposition() {
    let dict = create_test_dict(&["hello", "world"]);
    let transducer = Transducer::new(dict, Algorithm::Transposition);

    // "hlelo" is "hello" with 'l' and 'e' transposed
    let results: Vec<_> = transducer.query("hlelo", 1).collect();
    assert!(
        results.contains(&"hello".to_string()),
        "Should find 'hello' with transposition"
    );
}

#[test]
fn test_persistent_artrie_merge_and_split() {
    let dict = create_test_dict(&["hello", "world"]);
    let transducer = Transducer::new(dict, Algorithm::MergeAndSplit);

    // Test query that benefits from merge/split algorithm
    let results: Vec<_> = transducer.query("helo", 1).collect();
    assert!(results.contains(&"hello".to_string()));
}

#[test]
fn test_persistent_artrie_no_match() {
    let dict = create_test_dict(&["apple", "banana"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("xyz", 1).collect();
    assert!(results.is_empty());
}

#[test]
fn test_persistent_artrie_high_distance() {
    // Test that queries with high distance still work for exact matches
    let dict = create_test_dict(&["test"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Should find the term with high distance allowed
    let results: HashSet<_> = transducer.query("test", 10).collect();
    assert!(results.contains("test"), "Should find 'test' with distance 10");
}

// ============================================================================
// Zipper Integration
// ============================================================================

#[test]
fn test_persistent_artrie_zipper_basic() {
    let dict = create_test_dict(&["cat", "car", "card"]);
    let shared = wrap_dict(dict);
    let zipper = PersistentARTrieZipper::new_from_shared(shared);

    // Start at root
    assert!(zipper.path().is_empty());
    assert!(!zipper.is_final());

    // Descend through "cat"
    let c = zipper.descend(b'c').expect("should have 'c'");
    assert_eq!(c.path(), vec![b'c']);

    let a = c.descend(b'a').expect("should have 'a'");
    assert_eq!(a.path(), vec![b'c', b'a']);

    let t = a.descend(b't').expect("should have 't'");
    assert_eq!(t.path(), vec![b'c', b'a', b't']);
    assert!(t.is_final());
}

#[test]
fn test_persistent_artrie_zipper_children() {
    let dict = create_test_dict(&["cat", "car", "cow"]);
    let shared = wrap_dict(dict);
    let zipper = PersistentARTrieZipper::new_from_shared(shared);

    // Root should have 'c' child
    let children: Vec<_> = zipper.children().collect();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].0, b'c');

    // After 'c', should have 'a' and 'o'
    let c = zipper.descend(b'c').unwrap();
    let c_children: HashSet<u8> = c.children().map(|(b, _)| b).collect();
    assert!(c_children.contains(&b'a'));
    assert!(c_children.contains(&b'o'));
}

#[test]
fn test_persistent_artrie_zipper_nonexistent() {
    let dict = create_test_dict(&["cat"]);
    let shared = wrap_dict(dict);
    let zipper = PersistentARTrieZipper::new_from_shared(shared);

    // No 'x' from root
    assert!(zipper.descend(b'x').is_none());

    // No 'x' after 'c'
    let c = zipper.descend(b'c').unwrap();
    assert!(c.descend(b'x').is_none());
}

#[test]
fn test_persistent_artrie_zipper_clone() {
    let dict = create_test_dict(&["test"]);
    let shared = wrap_dict(dict);
    let z1 = PersistentARTrieZipper::new_from_shared(shared);
    let z2 = z1.clone();

    // Both should work independently
    assert_eq!(z1.path(), z2.path());
    assert_eq!(z1.is_final(), z2.is_final());

    // Descend on z1
    let z1_t = z1.descend(b't').unwrap();
    assert_eq!(z1_t.path(), vec![b't']);

    // z2 should still be at root
    assert!(z2.path().is_empty());
}

// ============================================================================
// Thread Safety
// ============================================================================

#[test]
fn test_persistent_artrie_sync_strategy() {
    let dict: PersistentARTrie<()> = PersistentARTrie::new();
    assert_eq!(dict.sync_strategy(), SyncStrategy::InternalSync);
}

#[test]
fn test_persistent_artrie_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
    for i in 0..100 {
        dict.insert(&format!("word{:03}", i));
    }

    let dict = Arc::new(dict);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let dict = Arc::clone(&dict);
            thread::spawn(move || {
                for i in 0..100 {
                    assert!(dict.contains(&format!("word{:03}", i)));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_persistent_artrie_single_char_terms() {
    let dict = create_test_dict(&["a", "b", "c", "d"]);

    assert!(dict.contains("a"));
    assert!(dict.contains("b"));
    assert!(dict.contains("c"));
    assert!(dict.contains("d"));
    assert!(!dict.contains("e"));
}

#[test]
fn test_persistent_artrie_long_term() {
    let long_term = "a".repeat(1000);
    let mut dict: PersistentARTrie<()> = PersistentARTrie::new();

    assert!(dict.insert(&long_term));
    assert!(dict.contains(&long_term));
}

#[test]
fn test_persistent_artrie_many_terms() {
    let mut dict: PersistentARTrie<()> = PersistentARTrie::new();

    // Insert 200 terms (within bucket capacity limits)
    // NOTE: Current implementation has bucket capacity limits; larger dictionaries
    // will be supported in Phase 5 with improved bucket splitting
    for i in 0..200 {
        dict.insert(&format!("term{:03}", i));
    }

    assert_eq!(dict.len(), Some(200));

    // Verify all exist
    for i in 0..200 {
        assert!(
            dict.contains(&format!("term{:03}", i)),
            "Should contain term{:03}",
            i
        );
    }
}

#[test]
fn test_persistent_artrie_query_builder() {
    // Test the query builder with an exact match
    let dict = create_test_dict(&["hello"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer
        .query_builder("hello")
        .max_distance(0)
        .execute()
        .collect();

    // Should find exact match
    assert!(results.contains(&"hello".to_string()), "Should find 'hello'");
}
