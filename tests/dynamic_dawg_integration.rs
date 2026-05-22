//! Integration tests for DynamicDawg with FuzzyMultiMap and eviction wrappers.
//!
//! These tests verify that DynamicDawg<V> correctly implements MappedDictionary
//! and works seamlessly with all cache and fuzzy query components.

#![cfg(feature = "pathmap-backend")]

use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::MappedDictionary;
use liblevenshtein::cache::eviction::{Age, Lfu, Lru, Ttl};
use liblevenshtein::cache::multimap::FuzzyMultiMap;
use liblevenshtein::prelude::*;
use std::collections::HashSet;
use std::thread;
use std::time::Duration;

#[test]
fn test_dynamic_dawg_with_fuzzy_multimap() {
    // Create DynamicDawg with HashSet values
    let dawg: DynamicDawg<HashSet<u32>> = DynamicDawg::new();
    dawg.insert_with_value("foo", HashSet::from([1, 2]));
    dawg.insert_with_value("bar", HashSet::from([3]));
    dawg.insert_with_value("baz", HashSet::from([4, 5]));

    let fuzzy = FuzzyMultiMap::new(dawg, Algorithm::Standard);

    // Query "bat" with distance 1
    // Should match "bar" (distance 1) and "baz" (distance 1)
    let result = fuzzy.query("bat", 1).unwrap();
    assert_eq!(result, HashSet::from([3, 4, 5]));
}

#[test]
fn test_dynamic_dawg_with_fuzzy_multimap_vec() {
    // Create DynamicDawg with Vec values
    let dawg: DynamicDawg<Vec<i32>> = DynamicDawg::new();
    dawg.insert_with_value("foo", vec![1, 2]);
    dawg.insert_with_value("fob", vec![3]);
    dawg.insert_with_value("fog", vec![4, 5]);

    let fuzzy = FuzzyMultiMap::new(dawg, Algorithm::Standard);

    // Query "fox" with distance 1
    // Should match "fob" and "fog"
    let result = fuzzy.query("fox", 1).unwrap();
    assert!(result.contains(&3));
    assert!(result.contains(&4));
    assert!(result.contains(&5));
}

#[test]
fn test_dynamic_dawg_with_lru_wrapper() {
    // Create DynamicDawg with u32 values
    let dawg: DynamicDawg<u32> = DynamicDawg::new();
    dawg.insert_with_value("foo", 42);
    dawg.insert_with_value("bar", 99);
    dawg.insert_with_value("baz", 123);

    let lru = Lru::new(dawg);

    // Access all entries in order
    assert_eq!(lru.get_value("foo"), Some(42));
    thread::sleep(Duration::from_millis(10));
    assert_eq!(lru.get_value("bar"), Some(99));
    thread::sleep(Duration::from_millis(10));
    assert_eq!(lru.get_value("baz"), Some(123));

    // "foo" should be least recently used
    let lru_term = lru.find_lru(&["foo", "bar", "baz"]);
    assert_eq!(lru_term, Some("foo".to_string()));
}

#[test]
fn test_dynamic_dawg_with_ttl_wrapper() {
    let dawg: DynamicDawg<String> = DynamicDawg::new();
    dawg.insert_with_value("test", "value".to_string());

    let ttl = Ttl::new(dawg, Duration::from_secs(1));

    // Value should be accessible initially
    assert_eq!(ttl.get_value("test"), Some("value".to_string()));

    // Wait for TTL to expire
    thread::sleep(Duration::from_secs(2));

    // Value should now return None (expired)
    assert_eq!(ttl.get_value("test"), None);
}

#[test]
fn test_dynamic_dawg_with_age_wrapper() {
    let dawg: DynamicDawg<i32> = DynamicDawg::new();
    dawg.insert_with_value("first", 1);

    let age = Age::new(dawg);
    assert_eq!(age.get_value("first"), Some(1));

    thread::sleep(Duration::from_millis(10));

    age.inner().insert_with_value("second", 2);
    assert_eq!(age.get_value("second"), Some(2));

    thread::sleep(Duration::from_millis(10));

    age.inner().insert_with_value("third", 3);
    assert_eq!(age.get_value("third"), Some(3));

    // "first" should be oldest
    let oldest = age.find_oldest(&["first", "second", "third"]);
    assert_eq!(oldest, Some("first".to_string()));
}

#[test]
fn test_dynamic_dawg_with_lfu_wrapper() {
    let dawg: DynamicDawg<u64> = DynamicDawg::new();
    dawg.insert_with_value("rare", 1);
    dawg.insert_with_value("common", 2);

    let lfu = Lfu::new(dawg);

    // Access "common" multiple times
    assert_eq!(lfu.get_value("rare"), Some(1));
    assert_eq!(lfu.get_value("common"), Some(2));
    assert_eq!(lfu.get_value("common"), Some(2));
    assert_eq!(lfu.get_value("common"), Some(2));

    // "rare" should be least frequently used
    let lfu_term = lfu.find_lfu(&["rare", "common"]);
    assert_eq!(lfu_term, Some("rare".to_string()));

    // Verify access counts
    assert_eq!(lfu.access_count("rare"), Some(1));
    assert_eq!(lfu.access_count("common"), Some(3));
}

#[test]
fn test_dynamic_dawg_with_composed_wrappers() {
    // Stack multiple wrappers: Lru -> Ttl -> Age -> DynamicDawg
    let dawg: DynamicDawg<u32> = DynamicDawg::new();
    dawg.insert_with_value("alpha", 1);
    dawg.insert_with_value("beta", 2);
    dawg.insert_with_value("gamma", 3);

    let age = Age::new(dawg);
    let ttl = Ttl::new(age, Duration::from_secs(2));
    let lru = Lru::new(ttl);

    // Access all entries
    assert_eq!(lru.get_value("alpha"), Some(1));
    thread::sleep(Duration::from_millis(10));
    assert_eq!(lru.get_value("beta"), Some(2));
    thread::sleep(Duration::from_millis(10));
    assert_eq!(lru.get_value("gamma"), Some(3));

    // LRU tracking works
    let lru_term = lru.find_lru(&["alpha", "beta", "gamma"]);
    assert_eq!(lru_term, Some("alpha".to_string()));

    // Age tracking (through ttl.inner()) works
    let oldest = lru.inner().inner().find_oldest(&["alpha", "beta", "gamma"]);
    assert_eq!(oldest, Some("alpha".to_string()));

    // After TTL expires, all return None
    thread::sleep(Duration::from_secs(3));
    assert_eq!(lru.get_value("alpha"), None);
}

#[test]
fn test_dynamic_dawg_dynamic_updates_with_fuzzy_multimap() {
    // Verify that dynamic insertions/removals work with FuzzyMultiMap
    let dawg: DynamicDawg<HashSet<u32>> = DynamicDawg::new();
    dawg.insert_with_value("foo", HashSet::from([1]));
    dawg.insert_with_value("bar", HashSet::from([2]));

    let fuzzy = FuzzyMultiMap::new(dawg.clone(), Algorithm::Standard);

    // Initial query
    let result = fuzzy.query("foo", 0).unwrap();
    assert_eq!(result, HashSet::from([1]));

    // Add a new term dynamically
    dawg.insert_with_value("foe", HashSet::from([3]));

    // Query should now include the new term
    let result = fuzzy.query("foe", 1).unwrap();
    assert!(result.contains(&1)); // "foo" matches
    assert!(result.contains(&3)); // "foe" matches

    // Remove a term
    dawg.remove("bar");

    // Query should not find removed term
    let result = fuzzy.query("bar", 0);
    assert!(result.is_none());
}

#[test]
fn test_dynamic_dawg_value_updates() {
    // Test that updating values works correctly
    let dawg: DynamicDawg<u32> = DynamicDawg::new();

    // Insert initial value
    assert!(dawg.insert_with_value("key", 42));
    assert_eq!(dawg.get_value("key"), Some(42));

    // Update value (insert returns false for existing key)
    assert!(!dawg.insert_with_value("key", 99));
    assert_eq!(dawg.get_value("key"), Some(99));

    // Verify with eviction wrapper
    let lru = Lru::new(dawg);
    assert_eq!(lru.get_value("key"), Some(99));
}

#[test]
fn test_dynamic_dawg_fuzzy_multimap_overlapping_values() {
    // Test HashSet union with overlapping values
    let dawg: DynamicDawg<HashSet<u32>> = DynamicDawg::new();
    dawg.insert_with_value("foo", HashSet::from([1, 2]));
    dawg.insert_with_value("foe", HashSet::from([2, 3]));
    dawg.insert_with_value("fog", HashSet::from([3, 4]));

    let fuzzy = FuzzyMultiMap::new(dawg, Algorithm::Standard);

    // All three should match "fox" at distance 1
    let result = fuzzy.query("fox", 1).unwrap();
    // Should be union: {1, 2, 3, 4}
    assert_eq!(result, HashSet::from([1, 2, 3, 4]));
}

#[test]
fn test_dynamic_dawg_fuzzy_multimap_no_matches() {
    let dawg: DynamicDawg<Vec<i32>> = DynamicDawg::new();
    dawg.insert_with_value("hello", vec![1, 2, 3]);

    let fuzzy = FuzzyMultiMap::new(dawg, Algorithm::Standard);

    // Query with no matches
    let result = fuzzy.query("xyz", 1);
    assert!(result.is_none());
}

#[test]
fn test_dynamic_dawg_with_eviction_and_removal() {
    // Test eviction wrapper behavior after term removal
    let dawg: DynamicDawg<u32> = DynamicDawg::new();
    dawg.insert_with_value("keep", 1);
    dawg.insert_with_value("remove", 2);

    let lru = Lru::new(dawg.clone());

    // Access both
    assert_eq!(lru.get_value("keep"), Some(1));
    assert_eq!(lru.get_value("remove"), Some(2));

    // Remove from underlying dictionary
    dawg.remove("remove");

    // Should no longer be accessible
    assert_eq!(lru.get_value("remove"), None);

    // "keep" should still work
    assert_eq!(lru.get_value("keep"), Some(1));
}

#[test]
fn test_dynamic_dawg_fuzzy_multimap_with_transposition() {
    let dawg: DynamicDawg<HashSet<u32>> = DynamicDawg::new();
    dawg.insert_with_value("hello", HashSet::from([1]));
    dawg.insert_with_value("ehllo", HashSet::from([2])); // Transposition

    let fuzzy = FuzzyMultiMap::new(dawg, Algorithm::Transposition);

    // Query with transposition algorithm
    let result = fuzzy.query("hello", 2).unwrap();
    assert!(result.contains(&1)); // Exact match
}

#[test]
fn test_dynamic_dawg_contains_with_value_predicate() {
    // Test MappedDictionary::contains_with_value
    let dawg: DynamicDawg<u32> = DynamicDawg::new();
    dawg.insert_with_value("small", 5);
    dawg.insert_with_value("medium", 50);
    dawg.insert_with_value("large", 500);

    // Check for values > 100
    assert!(!dawg.contains_with_value("small", |v| *v > 100));
    assert!(!dawg.contains_with_value("medium", |v| *v > 100));
    assert!(dawg.contains_with_value("large", |v| *v > 100));

    // Check for even values
    assert!(!dawg.contains_with_value("small", |v| v % 2 == 0));
    assert!(dawg.contains_with_value("medium", |v| v % 2 == 0));
    assert!(dawg.contains_with_value("large", |v| v % 2 == 0));
}

#[test]
fn test_dynamic_dawg_empty_with_fuzzy_multimap() {
    // Test FuzzyMultiMap with empty DynamicDawg
    let dawg: DynamicDawg<Vec<u32>> = DynamicDawg::new();
    let fuzzy = FuzzyMultiMap::new(dawg, Algorithm::Standard);

    // Should return None for any query
    assert!(fuzzy.query("anything", 5).is_none());
}

#[test]
fn test_dynamic_dawg_with_contextual_completion_basic() {
    use liblevenshtein::contextual::DynamicContextualCompletionEngine;

    // Create a simple DynamicDawg-backed contextual completion engine
    let dict: DynamicDawg<Vec<u32>> = DynamicDawg::new();
    let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);

    let ctx = engine.create_root_context(0);

    // Add a term
    engine.finalize_direct(ctx, "test").unwrap();

    // Query should find it  (exact match, distance 0)
    let completions = engine.complete_finalized(ctx, "test", 0);
    assert!(
        completions.iter().any(|c| c.term == "test"),
        "Should find exact match 'test'"
    );

    // Fuzzy query should also find it
    let completions = engine.complete_finalized(ctx, "tst", 1); // Missing 'e'
    assert!(
        completions.iter().any(|c| c.term == "test"),
        "Should find 'test' with distance 1"
    );
}

#[test]
fn test_dynamic_dawg_with_contextual_completion_hierarchical() {
    use liblevenshtein::contextual::DynamicContextualCompletionEngine;

    let dict: DynamicDawg<Vec<u32>> = DynamicDawg::new();
    let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);

    // Create hierarchical contexts
    let global = engine.create_root_context(0);
    let function = engine.create_child_context(1, global).unwrap();

    // Add terms to different scopes
    engine.finalize_direct(global, "global_var").unwrap();
    engine.finalize_direct(function, "local_var").unwrap();

    // Query from function scope - should see both
    let completions = engine.complete_finalized(function, "var", 10);
    assert!(completions.iter().any(|c| c.term == "global_var"));
    assert!(completions.iter().any(|c| c.term == "local_var"));

    // Query from global scope - should NOT see local_var
    let completions = engine.complete_finalized(global, "var", 10);
    assert!(completions.iter().any(|c| c.term == "global_var"));
    assert!(!completions.iter().any(|c| c.term == "local_var"));
}

#[test]
fn test_dynamic_dawg_contextual_with_draft_debug() {
    use liblevenshtein::contextual::DynamicContextualCompletionEngine;

    let dict: DynamicDawg<Vec<u32>> = DynamicDawg::new();
    let dict_clone = dict.clone();
    let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);

    let ctx = engine.create_root_context(0);

    // Insert draft text
    engine.insert_str(ctx, "hello").unwrap();
    assert_eq!(engine.get_draft(ctx).unwrap(), "hello");

    println!("Draft before finalize: {:?}", engine.get_draft(ctx));

    // Finalize the draft
    let term = engine.finalize(ctx).unwrap();
    assert_eq!(term, "hello");

    println!("Term finalized: {}", term);

    // Check if the term is in the underlying dictionary
    let value = dict_clone.get_value("hello");
    println!("Value in dict for 'hello': {:?}", value);

    // Check if dict contains the term
    println!("Dict contains 'hello': {}", dict_clone.contains("hello"));

    // Try querying with complete_finalized
    let completions = engine.complete_finalized(ctx, "hello", 0);
    println!("Exact match completions: {:?}", completions);

    let completions = engine.complete_finalized(ctx, "hel", 2); // Need distance 2: "hel" -> "hello"
    println!(
        "Fuzzy completions for 'hel' (distance 2): {:?}",
        completions
    );

    // Should now be queryable
    assert!(
        completions.iter().any(|c| c.term == "hello"),
        "Should find 'hello' in completions"
    );
}

#[test]
fn test_dynamic_dawg_contextual_fuzzy_matching() {
    use liblevenshtein::contextual::DynamicContextualCompletionEngine;

    let dict: DynamicDawg<Vec<u32>> = DynamicDawg::new();
    let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Transposition);

    let ctx = engine.create_root_context(0);

    // Add identifiers
    engine.finalize_direct(ctx, "calculate").unwrap();
    engine.finalize_direct(ctx, "calibrate").unwrap();
    engine.finalize_direct(ctx, "calendar").unwrap();

    // Fuzzy match with distance 2
    let completions = engine.complete_finalized(ctx, "calulate", 2); // Missing 'c'
    assert!(completions.iter().any(|c| c.term == "calculate"));

    // Transposition should find these
    let completions = engine.complete_finalized(ctx, "calcualte", 2); // Transposed 'ua'
    assert!(completions.iter().any(|c| c.term == "calculate"));
}
