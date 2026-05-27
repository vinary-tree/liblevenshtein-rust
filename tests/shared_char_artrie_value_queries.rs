#![cfg(feature = "persistent-artrie")]
//! G9 end-to-end: the value-aware transducer API over the persistent char trie.
//!
//! libdictenstein commit f10c43e added `impl MappedDictionaryNode for
//! PersistentARTrieCharNode<V>` (`value() -> Option<V>`). That impl is exactly
//! what lets `Transducer::query_values` / `query_filtered` / `query_by_value_set`
//! operate over `SharedCharARTrie<V>`: those methods live in the
//! `impl<D, P> Transducer<D, P> where D: MappedDictionary,
//! D::Node: MappedDictionaryNode<Value = D::Value>, ...` block, and for the
//! shared char trie `D::Node` is `PersistentARTrieCharNode<V>`. Before f10c43e
//! this file would not compile (the trait bound was unsatisfied); these tests
//! exercise the unlocked queries over Unicode keys.

use libdictenstein::artrie_trait::ARTrie;
use libdictenstein::persistent_artrie_char::SharedCharARTrie;
use libdictenstein::{MappedDictionary, MutableMappedDictionary};
use liblevenshtein::prelude::*;
use std::collections::HashSet;
use tempfile::{tempdir, TempDir};

/// Build a disk-backed shared char trie with the given key/value pairs.
fn build(dir: &TempDir, entries: &[(&str, i64)]) -> SharedCharARTrie<i64> {
    let trie: SharedCharARTrie<i64> =
        ARTrie::create(dir.path().join("q.trie")).expect("create char trie");
    for (term, value) in entries {
        assert!(
            MutableMappedDictionary::insert_with_value(&trie, term, *value),
            "insert {term:?}"
        );
    }
    trie
}

/// `query_values` yields `(term, distance, value)` reading each value during
/// traversal via `MappedDictionaryNode::value()`; every yielded value must agree
/// with a fresh `get_value` lookup, and the exact (Unicode) match is present.
#[test]
fn query_values_over_shared_char_artrie() {
    let dir = tempdir().expect("tempdir");
    let entries = [
        ("café", 1i64),
        ("cafe", 2),
        ("naïve", 3),
        ("caffe", 4),
        ("apple", 5),
    ];
    let transducer = Transducer::new(build(&dir, &entries), Algorithm::Standard);

    let results: Vec<(String, usize, i64)> = transducer.query_values("café", 1).collect();
    assert!(!results.is_empty(), "expected at least the exact match");

    // Each bundled value equals a fresh dictionary lookup (no second lookup was
    // needed — `value()` was read during traversal).
    for (term, _distance, value) in &results {
        assert_eq!(
            MappedDictionary::get_value(transducer.dictionary(), term),
            Some(*value),
            "query_values value for {term:?} disagrees with get_value"
        );
    }

    // The exact Unicode match is present at distance 0 with its value.
    assert!(
        results
            .iter()
            .any(|(t, d, v)| t == "café" && *d == 0 && *v == 1),
        "exact match café/0/1 missing from {results:?}"
    );
    // The single-substitution neighbour (é -> e) is found with its own value.
    assert!(
        results
            .iter()
            .any(|(t, d, v)| t == "cafe" && *d == 1 && *v == 2),
        "neighbour cafe/1/2 missing from {results:?}"
    );
}

/// `query_filtered` prunes by a value predicate during traversal: only matches
/// whose value satisfies the predicate are yielded.
#[test]
fn query_filtered_over_shared_char_artrie() {
    let dir = tempdir().expect("tempdir");
    let entries = [
        ("apple", 10i64),
        ("apply", 20),
        ("apples", 10),
        ("maple", 30),
    ];
    let transducer = Transducer::new(build(&dir, &entries), Algorithm::Standard);

    let terms: HashSet<String> = transducer
        .query_filtered("apple", 2, |v: &i64| *v == 10)
        .map(|c: Candidate| c.term)
        .collect();

    assert!(terms.contains("apple"), "apple (value 10) should pass");
    assert!(terms.contains("apples"), "apples (value 10) should pass");
    assert!(
        !terms.contains("apply"),
        "apply (value 20) is within distance 2 but must be filtered out: {terms:?}"
    );
}

/// `query_by_value_set` keeps only matches whose value is in the given set —
/// the hierarchical-scope use case the API was built for.
#[test]
fn query_by_value_set_over_shared_char_artrie() {
    let dir = tempdir().expect("tempdir");
    let entries = [
        ("println", 1i64),
        ("printf", 2),
        ("print", 1),
        ("sprint", 3),
    ];
    let transducer = Transducer::new(build(&dir, &entries), Algorithm::Standard);

    let allowed: HashSet<i64> = [1].into_iter().collect();
    let terms: HashSet<String> = transducer
        .query_by_value_set("print", 2, &allowed)
        .map(|c: Candidate| c.term)
        .collect();

    assert!(
        terms.contains("print"),
        "print (value 1) should be included"
    );
    assert!(
        terms.contains("println"),
        "println (value 1, distance 2) should be included: {terms:?}"
    );
    assert!(
        !terms.contains("printf"),
        "printf (value 2) is not in the value set: {terms:?}"
    );
    assert!(
        !terms.contains("sprint"),
        "sprint (value 3) is not in the value set: {terms:?}"
    );
}
