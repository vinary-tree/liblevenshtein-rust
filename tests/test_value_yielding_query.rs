//! Integration tests for `Transducer::query_values` (G9) across dictionary backends.
//!
//! `query_values` is generic over any `MappedDictionary`, so it must behave
//! identically on every backend and must agree with the headline contract:
//! "`query_values` == `query_with_distance` paired with `get_value`, with no
//! second lookup." These tests exercise the byte-level `DoubleArrayTrie` and
//! `DynamicDawg`, and (under `pathmap-backend`) the char-level Unicode backend.

use liblevenshtein::prelude::*;
use std::collections::HashSet;

/// Both byte-level backends must agree on the full `(term, distance, value)`
/// result, and each value must equal a fresh `get_value` lookup.
#[test]
fn test_query_values_multibackend_parity() {
    let entries = vec![
        ("apple", 10usize),
        ("apply", 11),
        ("apples", 12),
        ("ape", 13),
        ("maple", 14),
    ];

    let dat_t = Transducer::new(
        DoubleArrayTrie::<usize>::from_terms_with_values(entries.clone()),
        Algorithm::Standard,
    );
    let dawg_t = Transducer::new(
        DynamicDawg::<usize>::from_terms_with_values(entries.clone()),
        Algorithm::Standard,
    );

    let mut dat_results: Vec<(String, usize, usize)> = dat_t.query_values("apple", 2).collect();
    let mut dawg_results: Vec<(String, usize, usize)> = dawg_t.query_values("apple", 2).collect();
    dat_results.sort();
    dawg_results.sort();

    assert_eq!(
        dat_results, dawg_results,
        "DoubleArrayTrie and DynamicDawg disagree on query_values"
    );
    assert!(!dat_results.is_empty(), "expected at least the exact match");

    // Every yielded value matches a fresh dictionary lookup on BOTH backends.
    for (term, _d, value) in &dat_results {
        assert_eq!(Some(*value), dat_t.dictionary().get_value(term));
        assert_eq!(Some(*value), dawg_t.dictionary().get_value(term));
    }
}

/// The headline contract: `query_values` yields exactly the `(term, distance)`
/// pairs of `query_with_distance`, and the bundled value equals `get_value`.
#[test]
fn test_query_values_matches_query_plus_get_value() {
    let entries = vec![
        ("test", 1usize),
        ("tests", 2),
        ("tester", 3),
        ("testing", 4),
        ("best", 5),
    ];
    let transducer = Transducer::new(
        DoubleArrayTrie::<usize>::from_terms_with_values(entries),
        Algorithm::Standard,
    );

    let from_values: HashSet<(String, usize)> = transducer
        .query_values("test", 2)
        .map(|(t, d, _)| (t, d))
        .collect();
    let from_query: HashSet<(String, usize)> = transducer
        .query_with_distance("test", 2)
        .map(|c| (c.term, c.distance))
        .collect();
    assert_eq!(from_values, from_query, "query_values set != query set");

    for (term, _d, value) in transducer.query_values("test", 2) {
        assert_eq!(
            Some(value),
            transducer.dictionary().get_value(&term),
            "bundled value must equal a fresh lookup for {term:?}"
        );
    }
}

/// Char-level (Unicode) backend: values are read correctly for multi-byte chars.
#[cfg(feature = "pathmap-backend")]
#[test]
fn test_query_values_char_level_unicode() {
    use libdictenstein::pathmap_char::PathMapDictionaryChar;

    let dict: PathMapDictionaryChar<u32> = PathMapDictionaryChar::from_terms_with_values(vec![
        ("café", 1u32),
        ("cafe", 1),
        ("中文", 2),
        ("汉字", 2),
    ]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut results: Vec<(String, usize, u32)> = transducer.query_values("cafe", 1).collect();
    results.sort();

    // "cafe" (distance 0) and "café" (distance 1, é↔e); the CJK terms are far away.
    assert!(
        results
            .iter()
            .any(|(t, d, v)| t == "cafe" && *d == 0 && *v == 1),
        "expected (cafe, 0, 1) in {results:?}"
    );
    assert!(
        results
            .iter()
            .any(|(t, d, v)| t == "café" && *d == 1 && *v == 1),
        "expected (café, 1, 1) in {results:?}"
    );
    assert!(
        !results.iter().any(|(t, _, _)| t == "中文" || t == "汉字"),
        "CJK terms must be out of range: {results:?}"
    );

    for (term, _d, value) in &results {
        assert_eq!(Some(*value), transducer.dictionary().get_value(term));
    }
}
