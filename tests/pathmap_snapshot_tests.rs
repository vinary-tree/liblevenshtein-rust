//! End-to-end `Transducer` queries over the zero-plumbing PathMap dictionaries.
//!
//! These tests are the central correctness proof of the TrieRef rework: a
//! `Transducer` must run not only over an owned [`PathMapSnapshot`] but over a
//! **borrowed** [`PathMapRef`] that holds only a `&PathMap` — demonstrating that
//! no hidden `'static` bound anywhere in the `Transducer` / `QueryIterator` /
//! intersection stack blocks the borrowed (MORK-style, zero-copy) variant.
#![cfg(feature = "pathmap-backend")]

use libdictenstein::pathmap::{
    PathMapDictionary, PathMapDictionaryChar, PathMapRef, PathMapRefChar, PathMapSnapshot,
};
use liblevenshtein::prelude::*;
use pathmap::PathMap;

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v.dedup();
    v
}

/// Build a raw `PathMap<()>` from terms — a bare byte-keyed trie, as a MORK
/// `Space.btm` would hand us.
fn raw_map(terms: &[&str]) -> PathMap<()> {
    let mut map = PathMap::new();
    for t in terms {
        map.insert(t.as_bytes(), ());
    }
    map
}

#[test]
fn transducer_over_owned_snapshot() {
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(vec!["foo", "food", "fool", "bar"]);
    let t = Transducer::new(dict.snapshot(), Algorithm::Standard);

    let got = sorted(t.query("fooo", 1).collect());
    assert_eq!(
        got,
        vec!["foo".to_string(), "food".to_string(), "fool".to_string()]
    );

    // Distance-bearing + distance-ordered variants exercise the same node path.
    let with_dist: Vec<(String, usize)> = t
        .query_with_distance("foo", 1)
        .map(|c| (c.term, c.distance))
        .collect();
    assert!(with_dist.contains(&("foo".to_string(), 0)));

    let ordered: Vec<usize> = t.query_ordered("fooo", 2).map(|c| c.distance).collect();
    assert!(
        ordered.windows(2).all(|w| w[0] <= w[1]),
        "query_ordered must be distance-first"
    );
}

/// THE key proof: a borrowed [`PathMapRef`] (holding only `&PathMap`) drives a
/// `Transducer` end-to-end. A hidden `'static` bound in the query stack would
/// make this fail to compile.
#[test]
fn transducer_over_borrowed_ref_no_static_bound() {
    let map = raw_map(&["foo", "food", "fool", "bar", "baz"]);
    let dict = PathMapRef::from_map(&map); // zero-copy borrow of the live map
    let t = Transducer::new(dict, Algorithm::Standard);

    let got = sorted(t.query("fooo", 1).collect());
    assert_eq!(
        got,
        vec!["foo".to_string(), "food".to_string(), "fool".to_string()]
    );

    // "bar" and "baz" are within edit distance 1 (r↔z).
    let near_bar = sorted(t.query("bar", 1).collect());
    assert!(near_bar.contains(&"bar".to_string()));
    assert!(near_bar.contains(&"baz".to_string()));

    // The borrow of `map` outlives the transducer and its queries.
    drop(map);
}

#[test]
fn transducer_over_subtrie_from_read_zippers() {
    let map = raw_map(&["foobar", "foobaz", "foo", "other"]);

    // Borrowed read zipper over the whole map.
    let rz = map.read_zipper();
    let borrowed = PathMapRef::from_read_zipper(&rz);
    let t = Transducer::new(borrowed, Algorithm::Standard);
    assert!(t.query("foo", 0).any(|s| s == "foo"));
    drop(rz);

    // Owned read zipper rooted at the "foo" prefix => an owned snapshot of that
    // subtrie; remaining terms are relative to "foo".
    let owned_rz = map.clone().into_read_zipper(b"foo");
    let sub = PathMapSnapshot::from_read_zipper(owned_rz);
    let t2 = Transducer::new(sub, Algorithm::Standard);
    let got = sorted(t2.query("bar", 1).collect());
    assert!(
        got.contains(&"bar".to_string()),
        "foobar's remainder 'bar' must be reachable in the foo-rooted subtrie"
    );
}

#[test]
fn snapshot_isolation_at_transducer_level() {
    let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(vec!["alpha"]);
    let t = Transducer::new(dict.snapshot(), Algorithm::Standard);

    // Mutate the dictionary AFTER the transducer captured its snapshot.
    dict.insert("alphb"); // edit distance 1 from "alpha"

    // The transducer queries the pre-mutation snapshot, so "alphb" is invisible.
    let got = sorted(t.query("alpha", 1).collect());
    assert_eq!(
        got,
        vec!["alpha".to_string()],
        "snapshot must not observe post-snapshot inserts"
    );
}

#[test]
fn char_snapshot_transducer_unicode() {
    let dict: PathMapDictionaryChar<()> =
        PathMapDictionaryChar::from_terms(vec!["café", "cafe", "cafés"]);
    let t = Transducer::new(dict.snapshot(), Algorithm::Standard);

    // Within one *character* edit of "café".
    let got = sorted(t.query("café", 1).collect());
    assert!(got.contains(&"café".to_string()));
    assert!(got.contains(&"cafe".to_string()), "é↔e is one char edit");
    assert!(
        got.contains(&"cafés".to_string()),
        "appending 's' is one char edit"
    );
}

#[test]
fn borrowed_char_ref_transducer() {
    let map = raw_map(&["中文", "中華"]);
    let dict = PathMapRefChar::from_map(&map);
    let t = Transducer::new(dict, Algorithm::Standard);

    let got = sorted(t.query("中文", 1).collect());
    assert!(got.contains(&"中文".to_string()));
    assert!(got.contains(&"中華".to_string()), "文↔華 is one char edit");
    drop(map);
}
