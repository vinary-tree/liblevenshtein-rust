//! Deliverable 1 — the `u64`-unit query surface.
//!
//! `T_gram` (word-level correction over term-id sequences) needs the Levenshtein
//! automaton to accept a `&[u64]` token sequence and return matched token sequences,
//! not byte-packed strings. These tests exercise `Transducer::query_units`,
//! `query_units_with_distance`, and `query_units_weighted` over a `u64` token
//! dictionary, and pin the regression that the string entry point (`query`) cannot
//! serve token sequences.

use libdictenstein::dynamic_dawg::DynamicDawgU64;
use libdictenstein::{MappedDictionary, MappedDictionaryNode}; // DictionaryNode via prelude
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::{OperationCostsF64, UnitCandidate, UnitCandidateF64};

/// Build a small token-id (`u64`) dictionary of n-gram-like sequences.
fn token_dict() -> DynamicDawgU64 {
    let dict = DynamicDawgU64::new();
    dict.insert_sequence(&[10, 20, 30]); // the "correct" trigram
    dict.insert_sequence(&[10, 20, 40]); // one substitution away (30 -> 40)
    dict.insert_sequence(&[10, 20, 30, 50]); // one insertion away (+ 50)
    dict.insert_sequence(&[10, 20]); // one deletion away (- 30)
    dict.insert_sequence(&[99, 98, 97]); // unrelated
    dict
}

fn sorted(mut v: Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    v.sort();
    v
}

#[test]
fn query_units_exact_match_is_lossless() {
    let t = Transducer::new(token_dict(), Algorithm::Standard);
    let hits: Vec<Vec<u64>> = t.query_units(&[10, 20, 30], 0).collect();
    // Lossless: the matched key comes back as the exact u64 sequence, no string round-trip.
    assert_eq!(hits, vec![vec![10, 20, 30]]);
}

#[test]
fn query_units_recovers_word_substitution_insertion_deletion() {
    let t = Transducer::new(token_dict(), Algorithm::Standard);
    let hits = sorted(t.query_units(&[10, 20, 30], 1).collect());
    // All four token-level edits at distance <= 1 of [10,20,30]:
    assert!(hits.contains(&vec![10, 20, 30]), "exact");
    assert!(hits.contains(&vec![10, 20, 40]), "substitution (30->40)");
    assert!(hits.contains(&vec![10, 20, 30, 50]), "insertion (+50)");
    assert!(hits.contains(&vec![10, 20]), "deletion (-30)");
    // The unrelated trigram is far away and must not appear.
    assert!(!hits.contains(&vec![99, 98, 97]));
}

#[test]
fn query_units_with_distance_reports_token_edit_distance() {
    let t = Transducer::new(token_dict(), Algorithm::Standard);
    let mut got: Vec<(Vec<u64>, usize)> = t
        .query_units_with_distance(&[10, 20, 30], 1)
        .map(|c: UnitCandidate<u64>| (c.term, c.distance))
        .collect();
    got.sort();
    assert!(got.contains(&(vec![10, 20, 30], 0)));
    assert!(got.contains(&(vec![10, 20, 40], 1)));
    assert!(got.contains(&(vec![10, 20, 30, 50], 1)));
    assert!(got.contains(&(vec![10, 20], 1)));
}

/// Regression: the `&str` entry point byte-packs the query via `CharUnit::from_str`
/// and reconstructs matches by byte-**un**packing the `u64` labels, so its `String`
/// results over a token dictionary are control-byte garbage (e.g. `"\n\0\0…\u{14}"`),
/// never a usable token sequence. This is exactly why `query_units` exists.
#[test]
fn string_query_is_lossy_but_query_units_is_lossless() {
    let t = Transducer::new(token_dict(), Algorithm::Standard);

    // Small token ids (< 256) byte-unpack to bytes with NUL padding, so every string
    // the `&str` path can produce here is NUL-laden garbage — not a real term.
    let via_str: Vec<String> = t.query("abc", 2).collect();
    assert!(
        via_str.iter().all(|s| s.contains('\0')),
        "string results over a u64 dict should be byte-unpacked garbage, got: {via_str:?}"
    );

    // query_units recovers the exact token sequence, losslessly — the whole point.
    let via_units: Vec<Vec<u64>> = t.query_units(&[10, 20, 30], 0).collect();
    assert_eq!(via_units, vec![vec![10, 20, 30]]);
}

#[test]
fn query_units_transposition_algorithm() {
    let dict: DynamicDawgU64 = DynamicDawgU64::new();
    dict.insert_sequence(&[1, 2, 3, 4]);
    let t = Transducer::new(dict, Algorithm::Transposition);
    // Adjacent token transposition (2<->3) is distance 1 under Transposition.
    let hits: Vec<Vec<u64>> = t.query_units(&[1, 3, 2, 4], 1).collect();
    assert_eq!(hits, vec![vec![1, 2, 3, 4]]);
}

/// Weighted (`f64`) `u64` search: a single search prunes on per-operation cost, so a
/// down-/up-weighted token edit ranks differently than plain unit-cost distance.
#[test]
fn query_units_weighted_prunes_on_operation_costs() {
    let t = Transducer::new(token_dict(), Algorithm::Standard);

    // Make substitutions expensive (2.0) but keep insertion cheap (1.0).
    let mut costs = OperationCostsF64::standard();
    costs.substitution = 2.0;

    let hits: Vec<(Vec<u64>, f64)> = t
        .query_units_weighted(&[10, 20, 30], 1.0, costs)
        .map(|c: UnitCandidateF64<u64>| (c.term, c.distance))
        .collect();
    let terms: Vec<&Vec<u64>> = hits.iter().map(|(t, _)| t).collect();

    // Within a 1.0 budget: exact (0.0) and the insertion (1.0) survive...
    assert!(terms.contains(&&vec![10, 20, 30]));
    assert!(terms.contains(&&vec![10, 20, 30, 50]));
    assert!(terms.contains(&&vec![10, 20])); // deletion, 1.0
                                             // ...but the substitution (now 2.0) is pruned, whereas plain unit-cost distance 1
                                             // would have included it.
    assert!(
        !terms.contains(&&vec![10, 20, 40]),
        "expensive substitution should be pruned by the weighted cost budget"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Deliverable 2 — value-returning term-ids, dictionary-agnostic.
//
// The value-yielding `u64` query surface is NOT coupled to any particular backend:
// it is generic over `D: MappedDictionary` whose node unit is `u64`. The helper
// below is written once against those trait bounds and then driven over two entirely
// independent 64-bit dictionaries with very different storage models — the in-memory,
// lock-free `DynamicDawgU64` and the disk-backed (WAL + in-memory-cache overlay)
// `PersistentARTrieU64`, exercised here in its path-less cache mode — proving the
// surface works for arbitrary 64-bit-node dictionaries, not a specific implementation.
// ────────────────────────────────────────────────────────────────────────────

/// Backend-agnostic assertion: the SAME `Transducer::query_units_values` recovers the
/// corrected token sequence *and* its stored value over any `u64` `MappedDictionary`.
///
/// The dictionary is pre-populated by the caller (construction is necessarily
/// backend-specific); everything below depends only on the trait bounds.
fn assert_value_roundtrip<D>(dict: D)
where
    D: MappedDictionary<Value = u64>,
    D::Node: MappedDictionaryNode<Value = u64> + DictionaryNode<Unit = u64>,
{
    let t = Transducer::new(dict, Algorithm::Standard);
    let mut got: Vec<(Vec<u64>, usize, u64)> = t.query_units_values(&[10, 20, 30], 1).collect();
    got.sort();

    // Exact match: lossless token sequence + its stored value, no string round-trip.
    assert!(
        got.contains(&(vec![10, 20, 30], 0, 100)),
        "exact match should yield (tokens, 0, value=100); got {got:?}"
    );
    // A one-token substitution within distance 1 also yields its own value.
    assert!(
        got.contains(&(vec![10, 20, 40], 1, 200)),
        "substitution should yield (tokens, 1, value=200); got {got:?}"
    );
}

#[test]
fn value_roundtrip_over_dynamic_dawg_u64() {
    let dict: DynamicDawgU64<u64> = DynamicDawgU64::new();
    dict.insert_sequence_with_value(&[10, 20, 30], 100);
    dict.insert_sequence_with_value(&[10, 20, 40], 200);
    assert_value_roundtrip(dict);
}

/// The identical generic helper over a completely different 64-bit dictionary — the
/// persistent adaptive-radix trie — demonstrating the surface is backend-agnostic.
#[cfg(feature = "persistent-artrie")]
#[test]
fn value_roundtrip_over_persistent_artrie_u64() {
    use libdictenstein::persistent_artrie::PersistentARTrieU64;

    let dict: PersistentARTrieU64<u64> = PersistentARTrieU64::new();
    dict.insert_sequence_with_value(&[10, 20, 30], 100);
    dict.insert_sequence_with_value(&[10, 20, 40], 200);
    assert_value_roundtrip(dict);
}
