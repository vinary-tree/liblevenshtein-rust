//! Deliverables 2 (phonetic value-return) and 3 (full articulatory cost).
//!
//! These tests pin the two phonetic-path upstream-contract items:
//!
//! * **D3 — articulatory cost surfacing.** A phonetic query built with
//!   [`PhoneticTransducerChar::with_articulatory_costs`] scores each candidate by
//!   its *articulatory-weighted* alignment cost (`ProductAutomatonChar::min_cost`),
//!   so a sound-alike substitution (e.g. the voiced/voiceless pair `p`↔`b`) costs
//!   a fraction of a full edit and ranks ahead of a phonetically distant one at
//!   the same integer edit distance. `total_cost = edit_distance + phonetic_cost`,
//!   with `phonetic_cost ≤ 0` the articulatory discount (`0.0` for an exact match
//!   and for the default, non-articulatory path).
//!
//! * **D2 — value-returning term-ids.** `query_values` yields the dictionary's
//!   stored value (e.g. a term-id) at each matched term, so a lexical corrector
//!   can emit `(term_id, cost)` with no string round-trip. Exercised over both a
//!   character dictionary and a byte dictionary.
//!
//! A regression test also pins the traversal fix: terms that *extend* a matched
//! prefix (e.g. "phones" past a final "phone") must not be dropped.

#![cfg(feature = "phonetic-rules")]

use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::phonetic::nfa::{compile, compile_bytes};
use liblevenshtein::phonetic::regex::{parse, parse_bytes};
use liblevenshtein::transducer::{ArticulatoryCosts, PhoneticTransducer, PhoneticTransducerChar};

/// Compile a character-level phonetic NFA from a regex pattern.
fn nfa_char(pattern: &str) -> liblevenshtein::phonetic::nfa::NFAChar {
    compile(&parse(pattern).expect("pattern parses")).expect("pattern compiles")
}

// ────────────────────────────────────────────────────────────────────────────
// D3 — articulatory cost surfacing (character level)
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn articulatory_exact_match_has_zero_phonetic_cost() {
    let dict = DoubleArrayTrieChar::from_terms(["pat"]);
    let t = PhoneticTransducerChar::with_articulatory_costs(
        dict,
        nfa_char("pat"),
        1,
        ArticulatoryCosts::default(),
    );
    let hit = t
        .query("pat")
        .find(|c| c.term == "pat")
        .expect("exact term present");
    assert_eq!(hit.edit_distance, 0);
    assert_eq!(hit.phonetic_cost, 0.0, "exact match earns no discount");
    assert_eq!(hit.total_cost, 0.0, "total_cost == edit_distance == 0");
}

#[test]
fn articulatory_soundalike_substitution_is_discounted_below_a_full_edit() {
    // Pattern "pat"; "bat" is one substitution away (p→b, a voiced/voiceless
    // pair), which is articulatorily near and so should cost far less than 1.0.
    let dict = DoubleArrayTrieChar::from_terms(["pat", "bat"]);
    let t = PhoneticTransducerChar::with_articulatory_costs(
        dict,
        nfa_char("pat"),
        1,
        ArticulatoryCosts::default(),
    );
    let bat = t
        .query("pat")
        .find(|c| c.term == "bat")
        .expect("bat within distance 1");

    assert_eq!(bat.edit_distance, 1, "bat is one integer substitution away");
    assert!(
        bat.total_cost < 1.0,
        "a sound-alike substitution must cost less than a full edit; got {}",
        bat.total_cost
    );
    assert!(
        bat.phonetic_cost < 0.0,
        "phonetic_cost is the (negative) articulatory discount; got {}",
        bat.phonetic_cost
    );
    // The candidate constructor's invariant: total = edit + phonetic.
    assert!(
        (bat.total_cost - (f64::from(bat.edit_distance) + bat.phonetic_cost)).abs() < 1e-9,
        "total_cost must equal edit_distance + phonetic_cost"
    );
}

#[test]
fn articulatory_ranks_soundalike_ahead_of_phonetically_distant() {
    // Both "bat" (p→b, near) and "cat" (p→c, far) are edit distance 1, but the
    // articulatory cost must rank "bat" strictly cheaper than "cat".
    let dict = DoubleArrayTrieChar::from_terms(["pat", "bat", "cat"]);
    let t = PhoneticTransducerChar::with_articulatory_costs(
        dict,
        nfa_char("pat"),
        1,
        ArticulatoryCosts::default(),
    );
    let sorted = t.query_sorted("pat");
    let order: Vec<&str> = sorted.iter().map(|c| c.term.as_str()).collect();

    // Exact match first, then the near substitution, then the far one.
    assert_eq!(
        order.first().copied(),
        Some("pat"),
        "exact match ranks first; got {order:?}"
    );
    let pos_bat = order.iter().position(|&s| s == "bat").expect("bat present");
    let pos_cat = order.iter().position(|&s| s == "cat").expect("cat present");
    assert!(
        pos_bat < pos_cat,
        "sound-alike 'bat' must outrank distant 'cat'; order {order:?}"
    );

    let bat = sorted.iter().find(|c| c.term == "bat").unwrap();
    let cat = sorted.iter().find(|c| c.term == "cat").unwrap();
    assert!(
        bat.total_cost < cat.total_cost,
        "bat ({}) must be strictly cheaper than cat ({})",
        bat.total_cost,
        cat.total_cost
    );
    // Both are one edit; the *cost* differs only in the articulatory component.
    assert_eq!(bat.edit_distance, 1);
    assert_eq!(cat.edit_distance, 1);
}

#[test]
fn default_path_reports_zero_phonetic_cost() {
    // Without articulatory costs, the query is a pure integer edit-distance
    // search: phonetic_cost is 0 and total_cost == edit_distance for every hit.
    let dict = DoubleArrayTrieChar::from_terms(["pat", "bat", "cat"]);
    let t = PhoneticTransducerChar::new(dict, nfa_char("pat"), 1);
    for c in t.query("pat") {
        assert_eq!(
            c.phonetic_cost, 0.0,
            "default path has zero phonetic cost for {}",
            c.term
        );
        assert_eq!(
            c.total_cost,
            f64::from(c.edit_distance),
            "default total_cost == edit_distance for {}",
            c.term
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// D2 — value-returning phonetic queries (character level)
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn query_values_char_yields_stored_term_id() {
    // A vocabulary trie: term → term-id. The phonetic query must return the
    // stored id at each match, with no string round-trip.
    let dict = DoubleArrayTrieChar::from_terms_with_values([("phone", 100u64), ("phones", 200u64)]);
    let t = PhoneticTransducerChar::new(dict, nfa_char("phone"), 1);

    let mut got: Vec<(String, u8, u64)> = t
        .query_values("phone")
        .map(|c| (c.term, c.edit_distance, c.value))
        .collect();
    got.sort();

    assert!(
        got.contains(&("phone".to_string(), 0, 100)),
        "exact match yields (phone, 0, id=100); got {got:?}"
    );
    assert!(
        got.contains(&("phones".to_string(), 1, 200)),
        "insertion match yields (phones, 1, id=200); got {got:?}"
    );
}

#[test]
fn query_values_char_combines_articulatory_cost_and_value() {
    // Value-return AND articulatory weighting together: T_lex's exact need,
    // `(term_id, total_cost)` where the cost is articulatory-weighted.
    let dict = DoubleArrayTrieChar::from_terms_with_values([("pat", 1u64), ("bat", 2u64)]);
    let t = PhoneticTransducerChar::with_articulatory_costs(
        dict,
        nfa_char("pat"),
        1,
        ArticulatoryCosts::default(),
    );
    let bat = t
        .query_values("pat")
        .find(|c| c.term == "bat")
        .expect("bat present");

    assert_eq!(bat.value, 2, "carries bat's stored term-id");
    assert_eq!(bat.edit_distance, 1);
    assert!(
        bat.total_cost < 1.0 && bat.phonetic_cost < 0.0,
        "articulatory discount applies on the value path too: total={}, phon={}",
        bat.total_cost,
        bat.phonetic_cost
    );
}

#[test]
fn query_values_sorted_ranks_by_total_cost() {
    let dict =
        DoubleArrayTrieChar::from_terms_with_values([("pat", 1u64), ("bat", 2u64), ("cat", 3u64)]);
    let t = PhoneticTransducerChar::with_articulatory_costs(
        dict,
        nfa_char("pat"),
        1,
        ArticulatoryCosts::default(),
    );
    let sorted = t.query_values_sorted("pat");
    // Non-decreasing total_cost.
    for w in sorted.windows(2) {
        assert!(
            w[0].total_cost <= w[1].total_cost,
            "results must be sorted by total_cost: {} then {}",
            w[0].total_cost,
            w[1].total_cost
        );
    }
    assert_eq!(sorted.first().map(|c| c.term.as_str()), Some("pat"));
}

// ────────────────────────────────────────────────────────────────────────────
// D2 — value-returning phonetic queries (byte level)
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn query_values_byte_yields_stored_term_id() {
    let dict = DoubleArrayTrie::from_terms_with_values([("phone", 100u64), ("phones", 200u64)]);
    let nfa = compile_bytes(&parse_bytes(b"phone").expect("parse")).expect("compile");
    let t = PhoneticTransducer::new(dict, nfa, 1);

    let mut got: Vec<(Vec<u8>, u8, u64)> = t
        .query_values(b"phone")
        .map(|c| (c.term, c.edit_distance, c.value))
        .collect();
    got.sort();

    assert!(
        got.contains(&(b"phone".to_vec(), 0, 100)),
        "byte exact match yields (phone, 0, 100); got {got:?}"
    );
    assert!(
        got.contains(&(b"phones".to_vec(), 1, 200)),
        "byte insertion yields (phones, 1, 200); got {got:?}"
    );
    // The byte path is integer-only: phonetic_cost is always 0.
    for c in t.query_values(b"phone") {
        assert_eq!(c.phonetic_cost, 0.0, "byte phonetic_cost is structurally 0");
        assert_eq!(c.total_cost, f64::from(c.edit_distance));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Regression — a matched prefix's extensions must not be dropped
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn extensions_of_a_matched_prefix_are_not_dropped() {
    // "phone" is a final node AND an internal node (prefix of phones/phoned).
    // The traversal must explore its children, or those distance-1 matches are
    // silently lost.
    let dict = DoubleArrayTrieChar::from_terms(["phone", "phones", "phoned", "phoning"]);
    let t = PhoneticTransducerChar::new(dict, nfa_char("phone"), 2);
    let mut terms: Vec<String> = t.query("phone").map(|c| c.term).collect();
    terms.sort();

    assert!(terms.contains(&"phone".to_string()), "exact");
    assert!(
        terms.contains(&"phones".to_string()),
        "phones (dist 1) — an extension of the matched prefix — must be found; got {terms:?}"
    );
    assert!(
        terms.contains(&"phoned".to_string()),
        "phoned (dist 1) must be found; got {terms:?}"
    );
}

#[test]
fn value_query_extensions_of_a_matched_prefix_are_not_dropped() {
    // Same regression on the value-returning path.
    let dict = DoubleArrayTrieChar::from_terms_with_values([
        ("phone", 1u64),
        ("phones", 2u64),
        ("phoned", 3u64),
    ]);
    let t = PhoneticTransducerChar::new(dict, nfa_char("phone"), 2);
    let mut got: Vec<(String, u64)> = t.query_values("phone").map(|c| (c.term, c.value)).collect();
    got.sort();

    assert!(got.contains(&("phone".to_string(), 1)), "exact value");
    assert!(
        got.contains(&("phones".to_string(), 2)),
        "phones value must be reachable past the matched prefix; got {got:?}"
    );
}
