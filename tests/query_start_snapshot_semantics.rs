//! Query-start snapshot semantics for one long-lived transducer iterator,
//! including the canonical cross-language fixture replayed at the NATIVE
//! layer (no interop boundary at all — the semantic floor the binding and
//! C-ABI replays must agree with).
//!
//! INVARIANT-HOOK: VT-SNAP-1 — emitted matches are a subset of the captured
//! revision (spec: docs/verification/abi/theories/CursorSnapshotSemantics.v).

mod support;

use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::dynamic_dawg::u64::DynamicDawgU64;
use liblevenshtein::transducer::{Algorithm, Transducer};
use proptest::prelude::*;
use std::collections::BTreeMap;
use support::query_start_fixture::{self, FixturePhase};

fn sorted_values(iter: impl Iterator<Item = (String, usize, u64)>) -> Vec<(String, usize, u64)> {
    let mut values: Vec<_> = iter.collect();
    values.sort();
    values
}

fn sorted_unit_values(
    iter: impl Iterator<Item = (Vec<u64>, usize, u64)>,
) -> Vec<(Vec<u64>, usize, u64)> {
    let mut values: Vec<_> = iter.collect();
    values.sort();
    values
}

/// The canonical cross-language fixture
/// (vinary-tree-interop/conformance/query_start_snapshot.tsv) replayed at
/// the native layer. Native dictionaries carry non-optional values, so the
/// fixture's empty id maps to the documented native sentinel 0; the
/// OPTIONAL-value fidelity of the same script is pinned at the binding and
/// C-ABI layers.
#[test]
fn canonical_fixture_replays_at_the_native_layer() {
    let steps = query_start_fixture::load();
    let initial: Vec<(String, u64)> = steps
        .iter()
        .filter(|step| step.phase == FixturePhase::Initial)
        .map(|step| {
            assert_eq!(step.operation, "insert", "initial phase is insert-only");
            (step.term.clone(), step.id.unwrap_or(0))
        })
        .collect();
    let dict = DynamicDawgChar::<u64>::from_terms_with_values(
        initial.iter().map(|(term, value)| (term.as_str(), *value)),
    );
    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

    let frozen = sorted_values(transducer.query_values("cat", 2));
    assert_eq!(
        frozen,
        vec![
            ("cat".to_owned(), 0, 1),
            ("cot".to_owned(), 1, 2),
            ("cut".to_owned(), 1, 3),
            ("scat".to_owned(), 1, 0),
        ],
        "the pinned frozen set (0 = native sentinel for the empty fixture id)"
    );

    let mut cursor = transducer.query_values("cat", 2);
    let first = cursor.next().expect("fixture query has initial matches");

    for step in steps
        .iter()
        .filter(|step| step.phase == FixturePhase::Mutation)
    {
        match step.operation.as_str() {
            "insert" | "update" => {
                dict.insert_with_value(&step.term, step.id.unwrap_or(0));
            }
            "remove" => {
                dict.remove(&step.term);
            }
            // The native DAWG has one structural maintenance operation; the
            // provider-level checkpoint is a publication no-op here.
            "compact" | "checkpoint" => {
                dict.compact();
            }
            other => panic!("fixture mutation {other:?} is not a native operation"),
        }
    }

    assert_eq!(
        sorted_values(std::iter::once(first).chain(cursor)),
        frozen,
        "the mid-mutation native cursor stays frozen"
    );
    assert_eq!(
        sorted_values(transducer.query_values("cat", 2)),
        vec![
            ("cat".to_owned(), 0, 1),
            ("cit".to_owned(), 1, 5),
            ("cut".to_owned(), 1, 30),
            ("scat".to_owned(), 1, 0),
        ],
        "the pinned post-mutation set"
    );
}

#[test]
fn long_lived_query_iterator_has_query_start_snapshot_semantics() {
    let dict = DynamicDawgChar::<u64>::from_terms_with_values([
        ("cat", 1),
        ("cot", 2),
        ("cut", 3),
        ("scat", 4),
    ]);
    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
    let frozen = sorted_values(transducer.query_values("cat", 2));

    let mut cursor = transducer.query_values("cat", 2);
    let first = cursor.next().expect("query has multiple initial matches");

    dict.remove("cot");
    dict.insert_with_value("cit", 5);
    dict.insert_with_value("cut", 30);
    dict.compact();

    assert_eq!(
        sorted_values(std::iter::once(first).chain(cursor)),
        frozen,
        "the same iterator must retain terms, distances, and start-of-query values"
    );
    assert_eq!(
        sorted_values(transducer.query_values("cat", 2)),
        vec![
            ("cat".to_owned(), 0, 1),
            ("cit".to_owned(), 1, 5),
            ("cut".to_owned(), 1, 30),
            ("scat".to_owned(), 1, 4),
        ]
    );
}

#[test]
fn long_lived_ordered_iterator_keeps_its_exact_initial_sequence() {
    let dict = DynamicDawgChar::<u64>::from_terms_with_values([
        ("book", 1),
        ("books", 2),
        ("boon", 3),
        ("cook", 4),
    ]);
    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
    let frozen: Vec<_> = transducer.query_ordered("book", 2).collect();
    let mut cursor = transducer.query_ordered("book", 2);
    let first = cursor.next().expect("query has initial matches");

    dict.remove("books");
    dict.insert_with_value("brook", 5);
    dict.compact();

    let observed: Vec<_> = std::iter::once(first).chain(cursor).collect();
    assert_eq!(observed, frozen);
    assert_ne!(
        transducer.query_ordered("book", 2).collect::<Vec<_>>(),
        frozen,
        "a new iterator must use the newly published root"
    );
}

#[test]
fn long_lived_u64_query_iterator_keeps_its_sequence_and_values() {
    let dict = DynamicDawgU64::<u64>::new();
    dict.insert_sequence_with_value(&[10, 20, 30], 1);
    dict.insert_sequence_with_value(&[10, 21, 30], 2);
    dict.insert_sequence_with_value(&[11, 20, 30], 3);
    dict.insert_sequence_with_value(&[10, 20, 30, 40], 4);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let frozen = sorted_unit_values(transducer.query_units_values(&[10, 20, 30], 2));
    let mut cursor = transducer.query_units_values(&[10, 20, 30], 2);
    let first = cursor.next().expect("query has initial matches");

    transducer.dictionary().remove_sequence(&[10, 21, 30]);
    transducer
        .dictionary()
        .insert_sequence_with_value(&[10, 22, 30], 5);
    transducer
        .dictionary()
        .insert_sequence_with_value(&[10, 20, 30], 100);
    transducer.dictionary().compact();

    assert_eq!(
        sorted_unit_values(std::iter::once(first).chain(cursor)),
        frozen,
        "u64 terms and values must come from the root captured at query start"
    );
    assert_ne!(
        sorted_unit_values(transducer.query_units_values(&[10, 20, 30], 2)),
        frozen,
        "a fresh u64 query must observe the new revision"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn arbitrary_mid_query_mutations_preserve_the_original_revision(
        initial in prop::collection::vec(("[a-z]{1,8}", any::<u64>()), 2..40),
        mutations in prop::collection::vec((0u8..3, "[a-z]{1,8}", any::<u64>()), 0..32),
        query in "[a-z]{0,8}",
        algorithm_index in 0usize..4,
        prefix_seed in any::<usize>(),
    ) {
        let initial: BTreeMap<String, u64> = initial.into_iter().collect();
        prop_assume!(initial.len() >= 2);
        let algorithm = [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
            Algorithm::DamerauLevenshtein,
        ][algorithm_index];
        let dict = DynamicDawgChar::<u64>::from_terms_with_values(
            initial.iter().map(|(term, value)| (term.as_str(), *value)),
        );
        let transducer = Transducer::new(dict.clone(), algorithm);

        // Eight edits cover every generated query/term pair, ensuring that the
        // cursor is long-lived and has an unconsumed suffix after its prefix.
        let frozen = sorted_values(transducer.query_values(&query, 8));
        prop_assert!(frozen.len() >= 2);
        let mut cursor = transducer.query_values(&query, 8);
        let prefix_len = 1 + prefix_seed % (frozen.len() - 1);
        let prefix: Vec<_> = cursor.by_ref().take(prefix_len).collect();

        for (operation, term, value) in mutations {
            match operation {
                0 | 2 => { dict.insert_with_value(&term, value); }
                _ => { dict.remove(&term); }
            }
        }
        dict.compact();

        prop_assert_eq!(
            sorted_values(prefix.into_iter().chain(cursor)),
            frozen,
        );
    }

    #[test]
    fn arbitrary_ordered_cursor_retains_the_initial_order(
        initial in prop::collection::vec("[a-z]{1,8}", 2..36),
        inserted in "[a-z]{1,8}",
        removed in "[a-z]{1,8}",
        query in "[a-z]{0,8}",
        prefix_seed in any::<usize>(),
    ) {
        let initial: BTreeMap<String, u64> = initial
            .into_iter()
            .enumerate()
            .map(|(index, term)| (term, index as u64))
            .collect();
        prop_assume!(initial.len() >= 2);
        let dict = DynamicDawgChar::<u64>::from_terms_with_values(
            initial.iter().map(|(term, value)| (term.as_str(), *value)),
        );
        let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
        let frozen: Vec<_> = transducer.query_ordered(&query, 8).collect();
        prop_assert!(frozen.len() >= 2);
        let mut cursor = transducer.query_ordered(&query, 8);
        let prefix_len = 1 + prefix_seed % (frozen.len() - 1);
        let prefix: Vec<_> = cursor.by_ref().take(prefix_len).collect();

        dict.insert_with_value(&inserted, u64::MAX);
        dict.remove(&removed);
        dict.compact();

        let observed: Vec<_> = prefix.into_iter().chain(cursor).collect();
        prop_assert_eq!(observed, frozen);
    }

    #[test]
    fn arbitrary_u64_mutations_preserve_one_long_lived_query(
        initial in prop::collection::vec((prop::collection::vec(0u64..32, 1..7), any::<u64>()), 2..32),
        mutations in prop::collection::vec((0u8..3, prop::collection::vec(0u64..32, 1..7), any::<u64>()), 0..28),
        query in prop::collection::vec(0u64..32, 0..7),
        prefix_seed in any::<usize>(),
    ) {
        let initial: BTreeMap<Vec<u64>, u64> = initial.into_iter().collect();
        prop_assume!(initial.len() >= 2);
        let dict = DynamicDawgU64::<u64>::new();
        for (term, value) in &initial {
            dict.insert_sequence_with_value(term, *value);
        }
        let transducer = Transducer::new(dict, Algorithm::Standard);
        let frozen = sorted_unit_values(transducer.query_units_values(&query, 8));
        prop_assert!(frozen.len() >= 2);
        let mut cursor = transducer.query_units_values(&query, 8);
        let prefix_len = 1 + prefix_seed % (frozen.len() - 1);
        let prefix: Vec<_> = cursor.by_ref().take(prefix_len).collect();

        for (operation, term, value) in mutations {
            match operation {
                0 | 2 => { transducer.dictionary().insert_sequence_with_value(&term, value); }
                _ => { transducer.dictionary().remove_sequence(&term); }
            }
        }
        transducer.dictionary().compact();

        prop_assert_eq!(
            sorted_unit_values(prefix.into_iter().chain(cursor)),
            frozen,
        );
    }
}
