#![cfg(feature = "binding-integration-tests")]

use libdictenstein::bindings::{
    BindingUnitDomain, DoubleArrayTrieBinding, DynamicDawgBinding, ScdawgBinding,
};
use liblevenshtein::bindings::{
    Match, MatchBatch, MatchTerm, QueryCursor, QueryOrder, ResourceTransducer,
};
use liblevenshtein::transducer::Algorithm;
use proptest::prelude::*;
use std::collections::BTreeMap;

fn drain(cursor: &mut QueryCursor) -> Vec<Match> {
    let mut output = Vec::new();
    let mut batch = MatchBatch::default();
    loop {
        let written = cursor.next_batch(&mut batch, 3).unwrap();
        if written == 0 {
            return output;
        }
        output.extend_from_slice(batch.as_slice());
    }
}

fn map(matches: impl IntoIterator<Item = Match>) -> BTreeMap<String, Option<u64>> {
    matches
        .into_iter()
        .map(|item| match item.term {
            MatchTerm::Utf8(term) => (term, item.id),
            other => panic!("unexpected term domain: {other:?}"),
        })
        .collect()
}

#[test]
fn real_libdictenstein_resource_has_one_query_start_revision() {
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    for (term, value) in [("cat", 1), ("cot", 2), ("cut", 3), ("scat", 4)] {
        assert!(dictionary
            .insert_text(term.as_bytes(), Some(value))
            .unwrap());
    }
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let mut cursor = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let mut one = MatchBatch::default();
    assert_eq!(cursor.next_batch(&mut one, 1).unwrap(), 1);
    let first = one.as_slice()[0].clone();

    assert!(dictionary.remove_text(b"cot").unwrap());
    assert!(!dictionary.insert_text(b"cut", Some(30)).unwrap());
    assert!(dictionary.insert_text(b"cit", Some(5)).unwrap());
    dictionary.compact();
    dictionary.clear();
    assert!(dictionary.insert_text(b"new", Some(99)).unwrap());

    let fresh = map(drain(
        &mut transducer
            .query_utf8("cat", 8, QueryOrder::Traversal)
            .unwrap(),
    ));
    assert_eq!(fresh, BTreeMap::from([("new".to_owned(), Some(99))]));

    drop(transducer);
    drop(resource);
    drop(dictionary);
    let frozen = map(std::iter::once(first).chain(drain(&mut cursor)));
    assert_eq!(
        frozen,
        BTreeMap::from([
            ("cat".to_owned(), Some(1)),
            ("cot".to_owned(), Some(2)),
            ("cut".to_owned(), Some(3)),
            ("scat".to_owned(), Some(4)),
        ])
    );
}

#[test]
#[cfg(feature = "perf-instrumentation")]
fn distinct_snapshot_resources_share_identity_keyed_node_caches() {
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    for term in ["cat", "cot", "cut", "scat"] {
        dictionary
            .insert_text(term.as_bytes(), None)
            .expect("seed insert");
    }
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };

    let first = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let second = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    assert_eq!(first.snapshot_identity(), second.snapshot_identity());
    assert!(first.snapshot_identity().is_some());
    assert!(first.shares_node_cache_with(&second));

    dictionary.insert_text(b"cit", None).expect("mutate");
    let next = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let (producer, revision) = first.snapshot_identity().unwrap();
    let (next_producer, next_revision) = next.snapshot_identity().unwrap();
    assert_eq!(producer, next_producer);
    assert!(next_revision > revision);
    assert!(!first.shares_node_cache_with(&next));
}

#[test]
fn double_array_trie_resource_composes_without_serialization() {
    let dictionary = DoubleArrayTrieBinding::from_unicode_terms([
        ("café", Some(7)),
        ("caff", None),
        ("tea", Some(9)),
    ]);
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let observed = map(drain(
        &mut transducer
            .query_utf8("cafe", 2, QueryOrder::Traversal)
            .unwrap(),
    ));
    assert_eq!(
        observed,
        BTreeMap::from([("caff".to_owned(), None), ("café".to_owned(), Some(7)),])
    );
}

#[test]
fn scdawg_long_lived_cursor_keeps_its_published_revision() {
    let dictionary = ScdawgBinding::new_unicode();
    assert!(dictionary.insert("cat", Some(1)));
    assert!(dictionary.insert("cot", Some(2)));
    assert!(dictionary.insert("cut", None));
    let resource = dictionary.resource();
    let transducer = unsafe {
        ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
    };
    let mut cursor = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .unwrap();
    let mut one = MatchBatch::default();
    assert_eq!(cursor.next_batch(&mut one, 1).unwrap(), 1);
    let first = one.as_slice()[0].clone();

    assert!(dictionary.insert("cit", Some(4)));
    assert!(dictionary.contains_substring("it"));
    assert_eq!(dictionary.frequency("it"), 1);

    let frozen = map(std::iter::once(first).chain(drain(&mut cursor)));
    assert_eq!(
        frozen,
        BTreeMap::from([
            ("at".to_owned(), Some(1)),
            ("cat".to_owned(), Some(1)),
            ("cot".to_owned(), Some(2)),
            ("cut".to_owned(), None),
            ("ot".to_owned(), Some(2)),
            ("ut".to_owned(), None),
        ])
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn arbitrary_real_producer_mutations_do_not_change_a_partially_consumed_cursor(
        initial in prop::collection::vec(("[a-z]{1,8}", any::<u64>()), 2..40),
        mutations in prop::collection::vec((0u8..4, "[a-z]{1,8}", any::<u64>()), 0..40),
        prefix_seed in any::<usize>(),
    ) {
        let initial: BTreeMap<String, u64> = initial.into_iter().collect();
        prop_assume!(initial.len() >= 2);
        let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
        for (term, value) in &initial {
            dictionary.insert_text(term.as_bytes(), Some(*value)).unwrap();
        }
        let resource = dictionary.resource();
        let transducer = unsafe {
            ResourceTransducer::from_resource(resource.as_raw(), Algorithm::Standard).unwrap()
        };
        let mut cursor = transducer.query_utf8("", 16, QueryOrder::Traversal).unwrap();
        let prefix_len = 1 + prefix_seed % (initial.len() - 1);
        let mut prefix = Vec::new();
        let mut batch = MatchBatch::default();
        for _ in 0..prefix_len {
            prop_assert_eq!(cursor.next_batch(&mut batch, 1).unwrap(), 1);
            prefix.push(batch.as_slice()[0].clone());
        }

        for (operation, term, value) in mutations {
            match operation {
                0 | 1 => {
                    dictionary.insert_text(term.as_bytes(), Some(value)).unwrap();
                }
                2 => {
                    dictionary.remove_text(term.as_bytes()).unwrap();
                }
                _ => {
                    dictionary.compact();
                }
            }
        }

        let observed = map(prefix.into_iter().chain(drain(&mut cursor)));
        let expected = initial
            .into_iter()
            .map(|(term, value)| (term, Some(value)))
            .collect::<BTreeMap<_, _>>();
        prop_assert_eq!(observed, expected);
    }
}
