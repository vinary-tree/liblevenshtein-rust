//! Safe binding-model laws for one partially consumed, long-lived cursor.

#![cfg(feature = "bindings-core")]

mod support;

use liblevenshtein::bindings::{
    Match, MatchBatch, QueryOrder, ResourceTransducer, DEFAULT_MATCH_BATCH,
};
use liblevenshtein::transducer::Algorithm;
use proptest::prelude::*;
use std::collections::BTreeMap;
use support::interop_dictionary::TestDictionary;

fn drain(cursor: &mut liblevenshtein::bindings::QueryCursor) -> Vec<Match> {
    let mut result = Vec::new();
    let mut batch = MatchBatch::default();
    loop {
        let count = cursor
            .next_batch(&mut batch, DEFAULT_MATCH_BATCH)
            .expect("provider query batch");
        if count == 0 {
            break;
        }
        result.extend_from_slice(batch.as_slice());
    }
    result.sort_by(|left, right| format!("{:?}", left.term).cmp(&format!("{:?}", right.term)));
    result
}

#[test]
fn query_start_snapshot_survives_every_crud_publication_and_owner_drop() {
    let dictionary = TestDictionary::new([
        ("cat".to_owned(), Some(1)),
        ("cot".to_owned(), Some(2)),
        ("cut".to_owned(), Some(3)),
        ("scat".to_owned(), None),
    ]);
    let transducer = unsafe {
        ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard)
            .expect("retain test dictionary")
    };
    let mut frozen_cursor = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .expect("frozen query");
    let frozen = drain(&mut frozen_cursor);

    let mut cursor = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .expect("long-lived query");
    let mut one = MatchBatch::default();
    assert_eq!(cursor.next_batch(&mut one, 1).unwrap(), 1);
    let prefix = one.as_slice().to_vec();

    dictionary.remove("cot");
    dictionary.update("cut", Some(30));
    dictionary.insert("cit", Some(5));
    dictionary.compact();
    dictionary.checkpoint();

    let mut observed = prefix;
    observed.extend(drain(&mut cursor));
    observed.sort_by(|left, right| format!("{:?}", left.term).cmp(&format!("{:?}", right.term)));
    assert_eq!(observed, frozen);

    let fresh = drain(
        &mut transducer
            .query_utf8("cat", 2, QueryOrder::Traversal)
            .expect("fresh query"),
    );
    assert_ne!(fresh, frozen, "a later query must observe the new revision");

    let mut outliving = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .expect("outliving query");
    drop(transducer);
    drop(dictionary);
    assert!(!drain(&mut outliving).is_empty());
}

#[test]
fn clear_after_partial_consumption_does_not_change_the_old_cursor() {
    let dictionary = TestDictionary::new(
        ["alpha", "alpine", "aleph", "beta"]
            .into_iter()
            .enumerate()
            .map(|(id, term)| (term.to_owned(), Some(id as u64))),
    );
    let transducer = unsafe {
        ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard).unwrap()
    };
    let mut expected = transducer
        .query_utf8("alpha", 8, QueryOrder::Traversal)
        .unwrap();
    let expected = drain(&mut expected);
    let mut cursor = transducer
        .query_utf8("alpha", 8, QueryOrder::Traversal)
        .unwrap();
    let mut first = MatchBatch::default();
    cursor.next_batch(&mut first, 1).unwrap();
    let mut observed = first.as_slice().to_vec();

    dictionary.clear();
    dictionary.insert("replacement", Some(99));
    observed.extend(drain(&mut cursor));
    observed.sort_by(|left, right| format!("{:?}", left.term).cmp(&format!("{:?}", right.term)));
    assert_eq!(observed, expected);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn arbitrary_mid_query_mutations_preserve_the_captured_provider_revision(
        initial in prop::collection::vec(("[a-z]{1,8}", proptest::option::of(any::<u64>())), 2..32),
        mutations in prop::collection::vec((0u8..5, "[a-z]{1,8}", proptest::option::of(any::<u64>())), 0..24),
        query in "[a-z]{0,8}",
        prefix_seed in any::<usize>(),
    ) {
        let initial: BTreeMap<_, _> = initial.into_iter().collect();
        prop_assume!(initial.len() >= 2);
        let dictionary = TestDictionary::new(initial.into_iter());
        let transducer = unsafe {
            ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard).unwrap()
        };
        let mut frozen_cursor = transducer.query_utf8(&query, 8, QueryOrder::Traversal).unwrap();
        let frozen = drain(&mut frozen_cursor);
        prop_assume!(frozen.len() >= 2);

        let mut cursor = transducer.query_utf8(&query, 8, QueryOrder::Traversal).unwrap();
        let prefix_len = 1 + prefix_seed % (frozen.len() - 1);
        let mut batch = MatchBatch::default();
        cursor.next_batch(&mut batch, prefix_len).unwrap();
        let mut observed = batch.as_slice().to_vec();

        for (operation, term, value) in mutations {
            match operation {
                0 => dictionary.insert(&term, value),
                1 => dictionary.remove(&term),
                2 => dictionary.update(&term, value),
                3 => dictionary.compact(),
                _ => dictionary.checkpoint(),
            }
        }
        observed.extend(drain(&mut cursor));
        observed.sort_by(|left, right| format!("{:?}", left.term).cmp(&format!("{:?}", right.term)));
        prop_assert_eq!(observed, frozen);
    }
}

#[test]
fn provider_edges_cross_the_abi_in_batches_not_per_edge() {
    let terms = (0..300u32).map(|index| {
        let first = char::from_u32(0x1000 + index).unwrap();
        (format!("{first}tail"), Some(index as u64))
    });
    let dictionary = TestDictionary::new(terms);
    let transducer = unsafe {
        ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard).unwrap()
    };
    let before = dictionary.edge_batch_calls();
    let mut cursor = transducer.query_utf8("", 0, QueryOrder::Traversal).unwrap();
    let _ = drain(&mut cursor);
    let calls = dictionary.edge_batch_calls() - before;
    assert!(calls < 300, "root edges must not use one callback per edge");
    assert_eq!(dictionary.snapshot_calls(), 1);
}
