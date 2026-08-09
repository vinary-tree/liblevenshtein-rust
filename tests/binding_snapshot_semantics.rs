//! Safe binding-model laws for one partially consumed, long-lived cursor,
//! the canonical cross-language fixture replayed at this layer, and the
//! pinned cursor fault-channel semantics.
//!
//! INVARIANT-HOOK: VT-SNAP-2 — a fault-free drain yields exactly the captured
//! revision (spec: docs/verification/abi/theories/CursorSnapshotSemantics.v).
//! INVARIANT-HOOK: LLEV-CUR-1 — the cursor fault channel is take-once with
//! first-fault-wins and NO permanent latch: one recorded provider fault
//! surfaces as exactly one error, the pull that observed it is discarded,
//! and the cursor then resumes (or ends) cleanly
//! (`provider_fault_is_taken_once_then_the_cursor_resumes`,
//! `fault_on_the_final_match_surfaces_after_the_delivered_prefix`,
//! `first_recorded_fault_wins_when_multiple_callbacks_fail`; C-ABI face in
//! tests/ffi_provider_fault_injection.rs).

#![cfg(feature = "bindings-core")]

mod support;

use liblevenshtein::bindings::{
    BindingError, Match, MatchBatch, MatchTerm, QueryOrder, ResourceTransducer, DEFAULT_MATCH_BATCH,
};
use liblevenshtein::transducer::Algorithm;
use proptest::prelude::*;
use std::collections::BTreeMap;
use support::fault_dictionary::{FaultDictionary, FaultOp};
use support::interop_dictionary::TestDictionary;
use support::query_start_fixture::{self, FixturePhase};
use vinary_tree_interop::VtStatus;

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

/// Drain WITHOUT sorting: the exact delivery order.
fn ordered_drain(cursor: &mut liblevenshtein::bindings::QueryCursor) -> Vec<Match> {
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
    result
}

/// Expected-match builder sorted with the same comparator as `drain`.
fn utf8_matches(entries: &[(&str, usize, Option<u64>)]) -> Vec<Match> {
    let mut result: Vec<Match> = entries
        .iter()
        .map(|(term, distance, id)| Match {
            term: MatchTerm::Utf8((*term).to_owned()),
            distance: *distance,
            id: *id,
        })
        .collect();
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

/// The canonical cross-language fixture
/// (vinary-tree-interop/conformance/query_start_snapshot.tsv) replayed at
/// the safe binding layer: the exact script is pinned (drift guard), the
/// frozen and post-mutation result sets are the published cross-language
/// oracles, and a cursor started before the mutation phase observes the
/// frozen set exactly.
#[test]
fn canonical_fixture_replays_at_the_safe_binding_layer() {
    let steps = query_start_fixture::load();
    let script: Vec<(FixturePhase, &str, &str, Option<u64>)> = steps
        .iter()
        .map(|step| {
            (
                step.phase,
                step.operation.as_str(),
                step.term.as_str(),
                step.id,
            )
        })
        .collect();
    assert_eq!(
        script,
        vec![
            (FixturePhase::Initial, "insert", "cat", Some(1)),
            (FixturePhase::Initial, "insert", "cot", Some(2)),
            (FixturePhase::Initial, "insert", "cut", Some(3)),
            (FixturePhase::Initial, "insert", "scat", None),
            (FixturePhase::Mutation, "remove", "cot", None),
            (FixturePhase::Mutation, "update", "cut", Some(30)),
            (FixturePhase::Mutation, "insert", "cit", Some(5)),
            (FixturePhase::Mutation, "compact", "", None),
            (FixturePhase::Mutation, "checkpoint", "", None),
        ],
        "the canonical fixture script drifted"
    );

    let dictionary = TestDictionary::new(
        steps
            .iter()
            .filter(|step| step.phase == FixturePhase::Initial)
            .map(|step| {
                assert_eq!(step.operation, "insert", "initial phase is insert-only");
                (step.term.clone(), step.id)
            }),
    );
    let transducer = unsafe {
        ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard)
            .expect("fixture dictionary accepted")
    };

    let frozen = drain(
        &mut transducer
            .query_utf8("cat", 2, QueryOrder::Traversal)
            .expect("frozen fixture query"),
    );
    assert_eq!(
        frozen,
        utf8_matches(&[
            ("cat", 0, Some(1)),
            ("cot", 1, Some(2)),
            ("cut", 1, Some(3)),
            ("scat", 1, None),
        ]),
        "the pinned cross-language frozen set"
    );

    let mut cursor = transducer
        .query_utf8("cat", 2, QueryOrder::Traversal)
        .expect("long-lived fixture query");
    let mut one = MatchBatch::default();
    assert_eq!(cursor.next_batch(&mut one, 1).expect("one-match prefix"), 1);
    let mut observed = one.as_slice().to_vec();

    for step in steps
        .iter()
        .filter(|step| step.phase == FixturePhase::Mutation)
    {
        match step.operation.as_str() {
            "insert" => dictionary.insert(&step.term, step.id),
            "update" => dictionary.update(&step.term, step.id),
            "remove" => dictionary.remove(&step.term),
            "compact" => dictionary.compact(),
            "checkpoint" => dictionary.checkpoint(),
            other => panic!("fixture mutation {other:?} is not a provider operation"),
        }
    }

    observed.extend(drain(&mut cursor));
    observed.sort_by(|left, right| format!("{:?}", left.term).cmp(&format!("{:?}", right.term)));
    assert_eq!(observed, frozen, "the mid-mutation cursor stays frozen");

    let fresh = drain(
        &mut transducer
            .query_utf8("cat", 2, QueryOrder::Traversal)
            .expect("post-mutation fixture query"),
    );
    assert_eq!(
        fresh,
        utf8_matches(&[
            ("cat", 0, Some(1)),
            ("cit", 1, Some(5)),
            ("cut", 1, Some(30)),
            ("scat", 1, None),
        ]),
        "the pinned cross-language post-mutation set"
    );
}

/// LLEV-CUR-1 (take-once + resume): one one-shot provider fault costs the
/// faulted pull exactly once; the channel does not latch and the cursor
/// finishes cleanly. The pull that observes the fault is discarded whole —
/// here `b` faults and `c`, consumed by the same poisoned pull, is dropped
/// with it.
#[test]
fn provider_fault_is_taken_once_then_the_cursor_resumes() {
    let dictionary = FaultDictionary::new([
        ("a".to_owned(), Some(1)),
        ("b".to_owned(), Some(2)),
        ("c".to_owned(), Some(3)),
    ]);
    let transducer = unsafe {
        ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard)
            .expect("fault provider accepted")
    };
    let mut cursor = transducer
        .query_utf8("", 1, QueryOrder::Traversal)
        .expect("query start crosses only snapshot and root");
    // Fire on the SECOND value read: `a` is clean, `b` faults.
    dictionary.fail_op_once_after(FaultOp::NodeValueU64, VtStatus::IoError.to_raw(), 1);

    let mut batch = MatchBatch::default();
    assert_eq!(
        cursor
            .next_batch(&mut batch, 1)
            .expect("the first pull is untainted"),
        1
    );
    assert_eq!(batch.as_slice()[0].term, MatchTerm::Utf8("a".to_owned()));

    assert_eq!(
        cursor.next_batch(&mut batch, 1),
        Err(BindingError::Provider(VtStatus::IoError)),
        "the recorded fault surfaces exactly once"
    );

    assert_eq!(
        cursor
            .next_batch(&mut batch, 1)
            .expect("take-once: the channel is empty again"),
        0,
        "b faulted and c was consumed by the poisoned pull; nothing remains"
    );
}

/// LLEV-CUR-1 (delivered prefix): when the LAST match faults, everything
/// before it is delivered normally, the error surfaces once, and the cursor
/// ends cleanly.
#[test]
fn fault_on_the_final_match_surfaces_after_the_delivered_prefix() {
    let dictionary = FaultDictionary::new([
        ("a".to_owned(), Some(1)),
        ("b".to_owned(), Some(2)),
        ("c".to_owned(), Some(3)),
    ]);
    let transducer = unsafe {
        ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard)
            .expect("fault provider accepted")
    };
    let mut cursor = transducer
        .query_utf8("", 1, QueryOrder::Traversal)
        .expect("query starts");
    // Fire on the THIRD value read: `a` and `b` are clean, `c` faults.
    dictionary.fail_op_once_after(FaultOp::NodeValueU64, VtStatus::Closed.to_raw(), 2);

    let mut batch = MatchBatch::default();
    for expected in ["a", "b"] {
        assert_eq!(cursor.next_batch(&mut batch, 1).expect("clean prefix"), 1);
        assert_eq!(
            batch.as_slice()[0].term,
            MatchTerm::Utf8(expected.to_owned())
        );
    }
    assert_eq!(
        cursor.next_batch(&mut batch, 1),
        Err(BindingError::Provider(VtStatus::Closed))
    );
    assert_eq!(
        cursor.next_batch(&mut batch, 1).expect("clean end"),
        0,
        "after the take-once error the stream ends without latching"
    );
}

/// LLEV-CUR-1 (first-fault-wins): with several callbacks armed to fail, the
/// FIRST fault recorded during a pull is the one surfaced; later faults in
/// the same window are dropped by the channel, and the cursor still ends
/// cleanly.
#[test]
fn first_recorded_fault_wins_when_multiple_callbacks_fail() {
    let dictionary = FaultDictionary::new([("a".to_owned(), Some(1)), ("b".to_owned(), Some(2))]);
    let transducer = unsafe {
        ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard)
            .expect("fault provider accepted")
    };
    let mut cursor = transducer
        .query_utf8("", 1, QueryOrder::Traversal)
        .expect("query starts");
    // Both armed: the traversal reaches root edges FIRST, so Closed must win
    // and IoError must never surface.
    dictionary.fail_op(FaultOp::NodeEdges, VtStatus::Closed.to_raw());
    dictionary.fail_op(FaultOp::NodeValueU64, VtStatus::IoError.to_raw());

    let mut batch = MatchBatch::default();
    assert_eq!(
        cursor.next_batch(&mut batch, 8),
        Err(BindingError::Provider(VtStatus::Closed)),
        "the first recorded fault wins"
    );
    assert_eq!(
        cursor.next_batch(&mut batch, 8).expect("clean end"),
        0,
        "the pruned traversal ends; the losing fault never surfaces"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// DistanceThenTerm cursors preserve their EXACT initial sequence — not
    /// just the multiset — across arbitrary mid-query provider mutations.
    #[test]
    fn arbitrary_ordered_binding_cursors_keep_their_exact_sequence(
        initial in prop::collection::vec(("[a-z]{1,8}", proptest::option::of(any::<u64>())), 2..24),
        mutations in prop::collection::vec((0u8..5, "[a-z]{1,8}", proptest::option::of(any::<u64>())), 1..16),
        query in "[a-z]{0,8}",
        prefix_seed in any::<usize>(),
    ) {
        let initial: BTreeMap<_, _> = initial.into_iter().collect();
        prop_assume!(initial.len() >= 2);
        let dictionary = TestDictionary::new(initial.into_iter());
        let transducer = unsafe {
            ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard)
                .expect("ordered provider accepted")
        };
        let frozen = ordered_drain(
            &mut transducer
                .query_utf8(&query, 8, QueryOrder::DistanceThenTerm)
                .expect("frozen ordered query"),
        );
        prop_assume!(frozen.len() >= 2);

        let mut cursor = transducer
            .query_utf8(&query, 8, QueryOrder::DistanceThenTerm)
            .expect("long-lived ordered query");
        let prefix_len = 1 + prefix_seed % (frozen.len() - 1);
        let mut batch = MatchBatch::default();
        cursor.next_batch(&mut batch, prefix_len).expect("ordered prefix");
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

        observed.extend(ordered_drain(&mut cursor));
        prop_assert_eq!(observed, frozen);
    }
}
