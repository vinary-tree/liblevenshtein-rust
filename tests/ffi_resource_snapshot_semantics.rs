//! C-ABI lease and query-start snapshot integration tests, including the
//! canonical cross-language fixture replayed through the C ABI (the
//! conformance oracle every language binding must reproduce) and a
//! mutation-script property drain driven entirely through `llev_*`.

#![cfg(feature = "ffi")]

mod support;

use liblevenshtein::ffi::*;
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr;
use support::interop_dictionary::TestDictionary;
use support::query_start_fixture::{self, FixturePhase};

unsafe fn copy_batch(view: LlevMatchBatchView) -> Vec<(String, usize, Option<u64>)> {
    std::slice::from_raw_parts(view.matches, view.len)
        .iter()
        .map(|item| {
            assert_eq!(
                item.unit_domain,
                vinary_tree_interop::VtUnitDomain::UnicodeScalar
            );
            let bytes = std::slice::from_raw_parts(item.term_data.cast::<u8>(), item.byte_len);
            (
                std::str::from_utf8(bytes).unwrap().to_owned(),
                item.distance,
                (item.has_id == 1).then_some(item.id),
            )
        })
        .collect()
}

#[test]
fn c_abi_enforces_batch_leases_and_one_long_lived_snapshot() {
    unsafe {
        let dictionary = TestDictionary::new([
            ("cat".to_owned(), Some(1)),
            ("cot".to_owned(), Some(2)),
            ("cut".to_owned(), Some(3)),
            ("scat".to_owned(), Some(4)),
        ]);
        let resource = dictionary.resource();
        let mut transducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );

        let mut expected_cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_utf8(
                transducer,
                b"cat".as_ptr().cast(),
                3,
                2,
                LlevQueryOrder::Traversal as u32,
                &mut expected_cursor,
            ),
            LlevStatus::Ok
        );
        let mut expected = Vec::new();
        loop {
            let mut view = LlevMatchBatchView::default();
            match llev_query_cursor_next_batch(expected_cursor, 2, &mut view) {
                LlevStatus::Ok => {
                    expected.extend(copy_batch(view));
                    assert_eq!(
                        llev_query_cursor_release_batch(expected_cursor, view.generation),
                        LlevStatus::Ok
                    );
                }
                LlevStatus::End => break,
                status => panic!("unexpected status {status:?}"),
            }
        }
        assert_eq!(llev_query_cursor_free(expected_cursor), LlevStatus::Ok);
        expected.sort();

        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_utf8(
                transducer,
                b"cat".as_ptr().cast(),
                3,
                2,
                LlevQueryOrder::Traversal as u32,
                &mut cursor,
            ),
            LlevStatus::Ok
        );
        let mut first = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut first),
            LlevStatus::Ok
        );
        let mut observed = copy_batch(first);

        let mut blocked = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut blocked),
            LlevStatus::BatchInUse
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::BatchInUse);

        dictionary.remove("cot");
        dictionary.update("cut", Some(30));
        dictionary.insert("cit", Some(5));
        dictionary.compact();
        dictionary.checkpoint();

        assert_eq!(
            llev_query_cursor_release_batch(cursor, first.generation),
            LlevStatus::Ok
        );
        loop {
            let mut view = LlevMatchBatchView::default();
            match llev_query_cursor_next_batch(cursor, 2, &mut view) {
                LlevStatus::Ok => {
                    observed.extend(copy_batch(view));
                    assert_eq!(
                        llev_query_cursor_release_batch(cursor, view.generation),
                        LlevStatus::Ok
                    );
                }
                LlevStatus::End => break,
                status => panic!("unexpected status {status:?}"),
            }
        }
        observed.sort();
        assert_eq!(observed, expected);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        llev_transducer_free(transducer);
    }
}

struct Reduced {
    terms: Vec<String>,
    calls: usize,
}

unsafe extern "C" fn reducer(context: *mut c_void, matches: *const LlevMatch, len: usize) -> u32 {
    let output = &mut *context.cast::<Reduced>();
    output.calls += 1;
    for item in std::slice::from_raw_parts(matches, len) {
        let bytes = std::slice::from_raw_parts(item.term_data.cast::<u8>(), item.byte_len);
        output
            .terms
            .push(std::str::from_utf8(bytes).unwrap().to_owned());
    }
    // The reducer wire returns raw u32 (LLEV-B16): encode the enum.
    LlevStatus::Ok as u32
}

#[test]
fn c_reducer_uses_one_callback_per_batch_and_no_result_vector_abi() {
    unsafe {
        let dictionary = TestDictionary::new((0..40).map(|id| (format!("term{id:02}"), Some(id))));
        let resource = dictionary.resource();
        let mut transducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );
        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_utf8(
                transducer,
                b"term".as_ptr().cast(),
                4,
                4,
                LlevQueryOrder::Traversal as u32,
                &mut cursor,
            ),
            LlevStatus::Ok
        );
        let mut output = Reduced {
            terms: vec![],
            calls: 0,
        };
        let mut count = 0;
        assert_eq!(
            llev_query_cursor_reduce(
                cursor,
                7,
                Some(reducer),
                (&mut output as *mut Reduced).cast(),
                &mut count,
            ),
            LlevStatus::Ok
        );
        assert_eq!(count, output.terms.len());
        assert_eq!(output.calls, count.div_ceil(7));
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        llev_transducer_free(transducer);
    }
}

unsafe fn c_open(
    transducer: *mut LlevTransducer,
    query: &str,
    max_distance: usize,
) -> *mut LlevQueryCursor {
    let mut cursor = ptr::null_mut();
    assert_eq!(
        llev_transducer_query_utf8(
            transducer,
            query.as_ptr().cast(),
            query.len(),
            max_distance,
            LlevQueryOrder::Traversal as u32,
            &mut cursor,
        ),
        LlevStatus::Ok
    );
    cursor
}

/// Drain a cursor completely with `batch_size`-bounded leases; sorted output.
unsafe fn c_drain(
    cursor: *mut LlevQueryCursor,
    batch_size: usize,
) -> Vec<(String, usize, Option<u64>)> {
    let mut all = Vec::new();
    loop {
        let mut view = LlevMatchBatchView::default();
        match llev_query_cursor_next_batch(cursor, batch_size, &mut view) {
            LlevStatus::Ok => {
                all.extend(copy_batch(view));
                assert_eq!(
                    llev_query_cursor_release_batch(cursor, view.generation),
                    LlevStatus::Ok
                );
            }
            LlevStatus::End => break,
            status => panic!("unexpected status {status:?}"),
        }
    }
    all.sort();
    all
}

/// The canonical cross-language fixture
/// (vinary-tree-interop/conformance/query_start_snapshot.tsv) replayed
/// through the C ABI. The frozen and post-mutation result sets asserted
/// here are the conformance oracle: every language binding replaying the
/// same fixture must observe exactly these tuples.
#[test]
fn canonical_fixture_replays_through_the_c_abi() {
    let steps = query_start_fixture::load();
    let dictionary = TestDictionary::new(
        steps
            .iter()
            .filter(|step| step.phase == FixturePhase::Initial)
            .map(|step| {
                assert_eq!(step.operation, "insert", "initial phase is insert-only");
                (step.term.clone(), step.id)
            }),
    );
    let frozen_oracle = {
        let mut expected = vec![
            ("cat".to_owned(), 0usize, Some(1)),
            ("cot".to_owned(), 1, Some(2)),
            ("cut".to_owned(), 1, Some(3)),
            ("scat".to_owned(), 1, None),
        ];
        expected.sort();
        expected
    };
    let post_oracle = {
        let mut expected = vec![
            ("cat".to_owned(), 0usize, Some(1)),
            ("cit".to_owned(), 1, Some(5)),
            ("cut".to_owned(), 1, Some(30)),
            ("scat".to_owned(), 1, None),
        ];
        expected.sort();
        expected
    };

    unsafe {
        let resource = dictionary.resource();
        let mut transducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );

        let frozen_cursor = c_open(transducer, "cat", 2);
        let frozen = c_drain(frozen_cursor, 2);
        assert_eq!(llev_query_cursor_free(frozen_cursor), LlevStatus::Ok);
        assert_eq!(
            frozen, frozen_oracle,
            "the pinned cross-language frozen set"
        );

        // A cursor held across the fixture's whole mutation phase.
        let cursor = c_open(transducer, "cat", 2);
        let mut view = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut view),
            LlevStatus::Ok
        );
        let mut observed = copy_batch(view);
        assert_eq!(
            llev_query_cursor_release_batch(cursor, view.generation),
            LlevStatus::Ok
        );

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

        observed.extend(c_drain(cursor, 2));
        observed.sort();
        assert_eq!(
            observed, frozen_oracle,
            "the mid-mutation cursor stays frozen"
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);

        let fresh_cursor = c_open(transducer, "cat", 2);
        let fresh = c_drain(fresh_cursor, 2);
        assert_eq!(llev_query_cursor_free(fresh_cursor), LlevStatus::Ok);
        assert_eq!(fresh, post_oracle, "the pinned cross-language post set");

        llev_transducer_free(transducer);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Arbitrary mutation scripts driven WHILE a C-ABI cursor is partially
    /// consumed never change what that cursor observes, at every lease
    /// batch size.
    #[test]
    fn arbitrary_mutation_scripts_preserve_the_c_abi_snapshot(
        initial in prop::collection::vec(("[a-z]{1,6}", proptest::option::of(any::<u64>())), 2..20),
        mutations in prop::collection::vec((0u8..5, "[a-z]{1,6}", proptest::option::of(any::<u64>())), 0..16),
        query in "[a-z]{0,6}",
        batch_size in 1usize..8,
        prefix_seed in any::<usize>(),
    ) {
        let initial: BTreeMap<_, _> = initial.into_iter().collect();
        prop_assume!(initial.len() >= 2);
        let dictionary = TestDictionary::new(initial.into_iter());
        unsafe {
            let resource = dictionary.resource();
            let mut transducer = ptr::null_mut();
            prop_assert_eq!(
                llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
                LlevStatus::Ok
            );

            let frozen_cursor = c_open(transducer, &query, 8);
            let frozen = c_drain(frozen_cursor, batch_size);
            prop_assert_eq!(llev_query_cursor_free(frozen_cursor), LlevStatus::Ok);
            if frozen.len() < 2 {
                llev_transducer_free(transducer);
                // Not enough stream to leave a long-lived suffix.
                return Ok(());
            }

            let cursor = c_open(transducer, &query, 8);
            let prefix_len = 1 + prefix_seed % (frozen.len() - 1);
            let mut view = LlevMatchBatchView::default();
            prop_assert_eq!(
                llev_query_cursor_next_batch(cursor, prefix_len, &mut view),
                LlevStatus::Ok
            );
            let mut observed = copy_batch(view);
            prop_assert_eq!(
                llev_query_cursor_release_batch(cursor, view.generation),
                LlevStatus::Ok
            );

            for (operation, term, value) in mutations {
                match operation {
                    0 => dictionary.insert(&term, value),
                    1 => dictionary.remove(&term),
                    2 => dictionary.update(&term, value),
                    3 => dictionary.compact(),
                    _ => dictionary.checkpoint(),
                }
            }

            observed.extend(c_drain(cursor, batch_size));
            observed.sort();
            prop_assert_eq!(observed, frozen);
            prop_assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
            llev_transducer_free(transducer);
        }
    }
}
