//! C-ABI lease and query-start snapshot integration tests.

#![cfg(feature = "ffi")]

mod support;

use liblevenshtein::ffi::*;
use std::ffi::c_void;
use std::ptr;
use support::interop_dictionary::TestDictionary;

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
