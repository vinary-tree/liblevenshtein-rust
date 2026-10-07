#![cfg(feature = "ffi")]

mod support;

use liblevenshtein::ffi::*;
use std::ffi::c_void;
use std::ptr;
use support::interop_dictionary::TestDictionary;

#[test]
fn query_cursor_remains_send_for_shared_runtime_registries() {
    fn assert_send<T: Send>() {}
    assert_send::<liblevenshtein::bindings::QueryCursor>();
}

unsafe extern "C" fn soft_c_cost(
    _context: *mut c_void,
    operation: u32,
    view: *const LlevEditContextView,
    query: u32,
    candidate: u32,
) -> f64 {
    match operation {
        0 if query == candidate => 0.0,
        0 if query == u32::from('c') && candidate == u32::from('s') => {
            let view = &*view;
            let query = std::slice::from_raw_parts(view.query_units, view.query_len);
            if query.get(view.query_index + 1) == Some(&u32::from('e')) {
                0.25
            } else {
                1.0
            }
        }
        0..=2 => 1.0,
        _ => f64::NAN,
    }
}

unsafe extern "C" fn too_cheap(
    _context: *mut c_void,
    operation: u32,
    _view: *const LlevEditContextView,
    query: u32,
    candidate: u32,
) -> f64 {
    if operation == 0 && query == candidate {
        0.0
    } else {
        0.1
    }
}

unsafe extern "C" fn id_at_least_three(_context: *mut c_void, has_id: u8, id: u64) -> u8 {
    u8::from(has_id != 0 && id >= 3)
}

unsafe extern "C" fn abort_filter(_context: *mut c_void, _has_id: u8, _id: u64) -> u8 {
    2
}

unsafe extern "C" fn invalid_specialized_reducer(
    _context: *mut c_void,
    _matches: *const LlevSpecializedMatch,
    _len: usize,
) -> u32 {
    u32::MAX
}

#[derive(Default)]
struct PrefixState {
    stack: Vec<char>,
    entered: usize,
    left: usize,
}

unsafe extern "C" fn prefix_callback(
    context: *mut c_void,
    operation: u32,
    candidate: u32,
    query: u32,
    depth: usize,
    prefix: *const u32,
    prefix_len: usize,
    out_score: *mut f64,
) -> u8 {
    let state = &mut *context.cast::<PrefixState>();
    match operation {
        0 => u8::from(candidate == query),
        1 => {
            let unit = char::from_u32(candidate).unwrap();
            assert_eq!(depth, state.stack.len() + 1);
            state.stack.push(unit);
            state.entered += 1;
            u8::from(depth > 1 || unit == 'c')
        }
        2 => {
            assert_eq!(state.stack.pop(), char::from_u32(candidate));
            state.left += 1;
            0
        }
        3 => {
            let prefix = std::slice::from_raw_parts(prefix, prefix_len);
            u8::from(prefix.last() == Some(&u32::from('e')))
        }
        4 => {
            *out_score = prefix_len as f64;
            1
        }
        _ => 0,
    }
}

unsafe fn collect(
    cursor: *mut LlevSpecializedCursor,
    batch_size: usize,
) -> Vec<(String, f64, Option<f64>)> {
    let mut result = Vec::new();
    loop {
        let mut batch = LlevSpecializedBatchView::default();
        match llev_specialized_cursor_next_batch(cursor, batch_size, &mut batch) {
            LlevStatus::Ok => {
                for item in std::slice::from_raw_parts(batch.matches, batch.len) {
                    let bytes =
                        std::slice::from_raw_parts(item.term_data.cast::<u8>(), item.byte_len);
                    result.push((
                        std::str::from_utf8(bytes).unwrap().to_owned(),
                        item.cost,
                        (item.has_score != 0).then_some(item.score),
                    ));
                }
                assert_eq!(
                    llev_specialized_cursor_release_batch(cursor, batch.generation),
                    LlevStatus::Ok
                );
            }
            LlevStatus::End => break,
            other => panic!("unexpected specialized status {other:?}"),
        }
    }
    result
}

#[test]
fn contextual_query_checks_costs_leases_and_snapshot() {
    unsafe {
        let dictionary = TestDictionary::new([
            ("ce".into(), Some(1)),
            ("se".into(), Some(2)),
            ("sea".into(), Some(3)),
            ("ci".into(), Some(4)),
        ]);
        let resource = dictionary.resource();
        let mut transducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );

        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_contextual_utf8(
                transducer,
                b"ce".as_ptr().cast(),
                2,
                0.25,
                0.25,
                Some(soft_c_cost),
                ptr::null_mut(),
                &mut cursor
            ),
            LlevStatus::Ok
        );
        dictionary.remove("se");
        dictionary.insert("zz", Some(5));
        let mut first = LlevSpecializedBatchView::default();
        assert_eq!(
            llev_specialized_cursor_next_batch(cursor, 1, &mut first),
            LlevStatus::Ok
        );
        assert_eq!(first.len, 1);
        let first_item = &*first.matches;
        let first_term = std::str::from_utf8(std::slice::from_raw_parts(
            first_item.term_data.cast::<u8>(),
            first_item.byte_len,
        ))
        .unwrap()
        .to_owned();
        let first_cost = first_item.cost;
        let mut blocked = LlevSpecializedBatchView::default();
        assert_eq!(
            llev_specialized_cursor_next_batch(cursor, 1, &mut blocked),
            LlevStatus::BatchInUse
        );
        assert_eq!(llev_specialized_cursor_free(cursor), LlevStatus::BatchInUse);
        assert_eq!(
            llev_specialized_cursor_release_batch(cursor, first.generation + 1),
            LlevStatus::InvalidArgument
        );
        assert_eq!(
            llev_specialized_cursor_release_batch(cursor, first.generation),
            LlevStatus::Ok
        );
        let tail = collect(cursor, 1);
        let mut terms: Vec<_> = tail.iter().map(|item| (item.0.as_str(), item.1)).collect();
        terms.push((first_term.as_str(), first_cost));
        terms.sort_by(|left, right| left.0.cmp(right.0));
        assert_eq!(terms, [("ce", 0.0), ("se", 0.25)]);
        assert_eq!(llev_specialized_cursor_free(cursor), LlevStatus::Ok);

        let mut invalid = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_contextual_utf8(
                transducer,
                b"ce".as_ptr().cast(),
                2,
                -1.0,
                0.25,
                Some(soft_c_cost),
                ptr::null_mut(),
                &mut invalid
            ),
            LlevStatus::InvalidArgument
        );
        assert!(invalid.is_null());
        assert_eq!(
            llev_transducer_query_contextual_utf8(
                transducer,
                b"ce".as_ptr().cast(),
                2,
                1.0,
                1.0,
                Some(too_cheap),
                ptr::null_mut(),
                &mut invalid
            ),
            LlevStatus::InvalidArgument
        );
        assert!(invalid.is_null());
        let mut reducer_cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_contextual_utf8(
                transducer,
                b"ce".as_ptr().cast(),
                2,
                1.0,
                1.0,
                Some(soft_c_cost),
                ptr::null_mut(),
                &mut reducer_cursor
            ),
            LlevStatus::Ok
        );
        let mut delivered = 0;
        assert_eq!(
            llev_specialized_cursor_reduce(
                reducer_cursor,
                1,
                Some(invalid_specialized_reducer),
                ptr::null_mut(),
                &mut delivered
            ),
            LlevStatus::InvalidArgument
        );
        assert_eq!(llev_specialized_cursor_free(reducer_cursor), LlevStatus::Ok);
        llev_transducer_free(transducer);
    }
}

#[test]
fn prefix_pruner_balances_enter_leave_on_early_close() {
    unsafe {
        let dictionary = TestDictionary::new([
            ("ce".into(), Some(1)),
            ("ci".into(), Some(2)),
            ("cat".into(), Some(3)),
            ("se".into(), Some(4)),
        ]);
        let resource = dictionary.resource();
        let mut transducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );
        let mut state = PrefixState::default();
        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_pruned_utf8(
                transducer,
                b"ce".as_ptr().cast(),
                2,
                2,
                Some(prefix_callback),
                (&mut state as *mut PrefixState).cast(),
                &mut cursor
            ),
            LlevStatus::Ok
        );
        let matches = collect(cursor, 1);
        assert_eq!(matches, [("ce".into(), 0.0, Some(2.0))]);
        assert_eq!(llev_specialized_cursor_free(cursor), LlevStatus::Ok);
        assert!(state.stack.is_empty());
        assert_eq!(state.entered, state.left);

        let mut early = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_pruned_utf8(
                transducer,
                b"ce".as_ptr().cast(),
                2,
                2,
                Some(prefix_callback),
                (&mut state as *mut PrefixState).cast(),
                &mut early
            ),
            LlevStatus::Ok
        );
        let mut batch = LlevSpecializedBatchView::default();
        assert_eq!(
            llev_specialized_cursor_next_batch(early, 1, &mut batch),
            LlevStatus::Ok
        );
        assert_eq!(
            llev_specialized_cursor_release_batch(early, batch.generation),
            LlevStatus::Ok
        );
        assert_eq!(llev_specialized_cursor_free(early), LlevStatus::Ok);
        assert!(state.stack.is_empty());
        assert_eq!(state.entered, state.left);
        llev_transducer_free(transducer);
    }
}

#[test]
fn value_filter_checks_ids_before_term_construction_and_aborts() {
    unsafe {
        let dictionary = TestDictionary::new([
            ("ce".into(), Some(1)),
            ("se".into(), Some(2)),
            ("ci".into(), Some(3)),
            ("cat".into(), Some(4)),
            ("sea".into(), Some(5)),
        ]);
        let resource = dictionary.resource();
        let mut transducer = ptr::null_mut();
        assert_eq!(
            llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
            LlevStatus::Ok
        );

        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_filtered_utf8(
                transducer,
                b"ce".as_ptr().cast(),
                2,
                2,
                Some(id_at_least_three),
                ptr::null_mut(),
                &mut cursor
            ),
            LlevStatus::Ok
        );
        let mut observed = Vec::new();
        loop {
            let mut batch = LlevMatchBatchView::default();
            match llev_query_cursor_next_batch(cursor, 1, &mut batch) {
                LlevStatus::Ok => {
                    let item = &*batch.matches;
                    let term = std::str::from_utf8(std::slice::from_raw_parts(
                        item.term_data.cast::<u8>(),
                        item.byte_len,
                    ))
                    .unwrap()
                    .to_owned();
                    observed.push((term, item.id));
                    assert_eq!(item.has_id, 1);
                    assert_eq!(
                        llev_query_cursor_release_batch(cursor, batch.generation),
                        LlevStatus::Ok
                    );
                }
                LlevStatus::End => break,
                other => panic!("unexpected filtered status {other:?}"),
            }
        }
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        observed.sort();
        assert_eq!(
            observed,
            [("cat".into(), 4), ("ci".into(), 3), ("sea".into(), 5)]
        );

        let mut aborted = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_filtered_utf8(
                transducer,
                b"ce".as_ptr().cast(),
                2,
                2,
                Some(abort_filter),
                ptr::null_mut(),
                &mut aborted
            ),
            LlevStatus::Ok
        );
        let mut batch = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(aborted, 1, &mut batch),
            LlevStatus::InvalidArgument
        );
        assert_eq!(llev_query_cursor_free(aborted), LlevStatus::Ok);
        llev_transducer_free(transducer);
    }
}
