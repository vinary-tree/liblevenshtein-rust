//! Provider fault-injection pin matrix at the C ABI.
//!
//! Drives the programmable misbehaving provider in
//! `tests/support/fault_dictionary.rs` through `llev_*` and pins the exact
//! `LlevStatus` for every programmable provider failure, per
//! `src/ffi/index.rs::map_binding_error` over `src/bindings.rs::status`
//! decode (the raw-u32 status wire, ledger LLEV-B6):
//!
//! | injected raw status        | surfaced `LlevStatus`                     |
//! |----------------------------|-------------------------------------------|
//! | 1  (`End`)                 | `ProviderError` (LLEV-STAT-6)             |
//! | 2  (`InvalidArgument`)     | `InvalidArgument`                         |
//! | 3  (`NullPointer`)         | `NullPointer`                             |
//! | 4  (`Unsupported`)         | `Unsupported` (`ProviderError` from       |
//! |                            | `query_interface`: missing interface)     |
//! | 5  (`IoError`)             | `IoError`                                 |
//! | 6  (`Closed`)              | `Closed`                                  |
//! | 7  (`LimitExceeded`)       | `LimitExceeded`                           |
//! | 8  (`ProviderError`)       | `ProviderError`                           |
//! | 9, 42, `u32::MAX` (no      | `ProviderError` + "out-of-range status    |
//! | `VtStatus` discriminant)   | code" message (the LLEV-B6 regression)    |
//!
//! Structural misbehavior (well-formed status, malformed data) is rejected
//! as `ProviderError`-class failures — never undefined behavior: page-shape
//! lies fail the ConsumerAcceptance predicate ("invalid edge page lengths"),
//! `has_value = 7` / `is_final = 7` fail the boolean pins, nonzero
//! `VtOptionalU64::reserved` is rejected per VT-ABI-5 (ledger LLEV-B7), and
//! garbage node identifiers surface as the provider's own clean
//! `InvalidArgument`.
//!
//! Every fault path finishes with a full lifecycle-ledger balance check
//! (releases == retains + births, drops == births), which is also the
//! regression for the snapshot-decode retain leak fixed in
//! `src/bindings.rs::Provider::snapshot` during this wave.
//!
//! INVARIANT-HOOK: LLEV-STAT-6 — a provider callback returning `End` is
//! decoded as a provider error, never success or a silent stop
//! (`end_status_from_every_callback_is_a_provider_error`).

#![cfg(feature = "ffi")]

mod support;

use liblevenshtein::ffi::*;
use std::ffi::CStr;
use std::ptr;
use std::sync::{Arc, Barrier};
use support::fault_dictionary::{FaultDictionary, FaultOp};
use vinary_tree_interop::VtStatus;

/// Where an armed fault first crosses the C ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Surface {
    /// Surfaces from `llev_transducer_query_utf8` (query-start callbacks).
    AtQuery,
    /// Surfaces from `llev_query_cursor_next_batch` (traversal callbacks).
    AtDrain,
}

fn corpus() -> Vec<(String, Option<u64>)> {
    vec![
        ("cat".to_owned(), Some(1)),
        ("cot".to_owned(), Some(2)),
        ("cut".to_owned(), Some(3)),
        ("scat".to_owned(), None),
    ]
}

/// A corpus whose root fans out past one edge page, so paging reaches a
/// second `node_edges` call (required by the deflating-total misbehavior).
fn wide_corpus() -> Vec<(String, Option<u64>)> {
    let mut terms = Vec::with_capacity(300);
    for index in 0u32..300 {
        let first = char::from_u32(0x2000 + index).expect("BMP scalar");
        terms.push((format!("{first}x"), Some(u64::from(index))));
    }
    terms
}

fn last_message() -> String {
    unsafe { CStr::from_ptr(llev_last_error_message()) }
        .to_string_lossy()
        .into_owned()
}

unsafe fn new_transducer(dictionary: &FaultDictionary) -> *mut LlevTransducer {
    let resource = dictionary.resource();
    let mut transducer = ptr::null_mut();
    assert_eq!(
        llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
        LlevStatus::Ok,
        "transducer construction must succeed before faults are armed"
    );
    transducer
}

unsafe fn open_cursor(transducer: *mut LlevTransducer, query: &str) -> *mut LlevQueryCursor {
    let mut cursor = ptr::null_mut();
    assert_eq!(
        llev_transducer_query_utf8(
            transducer,
            query.as_ptr().cast(),
            query.len(),
            2,
            LlevQueryOrder::Traversal as u32,
            &mut cursor,
        ),
        LlevStatus::Ok,
        "query start must succeed before drain faults are armed"
    );
    cursor
}

fn assert_view_zeroed(view: &LlevMatchBatchView) {
    assert!(view.matches.is_null(), "non-Ok advance must zero the view");
    assert_eq!(view.len, 0, "non-Ok advance must zero the view length");
    assert_eq!(view.generation, 0, "non-Ok advance must not mint a lease");
}

/// Drain until `End`, tolerating only `allowed` as a non-Ok status; returns
/// the number of matches delivered. Bounded so a misbehaving latch cannot
/// hang the suite.
unsafe fn drain_tolerating(cursor: *mut LlevQueryCursor, allowed: LlevStatus) -> usize {
    let mut delivered = 0usize;
    for _ in 0..64 {
        let mut view = LlevMatchBatchView::default();
        match llev_query_cursor_next_batch(cursor, 8, &mut view) {
            LlevStatus::Ok => {
                delivered += view.len;
                assert_eq!(
                    llev_query_cursor_release_batch(cursor, view.generation),
                    LlevStatus::Ok
                );
            }
            LlevStatus::End => return delivered,
            status if status == allowed => assert_view_zeroed(&view),
            status => panic!("unexpected drain status {status:?}"),
        }
    }
    panic!("cursor did not reach End within the iteration bound");
}

/// The pinned `map_binding_error` image of one injected raw status.
fn expected_status(op: FaultOp, raw: u32) -> LlevStatus {
    match VtStatus::from_raw(raw) {
        // Out-of-range discriminants decode to InvalidProviderOutput.
        None => LlevStatus::ProviderError,
        // query_interface has one special row: a provider answering
        // Unsupported is reported as a missing dictionary interface.
        Some(VtStatus::Unsupported) if op == FaultOp::QueryInterface => LlevStatus::ProviderError,
        Some(VtStatus::InvalidArgument) => LlevStatus::InvalidArgument,
        Some(VtStatus::NullPointer) => LlevStatus::NullPointer,
        Some(VtStatus::Unsupported) => LlevStatus::Unsupported,
        Some(VtStatus::IoError) => LlevStatus::IoError,
        Some(VtStatus::Closed) => LlevStatus::Closed,
        Some(VtStatus::LimitExceeded) => LlevStatus::LimitExceeded,
        Some(VtStatus::End | VtStatus::ProviderError | VtStatus::BatchInUse) => {
            LlevStatus::ProviderError
        }
        Some(VtStatus::Ok) => unreachable!("success is never injected as a fault"),
    }
}

/// Run one (op, raw) matrix cell end to end, including the resume-after-heal
/// and lifecycle-balance postconditions.
fn run_matrix_cell(op: FaultOp, surface: Surface, raw: u32, check_message: Option<&str>) {
    let dictionary = FaultDictionary::new(corpus());
    let probe = dictionary.probe();
    let expected = expected_status(op, raw);
    unsafe {
        let transducer = new_transducer(&dictionary);
        match surface {
            Surface::AtQuery => {
                dictionary.fail_op(op, raw);
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
                    expected,
                    "query-start fault {op:?} raw {raw}"
                );
                assert!(cursor.is_null(), "no cursor may be written on failure");
                if let Some(needle) = check_message {
                    let message = last_message();
                    assert!(
                        message.contains(needle),
                        "message {message:?} must contain {needle:?}"
                    );
                }
                // Healed, the same transducer opens and drains cleanly.
                dictionary.clear_faults();
                let cursor = open_cursor(transducer, "cat");
                assert_eq!(drain_tolerating(cursor, LlevStatus::Ok), 4);
                assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
            }
            Surface::AtDrain => {
                let cursor = open_cursor(transducer, "cat");
                dictionary.fail_op(op, raw);
                let mut view = LlevMatchBatchView::default();
                assert_eq!(
                    llev_query_cursor_next_batch(cursor, 8, &mut view),
                    expected,
                    "drain fault {op:?} raw {raw}"
                );
                assert_view_zeroed(&view);
                if let Some(needle) = check_message {
                    let message = last_message();
                    assert!(
                        message.contains(needle),
                        "message {message:?} must contain {needle:?}"
                    );
                }
                // Healed, the cursor keeps working: the fault channel is
                // take-once, so whatever survives the pruned traversal
                // arrives, then a clean End.
                dictionary.clear_faults();
                let _ = drain_tolerating(cursor, expected);
                assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
            }
        }
        llev_transducer_free(transducer);
    }
    drop(dictionary);
    probe.assert_balanced();
}

/// The full valid-error-status matrix: every reachable callback times every
/// portable error status maps to its pinned `LlevStatus`.
#[test]
fn provider_status_codes_map_to_pinned_llev_statuses() {
    let cells = [
        (FaultOp::Snapshot, Surface::AtQuery),
        (FaultOp::Root, Surface::AtQuery),
        (FaultOp::NodeIsFinal, Surface::AtDrain),
        (FaultOp::NodeValueU64, Surface::AtDrain),
        (FaultOp::NodeEdges, Surface::AtDrain),
    ];
    let raws = [
        VtStatus::InvalidArgument.to_raw(),
        VtStatus::NullPointer.to_raw(),
        VtStatus::Unsupported.to_raw(),
        VtStatus::IoError.to_raw(),
        VtStatus::Closed.to_raw(),
        VtStatus::LimitExceeded.to_raw(),
        VtStatus::ProviderError.to_raw(),
    ];
    for (op, surface) in cells {
        for raw in raws {
            run_matrix_cell(op, surface, raw, None);
        }
    }
}

/// `query_interface` surfaces at two distinct decode points — transducer
/// construction and snapshot decode at query start — with the pinned special
/// row for `Unsupported`. The snapshot-decode half is also the regression
/// for the snapshot retain leak: on decode failure the consumer must release
/// the provider-transferred snapshot retain (fixed in
/// `src/bindings.rs::Provider::snapshot` this wave; the ledger balance below
/// fails without the fix).
#[test]
fn query_interface_failures_pin_at_construction_and_at_snapshot_decode() {
    let raws = [
        VtStatus::End.to_raw(),
        VtStatus::InvalidArgument.to_raw(),
        VtStatus::NullPointer.to_raw(),
        VtStatus::Unsupported.to_raw(),
        VtStatus::IoError.to_raw(),
        VtStatus::Closed.to_raw(),
        VtStatus::LimitExceeded.to_raw(),
        VtStatus::ProviderError.to_raw(),
        9,
        42,
        u32::MAX,
    ];
    for raw in raws {
        let expected = expected_status(FaultOp::QueryInterface, raw);

        // Surface 1: llev_transducer_new.
        let dictionary = FaultDictionary::new(corpus());
        let probe = dictionary.probe();
        dictionary.fail_op(FaultOp::QueryInterface, raw);
        unsafe {
            let resource = dictionary.resource();
            let mut transducer = ptr::null_mut();
            assert_eq!(
                llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer),
                expected,
                "construction-time query_interface fault raw {raw}"
            );
            assert!(transducer.is_null());
            dictionary.clear_faults();
            let transducer = new_transducer(&dictionary);
            llev_transducer_free(transducer);
        }
        drop(dictionary);
        probe.assert_balanced();

        // Surface 2: snapshot decode inside llev_transducer_query_utf8.
        let dictionary = FaultDictionary::new(corpus());
        let probe = dictionary.probe();
        unsafe {
            let transducer = new_transducer(&dictionary);
            dictionary.fail_op(FaultOp::QueryInterface, raw);
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
                expected,
                "snapshot-decode query_interface fault raw {raw}"
            );
            assert!(cursor.is_null());
            dictionary.clear_faults();
            let cursor = open_cursor(transducer, "cat");
            assert_eq!(drain_tolerating(cursor, LlevStatus::Ok), 4);
            assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
            llev_transducer_free(transducer);
        }
        drop(dictionary);
        // Without the snapshot-decode release fix, the snapshot context born
        // during the failed query never drops and this balance fails.
        probe.assert_balanced();
    }
}

/// LLEV-STAT-6: `End` (raw 1) is a valid wire value but NEVER a valid
/// callback result; every reachable callback returning it is decoded as
/// `ProviderError` — not success, not a silent stream stop.
#[test]
fn end_status_from_every_callback_is_a_provider_error() {
    let end = VtStatus::End.to_raw();
    for (op, surface) in [
        (FaultOp::Snapshot, Surface::AtQuery),
        (FaultOp::Root, Surface::AtQuery),
        (FaultOp::NodeIsFinal, Surface::AtDrain),
        (FaultOp::NodeValueU64, Surface::AtDrain),
        (FaultOp::NodeEdges, Surface::AtDrain),
    ] {
        assert_eq!(
            expected_status(op, end),
            LlevStatus::ProviderError,
            "the End row of the pin matrix must be ProviderError"
        );
        run_matrix_cell(op, surface, end, None);
    }
    // query_interface's End row is exercised (both surfaces) in
    // query_interface_failures_pin_at_construction_and_at_snapshot_decode.
}

/// The LLEV-B6 regression: raw statuses with no `VtStatus` discriminant —
/// including the first out-of-range value 10 and u32::MAX — become
/// `ProviderError` with the pinned decode message, never undefined behavior.
#[test]
fn out_of_range_status_codes_are_provider_errors() {
    for raw in [10u32, 42, u32::MAX] {
        for (op, surface) in [
            (FaultOp::Snapshot, Surface::AtQuery),
            (FaultOp::Root, Surface::AtQuery),
            (FaultOp::NodeIsFinal, Surface::AtDrain),
            (FaultOp::NodeValueU64, Surface::AtDrain),
            (FaultOp::NodeEdges, Surface::AtDrain),
        ] {
            run_matrix_cell(op, surface, raw, Some("out-of-range status code"));
        }
    }
}

/// Structural misbehavior: the provider reports success but writes malformed
/// data. Every mode is rejected as a clean `ProviderError` with its pinned
/// message; none is trusted, none is UB.
#[test]
fn structural_misbehavior_is_rejected_as_provider_error_never_ub() {
    struct ModeCase {
        name: &'static str,
        arm: fn(&FaultDictionary),
        surface: Surface,
        message: &'static str,
        wide: bool,
    }
    let cases = [
        ModeCase {
            name: "overshoot written",
            arm: |dictionary| dictionary.set_edges_overshoot_written(true),
            surface: Surface::AtDrain,
            message: "invalid edge page lengths",
            wide: false,
        },
        ModeCase {
            name: "deflating total across pages",
            arm: |dictionary| dictionary.set_edges_deflate_total(true),
            surface: Surface::AtDrain,
            message: "invalid edge page lengths",
            wide: true,
        },
        ModeCase {
            name: "inflated total (empty page with remaining)",
            arm: |dictionary| dictionary.set_edges_inflate_total(true),
            surface: Surface::AtDrain,
            message: "invalid edge page lengths",
            wide: false,
        },
        ModeCase {
            name: "garbage edge labels",
            arm: |dictionary| dictionary.set_edges_garbage_labels(true),
            surface: Surface::AtDrain,
            message: "edge label is outside its domain",
            wide: false,
        },
        ModeCase {
            name: "has_value seven",
            arm: |dictionary| dictionary.set_value_has_value_seven(true),
            surface: Surface::AtDrain,
            message: "has_value was not zero or one",
            wide: false,
        },
        ModeCase {
            // VT-ABI-5 / ledger LLEV-B7: reserved bytes are load-bearing ABI
            // surface; garbage there must be rejected, not reinterpreted.
            name: "nonzero reserved bytes",
            arm: |dictionary| dictionary.set_value_nonzero_reserved(true),
            surface: Surface::AtDrain,
            message: "reserved bytes were not zero",
            wide: false,
        },
        ModeCase {
            name: "is_final seven",
            arm: |dictionary| dictionary.set_is_final_seven(true),
            surface: Surface::AtDrain,
            message: "is_final was not zero or one",
            wide: false,
        },
        ModeCase {
            name: "null snapshot resource",
            arm: |dictionary| dictionary.set_snapshot_null(true),
            surface: Surface::AtQuery,
            message: "snapshot returned a null resource",
            wide: false,
        },
    ];

    for case in cases {
        let dictionary = if case.wide {
            FaultDictionary::new(wide_corpus())
        } else {
            FaultDictionary::new(corpus())
        };
        let probe = dictionary.probe();
        unsafe {
            let transducer = new_transducer(&dictionary);
            // The pinned message must be read IMMEDIATELY after the failing
            // call: any later successful call (the healed drain, the free)
            // clears the thread-local slot by design.
            let check_message = |name: &str, needle: &str| {
                let message = last_message();
                assert!(
                    message.contains(needle),
                    "{name}: message {message:?} must contain {needle:?}"
                );
            };
            match case.surface {
                Surface::AtQuery => {
                    (case.arm)(&dictionary);
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
                        LlevStatus::ProviderError,
                        "{} must be a ProviderError",
                        case.name
                    );
                    check_message(case.name, case.message);
                    assert!(cursor.is_null());
                }
                Surface::AtDrain => {
                    let cursor = open_cursor(transducer, "cat");
                    (case.arm)(&dictionary);
                    let mut view = LlevMatchBatchView::default();
                    assert_eq!(
                        llev_query_cursor_next_batch(cursor, 8, &mut view),
                        LlevStatus::ProviderError,
                        "{} must be a ProviderError",
                        case.name
                    );
                    check_message(case.name, case.message);
                    assert_view_zeroed(&view);
                    dictionary.clear_faults();
                    let _ = drain_tolerating(cursor, LlevStatus::ProviderError);
                    assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
                }
            }
            llev_transducer_free(transducer);
        }
        drop(dictionary);
        probe.assert_balanced();
    }
}

/// Garbage node identifiers (edges pointing at `u64::MAX`) surface as the
/// provider's own clean `InvalidArgument` on the next node callback — a
/// value-level rejection, never a dereference of the bogus identifier.
#[test]
fn garbage_node_identifiers_surface_as_clean_provider_rejections() {
    let dictionary = FaultDictionary::new(corpus());
    let probe = dictionary.probe();
    dictionary.set_edges_garbage_nodes(true);
    unsafe {
        let transducer = new_transducer(&dictionary);
        let cursor = open_cursor(transducer, "cat");
        let mut view = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 8, &mut view),
            LlevStatus::InvalidArgument,
            "the bogus node id is rejected by the provider's own bounds check"
        );
        assert_view_zeroed(&view);
        // Even after healing the mode, identifiers already absorbed into the
        // traversal keep being rejected as values until the frontier drains;
        // the cursor ends cleanly and never faults the process.
        dictionary.clear_faults();
        let _ = drain_tolerating(cursor, LlevStatus::InvalidArgument);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        llev_transducer_free(transducer);
    }
    drop(dictionary);
    probe.assert_balanced();
}

/// One-shot fault mid-stream at the C ABI: the error surfaces exactly once,
/// the cursor is not latched, and the untainted remainder arrives before a
/// clean End. (The safe-binding twin of this pin — with the registry row —
/// lives in tests/binding_snapshot_semantics.rs.)
#[test]
fn fault_status_does_not_latch_the_cursor_at_the_c_abi() {
    let dictionary = FaultDictionary::new(corpus());
    let probe = dictionary.probe();
    // Fire on the SECOND node_value_u64 call: "cat" is delivered, "cot"
    // takes the fault, and "cut" is consumed by the same poisoned pull.
    dictionary.fail_op_once_after(FaultOp::NodeValueU64, VtStatus::IoError.to_raw(), 1);
    unsafe {
        let transducer = new_transducer(&dictionary);
        let cursor = open_cursor(transducer, "cat");

        let mut view = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut view),
            LlevStatus::Ok
        );
        assert_eq!(view.len, 1);
        let first = std::slice::from_raw_parts(view.matches, view.len)[0];
        let bytes = std::slice::from_raw_parts(first.term_data.cast::<u8>(), first.byte_len);
        assert_eq!(bytes, b"cat");
        assert_eq!(
            llev_query_cursor_release_batch(cursor, view.generation),
            LlevStatus::Ok
        );

        let mut poisoned = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut poisoned),
            LlevStatus::IoError,
            "the one-shot fault surfaces exactly once"
        );
        assert_view_zeroed(&poisoned);

        let mut resumed = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut resumed),
            LlevStatus::Ok,
            "the fault does not latch: the cursor resumes"
        );
        assert_eq!(resumed.len, 1);
        let last = std::slice::from_raw_parts(resumed.matches, resumed.len)[0];
        let bytes = std::slice::from_raw_parts(last.term_data.cast::<u8>(), last.byte_len);
        assert_eq!(
            bytes, b"scat",
            "the poisoned pull cost cot (faulted) and cut (discarded); the \
             untainted remainder still arrives"
        );
        assert_eq!(
            llev_query_cursor_release_batch(cursor, resumed.generation),
            LlevStatus::Ok
        );

        let mut done = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut done),
            LlevStatus::End
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        llev_transducer_free(transducer);
    }
    drop(dictionary);
    probe.assert_balanced();
}

/// `llev_last_error_message` is thread-local: two threads failing with
/// different faults each read their own message, and a thread that only
/// succeeded keeps its empty message throughout.
#[test]
fn error_messages_are_thread_local() {
    // Establish a clean thread-local slot on the main thread.
    let clean = FaultDictionary::new(corpus());
    unsafe {
        let transducer = new_transducer(&clean);
        llev_transducer_free(transducer);
    }
    assert_eq!(last_message(), "");

    let rendezvous = Arc::new(Barrier::new(2));
    let mut workers = Vec::with_capacity(2);
    for (arm, needle) in [
        (
            (|dictionary: &FaultDictionary| {
                dictionary.fail_op(FaultOp::NodeEdges, VtStatus::IoError.to_raw())
            }) as fn(&FaultDictionary),
            "IoError",
        ),
        (
            (|dictionary: &FaultDictionary| dictionary.set_value_nonzero_reserved(true))
                as fn(&FaultDictionary),
            "reserved bytes were not zero",
        ),
    ] {
        let rendezvous = Arc::clone(&rendezvous);
        workers.push(std::thread::spawn(move || {
            let dictionary = FaultDictionary::new(corpus());
            let probe = dictionary.probe();
            unsafe {
                let transducer = new_transducer(&dictionary);
                let cursor = open_cursor(transducer, "cat");
                arm(&dictionary);
                let mut view = LlevMatchBatchView::default();
                let status = llev_query_cursor_next_batch(cursor, 8, &mut view);
                assert_ne!(status, LlevStatus::Ok);
                let message = last_message();
                assert!(
                    message.contains(needle),
                    "own-thread message {message:?} must contain {needle:?}"
                );
                // Let the other thread set ITS message, then re-read ours.
                rendezvous.wait();
                let message = last_message();
                assert!(
                    message.contains(needle),
                    "thread-local message {message:?} survived the other \
                     thread's fault"
                );
                dictionary.clear_faults();
                let _ = drain_tolerating(cursor, status);
                assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
                llev_transducer_free(transducer);
            }
            drop(dictionary);
            probe.assert_balanced();
        }));
    }
    for worker in workers {
        worker.join().expect("fault worker completes");
    }
    assert_eq!(
        last_message(),
        "",
        "the main thread's slot is untouched by other threads' faults"
    );
}

/// The traversal consumer never consults `node_transition`: the whole query
/// walk is edges-driven, which is why the garbage-node matrix row is
/// exercised through `node_edges` and why an armed transition fault is
/// inert during drains.
#[test]
fn node_transition_is_never_consulted_by_query_traversal() {
    let dictionary = FaultDictionary::new(corpus());
    let probe = dictionary.probe();
    dictionary.set_transition_garbage(true);
    dictionary.fail_op(FaultOp::NodeTransition, 42);
    unsafe {
        let transducer = new_transducer(&dictionary);
        let cursor = open_cursor(transducer, "cat");
        assert_eq!(drain_tolerating(cursor, LlevStatus::Ok), 4);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        llev_transducer_free(transducer);
    }
    assert_eq!(
        dictionary.calls(FaultOp::NodeTransition),
        0,
        "no query path may call node_transition"
    );
    drop(dictionary);
    probe.assert_balanced();
}

/// Sequential faults on one provider: every fault path releases exactly what
/// it retained, so the aggregated ledger still balances after arbitrarily
/// many fault-heal rounds.
#[test]
fn ledger_balances_across_sequential_fault_heal_rounds() {
    let dictionary = FaultDictionary::new(corpus());
    let probe = dictionary.probe();
    unsafe {
        let transducer = new_transducer(&dictionary);

        // Round 1: edges IoError.
        let cursor = open_cursor(transducer, "cat");
        dictionary.fail_op(FaultOp::NodeEdges, VtStatus::IoError.to_raw());
        let mut view = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 8, &mut view),
            LlevStatus::IoError
        );
        dictionary.clear_faults();
        let _ = drain_tolerating(cursor, LlevStatus::IoError);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);

        // Round 2: out-of-range value status.
        let cursor = open_cursor(transducer, "cat");
        dictionary.fail_op(FaultOp::NodeValueU64, u32::MAX);
        let mut view = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 8, &mut view),
            LlevStatus::ProviderError
        );
        dictionary.clear_faults();
        let _ = drain_tolerating(cursor, LlevStatus::ProviderError);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);

        // Round 3: fully healed drain.
        let cursor = open_cursor(transducer, "cat");
        assert_eq!(drain_tolerating(cursor, LlevStatus::Ok), 4);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);

        llev_transducer_free(transducer);
    }
    drop(dictionary);
    probe.assert_balanced();
}
