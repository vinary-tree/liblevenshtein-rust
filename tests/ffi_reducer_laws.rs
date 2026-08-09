//! Reducer-path laws for `llev_query_cursor_reduce`.
//!
//! Pinned semantics (from `src/ffi/index.rs`):
//! - a run over `n` matches with batch size `b` invokes the callback exactly
//!   `ceil(n / b)` times, every invocation seeing `0 < len <= b` matches,
//!   full batches until the (possibly short) last one;
//! - `out_count` counts every match DELIVERED to the callback — including
//!   the batch on which the callback returned `End` — and is written only on
//!   the `Ok` exits (run-to-end, early End stop, empty stream); it is left
//!   untouched on abort and argument errors;
//! - a callback returning `End` stops the reduction successfully; the
//!   cursor is NOT ended — the undelivered remainder stays drainable;
//! - any other VALID non-`Ok` status aborts the reduction and is returned
//!   VERBATIM (all eleven non-Ok/End statuses pinned), with the lease
//!   already auto-released, so the cursor resumes immediately;
//! - an OUT-OF-RANGE callback return (no `LlevStatus` discriminant — the
//!   reducer wire is raw u32, ledger LLEV-B16) aborts as `InvalidArgument`
//!   with the pinned decode message, never undefined behavior;
//! - reduce under a live manual lease is `BatchInUse` before any callback
//!   runs; batch size zero is `InvalidArgument`; null reducer/out_count are
//!   `NullPointer`.
//!
//! Documented-UB note (untestable): a reducer callback that REENTERS any
//! `llev_query_cursor_*` entry point on the SAME cursor violates the
//! exclusive-borrow contract of `reduce` (the cursor is `&mut` for the whole
//! call); doing so from a foreign language is undefined behavior by
//! contract, and constructing it from Rust would require aliasing a `&mut`
//! — so the law stays documented rather than executed.

#![cfg(feature = "ffi")]

mod support;

use liblevenshtein::ffi::*;
use proptest::prelude::*;
use std::ffi::{c_void, CStr};
use std::ptr;
use support::interop_dictionary::TestDictionary;

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------

struct ReduceHarness {
    transducer: *mut LlevTransducer,
    _dictionary: TestDictionary,
}

impl ReduceHarness {
    /// `count` terms all within distance 3 of the query "t".
    fn new(count: usize) -> Self {
        let dictionary = TestDictionary::new(
            (0..count).map(|index| (format!("t{index:03}"), Some(index as u64))),
        );
        let resource = dictionary.resource();
        let mut transducer = ptr::null_mut();
        let status = unsafe {
            llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer)
        };
        assert_eq!(status, LlevStatus::Ok);
        Self {
            transducer,
            _dictionary: dictionary,
        }
    }

    unsafe fn open(&self) -> *mut LlevQueryCursor {
        let mut cursor = ptr::null_mut();
        assert_eq!(
            llev_transducer_query_utf8(
                self.transducer,
                b"t".as_ptr().cast(),
                1,
                3,
                LlevQueryOrder::Traversal as u32,
                &mut cursor,
            ),
            LlevStatus::Ok
        );
        cursor
    }
}

impl Drop for ReduceHarness {
    fn drop(&mut self) {
        unsafe { llev_transducer_free(self.transducer) };
    }
}

fn last_message() -> String {
    unsafe { CStr::from_ptr(llev_last_error_message()) }
        .to_string_lossy()
        .into_owned()
}

unsafe fn drain_count(cursor: *mut LlevQueryCursor) -> usize {
    let mut total = 0usize;
    loop {
        let mut view = LlevMatchBatchView::default();
        match llev_query_cursor_next_batch(cursor, 513, &mut view) {
            LlevStatus::Ok => {
                total += view.len;
                assert_eq!(
                    llev_query_cursor_release_batch(cursor, view.generation),
                    LlevStatus::Ok
                );
            }
            LlevStatus::End => return total,
            status => panic!("drain hit {status:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Scripted callback (never panics: extern "C" unwind aborts the process).
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct ReduceProbe {
    calls: usize,
    delivered: usize,
    lens: Vec<usize>,
    /// 1-based invocation returning `stop_raw`; 0 = always Ok.
    stop_on: usize,
    stop_raw: u32,
}

impl ReduceProbe {
    fn plan(stop_on: usize, stop_raw: u32) -> Self {
        Self {
            stop_on,
            stop_raw,
            ..Self::default()
        }
    }
}

unsafe extern "C" fn scripted_reducer(
    context: *mut c_void,
    _matches: *const LlevMatch,
    len: usize,
) -> u32 {
    let probe = &mut *context.cast::<ReduceProbe>();
    probe.calls += 1;
    probe.delivered += len;
    probe.lens.push(len);
    if probe.stop_on != 0 && probe.calls == probe.stop_on {
        probe.stop_raw
    } else {
        LlevStatus::Ok as u32
    }
}

unsafe fn reduce(
    cursor: *mut LlevQueryCursor,
    batch: usize,
    probe: &mut ReduceProbe,
    out_count: &mut usize,
) -> LlevStatus {
    llev_query_cursor_reduce(
        cursor,
        batch,
        Some(scripted_reducer),
        (probe as *mut ReduceProbe).cast(),
        out_count,
    )
}

// ---------------------------------------------------------------------------
// Example anchors.
// ---------------------------------------------------------------------------

/// The batching law: exactly `ceil(n / b)` invocations, full batches until
/// the short last one, and `out_count == n`.
#[test]
fn reducer_sees_ceil_n_over_b_calls_each_nonempty_and_bounded() {
    for batch in [1usize, 3, 7, 40, 64] {
        let harness = ReduceHarness::new(40);
        unsafe {
            let cursor = harness.open();
            let mut probe = ReduceProbe::default();
            let mut out_count = usize::MAX;
            assert_eq!(
                reduce(cursor, batch, &mut probe, &mut out_count),
                LlevStatus::Ok
            );
            assert_eq!(out_count, 40, "batch {batch}");
            assert_eq!(probe.calls, 40usize.div_ceil(batch), "batch {batch}");
            assert_eq!(probe.delivered, 40);
            let full_calls = probe.calls - 1;
            for len in &probe.lens[..full_calls] {
                assert_eq!(*len, batch, "every non-final callback sees a full batch");
            }
            let last = probe.lens[full_calls];
            assert_eq!(last, 40 - batch * full_calls);
            assert!(last > 0 && last <= batch);
            assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        }
    }
}

/// Early stop: `End` from the k-th callback ends the reduction successfully,
/// `out_count` includes that k-th batch, and the cursor is NOT ended — the
/// remainder is still drainable through manual leases.
#[test]
fn early_end_stop_counts_delivered_and_the_cursor_resumes() {
    let harness = ReduceHarness::new(40);
    unsafe {
        let cursor = harness.open();
        let mut probe = ReduceProbe::plan(2, LlevStatus::End as u32);
        let mut out_count = usize::MAX;
        assert_eq!(
            reduce(cursor, 7, &mut probe, &mut out_count),
            LlevStatus::Ok
        );
        assert_eq!(probe.calls, 2);
        assert_eq!(
            out_count, 14,
            "out_count includes the batch that answered End"
        );
        assert_eq!(
            drain_count(cursor),
            26,
            "the early stop left the remainder drainable"
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// Abort: every valid non-Ok/End status — all eleven of them — is returned
/// VERBATIM; `out_count` stays unwritten, the pinned abort message is set,
/// the lease is already released, and the cursor resumes exactly where the
/// abort left it.
#[test]
fn abort_statuses_return_verbatim_release_the_lease_and_resume() {
    let aborts = [
        LlevStatus::InvalidArgument,
        LlevStatus::InvalidUtf8,
        LlevStatus::NullPointer,
        LlevStatus::Panic,
        LlevStatus::Unsupported,
        LlevStatus::IoError,
        LlevStatus::Closed,
        LlevStatus::LimitExceeded,
        LlevStatus::ProviderError,
        LlevStatus::BatchInUse,
        LlevStatus::DomainMismatch,
    ];
    for abort in aborts {
        let harness = ReduceHarness::new(40);
        unsafe {
            let cursor = harness.open();
            let mut probe = ReduceProbe::plan(1, abort as u32);
            let mut out_count = usize::MAX;
            assert_eq!(
                reduce(cursor, 7, &mut probe, &mut out_count),
                abort,
                "abort status must return verbatim"
            );
            assert_eq!(probe.calls, 1);
            assert_eq!(out_count, usize::MAX, "out_count must stay unwritten");
            let message = last_message();
            assert!(
                message.contains("batch reducer aborted the query"),
                "message {message:?}"
            );
            // The lease was auto-released before the abort surfaced: the
            // very next advance succeeds and the stream is intact minus the
            // batch already shown to the callback.
            assert_eq!(
                drain_count(cursor),
                33,
                "abort consumed exactly the one delivered batch"
            );
            assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        }
    }
}

/// The LLEV-B16 regression twin: a callback returning a raw status with no
/// `LlevStatus` discriminant aborts as `InvalidArgument` with the pinned
/// decode message — a value-level rejection of the untrusted return, never
/// an invalid-discriminant read.
#[test]
fn out_of_range_reducer_returns_abort_as_invalid_argument() {
    for raw in [13u32, 42, u32::MAX] {
        let harness = ReduceHarness::new(40);
        unsafe {
            let cursor = harness.open();
            let mut probe = ReduceProbe::plan(2, raw);
            let mut out_count = usize::MAX;
            assert_eq!(
                reduce(cursor, 7, &mut probe, &mut out_count),
                LlevStatus::InvalidArgument,
                "raw {raw}"
            );
            assert_eq!(probe.calls, 2);
            assert_eq!(out_count, usize::MAX);
            let message = last_message();
            assert!(
                message.contains("batch reducer returned an out-of-range status"),
                "message {message:?}"
            );
            // Lease released before the decode ran: the cursor resumes.
            assert_eq!(drain_count(cursor), 26, "raw {raw}");
            assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        }
    }
}

/// Reduce under a live manual lease is `BatchInUse` with zero callbacks;
/// releasing the lease unblocks the same reduce call.
#[test]
fn reduce_after_manual_lease_is_batch_in_use_until_released() {
    let harness = ReduceHarness::new(12);
    unsafe {
        let cursor = harness.open();
        let mut view = LlevMatchBatchView::default();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 5, &mut view),
            LlevStatus::Ok
        );
        let mut probe = ReduceProbe::default();
        let mut out_count = usize::MAX;
        assert_eq!(
            reduce(cursor, 4, &mut probe, &mut out_count),
            LlevStatus::BatchInUse
        );
        assert_eq!(probe.calls, 0, "no callback may run under a manual lease");
        assert_eq!(out_count, usize::MAX);
        assert_eq!(
            llev_query_cursor_release_batch(cursor, view.generation),
            LlevStatus::Ok
        );
        assert_eq!(
            reduce(cursor, 4, &mut probe, &mut out_count),
            LlevStatus::Ok
        );
        assert_eq!(out_count, 7, "the remainder after the manual lease");
        assert_eq!(probe.calls, 2);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// Batch size zero is rejected as `InvalidArgument` before any callback,
/// with `out_count` unwritten and the cursor untouched.
#[test]
fn reduce_with_zero_batch_is_invalid_argument() {
    let harness = ReduceHarness::new(6);
    unsafe {
        let cursor = harness.open();
        let mut probe = ReduceProbe::default();
        let mut out_count = usize::MAX;
        assert_eq!(
            reduce(cursor, 0, &mut probe, &mut out_count),
            LlevStatus::InvalidArgument
        );
        assert_eq!(probe.calls, 0);
        assert_eq!(out_count, usize::MAX);
        assert_eq!(drain_count(cursor), 6, "the rejection consumed nothing");
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// Null reducer and null out_count are `NullPointer` rejections.
#[test]
fn null_reducer_and_null_out_count_are_null_pointer() {
    let harness = ReduceHarness::new(3);
    unsafe {
        let cursor = harness.open();
        let mut out_count = usize::MAX;
        assert_eq!(
            llev_query_cursor_reduce(cursor, 4, None, ptr::null_mut(), &mut out_count),
            LlevStatus::NullPointer
        );
        assert_eq!(out_count, usize::MAX);
        let mut probe = ReduceProbe::default();
        assert_eq!(
            llev_query_cursor_reduce(
                cursor,
                4,
                Some(scripted_reducer),
                (&mut probe as *mut ReduceProbe).cast(),
                ptr::null_mut(),
            ),
            LlevStatus::NullPointer
        );
        assert_eq!(probe.calls, 0);
        assert_eq!(drain_count(cursor), 3);
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

// ---------------------------------------------------------------------------
// Property: arbitrary plans obey the partition + resume laws.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Plan {
    RunToEnd,
    StopWithEnd { on_call: usize },
    AbortValid { on_call: usize, status: LlevStatus },
    AbortRaw { on_call: usize, raw: u32 },
}

/// Oracle for one reduce call over `total` remaining matches.
struct Expected {
    status: LlevStatus,
    out_count: Option<usize>,
    calls: usize,
    delivered: usize,
}

fn simulate(total: usize, batch: usize, plan: &Plan) -> Expected {
    if batch == 0 {
        return Expected {
            status: LlevStatus::InvalidArgument,
            out_count: None,
            calls: 0,
            delivered: 0,
        };
    }
    let mut remaining = total;
    let mut calls = 0usize;
    let mut delivered = 0usize;
    loop {
        if remaining == 0 {
            return Expected {
                status: LlevStatus::Ok,
                out_count: Some(delivered),
                calls,
                delivered,
            };
        }
        let len = remaining.min(batch);
        remaining -= len;
        delivered += len;
        calls += 1;
        match plan {
            Plan::RunToEnd => {}
            Plan::StopWithEnd { on_call } if calls == *on_call => {
                return Expected {
                    status: LlevStatus::Ok,
                    out_count: Some(delivered),
                    calls,
                    delivered,
                };
            }
            Plan::AbortValid { on_call, status } if calls == *on_call => {
                return Expected {
                    status: *status,
                    out_count: None,
                    calls,
                    delivered,
                };
            }
            Plan::AbortRaw { on_call, .. } if calls == *on_call => {
                return Expected {
                    status: LlevStatus::InvalidArgument,
                    out_count: None,
                    calls,
                    delivered,
                };
            }
            _ => {}
        }
    }
}

fn plan_strategy() -> impl Strategy<Value = Plan> {
    prop_oneof![
        2 => Just(Plan::RunToEnd),
        1 => (1usize..5).prop_map(|on_call| Plan::StopWithEnd { on_call }),
        1 => (
            1usize..5,
            prop_oneof![
                Just(LlevStatus::IoError),
                Just(LlevStatus::Closed),
                Just(LlevStatus::LimitExceeded),
                Just(LlevStatus::Panic),
                Just(LlevStatus::DomainMismatch),
            ],
        )
            .prop_map(|(on_call, status)| Plan::AbortValid { on_call, status }),
        1 => (
            1usize..5,
            prop_oneof![Just(13u32), Just(42), Just(u32::MAX)],
        )
            .prop_map(|(on_call, raw)| Plan::AbortRaw { on_call, raw }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// For arbitrary stream sizes, batch sizes, and callback plans: the
    /// callback count, the per-invocation length bounds, the out_count
    /// write-or-not discipline, the verbatim/decode abort statuses, and the
    /// exact drainable residue all match the oracle.
    #[test]
    fn arbitrary_reduce_plans_obey_the_partition_and_resume_laws(
        total_index in 0usize..5,
        batch in prop_oneof![
            Just(0usize), Just(1), Just(2), Just(3), Just(7), Just(64), Just(300),
        ],
        plan in plan_strategy(),
    ) {
        let total = [0usize, 1, 13, 40, 120][total_index];
        let expected = simulate(total, batch, &plan);
        let harness = ReduceHarness::new(total);
        unsafe {
            let cursor = harness.open();
            let mut probe = match &plan {
                Plan::RunToEnd => ReduceProbe::default(),
                Plan::StopWithEnd { on_call } => {
                    ReduceProbe::plan(*on_call, LlevStatus::End as u32)
                }
                Plan::AbortValid { on_call, status } => {
                    ReduceProbe::plan(*on_call, *status as u32)
                }
                Plan::AbortRaw { on_call, raw } => ReduceProbe::plan(*on_call, *raw),
            };
            let mut out_count = usize::MAX;
            let status = reduce(cursor, batch, &mut probe, &mut out_count);
            prop_assert_eq!(status, expected.status);
            match expected.out_count {
                Some(count) => prop_assert_eq!(out_count, count),
                None => prop_assert_eq!(out_count, usize::MAX),
            }
            prop_assert_eq!(probe.calls, expected.calls);
            prop_assert_eq!(probe.delivered, expected.delivered);
            for len in &probe.lens {
                prop_assert!(*len > 0 && *len <= batch);
            }
            prop_assert_eq!(drain_count(cursor), total - expected.delivered);
            prop_assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        }
    }
}
