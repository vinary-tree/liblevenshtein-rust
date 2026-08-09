//! Correspondence anchor for the leased-batch cursor protocol.
//!
//! Spec: docs/verification/tla/LlevBatchLease.tla (TLC-checked with
//! LlevBatchLease.cfg). This file is the spec's companion Rust anchor: a
//! deep FSM proptest drives arbitrary operation scripts through the REAL C
//! ABI (`llev_transducer_new` over a live `TestDictionary` resource) in
//! lock-step with an oracle finite-state machine {Idle, Leased(generation),
//! Ended}, asserting the exact `LlevStatus` of every operation, plus one
//! example anchor per protocol law.
//!
//! Registry note: the seven lease invariants are TLA-homed; the registry
//! (docs/verification/ABI_INVARIANTS.tsv) is single-writer and owned by the
//! FV coordinator, who registers these rows against the hooks below.
//!
//! INVARIANT-HOOK: LLEV-LEASE-1 — AdvanceOnlyUnleased: a batch is
//! issued only from the unleased idle state
//! (`arbitrary_op_scripts_agree_with_the_lease_oracle`, and every example).
//! INVARIANT-HOOK: LLEV-LEASE-2 — BusyRejectionExact: BATCH_IN_USE
//! arises exactly from advance/reduce/free attempts under a live lease and
//! changes nothing
//! (`advance_reduce_and_free_are_rejected_under_a_live_lease`,
//! `batch_in_use_beats_invalid_argument_when_leased`).
//! INVARIANT-HOOK: LLEV-LEASE-3 — ReleaseNeedsExactGeneration:
//! stale and zero tags are INVALID_ARGUMENT and change nothing
//! (`release_requires_the_exact_live_generation`).
//! INVARIANT-HOOK: LLEV-LEASE-4 — GenerationsNeverZero: minted
//! tags are never zero and strictly advance
//! (`generations_start_at_one_strictly_increase_and_never_revisit_zero`).
//! INVARIANT-HOOK: LLEV-LEASE-5 — RejectedFreeKeepsOwnership: a
//! free under lease is refused AND the cursor stays fully usable
//! (`advance_reduce_and_free_are_rejected_under_a_live_lease`).
//! INVARIANT-HOOK: LLEV-LEASE-6 — ReduceNeverLeaks: the reducer
//! path auto-releases around every callback and never ends leased
//! (`reduce_auto_releases_between_callbacks_and_never_leaks_a_lease`).
//! INVARIANT-HOOK: LLEV-LEASE-7 — FreedIsTerminal: a successful
//! free is terminal (`free_on_idle_and_on_ended_cursors_succeeds`); the
//! not-observed-after-free half is an ownership rule whose violation is
//! use-after-free, outside the executable-test domain — the spec carries it
//! as an action property.

#![cfg(feature = "ffi")]

mod support;

use liblevenshtein::ffi::*;
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::ffi::c_void;
use std::ptr;
use support::interop_dictionary::TestDictionary;
use vinary_tree_interop::VtUnitDomain;

// ---------------------------------------------------------------------------
// Harness: a real interop provider behind the real C ABI.
// ---------------------------------------------------------------------------

/// One transducer over a `count`-term dictionary in which the query "t" at
/// distance 3 matches EVERY term (`t000`..), so the oracle's `remaining`
/// starts at exactly `count`.
struct CursorHarness {
    // Field order pins drop order: the transducer wrapper drops first, then
    // the dictionary releases the provider.
    transducer: *mut LlevTransducer,
    _dictionary: TestDictionary,
}

impl CursorHarness {
    fn new(count: usize) -> Self {
        let dictionary = TestDictionary::new(
            (0..count).map(|index| (format!("t{index:03}"), Some(index as u64))),
        );
        let resource = dictionary.resource();
        let mut transducer = ptr::null_mut();
        let status = unsafe {
            llev_transducer_new(&resource, LlevAlgorithm::Standard as u32, &mut transducer)
        };
        assert_eq!(status, LlevStatus::Ok, "harness transducer must construct");
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
            LlevStatus::Ok,
            "harness query must start"
        );
        cursor
    }

    /// Independent count of the query's total match volume.
    unsafe fn total(&self) -> usize {
        let cursor = self.open();
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
                LlevStatus::End => break,
                status => panic!("predrain hit {status:?}"),
            }
        }
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
        total
    }
}

impl Drop for CursorHarness {
    fn drop(&mut self) {
        unsafe { llev_transducer_free(self.transducer) };
    }
}

/// A sentinel-poisoned view: any non-Ok advance must overwrite every field
/// with the published default (null matches, zero length, zero generation).
fn poisoned_view() -> LlevMatchBatchView {
    LlevMatchBatchView {
        matches: std::ptr::NonNull::<LlevMatch>::dangling().as_ptr(),
        len: usize::MAX,
        generation: u64::MAX,
    }
}

fn assert_view_zeroed(view: &LlevMatchBatchView) {
    assert!(
        view.matches.is_null(),
        "non-Ok advance must null the matches"
    );
    assert_eq!(view.len, 0, "non-Ok advance must zero the length");
    assert_eq!(
        view.generation, 0,
        "non-Ok advance must zero the generation"
    );
}

// ---------------------------------------------------------------------------
// Scripted reducer callback (must never panic: extern "C" unwind aborts).
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct ReduceProbe {
    calls: usize,
    delivered: usize,
    lens: Vec<usize>,
    /// 1-based callback index returning `stop_raw`; 0 disables the plan.
    stop_on: usize,
    stop_raw: u32,
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

// ---------------------------------------------------------------------------
// Oracle FSM.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeaseState {
    Idle,
    Leased,
    Ended,
}

#[derive(Clone, Copy, Debug)]
struct Model {
    state: LeaseState,
    generation: u64,
    remaining: usize,
}

/// Everything the oracle predicts about one `Reduce` call.
#[derive(Debug)]
struct ReduceOutcome {
    status: LlevStatus,
    /// `Some(count)` exactly when the ABI writes `out_count`.
    out_count: Option<usize>,
    calls: usize,
    delivered: usize,
}

impl Model {
    fn new(remaining: usize) -> Self {
        Self {
            state: LeaseState::Idle,
            generation: 0,
            remaining,
        }
    }

    /// Oracle for `llev_query_cursor_next_batch`: status and delivered length.
    fn next_batch(&mut self, max: usize) -> (LlevStatus, usize) {
        if self.state == LeaseState::Leased {
            // The lease check precedes the empty-batch check.
            return (LlevStatus::BatchInUse, 0);
        }
        if max == 0 {
            return (LlevStatus::InvalidArgument, 0);
        }
        if self.remaining == 0 {
            self.state = LeaseState::Ended;
            return (LlevStatus::End, 0);
        }
        let len = self.remaining.min(max);
        self.remaining -= len;
        self.generation += 1;
        self.state = LeaseState::Leased;
        (LlevStatus::Ok, len)
    }

    fn release_live(&mut self) -> LlevStatus {
        if self.state == LeaseState::Leased {
            self.state = LeaseState::Idle;
            LlevStatus::Ok
        } else {
            LlevStatus::InvalidArgument
        }
    }

    /// Any tag other than the live one — stale, zero, or future — is
    /// rejected without a state change.
    fn release_mismatched(&self) -> LlevStatus {
        LlevStatus::InvalidArgument
    }

    fn reduce(&mut self, batch: usize, plan: &ReducePlan) -> ReduceOutcome {
        if self.state == LeaseState::Leased {
            return ReduceOutcome {
                status: LlevStatus::BatchInUse,
                out_count: None,
                calls: 0,
                delivered: 0,
            };
        }
        if batch == 0 {
            // The first internal fill rejects the empty batch even when the
            // stream is already exhausted.
            return ReduceOutcome {
                status: LlevStatus::InvalidArgument,
                out_count: None,
                calls: 0,
                delivered: 0,
            };
        }
        let mut calls = 0usize;
        let mut delivered = 0usize;
        loop {
            if self.remaining == 0 {
                self.state = LeaseState::Ended;
                return ReduceOutcome {
                    status: LlevStatus::Ok,
                    out_count: Some(delivered),
                    calls,
                    delivered,
                };
            }
            let len = self.remaining.min(batch);
            self.remaining -= len;
            self.generation += 1;
            delivered += len;
            calls += 1;
            match plan {
                ReducePlan::RunToEnd => {}
                ReducePlan::StopWithEnd { on_call } if calls == *on_call => {
                    self.state = LeaseState::Idle;
                    return ReduceOutcome {
                        status: LlevStatus::Ok,
                        out_count: Some(delivered),
                        calls,
                        delivered,
                    };
                }
                ReducePlan::Abort { on_call, status } if calls == *on_call => {
                    self.state = LeaseState::Idle;
                    return ReduceOutcome {
                        status: *status,
                        out_count: None,
                        calls,
                        delivered,
                    };
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Operation scripts.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum ReducePlan {
    RunToEnd,
    StopWithEnd { on_call: usize },
    Abort { on_call: usize, status: LlevStatus },
}

#[derive(Clone, Debug)]
enum Op {
    NextBatch { max: usize },
    ReleaseLive,
    ReleaseStale,
    ReleaseZero,
    Reduce { batch: usize, plan: ReducePlan },
    TryFree,
    UnitDomainProbe,
}

fn reduce_strategy() -> impl Strategy<Value = Op> {
    (
        prop_oneof![Just(0usize), Just(1), Just(7), Just(64), Just(256)],
        prop_oneof![
            2 => Just(ReducePlan::RunToEnd),
            1 => (1usize..4).prop_map(|on_call| ReducePlan::StopWithEnd { on_call }),
            1 => (
                1usize..4,
                prop_oneof![
                    Just(LlevStatus::IoError),
                    Just(LlevStatus::Closed),
                    Just(LlevStatus::LimitExceeded),
                    Just(LlevStatus::DomainMismatch),
                ],
            )
                .prop_map(|(on_call, status)| ReducePlan::Abort { on_call, status }),
        ],
    )
        .prop_map(|(batch, plan)| Op::Reduce { batch, plan })
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => prop_oneof![
            Just(0usize),
            Just(1),
            Just(255),
            Just(256),
            Just(257),
            Just(513),
        ]
        .prop_map(|max| Op::NextBatch { max }),
        4 => Just(Op::ReleaseLive),
        1 => Just(Op::ReleaseStale),
        1 => Just(Op::ReleaseZero),
        2 => reduce_strategy(),
        1 => Just(Op::TryFree),
        1 => Just(Op::UnitDomainProbe),
    ]
}

/// Execute one script against a fresh cursor, asserting oracle agreement at
/// every step, then prove the cursor still drains exactly the oracle's
/// residue and closes cleanly.
fn run_script(count: usize, ops: &[Op]) {
    let harness = CursorHarness::new(count);
    let total = unsafe { harness.total() };
    assert_eq!(total, count, "every tNNN term lies within distance 3 of t");

    let cursor = unsafe { harness.open() };
    let mut model = Model::new(total);
    let mut seen_terms: BTreeSet<String> = BTreeSet::new();

    for op in ops {
        match op {
            Op::NextBatch { max } => {
                let (expected, expected_len) = model.next_batch(*max);
                let mut view = poisoned_view();
                let status = unsafe { llev_query_cursor_next_batch(cursor, *max, &mut view) };
                assert_eq!(status, expected, "NextBatch({max}) status; model {model:?}");
                if expected == LlevStatus::Ok {
                    assert_eq!(view.len, expected_len, "delivered length");
                    assert!(!view.matches.is_null());
                    assert_eq!(view.generation, model.generation, "minted tag");
                    assert_ne!(view.generation, 0, "a live lease tag is never zero");
                    let items = unsafe { std::slice::from_raw_parts(view.matches, view.len) };
                    for item in items {
                        assert_eq!(item.unit_domain, VtUnitDomain::UnicodeScalar);
                        assert_eq!(item.has_id, 1, "every harness term carries an id");
                        let bytes = unsafe {
                            std::slice::from_raw_parts(item.term_data.cast::<u8>(), item.byte_len)
                        };
                        let term = std::str::from_utf8(bytes).expect("terms are UTF-8");
                        assert!(
                            seen_terms.insert(term.to_owned()),
                            "term {term:?} delivered twice"
                        );
                    }
                } else {
                    assert_view_zeroed(&view);
                }
            }
            Op::ReleaseLive => {
                let expected = model.release_live();
                let status = unsafe { llev_query_cursor_release_batch(cursor, model.generation) };
                assert_eq!(status, expected, "ReleaseLive; model {model:?}");
            }
            Op::ReleaseStale => {
                let expected = model.release_mismatched();
                let stale = model.generation.wrapping_sub(1);
                let status = unsafe { llev_query_cursor_release_batch(cursor, stale) };
                assert_eq!(status, expected, "ReleaseStale({stale}); model {model:?}");
            }
            Op::ReleaseZero => {
                let expected = model.release_mismatched();
                let status = unsafe { llev_query_cursor_release_batch(cursor, 0) };
                assert_eq!(status, expected, "ReleaseZero; model {model:?}");
            }
            Op::Reduce { batch, plan } => {
                let outcome = model.reduce(*batch, plan);
                let mut probe = ReduceProbe::default();
                if outcome.calls > 0 {
                    probe.lens.reserve(outcome.calls);
                }
                match plan {
                    ReducePlan::RunToEnd => {}
                    ReducePlan::StopWithEnd { on_call } => {
                        probe.stop_on = *on_call;
                        probe.stop_raw = LlevStatus::End as u32;
                    }
                    ReducePlan::Abort { on_call, status } => {
                        probe.stop_on = *on_call;
                        probe.stop_raw = *status as u32;
                    }
                }
                let mut out_count = usize::MAX;
                let status = unsafe {
                    llev_query_cursor_reduce(
                        cursor,
                        *batch,
                        Some(scripted_reducer),
                        (&mut probe as *mut ReduceProbe).cast(),
                        &mut out_count,
                    )
                };
                assert_eq!(status, outcome.status, "Reduce({batch}, {plan:?})");
                match outcome.out_count {
                    Some(expected_count) => assert_eq!(out_count, expected_count, "out_count"),
                    None => assert_eq!(out_count, usize::MAX, "out_count must stay unwritten"),
                }
                assert_eq!(probe.calls, outcome.calls, "callback invocations");
                assert_eq!(
                    probe.delivered, outcome.delivered,
                    "matches shown to callbacks"
                );
                for len in &probe.lens {
                    assert!(*len > 0 && *len <= *batch, "callback length law: {len}");
                }
            }
            Op::TryFree => {
                if model.state == LeaseState::Leased {
                    // LLEV-LEASE-5: refused, and ownership stays with us —
                    // every later step keeps operating on the same pointer.
                    let status = unsafe { llev_query_cursor_free(cursor) };
                    assert_eq!(status, LlevStatus::BatchInUse, "free under lease");
                }
                // A successful free would consume the cursor; the post-script
                // epilogue exercises that path exactly once per script.
            }
            Op::UnitDomainProbe => {
                let mut domain = VtUnitDomain::Byte;
                let status =
                    unsafe { llev_transducer_unit_domain(harness.transducer, &mut domain) };
                assert_eq!(status, LlevStatus::Ok);
                assert_eq!(domain, VtUnitDomain::UnicodeScalar);
            }
        }
    }

    // Epilogue: the oracle's residue is exactly drainable, then free is Ok.
    if model.state == LeaseState::Leased {
        assert_eq!(
            unsafe { llev_query_cursor_release_batch(cursor, model.generation) },
            LlevStatus::Ok
        );
        model.state = LeaseState::Idle;
    }
    let mut residue = 0usize;
    loop {
        let mut view = poisoned_view();
        match unsafe { llev_query_cursor_next_batch(cursor, 513, &mut view) } {
            LlevStatus::Ok => {
                residue += view.len;
                assert_eq!(
                    unsafe { llev_query_cursor_release_batch(cursor, view.generation) },
                    LlevStatus::Ok
                );
            }
            LlevStatus::End => break,
            status => panic!("epilogue drain hit {status:?}"),
        }
        assert!(
            residue <= model.remaining,
            "drained past the oracle residue"
        );
    }
    assert_eq!(residue, model.remaining, "oracle residue");
    assert_eq!(unsafe { llev_query_cursor_free(cursor) }, LlevStatus::Ok);
}

// ---------------------------------------------------------------------------
// Example anchors, one per protocol law.
// ---------------------------------------------------------------------------

/// LLEV-LEASE-4 face: tags mint at 1, advance strictly by one per issued
/// batch, and zero is never a live tag.
#[test]
fn generations_start_at_one_strictly_increase_and_never_revisit_zero() {
    let harness = CursorHarness::new(5);
    unsafe {
        let cursor = harness.open();
        for expected_generation in 1u64..=5 {
            let mut view = poisoned_view();
            assert_eq!(
                llev_query_cursor_next_batch(cursor, 1, &mut view),
                LlevStatus::Ok
            );
            assert_eq!(view.generation, expected_generation);
            assert_ne!(view.generation, 0);
            assert_eq!(
                llev_query_cursor_release_batch(cursor, view.generation),
                LlevStatus::Ok
            );
        }
        let mut view = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut view),
            LlevStatus::End
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// LLEV-LEASE-1/2/5/6 faces: under a live lease, advancing, reducing, and
/// freeing are all BATCH_IN_USE; none of the rejections damages the cursor,
/// which then finishes its stream exactly.
#[test]
fn advance_reduce_and_free_are_rejected_under_a_live_lease() {
    let harness = CursorHarness::new(4);
    unsafe {
        let cursor = harness.open();
        let mut view = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut view),
            LlevStatus::Ok
        );
        let live = view.generation;

        let mut blocked = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut blocked),
            LlevStatus::BatchInUse
        );
        assert_view_zeroed(&blocked);

        let mut probe = ReduceProbe::default();
        let mut out_count = usize::MAX;
        assert_eq!(
            llev_query_cursor_reduce(
                cursor,
                2,
                Some(scripted_reducer),
                (&mut probe as *mut ReduceProbe).cast(),
                &mut out_count,
            ),
            LlevStatus::BatchInUse
        );
        assert_eq!(probe.calls, 0, "no callback may run under a foreign lease");
        assert_eq!(out_count, usize::MAX, "out_count stays unwritten");

        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::BatchInUse);

        // Fully usable afterward: release, then drain the exact remainder.
        assert_eq!(
            llev_query_cursor_release_batch(cursor, live),
            LlevStatus::Ok
        );
        let mut rest = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 513, &mut rest),
            LlevStatus::Ok
        );
        assert_eq!(rest.len, 3, "the rejected operations consumed nothing");
        assert_eq!(
            llev_query_cursor_release_batch(cursor, rest.generation),
            LlevStatus::Ok
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// LLEV-LEASE-3 face: zero, stale, and future tags never release; only the
/// exact live generation does.
#[test]
fn release_requires_the_exact_live_generation() {
    let harness = CursorHarness::new(3);
    unsafe {
        let cursor = harness.open();
        let mut view = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 2, &mut view),
            LlevStatus::Ok
        );
        let live = view.generation;
        for wrong in [0u64, live.wrapping_sub(1), live + 1] {
            assert_eq!(
                llev_query_cursor_release_batch(cursor, wrong),
                LlevStatus::InvalidArgument,
                "tag {wrong} must not release live tag {live}"
            );
            let mut still = poisoned_view();
            assert_eq!(
                llev_query_cursor_next_batch(cursor, 1, &mut still),
                LlevStatus::BatchInUse,
                "the lease survived the mismatched tag {wrong}"
            );
        }
        assert_eq!(
            llev_query_cursor_release_batch(cursor, live),
            LlevStatus::Ok
        );
        assert_eq!(
            llev_query_cursor_release_batch(cursor, live),
            LlevStatus::InvalidArgument,
            "a released tag cannot release twice"
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// Ended is absorbing for advances, rejects releases, and reduces to zero
/// callbacks with a written zero count.
#[test]
fn ended_cursors_report_end_forever_and_reduce_without_callbacks() {
    let harness = CursorHarness::new(2);
    unsafe {
        let cursor = harness.open();
        let mut view = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 513, &mut view),
            LlevStatus::Ok
        );
        assert_eq!(view.len, 2);
        assert_eq!(
            llev_query_cursor_release_batch(cursor, view.generation),
            LlevStatus::Ok
        );
        for _ in 0..2 {
            let mut ended = poisoned_view();
            assert_eq!(
                llev_query_cursor_next_batch(cursor, 8, &mut ended),
                LlevStatus::End
            );
            assert_view_zeroed(&ended);
        }
        assert_eq!(
            llev_query_cursor_release_batch(cursor, view.generation),
            LlevStatus::InvalidArgument,
            "nothing is leased after End"
        );
        let mut probe = ReduceProbe::default();
        let mut out_count = usize::MAX;
        assert_eq!(
            llev_query_cursor_reduce(
                cursor,
                4,
                Some(scripted_reducer),
                (&mut probe as *mut ReduceProbe).cast(),
                &mut out_count,
            ),
            LlevStatus::Ok
        );
        assert_eq!(probe.calls, 0, "an ended stream produces no callbacks");
        assert_eq!(out_count, 0, "and writes a zero count");
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// LLEV-LEASE-6 face: a completed reduce leaves NO lease behind — the next
/// advance reports End instead of BATCH_IN_USE, and a second reduce is a
/// clean zero-callback success.
#[test]
fn reduce_auto_releases_between_callbacks_and_never_leaks_a_lease() {
    let harness = CursorHarness::new(40);
    unsafe {
        let cursor = harness.open();
        let mut probe = ReduceProbe::default();
        let mut out_count = usize::MAX;
        assert_eq!(
            llev_query_cursor_reduce(
                cursor,
                7,
                Some(scripted_reducer),
                (&mut probe as *mut ReduceProbe).cast(),
                &mut out_count,
            ),
            LlevStatus::Ok
        );
        assert_eq!(out_count, 40);
        assert_eq!(probe.calls, 40usize.div_ceil(7));
        let mut view = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 8, &mut view),
            LlevStatus::End,
            "no lease survives a completed reduce"
        );
        let mut again = ReduceProbe::default();
        let mut count_again = usize::MAX;
        assert_eq!(
            llev_query_cursor_reduce(
                cursor,
                7,
                Some(scripted_reducer),
                (&mut again as *mut ReduceProbe).cast(),
                &mut count_again,
            ),
            LlevStatus::Ok
        );
        assert_eq!((again.calls, count_again), (0, 0));
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// Check-order pin: the lease guard precedes the empty-batch guard, so a
/// zero `max_matches` is BATCH_IN_USE while leased and INVALID_ARGUMENT
/// otherwise (including on an ended cursor).
#[test]
fn batch_in_use_beats_invalid_argument_when_leased() {
    let harness = CursorHarness::new(2);
    unsafe {
        let cursor = harness.open();
        let mut view = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 0, &mut view),
            LlevStatus::InvalidArgument,
            "idle + zero max"
        );
        assert_view_zeroed(&view);
        let mut lease = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 1, &mut lease),
            LlevStatus::Ok
        );
        let mut zeroed = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 0, &mut zeroed),
            LlevStatus::BatchInUse,
            "leased + zero max: the lease guard wins"
        );
        assert_view_zeroed(&zeroed);
        assert_eq!(
            llev_query_cursor_release_batch(cursor, lease.generation),
            LlevStatus::Ok
        );
        let mut drained = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 513, &mut drained),
            LlevStatus::Ok
        );
        assert_eq!(
            llev_query_cursor_release_batch(cursor, drained.generation),
            LlevStatus::Ok
        );
        let mut ended = poisoned_view();
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 8, &mut ended),
            LlevStatus::End
        );
        assert_eq!(
            llev_query_cursor_next_batch(cursor, 0, &mut ended),
            LlevStatus::InvalidArgument,
            "ended + zero max"
        );
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);
    }
}

/// LLEV-LEASE-7 executable face: free succeeds exactly once from Idle and
/// from Ended; beyond that the terminal law is the ownership rule (a freed
/// cursor must never be passed again).
#[test]
fn free_on_idle_and_on_ended_cursors_succeeds() {
    let harness = CursorHarness::new(3);
    unsafe {
        // Freshly opened (Idle): free is immediate.
        let cursor = harness.open();
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);

        // Drained (Ended): free is equally clean.
        let cursor = harness.open();
        loop {
            let mut view = poisoned_view();
            match llev_query_cursor_next_batch(cursor, 513, &mut view) {
                LlevStatus::Ok => assert_eq!(
                    llev_query_cursor_release_batch(cursor, view.generation),
                    LlevStatus::Ok
                ),
                LlevStatus::End => break,
                status => panic!("unexpected {status:?}"),
            }
        }
        assert_eq!(llev_query_cursor_free(cursor), LlevStatus::Ok);

        // A null cursor free is the documented no-op success.
        assert_eq!(llev_query_cursor_free(ptr::null_mut()), LlevStatus::Ok);
    }
}

// ---------------------------------------------------------------------------
// The deep FSM proptest.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Arbitrary operation scripts over stream sizes {0, 1, 40, 300} agree
    /// with the oracle FSM on every status, every delivered length, every
    /// minted generation, every reducer invocation count, and the exact
    /// residue left in the stream.
    #[test]
    fn arbitrary_op_scripts_agree_with_the_lease_oracle(
        count_index in 0usize..4,
        ops in prop::collection::vec(op_strategy(), 1..24),
    ) {
        let count = [0usize, 1, 40, 300][count_index];
        run_script(count, &ops);
    }
}
