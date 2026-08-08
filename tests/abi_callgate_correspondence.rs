//! Correspondence anchor for the call-gate serialization model.
//!
//! Spec: docs/verification/tla/LlevCallGate.tla (TLC-checked; see
//! docs/verification/ABI_INVARIANTS.tsv).
//!
//! The gate's domain is ONE CAPTURED OBJECT (VT-GATE-1). In the real
//! consumer, every query captures its own snapshot object, so the one object
//! genuinely shared between concurrent queries is the BASE provider — whose
//! `snapshot()` callback runs at every query start. The provider therefore
//! records concurrency separately for `snapshot()` (base-object gate
//! observable, VT-GATE-2) and `node_edges` (aggregated across snapshot
//! objects — legally concurrent even for a serialized provider, the
//! object-domain face of VT-GATE-1). The first version of this file
//! aggregated all callbacks into one counter and "refuted" the gate; the
//! refutation was a test-model error the TLA+ model does not make, and the
//! per-object instrumentation below is the correction.
//!
//! INVARIANT-HOOK: VT-GATE-1 — serialization never crosses object or
//! provider boundaries (tests 3 and 4).
//! INVARIANT-HOOK: VT-GATE-2 — a non-reentrant object never observes two
//! concurrent callbacks (test 1); a PARALLEL_REENTRANT object may (test 2).
//! INVARIANT-HOOK: VT-GATE-3 — gate release on the error exit path is
//! modeled in the spec (ExitError); its Rust adversarial form lands with the
//! wave-W3 fault-injection provider and extends THIS file (reconciliation
//! rule: one file per protocol).

#![cfg(feature = "bindings-core")]

mod support;

use std::sync::Arc;
use std::time::Duration;

use liblevenshtein::bindings::{MatchBatch, QueryOrder, ResourceTransducer, DEFAULT_MATCH_BATCH};
use liblevenshtein::transducer::Algorithm;
use support::interop_dictionary::TestDictionary;

fn terms() -> Vec<(String, Option<u64>)> {
    // A fanned-out dictionary so every query crosses node_edges repeatedly.
    let mut entries = Vec::with_capacity(26 * 3);
    for a in 'a'..='z' {
        for suffix in ["at", "ot", "ut"] {
            entries.push((format!("{a}{suffix}"), Some(u64::from(a as u32))));
        }
    }
    entries
}

fn drain(transducer: &ResourceTransducer, query: &str) -> usize {
    let mut cursor = transducer
        .query_utf8(query, 2, QueryOrder::Traversal)
        .expect("query starts");
    let mut batch = MatchBatch::default();
    let mut count = 0usize;
    loop {
        let filled = cursor
            .next_batch(&mut batch, DEFAULT_MATCH_BATCH)
            .expect("provider never faults in this test");
        if filled == 0 {
            break;
        }
        count += filled;
    }
    count
}

fn wait_until(deadline: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if condition() {
            return true;
        }
        std::thread::yield_now();
    }
    condition()
}

/// VT-GATE-2: the base provider (the one object every concurrent query
/// shares) never observes two `snapshot()` callbacks at once through
/// CallGate::Serial, under a 72-query-start three-thread race.
#[test]
fn serial_query_starts_never_overlap_on_the_base_provider() {
    let dictionary = TestDictionary::new(terms());
    let transducer = Arc::new(
        unsafe { ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard) }
            .expect("resource accepted"),
    );

    let mut workers = Vec::with_capacity(3);
    for query in ["cat", "cot", "cut"] {
        let transducer = Arc::clone(&transducer);
        workers.push(std::thread::spawn(move || {
            let mut total = 0usize;
            for _ in 0..24 {
                total += drain(&transducer, query);
            }
            total
        }));
    }
    for worker in workers {
        let matches = worker.join().expect("worker thread completes");
        assert!(matches > 0, "every query round yields matches");
    }

    assert_eq!(
        dictionary.snapshot_calls(),
        72,
        "every query start captured exactly one snapshot"
    );
    assert!(
        dictionary.edge_batch_calls() > 0,
        "the stress actually crossed node_edges"
    );
    assert_eq!(
        dictionary.max_snapshots_in_flight(),
        1,
        "CallGate::Serial admitted two concurrent callbacks into the \
         non-reentrant base provider"
    );
}

/// Reentrancy honored: a provider claiming PARALLEL_REENTRANT provably hosts
/// two threads inside `snapshot()` simultaneously (deterministically, via the
/// provider-side hold — no timing assumptions).
#[test]
fn reentrant_provider_admits_concurrent_query_starts() {
    let dictionary = TestDictionary::with_reentrancy(terms(), true);
    let transducer = Arc::new(
        unsafe { ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard) }
            .expect("resource accepted"),
    );

    dictionary.pause_snapshots();
    let mut parked = Vec::with_capacity(2);
    for query in ["cat", "cot"] {
        let transducer = Arc::clone(&transducer);
        parked.push(std::thread::spawn(move || drain(&transducer, query)));
    }
    assert!(
        wait_until(Duration::from_secs(30), || dictionary.snapshots_in_flight()
            == 2),
        "both query starts should be inside snapshot() (in flight: {})",
        dictionary.snapshots_in_flight()
    );
    dictionary.resume_snapshots();
    for worker in parked {
        assert!(worker.join().expect("worker completes") > 0);
    }
    assert!(
        dictionary.max_snapshots_in_flight() >= 2,
        "PARALLEL_REENTRANT providers must be allowed concurrent callbacks"
    );
}

/// VT-GATE-1 (object-domain face): even with a SERIALIZED provider, two
/// queries traverse their own captured snapshot objects concurrently — the
/// gate serializes one object, never the family of objects a provider spawns.
#[test]
fn distinct_captured_objects_do_not_share_a_gate() {
    let dictionary = TestDictionary::new(terms());
    let transducer = Arc::new(
        unsafe { ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard) }
            .expect("resource accepted"),
    );

    dictionary.pause_edges();
    let mut parked = Vec::with_capacity(2);
    for query in ["cat", "cot"] {
        let transducer = Arc::clone(&transducer);
        parked.push(std::thread::spawn(move || drain(&transducer, query)));
    }
    // Each thread finishes its (serialized) query start, then parks inside
    // node_edges of its OWN snapshot object; both being parked at once is
    // exactly the cross-object concurrency VT-GATE-1 guarantees.
    assert!(
        wait_until(Duration::from_secs(30), || dictionary.edges_in_flight()
            == 2),
        "both threads should be inside their own snapshot's node_edges \
         (in flight: {})",
        dictionary.edges_in_flight()
    );
    dictionary.resume_edges();
    for worker in parked {
        assert!(worker.join().expect("worker completes") > 0);
    }
    assert!(dictionary.max_edges_in_flight() >= 2);
    assert_eq!(
        dictionary.max_snapshots_in_flight(),
        1,
        "the base object itself stayed serialized throughout"
    );
}

/// VT-GATE-1 (provider-domain face): a thread parked inside provider A's
/// callback does not serialize an independent query on provider B. The B
/// query COMPLETES while A is provably parked — an order-based, timing-free
/// witness of independence.
#[test]
fn gate_domain_is_per_provider_not_global() {
    let provider_a = TestDictionary::new(terms());
    let provider_b = TestDictionary::new(terms());
    let transducer_a = Arc::new(
        unsafe { ResourceTransducer::from_resource(provider_a.resource(), Algorithm::Standard) }
            .expect("resource A accepted"),
    );
    let transducer_b =
        unsafe { ResourceTransducer::from_resource(provider_b.resource(), Algorithm::Standard) }
            .expect("resource B accepted");

    provider_a.pause_edges();
    let parked = {
        let transducer_a = Arc::clone(&transducer_a);
        std::thread::spawn(move || drain(&transducer_a, "cat"))
    };
    assert!(
        wait_until(Duration::from_secs(30), || provider_a.edges_in_flight()
            == 1),
        "the A thread should be parked inside provider A's callback"
    );

    // With A's callback parked, B must run to completion.
    let matches_b = drain(&transducer_b, "cot");
    assert!(
        matches_b > 0,
        "provider B query completed while A was parked"
    );
    assert_eq!(
        provider_a.edges_in_flight(),
        1,
        "the A thread is still parked — B never waited on anything of A's"
    );

    provider_a.resume_edges();
    assert!(parked.join().expect("A thread completes") > 0);
    assert_eq!(provider_b.max_snapshots_in_flight(), 1);
}
