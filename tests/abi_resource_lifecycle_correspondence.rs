//! Correspondence anchor for the resource-lifecycle protocol model.
//!
//! Spec: docs/verification/tla/AbiResourceLifecycle.tla (TLC-checked; see
//! docs/verification/ABI_INVARIANTS.tsv). Every test here mirrors a named
//! model invariant against the REAL counting provider in
//! tests/support/interop_dictionary.rs, which implements the interop vtable
//! with genuine `Arc` reference counting plus an observable ledger
//! (retain/release calls and context destructions).
//!
//! INVARIANT-HOOK: VT-LIFE-1 — provider counter == outstanding owned retains
//! (ledger balance: releases == retains + births at full teardown).
//! INVARIANT-HOOK: VT-LIFE-2 — callbacks only under a live retain.
//! INVARIANT-HOOK: VT-LIFE-3 — the ledger never goes negative.
//! INVARIANT-HOOK: VT-LIFE-4 — copying the two words never moves the counter.
//! INVARIANT-HOOK: VT-LIFE-5 — every context reaches destruction exactly once.
//! INVARIANT-HOOK: VT-LIFE-6 — destruction is final: drops == births, never more.
//! INVARIANT-HOOK: VT-QI-1..3 — negotiation surface (pinned crate-side in
//! vinary-tree-interop/tests/vtable_evolution.rs; exercised here live).

#![cfg(feature = "bindings-core")]

mod support;

use std::ffi::c_void;

use proptest::prelude::*;
use support::interop_dictionary::TestDictionary;
use vinary_tree_interop::{
    VtDictionaryVTable, VtResource, VtStatus, VT_DICTIONARY_INTERFACE_ID,
    VT_DICTIONARY_INTERFACE_VERSION,
};

/// Raw vtable retain on a resource the test owns a reference to.
unsafe fn vt_retain(resource: VtResource) {
    let vtable = &*resource.vtable;
    (vtable.retain.expect("provider publishes retain"))(resource.context);
}

/// Raw vtable release of one owned retain.
unsafe fn vt_release(resource: VtResource) {
    let vtable = &*resource.vtable;
    (vtable.release.expect("provider publishes release"))(resource.context);
}

/// Negotiate the dictionary interface and read the root node — one full
/// "callback under a live retain" probe (VT-LIFE-2 discipline).
unsafe fn vt_probe_root(resource: VtResource) -> VtStatus {
    let vtable = &*resource.vtable;
    let mut dictionary: *const c_void = std::ptr::null();
    let raw = (vtable.query_interface.expect("query_interface published"))(
        resource.context,
        &VT_DICTIONARY_INTERFACE_ID,
        VT_DICTIONARY_INTERFACE_VERSION,
        &mut dictionary,
    );
    let status = VtStatus::from_raw(raw).expect("provider returned an out-of-range status");
    if status != VtStatus::Ok {
        return status;
    }
    let dictionary = &*(dictionary as *const VtDictionaryVTable);
    let mut root = 0u64;
    let raw = (dictionary.root.expect("root published"))(resource.context, &mut root);
    VtStatus::from_raw(raw).expect("provider returned an out-of-range status")
}

/// Capture a snapshot resource; the caller owns the returned retain.
unsafe fn vt_snapshot(resource: VtResource) -> VtResource {
    let vtable = &*resource.vtable;
    let mut dictionary: *const c_void = std::ptr::null();
    let raw = (vtable.query_interface.expect("query_interface published"))(
        resource.context,
        &VT_DICTIONARY_INTERFACE_ID,
        VT_DICTIONARY_INTERFACE_VERSION,
        &mut dictionary,
    );
    let status = VtStatus::from_raw(raw).expect("provider returned an out-of-range status");
    assert_eq!(status, VtStatus::Ok, "snapshot negotiation must succeed");
    let dictionary = &*(dictionary as *const VtDictionaryVTable);
    let mut captured = VtResource::NULL;
    let raw = (dictionary.snapshot.expect("snapshot published"))(resource.context, &mut captured);
    let status = VtStatus::from_raw(raw).expect("provider returned an out-of-range status");
    assert_eq!(status, VtStatus::Ok, "snapshot capture must succeed");
    assert!(!captured.is_null(), "captured snapshot must be non-null");
    captured
}

/// One scripted lifecycle operation over a slot pool (slot 0 = the root
/// resource owned by the provider handle; slots 1.. = captured snapshots the
/// script owns).
#[derive(Clone, Debug)]
enum LifecycleOp {
    /// Add one owned retain to a live slot.
    Retain(usize),
    /// Release one of the script's own retains on a live slot.
    Release(usize),
    /// Capture a snapshot from a live slot into a new slot.
    Snapshot(usize),
    /// Negotiate + read the root under a live retain.
    Probe(usize),
    /// Bit-copy the two words and drop the copy (must not move the ledger).
    CopyWords(usize),
}

fn scripts() -> impl Strategy<Value = Vec<LifecycleOp>> {
    let op = prop_oneof![
        (0usize..5).prop_map(LifecycleOp::Retain),
        (0usize..5).prop_map(LifecycleOp::Release),
        (0usize..5).prop_map(LifecycleOp::Snapshot),
        (0usize..5).prop_map(LifecycleOp::Probe),
        (0usize..5).prop_map(LifecycleOp::CopyWords),
    ];
    proptest::collection::vec(op, 1..64)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// VT-LIFE-1/3/4/5/6: random correct-consumer lifecycle scripts keep the
    /// provider ledger balanced, never negative, unmoved by raw copies, and
    /// drain every context to exactly one destruction at teardown.
    #[test]
    fn ledger_balances_under_random_lifecycle_scripts(script in scripts()) {
        let dictionary = TestDictionary::new([
            ("cat".to_string(), Some(1)),
            ("cot".to_string(), Some(2)),
            ("cut".to_string(), Some(3)),
            ("scat".to_string(), None),
        ]);
        let probe = dictionary.metrics_probe();

        // Slot pool: (resource, my_owned_retains, alive). Slot 0 is the root:
        // the provider handle owns its birth retain, the script owns zero.
        let mut slots: Vec<(VtResource, usize, bool)> =
            vec![(dictionary.resource(), 0usize, true)];
        let mut my_retains = 0usize;
        let mut my_releases = 0usize;
        let mut births = 1usize; // the root context

        for op in script {
            match op {
                LifecycleOp::Retain(index) => {
                    if let Some(&(resource, owned, alive)) = slots.get(index) {
                        // Correct-consumer rule: retain only through a live
                        // reference (slot 0 stays provider-live; snapshots
                        // need one of the script's own retains).
                        if alive && (index == 0 || owned > 0) {
                            unsafe { vt_retain(resource) };
                            slots[index].1 = owned + 1;
                            my_retains += 1;
                        }
                    }
                }
                LifecycleOp::Release(index) => {
                    if let Some(&(resource, owned, alive)) = slots.get(index) {
                        if alive && owned > 0 {
                            unsafe { vt_release(resource) };
                            slots[index].1 = owned - 1;
                            my_releases += 1;
                            // A snapshot slot whose last script retain is gone
                            // is dead to the script (its birth retain was the
                            // one the script released first — see Snapshot).
                            if index != 0 && slots[index].1 == 0 {
                                slots[index].2 = false;
                            }
                        }
                    }
                }
                LifecycleOp::Snapshot(index) => {
                    if slots.len() < 5 {
                        if let Some(&(resource, owned, alive)) = slots.get(index) {
                            if alive && (index == 0 || owned > 0) {
                                let captured = unsafe { vt_snapshot(resource) };
                                births += 1;
                                // The capture's birth retain belongs to the
                                // script (out-param ownership transfer).
                                slots.push((captured, 1, true));
                            }
                        }
                    }
                }
                LifecycleOp::Probe(index) => {
                    if let Some(&(resource, owned, alive)) = slots.get(index) {
                        if alive && (index == 0 || owned > 0) {
                            // VT-LIFE-2: probing only under a live retain
                            // must always succeed.
                            prop_assert_eq!(unsafe { vt_probe_root(resource) }, VtStatus::Ok);
                        }
                    }
                }
                LifecycleOp::CopyWords(index) => {
                    if let Some(&(resource, _, alive)) = slots.get(index) {
                        if alive {
                            let before = probe.retain_calls();
                            // VT-LIFE-4: a bit copy of the two words is not a
                            // retain, and dropping the copy is not a release.
                            let copied: VtResource = resource;
                            prop_assert!(!copied.is_null());
                            let _ = copied; // discarding a raw copy is not a release
                            prop_assert_eq!(probe.retain_calls(), before);
                        }
                    }
                }
            }

            // VT-LIFE-3 (running): the provider ledger can never show more
            // releases than retains-plus-births.
            prop_assert!(
                probe.release_calls() <= probe.retain_calls() + births,
                "ledger went negative: releases {} > retains {} + births {}",
                probe.release_calls(),
                probe.retain_calls(),
                births
            );
        }

        // Teardown: release everything the script still owns, then the
        // provider handle's own root retain via drop.
        for &(resource, owned, alive) in &slots {
            if alive {
                for _ in 0..owned {
                    unsafe { vt_release(resource) };
                    my_releases += 1;
                }
            }
        }
        drop(dictionary);

        // VT-LIFE-1: the script's view and the provider's view agree. The
        // script's releases cover every retain it made PLUS each captured
        // snapshot's birth retain (ownership of which transferred to the
        // script through the out parameter); the root's birth retain is the
        // provider handle's to release.
        prop_assert_eq!(probe.retain_calls(), my_retains);
        prop_assert_eq!(
            my_releases,
            my_retains + (births - 1),
            "script releases every retain it made plus each snapshot's birth retain"
        );
        // Every context's birth retain was released exactly once (script for
        // snapshots, provider-handle drop for the root):
        prop_assert_eq!(probe.release_calls(), probe.retain_calls() + births);
        // VT-LIFE-5/6: every context was destroyed exactly once, none twice.
        prop_assert_eq!(probe.context_drops(), births);
    }
}

/// VT-LIFE-4, example form: publishing and dropping raw copies of the two
/// words is completely invisible to the provider.
#[test]
fn copying_the_two_words_is_not_a_retain() {
    let dictionary = TestDictionary::new([("cat".to_string(), Some(1))]);
    let resource = dictionary.resource();
    let copies: Vec<VtResource> = (0..64).map(|_| resource).collect();
    assert_eq!(dictionary.retain_calls(), 0);
    assert_eq!(dictionary.release_calls(), 0);
    drop(copies);
    assert_eq!(dictionary.retain_calls(), 0);
    assert_eq!(dictionary.release_calls(), 0);
}

/// VT-SNAP-3 prelude + VT-LIFE-5: a captured snapshot outlives the release of
/// every other handle, and the full teardown drains both contexts.
#[test]
fn snapshot_survives_root_release_and_teardown_drains_everything() {
    let dictionary =
        TestDictionary::new([("cat".to_string(), Some(1)), ("cot".to_string(), Some(2))]);
    let probe = dictionary.metrics_probe();
    let captured = unsafe { vt_snapshot(dictionary.resource()) };

    // Drop the provider handle: the root's birth retain is released, but the
    // captured snapshot (its own context, script-owned retain) stays usable.
    drop(dictionary);
    assert_eq!(unsafe { vt_probe_root(captured) }, VtStatus::Ok);

    unsafe { vt_release(captured) };
    assert_eq!(
        probe.context_drops(),
        2,
        "root and snapshot contexts both destroyed exactly once"
    );
    assert_eq!(probe.release_calls(), probe.retain_calls() + 2);
}
