//! Correspondence anchor for the interop paging laws and the harmonized
//! consumer-acceptance predicate.
//!
//! Specs: docs/verification/abi/theories/PagingLaws.v (the page shape and its
//! six laws) and ConsumerAcceptance.v (the single `accepts_dec` predicate the
//! consumer runs). This test drives the REAL consumer paging path
//! (`ForeignNode::expanded_edges`, exercised through a full query over a
//! high-degree provider) and an adversarial provider whose misbehaviors are
//! exactly the `reject_*` lemmas: each must be rejected as a provider error,
//! and — the point of finding LLEV-B8 — an inflated `out_total` must never
//! drive an allocation abort.
//!
//! INVARIANT-HOOK: VT-PAGE-1 — written <= capacity.
//! INVARIANT-HOOK: VT-PAGE-2 — written <= total - start (never past the end).
//! INVARIANT-HOOK: VT-PAGE-3 — a full page exists exactly while the window fits.
//! INVARIANT-HOOK: VT-PAGE-4 — start >= total yields the empty (End) page.
//! INVARIANT-HOOK: VT-PAGE-5 — out_total is window-independent.
//! INVARIANT-HOOK: VT-PAGE-6 — pages concatenate losslessly to the full node.
//! INVARIANT-HOOK: VT-SNAP-3 — a latched provider fault surfaces on the next
//! advance and no match is fabricated past it (adversarial-page rejection;
//! spec: docs/verification/abi/theories/CursorSnapshotSemantics.v).

#![cfg(feature = "bindings-core")]

mod support;

use liblevenshtein::bindings::{QueryOrder, ResourceTransducer};
use liblevenshtein::transducer::Algorithm;
use proptest::prelude::*;
use support::high_degree_dictionary::HighDegreeDictionary;

/// A pure re-statement of ConsumerAcceptance.accepts_dec, so the test asserts
/// the SAME boolean the Rust consumer's inline check computes and the Coq
/// proof certifies. `max_total` is the consumer's allocation ceiling; the real
/// consumer realizes it structurally (it never sizes an allocation from
/// `total`), so here we pass the true node degree as the honest ceiling.
fn accepts(start: usize, capacity: usize, max_total: usize, reply: (usize, usize, usize)) -> bool {
    let (written, total, page_len) = reply;
    page_len == written
        && written <= capacity
        && written <= total.saturating_sub(start)
        && (written != 0 || capacity == 0 || total <= start)
        && total <= max_total
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// VT-PAGE-1..6: for a node of arbitrary degree paged at an arbitrary
    /// (start, capacity), the honest reply the provider produces is accepted,
    /// the page holds exactly `written` items, and paging from 0 by the
    /// recommended capacity reproduces the whole edge set losslessly.
    #[test]
    fn honest_paging_is_accepted_and_lossless(
        degree in 0usize..600,
        start in 0usize..700,
        capacity in 0usize..300,
    ) {
        let dictionary = HighDegreeDictionary::single_node(degree);
        let reply = dictionary.raw_node_edges(start, capacity);
        // Honest replies satisfy the predicate against the true-degree ceiling.
        prop_assert!(accepts(start, capacity, degree, reply));
        let (written, total, page_len) = reply;
        prop_assert_eq!(total, degree);            // VT-PAGE-5 window-independent
        prop_assert_eq!(page_len, written);        // page holds exactly written
        prop_assert!(written <= capacity);         // VT-PAGE-1
        prop_assert!(written <= total.saturating_sub(start)); // VT-PAGE-2
        if start.saturating_add(capacity) <= total {
            prop_assert_eq!(written, capacity);    // VT-PAGE-3 full page
        }
        if start >= total {
            prop_assert_eq!(written, 0);           // VT-PAGE-4 End
        }

        // VT-PAGE-6: the full consumer walk (recommended capacity from 0)
        // yields every label exactly once, in order.
        let walked = dictionary.walk_all_labels();
        prop_assert_eq!(walked, (0..degree as u32).collect::<Vec<_>>());
    }
}

/// The consumer's real paging path over a very high-degree node crossing the
/// 256 recommended batch many times returns every edge — the end-to-end
/// lossless-decomposition check through `expanded_edges`.
#[test]
fn full_query_traverses_a_high_degree_node_losslessly() {
    // A star of 1000 one-edge terms under a shared prefix forces the root's
    // expansion to page across the recommended batch ~4 times.
    let dictionary = HighDegreeDictionary::star(1000);
    let transducer =
        unsafe { ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard) }
            .expect("resource accepted");
    let mut cursor = transducer
        .query_utf8(dictionary.query(), 1, QueryOrder::Traversal)
        .expect("query starts");
    let mut matches = 0usize;
    let mut batch = liblevenshtein::bindings::MatchBatch::default();
    loop {
        let filled = cursor
            .next_batch(&mut batch, liblevenshtein::bindings::DEFAULT_MATCH_BATCH)
            .expect("honest provider never faults");
        if filled == 0 {
            break;
        }
        matches += filled;
    }
    assert!(
        matches >= 1000,
        "every star leaf within distance 1 should match (got {matches})"
    );
    assert!(
        dictionary.max_edge_page_capacity() <= 256,
        "no page ever exceeded the recommended batch"
    );
}

/// The rejection lemmas, live: each adversarial misbehavior surfaces as a
/// provider error from the consumer, and the inflated-total case does NOT
/// abort (finding LLEV-B8 — the consumer never allocates on the claimed total).
#[test]
fn adversarial_pages_are_rejected_without_aborting() {
    use support::high_degree_dictionary::Misbehavior;
    for misbehavior in [
        Misbehavior::Overfill,
        Misbehavior::PastEnd,
        Misbehavior::StalledProgress,
        Misbehavior::InflatedTotal,
    ] {
        let dictionary = HighDegreeDictionary::misbehaving(8, misbehavior);
        let transducer = unsafe {
            ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard)
        }
        .expect("resource accepted");
        let mut cursor = transducer
            .query_utf8(dictionary.query(), 1, QueryOrder::Traversal)
            .expect("query starts");
        let mut batch = liblevenshtein::bindings::MatchBatch::default();
        // The traversal must terminate with a provider error (never hang, never
        // abort the process on an allocation).
        let outcome = cursor.next_batch(&mut batch, liblevenshtein::bindings::DEFAULT_MATCH_BATCH);
        assert!(
            outcome.is_err(),
            "misbehavior {misbehavior:?} must surface as a provider error"
        );
    }
}
