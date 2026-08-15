//! Boundary-crossing census (wave W8).
//!
//! The batched-cursor design amortizes the FFI boundary: a query that returns
//! `M` matches crosses the consumer↔cursor boundary `⌈M/cap⌉ + 1` times at batch
//! capacity `cap` (the `+1` is the terminal `End`-returning pull), and it crosses
//! the cursor↔provider boundary exactly once for the query-start snapshot
//! (`snapshot_calls == 1`) plus a bounded number of `node_edges` expansions. This
//! test measures those crossings through a real `ResourceTransducer` over the
//! metrics-instrumented provider and pins the laws that
//! `docs/verification/tla/LlevBatchLease.tla` and `abi_paging_correspondence.rs`
//! certify, then writes a human-readable census table to
//! `target/ffi-census/boundary_crossing_census.tsv` for the record.
//!
//! Run: `cargo test --features binding-integration-tests --test ffi_boundary_census`

#![cfg(feature = "binding-integration-tests")]

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use liblevenshtein::bindings::{MatchBatch, QueryOrder, ResourceTransducer};
use liblevenshtein::transducer::Algorithm;

mod support;
use support::interop_dictionary::TestDictionary;

/// Ceiling division, the closed form for the number of full-plus-partial pages
/// `M` matches occupy at capacity `cap`.
fn div_ceil(numerator: usize, denominator: usize) -> usize {
    numerator.div_ceil(denominator)
}

/// Drain a fresh query completely at batch capacity `cap`, returning
/// `(matches, next_batch_calls)`. `next_batch_calls` counts every crossing of the
/// consumer↔cursor boundary, including the terminal pull that returns `End`.
fn drain_at_capacity(
    dictionary: &TestDictionary,
    query: &str,
    k: usize,
    cap: usize,
) -> (usize, usize) {
    let transducer =
        unsafe { ResourceTransducer::from_resource(dictionary.resource(), Algorithm::Standard) }
            .expect("resource accepted");
    let mut cursor = transducer
        .query_utf8(query, k, QueryOrder::Traversal)
        .expect("query starts");
    let mut batch = MatchBatch::default();
    let mut matches = 0usize;
    let mut crossings = 0usize;
    loop {
        let filled = cursor
            .next_batch(&mut batch, cap)
            .expect("honest provider never faults");
        crossings += 1;
        if filled == 0 {
            break;
        }
        matches += filled;
    }
    (matches, crossings)
}

/// The consumer↔cursor crossing count obeys `⌈M/cap⌉ + 1` across every batch
/// capacity, and the query captures its provider snapshot exactly once. A census
/// row is emitted per capacity to `target/ffi-census/boundary_crossing_census.tsv`.
#[test]
fn boundary_crossings_follow_the_paging_law_and_snapshot_is_captured_once() {
    // Twenty single-character terms are each within edit distance 1 of the query
    // "a" (one substitution), so a distance-1 query returns all twenty — a fixed,
    // deterministic match set to sweep the batch capacity against.
    let entries: Vec<(String, Option<u64>)> = ('a'..='t')
        .enumerate()
        .map(|(index, ch)| (ch.to_string(), Some(index as u64)))
        .collect();
    let expected_matches = entries.len();
    let dictionary = TestDictionary::new(entries);

    let mut census = String::new();
    writeln!(
        census,
        "capacity\tmatches\tnext_batch_calls\texpected_calls\tsnapshot_calls\tedge_batch_calls\tretain_calls\trelease_calls\tcontext_drops"
    )
    .expect("write header");

    // Capacity 1 is the one-match-per-crossing baseline; the recommended default
    // (256) and beyond collapse the twenty matches into a single filled page.
    for &cap in &[1usize, 4, 8, 16, 20, 256] {
        let snapshots_before = dictionary.snapshot_calls();
        let (matches, crossings) = drain_at_capacity(&dictionary, "a", 1, cap);

        let expected_calls = div_ceil(expected_matches, cap) + 1;
        assert_eq!(
            matches, expected_matches,
            "distance-1 query over single-character terms returns every term (cap {cap})"
        );
        assert_eq!(
            crossings, expected_calls,
            "consumer<->cursor crossings must equal ceil(M/cap)+1 at capacity {cap}"
        );
        assert_eq!(
            dictionary.snapshot_calls() - snapshots_before,
            1,
            "each query captures its provider snapshot exactly once (cap {cap})"
        );

        writeln!(
            census,
            "{cap}\t{matches}\t{crossings}\t{expected_calls}\t{}\t{}\t{}\t{}\t{}",
            dictionary.snapshot_calls(),
            dictionary.edge_batch_calls(),
            dictionary.retain_calls(),
            dictionary.release_calls(),
            dictionary.context_drops(),
        )
        .expect("write row");
    }

    // A larger capacity never costs more crossings than a smaller one: the census
    // column is monotonically non-increasing in capacity.
    let mut previous = usize::MAX;
    for &cap in &[1usize, 4, 8, 16, 20, 256] {
        let (_matches, crossings) = drain_at_capacity(&dictionary, "a", 1, cap);
        assert!(
            crossings <= previous,
            "crossings must not increase with capacity (cap {cap}: {crossings} > {previous})"
        );
        previous = crossings;
    }

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/ffi-census");
    fs::create_dir_all(&out_dir).expect("create census dir");
    let out_path = out_dir.join("boundary_crossing_census.tsv");
    fs::write(&out_path, &census).expect("write census tsv");
    // Also surface the table in `--nocapture` runs for the record.
    print!("boundary-crossing census ->\n{census}");
}
