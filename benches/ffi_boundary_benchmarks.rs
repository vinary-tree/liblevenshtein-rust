//! FFI boundary-crossing benchmarks (wave W8).
//!
//! Evidence base for the batched-cursor design: the resource ABI hands matches
//! back in leased batches so a query crosses the FFI boundary about
//! ceil(matches / batch) times rather than once per match. This benchmark
//! drains the same query at batch capacities 1, 16, and 256 over a real
//! `DynamicDawgBinding` dictionary consumed through `ResourceTransducer`, so the
//! per-match boundary cost of a large batch can be read against a
//! one-match-per-crossing baseline. Hardware for recorded results:
//! /home/dylon/.claude/hardware-specifications.md.
//!
//! Run: `cargo bench --features binding-integration-tests --bench ffi_boundary_benchmarks`
//! (pin with taskset + the performance governor for stable numbers).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use libdictenstein::bindings::{BindingUnitDomain, DynamicDawgBinding};
use liblevenshtein::bindings::{MatchBatch, QueryOrder, ResourceTransducer};
use liblevenshtein::transducer::Algorithm;

/// Build a Unicode-scalar dictionary of `n` distinct terms.
fn build_dict(n: usize) -> DynamicDawgBinding {
    let dictionary = DynamicDawgBinding::new(BindingUnitDomain::UnicodeScalar);
    for index in 0..n {
        let term = format!("term{index:06}");
        dictionary
            .insert_text(term.as_bytes(), None)
            .expect("insert_text succeeds");
    }
    dictionary
}

/// Draining a query at a larger batch capacity amortizes the FFI boundary
/// crossing across more matches, so cost per match should fall as the batch
/// grows from 1 to 256.
fn bench_cursor_drain_by_batch(criterion: &mut Criterion) {
    let dictionary = build_dict(2_000);
    let source = dictionary.resource();
    let mut group = criterion.benchmark_group("cursor_drain_by_batch");
    // Batch-capacity sweep (wave W8): capacity 1 is the one-match-per-crossing
    // baseline; 32..1024 brackets the recommended default (256) so the marginal
    // gain from a larger batch can be read directly and the default justified on
    // evidence rather than assumption. See docs/bindings/FINDINGS_LEDGER.md
    // (batch-size sweep) for the recorded decision.
    for &cap in &[1usize, 32, 64, 128, 256, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(cap), &cap, |b, &cap| {
            b.iter(|| {
                let transducer = unsafe {
                    ResourceTransducer::from_resource(source.as_raw(), Algorithm::Standard)
                }
                .expect("resource accepted");
                let mut cursor = transducer
                    .query_utf8("term000000", 2, QueryOrder::Traversal)
                    .expect("query starts");
                let mut batch = MatchBatch::default();
                let mut total = 0usize;
                loop {
                    let filled = cursor
                        .next_batch(&mut batch, cap)
                        .expect("honest provider never faults");
                    if filled == 0 {
                        break;
                    }
                    total += filled;
                }
                black_box(total);
            });
        });
    }
    group.finish();
}

/// The resource handoff itself (from_resource + query start) must be O(1) in the
/// dictionary size: the transducer captures a snapshot and walks lazily.
fn bench_resource_handoff_flat(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("resource_handoff_vs_dict_size");
    for &n in &[64usize, 1_024, 16_384] {
        let dictionary = build_dict(n);
        let source = dictionary.resource();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let transducer = unsafe {
                    ResourceTransducer::from_resource(black_box(source.as_raw()), Algorithm::Standard)
                }
                .expect("resource accepted");
                let cursor = transducer
                    .query_utf8("term000000", 1, QueryOrder::Traversal)
                    .expect("query starts");
                black_box(&cursor);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cursor_drain_by_batch, bench_resource_handoff_flat);
criterion_main!(benches);
