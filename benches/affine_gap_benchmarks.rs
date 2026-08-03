//! Frozen Phase-7 affine-gap cost report.
//!
//! For lengths 8, 16, and 32, compare Standard at budget 2 with the
//! Levenshtein-degenerate affine parameters `(0, 1, 1)` at budget 2 and the
//! gap-favoring parameters `(2, 1, 2)` at budget 6. The matching scientific
//! ledger records environment and results; this benchmark has no performance
//! acceptance threshold.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::cost::CostScale;
use liblevenshtein::transducer::{AffineGapParams, Algorithm, Transducer};
use std::hint::black_box;

fn patterned_word(length: usize, seed: usize) -> String {
    (0..length)
        .map(|index| {
            let place = 26usize.pow((index % 3) as u32);
            let digit = (seed / place) % 26;
            char::from(b'a' + ((index * 7 + digit) % 26) as u8)
        })
        .collect()
}

fn corpus(length: usize) -> (String, Vec<String>) {
    let query = patterned_word(length, 0);
    let mut terms = Vec::with_capacity(260);
    terms.push(query.clone());

    let mut substitution = query.as_bytes().to_vec();
    substitution[length / 2] = b'z';
    terms.push(String::from_utf8(substitution).expect("ASCII benchmark term"));

    let mut single_gap = query.clone();
    single_gap.insert_str(length / 2, "qqq");
    terms.push(single_gap);

    let mut two_gaps = query.clone();
    two_gaps.insert(length / 3, 'q');
    two_gaps.insert(2 * length / 3, 'z');
    terms.push(two_gaps);

    for seed in 1..=256 {
        terms.push(patterned_word(length, seed));
    }
    terms.sort_unstable();
    terms.dedup();
    assert_eq!(terms.len(), 260, "benchmark corpus must remain unique");
    (query, terms)
}

fn scaled(open: usize, extend: usize, substitution: usize) -> AffineGapParams {
    AffineGapParams::from_scaled(
        CostScale::new(1).expect("unit benchmark scale"),
        open,
        extend,
        substitution,
    )
}

fn affine_gap_cost_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_gap/query_cost");
    let levenshtein = scaled(0, 1, 1);
    let gap_favoring = scaled(2, 1, 2);

    for length in [8usize, 16, 32] {
        let (query, terms) = corpus(length);
        let transducer = Transducer::new(DoubleArrayTrie::from_terms(&terms), Algorithm::Standard);
        group.throughput(Throughput::Elements(terms.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("standard_k2", length),
            &query,
            |b, query| {
                b.iter(|| {
                    black_box(
                        transducer
                            .query_with_distance(black_box(query), black_box(2))
                            .count(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("affine_0_1_1_k2", length),
            &query,
            |b, query| {
                b.iter(|| {
                    black_box(
                        transducer
                            .query_affine_scaled(
                                black_box(query),
                                black_box(2),
                                black_box(levenshtein),
                            )
                            .count(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("affine_2_1_2_k6", length),
            &query,
            |b, query| {
                b.iter(|| {
                    black_box(
                        transducer
                            .query_affine_scaled(
                                black_box(query),
                                black_box(6),
                                black_box(gap_favoring),
                            )
                            .count(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, affine_gap_cost_report);
criterion_main!(benches);
