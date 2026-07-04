//! Profiling benchmark for fuzzy map operations.
//!
//! This benchmark is designed to run with flamegraph to identify
//! bottlenecks in value-filtered queries vs post-filtered queries.

use criterion::{criterion_group, criterion_main, Criterion};
use liblevenshtein::prelude::*;
use std::hint::black_box;

/// Create a dictionary with u32 scope IDs
fn create_dict_with_scopes(size: usize, num_scopes: usize) -> PathMapDictionary<u32> {
    let terms: Vec<(String, u32)> = (0..size)
        .map(|i| {
            let term = format!("term{:04}", i);
            let scope = (i % num_scopes) as u32;
            (term, scope)
        })
        .collect();
    PathMapDictionary::from_terms_with_values(terms)
}

/// Profile: Unfiltered query (baseline)
fn profile_unfiltered(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 10);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("unfiltered_query", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer.query(black_box("term"), black_box(2)).collect();
            black_box(results)
        })
    });
}

/// Profile: Value-filtered query (during traversal)
fn profile_value_filtered(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 10);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("value_filtered_query", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_filtered(black_box("term"), black_box(2), |scope| *scope == 5)
                .collect();
            black_box(results)
        })
    });
}

/// Profile: Post-filtered query (filter after query)
fn profile_post_filtered(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 10);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("post_filtered_query", |b| {
        b.iter(|| {
            let dict_ref = transducer.dictionary();
            let results: Vec<_> = transducer
                .query(black_box("term"), black_box(2))
                .filter(|term| dict_ref.get_value(term) == Some(5))
                .collect();
            black_box(results)
        })
    });
}

/// Profile: Value-filtered with high selectivity (50%)
fn profile_value_filtered_high_selectivity(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 2); // 50% selectivity
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("value_filtered_high_selectivity", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_filtered(black_box("term"), black_box(2), |scope| *scope == 0)
                .collect();
            black_box(results)
        })
    });
}

/// Profile: Post-filtered with high selectivity (50%)
fn profile_post_filtered_high_selectivity(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 2); // 50% selectivity
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("post_filtered_high_selectivity", |b| {
        b.iter(|| {
            let dict_ref = transducer.dictionary();
            let results: Vec<_> = transducer
                .query(black_box("term"), black_box(2))
                .filter(|term| dict_ref.get_value(term) == Some(0))
                .collect();
            black_box(results)
        })
    });
}

/// Profile: Dictionary value access operations
fn profile_value_access(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 10);

    c.bench_function("value_access", |b| {
        b.iter(|| {
            // Access values for various terms
            for i in 0..100 {
                let term = format!("term{:04}", i * 10);
                let value = dict.get_value(black_box(&term));
                black_box(value);
            }
        })
    });
}

criterion_group!(
    benches,
    profile_unfiltered,
    profile_value_filtered,
    profile_post_filtered,
    profile_value_filtered_high_selectivity,
    profile_post_filtered_high_selectivity,
    profile_value_access,
);
criterion_main!(benches);
