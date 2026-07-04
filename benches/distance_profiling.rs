//! Simplified benchmark for profiling purposes.
//!
//! This benchmark runs fewer iterations with longer runtime to make
//! profiling more effective (easier to see hot paths).

use criterion::{criterion_group, criterion_main, Criterion};
use liblevenshtein::distance::*;
use std::hint::black_box;

fn profile_standard_iterative(c: &mut Criterion) {
    c.bench_function("profile/standard_iterative", |b| {
        let source = "programming language design";
        let target = "programing languag design";
        b.iter(|| {
            for _ in 0..1000 {
                black_box(standard_distance(black_box(source), black_box(target)));
            }
        });
    });
}

fn profile_standard_recursive(c: &mut Criterion) {
    c.bench_function("profile/standard_recursive", |b| {
        let source = "programming language design";
        let target = "programing languag design";
        let cache = create_memo_cache();
        b.iter(|| {
            for _ in 0..1000 {
                black_box(standard_distance_recursive(
                    black_box(source),
                    black_box(target),
                    &cache,
                ));
            }
        });
    });
}

fn profile_transposition_iterative(c: &mut Criterion) {
    c.bench_function("profile/transposition_iterative", |b| {
        let source = "programming language design";
        let target = "programing languag design";
        b.iter(|| {
            for _ in 0..1000 {
                black_box(transposition_distance(black_box(source), black_box(target)));
            }
        });
    });
}

fn profile_merge_split(c: &mut Criterion) {
    c.bench_function("profile/merge_split", |b| {
        let source = "programming language design";
        let target = "programing languag design";
        let cache = create_memo_cache();
        b.iter(|| {
            for _ in 0..1000 {
                black_box(merge_and_split_distance(
                    black_box(source),
                    black_box(target),
                    &cache,
                ));
            }
        });
    });
}

criterion_group!(
    benches,
    profile_standard_iterative,
    profile_standard_recursive,
    profile_transposition_iterative,
    profile_merge_split,
);

criterion_main!(benches);
