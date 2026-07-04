//! Profiling benchmark for generating flame graphs of query operations.
//!
//! This benchmark is specifically designed for use with cargo-flamegraph
//! to identify performance bottlenecks in query iterators.
//!
//! Usage:
//!   RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
//!   cargo flamegraph --bench query_profiling -- --bench --profile-time 30

use criterion::{criterion_group, criterion_main, Criterion};
use liblevenshtein::prelude::*;
use std::fs;
use std::hint::black_box;

/// Load a large dictionary from system dictionary or create one
fn load_large_dictionary() -> DoubleArrayTrie {
    // Try to load system dictionary first
    if let Ok(contents) = fs::read_to_string("/usr/share/dict/words") {
        let words: Vec<&str> = contents.lines().take(10000).collect();
        return DoubleArrayTrie::from_terms(words);
    }

    // Fallback: create synthetic dictionary
    let words: Vec<String> = (0..10000)
        .map(|i| match i % 10 {
            0 => format!("test{}", i),
            1 => format!("best{}", i),
            2 => format!("rest{}", i),
            3 => format!("nest{}", i),
            4 => format!("word{}", i),
            5 => format!("term{}", i),
            6 => format!("item{}", i),
            7 => format!("code{}", i),
            8 => format!("data{}", i),
            _ => format!("info{}", i),
        })
        .collect();
    DoubleArrayTrie::from_terms(words)
}

/// Profile ordered query with moderate distance
fn profile_ordered_query_moderate_distance(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("ordered_query_distance_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_ordered(black_box("testing"), black_box(2))
                .collect();
            black_box(results);
        });
    });
}

/// Profile ordered query with large distance
fn profile_ordered_query_large_distance(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("ordered_query_distance_10", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_ordered(black_box("testing"), black_box(10))
                .collect();
            black_box(results);
        });
    });
}

/// Profile ordered query sorting overhead
fn profile_ordered_query_sorting(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("ordered_query_sorting_stress", |b| {
        b.iter(|| {
            // Query that produces many results to stress the sorting
            let results: Vec<_> = transducer
                .query_ordered(black_box("xyz"), black_box(3))
                .collect();
            black_box(results);
        });
    });
}

/// Profile prefix mode query
fn profile_prefix_query(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("prefix_query_distance_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_ordered(black_box("test"), black_box(2))
                .prefix()
                .collect();
            black_box(results);
        });
    });
}

/// Profile unordered query for comparison
fn profile_unordered_query(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("unordered_query_distance_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query(black_box("testing"), black_box(2))
                .collect();
            black_box(results);
        });
    });
}

/// Profile ordered query with early termination
fn profile_ordered_query_early_termination(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("ordered_query_take_10", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_ordered(black_box("testing"), black_box(5))
                .take(10)
                .collect();
            black_box(results);
        });
    });
}

/// Profile ordered query with take_while
fn profile_ordered_query_take_while(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("ordered_query_take_while_dist_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_ordered(black_box("testing"), black_box(5))
                .take_while(|c| c.distance <= 2)
                .collect();
            black_box(results);
        });
    });
}

/// Profile advance() method hotpath
fn profile_ordered_advance_hotpath(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("ordered_advance_hotpath", |b| {
        b.iter(|| {
            // This stresses the advance() method with multiple distance levels
            let results: Vec<_> = transducer
                .query_ordered(black_box("abcxyz"), black_box(4))
                .collect();
            black_box(results);
        });
    });
}

/// Profile buffer sorting in ordered query
fn profile_buffer_sorting(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("buffer_sorting_many_results", |b| {
        b.iter(|| {
            // Query that generates many results at the same distance
            let results: Vec<_> = transducer
                .query_ordered(black_box("qqqq"), black_box(4))
                .collect();
            black_box(results);
        });
    });
}

/// Profile transposition algorithm with ordered query
fn profile_transposition_ordered(c: &mut Criterion) {
    let dict = load_large_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Transposition);

    c.bench_function("transposition_ordered_query", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_ordered(black_box("testing"), black_box(2))
                .collect();
            black_box(results);
        });
    });
}

criterion_group!(
    query_profiling,
    profile_ordered_query_moderate_distance,
    profile_ordered_query_large_distance,
    profile_ordered_query_sorting,
    profile_prefix_query,
    profile_unordered_query,
    profile_ordered_query_early_termination,
    profile_ordered_query_take_while,
    profile_ordered_advance_hotpath,
    profile_buffer_sorting,
    profile_transposition_ordered,
);

criterion_main!(query_profiling);
