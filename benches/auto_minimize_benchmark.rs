use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::dynamic_dawg::DynamicDawg;

fn generate_terms(size: usize) -> Vec<String> {
    let mut terms = Vec::with_capacity(size);
    let prefixes = [
        "pre", "un", "re", "in", "dis", "en", "non", "over", "mis", "sub",
    ];
    let roots = [
        "test", "code", "data", "work", "play", "read", "write", "run", "walk", "talk",
    ];
    let suffixes = [
        "ing", "ed", "er", "est", "ly", "ness", "ment", "tion", "able", "ful",
    ];

    for i in 0..size {
        let prefix = prefixes[i % prefixes.len()];
        let root = roots[(i / prefixes.len()) % roots.len()];
        let suffix = suffixes[(i / (prefixes.len() * roots.len())) % suffixes.len()];
        terms.push(format!("{}{}{}", prefix, root, suffix));
    }

    terms
}

/// Benchmark insertion without auto-minimize (baseline)
fn bench_insert_no_auto_minimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_no_auto_minimize");

    for size in [100, 500, 1000].iter() {
        let terms = generate_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let dawg: DynamicDawg<()> = DynamicDawg::new(); // No auto-minimize (f32::INFINITY)
                for term in &terms {
                    dawg.insert(black_box(term));
                }
                black_box(dawg);
            });
        });
    }
    group.finish();
}

/// Benchmark insertion with auto-minimize threshold 1.5 (50% bloat)
fn bench_insert_with_auto_minimize_1_5(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_auto_minimize_1.5");

    for size in [100, 500, 1000].iter() {
        let terms = generate_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let dawg: DynamicDawg<()> = DynamicDawg::with_auto_minimize_threshold(1.5);
                for term in &terms {
                    dawg.insert(black_box(term));
                }
                black_box(dawg);
            });
        });
    }
    group.finish();
}

/// Benchmark insertion with auto-minimize threshold 2.0 (100% bloat)
fn bench_insert_with_auto_minimize_2_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_auto_minimize_2.0");

    for size in [100, 500, 1000].iter() {
        let terms = generate_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let dawg: DynamicDawg<()> = DynamicDawg::with_auto_minimize_threshold(2.0);
                for term in &terms {
                    dawg.insert(black_box(term));
                }
                black_box(dawg);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_insert_no_auto_minimize,
    bench_insert_with_auto_minimize_1_5,
    bench_insert_with_auto_minimize_2_0,
);
criterion_main!(benches);
