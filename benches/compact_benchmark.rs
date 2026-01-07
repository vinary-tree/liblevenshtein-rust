use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
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

/// Benchmark compact() after many insertions
fn bench_compact_after_insertions(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_after_insertions");

    for size in [500, 1000, 2000, 5000].iter() {
        let terms = generate_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter_batched(
                || {
                    // Setup: Create DAWG with terms
                    let dawg: DynamicDawg<()> = DynamicDawg::new();
                    for term in &terms {
                        dawg.insert(term);
                    }
                    dawg
                },
                |dawg| {
                    // Measure: compact()
                    let removed = dawg.compact();
                    black_box(removed);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Benchmark compact() after deletions (creates fragmentation)
fn bench_compact_after_deletions(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_after_deletions");

    for size in [500, 1000, 2000].iter() {
        let terms = generate_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter_batched(
                || {
                    // Setup: Create DAWG, insert terms, delete half
                    let dawg: DynamicDawg<()> = DynamicDawg::new();
                    for term in &terms {
                        dawg.insert(term);
                    }
                    // Delete every other term to create fragmentation
                    for i in (0..terms.len()).step_by(2) {
                        dawg.remove(&terms[i]);
                    }
                    dawg
                },
                |dawg| {
                    // Measure: compact()
                    let removed = dawg.compact();
                    black_box(removed);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Benchmark minimize() vs compact()
fn bench_minimize_vs_compact(c: &mut Criterion) {
    let mut group = c.benchmark_group("minimize_vs_compact");

    for size in [1000, 2000].iter() {
        let terms = generate_terms(*size);

        // Minimize benchmark
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("minimize", size), size, |b, _| {
            b.iter_batched(
                || {
                    let dawg: DynamicDawg<()> = DynamicDawg::new();
                    for term in &terms {
                        dawg.insert(term);
                    }
                    dawg
                },
                |dawg| {
                    let merged = dawg.minimize();
                    black_box(merged);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // Compact benchmark
        group.bench_with_input(BenchmarkId::new("compact", size), size, |b, _| {
            b.iter_batched(
                || {
                    let dawg: DynamicDawg<()> = DynamicDawg::new();
                    for term in &terms {
                        dawg.insert(term);
                    }
                    dawg
                },
                |dawg| {
                    let removed = dawg.compact();
                    black_box(removed);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compact_after_insertions,
    bench_compact_after_deletions,
    bench_minimize_vs_compact,
);
criterion_main!(benches);
