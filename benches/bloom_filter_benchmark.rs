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

/// Benchmark contains() WITHOUT Bloom filter
fn bench_contains_no_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_no_bloom");

    for size in [100, 500, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let dawg: DynamicDawg<()> = DynamicDawg::with_config(f32::INFINITY, None); // No Bloom filter

        for term in &terms {
            dawg.insert(term);
        }

        // Create test queries: 50% hits, 50% misses
        let mut test_queries = Vec::new();
        for i in 0..50 {
            test_queries.push(terms[i * (terms.len() / 100)].clone()); // Hits
            test_queries.push(format!("nonexistent{}", i)); // Misses
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut count = 0;
                for query in &test_queries {
                    if dawg.contains(black_box(query)) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

/// Benchmark contains() WITH Bloom filter
fn bench_contains_with_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_with_bloom");

    for size in [100, 500, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let dawg: DynamicDawg<()> = DynamicDawg::with_config(f32::INFINITY, Some(*size)); // With Bloom filter

        for term in &terms {
            dawg.insert(term);
        }

        // Create test queries: 50% hits, 50% misses
        let mut test_queries = Vec::new();
        for i in 0..50 {
            test_queries.push(terms[i * (terms.len() / 100)].clone()); // Hits
            test_queries.push(format!("nonexistent{}", i)); // Misses
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut count = 0;
                for query in &test_queries {
                    if dawg.contains(black_box(query)) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

/// Benchmark contains() - mostly negative lookups (90% misses)
fn bench_contains_mostly_negative_no_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_mostly_negative_no_bloom");

    for size in [1000, 5000].iter() {
        let terms = generate_terms(*size);
        let dawg: DynamicDawg<()> = DynamicDawg::with_config(f32::INFINITY, None); // No Bloom filter

        for term in &terms {
            dawg.insert(term);
        }

        // Create test queries: 10% hits, 90% misses
        let mut test_queries = Vec::new();
        for i in 0..10 {
            test_queries.push(terms[i * (terms.len() / 100)].clone()); // Hits
        }
        for i in 0..90 {
            test_queries.push(format!("nonexistent{}", i)); // Misses
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut count = 0;
                for query in &test_queries {
                    if dawg.contains(black_box(query)) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

/// Benchmark contains() - mostly negative lookups WITH Bloom filter (90% misses)
fn bench_contains_mostly_negative_with_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_mostly_negative_with_bloom");

    for size in [1000, 5000].iter() {
        let terms = generate_terms(*size);
        let dawg: DynamicDawg<()> = DynamicDawg::with_config(f32::INFINITY, Some(*size)); // With Bloom filter

        for term in &terms {
            dawg.insert(term);
        }

        // Create test queries: 10% hits, 90% misses
        let mut test_queries = Vec::new();
        for i in 0..10 {
            test_queries.push(terms[i * (terms.len() / 100)].clone()); // Hits
        }
        for i in 0..90 {
            test_queries.push(format!("nonexistent{}", i)); // Misses
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut count = 0;
                for query in &test_queries {
                    if dawg.contains(black_box(query)) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_contains_no_bloom,
    bench_contains_with_bloom,
    bench_contains_mostly_negative_no_bloom,
    bench_contains_mostly_negative_with_bloom,
);
criterion_main!(benches);
