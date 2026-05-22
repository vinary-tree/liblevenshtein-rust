//! Analysis benchmark for small vs large substitution sets
//!
//! This benchmark measures the performance crossover point between linear scan
//! and hash-based lookup for small substitution sets.
//!
//! Specifically designed to inform Hypothesis 3: Hybrid Small/Large Strategy
//!
//! Run with: cargo bench --bench small_set_analysis --features rand

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::SubstitutionSet;
use rand::prelude::*;

/// Generate random byte pairs for testing
fn generate_byte_pairs(count: usize, seed: u64) -> Vec<(u8, u8)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let a = rng.gen_range(b'a'..=b'z');
            let b = rng.gen_range(b'a'..=b'z');
            (a, b)
        })
        .collect()
}

/// Linear scan implementation for comparison
#[derive(Clone, Debug)]
struct LinearSubstitutionSet {
    pairs: Vec<(u8, u8)>,
}

impl LinearSubstitutionSet {
    fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    fn allow_byte(&mut self, a: u8, b: u8) {
        self.pairs.push((a, b));
    }

    #[inline]
    fn contains(&self, dict_char: u8, query_char: u8) -> bool {
        self.pairs
            .iter()
            .any(|&(a, b)| a == dict_char && b == query_char)
    }

    fn from_pairs(pairs: &[(u8, u8)]) -> Self {
        Self {
            pairs: pairs.to_vec(),
        }
    }
}

/// SmallVec-based implementation (inline storage)
use smallvec::{smallvec, SmallVec};

/// SmallVec with 8-element inline capacity
#[derive(Clone, Debug)]
struct SmallVec8SubstitutionSet {
    pairs: SmallVec<[(u8, u8); 8]>,
}

impl SmallVec8SubstitutionSet {
    fn new() -> Self {
        Self { pairs: smallvec![] }
    }

    fn allow_byte(&mut self, a: u8, b: u8) {
        self.pairs.push((a, b));
    }

    #[inline]
    fn contains(&self, dict_char: u8, query_char: u8) -> bool {
        self.pairs
            .iter()
            .any(|&(a, b)| a == dict_char && b == query_char)
    }

    fn from_pairs(pairs: &[(u8, u8)]) -> Self {
        Self {
            pairs: SmallVec::from_slice(pairs),
        }
    }
}

/// SmallVec with 16-element inline capacity
#[derive(Clone, Debug)]
struct SmallVec16SubstitutionSet {
    pairs: SmallVec<[(u8, u8); 16]>,
}

impl SmallVec16SubstitutionSet {
    fn new() -> Self {
        Self { pairs: smallvec![] }
    }

    fn allow_byte(&mut self, a: u8, b: u8) {
        self.pairs.push((a, b));
    }

    #[inline]
    fn contains(&self, dict_char: u8, query_char: u8) -> bool {
        self.pairs
            .iter()
            .any(|&(a, b)| a == dict_char && b == query_char)
    }

    fn from_pairs(pairs: &[(u8, u8)]) -> Self {
        Self {
            pairs: SmallVec::from_slice(pairs),
        }
    }
}

/// Benchmark: Crossover point analysis - varying set sizes (1-20 pairs)
fn bench_crossover_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_set/crossover");

    // Test very small sizes to find exact crossover point
    for size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20].iter() {
        let pairs = generate_byte_pairs(*size, 42);

        // Hash-based (FxHashSet)
        let mut hash_set = SubstitutionSet::new();
        for &(a, b) in &pairs {
            hash_set.allow_byte(a, b);
        }

        // Linear scan (Vec)
        let linear_set = LinearSubstitutionSet::from_pairs(&pairs);

        // SmallVec<8> (most likely optimal)
        let smallvec8_set = SmallVec8SubstitutionSet::from_pairs(&pairs);

        // SmallVec<16> (for comparison)
        let smallvec16_set = SmallVec16SubstitutionSet::from_pairs(&pairs);

        // Test queries: 50% hits, 50% misses
        let test_queries: Vec<(u8, u8)> = (0..100)
            .map(|i| {
                if i < 50 {
                    pairs[i % pairs.len()]
                } else {
                    (b'A' + (i as u8), b'Z')
                }
            })
            .collect();

        group.throughput(Throughput::Elements(100));

        group.bench_with_input(BenchmarkId::new("hash", size), size, |b, _| {
            b.iter(|| {
                for &(a, c) in &test_queries {
                    black_box(hash_set.contains(black_box(a), black_box(c)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("linear", size), size, |b, _| {
            b.iter(|| {
                for &(a, c) in &test_queries {
                    black_box(linear_set.contains(black_box(a), black_box(c)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("smallvec8", size), size, |b, _| {
            b.iter(|| {
                for &(a, c) in &test_queries {
                    black_box(smallvec8_set.contains(black_box(a), black_box(c)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("smallvec16", size), size, |b, _| {
            b.iter(|| {
                for &(a, c) in &test_queries {
                    black_box(smallvec16_set.contains(black_box(a), black_box(c)));
                }
            });
        });
    }
    group.finish();
}

/// Benchmark: Hit rate sensitivity for small sets
fn bench_hit_rate_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_set/hit_rate");

    let size = 5; // Representative small size
    let pairs = generate_byte_pairs(size, 42);

    let mut hash_set = SubstitutionSet::new();
    for &(a, b) in &pairs {
        hash_set.allow_byte(a, b);
    }

    let linear_set = LinearSubstitutionSet::from_pairs(&pairs);

    for hit_rate in [10, 50, 90].iter() {
        let test_queries: Vec<(u8, u8)> = (0..1000)
            .map(|i| {
                if (i % 100) < *hit_rate {
                    pairs[i % pairs.len()]
                } else {
                    (b'A' + ((i % 26) as u8), b'Z')
                }
            })
            .collect();

        group.throughput(Throughput::Elements(1000));

        group.bench_with_input(
            BenchmarkId::new("hash", format!("{}%", hit_rate)),
            hit_rate,
            |b, _| {
                b.iter(|| {
                    for &(a, c) in &test_queries {
                        black_box(hash_set.contains(black_box(a), black_box(c)));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("linear", format!("{}%", hit_rate)),
            hit_rate,
            |b, _| {
                b.iter(|| {
                    for &(a, c) in &test_queries {
                        black_box(linear_set.contains(black_box(a), black_box(c)));
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Initialization overhead for small sets
fn bench_init_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_set/init");

    for size in [1, 3, 5, 10, 20].iter() {
        let pairs = generate_byte_pairs(*size, 42);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("hash", size), size, |b, _| {
            b.iter(|| {
                let mut set = SubstitutionSet::new();
                for &(a, c) in &pairs {
                    set.allow_byte(black_box(a), black_box(c));
                }
                black_box(set)
            });
        });

        group.bench_with_input(BenchmarkId::new("linear", size), size, |b, _| {
            b.iter(|| {
                let mut set = LinearSubstitutionSet::new();
                for &(a, c) in &pairs {
                    set.allow_byte(black_box(a), black_box(c));
                }
                black_box(set)
            });
        });

        group.bench_with_input(BenchmarkId::new("smallvec8", size), size, |b, _| {
            b.iter(|| {
                let mut set = SmallVec8SubstitutionSet::new();
                for &(a, c) in &pairs {
                    set.allow_byte(black_box(a), black_box(c));
                }
                black_box(set)
            });
        });
    }
    group.finish();
}

/// Benchmark: Memory access patterns (single lookup)
fn bench_single_lookup_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_set/single_lookup");

    for size in [1, 3, 5, 10].iter() {
        let pairs = generate_byte_pairs(*size, 42);

        let mut hash_set = SubstitutionSet::new();
        for &(a, b) in &pairs {
            hash_set.allow_byte(a, b);
        }

        let linear_set = LinearSubstitutionSet::from_pairs(&pairs);
        let smallvec_set = SmallVec8SubstitutionSet::from_pairs(&pairs);

        let (test_a, test_b) = pairs[0];

        group.bench_with_input(BenchmarkId::new("hash/hit", size), size, |b, _| {
            b.iter(|| black_box(hash_set.contains(black_box(test_a), black_box(test_b))));
        });

        group.bench_with_input(BenchmarkId::new("linear/hit", size), size, |b, _| {
            b.iter(|| black_box(linear_set.contains(black_box(test_a), black_box(test_b))));
        });

        group.bench_with_input(BenchmarkId::new("smallvec/hit", size), size, |b, _| {
            b.iter(|| black_box(smallvec_set.contains(black_box(test_a), black_box(test_b))));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_crossover_point,
    bench_hit_rate_small,
    bench_init_small,
    bench_single_lookup_small,
);

criterion_main!(benches);
