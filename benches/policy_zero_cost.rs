use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::{Restricted, SubstitutionSet};

fn benchmark_unrestricted_policy(c: &mut Criterion) {
    let dict = DoubleArrayTrie::from_terms(vec![
        "test", "testing", "tester", "best", "rest", "nest", "cat", "dog", "bird", "fish", "mouse",
        "elephant",
    ]);

    // Standard transducer (uses Unrestricted by default)
    let transducer = Transducer::standard(dict);

    c.bench_function("query_unrestricted", |b| {
        b.iter(|| {
            let results: Vec<String> = transducer.query(black_box("test"), black_box(1)).collect();
            black_box(results)
        })
    });
}

fn benchmark_restricted_policy(c: &mut Criterion) {
    let dict = DoubleArrayTrie::from_terms(vec![
        "test", "testing", "tester", "best", "rest", "nest", "cat", "dog", "bird", "fish", "mouse",
        "elephant",
    ]);

    // Create a policy with some substitutions
    let mut set = SubstitutionSet::new();
    set.allow('c', 'k');
    set.allow('k', 'c');
    set.allow('f', 'p');
    set.allow('p', 'f');
    let policy = Restricted::new(&set);

    let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);

    c.bench_function("query_restricted", |b| {
        b.iter(|| {
            let results: Vec<String> = transducer.query(black_box("test"), black_box(1)).collect();
            black_box(results)
        })
    });
}

criterion_group!(
    benches,
    benchmark_unrestricted_policy,
    benchmark_restricted_policy
);
criterion_main!(benches);
