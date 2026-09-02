use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::universal::{Standard as UniversalStandard, UniversalAutomaton};
use liblevenshtein::transducer::{
    Restricted, RestrictedChar, SubstitutionSet, SubstitutionSetChar, Unrestricted,
};
use std::hint::black_box;

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

fn benchmark_universal_policy_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_policy_encoding");
    let word = "encyclopædia-café-characteristic-vector";
    let exact_query = "encyclopædia-café-characteristic-vector";
    let equivalent_query = "encyclopædia-cafe-characteristic-vector";
    group.throughput(Throughput::Elements(word.chars().count() as u64));

    let default_automaton = UniversalAutomaton::<UniversalStandard>::new(2);
    group.bench_with_input(
        BenchmarkId::new("unrestricted_default", "exact"),
        &exact_query,
        |b, query| {
            b.iter(|| {
                black_box(default_automaton.accepts(black_box(word), black_box(query)));
            });
        },
    );

    let explicit_unrestricted =
        UniversalAutomaton::<UniversalStandard, _>::with_policy(2, Unrestricted);
    group.bench_with_input(
        BenchmarkId::new("unrestricted_explicit", "exact"),
        &exact_query,
        |b, query| {
            b.iter(|| {
                black_box(explicit_unrestricted.accepts(black_box(word), black_box(query)));
            });
        },
    );

    let mut substitutions = SubstitutionSetChar::new();
    substitutions.allow('é', 'e');
    let restricted = UniversalAutomaton::<UniversalStandard, _>::with_policy(
        2,
        RestrictedChar::new(&substitutions),
    );
    group.bench_with_input(
        BenchmarkId::new("restricted_unicode", "equivalence_hit"),
        &equivalent_query,
        |b, query| {
            b.iter(|| {
                black_box(restricted.accepts(black_box(word), black_box(query)));
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    benchmark_unrestricted_policy,
    benchmark_restricted_policy,
    benchmark_universal_policy_encoding
);
criterion_main!(benches);
