//! Price the loss of characteristic-vector reuse for contextual costs.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::Dictionary;
use liblevenshtein::transducer::{
    Algorithm, ContextualQueryIterator, OperationCostsF64, QueryIteratorF64,
};
use std::hint::black_box;

fn word(mut seed: usize, length: usize) -> String {
    let mut bytes = vec![b'a'; length];
    for byte in bytes.iter_mut().rev() {
        *byte = b'a' + (seed % 26) as u8;
        seed /= 26;
    }
    String::from_utf8(bytes).expect("lowercase benchmark word")
}

fn contextual_cost_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("contextual_cost/context_free_adapter");
    for size in [1_000usize, 10_000] {
        let mut terms: Vec<_> = (0..size).map(|seed| word(seed, 12)).collect();
        terms.sort_unstable();
        terms.dedup();
        let query = terms[terms.len() / 2].clone();
        let dictionary = DoubleArrayTrieChar::from_terms(&terms);
        let costs = OperationCostsF64::standard();
        group.throughput(Throughput::Elements(terms.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("query_iterator_f64", size),
            &query,
            |bencher, query| {
                bencher.iter(|| {
                    black_box(
                        QueryIteratorF64::<_, String>::new(
                            dictionary.root(),
                            black_box(query.clone()),
                            black_box(2.0),
                            Algorithm::Standard,
                            costs,
                        )
                        .count(),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("contextual_adapter", size),
            &query,
            |bencher, query| {
                bencher.iter(|| {
                    black_box(
                        ContextualQueryIterator::from_dictionary(
                            &dictionary,
                            black_box(query.chars().collect()),
                            black_box(2.0),
                            costs,
                        )
                        .expect("benchmark configuration is valid")
                        .count(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, contextual_cost_report);
criterion_main!(benches);
