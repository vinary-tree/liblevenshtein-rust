use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::transition::{initial_state, transition_state};
use liblevenshtein::transducer::{Algorithm, Unrestricted};

fn benchmark_prefix_walks(c: &mut Criterion) {
    let mut group = c.benchmark_group("true_damerau/repeated_prefix_walk");

    for k in 1usize..=3 {
        let length = 2 * k + 4;
        let query = vec![b'a'; length];
        let dictionary = vec![b'a'; length];
        group.throughput(Throughput::Elements(length as u64));

        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::DamerauLevenshtein,
        ] {
            group.bench_with_input(
                BenchmarkId::new(algorithm.name(), format!("k={k}")),
                &(query.as_slice(), dictionary.as_slice()),
                |b, &(query, dictionary)| {
                    b.iter(|| {
                        let mut state = initial_state(query.len(), k, algorithm);
                        for &unit in dictionary {
                            let Some(next) = transition_state(
                                &state,
                                Unrestricted,
                                black_box(unit),
                                black_box(query),
                                k,
                                algorithm,
                                false,
                            ) else {
                                break;
                            };
                            state = next;
                        }
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_prefix_walks);
criterion_main!(benches);
