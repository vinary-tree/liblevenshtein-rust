// Comprehensive benchmarks for generalized automaton core operations
//
// Tests hypotheses H1, H2, and H3:
// - H1: Successor generation dominates runtime (60-80%)
// - H2: Repeated char().collect() causes 10-15% overhead
// - H3: Subsumption creates O(n²) bottleneck for large states
//
// Hardware: Intel Xeon E5-2699 v3 @ 2.30GHz
// Run with: taskset -c 0 cargo bench --bench generalized_automaton_benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::generalized::GeneralizedAutomaton;

/// Benchmark state transition performance with varying input lengths
fn bench_state_transition_by_input_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_transition/by_input_length");

    let max_distance = 2;
    let automaton = GeneralizedAutomaton::new(max_distance);

    // Test with different input lengths: 3, 5, 8, 12, 15 characters
    for input_len in [3, 5, 8, 12, 15].iter() {
        let word = "benchmark".chars().take(*input_len).collect::<String>();
        let input = "bencmark".chars().take(*input_len).collect::<String>(); // 1 error (h missing)

        group.throughput(Throughput::Elements(*input_len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(input_len), input_len, |b, _| {
            b.iter(|| black_box(automaton.accepts(black_box(&word), black_box(&input))));
        });
    }

    group.finish();
}

/// Benchmark state transition with varying edit distances
fn bench_state_transition_by_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_transition/by_distance");

    let word = "benchmark";
    let input = "bencmark"; // 1 error

    // Test with different maximum distances: 0, 1, 2, 3
    for max_distance in [0, 1, 2, 3].iter() {
        group.throughput(Throughput::Elements(word.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(max_distance),
            max_distance,
            |b, &dist| {
                let automaton = GeneralizedAutomaton::new(dist);
                b.iter(|| black_box(automaton.accepts(black_box(word), black_box(input))));
            },
        );
    }

    group.finish();
}

/// Benchmark different input scenarios: best case (match), worst case (no match), average
fn bench_input_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_transition/scenarios");

    let max_distance = 2;
    let automaton = GeneralizedAutomaton::new(max_distance);
    let word = "benchmark";

    // Best case: exact match (distance 0)
    group.bench_function("exact_match", |b| {
        b.iter(|| black_box(automaton.accepts(black_box(word), black_box("benchmark"))));
    });

    // Average case: 1 error
    group.bench_function("one_error", |b| {
        b.iter(|| {
            black_box(automaton.accepts(
                black_box(word),
                black_box("bencmark"), // missing 'h'
            ))
        });
    });

    // Worst case: 2 errors
    group.bench_function("two_errors", |b| {
        b.iter(|| {
            black_box(automaton.accepts(
                black_box(word),
                black_box("bencmrk"), // missing 'h' and 'a'
            ))
        });
    });

    // Rejection case: too many errors
    group.bench_function("reject_distance_3", |b| {
        b.iter(|| {
            black_box(automaton.accepts(
                black_box(word),
                black_box("bncmrk"), // missing 'e', 'h', 'a' (3 errors)
            ))
        });
    });

    group.finish();
}

/// Benchmark with words of different lengths to test state size growth
fn bench_word_length_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_transition/word_length_scaling");

    let max_distance = 2;
    let automaton = GeneralizedAutomaton::new(max_distance);

    // Test with increasing word lengths
    for word_len in [5, 10, 15, 20].iter() {
        let word = "a".repeat(*word_len);
        let input = "a".repeat(*word_len - 1); // 1 deletion

        group.throughput(Throughput::Elements(*word_len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(word_len), word_len, |b, _| {
            b.iter(|| black_box(automaton.accepts(black_box(&word), black_box(&input))));
        });
    }

    group.finish();
}

/// Benchmark specific operation types to identify hotspots
fn bench_operation_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations/types");

    let max_distance = 1;
    let automaton = GeneralizedAutomaton::new(max_distance);

    // Match operation (0 errors)
    group.bench_function("match", |b| {
        b.iter(|| black_box(automaton.accepts(black_box("test"), black_box("test"))));
    });

    // Delete operation
    group.bench_function("delete", |b| {
        b.iter(|| {
            black_box(automaton.accepts(
                black_box("test"),
                black_box("tes"), // delete 't'
            ))
        });
    });

    // Insert operation
    group.bench_function("insert", |b| {
        b.iter(|| {
            black_box(automaton.accepts(
                black_box("test"),
                black_box("tesst"), // insert 's'
            ))
        });
    });

    // Substitute operation
    group.bench_function("substitute", |b| {
        b.iter(|| {
            black_box(automaton.accepts(
                black_box("test"),
                black_box("tezt"), // substitute 's' with 'z'
            ))
        });
    });

    group.finish();
}

/// Benchmark state size impact on performance (tests H3: O(n²) subsumption)
fn bench_state_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("subsumption/state_size");

    // Create scenarios that generate different state sizes
    // Higher distance = larger states

    for max_distance in [1, 2, 3].iter() {
        let automaton = GeneralizedAutomaton::new(*max_distance);
        let word = "abcdefghij"; // 10 chars
        let input = "abcdefghij";

        group.throughput(Throughput::Elements(word.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("exact_match", max_distance),
            max_distance,
            |b, _| {
                b.iter(|| black_box(automaton.accepts(black_box(word), black_box(input))));
            },
        );
    }

    // Test with inputs that maximize state size (all positions non-subsumed)
    for max_distance in [1, 2, 3].iter() {
        let automaton = GeneralizedAutomaton::new(*max_distance);
        let word = "aaaaaaaaaa"; // Repetitive word
        let input = "bbbbbbbbbb"; // All mismatches

        group.throughput(Throughput::Elements(word.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("max_state", max_distance),
            max_distance,
            |b, _| {
                b.iter(|| black_box(automaton.accepts(black_box(word), black_box(input))));
            },
        );
    }

    group.finish();
}

/// Benchmark realistic word pairs for end-to-end performance
fn bench_realistic_words(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic/word_pairs");

    let max_distance = 2;
    let automaton = GeneralizedAutomaton::new(max_distance);

    let test_cases = vec![
        ("color", "colour", "color_colour"),                   // 1 insert
        ("gray", "grey", "gray_grey"),                         // 1 substitute
        ("theater", "theatre", "theater_theatre"),             // 1 substitute
        ("organize", "organise", "organize_organise"),         // 1 substitute
        ("definitely", "definately", "definitely_definately"), // 2 substitutes
    ];

    for (word, input, name) in test_cases {
        group.bench_function(name, |b| {
            b.iter(|| black_box(automaton.accepts(black_box(word), black_box(input))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_state_transition_by_input_length,
    bench_state_transition_by_distance,
    bench_input_scenarios,
    bench_word_length_scaling,
    bench_operation_types,
    bench_state_size_impact,
    bench_realistic_words,
);

criterion_main!(benches);
