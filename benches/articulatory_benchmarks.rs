//! Benchmarks for articulatory phonetic distance computation.
//!
//! Measures:
//! - Character-to-character articulatory distance
//! - Full edit distance with articulatory costs vs standard
//! - Overhead of phonetic feature lookup

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::distance::standard_distance;
use liblevenshtein::phonetic::feature_distance::{
    articulatory_distance, articulatory_distance_weighted, articulatory_edit_distance,
    articulatory_edit_distance_weighted, is_free_substitution, FeatureDistanceWeights,
};

// ============================================================================
// Test Data
// ============================================================================

/// IPA character pairs for single-character distance benchmarks
fn ipa_char_pairs() -> Vec<(&'static str, char, char)> {
    vec![
        // Identical sounds
        ("identical_p", 'p', 'p'),
        ("identical_a", 'a', 'a'),
        // Voicing only
        ("voicing_pb", 'p', 'b'),
        ("voicing_td", 't', 'd'),
        ("voicing_kg", 'k', 'g'),
        ("voicing_sz", 's', 'z'),
        // Adjacent place
        ("place_adjacent_pt", 'p', 't'),
        ("place_adjacent_tk", 't', 'k'),
        // Distant place
        ("place_distant_pk", 'p', 'k'),
        ("place_distant_ph", 'p', 'h'),
        // Manner difference
        ("manner_ps", 'p', 's'),
        ("manner_tn", 't', 'n'),
        // Vowels
        ("vowel_ai", 'a', 'i'),
        ("vowel_ae", 'a', 'e'),
        // Non-IPA characters
        ("non_ipa", 'x', 'y'),
        // Mixed ASCII
        ("ascii_ab", 'a', 'b'),
    ]
}

/// String pairs for edit distance benchmarks
fn edit_distance_pairs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // Identical
        ("identical_short", "pat", "pat"),
        ("identical_medium", "pattern", "pattern"),
        // Single substitution - phonetically similar
        ("sub_voicing", "pat", "bat"),
        ("sub_voicing_med", "pattern", "battern"),
        // Single substitution - phonetically different
        ("sub_different", "pat", "hat"),
        ("sub_different_med", "pattern", "hattern"),
        // Multiple substitutions
        ("multi_sub", "pit", "bed"),
        // Insertion/deletion
        ("insertion", "pat", "prat"),
        ("deletion", "prat", "pat"),
        // Mixed operations
        ("mixed", "kitten", "sitting"),
        ("mixed_long", "intention", "execution"),
        // Real words with phonetic similarity
        ("phonetic_similar", "phone", "foam"),
        ("phonetic_different", "phone", "stone"),
    ]
}

// ============================================================================
// Single Character Distance Benchmarks
// ============================================================================

fn bench_articulatory_char_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory/char_distance");

    for (name, c1, c2) in ipa_char_pairs() {
        group.bench_function(name, |b| {
            b.iter(|| articulatory_distance(black_box(c1), black_box(c2)));
        });
    }

    group.finish();
}

fn bench_is_free_substitution(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory/is_free_substitution");

    let pairs = [
        ("voicing_pb", 'p', 'b'),   // Should be free (similar)
        ("voicing_td", 't', 'd'),   // Should be free (similar)
        ("different_ph", 'p', 'h'), // Should not be free
        ("identical", 'p', 'p'),    // Should be free
    ];

    for (name, c1, c2) in pairs {
        group.bench_function(name, |b| {
            b.iter(|| is_free_substitution(black_box(c1), black_box(c2)));
        });
    }

    group.finish();
}

// ============================================================================
// Edit Distance Benchmarks
// ============================================================================

fn bench_articulatory_edit_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory/edit_distance");

    for (name, s1, s2) in edit_distance_pairs() {
        let bytes = (s1.len() + s2.len()) as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(
            BenchmarkId::new("articulatory", name),
            &(s1, s2),
            |b, (s1, s2)| {
                b.iter(|| articulatory_edit_distance(black_box(s1), black_box(s2)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Comparison: Articulatory vs Standard Distance
// ============================================================================

fn bench_articulatory_vs_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory_vs_standard");

    let pairs = [
        ("short", "pat", "bat"),
        ("medium", "pattern", "battern"),
        ("long", "information", "confirmation"),
        ("classic", "kitten", "sitting"),
    ];

    for (name, s1, s2) in pairs {
        // Standard distance
        group.bench_with_input(
            BenchmarkId::new("standard", name),
            &(s1, s2),
            |b, (s1, s2)| {
                b.iter(|| standard_distance(black_box(s1), black_box(s2)));
            },
        );

        // Articulatory distance
        group.bench_with_input(
            BenchmarkId::new("articulatory", name),
            &(s1, s2),
            |b, (s1, s2)| {
                b.iter(|| articulatory_edit_distance(black_box(s1), black_box(s2)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Throughput Benchmarks
// ============================================================================

fn bench_articulatory_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory/throughput");

    // Generate many pairs for throughput measurement
    let base_words = ["pattern", "kitten", "sitting", "written", "bitten"];
    let targets = ["battern", "sitting", "fitting", "rotten", "mitten"];

    let pairs: Vec<_> = base_words.iter().zip(targets.iter()).collect();

    group.throughput(Throughput::Elements(pairs.len() as u64));

    group.bench_function("batch_standard", |b| {
        b.iter(|| {
            pairs
                .iter()
                .map(|(s1, s2)| standard_distance(black_box(s1), black_box(s2)))
                .sum::<usize>()
        });
    });

    group.bench_function("batch_articulatory", |b| {
        b.iter(|| {
            pairs
                .iter()
                .map(|(s1, s2)| articulatory_edit_distance(black_box(s1), black_box(s2)))
                .sum::<f64>()
        });
    });

    group.finish();
}

// ============================================================================
// Feature Lookup Overhead
// ============================================================================

fn bench_feature_lookup_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory/feature_lookup");

    // Benchmark the overhead of looking up phonetic features
    // by comparing strings with IPA vs non-IPA characters

    let ipa_pairs = [("p", "b"), ("t", "d"), ("k", "g")];

    let ascii_pairs = [("x", "y"), ("q", "w"), ("j", "z")];

    group.bench_function("ipa_chars", |b| {
        b.iter(|| {
            ipa_pairs
                .iter()
                .map(|(s1, s2)| articulatory_edit_distance(black_box(s1), black_box(s2)))
                .sum::<f64>()
        });
    });

    group.bench_function("ascii_chars", |b| {
        b.iter(|| {
            ascii_pairs
                .iter()
                .map(|(s1, s2)| articulatory_edit_distance(black_box(s1), black_box(s2)))
                .sum::<f64>()
        });
    });

    group.finish();
}

// ============================================================================
// Product Automaton Benchmarks (Articulatory vs Fixed Costs)
// ============================================================================

use liblevenshtein::phonetic::nfa::compiler::compile;
use liblevenshtein::phonetic::nfa::product::ProductAutomatonChar;
use liblevenshtein::phonetic::regex::parse;
use liblevenshtein::transducer::{Algorithm, ArticulatoryCosts};

/// Benchmark ProductAutomatonChar with fixed vs articulatory costs.
fn bench_product_automaton_articulatory(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_automaton/articulatory");

    // Simple pattern NFA
    let nfa = compile(&parse("pattern").unwrap()).unwrap();
    let costs = ArticulatoryCosts::default();

    // Create both product automaton variants
    let product_fixed = ProductAutomatonChar::new(nfa.clone(), 2);
    let product_articulatory =
        ProductAutomatonChar::with_articulatory_costs(nfa.clone(), 2.0, Algorithm::Standard, costs);

    // Benchmark transition with exact match
    group.bench_function("fixed/transition_match", |b| {
        let initial = product_fixed.initial_state();
        b.iter(|| product_fixed.transition(black_box(&initial), black_box('p')));
    });

    group.bench_function("articulatory/transition_match", |b| {
        let initial = product_articulatory.initial_state();
        b.iter(|| product_articulatory.transition(black_box(&initial), black_box('p')));
    });

    // Benchmark transition with substitution (similar sound: p→b)
    group.bench_function("fixed/transition_sub_similar", |b| {
        let initial = product_fixed.initial_state();
        b.iter(|| product_fixed.transition(black_box(&initial), black_box('b')));
    });

    group.bench_function("articulatory/transition_sub_similar", |b| {
        let initial = product_articulatory.initial_state();
        b.iter(|| product_articulatory.transition(black_box(&initial), black_box('b')));
    });

    // Benchmark transition with substitution (different sound: p→k)
    group.bench_function("fixed/transition_sub_different", |b| {
        let initial = product_fixed.initial_state();
        b.iter(|| product_fixed.transition(black_box(&initial), black_box('k')));
    });

    group.bench_function("articulatory/transition_sub_different", |b| {
        let initial = product_articulatory.initial_state();
        b.iter(|| product_articulatory.transition(black_box(&initial), black_box('k')));
    });

    group.finish();
}

/// Benchmark full accepts() with fixed vs articulatory costs.
fn bench_product_automaton_accepts(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_automaton/accepts");

    let nfa = compile(&parse("pattern").unwrap()).unwrap();
    let costs = ArticulatoryCosts::default();

    let product_fixed = ProductAutomatonChar::new(nfa.clone(), 2);
    let product_articulatory =
        ProductAutomatonChar::with_articulatory_costs(nfa.clone(), 2.0, Algorithm::Standard, costs);

    // Exact match
    group.bench_function("fixed/exact_match", |b| {
        b.iter(|| product_fixed.accepts(black_box("pattern")));
    });

    group.bench_function("articulatory/exact_match", |b| {
        b.iter(|| product_articulatory.accepts(black_box("pattern")));
    });

    // One substitution (similar sound)
    group.bench_function("fixed/one_sub_similar", |b| {
        b.iter(|| product_fixed.accepts(black_box("battern")));
    });

    group.bench_function("articulatory/one_sub_similar", |b| {
        b.iter(|| product_articulatory.accepts(black_box("battern")));
    });

    // One substitution (different sound)
    group.bench_function("fixed/one_sub_different", |b| {
        b.iter(|| product_fixed.accepts(black_box("hattern")));
    });

    group.bench_function("articulatory/one_sub_different", |b| {
        b.iter(|| product_articulatory.accepts(black_box("hattern")));
    });

    // No match (distant)
    group.bench_function("fixed/no_match", |b| {
        b.iter(|| product_fixed.accepts(black_box("zzzzzzz")));
    });

    group.bench_function("articulatory/no_match", |b| {
        b.iter(|| product_articulatory.accepts(black_box("zzzzzzz")));
    });

    group.finish();
}

/// Benchmark substitution_cost computation overhead.
fn bench_substitution_cost_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_automaton/substitution_cost");

    let nfa = compile(&parse("p").unwrap()).unwrap();
    let costs = ArticulatoryCosts::default();

    let product =
        ProductAutomatonChar::with_articulatory_costs(nfa, 2.0, Algorithm::Standard, costs);

    // Test various input characters (pattern expects 'p')
    let chars = [
        ("identical", 'p'),        // Matches pattern
        ("voicing_pair", 'b'),     // p→b voicing
        ("adjacent_place", 't'),   // p→t adjacent place
        ("distant_place", 'k'),    // p→k distant place
        ("different_manner", 's'), // p→s different manner
        ("vowel", 'a'),            // Completely different
        ("non_ipa", 'x'),          // Non-IPA
    ];

    for (name, input_char) in chars {
        group.bench_function(name, |b| {
            b.iter(|| {
                // Access substitution_cost indirectly through a transition
                // since it's a private method
                let initial = product.initial_state();
                product.transition(black_box(&initial), black_box(input_char))
            });
        });
    }

    group.finish();
}

// ============================================================================
// Weighted Variant Benchmarks (G4)
// ============================================================================

/// Quantify the overhead of routing through the per-dimension weighted leaf.
/// With `FeatureDistanceWeights::standard()` the weighted path is numerically
/// identical to the unweighted one (the unweighted fns delegate to it), so this
/// measures only the `&weights` indirection — expected to be ~zero.
fn bench_articulatory_weighted(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory/weighted");
    let standard = FeatureDistanceWeights::standard();
    // A configuration that emphasizes place differences.
    let custom = FeatureDistanceWeights {
        place_step: 0.3,
        ..FeatureDistanceWeights::standard()
    };

    for (name, c1, c2) in ipa_char_pairs() {
        group.bench_with_input(BenchmarkId::new("unweighted", name), &(c1, c2), |b, &(c1, c2)| {
            b.iter(|| articulatory_distance(black_box(c1), black_box(c2)));
        });
        group.bench_with_input(
            BenchmarkId::new("weighted_standard", name),
            &(c1, c2),
            |b, &(c1, c2)| {
                b.iter(|| articulatory_distance_weighted(black_box(c1), black_box(c2), &standard));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("weighted_custom", name),
            &(c1, c2),
            |b, &(c1, c2)| {
                b.iter(|| articulatory_distance_weighted(black_box(c1), black_box(c2), &custom));
            },
        );
    }

    group.finish();
}

/// Weighted edit distance vs the unweighted (default-weight) baseline.
fn bench_articulatory_edit_distance_weighted(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory/edit_distance_weighted");
    let standard = FeatureDistanceWeights::standard();

    for (name, s1, s2) in edit_distance_pairs() {
        let bytes = (s1.len() + s2.len()) as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::new("unweighted", name), &(s1, s2), |b, (s1, s2)| {
            b.iter(|| articulatory_edit_distance(black_box(s1), black_box(s2)));
        });
        group.bench_with_input(
            BenchmarkId::new("weighted_standard", name),
            &(s1, s2),
            |b, (s1, s2)| {
                b.iter(|| {
                    articulatory_edit_distance_weighted(black_box(s1), black_box(s2), &standard)
                });
            },
        );
    }

    group.finish();
}

/// `ArticulatoryCosts::with_feature_weights` substitution-cost path.
fn bench_articulatory_costs_with_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("articulatory/costs_with_feature_weights");

    let default_costs = ArticulatoryCosts::default();
    let heavy_place = ArticulatoryCosts::with_feature_weights(FeatureDistanceWeights {
        place_step: 0.4,
        ..FeatureDistanceWeights::standard()
    });

    let pairs = [
        ("voicing_pb", 'p', 'b'),
        ("place_pt", 'p', 't'),
        ("manner_pm", 'p', 'm'),
        ("vowel_ai", 'a', 'i'),
    ];

    for (name, from, to) in pairs {
        group.bench_with_input(BenchmarkId::new("default", name), &(from, to), |b, &(from, to)| {
            b.iter(|| default_costs.substitution_cost(black_box(from), black_box(to)));
        });
        group.bench_with_input(
            BenchmarkId::new("heavy_place", name),
            &(from, to),
            |b, &(from, to)| {
                b.iter(|| heavy_place.substitution_cost(black_box(from), black_box(to)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    char_benches,
    bench_articulatory_char_distance,
    bench_is_free_substitution,
);

criterion_group!(
    edit_benches,
    bench_articulatory_edit_distance,
    bench_articulatory_vs_standard,
);

criterion_group!(
    throughput_benches,
    bench_articulatory_throughput,
    bench_feature_lookup_overhead,
);

criterion_group!(
    product_benches,
    bench_product_automaton_articulatory,
    bench_product_automaton_accepts,
    bench_substitution_cost_overhead,
);

criterion_group!(
    weighted_benches,
    bench_articulatory_weighted,
    bench_articulatory_edit_distance_weighted,
    bench_articulatory_costs_with_weights,
);

criterion_main!(
    char_benches,
    edit_benches,
    throughput_benches,
    product_benches,
    weighted_benches
);
