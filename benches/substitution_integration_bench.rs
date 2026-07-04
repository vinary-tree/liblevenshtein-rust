//! Integration benchmarks for SubstitutionSet with real queries
//!
//! Tests performance impact of substitution policies in realistic fuzzy search
//! scenarios using actual dictionary traversal and query evaluation.
//!
//! Benchmarks cover:
//! - Unrestricted (baseline) vs Restricted policy overhead
//! - Phonetic preset performance (substitution-heavy queries)
//! - Keyboard preset performance (typo-focused queries)
//! - Custom small substitution sets
//! - Integration with dictionary queries at various distances
//!
//! Run with: cargo bench --bench substitution_integration_bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::transducer::{Algorithm, Restricted, SubstitutionSet, Transducer};
use std::hint::black_box;

/// Create a test dictionary with common English words
fn create_test_dictionary() -> DoubleArrayTrie {
    let words = vec![
        "apple",
        "application",
        "apply",
        "appreciate",
        "approach",
        "banana",
        "band",
        "bank",
        "bar",
        "base",
        "basic",
        "basis",
        "basket",
        "bear",
        "beat",
        "beautiful",
        "beauty",
        "because",
        "become",
        "bed",
        "before",
        "begin",
        "behavior",
        "behind",
        "believe",
        "benefit",
        "best",
        "better",
        "between",
        "beyond",
        "big",
        "bill",
        "billion",
        "bit",
        "black",
        "blood",
        "blue",
        "board",
        "body",
        "book",
        "born",
        "both",
        "box",
        "boy",
        "break",
        "bring",
        "brother",
        "budget",
        "build",
        "building",
        "business",
        "but",
        "buy",
        "call",
        "camera",
        "campaign",
        "can",
        "cancer",
        "candidate",
        "capital",
        "car",
        "card",
        "care",
        "career",
        "carry",
        "case",
        "catch",
        "cause",
        "cell",
        "center",
        "central",
        "century",
        "certain",
        "certainly",
        "chair",
        "challenge",
        "chance",
        "change",
        "character",
        "charge",
        "check",
        "child",
        "choice",
        "choose",
        "church",
        "citizen",
        "city",
        "civil",
        "claim",
        "class",
        "clear",
        "clearly",
        "close",
        "coach",
        "cold",
        "collection",
        "college",
        "color",
        "come",
        "commercial",
        "common",
        "community",
        "company",
        "compare",
        "computer",
        "concern",
        "condition",
        "conference",
        "congress",
        "consider",
        "consumer",
        "contain",
        "continue",
        "control",
        "cost",
        "could",
        "country",
        "couple",
        "course",
        "court",
        "cover",
        "create",
        "crime",
        "cultural",
        "culture",
        "cup",
        "current",
        "customer",
        "cut",
        "dark",
        "data",
        "daughter",
        "day",
        "dead",
        "deal",
        "death",
        "debate",
        "decade",
        "decide",
        "decision",
        "deep",
        "defense",
        "degree",
        "democratic",
        "describe",
        "design",
        "despite",
        "detail",
        "determine",
        "develop",
        "development",
        "die",
        "difference",
        "different",
        "difficult",
        "dinner",
        "direction",
        "director",
        "discover",
        "discuss",
        "discussion",
        "disease",
        "do",
        "doctor",
        "dog",
        "door",
        "down",
        "draw",
        "dream",
        "drive",
        "drop",
        "drug",
        "during",
        "each",
        "early",
        "east",
        "easy",
        "eat",
        "economic",
        "economy",
        "edge",
        "education",
        "effect",
        "effort",
        "eight",
        "either",
        "election",
        "else",
        "employee",
        "end",
        "energy",
        "enjoy",
        "enough",
        "enter",
        "entire",
        "environment",
        "environmental",
        "especially",
        "establish",
        "even",
        "evening",
        "event",
        "ever",
        "every",
        "everybody",
        "everyone",
        "everything",
        "evidence",
        "exactly",
        "example",
        "executive",
        "exist",
        "expect",
        "experience",
        "expert",
        "explain",
        "eye",
        "face",
        "fact",
        "factor",
        "fail",
        "fall",
        "family",
        "far",
        "fast",
        "father",
        "fear",
        "federal",
        "feel",
        "feeling",
        "few",
        "field",
        "fight",
        "figure",
        "fill",
        "film",
        "final",
        "finally",
        "financial",
        "find",
        "fine",
        "finger",
        "finish",
        "fire",
        "firm",
        "first",
        "fish",
        "five",
        "floor",
        "fly",
        "focus",
        "follow",
        "food",
        "foot",
        "for",
        "force",
        "foreign",
        "forget",
        "form",
        "former",
        "forward",
        "four",
        "free",
        "friend",
        "from",
        "front",
        "full",
        "fund",
        "future",
    ];
    DoubleArrayTrie::from_terms(words)
}

/// Benchmark: Unrestricted (baseline) query performance
fn bench_unrestricted_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_integration/unrestricted");
    let dict = create_test_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Test queries with typos
    let test_queries = [
        ("aple", 1),       // apple with 1 deletion
        ("appl", 1),       // apple with 1 deletion
        ("aplpy", 2),      // apply with 1 substitution + 1 transposition
        ("banan", 1),      // banana with 1 deletion
        ("beutiful", 2),   // beautiful with 2 substitutions
        ("buisness", 2),   // business with 2 substitutions
        ("computr", 1),    // computer with 1 deletion
        ("famly", 1),      // family with 1 deletion
        ("govrment", 2),   // government with 2 deletions (not in dict)
        ("intresting", 3), // interesting with substitution (not in dict)
    ];

    for (query, distance) in test_queries.iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("distance", format!("{}/{}", query, distance)),
            &(query, distance),
            |b, &(q, d)| {
                b.iter(|| {
                    let results: Vec<_> = transducer.query(black_box(q), black_box(*d)).collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Phonetic preset performance
fn bench_phonetic_preset(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_integration/phonetic");
    let dict = create_test_dictionary();

    // Phonetic substitutions: allow common sound-alike replacements
    let policy_set = SubstitutionSet::phonetic_basic();
    let policy = Restricted::new(&policy_set);
    let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);

    // Test queries with phonetic typos
    let test_queries = [
        ("aple", 1),    // apple -> should match with substitutions
        ("senter", 2),  // center -> c/s substitution
        ("kollege", 2), // college -> c/k substitution
        ("foto", 2),    // photo (not in dict) -> f/ph substitution
        ("nite", 2),    // night (not in dict) -> ight/ite
        ("kwick", 2),   // quick (not in dict) -> qu/k
    ];

    for (query, distance) in test_queries.iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("distance", format!("{}/{}", query, distance)),
            &(query, distance),
            |b, &(q, d)| {
                b.iter(|| {
                    let results: Vec<_> = transducer.query(black_box(q), black_box(*d)).collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Keyboard preset performance
fn bench_keyboard_preset(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_integration/keyboard");
    let dict = create_test_dictionary();

    // Keyboard substitutions: allow adjacent key typos
    let policy_set = SubstitutionSet::keyboard_qwerty();
    let policy = Restricted::new(&policy_set);
    let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);

    // Test queries with keyboard typos (adjacent keys)
    let test_queries = [
        ("aoole", 2),    // apple -> p/o substitution (adjacent)
        ("bannna", 2),   // banana -> a/n substitution
        ("vook", 1),     // book -> b/v substitution
        ("cimputer", 2), // computer -> o/i substitution
        ("familh", 1),   // family -> y/h substitution
    ];

    for (query, distance) in test_queries.iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("distance", format!("{}/{}", query, distance)),
            &(query, distance),
            |b, &(q, d)| {
                b.iter(|| {
                    let results: Vec<_> = transducer.query(black_box(q), black_box(*d)).collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Custom small substitution set
fn bench_custom_small_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_integration/custom_small");
    let dict = create_test_dictionary();

    // Custom small set: only allow a few specific substitutions
    let mut policy_set = SubstitutionSet::new();
    policy_set.allow_byte(b'a', b'e');
    policy_set.allow_byte(b'e', b'a');
    policy_set.allow_byte(b'i', b'e');
    policy_set.allow_byte(b'e', b'i');
    policy_set.allow_byte(b'o', b'u');
    policy_set.allow_byte(b'u', b'o');

    let policy = Restricted::new(&policy_set);
    let transducer = Transducer::with_policy(dict, Algorithm::Standard, policy);

    // Test queries using only allowed substitutions
    let test_queries = [
        ("epple", 1), // apple -> a/e substitution
        ("benen", 2), // banana -> a/e substitutions
        ("bist", 1),  // best -> e/i substitution
        ("bux", 1),   // box -> o/u substitution
    ];

    for (query, distance) in test_queries.iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("distance", format!("{}/{}", query, distance)),
            &(query, distance),
            |b, &(q, d)| {
                b.iter(|| {
                    let results: Vec<_> = transducer.query(black_box(q), black_box(*d)).collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Comparison of policy overhead at different distances
fn bench_policy_overhead_by_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_integration/overhead_by_distance");

    let query = "computer"; // Exact match in dictionary

    for distance in [1, 2, 3].iter() {
        // Unrestricted (baseline)
        let dict_unrestricted = create_test_dictionary();
        let transducer_unrestricted = Transducer::new(dict_unrestricted, Algorithm::Standard);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("unrestricted", distance),
            distance,
            |b, &d| {
                b.iter(|| {
                    let results: Vec<_> = transducer_unrestricted
                        .query(black_box(query), black_box(d))
                        .collect();
                    black_box(results)
                });
            },
        );

        // Phonetic (restricted)
        let dict_phonetic = create_test_dictionary();
        let phonetic_set = SubstitutionSet::phonetic_basic();
        let phonetic_policy = Restricted::new(&phonetic_set);
        let transducer_phonetic =
            Transducer::with_policy(dict_phonetic, Algorithm::Standard, phonetic_policy);
        group.bench_with_input(BenchmarkId::new("phonetic", distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer_phonetic
                    .query(black_box(query), black_box(d))
                    .collect();
                black_box(results)
            });
        });

        // Small custom (restricted)
        let dict_custom = create_test_dictionary();
        let mut custom_set = SubstitutionSet::new();
        custom_set.allow_byte(b'e', b'i');
        custom_set.allow_byte(b'i', b'e');
        let custom_policy = Restricted::new(&custom_set);
        let transducer_custom =
            Transducer::with_policy(dict_custom, Algorithm::Standard, custom_policy);
        group.bench_with_input(
            BenchmarkId::new("custom_small", distance),
            distance,
            |b, &d| {
                b.iter(|| {
                    let results: Vec<_> = transducer_custom
                        .query(black_box(query), black_box(d))
                        .collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// Note: Unicode char benchmarks with RestrictedChar would require a char-based dictionary
// backend (e.g., PathMapDictionary<()>). DoubleArrayTrie is byte-based (u8) only.

criterion_group!(
    benches,
    bench_unrestricted_baseline,
    bench_phonetic_preset,
    bench_keyboard_preset,
    bench_custom_small_set,
    bench_policy_overhead_by_distance,
);

criterion_main!(benches);
