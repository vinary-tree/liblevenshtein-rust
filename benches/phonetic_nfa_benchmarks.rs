//! Benchmarks for NFA-based phonetic regex functionality.
//!
//! These benchmarks measure:
//! - NFA pattern matching throughput
//! - Lazy DFA caching efficiency
//! - Product automaton (fuzzy regex) performance
//! - PhoneticTransducer dictionary query speed

#![cfg(feature = "phonetic-rules")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::double_array_trie_char::DoubleArrayTrieChar;
use liblevenshtein::phonetic::nfa::{
    compile, IncrementalMatcherChar, LazyDFAChar, MemoizedMatcherChar, ProductAutomatonChar,
};
use liblevenshtein::phonetic::regex::parse;
use liblevenshtein::phonetic::verified::zompist_nfa_char;
use liblevenshtein::transducer::PhoneticTransducerChar;

// ============================================================================
// NFA Pattern Matching Benchmarks
// ============================================================================

fn bench_nfa_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("nfa_pattern_matching");

    // Simple literal pattern
    let nfa_simple = compile(&parse("phone").expect("parse")).expect("compile");

    // Alternation pattern
    let nfa_alt = compile(&parse("(ph|f)one").expect("parse")).expect("compile");

    // Complex pattern with repetition
    let nfa_complex = compile(&parse("(a|b|c)+").expect("parse")).expect("compile");

    // Test inputs
    let inputs_simple = ["phone", "phon", "phones", "xyz"];
    let inputs_alt = ["phone", "fone", "bone", "tone"];
    let inputs_complex = ["a", "abc", "aabbcc", "aaabbbccc"];

    group.throughput(Throughput::Elements(inputs_simple.len() as u64));

    group.bench_function("simple_literal", |b| {
        b.iter(|| {
            for input in &inputs_simple {
                black_box(nfa_simple.accepts(input));
            }
        })
    });

    group.bench_function("alternation", |b| {
        b.iter(|| {
            for input in &inputs_alt {
                black_box(nfa_alt.accepts(input));
            }
        })
    });

    group.bench_function("complex_repetition", |b| {
        b.iter(|| {
            for input in &inputs_complex {
                black_box(nfa_complex.accepts(input));
            }
        })
    });

    group.finish();
}

// ============================================================================
// Lazy DFA Benchmarks
// ============================================================================

fn bench_lazy_dfa(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_dfa");

    let nfa = compile(&parse("(a|b|c)*").expect("parse")).expect("compile");
    let mut dfa = LazyDFAChar::new(nfa.clone());

    // Warm up the cache
    for len in [1, 5, 10, 20] {
        let input: String = (0..len).map(|i| ['a', 'b', 'c'][i % 3]).collect();
        dfa.accepts(&input);
    }

    group.bench_function("cached_lookup", |b| {
        b.iter(|| black_box(dfa.accepts("abc")))
    });

    // Fresh DFA (no cache)
    group.bench_function("fresh_lookup", |b| {
        b.iter(|| {
            let mut fresh_dfa = LazyDFAChar::new(nfa.clone());
            black_box(fresh_dfa.accepts("abc"))
        })
    });

    // Different input lengths
    for len in [5, 10, 20, 50] {
        let input: String = (0..len).map(|i| ['a', 'b', 'c'][i % 3]).collect();
        group.throughput(Throughput::Bytes(len as u64));

        group.bench_with_input(BenchmarkId::new("length", len), &input, |b, input| {
            let mut local_dfa = LazyDFAChar::new(nfa.clone());
            b.iter(|| black_box(local_dfa.accepts(input)))
        });
    }

    group.finish();
}

// ============================================================================
// Product Automaton (Fuzzy Regex) Benchmarks
// ============================================================================

fn bench_product_automaton(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_automaton");

    let nfa = compile(&parse("phone").expect("parse")).expect("compile");
    let product = ProductAutomatonChar::new(nfa.clone(), 5);

    // Benchmark distance computation
    group.bench_function("exact_match", |b| {
        b.iter(|| black_box(product.min_distance("phone")))
    });

    group.bench_function("one_edit", |b| {
        b.iter(|| black_box(product.min_distance("phon")))
    });

    group.bench_function("two_edits", |b| {
        b.iter(|| black_box(product.min_distance("pho")))
    });

    // Benchmark with alternation pattern
    let nfa_alt = compile(&parse("(ph|f)one").expect("parse")).expect("compile");
    let product_alt = ProductAutomatonChar::new(nfa_alt, 5);

    group.bench_function("alternation_exact", |b| {
        b.iter(|| black_box(product_alt.min_distance("phone")))
    });

    group.bench_function("alternation_variant", |b| {
        b.iter(|| black_box(product_alt.min_distance("fone")))
    });

    group.finish();
}

// ============================================================================
// Incremental Matcher Benchmarks
// ============================================================================

fn bench_incremental_matcher(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_matcher");

    let nfa = compile(&parse("hello").expect("parse")).expect("compile");

    group.bench_function("feed_string", |b| {
        b.iter(|| {
            let mut matcher = IncrementalMatcherChar::new(nfa.clone());
            matcher.feed_str("hello");
            black_box(matcher.is_accepting())
        })
    });

    group.bench_function("feed_chars", |b| {
        b.iter(|| {
            let mut matcher = IncrementalMatcherChar::new(nfa.clone());
            for c in "hello".chars() {
                matcher.feed(c);
            }
            black_box(matcher.is_accepting())
        })
    });

    // Longer strings
    for len in [10, 50, 100] {
        let pattern: String = (0..len / 5).map(|_| "hello").collect();
        let nfa_long = compile(&parse(&pattern).expect("parse")).expect("compile");
        let input: String = (0..len / 5).map(|_| "hello").collect();

        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_with_input(BenchmarkId::new("length", len), &input, |b, input| {
            b.iter(|| {
                let mut matcher = IncrementalMatcherChar::new(nfa_long.clone());
                matcher.feed_str(input);
                black_box(matcher.is_accepting())
            })
        });
    }

    group.finish();
}

// ============================================================================
// Memoized Matcher Benchmarks
// ============================================================================

fn bench_memoized_matcher(c: &mut Criterion) {
    let mut group = c.benchmark_group("memoized_matcher");

    let nfa = compile(&parse("test").expect("parse")).expect("compile");
    let product = ProductAutomatonChar::new(nfa.clone(), 5);
    let mut matcher = MemoizedMatcherChar::new(product.clone(), 1000);

    // Warm up cache
    for _ in 0..10 {
        matcher.accepts("test");
        matcher.accepts("tst");
        matcher.accepts("testing");
    }

    group.bench_function("cached_hit", |b| {
        b.iter(|| black_box(matcher.accepts("test")))
    });

    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            let mut fresh = MemoizedMatcherChar::new(product.clone(), 1000);
            black_box(fresh.accepts("test"))
        })
    });

    group.finish();
}

// ============================================================================
// Verified Rules Benchmarks
// ============================================================================

fn bench_verified_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("verified_rules");

    // Build Zompist NFA
    group.bench_function("build_zompist_nfa", |b| {
        b.iter(|| black_box(zompist_nfa_char()))
    });

    let zompist_nfa = zompist_nfa_char();

    // Pattern recognition
    let patterns = ["ch", "sh", "ph", "gh", "th", "qu", "kw", "c", "g", "e", "x", "y"];

    group.throughput(Throughput::Elements(patterns.len() as u64));
    group.bench_function("pattern_recognition", |b| {
        b.iter(|| {
            for pattern in &patterns {
                black_box(zompist_nfa.accepts(pattern));
            }
        })
    });

    group.finish();
}

// ============================================================================
// PhoneticTransducer Benchmarks
// ============================================================================

fn bench_phonetic_transducer(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonetic_transducer");

    // Small dictionary
    let small_dict = DoubleArrayTrieChar::from_terms([
        "phone", "phones", "phoned", "phoning", "fone", "fones",
    ]);

    // Medium dictionary
    let medium_terms: Vec<_> = (0..1000).map(|i| format!("word{}", i)).collect();
    let medium_refs: Vec<_> = medium_terms.iter().map(|s| s.as_str()).collect();
    let medium_dict = DoubleArrayTrieChar::from_terms(medium_refs);

    let nfa = compile(&parse("phone").expect("parse")).expect("compile");

    let small_transducer = PhoneticTransducerChar::new(small_dict.clone(), nfa.clone(), 2);
    let medium_transducer = PhoneticTransducerChar::new(medium_dict, nfa.clone(), 2);

    group.bench_function("small_dict_query", |b| {
        b.iter(|| {
            let results: Vec<_> = small_transducer.query("phone").collect();
            black_box(results)
        })
    });

    group.bench_function("small_dict_sorted", |b| {
        b.iter(|| {
            let results = small_transducer.query_sorted("phone");
            black_box(results)
        })
    });

    group.bench_function("medium_dict_query", |b| {
        b.iter(|| {
            let results: Vec<_> = medium_transducer.query("word500").collect();
            black_box(results)
        })
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_nfa_pattern_matching,
    bench_lazy_dfa,
    bench_product_automaton,
    bench_incremental_matcher,
    bench_memoized_matcher,
    bench_verified_rules,
    bench_phonetic_transducer,
);

criterion_main!(benches);
