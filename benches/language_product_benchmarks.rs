//! Phase-0 baseline for language-product frontier canonicalization.
//!
//! The legacy arm retains a separate state for each `(NFA state set, cost)`
//! pair. The candidate arm is the shipped generic `LanguageProduct`: it unions
//! all states at one exact cost and removes states already present more cheaply.
//! Both arms use the same NFA transitions; the comparison isolates the product
//! frontier representation and expansion count.

#![cfg(feature = "phonetic-rules")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use liblevenshtein::phonetic::nfa::{compile, NFAChar, ProductAutomatonChar, StateSet};
use liblevenshtein::phonetic::regex::parse;
use liblevenshtein::transducer::language::{Frontier, LanguageProduct};
use std::hint::black_box;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrontierResult {
    accepting_distance: Option<u8>,
    states_expanded: usize,
    final_frontier: usize,
}

fn run_legacy(product: &ProductAutomatonChar, input: &str) -> FrontierResult {
    let mut frontier = product.initial_frontier();
    let mut states_expanded = 0;
    for ch in input.chars() {
        states_expanded += frontier.len();
        frontier = product.transition_frontier(&frontier, ch);
    }
    FrontierResult {
        accepting_distance: product.min_accepting_distance(&frontier),
        states_expanded,
        final_frontier: frontier.len(),
    }
}

fn active_states(frontier: &Frontier<StateSet>) -> usize {
    (0..frontier.len())
        .filter_map(|level| frontier.level(level))
        .map(StateSet::len)
        .sum()
}

fn run_cost_indexed(product: &LanguageProduct<char, NFAChar>, input: &str) -> FrontierResult {
    let mut frontier = product.initial_frontier();
    let mut states_expanded = 0;
    for ch in input.chars() {
        states_expanded += active_states(&frontier);
        frontier = product.step(&frontier, &ch);
    }
    FrontierResult {
        accepting_distance: product.min_accepting_distance(&frontier),
        states_expanded,
        final_frontier: active_states(&frontier),
    }
}

fn benchmark_language_product(c: &mut Criterion) {
    let cases = [
        ("literal", "characteristic", "charactaristik"),
        ("alternation", "(ph|f|v)(one|un|on)e?", "fone"),
        (
            "branching",
            "(a|ab|abc|abcd|abcde)*(x|xy|xyz)",
            "abcdeabcdeabxyz",
        ),
    ];

    let mut group = c.benchmark_group("language_product_frontier");
    for (name, pattern, input) in cases {
        let nfa = compile(&parse(pattern).expect("benchmark pattern must parse"))
            .expect("benchmark pattern must compile");
        let legacy_product = ProductAutomatonChar::new(nfa.clone(), 3);
        let canonical_product = LanguageProduct::new(nfa, 3);
        let legacy = run_legacy(&legacy_product, input);
        let canonical = run_cost_indexed(&canonical_product, input);
        assert_eq!(canonical.accepting_distance, legacy.accepting_distance);
        eprintln!(
            "frontier pattern={name} legacy_expanded={} canonical_expanded={} legacy_final={} canonical_final={}",
            legacy.states_expanded,
            canonical.states_expanded,
            legacy.final_frontier,
            canonical.final_frontier,
        );

        group.bench_with_input(
            BenchmarkId::new("legacy_cost_distinct", name),
            &(),
            |b, _| b.iter(|| black_box(run_legacy(&legacy_product, black_box(input)))),
        );
        group.bench_with_input(BenchmarkId::new("canonical_min_cost", name), &(), |b, _| {
            b.iter(|| black_box(run_cost_indexed(&canonical_product, black_box(input))))
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_language_product);
criterion_main!(benches);
