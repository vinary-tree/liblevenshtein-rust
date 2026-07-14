//! Regression oracle for the weighted-`f64` subsumption soundness bug
//! (pgmcp `weighted-f64-subsumption-over-prunes-...`, Phase 2 item 1764).
//!
//! `PositionF64::subsumes` previously tested `|i - j| <= (f - e)`, implicitly
//! assuming each term-index realignment step costs exactly `1.0`. With a custom
//! [`OperationCostsF64`] whose insertion/deletion cost exceeds `1.0`, that bound
//! over-subsumes and can delete a position that leads to the sole in-budget
//! match, yielding a *missed term* or an *overestimated distance*.
//!
//! The fix scales the index difference by `max(insertion, deletion)`. These
//! tests pin the fix by cross-checking the weighted automaton against a
//! reference weighted-Levenshtein dynamic program over many randomized cases
//! (including insertion = deletion > 1), plus the exact counterexample from the
//! audit. Insertion and deletion are kept equal so the reference DP is
//! direction-independent; that already exercises the `max(ins, del) > 1` regime
//! in which the old bound was unsound.

use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::Dictionary;
use liblevenshtein::transducer::{Algorithm, CandidateIteratorF64, OperationCostsF64};

const EPS: f64 = 1e-6;

/// Reference weighted Levenshtein distance (Standard algorithm: substitution,
/// insertion, deletion; match is free). With `indel = insertion = deletion`
/// the metric is symmetric, so argument order does not matter.
fn weighted_levenshtein(a: &[char], b: &[char], sub: f64, indel: f64) -> f64 {
    let m = a.len();
    let n = b.len();
    let mut prev: Vec<f64> = (0..=n).map(|j| j as f64 * indel).collect();
    let mut curr: Vec<f64> = vec![0.0; n + 1];
    for i in 1..=m {
        curr[0] = i as f64 * indel;
        for j in 1..=n {
            let sub_cost = if a[i - 1] == b[j - 1] { 0.0 } else { sub };
            curr[j] = (prev[j - 1] + sub_cost)
                .min(prev[j] + indel)
                .min(curr[j - 1] + indel);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Run the weighted automaton over `terms` for `query` and collect
/// `term -> distance` for every returned candidate.
fn automaton_matches(
    terms: &[String],
    query: &str,
    max_cost: f64,
    costs: OperationCostsF64,
) -> Vec<(String, f64)> {
    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
    let dict = DoubleArrayTrie::from_terms(refs);
    CandidateIteratorF64::new(
        dict.root(),
        query.to_string(),
        max_cost,
        Algorithm::Standard,
        costs,
    )
    .map(|c| (c.term, c.distance))
    .collect()
}

/// Assert full soundness of the automaton against the reference DP for one case.
fn assert_sound(terms: &[String], query: &str, sub: f64, indel: f64, max_cost: f64) {
    let costs = OperationCostsF64::custom(sub, indel, indel, 1.0, 1.0, 1.0);
    let matches = automaton_matches(terms, query, max_cost, costs);
    let query_chars: Vec<char> = query.chars().collect();

    // (1) Completeness + exactness: every term whose true weighted distance is
    // within budget MUST be returned with exactly that distance. This is the
    // property the old over-pruning broke.
    for term in terms {
        let term_chars: Vec<char> = term.chars().collect();
        let truth = weighted_levenshtein(&query_chars, &term_chars, sub, indel);
        let found = matches.iter().find(|(t, _)| t == term).map(|(_, d)| *d);
        if truth <= max_cost - EPS {
            let got = found.unwrap_or_else(|| {
                panic!(
                    "MISSED in-budget term {term:?} for query {query:?} \
                     (truth={truth}, max_cost={max_cost}, sub={sub}, indel={indel})"
                )
            });
            assert!(
                (got - truth).abs() < 1e-6,
                "distance mismatch for {term:?} query {query:?}: automaton={got} truth={truth} \
                 (sub={sub}, indel={indel}, max_cost={max_cost})"
            );
        }
    }

    // (2) Soundness of results: everything returned is within budget and equals
    // its true distance (no spurious or misreported matches).
    for (term, dist) in &matches {
        let term_chars: Vec<char> = term.chars().collect();
        let truth = weighted_levenshtein(&query_chars, &term_chars, sub, indel);
        assert!(
            *dist <= max_cost + EPS,
            "returned {term:?} with distance {dist} > max_cost {max_cost}"
        );
        assert!(
            (dist - truth).abs() < 1e-6,
            "reported distance {dist} != truth {truth} for {term:?} query {query:?}"
        );
    }
}

/// The exact counterexample from the audit: substitution cheap, indels = 2.0.
/// Query "ab" vs term "b" has true distance = one deletion = 2.0; with
/// `max_cost = 2.0` it must be found. The old unit-cost bound could prune the
/// position that reaches it.
#[test]
fn audit_counterexample_indel_cost_two_is_not_over_pruned() {
    let terms: Vec<String> = ["b", "ab", "abc", "xy"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    // indel = 2.0, substitution = 1.0.
    assert_sound(&terms, "ab", 1.0, 2.0, 2.0);
    // A tighter budget where only the exact and the single-substitution match fit.
    assert_sound(&terms, "ab", 1.0, 2.0, 1.0);
}

/// Deterministic pseudo-random cross-check across many (dict, query, cost, budget)
/// combinations, emphasising insertion = deletion > 1 where the old bound failed.
#[test]
fn weighted_automaton_matches_reference_dp_fuzz() {
    // Small, self-contained LCG (avoids Date/rand nondeterminism; fully reproducible).
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = |bound: u64| -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };

    const ALPHABET: &[u8] = b"abcd";
    let subs = [0.5f64, 1.0, 1.5, 2.0];
    let indels = [1.0f64, 1.5, 2.0, 3.0];
    let budgets = [0.0f64, 1.0, 1.5, 2.0, 3.0, 4.0];

    let rand_word = |next: &mut dyn FnMut(u64) -> u64| -> String {
        let len = next(6) as usize; // 0..=5, exercises the empty word
        (0..len)
            .map(|_| ALPHABET[next(ALPHABET.len() as u64) as usize] as char)
            .collect()
    };

    for _ in 0..1200 {
        let dict_size = 3 + next(6) as usize; // 3..=8 terms
        let mut terms: Vec<String> = Vec::with_capacity(dict_size);
        for _ in 0..dict_size {
            let w = rand_word(&mut next);
            if !terms.contains(&w) {
                terms.push(w);
            }
        }
        if terms.is_empty() {
            continue;
        }
        let query = rand_word(&mut next);
        let sub = subs[next(subs.len() as u64) as usize];
        let indel = indels[next(indels.len() as u64) as usize];
        let max_cost = budgets[next(budgets.len() as u64) as usize];
        assert_sound(&terms, &query, sub, indel, max_cost);
    }
}

/// Unit costs (insertion = deletion = 1) must still behave exactly as classic
/// Levenshtein — the fix must not perturb the common case.
#[test]
fn unit_costs_still_exact() {
    let terms: Vec<String> = ["test", "best", "rest", "tent", "tests", "te"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for &max_cost in &[0.0f64, 1.0, 2.0, 3.0] {
        assert_sound(&terms, "test", 1.0, 1.0, max_cost);
    }
}
