//! Word-level (term-id) correction over a `u64` alphabet.
//!
//! The Levenshtein engine is unit-generic: the same automaton that corrects
//! *characters* of a word can correct *words* of a sentence, once each word is
//! mapped to an integer term-id. This example builds a small dictionary of
//! term-id sequences (think: known n-grams of a vocabulary) and corrects a
//! corrupted sequence three ways:
//!
//! 1. `query_units` — integer edit distance over `&[u64]`, matches returned as
//!    `Vec<u64>` (no lossy string round-trip).
//! 2. `query_units_weighted` — the same search but pruned on a per-operation
//!    **`f64` cost budget** (e.g. a `−log P` language-model weight), so an
//!    expensive edit can be ranked out even within the integer distance bound.
//! 3. `query_units_values` — returns the dictionary's stored **value** (here an
//!    n-gram frequency / id) at each corrected sequence, in one pass.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example u64_word_correction
//! ```

use libdictenstein::dynamic_dawg::DynamicDawgU64;
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::{OperationCostsF64, UnitCandidate, UnitCandidateF64};

/// Pretty-print a term-id sequence.
fn show(units: &[u64]) -> String {
    let inner = units
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn main() {
    // A vocabulary-trie of known word-id sequences (n-grams). The value stored
    // at each sequence is a stand-in for an n-gram frequency / model score.
    let dict: DynamicDawgU64<u64> = DynamicDawgU64::new();
    dict.insert_sequence_with_value(&[10, 20, 30], 900); // "the quick fox"
    dict.insert_sequence_with_value(&[10, 20, 40], 150); // "the quick dog"
    dict.insert_sequence_with_value(&[10, 20, 30, 50], 40); // "the quick fox jumps"
    dict.insert_sequence_with_value(&[10, 20], 500); // "the quick"
    dict.insert_sequence_with_value(&[99, 98, 97], 5); // unrelated

    let transducer = Transducer::new(dict, Algorithm::Standard);

    // A corrupted observation: the third word-id is wrong (30 was seen as 40).
    let observed = [10u64, 20, 40];
    println!("Observed word-id sequence: {}", show(&observed));
    println!();

    // ── 1. Integer edit-distance correction ────────────────────────────────
    println!("query_units (integer distance ≤ 1) — corrected sequences:");
    let mut hits: Vec<Vec<u64>> = transducer.query_units(&observed, 1).collect();
    hits.sort();
    for h in &hits {
        println!("    {}", show(h));
    }
    println!();

    // ── 2. Same search, but reported with each candidate's edit distance ────
    println!("query_units_with_distance (≤ 1) — (sequence, distance):");
    let mut ranked: Vec<UnitCandidate<u64>> =
        transducer.query_units_with_distance(&observed, 1).collect();
    ranked.sort_by(|a, b| {
        a.distance
            .cmp(&b.distance)
            .then_with(|| a.term.cmp(&b.term))
    });
    for c in &ranked {
        println!("    {} @ distance {}", show(&c.term), c.distance);
    }
    println!();

    // ── 3. Weighted (f64) correction ───────────────────────────────────────
    // Make substitutions expensive (2.0) while keeping insertion/deletion at
    // 1.0. Within a 1.0 budget the *exact* match survives, but the lone
    // substitution (30↔40) that a unit-cost search would accept is now pruned.
    let mut costs = OperationCostsF64::standard();
    costs.substitution = 2.0;
    println!("query_units_weighted (cost ≤ 1.0, substitution = 2.0) — (sequence, cost):");
    let mut weighted: Vec<UnitCandidateF64<u64>> = transducer
        .query_units_weighted(&observed, 1.0, costs)
        .collect();
    weighted.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.term.cmp(&b.term))
    });
    for c in &weighted {
        println!("    {} @ cost {:.1}", show(&c.term), c.distance);
    }
    println!();

    // ── 4. Value-returning correction ──────────────────────────────────────
    // One pass yields the corrected sequence, its edit distance, AND the stored
    // value (n-gram frequency / id) — no second lookup.
    println!("query_units_values (≤ 1) — (sequence, distance, stored value):");
    let mut valued: Vec<(Vec<u64>, usize, u64)> =
        transducer.query_units_values(&observed, 1).collect();
    valued.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    for (seq, dist, value) in &valued {
        println!("    {} @ distance {dist}, value {value}", show(seq));
    }
    println!();

    // A caller can now pick the best correction by combining edit distance with
    // the model score (value) — e.g. prefer the highest-frequency sequence among
    // the closest matches.
    if let Some((best_seq, best_dist, best_value)) = valued
        .iter()
        .min_by(|a, b| a.1.cmp(&b.1).then_with(|| b.2.cmp(&a.2)))
    {
        println!(
            "Best correction (closest, then highest value): {} (distance {best_dist}, value {best_value})",
            show(best_seq)
        );
    }
}
