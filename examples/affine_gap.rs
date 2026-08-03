//! Exact affine-gap search with one opening charge per contiguous gap.

use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::distance::affine_gap_distance;
use liblevenshtein::transducer::{AffineGapParams, Algorithm, Transducer};

fn main() {
    let dictionary = DoubleArrayTrie::from_terms(["a", "abcd", "kitten", "sitting"]);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);
    let costs = AffineGapParams::new(3.0, 2.0, 10.0).expect("costs are exact decimals");

    let candidates: Vec<_> = transducer
        .query_affine("a", 9.0, costs)
        .expect("budget is exact at the cost scale")
        .collect();
    let expanded = candidates
        .iter()
        .find(|candidate| candidate.term == "abcd")
        .expect("one length-three dictionary gap is affordable");

    assert_eq!(expanded.distance, 9.0); // gap-open 3 + 3 extensions * 2
    assert_eq!(expanded.scaled_distance, 9);
    assert_eq!(affine_gap_distance("a", "abcd", costs), Some(9));

    println!("{}: {}", expanded.term, expanded.distance);
}
