use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::distance::{damerau_levenshtein_distance, transposition_distance};
use liblevenshtein::prelude::*;

fn main() {
    let dictionary = DoubleArrayTrie::from_terms(["AC", "ABC", "CA", "CBA"]);
    let transducer = Transducer::with_damerau_levenshtein(dictionary);
    let mut matches: Vec<_> = transducer
        .query_with_distance("CA", 2)
        .map(|candidate| (candidate.term, candidate.distance))
        .collect();
    matches.sort();

    assert!(matches.contains(&("ABC".to_owned(), 2)));
    assert_eq!(damerau_levenshtein_distance("CA", "ABC"), 2);
    assert_eq!(transposition_distance("CA", "ABC"), 3);

    println!("true Damerau matches for CA at budget 2: {matches:?}");
}
