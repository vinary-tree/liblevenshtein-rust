#![no_main]

use libdictenstein::double_array_trie::DoubleArrayTrieChar;
use libfuzzer_sys::fuzz_target;
use liblevenshtein::distance::damerau_levenshtein_distance;
use liblevenshtein::transducer::{Algorithm, Transducer};
use std::collections::BTreeMap;

fuzz_target!(|data: &[u8]| {
    let Some((&budget, body)) = data.split_first() else {
        return;
    };
    // Keep each generated dictionary bounded while leaving the complete u8
    // budget domain attacker-controlled.
    let fields: Vec<String> = body[..body.len().min(512)]
        .split(|byte| *byte == 0)
        .take(9)
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .collect();
    let Some((query, terms)) = fields.split_last() else {
        return;
    };

    // The reference distance counts Unicode scalar values.  Use the matching
    // character-labelled backend so arbitrary valid UTF-8 (including the
    // replacement characters introduced by `from_utf8_lossy`) has the same
    // unit on both sides of the differential oracle.
    let dictionary = DoubleArrayTrieChar::from_terms(terms);
    let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);
    let actual: BTreeMap<_, _> = transducer
        .query_with_distance(query, usize::from(budget))
        .map(|candidate| (candidate.term, candidate.distance))
        .collect();
    let expected: BTreeMap<_, _> = terms
        .iter()
        .filter_map(|term| {
            let distance = damerau_levenshtein_distance(query, term);
            (distance <= usize::from(budget)).then(|| (term.clone(), distance))
        })
        .collect();
    assert_eq!(actual, expected);
});
