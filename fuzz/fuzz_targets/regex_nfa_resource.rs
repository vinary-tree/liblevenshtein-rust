#![no_main]

use libdictenstein::double_array_trie::DoubleArrayTrieChar;
use libfuzzer_sys::fuzz_target;
use liblevenshtein::transducer::{Algorithm, Transducer};

fuzz_target!(|data: &[u8]| {
    let pattern = String::from_utf8_lossy(data);
    let dictionary = DoubleArrayTrieChar::from_terms([""]);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);

    // `query_regex` preflights source and expanded Thompson-state counts before
    // a compiled automaton can enter the 4,096-state product.
    if let Ok(iterator) = transducer.query_regex(&pattern, 1) {
        drop(iterator);
    }
});
