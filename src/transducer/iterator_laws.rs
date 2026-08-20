use super::*;
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::Dictionary;
use std::collections::HashSet;
use std::iter::FusedIterator;

fn assert_query_iterator_laws<I>(mut iterator: I)
where
    I: FusedIterator,
{
    assert_eq!(iterator.size_hint(), (0, None));
    while iterator.next().is_some() {
        assert_eq!(iterator.size_hint(), (0, None));
    }
    assert!(iterator.next().is_none());
    assert!(iterator.next().is_none());
    assert_eq!(iterator.size_hint(), (0, None));
}

#[test]
fn query_families_share_conservative_fused_iterator_laws() {
    let dictionary = DoubleArrayTrie::from_terms(["cat", "bat", "coat", "dog"]);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);

    assert_query_iterator_laws(transducer.query("cat", 2));
    assert_query_iterator_laws(transducer.query_with_distance("cat", 2));
    assert_query_iterator_laws(transducer.query_units_weighted(
        b"cat",
        2.0,
        OperationCostsF64::standard(),
    ));

    let affine = AffineGapParams::new(1.0, 1.0, 1.0).expect("valid affine costs");
    assert_query_iterator_laws(
        transducer
            .query_affine("cat", 2.0, affine)
            .expect("exact affine budget"),
    );

    assert_query_iterator_laws(transducer.query_ordered("cat", 2));
    assert_query_iterator_laws(transducer.query_ordered("cat", 2).filter(|_| true));
    assert_query_iterator_laws(transducer.query_ordered("cat", 2).prefix());
    assert_query_iterator_laws(
        transducer
            .query_mode(
                "cat",
                MatchMode::Range {
                    min_distance: 1,
                    max_distance: 2,
                },
            )
            .expect("valid match mode"),
    );

    assert_query_iterator_laws(transducer.query_with_pruner("cat", 2, NoPruning));
    assert_query_iterator_laws(SubsequenceQueryIterator::from_dictionary(
        transducer.dictionary(),
        b"ct".to_vec(),
    ));
    assert_query_iterator_laws(PriorityQueryIterator::new(
        transducer.dictionary().root(),
        "cat",
        2,
        Algorithm::Standard,
    ));
    assert_query_iterator_laws(
        ContextualQueryIterator::from_dictionary(
            transducer.dictionary(),
            b"cat".to_vec(),
            2.0,
            OperationCostsF64::standard(),
        )
        .expect("valid contextual costs"),
    );

    let mut language = language::SmallDfa::new();
    let accepting = language.add_state(true).expect("state within test limit");
    language
        .add_transition(0, b'c', accepting)
        .expect("valid test transition");
    assert_query_iterator_laws(transducer.query_language(language, 2));
}

#[test]
fn value_query_adapters_share_conservative_fused_iterator_laws() {
    let dictionary: DoubleArrayTrie<u64> =
        DoubleArrayTrie::from_terms_with_values([("cat", 10), ("bat", 5), ("dog", 1)]);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);

    assert_query_iterator_laws(transducer.query_filtered("cat", 2, |_| true));
    assert_query_iterator_laws(transducer.query_values("cat", 2));
    assert_query_iterator_laws(transducer.query_suggestions("cat", 2, LogFrequencyScorer));

    let values = HashSet::from([5, 10]);
    assert_query_iterator_laws(transducer.query_by_value_set("cat", 2, &values));
}

#[cfg(feature = "pathmap-backend")]
#[test]
fn zipper_query_shares_conservative_fused_iterator_laws() {
    use libdictenstein::pathmap::zipper::PathMapZipper;
    use libdictenstein::pathmap::PathMapDictionary;

    let dictionary = PathMapDictionary::<()>::new();
    dictionary.insert("cat");
    dictionary.insert("bat");
    assert_query_iterator_laws(ZipperQueryIterator::new(
        PathMapZipper::new_from_dict(&dictionary),
        "cat",
        1,
        Algorithm::Standard,
    ));
}

#[cfg(feature = "phonetic-rules")]
#[test]
fn phonetic_query_adapters_share_conservative_fused_iterator_laws() {
    use crate::phonetic::nfa::compiler::{compile, compile_bytes};
    use crate::phonetic::regex::{parse, parse_bytes};
    use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;

    let char_nfa = compile(&parse("cat").expect("parse character pattern"))
        .expect("compile character pattern");
    let char_transducer =
        PhoneticTransducerChar::new(DoubleArrayTrieChar::from_terms(["cat", "bat"]), char_nfa, 1);
    assert_query_iterator_laws(char_transducer.query("cat"));
    #[cfg(feature = "benchmark-controls")]
    assert_query_iterator_laws(char_transducer.query_legacy_unit_cost_retention_control("cat"));

    let mapped_char_transducer = PhoneticTransducerChar::new(
        DoubleArrayTrieChar::from_terms_with_values([("cat", 1_u64), ("bat", 2)]),
        compile(&parse("cat").expect("parse mapped character pattern"))
            .expect("compile mapped character pattern"),
        1,
    );
    assert_query_iterator_laws(mapped_char_transducer.query_values("cat"));

    let byte_nfa = compile_bytes(&parse_bytes(b"cat").expect("parse byte pattern"))
        .expect("compile byte pattern");
    let byte_transducer =
        PhoneticTransducer::new(DoubleArrayTrie::from_terms(["cat", "bat"]), byte_nfa, 1);
    assert_query_iterator_laws(byte_transducer.query(b"cat"));

    let mapped_byte_transducer = PhoneticTransducer::new(
        DoubleArrayTrie::from_terms_with_values([("cat", 1_u64), ("bat", 2)]),
        compile_bytes(&parse_bytes(b"cat").expect("parse mapped byte pattern"))
            .expect("compile mapped byte pattern"),
        1,
    );
    assert_query_iterator_laws(mapped_byte_transducer.query_values(b"cat"));
}
