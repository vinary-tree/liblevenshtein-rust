/// Cross-validation test: Universal vs Parameterized Levenshtein Automata
///
/// This test validates that the Universal Levenshtein Automaton (Mitankin 2005)
/// produces the same acceptance results as the Parameterized Automaton (Schulz & Mihov 2002).
///
/// The universal automaton should accept/reject the same word/input pairs as the
/// parameterized version, just without needing to be constructed for each specific word.
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::universal::{
    MergeAndSplit as UniversalMergeAndSplit, Standard as UniversalStandard,
    Transposition as UniversalTransposition, UniversalAutomaton,
};
use liblevenshtein::transducer::{Algorithm, Restricted, SubstitutionSet};

fn parameterized_accepts_with_policy(
    word: &str,
    input: &str,
    max_distance: usize,
    algorithm: Algorithm,
    substitutions: &SubstitutionSet,
) -> bool {
    let dictionary: DynamicDawg<()> = DynamicDawg::default();
    dictionary.insert(word);
    let transducer = Transducer::with_policy(dictionary, algorithm, Restricted::new(substitutions));
    transducer
        .query(input, max_distance)
        .any(|candidate| candidate == word)
}

#[test]
fn policy_aware_universal_variants_match_singleton_dictionary_transducers() {
    let mut substitutions = SubstitutionSet::new();
    substitutions.allow_byte(b'k', b'c');
    let policy = Restricted::new(&substitutions);

    let standard = UniversalAutomaton::<UniversalStandard, _>::with_policy(0, policy);
    assert_eq!(
        standard.accepts_bytes(b"kit", b"cit"),
        parameterized_accepts_with_policy("kit", "cit", 0, Algorithm::Standard, &substitutions,)
    );

    let transposition = UniversalAutomaton::<UniversalTransposition, _>::with_policy(1, policy);
    assert_eq!(
        transposition.accepts_bytes(b"abk", b"bac"),
        parameterized_accepts_with_policy(
            "abk",
            "bac",
            1,
            Algorithm::Transposition,
            &substitutions,
        )
    );

    let merge_and_split = UniversalAutomaton::<UniversalMergeAndSplit, _>::with_policy(1, policy);
    assert_eq!(
        merge_and_split.accepts_bytes(b"ak", b"abc"),
        parameterized_accepts_with_policy("ak", "abc", 1, Algorithm::MergeAndSplit, &substitutions,)
    );
}

#[test]
fn test_substitution_test_to_text() {
    // This is the failing case: "test" → "text" (one substitution)
    let word = "test";
    let input = "text";
    let max_distance = 2;

    // Test with universal automaton
    let universal = UniversalAutomaton::<UniversalStandard>::new(max_distance);
    let universal_result = universal.accepts(word, input);

    // Test with parameterized automaton
    let dict: DynamicDawg<()> = DynamicDawg::default();
    dict.insert(word);

    let transducer = Transducer::standard(dict);

    let parameterized_results: Vec<_> = transducer.query(input, max_distance as usize).collect();

    let parameterized_result = parameterized_results.iter().any(|w| w == word);

    println!(
        "Word: '{}', Input: '{}', Max distance: {}",
        word, input, max_distance
    );
    println!("Universal result: {}", universal_result);
    println!("Parameterized result: {}", parameterized_result);

    if parameterized_result {
        println!("Parameterized matches: {:?}", parameterized_results);
    }

    assert_eq!(
        universal_result, parameterized_result,
        "Universal and parameterized automata disagree on '{}' → '{}' (distance ≤ {})",
        word, input, max_distance
    );
}

#[test]
fn test_substitution_test_to_best() {
    let word = "test";
    let input = "best";
    let max_distance = 2;

    let universal = UniversalAutomaton::<UniversalStandard>::new(max_distance);
    let universal_result = universal.accepts(word, input);

    let dict: DynamicDawg<()> = DynamicDawg::default();
    dict.insert(word);

    let transducer = Transducer::standard(dict);

    let parameterized_results: Vec<_> = transducer.query(input, max_distance as usize).collect();
    let parameterized_result = parameterized_results.iter().any(|w| w == word);

    println!(
        "Word: '{}', Input: '{}', Max distance: {}",
        word, input, max_distance
    );
    println!("Universal result: {}", universal_result);
    println!("Parameterized result: {}", parameterized_result);

    assert_eq!(
        universal_result, parameterized_result,
        "Universal and parameterized automata disagree on '{}' → '{}' (distance ≤ {})",
        word, input, max_distance
    );
}

#[test]
fn test_cross_validation_suite() {
    // Test a variety of cases
    let test_cases = vec![
        ("test", "test", 2, true),           // Exact match
        ("test", "text", 2, true),           // 1 substitution
        ("test", "best", 2, true),           // 1 substitution
        ("test", "tet", 2, true),            // 1 deletion
        ("test", "teast", 2, true),          // 1 insertion
        ("test", "hello", 2, false),         // Too many edits
        ("algorithm", "algorythm", 2, true), // 1 substitution
        ("", "", 2, true),                   // Empty strings
        ("ab", "", 2, true),                 // Delete all
        ("", "ab", 2, true),                 // Insert all
    ];

    for (word, input, max_distance, expected) in test_cases {
        let universal = UniversalAutomaton::<UniversalStandard>::new(max_distance);
        let universal_result = universal.accepts(word, input);

        let dict: DynamicDawg<()> = DynamicDawg::default();
        dict.insert(word);

        let transducer = Transducer::standard(dict);

        let parameterized_results: Vec<_> =
            transducer.query(input, max_distance as usize).collect();
        let parameterized_result = parameterized_results.iter().any(|w| w == word);

        // Check both agree with expected
        assert_eq!(
            universal_result, expected,
            "Universal automaton wrong for '{}' → '{}' (expected {})",
            word, input, expected
        );

        assert_eq!(
            parameterized_result, expected,
            "Parameterized automaton wrong for '{}' → '{}' (expected {})",
            word, input, expected
        );

        // Check they agree with each other
        assert_eq!(
            universal_result, parameterized_result,
            "Automata disagree on '{}' → '{}': universal={}, parameterized={}",
            word, input, universal_result, parameterized_result
        );
    }
}
