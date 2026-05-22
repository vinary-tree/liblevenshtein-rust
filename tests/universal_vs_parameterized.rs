/// Cross-validation test: Universal vs Parameterized Levenshtein Automata
///
/// This test validates that the Universal Levenshtein Automaton (Mitankin 2005)
/// produces the same acceptance results as the Parameterized Automaton (Schulz & Mihov 2002).
///
/// The universal automaton should accept/reject the same word/input pairs as the
/// parameterized version, just without needing to be constructed for each specific word.
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::universal::{Standard as UniversalStandard, UniversalAutomaton};

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
