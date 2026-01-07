//! Tests for DynamicDawgChar with Unicode support.
//!
//! These tests verify that DynamicDawgChar correctly handles Unicode characters
//! at the character level (not byte level), providing correct Levenshtein distances
//! for multi-byte UTF-8 sequences while supporting dynamic insert and remove operations.

use libdictenstein::dynamic_dawg_char::DynamicDawgChar;
use libdictenstein::MappedDictionary;
use liblevenshtein::prelude::*;

// ===== Basic Dictionary Operations =====

#[test]
fn test_dynamic_dawg_char_empty_query_unicode() {
    println!("\n=== DynamicDawgChar: Empty Query → Unicode Character ===\n");

    // This was the original problem: "" → "¡" should be distance 1, not 2
    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["¡"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"¡\"]");
    println!("Query: \"\" (empty)");
    println!("Max distance: 1");
    println!("Character '¡': {:?}", '¡');
    println!("Bytes: {:?} (length: {})\n", "¡".as_bytes(), "¡".len());

    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Results: {:?}\n", results);

    assert!(
        results.contains(&"¡".to_string()),
        "Empty query should match \"¡\" at distance 1 (one character insertion)"
    );

    println!("✅ SUCCESS: Char-level correctly treats \"¡\" as distance 1 from empty string");
}

#[test]
fn test_dynamic_dawg_char_exact_match() {
    println!("\n=== DynamicDawgChar: Exact Match ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["café", "naïve", "résumé"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"café\", \"naïve\", \"résumé\"]");

    // Exact matches at distance 0
    let results: Vec<_> = transducer.query("café", 0).collect();
    println!("Query \"café\" at distance 0: {:?}", results);
    assert!(results.contains(&"café".to_string()));

    let results: Vec<_> = transducer.query("naïve", 0).collect();
    println!("Query \"naïve\" at distance 0: {:?}", results);
    assert!(results.contains(&"naïve".to_string()));

    println!("\n✅ SUCCESS: Exact matches work correctly");
}

#[test]
fn test_dynamic_dawg_char_one_edit_distance() {
    println!("\n=== DynamicDawgChar: One Edit Distance ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["café", "cafe"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"café\", \"cafe\"]");

    // Query "cafe" at distance 1 should find "café" (substitute e→é)
    let results: Vec<_> = transducer.query("cafe", 1).collect();
    println!("Query \"cafe\" at distance 1: {:?}", results);
    assert!(results.contains(&"cafe".to_string())); // exact match
    assert!(results.contains(&"café".to_string())); // one substitution

    // Query "café" at distance 1 should find "cafe" (substitute é→e)
    let results: Vec<_> = transducer.query("café", 1).collect();
    println!("Query \"café\" at distance 1: {:?}", results);
    assert!(results.contains(&"café".to_string())); // exact match
    assert!(results.contains(&"cafe".to_string())); // one substitution

    println!("\n✅ SUCCESS: Accented characters are single character edits");
}

// ===== Unicode Character Types =====

#[test]
fn test_dynamic_dawg_char_emoji_distance() {
    println!("\n=== DynamicDawgChar: Emoji Distances ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["🎉"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"🎉\"]");
    println!("Emoji '🎉' is 1 character (4 bytes in UTF-8)\n");

    // Empty query at distance 1 should find solo emoji
    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Empty query at distance 1: {:?}", results);
    assert!(results.contains(&"🎉".to_string()));

    println!("\n✅ SUCCESS: Emoji treated as single character");
}

#[test]
fn test_dynamic_dawg_char_emoji_with_text() {
    println!("\n=== DynamicDawgChar: Emoji with Text ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["hello🎉", "world🌍"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"hello🎉\", \"world🌍\"]");

    // Query "hello" at distance 1 should find "hello🎉" (insert emoji at end)
    let results: Vec<_> = transducer.query("hello", 1).collect();
    println!("Query \"hello\" at distance 1: {:?}", results);
    assert!(results.contains(&"hello🎉".to_string()));

    println!("\n✅ SUCCESS: Emoji appending works as single character insertion");
}

#[test]
fn test_dynamic_dawg_char_cjk_distance() {
    println!("\n=== DynamicDawgChar: CJK Character Distances ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["中", "中文", "中文字"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"中\", \"中文\", \"中文字\"]");
    println!("Each CJK character is 1 character (3 bytes in UTF-8)\n");

    // Empty query at distance 1: should find "中" (1 insertion)
    let results_1: Vec<_> = transducer.query("", 1).collect();
    println!("Distance 1: {:?}", results_1);
    assert!(results_1.contains(&"中".to_string()));

    // Empty query at distance 2: should find "中" and "中文" (2 insertions)
    let results_2: Vec<_> = transducer.query("", 2).collect();
    println!("Distance 2: {:?}", results_2);
    assert!(results_2.contains(&"中".to_string()));
    assert!(results_2.contains(&"中文".to_string()));

    // Query "中" at distance 1: should find "中文" (insert '文')
    let results: Vec<_> = transducer.query("中", 1).collect();
    println!("Query \"中\" at distance 1: {:?}", results);
    assert!(results.contains(&"中".to_string())); // exact match
    assert!(results.contains(&"中文".to_string())); // insert '文'

    println!("\n✅ SUCCESS: CJK characters treated correctly");
}

#[test]
fn test_dynamic_dawg_char_mixed_unicode() {
    println!("\n=== DynamicDawgChar: Mixed Unicode Characters ===\n");

    let dict: DynamicDawgChar<()> =
        DynamicDawgChar::from_terms(vec!["hello", "café", "中文", "🎉", "test123"]);

    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary contains: ASCII, accented, CJK, emoji, alphanumeric\n");

    // Query each with exact match
    assert!(transducer
        .query("hello", 0)
        .collect::<Vec<_>>()
        .contains(&"hello".to_string()));
    assert!(transducer
        .query("café", 0)
        .collect::<Vec<_>>()
        .contains(&"café".to_string()));
    assert!(transducer
        .query("中文", 0)
        .collect::<Vec<_>>()
        .contains(&"中文".to_string()));
    assert!(transducer
        .query("🎉", 0)
        .collect::<Vec<_>>()
        .contains(&"🎉".to_string()));

    println!("✅ SUCCESS: Mixed Unicode content works correctly");
}

// ===== Transposition Algorithm =====

#[test]
fn test_dynamic_dawg_char_transposition_unicode() {
    println!("\n=== DynamicDawgChar: Transposition with Unicode ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["café", "éfac"]);
    let transducer = Transducer::new(dict, Algorithm::Transposition);

    println!("Dictionary: [\"café\", \"éfac\"]");
    println!("Using Transposition algorithm\n");

    // Swap 'é' and 'f' in "éfac" → "féac"
    let results: Vec<_> = transducer.query("féac", 1).collect();
    println!("Query \"féac\" at distance 1: {:?}", results);

    // Should find "éfac" via one transposition
    assert!(results.contains(&"éfac".to_string()));

    println!("\n✅ SUCCESS: Transposition works with Unicode characters");
}

// ===== Query with Distance =====

#[test]
fn test_dynamic_dawg_char_query_with_distance() {
    println!("\n=== DynamicDawgChar: Query with Distance (Unicode) ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["café", "naïve"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Query and get distances
    let results: Vec<_> = transducer.query_with_distance("cafe", 2).collect();

    println!("Query \"cafe\" at max_distance 2:");
    for candidate in &results {
        println!("  {}: distance {}", candidate.term, candidate.distance);
    }

    // Find "café" - should be distance 1 (substitute e→é)
    let cafe_result = results.iter().find(|c| c.term == "café");
    assert!(cafe_result.is_some());
    assert_eq!(cafe_result.unwrap().distance, 1);

    println!("\n✅ SUCCESS: Distances correctly computed for Unicode");
}

// ===== Various Distance Levels =====

#[test]
fn test_dynamic_dawg_char_various_distances() {
    println!("\n=== DynamicDawgChar: Various Distances ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["é", "ée", "éée"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"é\", \"ée\", \"éée\"]");
    println!("Each 'é' is 1 character (2 bytes in UTF-8)\n");

    // At distance 1: can insert 1 character → should find "é"
    let results_1: Vec<_> = transducer.query("", 1).collect();
    println!("Distance 1: {:?}", results_1);
    assert!(results_1.contains(&"é".to_string()));

    // At distance 2: can insert 2 characters → should find "é" and "ée"
    let results_2: Vec<_> = transducer.query("", 2).collect();
    println!("Distance 2: {:?}", results_2);
    assert!(results_2.contains(&"é".to_string()));
    assert!(results_2.contains(&"ée".to_string()));

    // At distance 3: should find all
    let results_3: Vec<_> = transducer.query("", 3).collect();
    println!("Distance 3: {:?}", results_3);
    assert!(results_3.contains(&"é".to_string()));
    assert!(results_3.contains(&"ée".to_string()));
    assert!(results_3.contains(&"éée".to_string()));

    println!("\n✅ SUCCESS: Character-level distances work correctly");
}

// ===== Dynamic Operations (Insert/Remove) =====

#[test]
fn test_dynamic_dawg_char_insert_unicode() {
    println!("\n=== DynamicDawgChar: Insert Unicode Terms ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
    assert_eq!(dict.term_count(), 0);

    // Insert various Unicode terms
    assert!(dict.insert("café"));
    assert!(dict.insert("中文"));
    assert!(dict.insert("🎉"));
    assert_eq!(dict.term_count(), 3);

    // Verify all inserted
    assert!(dict.contains("café"));
    assert!(dict.contains("中文"));
    assert!(dict.contains("🎉"));

    // Insert duplicate
    assert!(!dict.insert("café"));
    assert_eq!(dict.term_count(), 3);

    println!("✅ SUCCESS: Unicode insertions work correctly");
}

#[test]
fn test_dynamic_dawg_char_remove_unicode() {
    println!("\n=== DynamicDawgChar: Remove Unicode Terms ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["café", "中文", "🎉"]);
    assert_eq!(dict.term_count(), 3);

    // Remove Unicode terms
    assert!(dict.remove("café"));
    assert_eq!(dict.term_count(), 2);
    assert!(!dict.contains("café"));
    assert!(dict.contains("中文"));
    assert!(dict.contains("🎉"));

    // Remove non-existent
    assert!(!dict.remove("missing"));
    assert_eq!(dict.term_count(), 2);

    println!("✅ SUCCESS: Unicode removal works correctly");
}

#[test]
fn test_dynamic_dawg_char_dynamic_updates_with_fuzzy() {
    println!("\n=== DynamicDawgChar: Dynamic Updates with Fuzzy Queries ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
    dict.insert("café");
    dict.insert("naïve");

    let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

    // Initial query
    let results: Vec<_> = transducer.query("café", 0).collect();
    println!("Query \"café\" at distance 0: {:?}", results);
    assert!(results.contains(&"café".to_string()));

    // Add a new Unicode term dynamically
    dict.insert("新しい");

    // Query should now include the new term (distance 3 from empty is enough for 3-char term)
    let results: Vec<_> = transducer.query("", 3).collect();
    println!("Query \"\" at distance 3: {:?}", results);
    assert!(results.contains(&"新しい".to_string()));

    // Remove a term
    dict.remove("naïve");

    // Query should not find removed term
    let results: Vec<_> = transducer.query("naïve", 0).collect();
    println!("Query \"naïve\" (after removal): {:?}", results);
    assert!(!results.contains(&"naïve".to_string()));

    println!("\n✅ SUCCESS: Dynamic updates work with fuzzy queries");
}

// ===== Value Mapping =====

#[test]
fn test_dynamic_dawg_char_with_values() {
    println!("\n=== DynamicDawgChar: Value Mapping ===\n");

    let dict: DynamicDawgChar<u32> = DynamicDawgChar::new();
    dict.insert_with_value("café", 1);
    dict.insert_with_value("中文", 2);
    dict.insert_with_value("🎉", 3);

    println!("Dictionary with scope IDs:");
    println!("  \"café\" -> 1");
    println!("  \"中文\" -> 2");
    println!("  \"🎉\" -> 3\n");

    assert_eq!(dict.get_value("café"), Some(1));
    assert_eq!(dict.get_value("中文"), Some(2));
    assert_eq!(dict.get_value("🎉"), Some(3));
    assert_eq!(dict.get_value("missing"), None);

    println!("✅ SUCCESS: Value mapping works with Unicode");
}

#[test]
fn test_dynamic_dawg_char_value_updates() {
    println!("\n=== DynamicDawgChar: Value Updates ===\n");

    let dict: DynamicDawgChar<u32> = DynamicDawgChar::new();

    // Insert initial value
    assert!(dict.insert_with_value("café", 42));
    assert_eq!(dict.get_value("café"), Some(42));

    // Update value (insert returns false for existing key)
    assert!(!dict.insert_with_value("café", 99));
    assert_eq!(dict.get_value("café"), Some(99));

    println!("✅ SUCCESS: Value updates work with Unicode");
}

#[test]
#[cfg(feature = "pathmap-backend")]
fn test_dynamic_dawg_char_value_filtered_query() {
    use liblevenshtein::cache::multimap::FuzzyMultiMap;
    use std::collections::HashSet;

    println!("\n=== DynamicDawgChar: Value-Filtered Query ===\n");

    let dict: DynamicDawgChar<HashSet<u32>> = DynamicDawgChar::new();
    dict.insert_with_value("café", HashSet::from([1])); // scope 1
    dict.insert_with_value("cafe", HashSet::from([1])); // scope 1
    dict.insert_with_value("中文", HashSet::from([2])); // scope 2

    let fuzzy = FuzzyMultiMap::new(dict, Algorithm::Standard);

    println!("Dictionary with scopes:");
    println!("  Scope 1: \"café\", \"cafe\"");
    println!("  Scope 2: \"中文\"\n");

    // Query "cafe" with distance 1
    let result = fuzzy.query("cafe", 1).unwrap();
    println!("Query \"cafe\" at distance 1 (all scopes): {:?}", result);
    assert!(result.contains(&1)); // Both "café" and "cafe" are in scope 1

    println!("\n✅ SUCCESS: Value-filtered queries work with Unicode");
}

// ===== Edge Cases =====

#[test]
fn test_dynamic_dawg_char_empty_dictionary() {
    println!("\n=== DynamicDawgChar: Empty Dictionary ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::new();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let results: Vec<_> = transducer.query("café", 5).collect();
    println!("Query \"café\" on empty dictionary: {:?}", results);
    assert!(results.is_empty());

    println!("✅ SUCCESS: Empty dictionary handles queries correctly");
}

#[test]
fn test_dynamic_dawg_char_single_character_terms() {
    println!("\n=== DynamicDawgChar: Single Character Terms ===\n");

    let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["a", "é", "中", "🎉"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    println!("Dictionary: [\"a\", \"é\", \"中\", \"🎉\"]");

    // Empty query at distance 1 should find all (each requires 1 insertion)
    let results: Vec<_> = transducer.query("", 1).collect();
    println!("Empty query at distance 1: {:?}", results);

    assert_eq!(results.len(), 4);
    assert!(results.contains(&"a".to_string()));
    assert!(results.contains(&"é".to_string()));
    assert!(results.contains(&"中".to_string()));
    assert!(results.contains(&"🎉".to_string()));

    println!("\n✅ SUCCESS: Single character terms with various Unicode");
}

#[test]
fn test_dynamic_dawg_char_normalization_caveat() {
    println!("\n=== DynamicDawgChar: Unicode Normalization Caveat ===\n");

    // "é" can be represented as:
    // - NFC (composed): '\u{00E9}' - single code point
    // - NFD (decomposed): 'e' + '\u{0301}' (combining acute) - two code points

    let dict_nfc: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["café"]); // NFC form
    let transducer = Transducer::new(dict_nfc, Algorithm::Standard);

    let results: Vec<_> = transducer.query("café", 0).collect(); // exact match
    println!("Query \"café\" (NFC) for exact match: {:?}", results);
    assert!(results.contains(&"café".to_string()));

    println!("\n✅ SUCCESS: NFC Unicode handled correctly");
    println!("Note: NFD (decomposed) would be treated as separate characters");
}

// ===== Thread Safety =====

#[test]
fn test_dynamic_dawg_char_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    println!("\n=== DynamicDawgChar: Thread Safety ===\n");

    let dict: Arc<DynamicDawgChar<()>> =
        Arc::new(DynamicDawgChar::from_terms(vec!["café", "中文", "🎉"]));

    let dict1 = Arc::clone(&dict);
    let handle1 = thread::spawn(move || {
        dict1.insert("新しい");
        assert!(dict1.contains("新しい"));
    });

    let dict2 = Arc::clone(&dict);
    let handle2 = thread::spawn(move || {
        dict2.insert("日本語");
        assert!(dict2.contains("日本語"));
    });

    handle1.join().unwrap();
    handle2.join().unwrap();

    // Both terms should be present
    assert!(dict.contains("新しい"));
    assert!(dict.contains("日本語"));
    assert_eq!(dict.term_count(), 5);

    println!("✅ SUCCESS: Thread-safe insertions work correctly");
}

// ===== Comparison with Byte-level =====

#[test]
fn test_dynamic_dawg_char_vs_byte_level() {
    use libdictenstein::dynamic_dawg::DynamicDawg;

    println!("\n=== DynamicDawgChar: Comparison with Byte-level ===\n");

    // Byte-level (incorrect for Unicode)
    let dict_byte: DynamicDawg<()> = DynamicDawg::from_terms(vec!["¡"]);
    let trans_byte = Transducer::new(dict_byte, Algorithm::Standard);

    // Character-level (correct for Unicode)
    let dict_char: DynamicDawgChar<()> = DynamicDawgChar::from_terms(vec!["¡"]);
    let trans_char = Transducer::new(dict_char, Algorithm::Standard);

    println!("Query empty string \"\" with distance 1 for \"¡\":");

    // Byte-level incorrectly requires distance 2 (¡ is 2 bytes: 0xC2 0xA1)
    let results_byte: Vec<_> = trans_byte.query("", 1).collect();
    println!("  Byte-level results: {:?}", results_byte);
    assert!(
        results_byte.is_empty(),
        "Byte-level should NOT find \"¡\" at distance 1"
    );

    // Character-level correctly requires distance 1 (¡ is 1 character)
    let results_char: Vec<_> = trans_char.query("", 1).collect();
    println!("  Char-level results: {:?}", results_char);
    assert!(
        results_char.contains(&"¡".to_string()),
        "Char-level SHOULD find \"¡\" at distance 1"
    );

    println!("\n✅ SUCCESS: Char-level provides correct Unicode semantics");
}
