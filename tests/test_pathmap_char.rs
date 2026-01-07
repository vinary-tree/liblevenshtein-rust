#[cfg(feature = "pathmap-backend")]
mod pathmap_char_tests {
    use libdictenstein::pathmap_char::PathMapDictionaryChar;
    use libdictenstein::Dictionary;
    use liblevenshtein::prelude::*;

    // ===== Basic Dictionary Operations =====

    #[test]
    fn test_pathmap_char_empty_query_unicode() {
        println!("\n=== PathMapChar: Empty Query → Unicode Character ===\n");

        // This was the original problem: "" → "¡" should be distance 1, not 2
        let dict: PathMapDictionaryChar<()> = PathMapDictionaryChar::from_terms(vec!["¡"]);
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
    fn test_pathmap_char_exact_match() {
        println!("\n=== PathMapChar: Exact Match ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["café", "naïve", "résumé"]);
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
    fn test_pathmap_char_one_edit_distance() {
        println!("\n=== PathMapChar: One Edit Distance ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["café", "cafe"]);
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
    fn test_pathmap_char_emoji_distance() {
        println!("\n=== PathMapChar: Emoji Distances ===\n");

        let dict: PathMapDictionaryChar<()> = PathMapDictionaryChar::from_terms(vec!["🎉"]);
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
    fn test_pathmap_char_emoji_with_text() {
        println!("\n=== PathMapChar: Emoji with Text ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["hello🎉", "world🌍"]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        println!("Dictionary: [\"hello🎉\", \"world🌍\"]");

        // Query "hello" at distance 1 should find "hello🎉" (insert emoji at end)
        let results: Vec<_> = transducer.query("hello", 1).collect();
        println!("Query \"hello\" at distance 1: {:?}", results);
        assert!(results.contains(&"hello🎉".to_string()));

        println!("\n✅ SUCCESS: Emoji appending works as single character insertion");
    }

    #[test]
    fn test_pathmap_char_cjk_distance() {
        println!("\n=== PathMapChar: CJK Character Distances ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["中", "中文", "中文字"]);
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
    fn test_pathmap_char_mixed_unicode() {
        println!("\n=== PathMapChar: Mixed Unicode Characters ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["hello", "café", "中文", "🎉", "test123"]);

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
    fn test_pathmap_char_transposition_unicode() {
        println!("\n=== PathMapChar: Transposition with Unicode ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["café", "éfac"]);
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
    fn test_pathmap_char_query_with_distance() {
        println!("\n=== PathMapChar: Query with Distance (Unicode) ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["café", "naïve"]);
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
    fn test_pathmap_char_various_distances() {
        println!("\n=== PathMapChar: Various Distances ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["é", "ée", "éée"]);
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

    // ===== Dynamic Operations =====

    #[test]
    fn test_pathmap_char_insert_unicode() {
        println!("\n=== PathMapChar: Insert Unicode Terms ===\n");

        let dict: PathMapDictionaryChar<()> = PathMapDictionaryChar::new();
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
    fn test_pathmap_char_remove_unicode() {
        println!("\n=== PathMapChar: Remove Unicode Terms ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["café", "中文", "🎉"]);
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
    fn test_pathmap_char_clear() {
        println!("\n=== PathMapChar: Clear Dictionary ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["café", "中文", "🎉"]);
        assert_eq!(dict.term_count(), 3);

        dict.clear();
        assert_eq!(dict.term_count(), 0);
        assert!(!dict.contains("café"));
        assert!(!dict.contains("中文"));

        println!("✅ SUCCESS: Clear works correctly");
    }

    // ===== Value Mapping =====

    #[test]
    fn test_pathmap_char_with_values() {
        println!("\n=== PathMapChar: Value Mapping ===\n");

        let terms_with_values = vec![("café", 1u32), ("中文", 2u32), ("🎉", 3u32)];
        let dict: PathMapDictionaryChar<u32> =
            PathMapDictionaryChar::from_terms_with_values(terms_with_values);

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
    fn test_pathmap_char_value_filtered_query() {
        println!("\n=== PathMapChar: Value-Filtered Query ===\n");

        let terms_with_scopes = vec![
            ("café", 1u32), // scope 1
            ("cafe", 1u32), // scope 1
            ("中文", 2u32), // scope 2
            ("汉字", 2u32), // scope 2
        ];
        let dict: PathMapDictionaryChar<u32> =
            PathMapDictionaryChar::from_terms_with_values(terms_with_scopes);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        println!("Dictionary with scopes:");
        println!("  Scope 1: \"café\", \"cafe\"");
        println!("  Scope 2: \"中文\", \"汉字\"\n");

        // Query with scope filter for scope 1
        let results: Vec<_> = transducer
            .query_filtered("cafe", 1, |&scope| scope == 1)
            .collect();

        println!("Query \"cafe\" at distance 1, scope 1 only: {:?}", results);
        let result_terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();
        assert!(result_terms.contains(&"cafe"));
        assert!(result_terms.contains(&"café"));
        assert!(!result_terms.iter().any(|&s| s == "中文" || s == "汉字"));

        println!("\n✅ SUCCESS: Value-filtered queries work with Unicode");
    }

    #[test]
    fn test_pathmap_char_value_set_filtered_query() {
        use std::collections::HashSet;

        println!("\n=== PathMapChar: Value Set Filtered Query ===\n");

        let terms_with_scopes = vec![
            ("café", 1u32),
            ("naïve", 1u32),
            ("中文", 2u32),
            ("日本語", 3u32),
        ];
        let dict: PathMapDictionaryChar<u32> =
            PathMapDictionaryChar::from_terms_with_values(terms_with_scopes);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        println!("Dictionary with scopes:");
        println!("  Scope 1: \"café\", \"naïve\"");
        println!("  Scope 2: \"中文\"");
        println!("  Scope 3: \"日本語\"\n");

        // Query with scope set {1, 2}
        let scopes: HashSet<u32> = [1u32, 2u32].iter().copied().collect();
        let results: Vec<_> = transducer.query_by_value_set("cafe", 1, &scopes).collect();

        println!(
            "Query \"cafe\" at distance 1, scopes {{1, 2}}: {:?}",
            results
        );
        let result_terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();
        assert!(result_terms
            .iter()
            .any(|&s| s == "café" || s == "naïve" || s == "中文"));
        assert!(!result_terms.contains(&"日本語")); // scope 3 excluded

        println!("\n✅ SUCCESS: Value set filtering works with Unicode");
    }

    // ===== Edge Cases =====

    #[test]
    fn test_pathmap_char_empty_dictionary() {
        println!("\n=== PathMapChar: Empty Dictionary ===\n");

        let dict: PathMapDictionaryChar<()> = PathMapDictionaryChar::new();
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<_> = transducer.query("café", 5).collect();
        println!("Query \"café\" on empty dictionary: {:?}", results);
        assert!(results.is_empty());

        println!("✅ SUCCESS: Empty dictionary handles queries correctly");
    }

    #[test]
    fn test_pathmap_char_single_character_terms() {
        println!("\n=== PathMapChar: Single Character Terms ===\n");

        let dict: PathMapDictionaryChar<()> =
            PathMapDictionaryChar::from_terms(vec!["a", "é", "中", "🎉"]);
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
    fn test_pathmap_char_normalization_caveat() {
        println!("\n=== PathMapChar: Unicode Normalization Caveat ===\n");

        // "é" can be represented as:
        // - NFC (composed): '\u{00E9}' - single code point
        // - NFD (decomposed): 'e' + '\u{0301}' (combining acute) - two code points

        let dict_nfc: PathMapDictionaryChar<()> = PathMapDictionaryChar::from_terms(vec!["café"]); // NFC form
        let transducer = Transducer::new(dict_nfc, Algorithm::Standard);

        let results: Vec<_> = transducer.query("café", 0).collect(); // exact match
        println!("Query \"café\" (NFC) for exact match: {:?}", results);
        assert!(results.contains(&"café".to_string()));

        println!("\n✅ SUCCESS: NFC Unicode handled correctly");
        println!("Note: NFD (decomposed) would be treated as separate characters");
    }
}
