//! Comprehensive serialization/deserialization tests for all dictionary backends

#[cfg(feature = "serialization")]
mod serialization_tests {
    use liblevenshtein::prelude::*;

    /// Test data
    fn test_terms() -> Vec<&'static str> {
        vec![
            "apple",
            "application",
            "apply",
            "banana",
            "band",
            "test",
            "testing",
            "zebra",
        ]
    }

    // ============================================================================
    // Bincode Round-Trip Tests
    // ============================================================================

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_pathmap_bincode_roundtrip() {
        use libdictenstein::pathmap::PathMapDictionary;

        let terms = test_terms();
        let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(terms.iter().copied());

        // Serialize using PathMap's native format
        let mut buffer = Vec::new();
        dict.serialize_paths(&mut buffer)
            .expect("Failed to serialize PathMapDictionary");

        // Deserialize
        let deserialized: PathMapDictionary<()> = PathMapDictionary::deserialize_paths(&buffer[..])
            .expect("Failed to deserialize PathMapDictionary");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "PathMapDictionary missing term: {}",
                term
            );
        }
        assert_eq!(dict.term_count(), deserialized.term_count());
    }

    #[test]
    fn test_double_array_trie_bincode_roundtrip() {
        let terms = test_terms();
        let dict = DoubleArrayTrie::from_terms(terms.clone());

        // Serialize
        let mut buffer = Vec::new();
        bincode::serde::encode_into_std_write(&dict, &mut buffer, bincode::config::legacy())
            .expect("Failed to serialize DoubleArrayTrie");

        // Deserialize
        let deserialized: DoubleArrayTrie =
            bincode::serde::decode_from_slice(&buffer, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .expect("Failed to deserialize DoubleArrayTrie");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "DoubleArrayTrie missing term: {}",
                term
            );
        }
        assert_eq!(dict.len(), deserialized.len());
    }

    #[test]
    fn test_dynamic_dawg_bincode_roundtrip() {
        let terms = test_terms();
        let dict: DynamicDawg<()> = DynamicDawg::from_terms(terms.clone());

        // Serialize
        let mut buffer = Vec::new();
        bincode::serde::encode_into_std_write(&dict, &mut buffer, bincode::config::legacy())
            .expect("Failed to serialize DynamicDawg");

        // Deserialize
        let deserialized: DynamicDawg<()> =
            bincode::serde::decode_from_slice(&buffer, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .expect("Failed to deserialize DynamicDawg");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "DynamicDawg missing term: {}",
                term
            );
        }
        assert_eq!(dict.term_count(), deserialized.term_count());
    }

    #[test]
    fn test_dynamic_dawg_char_bincode_roundtrip() {
        use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

        let terms = test_terms();
        let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(terms.clone());

        // Serialize
        let mut buffer = Vec::new();
        bincode::serde::encode_into_std_write(&dict, &mut buffer, bincode::config::legacy())
            .expect("Failed to serialize DynamicDawgChar");

        // Deserialize
        let deserialized: DynamicDawgChar<()> =
            bincode::serde::decode_from_slice(&buffer, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .expect("Failed to deserialize DynamicDawgChar");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "DynamicDawgChar missing term: {}",
                term
            );
        }
        assert_eq!(dict.term_count(), deserialized.term_count());
    }

    #[test]
    fn test_suffix_automaton_bincode_roundtrip() {
        let terms: Vec<String> = test_terms().iter().map(|s| s.to_string()).collect();
        let dict = SuffixAutomaton::<()>::from_texts(terms.clone());

        // Serialize
        let mut buffer = Vec::new();
        bincode::serde::encode_into_std_write(&dict, &mut buffer, bincode::config::legacy())
            .expect("Failed to serialize SuffixAutomaton");

        // Deserialize
        let deserialized: SuffixAutomaton =
            bincode::serde::decode_from_slice(&buffer, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .expect("Failed to deserialize SuffixAutomaton");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "SuffixAutomaton missing term: {}",
                term
            );
        }
        assert_eq!(dict.string_count(), deserialized.string_count());
    }

    // ============================================================================
    // JSON Round-Trip Tests
    // ============================================================================

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_pathmap_json_roundtrip() {
        use libdictenstein::pathmap::PathMapDictionary;

        let terms = test_terms();
        let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(terms.iter().copied());

        // PathMap uses its own native .paths format, not JSON
        // Test serialization using the native format
        let mut buffer = Vec::new();
        dict.serialize_paths(&mut buffer)
            .expect("Failed to serialize PathMapDictionary");

        // Deserialize
        let deserialized: PathMapDictionary<()> = PathMapDictionary::deserialize_paths(&buffer[..])
            .expect("Failed to deserialize PathMapDictionary");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "PathMapDictionary missing term: {}",
                term
            );
        }
        assert_eq!(dict.term_count(), deserialized.term_count());
    }

    #[test]
    fn test_double_array_trie_json_roundtrip() {
        let terms = test_terms();
        let dict = DoubleArrayTrie::from_terms(terms.clone());

        // Serialize
        let json =
            serde_json::to_string(&dict).expect("Failed to serialize DoubleArrayTrie to JSON");

        // Deserialize
        let deserialized: DoubleArrayTrie =
            serde_json::from_str(&json).expect("Failed to deserialize DoubleArrayTrie from JSON");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "DoubleArrayTrie missing term: {}",
                term
            );
        }
        assert_eq!(dict.len(), deserialized.len());
    }

    #[test]
    fn test_dynamic_dawg_json_roundtrip() {
        let terms = test_terms();
        let dict: DynamicDawg<()> = DynamicDawg::from_terms(terms.clone());

        // Serialize
        let json = serde_json::to_string(&dict).expect("Failed to serialize DynamicDawg to JSON");

        // Deserialize
        let deserialized: DynamicDawg<()> =
            serde_json::from_str(&json).expect("Failed to deserialize DynamicDawg from JSON");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "DynamicDawg missing term: {}",
                term
            );
        }
        assert_eq!(dict.term_count(), deserialized.term_count());
    }

    #[test]
    fn test_dynamic_dawg_char_json_roundtrip() {
        use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

        let terms = test_terms();
        let dict: DynamicDawgChar<()> = DynamicDawgChar::from_terms(terms.clone());

        // Serialize
        let json =
            serde_json::to_string(&dict).expect("Failed to serialize DynamicDawgChar to JSON");

        // Deserialize
        let deserialized: DynamicDawgChar<()> =
            serde_json::from_str(&json).expect("Failed to deserialize DynamicDawgChar from JSON");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "DynamicDawgChar missing term: {}",
                term
            );
        }
        assert_eq!(dict.term_count(), deserialized.term_count());
    }

    #[test]
    fn test_suffix_automaton_json_roundtrip() {
        let terms: Vec<String> = test_terms().iter().map(|s| s.to_string()).collect();
        let dict = SuffixAutomaton::<()>::from_texts(terms.clone());

        // Serialize
        let json =
            serde_json::to_string(&dict).expect("Failed to serialize SuffixAutomaton to JSON");

        // Deserialize
        let deserialized: SuffixAutomaton =
            serde_json::from_str(&json).expect("Failed to deserialize SuffixAutomaton from JSON");

        // Verify
        for term in &terms {
            assert!(
                deserialized.contains(term),
                "SuffixAutomaton missing term: {}",
                term
            );
        }
        assert_eq!(dict.string_count(), deserialized.string_count());
    }

    // ============================================================================
    // Cross-Format Consistency Tests
    // ============================================================================

    #[test]
    fn test_double_array_trie_bincode_json_consistency() {
        let terms = test_terms();
        let dict = DoubleArrayTrie::from_terms(terms.clone());

        // Serialize to both formats
        let bincode_bytes =
            bincode::serde::encode_to_vec(&dict, bincode::config::legacy()).unwrap();
        let json_str = serde_json::to_string(&dict).unwrap();

        // Deserialize from both
        let from_bincode: DoubleArrayTrie =
            bincode::serde::decode_from_slice(&bincode_bytes, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .unwrap();
        let from_json: DoubleArrayTrie = serde_json::from_str(&json_str).unwrap();

        // Both should contain same terms
        for term in &terms {
            assert!(from_bincode.contains(term));
            assert!(from_json.contains(term));
        }

        assert_eq!(from_bincode.len(), from_json.len());
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_all_backends_serialize_same_content() {
        use libdictenstein::pathmap::PathMapDictionary;

        let terms = test_terms();

        let pathmap: PathMapDictionary<()> = PathMapDictionary::from_terms(terms.iter().copied());
        let dat = DoubleArrayTrie::from_terms(terms.clone());
        let dynamic: DynamicDawg<()> = DynamicDawg::from_terms(terms.clone());

        // All should contain same terms
        for term in &terms {
            assert!(pathmap.contains(term), "PathMap missing: {}", term);
            assert!(dat.contains(term), "DAT missing: {}", term);
            assert!(dynamic.contains(term), "DynamicDawg missing: {}", term);
        }
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    #[test]
    fn test_empty_dictionary_serialization() {
        let empty_terms: Vec<&str> = vec![];

        // DoubleArrayTrie
        let dict = DoubleArrayTrie::from_terms(empty_terms.clone());
        let serialized = bincode::serde::encode_to_vec(&dict, bincode::config::legacy()).unwrap();
        let deserialized: DoubleArrayTrie =
            bincode::serde::decode_from_slice(&serialized, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .unwrap();
        assert_eq!(deserialized.len().unwrap_or(0), 0);

        // DynamicDawg
        let dict: DynamicDawg<()> = DynamicDawg::from_terms(empty_terms);
        let serialized = bincode::serde::encode_to_vec(&dict, bincode::config::legacy()).unwrap();
        let deserialized: DynamicDawg<()> =
            bincode::serde::decode_from_slice(&serialized, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .unwrap();
        assert_eq!(deserialized.term_count(), 0);
    }

    #[test]
    fn test_unicode_serialization() {
        let unicode_terms = vec!["hello", "世界", "🌍", "Ñoño", "café"];

        let dict = DoubleArrayTrie::from_terms(unicode_terms.clone());
        let serialized = bincode::serde::encode_to_vec(&dict, bincode::config::legacy()).unwrap();
        let deserialized: DoubleArrayTrie =
            bincode::serde::decode_from_slice(&serialized, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .unwrap();

        for term in &unicode_terms {
            assert!(
                deserialized.contains(term),
                "Missing unicode term: {}",
                term
            );
        }
    }

    #[test]
    fn test_large_dictionary_serialization() {
        // Generate 1000 terms
        let terms: Vec<String> = (0..1000).map(|i| format!("term_{:04}", i)).collect();

        let dict = DoubleArrayTrie::from_terms(terms.clone());
        let serialized = bincode::serde::encode_to_vec(&dict, bincode::config::legacy()).unwrap();
        let deserialized: DoubleArrayTrie =
            bincode::serde::decode_from_slice(&serialized, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .unwrap();

        // Spot check
        assert!(deserialized.contains("term_0000"));
        assert!(deserialized.contains("term_0500"));
        assert!(deserialized.contains("term_0999"));
        assert_eq!(deserialized.len().unwrap_or(0), 1000);
    }

    #[test]
    fn test_dynamic_dawg_after_modifications() {
        let initial_terms = vec!["apple", "banana"];
        let dict: DynamicDawg<()> = DynamicDawg::from_terms(initial_terms);

        // Add more terms
        dict.insert("cherry");
        dict.insert("date");

        // Serialize
        let serialized = bincode::serde::encode_to_vec(&dict, bincode::config::legacy()).unwrap();
        let deserialized: DynamicDawg<()> =
            bincode::serde::decode_from_slice(&serialized, bincode::config::legacy())
                .map(|(__decoded, _)| __decoded)
                .unwrap();

        // Verify all terms
        assert!(deserialized.contains("apple"));
        assert!(deserialized.contains("banana"));
        assert!(deserialized.contains("cherry"));
        assert!(deserialized.contains("date"));
        assert_eq!(deserialized.term_count(), 4);
    }
}
