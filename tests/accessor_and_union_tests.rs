//! Integration tests for dictionary accessor methods and union operations.

#[cfg(feature = "pathmap-backend")]
mod pathmap_tests {
    use libdictenstein::pathmap::char::PathMapDictionaryChar;
    use libdictenstein::pathmap::PathMapDictionary;
    use libdictenstein::{Dictionary, MutableMappedDictionary};
    use liblevenshtein::contextual::{ContextId, DynamicContextualCompletionEngine};
    use liblevenshtein::transducer::{Algorithm, Transducer};

    #[test]
    fn test_transducer_into_inner() {
        let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
            ("hello", 1),
            ("world", 2),
            ("test", 3),
        ]);

        let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

        // Extract the dictionary
        let extracted_dict = transducer.into_inner();

        // Verify the extracted dictionary has the same content
        assert!(extracted_dict.contains("hello"));
        assert!(extracted_dict.contains("world"));
        assert!(extracted_dict.contains("test"));
        assert_eq!(extracted_dict.len().unwrap(), 3);
    }

    #[test]
    fn test_transducer_into_dictionary() {
        let dict: PathMapDictionary<u32> =
            PathMapDictionary::from_terms_with_values(vec![("foo", 10), ("bar", 20)]);

        let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

        // Extract using into_dictionary() alias
        let extracted_dict = transducer.into_dictionary();

        assert!(extracted_dict.contains("foo"));
        assert!(extracted_dict.contains("bar"));
        assert_eq!(extracted_dict.len().unwrap(), 2);
    }

    #[test]
    fn test_engine_transducer_accessor() {
        let dict: PathMapDictionary<Vec<ContextId>> = PathMapDictionary::new();
        dict.insert_with_value("hello", vec![1, 2, 3]);
        dict.insert_with_value("world", vec![4, 5, 6]);

        let engine = DynamicContextualCompletionEngine::with_dictionary(dict, Algorithm::Standard);

        // Access the dictionary through the transducer
        let len = engine.with_transducer(|transducer| {
            let dictionary = transducer.dictionary();
            assert!(dictionary.contains("hello"));
            assert!(dictionary.contains("world"));
            dictionary.len().unwrap()
        });
        assert_eq!(len, 2);
    }

    #[test]
    fn test_pathmap_union_with_no_conflicts() {
        let dict1: PathMapDictionary<u32> = PathMapDictionary::new();
        dict1.insert_with_value("apple", 1);
        dict1.insert_with_value("banana", 2);

        let dict2: PathMapDictionary<u32> = PathMapDictionary::new();
        dict2.insert_with_value("cherry", 3);
        dict2.insert_with_value("date", 4);

        // Union with no conflicts (merge function won't be called)
        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 2); // 2 entries from dict2
        assert_eq!(dict1.len().unwrap(), 4); // 2 original + 2 from union

        assert_eq!(dict1.get_value("apple"), Some(1));
        assert_eq!(dict1.get_value("banana"), Some(2));
        assert_eq!(dict1.get_value("cherry"), Some(3));
        assert_eq!(dict1.get_value("date"), Some(4));
    }

    #[test]
    fn test_pathmap_union_with_conflicts_merge() {
        let dict1: PathMapDictionary<u32> = PathMapDictionary::new();
        dict1.insert_with_value("apple", 1);
        dict1.insert_with_value("banana", 2);

        let dict2: PathMapDictionary<u32> = PathMapDictionary::new();
        dict2.insert_with_value("apple", 10); // Conflict!
        dict2.insert_with_value("cherry", 3);

        // Union with merge function (sum the values)
        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 2); // 2 entries from dict2
        assert_eq!(dict1.len().unwrap(), 3); // 2 original + 1 new (apple already existed)

        assert_eq!(dict1.get_value("apple"), Some(11)); // 1 + 10 = 11
        assert_eq!(dict1.get_value("banana"), Some(2));
        assert_eq!(dict1.get_value("cherry"), Some(3));
    }

    #[test]
    fn test_pathmap_union_replace() {
        let dict1: PathMapDictionary<u32> = PathMapDictionary::new();
        dict1.insert_with_value("apple", 1);
        dict1.insert_with_value("banana", 2);

        let dict2: PathMapDictionary<u32> = PathMapDictionary::new();
        dict2.insert_with_value("apple", 100); // Will replace
        dict2.insert_with_value("cherry", 3);

        // Use union_replace (last writer wins)
        let processed = dict1.union_replace(&dict2);

        assert_eq!(processed, 2);
        assert_eq!(dict1.len().unwrap(), 3);

        assert_eq!(dict1.get_value("apple"), Some(100)); // Replaced!
        assert_eq!(dict1.get_value("banana"), Some(2));
        assert_eq!(dict1.get_value("cherry"), Some(3));
    }

    #[test]
    fn test_pathmap_union_with_vec_values() {
        let dict1: PathMapDictionary<Vec<u32>> = PathMapDictionary::new();
        dict1.insert_with_value("apple", vec![1, 2]);
        dict1.insert_with_value("banana", vec![3]);

        let dict2: PathMapDictionary<Vec<u32>> = PathMapDictionary::new();
        dict2.insert_with_value("apple", vec![4, 5]); // Conflict with vector value
        dict2.insert_with_value("cherry", vec![6, 7]);

        // Merge by concatenating and deduplicating vectors
        let processed = dict1.union_with(&dict2, |left, right| {
            let mut merged = left.clone();
            merged.extend(right.clone());
            merged.sort();
            merged.dedup();
            merged
        });

        assert_eq!(processed, 2);
        assert_eq!(dict1.len().unwrap(), 3);

        assert_eq!(dict1.get_value("apple"), Some(vec![1, 2, 4, 5]));
        assert_eq!(dict1.get_value("banana"), Some(vec![3]));
        assert_eq!(dict1.get_value("cherry"), Some(vec![6, 7]));
    }

    #[test]
    fn test_pathmap_char_union_with_conflicts() {
        let dict1: PathMapDictionaryChar<u32> = PathMapDictionaryChar::new();
        dict1.insert_with_value("hello", 1);
        dict1.insert_with_value("world", 2);

        let dict2: PathMapDictionaryChar<u32> = PathMapDictionaryChar::new();
        dict2.insert_with_value("hello", 10); // Conflict
        dict2.insert_with_value("test", 3);

        // Union with custom merge (take maximum)
        let processed = dict1.union_with(&dict2, |left, right| (*left).max(*right));

        assert_eq!(processed, 2);
        assert_eq!(dict1.len().unwrap(), 3);

        assert_eq!(dict1.get_value("hello"), Some(10)); // max(1, 10)
        assert_eq!(dict1.get_value("world"), Some(2));
        assert_eq!(dict1.get_value("test"), Some(3));
    }

    #[test]
    fn test_empty_union() {
        let dict1: PathMapDictionary<u32> = PathMapDictionary::new();
        dict1.insert_with_value("apple", 1);

        let dict2: PathMapDictionary<u32> = PathMapDictionary::new();
        // dict2 is empty

        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 0); // Nothing to process
        assert_eq!(dict1.len().unwrap(), 1); // Unchanged
        assert_eq!(dict1.get_value("apple"), Some(1));
    }

    #[test]
    fn test_union_into_empty() {
        let dict1: PathMapDictionary<u32> = PathMapDictionary::new();
        // dict1 is empty

        let dict2: PathMapDictionary<u32> = PathMapDictionary::new();
        dict2.insert_with_value("apple", 1);
        dict2.insert_with_value("banana", 2);

        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 2);
        assert_eq!(dict1.len().unwrap(), 2);
        assert_eq!(dict1.get_value("apple"), Some(1));
        assert_eq!(dict1.get_value("banana"), Some(2));
    }

    #[test]
    fn test_serialization_after_extraction() {
        let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(vec!["hello", "world"]);

        let transducer = Transducer::new(dict.clone(), Algorithm::Standard);

        // Extract the dictionary
        let extracted_dict = transducer.into_inner();

        // Verify we can still access and use the dictionary
        assert!(extracted_dict.contains("hello"));
        assert!(extracted_dict.contains("world"));

        let mut serialized = Vec::new();
        extracted_dict
            .serialize_paths(&mut serialized)
            .expect("failed to serialize extracted PathMap dictionary");
        assert!(
            !serialized.is_empty(),
            "serialized PathMap payload should not be empty"
        );

        let deserialized: PathMapDictionary<()> =
            PathMapDictionary::deserialize_paths(&serialized[..])
                .expect("failed to deserialize extracted PathMap dictionary");
        assert!(deserialized.contains("hello"));
        assert!(deserialized.contains("world"));
        assert_eq!(deserialized.term_count(), 2);

        let new_transducer = Transducer::new(deserialized, Algorithm::Standard);
        let results: Vec<_> = new_transducer.query("helo", 1).collect();
        assert!(results.contains(&"hello".to_string()));
    }
}

#[cfg(test)]
mod dynamic_dawg_tests {
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use libdictenstein::{Dictionary, MutableMappedDictionary};

    #[test]
    fn test_dynamic_dawg_union_no_conflicts() {
        let dict1: DynamicDawg<u32> = DynamicDawg::new();
        dict1.insert_with_value("apple", 1);
        dict1.insert_with_value("banana", 2);

        let dict2: DynamicDawg<u32> = DynamicDawg::new();
        dict2.insert_with_value("cherry", 3);
        dict2.insert_with_value("date", 4);

        // Union with no conflicts
        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 2); // 2 entries from dict2
        assert_eq!(dict1.len().unwrap(), 4); // 2 original + 2 from union

        assert_eq!(dict1.get_value("apple"), Some(1));
        assert_eq!(dict1.get_value("banana"), Some(2));
        assert_eq!(dict1.get_value("cherry"), Some(3));
        assert_eq!(dict1.get_value("date"), Some(4));
    }

    #[test]
    fn test_dynamic_dawg_union_with_conflicts_merge() {
        let dict1: DynamicDawg<u32> = DynamicDawg::new();
        dict1.insert_with_value("apple", 1);
        dict1.insert_with_value("banana", 2);

        let dict2: DynamicDawg<u32> = DynamicDawg::new();
        dict2.insert_with_value("apple", 10); // Conflict!
        dict2.insert_with_value("cherry", 3);

        // Union with merge function (sum the values)
        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 2); // 2 entries from dict2
        assert_eq!(dict1.len().unwrap(), 3); // 2 original + 1 new (apple merged)

        assert_eq!(dict1.get_value("apple"), Some(11)); // 1 + 10 = 11
        assert_eq!(dict1.get_value("banana"), Some(2));
        assert_eq!(dict1.get_value("cherry"), Some(3));
    }

    #[test]
    fn test_dynamic_dawg_union_replace() {
        let dict1: DynamicDawg<u32> = DynamicDawg::new();
        dict1.insert_with_value("apple", 1);
        dict1.insert_with_value("banana", 2);

        let dict2: DynamicDawg<u32> = DynamicDawg::new();
        dict2.insert_with_value("apple", 100); // Will replace
        dict2.insert_with_value("cherry", 3);

        // Use union_replace (last writer wins)
        let processed = dict1.union_replace(&dict2);

        assert_eq!(processed, 2);
        assert_eq!(dict1.len().unwrap(), 3);

        assert_eq!(dict1.get_value("apple"), Some(100)); // Replaced!
        assert_eq!(dict1.get_value("banana"), Some(2));
        assert_eq!(dict1.get_value("cherry"), Some(3));
    }

    #[test]
    fn test_dynamic_dawg_union_with_vec_values() {
        let dict1: DynamicDawg<Vec<u32>> = DynamicDawg::new();
        dict1.insert_with_value("apple", vec![1, 2]);
        dict1.insert_with_value("banana", vec![3]);

        let dict2: DynamicDawg<Vec<u32>> = DynamicDawg::new();
        dict2.insert_with_value("apple", vec![4, 5]); // Conflict with vector value
        dict2.insert_with_value("cherry", vec![6, 7]);

        // Merge by concatenating and deduplicating vectors
        let processed = dict1.union_with(&dict2, |left, right| {
            let mut merged = left.clone();
            merged.extend(right.clone());
            merged.sort();
            merged.dedup();
            merged
        });

        assert_eq!(processed, 2);
        assert_eq!(dict1.len().unwrap(), 3);

        assert_eq!(dict1.get_value("apple"), Some(vec![1, 2, 4, 5]));
        assert_eq!(dict1.get_value("banana"), Some(vec![3]));
        assert_eq!(dict1.get_value("cherry"), Some(vec![6, 7]));
    }

    #[test]
    fn test_dynamic_dawg_char_union_with_conflicts() {
        let dict1: DynamicDawgChar<u32> = DynamicDawgChar::new();
        dict1.insert_with_value("hello", 1);
        dict1.insert_with_value("world", 2);

        let dict2: DynamicDawgChar<u32> = DynamicDawgChar::new();
        dict2.insert_with_value("hello", 10); // Conflict
        dict2.insert_with_value("test", 3);

        // Union with custom merge (take maximum)
        let processed = dict1.union_with(&dict2, |left, right| (*left).max(*right));

        assert_eq!(processed, 2);
        assert_eq!(dict1.len().unwrap(), 3);

        assert_eq!(dict1.get_value("hello"), Some(10)); // max(1, 10)
        assert_eq!(dict1.get_value("world"), Some(2));
        assert_eq!(dict1.get_value("test"), Some(3));
    }

    #[test]
    fn test_dynamic_dawg_empty_union() {
        let dict1: DynamicDawg<u32> = DynamicDawg::new();
        dict1.insert_with_value("apple", 1);

        let dict2: DynamicDawg<u32> = DynamicDawg::new();
        // dict2 is empty

        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 0); // Nothing to process
        assert_eq!(dict1.len().unwrap(), 1); // Unchanged
        assert_eq!(dict1.get_value("apple"), Some(1));
    }

    #[test]
    fn test_dynamic_dawg_union_into_empty() {
        let dict1: DynamicDawg<u32> = DynamicDawg::new();
        // dict1 is empty

        let dict2: DynamicDawg<u32> = DynamicDawg::new();
        dict2.insert_with_value("apple", 1);
        dict2.insert_with_value("banana", 2);

        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 2);
        assert_eq!(dict1.len().unwrap(), 2);
        assert_eq!(dict1.get_value("apple"), Some(1));
        assert_eq!(dict1.get_value("banana"), Some(2));
    }

    #[test]
    fn test_dynamic_dawg_union_with_shared_prefixes() {
        // Test that union works correctly with terms that share prefixes
        let dict1: DynamicDawg<u32> = DynamicDawg::new();
        dict1.insert_with_value("test", 1);
        dict1.insert_with_value("testing", 2);
        dict1.insert_with_value("tester", 3);

        let dict2: DynamicDawg<u32> = DynamicDawg::new();
        dict2.insert_with_value("test", 10); // Conflict
        dict2.insert_with_value("tested", 4); // New term with shared prefix
        dict2.insert_with_value("testimony", 5); // Another shared prefix

        let processed = dict1.union_with(&dict2, |left, right| *left + *right);

        assert_eq!(processed, 3);
        assert_eq!(dict1.len().unwrap(), 5);

        assert_eq!(dict1.get_value("test"), Some(11)); // 1 + 10
        assert_eq!(dict1.get_value("testing"), Some(2));
        assert_eq!(dict1.get_value("tester"), Some(3));
        assert_eq!(dict1.get_value("tested"), Some(4));
        assert_eq!(dict1.get_value("testimony"), Some(5));
    }
}
