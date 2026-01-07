//! Integration tests for cache eviction wrapper composition.
//!
//! These tests verify that multiple eviction wrappers can be composed
//! and work together correctly.

#[cfg(feature = "pathmap-backend")]
mod wrapper_composition {
    use liblevenshtein::cache::eviction::{Age, CostAware, Lfu, Lru, MemoryPressure, Noop, Ttl};
    use libdictenstein::MappedDictionary;
    use liblevenshtein::prelude::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_lru_ttl_composition() {
        // TTL filters expired entries, LRU tracks recency of remaining entries
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let ttl = Ttl::new(dict, Duration::from_secs(1));
        let lru = Lru::new(ttl);

        // Access all entries
        assert_eq!(lru.get_value("foo"), Some(42));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("bar"), Some(99));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("baz"), Some(123));

        // foo should be least recently used
        let lru_term = lru.find_lru(&["foo", "bar", "baz"]);
        assert_eq!(lru_term, Some("foo".to_string()));

        // After TTL expires, entries should return None
        thread::sleep(Duration::from_secs(2));
        assert_eq!(lru.get_value("foo"), None);
        assert_eq!(lru.get_value("bar"), None);
        assert_eq!(lru.get_value("baz"), None);
    }

    #[test]
    fn test_lfu_age_composition() {
        // Age tracks insertion time, LFU tracks access frequency
        let dict =
            PathMapDictionary::from_terms_with_values([("first", 1), ("second", 2), ("third", 3)]);

        let age = Age::new(dict);
        let lfu = Lfu::new(age);

        // Access with different frequencies
        assert_eq!(lfu.get_value("first"), Some(1));

        thread::sleep(Duration::from_millis(10));
        assert_eq!(lfu.get_value("second"), Some(2));
        assert_eq!(lfu.get_value("second"), Some(2));

        thread::sleep(Duration::from_millis(10));
        assert_eq!(lfu.get_value("third"), Some(3));
        assert_eq!(lfu.get_value("third"), Some(3));
        assert_eq!(lfu.get_value("third"), Some(3));

        // first is least frequently used (1 access)
        let lfu_term = lfu.find_lfu(&["first", "second", "third"]);
        assert_eq!(lfu_term, Some("first".to_string()));

        // first is also oldest
        let oldest = lfu.inner().find_oldest(&["first", "second", "third"]);
        assert_eq!(oldest, Some("first".to_string()));
    }

    #[test]
    fn test_memory_pressure_cost_aware_composition() {
        // CostAware balances age/size/hits, MemoryPressure tracks size/hit_rate
        let dict = PathMapDictionary::from_terms_with_values([
            ("large", vec![1, 2, 3, 4, 5]),
            ("medium", vec![1, 2, 3]),
            ("small", vec![1]),
        ]);

        let memory = MemoryPressure::new(dict);
        let cost = CostAware::new(memory);

        // Access all entries
        cost.get_value("large");
        cost.get_value("medium");
        cost.get_value("small");

        thread::sleep(Duration::from_millis(20));

        // All should have cost scores
        assert!(cost.cost_score("large").is_some());
        assert!(cost.cost_score("medium").is_some());
        assert!(cost.cost_score("small").is_some());

        // Inner MemoryPressure should also have scores
        assert!(cost.inner().memory_pressure_score("large").is_some());
        assert!(cost.inner().memory_pressure_score("medium").is_some());
        assert!(cost.inner().memory_pressure_score("small").is_some());
    }

    #[test]
    fn test_triple_wrapper_composition() {
        // Stack three wrappers: Lru -> Ttl -> Age
        let dict =
            PathMapDictionary::from_terms_with_values([("alpha", 1), ("beta", 2), ("gamma", 3)]);

        let age = Age::new(dict);
        let ttl = Ttl::new(age, Duration::from_secs(2));
        let lru = Lru::new(ttl);

        // Access all entries
        assert_eq!(lru.get_value("alpha"), Some(1));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("beta"), Some(2));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("gamma"), Some(3));

        // LRU tracking works
        let lru_term = lru.find_lru(&["alpha", "beta", "gamma"]);
        assert_eq!(lru_term, Some("alpha".to_string()));

        // Age tracking (through ttl.inner()) works
        let oldest = lru.inner().inner().find_oldest(&["alpha", "beta", "gamma"]);
        assert_eq!(oldest, Some("alpha".to_string()));

        // After TTL expires, all return None
        thread::sleep(Duration::from_secs(3));
        assert_eq!(lru.get_value("alpha"), None);
    }

    #[test]
    fn test_noop_passthrough() {
        // Noop should not affect behavior
        let dict = PathMapDictionary::from_terms_with_values([("test", 42)]);

        let noop = Noop::new(dict);
        let lru = Lru::new(noop);

        assert_eq!(lru.get_value("test"), Some(42));

        // Can unwrap through layers
        let unwrapped = lru.into_inner().into_inner();
        assert_eq!(unwrapped.get_value("test"), Some(42));
    }

    #[test]
    fn test_into_inner_unwrapping() {
        // Test unwrapping composed wrappers
        let dict = PathMapDictionary::from_terms_with_values([("data", 100)]);

        let lfu = Lfu::new(dict);
        let lru = Lru::new(lfu);

        // Access through composed wrapper
        assert_eq!(lru.get_value("data"), Some(100));

        // Unwrap one layer
        let lfu_wrapper = lru.into_inner();
        assert_eq!(lfu_wrapper.get_value("data"), Some(100));

        // Unwrap to base dictionary
        let base_dict = lfu_wrapper.into_inner();
        assert_eq!(base_dict.get_value("data"), Some(100));
    }

    #[test]
    fn test_metadata_independence() {
        // Each wrapper maintains independent metadata
        let dict = PathMapDictionary::from_terms_with_values([("item", 50)]);

        let lfu = Lfu::new(dict);
        let lru = Lru::new(lfu);

        // First access
        assert_eq!(lru.get_value("item"), Some(50));

        // LRU metadata exists
        assert!(lru.recency("item").is_some());

        // LFU metadata exists (through inner)
        assert!(lru.inner().access_count("item").is_some());
        assert_eq!(lru.inner().access_count("item"), Some(1));

        // Second access
        assert_eq!(lru.get_value("item"), Some(50));

        // LFU count incremented
        assert_eq!(lru.inner().access_count("item"), Some(2));

        // LRU recency updated (time increased since last access)
        assert!(lru.recency("item").is_some());
    }

    #[test]
    fn test_eviction_coordination() {
        // Test that eviction works correctly with composed wrappers
        let dict = PathMapDictionary::from_terms_with_values([("keep", 1), ("evict", 2)]);

        let lru = Lru::new(dict);

        // Access both
        assert_eq!(lru.get_value("keep"), Some(1));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("evict"), Some(2));
        thread::sleep(Duration::from_millis(10));

        // Access "keep" again to make it most recently used
        assert_eq!(lru.get_value("keep"), Some(1));

        // "evict" should be LRU
        let lru_term = lru.find_lru(&["keep", "evict"]);
        assert_eq!(lru_term, Some("evict".to_string()));

        // Evict LRU
        let evicted = lru.evict_lru(&["keep", "evict"]);
        assert_eq!(evicted, Some("evict".to_string()));

        // Evicted entry has no metadata
        assert_eq!(lru.recency("evict"), None);

        // But data still in dictionary (wrapper only tracks metadata)
        assert_eq!(lru.get_value("evict"), Some(2));
    }

    #[test]
    fn test_clear_metadata_composition() {
        // Test clearing metadata in composed wrappers
        let dict = PathMapDictionary::from_terms_with_values([("data", 42)]);

        let lfu = Lfu::new(dict);
        let lru = Lru::new(lfu);

        // Create metadata
        assert_eq!(lru.get_value("data"), Some(42));
        assert!(lru.recency("data").is_some());
        assert!(lru.inner().access_count("data").is_some());

        // Clear outer metadata
        lru.clear_metadata();
        assert_eq!(lru.recency("data"), None);

        // Inner metadata still exists
        assert!(lru.inner().access_count("data").is_some());

        // Clear inner metadata
        lru.inner().clear_metadata();
        assert_eq!(lru.inner().access_count("data"), None);
    }
}
