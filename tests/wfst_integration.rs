//! Integration tests for WFST functionality.
//!
//! These tests verify that liblevenshtein integrates correctly with
//! lling-llang's WFST infrastructure.

#![cfg(feature = "wfst")]

use libdictenstein::dynamic_dawg_char::DynamicDawgChar;
use liblevenshtein::wfst::{
    DictionaryBackend, LevenshteinStateSource, LevenshteinWfst, Semiring, StateSource,
    TropicalWeight, Wfst,
};
use lling_llang::backend::LatticeBackend;
use lling_llang::wfst::{LazyWfst, LazyWfstWrapper};

#[test]
fn test_levenshtein_wfst_basic() {
    // Create a simple dictionary
    let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world", "hell"]);

    // Create WFST for query "helo" with max distance 2
    let wfst = LevenshteinWfst::new(&dict, "helo", 2);

    // Verify basic properties
    assert_eq!(wfst.max_distance(), 2);
    assert_eq!(wfst.query(), "helo");
    assert!(!wfst.is_empty());
}

#[test]
fn test_levenshtein_wfst_start_state() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
    let wfst = LevenshteinWfst::new(&dict, "tset", 2);

    // Start state should exist
    let start = wfst.start();
    assert!(wfst.is_valid_state(start));
}

#[test]
fn test_levenshtein_wfst_lazy_expansion() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help"]);
    let mut wfst = LevenshteinWfst::new(&dict, "helo", 2);

    // Initially nothing expanded
    let start = wfst.start();
    assert!(!wfst.is_expanded(start));
    assert_eq!(wfst.computed_states(), 0);

    // Expand start state
    wfst.expand(start);
    assert!(wfst.is_expanded(start));
    assert!(wfst.computed_states() >= 1);
}

#[test]
fn test_levenshtein_wfst_clone() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
    let wfst = LevenshteinWfst::new(&dict, "tset", 1);
    let cloned = wfst.clone();

    assert_eq!(wfst.start(), cloned.start());
    assert_eq!(wfst.max_distance(), cloned.max_distance());
}

#[test]
fn test_levenshtein_state_source_basic() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
    let source = LevenshteinStateSource::new(&dict, "helo", 2);

    // Start state should be 0
    let start = source.start();
    assert_eq!(start, 0);

    // Should have a state hint
    assert!(source.num_states_hint().is_some());
}

#[test]
fn test_levenshtein_state_source_compute_state() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help"]);
    let source = LevenshteinStateSource::new(&dict, "helo", 2);

    // Compute start state
    let state = source.compute_state(source.start());
    assert!(state.is_computed());
}

#[test]
fn test_levenshtein_state_source_with_lazy_wrapper() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help"]);
    let source = LevenshteinStateSource::new(&dict, "helo", 2);

    // Wrap in LazyWfstWrapper
    let mut lazy_wfst = LazyWfstWrapper::new(source);

    // Initially no states computed
    assert_eq!(lazy_wfst.computed_states(), 0);

    // Access start state
    let start = lazy_wfst.start();
    let _transitions = lazy_wfst.transitions_lazy(start);

    // Now should have computed at least 1 state
    assert!(lazy_wfst.computed_states() >= 1);
}

#[test]
fn test_dictionary_backend_basic() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
    let mut backend = DictionaryBackend::new(dict);

    // Initially empty vocabulary
    assert_eq!(backend.vocab_size(), 0);

    // Intern some words
    let id1 = backend.intern("hello");
    let id2 = backend.intern("world");

    assert_eq!(backend.vocab_size(), 2);
    assert_ne!(id1, id2);

    // Same word should return same ID
    let id3 = backend.intern("hello");
    assert_eq!(id1, id3);
}

#[test]
fn test_dictionary_backend_lookup() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
    let mut backend = DictionaryBackend::new(dict);

    let id = backend.intern("test");
    assert_eq!(backend.lookup(id), Some("test"));
    assert_eq!(backend.lookup(999), None);
}

#[test]
fn test_dictionary_backend_contains() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
    let backend = DictionaryBackend::new(dict);

    // Contains checks the underlying dictionary
    assert!(backend.contains("hello"));
    assert!(backend.contains("world"));
    assert!(!backend.contains("missing"));
}

#[test]
fn test_dictionary_backend_iter() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["a", "b", "c"]);
    let mut backend = DictionaryBackend::new(dict);

    backend.intern("a");
    backend.intern("b");
    backend.intern("c");

    let entries: Vec<_> = backend.iter().collect();
    assert_eq!(entries.len(), 3);
}

#[test]
fn test_tropical_weight_semantics() {
    // Verify TropicalWeight behaves as expected
    let w1 = TropicalWeight::new(1.0);
    let w2 = TropicalWeight::new(2.0);

    // Plus is min
    let sum = w1.plus(&w2);
    assert_eq!(sum.value(), 1.0);

    // Times is add
    let prod = w1.times(&w2);
    assert_eq!(prod.value(), 3.0);

    // Zero is infinity
    let zero = TropicalWeight::zero();
    assert!(zero.is_infinite());

    // One is 0.0
    let one = TropicalWeight::one();
    assert_eq!(one.value(), 0.0);
}

#[test]
fn test_wfst_transposition_algorithm() {
    use liblevenshtein::transducer::Algorithm;

    let dict = DynamicDawgChar::<()>::from_terms(vec!["test", "tset"]);
    let wfst = LevenshteinWfst::with_algorithm(&dict, "tset", 1, Algorithm::Transposition);

    assert_eq!(wfst.algorithm(), Algorithm::Transposition);
}

#[test]
fn test_dictionary_backend_with_vocabulary() {
    let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world", "test"]);
    let terms = vec!["hello".to_string(), "world".to_string()];
    let backend = DictionaryBackend::with_vocabulary(dict, terms);

    assert_eq!(backend.vocab_size(), 2);
    assert!(backend.get_id("hello").is_some());
    assert!(backend.get_id("world").is_some());
    assert!(backend.get_id("test").is_none()); // Not pre-populated
}

// MSM (Move-Split-Merge) WFST Tests
mod msm_tests {
    use liblevenshtein::time_series::MsmConfig;
    use liblevenshtein::wfst::msm::{MsmWfstBuilder, MsmWeight};
    use lling_llang::prelude::{LazyWfst, Semiring, Wfst};

    #[test]
    fn test_msm_wfst_creation() {
        let wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0, 3.0])
            .msm_config(MsmConfig::new(1.0))
            .max_cost(10.0)
            .add_target(0, &[1.0, 2.0, 3.0])
            .build()
            .expect("build failed");

        assert!(!wfst.is_empty());
        assert_eq!(wfst.query().len(), 3);
        assert_eq!(wfst.max_cost(), 10.0);
    }

    #[test]
    fn test_msm_wfst_identical_series() {
        let query = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];

        let mut wfst = MsmWfstBuilder::new()
            .query(&query)
            .msm_config(MsmConfig::new(1.0))
            .max_cost(10.0)
            .add_target(0, &target)
            .build()
            .expect("build failed");

        let start = Wfst::start(&wfst);
        wfst.expand(start);

        // Start state should be valid
        assert!(wfst.is_valid_state(start));
        assert!(wfst.is_expanded(start));
    }

    #[test]
    fn test_msm_wfst_multiple_targets() {
        let query = vec![1.0, 2.0, 3.0];

        let wfst = MsmWfstBuilder::new()
            .query(&query)
            .msm_config(MsmConfig::new(1.0))
            .max_cost(20.0)
            .add_target(0, &[1.0, 2.0, 3.0])
            .add_target(1, &[2.0, 3.0, 4.0])
            .add_target(2, &[1.5, 2.5, 3.5])
            .build()
            .expect("build failed");

        assert!(!wfst.is_empty());
        // Should have states for multiple series
        assert!(wfst.num_states() >= 3);
    }

    #[test]
    fn test_msm_wfst_lazy_expansion() {
        let mut wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0])
            .add_target(0, &[1.0, 2.0])
            .build()
            .expect("build failed");

        let start = Wfst::start(&wfst);
        let initial = wfst.computed_states();

        wfst.expand(start);
        assert!(wfst.is_expanded(start));
        assert!(wfst.computed_states() >= initial);
    }

    #[test]
    fn test_msm_weight_semiring() {
        // Zero element
        let zero = MsmWeight::zero();
        assert!(zero.cost().is_infinite());

        // One element
        let one = MsmWeight::one();
        assert!((one.cost() - 0.0).abs() < 1e-9);

        // Plus is min
        let w1 = MsmWeight::new(1.0, 0.0, 0.0, 1.0);
        let w2 = MsmWeight::new(2.0, 0.0, 0.0, 1.0);
        let plus = w1.plus(&w2);
        assert!((plus.cost() - 1.0).abs() < 1e-9);

        // Times is add
        let times = w1.times(&w2);
        assert!((times.cost() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_msm_weight_operations() {
        let w = MsmWeight::initial(1.0, 2.0, 1.0);

        // Move operation: cost = |query - target|
        let w_move = w.with_move(1.5, 2.5);
        assert!((w_move.cost() - 1.0).abs() < 1e-9); // |1.5 - 2.5| = 1.0
        assert!((w_move.last_query_value() - 1.5).abs() < 1e-9);
        assert!((w_move.last_target_value() - 2.5).abs() < 1e-9);

        // Merge operation uses C function
        let w_merge = w.with_merge(1.5);
        assert!((w_merge.last_query_value() - 1.5).abs() < 1e-9);
        assert!((w_merge.last_target_value() - 2.0).abs() < 1e-9);

        // Split operation uses C function
        let w_split = w.with_split(2.5);
        assert!((w_split.last_query_value() - 1.0).abs() < 1e-9);
        assert!((w_split.last_target_value() - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_msm_wfst_cache_operations() {
        let mut wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0])
            .add_target(0, &[1.0, 2.0])
            .build()
            .expect("build failed");

        wfst.expand(0);
        let before = wfst.computed_states();

        wfst.clear_cache();
        assert_eq!(wfst.computed_states(), 0);
        assert!(before > 0);
    }

    #[test]
    fn test_msm_wfst_high_cost_threshold() {
        let mut wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0, 3.0])
            .msm_config(MsmConfig::new(1.0))
            .max_cost(100.0) // High threshold
            .add_target(0, &[10.0, 20.0, 30.0])
            .build()
            .expect("build failed");

        let start = Wfst::start(&wfst);
        wfst.expand(start);

        // Should have transitions even with very different values
        let _transitions = wfst.transitions(start);
        // Start state should have some transitions
        assert!(wfst.is_valid_state(start));
    }

    #[test]
    fn test_msm_wfst_low_cost_threshold() {
        let wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0, 3.0])
            .msm_config(MsmConfig::new(1.0))
            .max_cost(0.5) // Very restrictive
            .add_target(0, &[10.0, 20.0, 30.0])
            .build()
            .expect("build failed");

        // With such a low threshold, most states should be pruned
        // But the structure should still be valid
        assert!(!wfst.is_empty());
    }
}

// Generalized Automata WFST Tests
mod generalized_tests {
    use libdictenstein::dynamic_dawg_char::DynamicDawgChar;
    use liblevenshtein::transducer::OperationSet;
    use liblevenshtein::wfst::{GeneralizedWfst, GeneralizedWfstBuilder, Wfst};
    use lling_llang::prelude::LazyWfst;

    #[test]
    fn test_generalized_wfst_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = GeneralizedWfst::new(&dict, "helo", 2, OperationSet::standard());

        assert!(!wfst.is_empty());
        assert_eq!(wfst.query(), "helo");
        assert_eq!(wfst.max_distance(), 2);
    }

    #[test]
    fn test_generalized_wfst_lazy_expansion() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "help"]);
        let mut wfst = GeneralizedWfst::new(&dict, "helo", 2, OperationSet::standard());

        let start = Wfst::start(&wfst);
        assert!(!wfst.is_expanded(start));

        wfst.expand(start);
        assert!(wfst.is_expanded(start));
    }

    #[test]
    fn test_generalized_wfst_with_transposition() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test", "tset"]);
        let result = GeneralizedWfstBuilder::new(&dict)
            .query("tset")
            .max_distance(1)
            .with_transposition()
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_generalized_wfst_builder() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["hello", "world"]);
        let result = GeneralizedWfstBuilder::new(&dict)
            .query("helo")
            .max_distance(2)
            .with_standard_ops()
            .build();

        assert!(result.is_ok());
        let wfst = result.unwrap();
        assert_eq!(wfst.query(), "helo");
    }

    #[test]
    fn test_generalized_wfst_cache_operations() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let mut wfst = GeneralizedWfst::new(&dict, "test", 1, OperationSet::standard());

        wfst.expand(0);
        let before = wfst.computed_states();

        wfst.clear_cache();
        assert_eq!(wfst.computed_states(), 0);
        assert!(before > 0);
    }
}

// WallBreaker WFST Tests
mod wallbreaker_tests {
    use libdictenstein::scdawg::Scdawg;
    use liblevenshtein::transducer::Algorithm;
    use liblevenshtein::wfst::{StateSource, WallBreakerWfst, WallBreakerWfstBuilder, Wfst};
    use lling_llang::prelude::LazyWfst;

    #[test]
    fn test_wallbreaker_wfst_creation() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = WallBreakerWfst::new(&dict, "helo", 2);

        assert!(!wfst.is_empty());
        assert_eq!(wfst.query(), "helo");
        assert_eq!(wfst.max_distance(), 2);
    }

    #[test]
    fn test_wallbreaker_wfst_lazy_expansion() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "help"]);
        let mut wfst = WallBreakerWfst::new(&dict, "helo", 2);

        let start = Wfst::start(&wfst);
        assert!(!wfst.is_expanded(start));

        wfst.expand(start);
        assert!(wfst.is_expanded(start));
    }

    #[test]
    fn test_wallbreaker_wfst_with_transposition() {
        let dict = Scdawg::<()>::from_terms(vec!["test", "tset"]);
        let wfst = WallBreakerWfst::with_algorithm(&dict, "tset", 1, Algorithm::Transposition);

        assert!(matches!(wfst.algorithm(), Algorithm::Transposition));
    }

    #[test]
    fn test_wallbreaker_wfst_builder() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let result = WallBreakerWfstBuilder::new(&dict)
            .query("helo")
            .max_distance(2)
            .standard()
            .build();

        assert!(result.is_ok());
        let wfst = result.unwrap();
        assert_eq!(wfst.query(), "helo");
    }

    #[test]
    fn test_wallbreaker_wfst_num_results() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "help", "world"]);
        let wfst = WallBreakerWfst::new(&dict, "helo", 2);

        // Should have found matches
        assert!(wfst.num_results() > 0);
    }

    #[test]
    fn test_wallbreaker_wfst_cache_operations() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let mut wfst = WallBreakerWfst::new(&dict, "test", 1);

        wfst.expand(0);
        let before = wfst.computed_states();

        wfst.clear_cache();
        assert_eq!(wfst.computed_states(), 0);
        assert!(before > 0);
    }

    #[test]
    fn test_wallbreaker_wfst_state_hint() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let wfst = WallBreakerWfst::new(&dict, "helo", 2);

        let hint = StateSource::num_states_hint(&wfst);
        assert!(hint.is_some());
        assert!(hint.unwrap() > 0);
    }
}
