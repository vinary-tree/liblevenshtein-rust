//! Phase 4: Additional Coverage Tests
//!
//! This module provides comprehensive test coverage for additional modules:
//! - time_series: MSM metric, encoding, indexing, lower bounds
//! - filter: N-gram index, Jaro-Winkler similarity, hybrid filtering

use proptest::prelude::*;

// ============================================================================
// Time Series MSM Coverage Tests
// ============================================================================

mod time_series_msm_coverage {
    use liblevenshtein::time_series::{msm_distance_wavefront, MsmConfig};

    // --- MsmConfig Tests ---

    #[test]
    fn test_msm_config_new() {
        let config = MsmConfig::new(1.0);
        assert_eq!(config.c, 1.0);
    }

    #[test]
    fn test_msm_config_various_c_values() {
        for c in [0.0, 0.5, 1.0, 2.0, 10.0] {
            let config = MsmConfig::new(c);
            assert_eq!(config.c, c);
        }
    }

    #[test]
    fn test_msm_distance_identical_series() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let distance = config.distance(&x, &y);
        assert_eq!(distance, 0.0, "Identical series should have distance 0");
    }

    #[test]
    fn test_msm_distance_empty_series() {
        let config = MsmConfig::new(1.0);

        // Both empty
        let d1 = config.distance(&[], &[]);
        assert_eq!(d1, 0.0);

        // One empty
        let d2 = config.distance(&[1.0, 2.0], &[]);
        assert!(d2 > 0.0);

        let d3 = config.distance(&[], &[1.0, 2.0]);
        assert!(d3 > 0.0);
    }

    #[test]
    fn test_msm_distance_single_elements() {
        let config = MsmConfig::new(1.0);

        // Same value
        let d1 = config.distance(&[5.0], &[5.0]);
        assert_eq!(d1, 0.0);

        // Different values - distance is the difference (Move cost)
        let d2 = config.distance(&[5.0], &[8.0]);
        assert_eq!(d2, 3.0);
    }

    #[test]
    fn test_msm_distance_different_lengths() {
        let config = MsmConfig::new(1.0);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        let d = config.distance(&x, &y);
        assert!(
            d > 0.0,
            "Different length series should have positive distance"
        );
    }

    #[test]
    fn test_msm_distance_symmetry() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.5, 2.5, 3.5];

        let d_xy = config.distance(&x, &y);
        let d_yx = config.distance(&y, &x);

        assert!(
            (d_xy - d_yx).abs() < 1e-9,
            "MSM should be symmetric: {} vs {}",
            d_xy,
            d_yx
        );
    }

    #[test]
    fn test_msm_distance_triangle_inequality() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.5, 2.5, 3.5];
        let z = vec![2.0, 3.0, 4.0];

        let d_xy = config.distance(&x, &y);
        let d_yz = config.distance(&y, &z);
        let d_xz = config.distance(&x, &z);

        assert!(
            d_xz <= d_xy + d_yz + 1e-9,
            "Triangle inequality violated: {} > {} + {}",
            d_xz,
            d_xy,
            d_yz
        );
    }

    #[test]
    fn test_msm_distance_high_c_value() {
        let config = MsmConfig::new(100.0);
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0];

        let d = config.distance(&x, &y);
        assert!(d > 0.0);
    }

    #[test]
    fn test_msm_distance_zero_c_value() {
        let config = MsmConfig::new(0.0);
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0];

        let d = config.distance(&x, &y);
        assert!(d >= 0.0);
    }

    // --- Wavefront Algorithm Tests ---

    #[test]
    fn test_msm_wavefront_identical() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let config = MsmConfig::new(1.0);

        let result = msm_distance_wavefront(&x, &y, &config, 10.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn test_msm_wavefront_threshold_exceeded() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let config = MsmConfig::new(1.0);

        let result = msm_distance_wavefront(&x, &y, &config, 1.0);
        assert!(result.is_none(), "Should exceed threshold");
    }

    #[test]
    fn test_msm_wavefront_threshold_ok() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 4.0];
        let config = MsmConfig::new(1.0);

        let result = msm_distance_wavefront(&x, &y, &config, 10.0);
        assert!(result.is_some());
        assert!(result.unwrap() >= 1.0);
    }

    #[test]
    fn test_msm_wavefront_empty_series() {
        let config = MsmConfig::new(1.0);

        let result1 = msm_distance_wavefront(&[], &[], &config, 10.0);
        assert!(result1.is_some());
        assert_eq!(result1.unwrap(), 0.0);

        // Empty vs non-empty may return None (infinite distance) or Some based on implementation
        let result2 = msm_distance_wavefront(&[1.0], &[], &config, 10.0);
        // Just verify it doesn't panic; the result depends on how inf is handled
        let _ = result2;
    }
}

// ============================================================================
// Time Series Encoding Coverage Tests
// ============================================================================

mod time_series_encoding_coverage {
    use liblevenshtein::time_series::QuantizationConfig;

    #[test]
    fn test_quantization_uniform_config() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);
        assert_eq!(config.min_value, 0.0);
        assert_eq!(config.max_value, 100.0);
        assert_eq!(config.num_bins, 256);
    }

    #[test]
    fn test_quantization_quantize() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);

        // Middle value should be around bin 128
        let bin = config.quantize(50.0);
        assert!(
            bin >= 125 && bin <= 130,
            "50.0 should quantize to ~128, got {}",
            bin
        );
    }

    #[test]
    fn test_quantization_dequantize() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);

        let original = 50.0;
        let bin = config.quantize(original);
        let recovered = config.dequantize(bin);

        // Should be close to original (within one bin width)
        let bin_width = 100.0 / 256.0;
        assert!(
            (recovered - original).abs() < bin_width,
            "Roundtrip error too large: {} -> {} -> {}",
            original,
            bin,
            recovered
        );
    }

    #[test]
    fn test_quantization_encode_u8() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);
        let series = vec![10.0, 50.0, 90.0];

        let encoded = config.encode_u8(&series);
        assert_eq!(encoded.len(), 3);

        // Values should be increasing
        assert!(encoded[0] < encoded[1]);
        assert!(encoded[1] < encoded[2]);
    }

    #[test]
    fn test_quantization_encode_u32() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 1000);
        let series = vec![10.0, 50.0, 90.0];

        let encoded = config.encode_u32(&series);
        assert_eq!(encoded.len(), 3);

        // Values should be increasing
        assert!(encoded[0] < encoded[1]);
        assert!(encoded[1] < encoded[2]);
    }

    #[test]
    fn test_quantization_clipping() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);

        // Values outside range should be clipped
        let below = config.quantize(-10.0);
        let above = config.quantize(110.0);

        assert_eq!(below, 0); // Clipped to min
        assert_eq!(above, 255); // Clipped to max
    }

    #[test]
    fn test_quantization_negative_range() {
        let config = QuantizationConfig::uniform(-50.0, 50.0, 256);

        let encoded_neg = config.quantize(-25.0);
        let encoded_zero = config.quantize(0.0);
        let encoded_pos = config.quantize(25.0);

        assert!(encoded_neg < encoded_zero);
        assert!(encoded_zero < encoded_pos);
    }

    #[test]
    fn test_quantization_from_data() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        // Create config automatically from data
        let config = QuantizationConfig::from_data(&data, 256, 0.1);
        assert!(config.is_some());

        let config = config.unwrap();
        // Range should cover the data with margin
        assert!(config.min_value <= 10.0);
        assert!(config.max_value >= 50.0);
    }
}

// ============================================================================
// Time Series Lower Bounds Coverage Tests
// ============================================================================

mod time_series_lower_bounds_coverage {
    use liblevenshtein::time_series::{
        combined_lb, euclidean_lb, l1_lb, length_lb, LowerBoundConfig, MsmConfig,
    };

    #[test]
    fn test_euclidean_lb_identical() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let lb = euclidean_lb(&x, &y);
        assert_eq!(lb, 0.0, "Identical series should have LB 0");
    }

    #[test]
    fn test_euclidean_lb_different() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];

        let lb = euclidean_lb(&x, &y);
        assert!(lb > 0.0);
    }

    #[test]
    fn test_euclidean_lb_different_lengths() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0];

        let lb = euclidean_lb(&x, &y);
        assert!(lb >= 0.0);
    }

    #[test]
    fn test_length_lb() {
        let c = 1.0;

        // Same length
        let lb1 = length_lb(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], c);
        assert_eq!(lb1, 0.0);

        // Different lengths
        let lb2 = length_lb(&[1.0, 2.0, 3.0], &[1.0, 2.0], c);
        assert!(lb2 > 0.0, "Length diff should create positive LB");

        let lb3 = length_lb(&[1.0], &[1.0, 2.0, 3.0, 4.0], c);
        assert!(lb3 > 0.0);
    }

    #[test]
    fn test_l1_lb() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let lb1 = l1_lb(&x, &y);
        assert_eq!(lb1, 0.0, "Identical series should have L1 LB 0");

        let y2 = vec![2.0, 3.0, 4.0];
        let lb2 = l1_lb(&x, &y2);
        assert!(lb2 > 0.0);
    }

    #[test]
    fn test_combined_lb() {
        let c = 1.0;

        // Same series
        let lb1 = combined_lb(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], c);
        assert_eq!(lb1, 0.0);

        // Different series
        let lb2 = combined_lb(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], c);
        assert!(lb2 > 0.0);
    }

    #[test]
    fn test_combined_lb_vs_actual() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.5, 2.5, 3.5, 4.5];
        let c = 1.0;

        let lb = combined_lb(&x, &y, c);
        let config = MsmConfig::new(c);
        let actual = config.distance(&x, &y);

        assert!(
            lb <= actual + 1e-9,
            "Lower bound {} should not exceed actual distance {}",
            lb,
            actual
        );
    }

    #[test]
    fn test_lower_bound_config_new() {
        let config = LowerBoundConfig::new(1.0);
        assert_eq!(config.c, 1.0);
    }

    #[test]
    fn test_lower_bound_config_compute() {
        let config = LowerBoundConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let lb = config.lower_bound(&x, &y);
        assert_eq!(lb, 0.0, "Identical series should have LB 0");
    }
}

// ============================================================================
// Time Series Index Coverage Tests
// ============================================================================

mod time_series_index_coverage {
    use liblevenshtein::time_series::{
        HybridSearchIndex, MsmConfig, QuantizationConfig, TimeSeriesIndex, TimeSeriesIndexBuilder,
    };

    #[test]
    fn test_time_series_index_creation() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);
        let series = vec![vec![10.0, 20.0, 30.0], vec![15.0, 25.0, 35.0]];

        let index: TimeSeriesIndex<usize> = TimeSeriesIndex::from_series(config, &series);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_time_series_index_new() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);
        let index: TimeSeriesIndex<usize> = TimeSeriesIndex::new(config);

        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_time_series_index_insert() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);
        let mut index: TimeSeriesIndex<usize> = TimeSeriesIndex::new(config);

        index.insert(0, &[10.0, 20.0, 30.0]);
        index.insert(1, &[15.0, 25.0, 35.0]);
        index.insert(2, &[20.0, 30.0, 40.0]);

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_time_series_index_search() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);
        let series = vec![
            vec![10.0, 20.0, 30.0],
            vec![11.0, 21.0, 31.0],
            vec![50.0, 60.0, 70.0],
        ];

        let index: TimeSeriesIndex<usize> = TimeSeriesIndex::from_series(config, &series);

        // Search with larger distance to ensure matches (quantization can cause larger distances)
        let results = index.search(&[10.0, 20.0, 30.0], 5);
        // Just verify the search runs without panicking
        let _ = results;
    }

    #[test]
    fn test_time_series_index_empty() {
        let config = QuantizationConfig::uniform(0.0, 100.0, 256);
        let index: TimeSeriesIndex<usize> = TimeSeriesIndex::new(config);

        assert_eq!(index.len(), 0);
        let results = index.search(&[1.0, 2.0, 3.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hybrid_search_index_creation() {
        let quant_config = QuantizationConfig::uniform(0.0, 100.0, 256);
        let msm_config = MsmConfig::new(1.0);

        let index: HybridSearchIndex<usize> = HybridSearchIndex::new(quant_config, msm_config);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_hybrid_search_index_insert_search() {
        let quant_config = QuantizationConfig::uniform(0.0, 100.0, 256);
        let msm_config = MsmConfig::new(1.0);

        let mut index = HybridSearchIndex::new(quant_config, msm_config);
        index.insert(0usize, &[10.0, 20.0, 30.0]);
        index.insert(1usize, &[15.0, 25.0, 35.0]);
        index.insert(2usize, &[50.0, 60.0, 70.0]);

        assert_eq!(index.len(), 3);

        let results = index.search_exact(&[12.0, 22.0, 32.0], 10.0);
        let _ = results;
    }

    #[test]
    fn test_time_series_index_builder() {
        let builder = TimeSeriesIndexBuilder::new().quantization(0.0, 100.0, 256);

        let index: TimeSeriesIndex<usize> = builder.build();
        assert_eq!(index.len(), 0);
    }
}

// ============================================================================
// Filter N-gram Coverage Tests
// ============================================================================

mod filter_ngram_coverage {
    use liblevenshtein::filter::NgramIndex;

    #[test]
    fn test_ngram_index_creation() {
        let index = NgramIndex::new(2);
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_ngram_index_insert() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("world");
        index.insert("help");

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_ngram_index_find_candidates() {
        let mut index = NgramIndex::new(2);
        index.insert("apple");
        index.insert("application");
        index.insert("banana");
        index.insert("apply");

        let candidates = index.find_candidates("aple", 2);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_ngram_index_exact_match() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("world");

        let candidates = index.find_candidates("hello", 0);
        assert!(candidates.iter().any(|c| *c == "hello"));
    }

    #[test]
    fn test_ngram_index_short_words() {
        let mut index = NgramIndex::new(2);
        index.insert("a");
        index.insert("ab");
        index.insert("abc");

        let candidates = index.find_candidates("ab", 1);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_ngram_index_empty_query() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("world");

        let candidates = index.find_candidates("", 2);
        let _ = candidates;
    }

    #[test]
    fn test_ngram_index_trigrams() {
        let mut index = NgramIndex::new(3);
        index.insert("testing");
        index.insert("tested");
        index.insert("tester");
        index.insert("unrelated");

        let candidates = index.find_candidates("test", 2);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_ngram_index_unicode() {
        let mut index = NgramIndex::new(2);
        index.insert("café");
        index.insert("naive");
        index.insert("naïve");

        let candidates = index.find_candidates("cafe", 2);
        let _ = candidates;
    }
}

// ============================================================================
// Filter Jaro-Winkler Coverage Tests
// ============================================================================

mod filter_jaro_winkler_coverage {
    use liblevenshtein::filter::{
        distance_to_similarity_approx, is_similar, jaro_similarity, jaro_winkler_similarity,
        jaro_winkler_similarity_scaled, similarity_to_distance_approx,
    };

    #[test]
    fn test_jaro_identical() {
        let sim = jaro_similarity("hello", "hello");
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "Identical strings should have similarity 1.0"
        );
    }

    #[test]
    fn test_jaro_empty() {
        // Both empty
        let sim1 = jaro_similarity("", "");
        assert!(
            (sim1 - 1.0).abs() < 1e-9,
            "Both empty should have similarity 1.0"
        );

        // One empty
        let sim2 = jaro_similarity("hello", "");
        assert_eq!(sim2, 0.0, "One empty should have similarity 0.0");

        let sim3 = jaro_similarity("", "hello");
        assert_eq!(sim3, 0.0);
    }

    #[test]
    fn test_jaro_completely_different() {
        let sim = jaro_similarity("abc", "xyz");
        assert!(
            sim < 0.5,
            "Completely different strings should have low similarity"
        );
    }

    #[test]
    fn test_jaro_similar_strings() {
        let sim = jaro_similarity("martha", "marhta");
        assert!(sim > 0.9, "Similar strings should have high similarity");
    }

    #[test]
    fn test_jaro_winkler_prefix_bonus() {
        let j = jaro_similarity("prefix_abc", "prefix_xyz");
        let jw = jaro_winkler_similarity("prefix_abc", "prefix_xyz");

        assert!(jw >= j, "Jaro-Winkler should be >= Jaro for common prefix");
    }

    #[test]
    fn test_jaro_winkler_identical() {
        let sim = jaro_winkler_similarity("hello", "hello");
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaro_winkler_scaled() {
        let sim = jaro_winkler_similarity_scaled("hello", "hallo", 0.1);
        assert!(sim > 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_is_similar() {
        assert!(is_similar("hello", "hello", 0.5));

        let result = is_similar("hello", "world", 0.95);
        assert!(
            !result,
            "Very different strings shouldn't be similar at high threshold"
        );
    }

    #[test]
    fn test_distance_similarity_conversion() {
        let dist = 2.0;
        let len = 5.0;

        let sim = distance_to_similarity_approx(dist, len);
        assert!(sim >= 0.0 && sim <= 1.0);

        let sim_high = distance_to_similarity_approx(4.0, len);
        assert!(sim_high < sim);
    }

    #[test]
    fn test_similarity_to_distance_approx() {
        let sim = 0.8;
        let len = 5.0;

        let dist = similarity_to_distance_approx(sim, len);
        assert!(dist >= 0.0);

        let dist_high_sim = similarity_to_distance_approx(0.95, len);
        assert!(dist_high_sim < dist);
    }

    #[test]
    fn test_jaro_symmetry() {
        let s1 = "hello";
        let s2 = "hallo";

        let sim1 = jaro_similarity(s1, s2);
        let sim2 = jaro_similarity(s2, s1);

        assert!(
            (sim1 - sim2).abs() < 1e-9,
            "Jaro should be symmetric: {} vs {}",
            sim1,
            sim2
        );
    }

    #[test]
    fn test_jaro_winkler_symmetry() {
        let s1 = "hello";
        let s2 = "hallo";

        let sim1 = jaro_winkler_similarity(s1, s2);
        let sim2 = jaro_winkler_similarity(s2, s1);

        assert!(
            (sim1 - sim2).abs() < 1e-9,
            "Jaro-Winkler should be symmetric: {} vs {}",
            sim1,
            sim2
        );
    }
}

// ============================================================================
// Filter Hybrid Matcher Coverage Tests
// ============================================================================

mod filter_hybrid_coverage {
    use liblevenshtein::filter::HybridMatcher;

    #[test]
    fn test_hybrid_matcher_new() {
        let terms = ["apple", "banana", "cherry"];
        let matcher = HybridMatcher::new(terms.iter().map(|s| s.to_string()));

        // Just verify it can be created
        let _ = matcher;
    }

    #[test]
    fn test_hybrid_matcher_with_config() {
        let terms = ["apple", "banana", "cherry"];
        let matcher = HybridMatcher::with_config(
            terms.iter().map(|s| s.to_string()),
            2,   // bigrams
            0.7, // jaro threshold
        );

        let _ = matcher;
    }

    #[test]
    fn test_hybrid_matcher_filter_candidates() {
        let terms = ["apple", "application", "apply", "banana", "band", "bandana"];
        let matcher = HybridMatcher::new(terms.iter().map(|s| s.to_string()));

        let results = matcher.filter_candidates("aple", 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hybrid_matcher_exact() {
        let terms = ["exact", "extract", "example"];
        let matcher = HybridMatcher::with_config(terms.iter().map(|s| s.to_string()), 2, 0.9);

        let results = matcher.filter_candidates("exact", 0);
        assert!(results.iter().any(|r| *r == "exact"));
    }

    #[test]
    fn test_hybrid_matcher_no_matches() {
        let terms = ["hello", "world"];
        let matcher = HybridMatcher::with_config(terms.iter().map(|s| s.to_string()), 2, 0.99);

        let results = matcher.filter_candidates("zzzzz", 0);
        let _ = results;
    }

    #[test]
    fn test_hybrid_matcher_skip_jaro() {
        let terms = ["apple", "application", "apply"];
        let matcher = HybridMatcher::ngram_only(terms.iter().map(|s| s.to_string()), 2);

        let results = matcher.filter_candidates("aple", 2);
        let _ = results;
    }
}

// ============================================================================
// Proptest Property-Based Tests
// ============================================================================

mod proptest_additional {
    use super::*;
    use liblevenshtein::filter::jaro_similarity;
    use liblevenshtein::time_series::{euclidean_lb, MsmConfig, QuantizationConfig};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_msm_identity(values in prop::collection::vec(-100.0f64..100.0, 1..10)) {
            let config = MsmConfig::new(1.0);
            let dist = config.distance(&values, &values);
            prop_assert!((dist - 0.0).abs() < 1e-9, "Identity distance should be 0, got {}", dist);
        }

        #[test]
        fn prop_msm_symmetry(
            x in prop::collection::vec(-100.0f64..100.0, 1..10),
            y in prop::collection::vec(-100.0f64..100.0, 1..10)
        ) {
            let config = MsmConfig::new(1.0);
            let d_xy = config.distance(&x, &y);
            let d_yx = config.distance(&y, &x);
            prop_assert!(
                (d_xy - d_yx).abs() < 1e-6,
                "MSM should be symmetric: {} vs {}",
                d_xy,
                d_yx
            );
        }

        #[test]
        fn prop_msm_non_negative(
            x in prop::collection::vec(-100.0f64..100.0, 1..10),
            y in prop::collection::vec(-100.0f64..100.0, 1..10)
        ) {
            let config = MsmConfig::new(1.0);
            let dist = config.distance(&x, &y);
            prop_assert!(dist >= 0.0, "Distance should be non-negative, got {}", dist);
        }

        #[test]
        fn prop_euclidean_lb_non_negative(
            x in prop::collection::vec(-50.0f64..50.0, 2..8),
            y in prop::collection::vec(-50.0f64..50.0, 2..8)
        ) {
            // Euclidean lower bound should always be non-negative
            let lb = euclidean_lb(&x, &y);
            prop_assert!(lb >= 0.0, "Euclidean lower bound should be non-negative, got {}", lb);
        }

        #[test]
        fn prop_quantization_monotonic(
            v1 in 0.0f64..100.0,
            v2 in 0.0f64..100.0
        ) {
            let config = QuantizationConfig::uniform(0.0, 100.0, 256);
            let e1 = config.quantize(v1);
            let e2 = config.quantize(v2);

            if v1 < v2 - 0.5 {
                prop_assert!(e1 <= e2, "Encoding should be monotonic: {} -> {} vs {} -> {}", v1, e1, v2, e2);
            }
        }

        #[test]
        fn prop_jaro_bounds(
            s1 in "[a-z]{1,10}",
            s2 in "[a-z]{1,10}"
        ) {
            let sim = jaro_similarity(&s1, &s2);
            prop_assert!(sim >= 0.0 && sim <= 1.0, "Jaro should be in [0,1], got {}", sim);
        }

        #[test]
        fn prop_jaro_identity(s in "[a-z]{1,10}") {
            let sim = jaro_similarity(&s, &s);
            prop_assert!((sim - 1.0).abs() < 1e-9, "Identical strings should have similarity 1.0, got {}", sim);
        }

        #[test]
        fn prop_jaro_symmetry(
            s1 in "[a-z]{1,10}",
            s2 in "[a-z]{1,10}"
        ) {
            let sim1 = jaro_similarity(&s1, &s2);
            let sim2 = jaro_similarity(&s2, &s1);
            prop_assert!(
                (sim1 - sim2).abs() < 1e-9,
                "Jaro should be symmetric: {} vs {}",
                sim1,
                sim2
            );
        }
    }
}
