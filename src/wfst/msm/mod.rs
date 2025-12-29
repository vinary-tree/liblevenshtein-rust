//! MSM (Move-Split-Merge) WFST integration for time series similarity.
//!
//! This module provides WFST wrappers for the MSM time series metric,
//! enabling composition with other WFSTs in similarity search pipelines.
//!
//! # Overview
//!
//! The MSM metric is a metric for time series that allows:
//! - **Move**: Change a value by some amount (cost = |change|)
//! - **Split**: Duplicate a value into two consecutive elements (cost = c)
//! - **Merge**: Combine two consecutive equal-value elements (cost = c)
//!
//! Unlike string Levenshtein distance, MSM has **data-dependent costs** where
//! the split/merge cost depends on the actual values being compared through
//! the C(a, b, c) function.
//!
//! # Key Types
//!
//! - [`MsmWeight`]: Custom semiring weight that tracks data values
//! - [`MsmCompositeState`]: State encoding with value tracking
//! - [`MsmStateRegistry`]: State deduplication registry
//! - [`MsmWfst`]: Main WFST wrapper
//! - [`MsmStateSource`]: Lazy state computation source
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::wfst::msm::{MsmWfst, MsmWfstBuilder};
//! use liblevenshtein::time_series::MsmConfig;
//!
//! // Create query and target series
//! let query = vec![1.0, 2.0, 3.0];
//! let target1 = vec![1.0, 2.0, 3.0];  // Identical
//! let target2 = vec![2.0, 3.0, 4.0];  // Shifted
//!
//! // Build MSM WFST
//! let wfst = MsmWfstBuilder::new()
//!     .query(&query)
//!     .msm_config(MsmConfig::new(1.0))
//!     .max_cost(10.0)
//!     .add_target(0, &target1)
//!     .add_target(1, &target2)
//!     .build()
//!     .expect("build failed");
//!
//! // Use in WFST pipeline...
//! ```
//!
//! # Feature Gate
//!
//! This module requires the `wfst` feature:
//!
//! ```toml
//! [dependencies]
//! liblevenshtein = { version = "0.8", features = ["wfst"] }
//! ```

mod adaptive;
mod state;
mod state_source;
mod weight;
mod wrapper;

// Re-export public types
pub use adaptive::{AdaptiveMsm, AdaptiveMsmBuilder, AdaptiveMsmConfig, AdaptiveMsmStatistics};
pub use state::{
    decode_msm_state, encode_msm_state, estimate_msm_states, MsmCompositeState, MsmStateKey,
    MsmStateRegistry, MsmTransition,
};
pub use state_source::{MsmStateSource, MsmStateSourceBuilder};
pub use weight::MsmWeight;
pub use wrapper::{MsmWfst, MsmWfstBuilder};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::time_series::MsmConfig;
    use lling_llang::prelude::{LazyWfst, StateSource, Wfst};

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
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

        assert!(!wfst.is_empty());

        // Expand start state
        let start = Wfst::start(&wfst);
        wfst.expand(start);

        // Should be a valid starting point
        assert!(wfst.is_valid_state(start));
    }

    #[test]
    fn test_msm_wfst_shifted_series() {
        let query = vec![1.0, 2.0, 3.0];
        let target = vec![2.0, 3.0, 4.0]; // All values shifted by 1

        let mut wfst = MsmWfstBuilder::new()
            .query(&query)
            .msm_config(MsmConfig::new(1.0))
            .max_cost(10.0)
            .add_target(0, &target)
            .build()
            .expect("build failed");

        // Expand to compute states
        let start = Wfst::start(&wfst);
        wfst.expand(start);

        // Should find transitions
        let transitions = wfst.transitions(start);
        // Initial transitions may exist depending on state
    }

    #[test]
    fn test_msm_wfst_with_threshold() {
        let query = vec![1.0, 2.0, 3.0];
        let target = vec![10.0, 20.0, 30.0]; // Very different

        // Low threshold should restrict states
        let wfst = MsmWfstBuilder::new()
            .query(&query)
            .msm_config(MsmConfig::new(1.0))
            .max_cost(1.0) // Very restrictive
            .add_target(0, &target)
            .build()
            .expect("build failed");

        // With such a low threshold, most paths should be pruned
    }

    #[test]
    fn test_msm_weight_operations() {
        let w1 = MsmWeight::initial(1.0, 2.0, 1.0);

        // Move operation
        let w2 = w1.with_move(1.5, 2.5);
        assert!(approx_eq(w2.cost(), 1.0)); // |1.5 - 2.5| = 1.0
        assert!(approx_eq(w2.last_query_value(), 1.5));
        assert!(approx_eq(w2.last_target_value(), 2.5));

        // Merge operation (from w1)
        let w3 = w1.with_merge(1.5); // C(1.5, 1.0, 2.0) = 1.0 (1.5 between 1.0 and 2.0)
        assert!(approx_eq(w3.cost(), 1.0));
        assert!(approx_eq(w3.last_query_value(), 1.5));
        assert!(approx_eq(w3.last_target_value(), 2.0)); // unchanged

        // Split operation (from w1)
        let w4 = w1.with_split(2.5); // C(2.5, 1.0, 2.0) = 1.0 + 0.5 = 1.5 (2.5 > 2.0)
        assert!(approx_eq(w4.cost(), 1.5));
        assert!(approx_eq(w4.last_query_value(), 1.0)); // unchanged
        assert!(approx_eq(w4.last_target_value(), 2.5));
    }

    #[test]
    fn test_msm_state_encoding() {
        let max_q = 5;
        let max_t = 8;

        // Test roundtrip for various states
        for series_id in [0, 1, 5] {
            for q_idx in [0, 2, 5] {
                for t_idx in [0, 4, 8] {
                    let encoded = encode_msm_state(series_id, q_idx, t_idx, max_q, max_t);
                    let (dec_s, dec_q, dec_t) = decode_msm_state(encoded, max_q, max_t);

                    assert_eq!(dec_s, series_id);
                    assert_eq!(dec_q, q_idx);
                    assert_eq!(dec_t, t_idx);
                }
            }
        }
    }

    #[test]
    fn test_msm_state_registry() {
        let mut registry = MsmStateRegistry::new(10, 10);

        // Add states
        let s1 = MsmCompositeState::new(0, 0, 0, 1.0, 2.0);
        let id1 = registry.get_or_create(s1);

        let s2 = MsmCompositeState::new(0, 1, 0, 1.5, 2.0);
        let id2 = registry.get_or_create(s2);

        assert_ne!(id1, id2);
        assert_eq!(registry.len(), 2);

        // Same key should return same ID
        let s3 = MsmCompositeState::new(0, 0, 0, 1.0, 2.0);
        let id3 = registry.get_or_create(s3);
        assert_eq!(id1, id3);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_msm_composite_state() {
        let state = MsmCompositeState::initial(5, 1.0, 2.0);

        assert_eq!(state.series_id, 5);
        assert_eq!(state.query_index, 0);
        assert_eq!(state.target_index, 0);
        assert!(!state.is_final(3, 4));

        let final_state = MsmCompositeState::new(5, 3, 4, 1.0, 2.0);
        assert!(final_state.is_final(3, 4));
    }

    #[test]
    fn test_msm_state_source() {
        let source = MsmStateSource::new(
            vec![1.0, 2.0, 3.0],
            MsmConfig::new(1.0),
            10.0,
            vec![(0, vec![1.0, 2.0, 3.0])],
        );

        let (min, max) = source.value_range();
        assert!(approx_eq(min, 1.0));
        assert!(approx_eq(max, 3.0));

        let hint = source.num_states_hint();
        assert!(hint.is_some());
        assert!(hint.unwrap() > 0);
    }
}
