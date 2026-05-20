//! State source for lazy MSM WFST computation.
//!
//! This module provides [`MsmStateSource`], a [`StateSource`] implementation
//! that enables lazy state computation for MSM WFSTs. This is useful for
//! composition with other WFSTs where only accessed states need to be computed.
//!
//! # Architecture
//!
//! The state source maintains:
//! - A reference to the query time series
//! - A reference to the target series index
//! - MSM configuration
//! - State registry for deduplication
//!
//! States are computed on-demand using the MSM transition rules.

use lling_llang::prelude::{LazyState, StateId, StateSource, TropicalWeight, WeightedTransition};
use smallvec::SmallVec;

use super::state::MsmCompositeState;
use super::weight::MsmWeight;
use crate::time_series::MsmConfig;

/// State source for lazy MSM WFST state computation.
///
/// This enables demand-driven computation of MSM WFST states, computing
/// only the states that are actually accessed during traversal.
#[derive(Clone)]
pub struct MsmStateSource {
    /// The query time series.
    query: Vec<f64>,

    /// MSM configuration.
    msm_config: MsmConfig,

    /// Maximum cost threshold.
    max_cost: f64,

    /// Target series data.
    target_series: Vec<(u32, Vec<f64>)>,

    /// Value range for quantization.
    value_range: (f64, f64),
}

impl MsmStateSource {
    /// Create a new MSM state source.
    pub fn new(
        query: Vec<f64>,
        msm_config: MsmConfig,
        max_cost: f64,
        target_series: Vec<(u32, Vec<f64>)>,
    ) -> Self {
        // Compute value range from all series
        let all_values = query.iter().chain(
            target_series.iter().flat_map(|(_, s)| s.iter())
        );
        let min_val = all_values.clone().cloned().fold(f64::INFINITY, f64::min);
        let max_val = all_values.cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            query,
            msm_config,
            max_cost,
            target_series,
            value_range: (min_val, max_val),
        }
    }

    /// Get the query series.
    #[inline]
    pub fn query(&self) -> &[f64] {
        &self.query
    }

    /// Get the MSM configuration.
    #[inline]
    pub fn msm_config(&self) -> &MsmConfig {
        &self.msm_config
    }

    /// Get the maximum cost threshold.
    #[inline]
    pub fn max_cost(&self) -> f64 {
        self.max_cost
    }

    /// Get the value range.
    #[inline]
    pub fn value_range(&self) -> (f64, f64) {
        self.value_range
    }

    /// Compute a state given its composite representation.
    pub fn compute_composite_state(
        &self,
        state: &MsmCompositeState,
        weight: &MsmWeight,
    ) -> LazyState<u8, TropicalWeight> {
        let target = self
            .target_series
            .iter()
            .find(|(id, _)| *id == state.series_id)
            .map(|(_, s)| s);

        let target = match target {
            Some(t) => t,
            None => return LazyState::Computed {
                is_final: false,
                final_weight: TropicalWeight::infinity(),
                transitions: SmallVec::new(),
            },
        };

        let query_len = self.query.len();
        let target_len = target.len();

        let is_final = state.is_final(query_len, target_len);
        let final_weight = if is_final {
            TropicalWeight::new(weight.cost())
        } else {
            TropicalWeight::infinity()
        };

        let transitions = self.compute_transitions(state, weight, target);

        LazyState::Computed {
            is_final,
            final_weight,
            transitions,
        }
    }

    /// Compute transitions for a state.
    fn compute_transitions(
        &self,
        state: &MsmCompositeState,
        weight: &MsmWeight,
        target: &[f64],
    ) -> SmallVec<[WeightedTransition<u8, TropicalWeight>; 4]> {
        let mut transitions = SmallVec::new();

        let query_len = self.query.len();
        let target_len = target.len();

        // Get current values
        let query_value = if (state.query_index as usize) < query_len {
            Some(self.query[state.query_index as usize])
        } else {
            None
        };

        let target_value = if (state.target_index as usize) < target_len {
            Some(target[state.target_index as usize])
        } else {
            None
        };

        // Move transition
        if let (Some(qv), Some(tv)) = (query_value, target_value) {
            let new_weight = weight.with_move(qv, tv);
            if new_weight.cost() <= self.max_cost {
                let label = self.quantize(tv);
                // Note: target state ID will be resolved by the wrapper
                transitions.push(WeightedTransition::new(
                    0, // Placeholder, resolved by caller
                    Some(label),
                    Some(label),
                    0, // Placeholder
                    TropicalWeight::new((qv - tv).abs()),
                ));
            }
        }

        // Merge transition
        if let Some(qv) = query_value {
            let new_weight = weight.with_merge(qv);
            if new_weight.cost() <= self.max_cost {
                let cost = self.msm_config.c_func(
                    qv,
                    weight.last_query_value(),
                    weight.last_target_value(),
                );
                transitions.push(WeightedTransition::new(
                    0,
                    None,
                    None,
                    0,
                    TropicalWeight::new(cost),
                ));
            }
        }

        // Split transition
        if let Some(tv) = target_value {
            let new_weight = weight.with_split(tv);
            if new_weight.cost() <= self.max_cost {
                let label = self.quantize(tv);
                let cost = self.msm_config.c_func(
                    tv,
                    weight.last_query_value(),
                    weight.last_target_value(),
                );
                transitions.push(WeightedTransition::new(
                    0,
                    None,
                    Some(label),
                    0,
                    TropicalWeight::new(cost),
                ));
            }
        }

        transitions
    }

    /// Quantize a value to u8.
    #[inline]
    fn quantize(&self, value: f64) -> u8 {
        let (min_val, max_val) = self.value_range;
        let range = max_val - min_val;
        if range < 1e-9 {
            return 127; // Mid-point if range is zero
        }
        let normalized = (value - min_val) / range;
        (normalized.clamp(0.0, 1.0) * 255.0) as u8
    }
}

impl StateSource<u8, TropicalWeight> for MsmStateSource {
    fn compute_state(&self, _state: StateId) -> LazyState<u8, TropicalWeight> {
        // This requires mapping StateId back to composite state, which needs
        // the registry. For a pure StateSource implementation, we return Pending
        // and let the wrapper handle the actual computation.
        LazyState::Pending
    }

    fn start(&self) -> StateId {
        0
    }

    fn num_states_hint(&self) -> Option<usize> {
        let query_len = self.query.len();
        let max_target_len = self.target_series
            .iter()
            .map(|(_, s)| s.len())
            .max()
            .unwrap_or(0);
        let num_series = self.target_series.len();

        // Estimate: positions * series * ~0.5 for pruning
        Some((query_len + 1) * (max_target_len + 1) * num_series / 2)
    }
}

/// Builder for MSM state source.
pub struct MsmStateSourceBuilder {
    query: Option<Vec<f64>>,
    msm_config: MsmConfig,
    max_cost: f64,
    target_series: Vec<(u32, Vec<f64>)>,
}

impl MsmStateSourceBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            query: None,
            msm_config: MsmConfig::default(),
            max_cost: f64::INFINITY,
            target_series: Vec::new(),
        }
    }

    /// Set the query.
    pub fn query(mut self, query: &[f64]) -> Self {
        self.query = Some(query.to_vec());
        self
    }

    /// Set the MSM configuration.
    pub fn msm_config(mut self, config: MsmConfig) -> Self {
        self.msm_config = config;
        self
    }

    /// Set the maximum cost.
    pub fn max_cost(mut self, cost: f64) -> Self {
        self.max_cost = cost;
        self
    }

    /// Add a target series.
    pub fn add_target(mut self, id: u32, series: &[f64]) -> Self {
        self.target_series.push((id, series.to_vec()));
        self
    }

    /// Build the state source.
    pub fn build(self) -> Result<MsmStateSource, String> {
        let query = self.query.ok_or_else(|| "Query not set".to_string())?;
        if self.target_series.is_empty() {
            return Err("No target series".to_string());
        }
        Ok(MsmStateSource::new(
            query,
            self.msm_config,
            self.max_cost,
            self.target_series,
        ))
    }
}

impl Default for MsmStateSourceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_source_creation() {
        let source = MsmStateSource::new(
            vec![1.0, 2.0, 3.0],
            MsmConfig::new(1.0),
            10.0,
            vec![(0, vec![1.0, 2.0, 3.0])],
        );

        assert_eq!(source.query().len(), 3);
        assert_eq!(source.max_cost(), 10.0);
    }

    #[test]
    fn test_state_source_value_range() {
        let source = MsmStateSource::new(
            vec![0.0, 50.0],
            MsmConfig::new(1.0),
            10.0,
            vec![(0, vec![25.0, 100.0])],
        );

        let (min, max) = source.value_range();
        assert!((min - 0.0).abs() < 1e-9);
        assert!((max - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_state_source_quantize() {
        let source = MsmStateSource::new(
            vec![0.0, 100.0],
            MsmConfig::new(1.0),
            10.0,
            vec![(0, vec![50.0])],
        );

        assert_eq!(source.quantize(0.0), 0);
        assert_eq!(source.quantize(100.0), 255);
        assert_eq!(source.quantize(50.0), 127);
    }

    #[test]
    fn test_state_source_start() {
        let source = MsmStateSource::new(
            vec![1.0],
            MsmConfig::new(1.0),
            10.0,
            vec![(0, vec![1.0])],
        );

        assert_eq!(StateSource::<u8, TropicalWeight>::start(&source), 0);
    }

    #[test]
    fn test_state_source_hint() {
        let source = MsmStateSource::new(
            vec![1.0, 2.0, 3.0],  // len 3
            MsmConfig::new(1.0),
            10.0,
            vec![
                (0, vec![1.0, 2.0, 3.0, 4.0]),  // len 4
                (1, vec![1.0, 2.0]),            // len 2
            ],
        );

        let hint = source.num_states_hint();
        assert!(hint.is_some());
        // (3+1) * (4+1) * 2 / 2 = 20
        assert!(hint.expect("expected Some hint in test") > 0);
    }

    #[test]
    fn test_builder() {
        let result = MsmStateSourceBuilder::new()
            .query(&[1.0, 2.0])
            .msm_config(MsmConfig::new(0.5))
            .max_cost(5.0)
            .add_target(0, &[1.0, 2.0])
            .build();

        assert!(result.is_ok());
        let source = result.expect("test fixture: build must be Ok");
        assert_eq!(source.query().len(), 2);
        assert_eq!(source.msm_config().c, 0.5);
        assert_eq!(source.max_cost(), 5.0);
    }

    #[test]
    fn test_builder_no_query() {
        let result = MsmStateSourceBuilder::new()
            .add_target(0, &[1.0])
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_no_targets() {
        let result = MsmStateSourceBuilder::new()
            .query(&[1.0])
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_compute_composite_state() {
        let source = MsmStateSource::new(
            vec![1.0, 2.0, 3.0],
            MsmConfig::new(1.0),
            100.0,
            vec![(0, vec![1.0, 2.0, 3.0])],
        );

        let state = MsmCompositeState::initial(0, 1.0, 1.0);
        let weight = MsmWeight::initial(1.0, 1.0, 1.0);

        let lazy = source.compute_composite_state(&state, &weight);

        match lazy {
            LazyState::Computed { is_final, transitions, .. } => {
                assert!(!is_final); // Not at end yet
                assert!(!transitions.is_empty()); // Should have transitions
            }
            LazyState::Pending => panic!("Expected Computed state"),
        }
    }

    #[test]
    fn test_compute_final_state() {
        let source = MsmStateSource::new(
            vec![1.0, 2.0],
            MsmConfig::new(1.0),
            100.0,
            vec![(0, vec![1.0, 2.0])],
        );

        // Final state: consumed all of both series
        let state = MsmCompositeState::new(0, 2, 2, 2.0, 2.0);
        let weight = MsmWeight::new(1.5, 2.0, 2.0, 1.0);

        let lazy = source.compute_composite_state(&state, &weight);

        match lazy {
            LazyState::Computed { is_final, final_weight, .. } => {
                assert!(is_final);
                assert!((final_weight.value() - 1.5).abs() < 1e-9);
            }
            LazyState::Pending => panic!("Expected Computed state"),
        }
    }
}
