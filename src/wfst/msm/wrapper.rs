//! MSM WFST wrapper for time series similarity search.
//!
//! This module provides [`MsmWfst`], a lazy WFST wrapper that exposes the MSM
//! (Move-Split-Merge) time series metric as a weighted transducer compatible
//! with the lling-llang WFST framework.
//!
//! # Architecture
//!
//! The MSM WFST represents the product of:
//! - A query time series (the input to be matched)
//! - A time series index/dictionary (the candidates)
//! - The MSM automaton (computing data-dependent costs)
//!
//! Unlike string Levenshtein WFSTs, MSM has **data-dependent transition costs**
//! that depend on the actual values being compared, not just position indices.
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::wfst::msm::{MsmWfst, MsmWfstBuilder};
//! use liblevenshtein::time_series::{TimeSeriesIndex, QuantizationConfig, MsmConfig};
//!
//! // Build a time series index
//! let config = QuantizationConfig::for_u8(0.0, 100.0);
//! let index = TimeSeriesIndex::from_series(config, &[
//!     vec![10.0, 20.0, 30.0],
//!     vec![15.0, 25.0, 35.0],
//! ]);
//!
//! // Create MSM WFST for query
//! let query = vec![12.0, 22.0, 32.0];
//! let msm_config = MsmConfig::new(1.0);
//! let wfst = MsmWfstBuilder::new()
//!     .query(&query)
//!     .msm_config(msm_config)
//!     .max_cost(10.0)
//!     .build();
//! ```

use lling_llang::prelude::{
    LazyState, LazyWfst, Semiring, StateId, StateSource, TropicalWeight, Wfst,
    WeightedTransition,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::state::{MsmCompositeState, MsmStateKey, MsmStateRegistry};
use super::weight::MsmWeight;
use crate::time_series::MsmConfig;

/// Cached state information for the MSM WFST.
#[derive(Clone)]
struct CachedMsmState {
    /// Whether this is a final state.
    is_final: bool,
    /// Final weight if final.
    final_weight: TropicalWeight,
    /// Outgoing transitions.
    transitions: SmallVec<[WeightedTransition<u8, TropicalWeight>; 8]>,
}

/// MSM WFST for time series similarity search.
///
/// This WFST computes MSM distances between a query time series and
/// candidate series stored in an index.
///
/// # Type Parameters
///
/// The WFST uses `u8` labels for quantized time series values and
/// `TropicalWeight` for compatibility with lling-llang composition.
/// The actual MSM weight tracking is done internally.
#[derive(Clone)]
pub struct MsmWfst {
    /// The query time series.
    query: Vec<f64>,

    /// MSM configuration (c constant).
    msm_config: MsmConfig,

    /// Maximum cost threshold.
    max_cost: f64,

    /// State registry for composite states.
    state_registry: MsmStateRegistry,

    /// Cached state computations.
    cache: FxHashMap<StateId, CachedMsmState>,

    /// Map from state key to MSM weight tracking.
    state_weights: FxHashMap<MsmStateKey, MsmWeight>,

    /// Cache policy.
    cache_policy: lling_llang::wfst::CachePolicy,

    /// Maximum cache size.
    max_cache_size: usize,

    /// Target series data (for computing transitions).
    /// Each entry is (series_id, series_data).
    target_series: Vec<(u32, Vec<f64>)>,
}

/// Default maximum cache size (50,000 states).
const DEFAULT_MAX_CACHE_SIZE: usize = 50_000;

impl MsmWfst {
    /// Create a new MSM WFST.
    ///
    /// # Arguments
    ///
    /// * `query` - The query time series
    /// * `msm_config` - MSM configuration
    /// * `max_cost` - Maximum cost threshold
    /// * `target_series` - Vector of (series_id, series_data) pairs
    pub fn new(
        query: Vec<f64>,
        msm_config: MsmConfig,
        max_cost: f64,
        target_series: Vec<(u32, Vec<f64>)>,
    ) -> Self {
        let max_target_len = target_series
            .iter()
            .map(|(_, s)| s.len())
            .max()
            .unwrap_or(0) as u32;

        let mut wfst = Self {
            query: query.clone(),
            msm_config,
            max_cost,
            state_registry: MsmStateRegistry::new(query.len() as u32, max_target_len),
            cache: FxHashMap::default(),
            state_weights: FxHashMap::default(),
            cache_policy: lling_llang::wfst::CachePolicy::CacheAll,
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
            target_series,
        };

        // Register initial states for each target series
        wfst.initialize_states();
        wfst
    }

    /// Initialize start states for all target series.
    fn initialize_states(&mut self) {
        if self.query.is_empty() {
            return;
        }

        let first_query_value = self.query[0];
        let query_len = self.query.len();

        // Collect series info upfront to avoid borrow conflicts
        let series_info: Vec<_> = self
            .target_series
            .iter()
            .filter(|(_, target)| !target.is_empty())
            .map(|(series_id, target)| {
                (*series_id, target[0], target.len())
            })
            .collect();

        for (series_id, first_target_value, target_len) in series_info {
            let state = MsmCompositeState::initial(series_id, first_query_value, first_target_value);
            let id = self.state_registry.get_or_create(state);

            // Initialize MSM weight for this state
            let weight = MsmWeight::initial(first_query_value, first_target_value, self.msm_config.c);
            self.state_weights.insert(state.key(), weight);

            // Cache the initial state
            let initial_cost = (first_query_value - first_target_value).abs();
            if initial_cost <= self.max_cost {
                let transitions = self.compute_transitions_for_state(id, &state);
                self.cache.insert(
                    id,
                    CachedMsmState {
                        is_final: state.is_final(query_len, target_len),
                        final_weight: if state.is_final(query_len, target_len) {
                            TropicalWeight::new(initial_cost)
                        } else {
                            TropicalWeight::infinity()
                        },
                        transitions,
                    },
                );
            }
        }
    }

    /// Compute transitions for a state.
    fn compute_transitions_for_state(
        &mut self,
        _state_id: StateId,
        state: &MsmCompositeState,
    ) -> SmallVec<[WeightedTransition<u8, TropicalWeight>; 8]> {
        let mut transitions = SmallVec::new();

        let target = self
            .target_series
            .iter()
            .find(|(id, _)| *id == state.series_id)
            .map(|(_, s)| s);

        let target = match target {
            Some(t) => t,
            None => return transitions,
        };

        let query_len = self.query.len();
        let target_len = target.len();

        // Get current MSM weight
        let current_weight = self
            .state_weights
            .get(&state.key())
            .copied()
            .unwrap_or_else(|| {
                MsmWeight::initial(
                    state.last_query_value,
                    state.last_target_value,
                    self.msm_config.c,
                )
            });

        // Get values for transitions
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

        // Move transition: consume one from both
        if let (Some(qv), Some(tv)) = (query_value, target_value) {
            let new_weight = current_weight.with_move(qv, tv);
            if new_weight.cost() <= self.max_cost {
                let new_state = MsmCompositeState::new(
                    state.series_id,
                    state.query_index + 1,
                    state.target_index + 1,
                    qv,
                    tv,
                );
                let new_id = self.state_registry.get_or_create(new_state);
                self.state_weights.insert(new_state.key(), new_weight);

                // Use quantized value as label (0-255 range)
                let label = quantize_value(tv, 0.0, 100.0);
                transitions.push(WeightedTransition::new(
                    _state_id,
                    Some(label),
                    Some(label),
                    new_id,
                    TropicalWeight::new((qv - tv).abs()),
                ));
            }
        }

        // Merge-like transition: consume one from query only
        if let Some(qv) = query_value {
            let new_weight = current_weight.with_merge(qv);
            if new_weight.cost() <= self.max_cost {
                let new_state = MsmCompositeState::new(
                    state.series_id,
                    state.query_index + 1,
                    state.target_index,
                    qv,
                    state.last_target_value,
                );
                let new_id = self.state_registry.get_or_create(new_state);
                self.state_weights.insert(new_state.key(), new_weight);

                // Epsilon transition for merge (no output consumed)
                let cost = self.msm_config.c_func(qv, current_weight.last_query_value(), current_weight.last_target_value());
                transitions.push(WeightedTransition::new(
                    _state_id,
                    None,  // Epsilon on input
                    None,  // Epsilon on output
                    new_id,
                    TropicalWeight::new(cost),
                ));
            }
        }

        // Split-like transition: consume one from target only
        if let Some(tv) = target_value {
            let new_weight = current_weight.with_split(tv);
            if new_weight.cost() <= self.max_cost {
                let new_state = MsmCompositeState::new(
                    state.series_id,
                    state.query_index,
                    state.target_index + 1,
                    state.last_query_value,
                    tv,
                );
                let new_id = self.state_registry.get_or_create(new_state);
                self.state_weights.insert(new_state.key(), new_weight);

                // Epsilon on input (only target consumed)
                let label = quantize_value(tv, 0.0, 100.0);
                let cost = self.msm_config.c_func(tv, current_weight.last_query_value(), current_weight.last_target_value());
                transitions.push(WeightedTransition::new(
                    _state_id,
                    None,        // Epsilon on input
                    Some(label), // Output the target value
                    new_id,
                    TropicalWeight::new(cost),
                ));
            }
        }

        transitions
    }

    /// Ensure a state is computed and cached.
    fn ensure_state(&mut self, state_id: StateId) {
        if self.cache.contains_key(&state_id) {
            return;
        }

        let state = match self.state_registry.get(state_id) {
            Some(s) => *s,
            None => return,
        };

        let target = self
            .target_series
            .iter()
            .find(|(id, _)| *id == state.series_id)
            .map(|(_, s)| s.clone());

        let target = match target {
            Some(t) => t,
            None => return,
        };

        let is_final = state.is_final(self.query.len(), target.len());
        let final_weight = if is_final {
            let weight = self
                .state_weights
                .get(&state.key())
                .map(|w| w.cost())
                .unwrap_or(f64::INFINITY);
            TropicalWeight::new(weight)
        } else {
            TropicalWeight::infinity()
        };

        let transitions = self.compute_transitions_for_state(state_id, &state);

        // Apply cache eviction if needed
        if let lling_llang::wfst::CachePolicy::Lru { max_states } = self.cache_policy {
            let limit = if max_states > 0 { max_states } else { self.max_cache_size };
            if self.cache.len() >= limit {
                let to_remove = (self.cache.len() / 10).max(1);
                let keys: Vec<_> = self.cache.keys().take(to_remove).copied().collect();
                for key in keys {
                    self.cache.remove(&key);
                }
            }
        }

        self.cache.insert(
            state_id,
            CachedMsmState {
                is_final,
                final_weight,
                transitions,
            },
        );
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

    /// Set the maximum cache size.
    pub fn set_max_cache_size(&mut self, size: usize) {
        self.max_cache_size = size;
    }
}

impl Wfst<u8, TropicalWeight> for MsmWfst {
    fn start(&self) -> StateId {
        0 // The first registered state is always the start
    }

    fn is_final(&self, state: StateId) -> bool {
        self.cache
            .get(&state)
            .map(|s| s.is_final)
            .unwrap_or(false)
    }

    fn final_weight(&self, state: StateId) -> TropicalWeight {
        self.cache
            .get(&state)
            .map(|s| s.final_weight)
            .unwrap_or_else(TropicalWeight::infinity)
    }

    fn transitions(&self, state: StateId) -> &[WeightedTransition<u8, TropicalWeight>] {
        static EMPTY: &[WeightedTransition<u8, TropicalWeight>] = &[];
        self.cache
            .get(&state)
            .map(|s| s.transitions.as_slice())
            .unwrap_or(EMPTY)
    }

    fn num_states(&self) -> usize {
        self.state_registry.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.query.is_empty() || self.target_series.is_empty()
    }

    #[inline]
    fn is_valid_state(&self, state: StateId) -> bool {
        self.state_registry.get(state).is_some()
    }
}

impl LazyWfst<u8, TropicalWeight> for MsmWfst {
    fn is_expanded(&self, state: StateId) -> bool {
        self.cache.contains_key(&state)
    }

    fn expand(&mut self, state: StateId) {
        self.ensure_state(state);
    }

    fn transitions_lazy(&mut self, state: StateId) -> &[WeightedTransition<u8, TropicalWeight>] {
        self.ensure_state(state);
        self.transitions(state)
    }

    fn cache_policy(&self) -> lling_llang::wfst::CachePolicy {
        self.cache_policy
    }

    fn set_cache_policy(&mut self, policy: lling_llang::wfst::CachePolicy) {
        self.cache_policy = policy;
    }

    fn computed_states(&self) -> usize {
        self.cache.len()
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl StateSource<u8, TropicalWeight> for MsmWfst {
    fn compute_state(&self, _state: StateId) -> LazyState<u8, TropicalWeight> {
        // For StateSource, we need a non-mutable version
        // Return Pending since we need mutable access
        LazyState::Pending
    }

    fn start(&self) -> StateId {
        0
    }

    fn num_states_hint(&self) -> Option<usize> {
        Some(self.state_registry.len() * 3)
    }
}

/// Quantize a float value to u8 range.
#[inline]
fn quantize_value(value: f64, min_val: f64, max_val: f64) -> u8 {
    let normalized = (value - min_val) / (max_val - min_val);
    (normalized.clamp(0.0, 1.0) * 255.0) as u8
}

/// Builder for MSM WFST.
pub struct MsmWfstBuilder {
    query: Option<Vec<f64>>,
    msm_config: MsmConfig,
    max_cost: f64,
    target_series: Vec<(u32, Vec<f64>)>,
}

impl MsmWfstBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            query: None,
            msm_config: MsmConfig::default(),
            max_cost: f64::INFINITY,
            target_series: Vec::new(),
        }
    }

    /// Set the query time series.
    pub fn query(mut self, query: &[f64]) -> Self {
        self.query = Some(query.to_vec());
        self
    }

    /// Set the MSM configuration.
    pub fn msm_config(mut self, config: MsmConfig) -> Self {
        self.msm_config = config;
        self
    }

    /// Set the maximum cost threshold.
    pub fn max_cost(mut self, cost: f64) -> Self {
        self.max_cost = cost;
        self
    }

    /// Add a target series.
    pub fn add_target(mut self, series_id: u32, series: &[f64]) -> Self {
        self.target_series.push((series_id, series.to_vec()));
        self
    }

    /// Add multiple target series.
    pub fn add_targets(mut self, series: impl IntoIterator<Item = (u32, Vec<f64>)>) -> Self {
        self.target_series.extend(series);
        self
    }

    /// Build the MSM WFST.
    pub fn build(self) -> Result<MsmWfst, String> {
        let query = self.query.ok_or_else(|| "Query not set".to_string())?;
        if self.target_series.is_empty() {
            return Err("No target series added".to_string());
        }
        Ok(MsmWfst::new(
            query,
            self.msm_config,
            self.max_cost,
            self.target_series,
        ))
    }
}

impl Default for MsmWfstBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_quantize_value() {
        assert_eq!(quantize_value(0.0, 0.0, 100.0), 0);
        assert_eq!(quantize_value(100.0, 0.0, 100.0), 255);
        assert_eq!(quantize_value(50.0, 0.0, 100.0), 127); // ~128
    }

    #[test]
    fn test_builder_creation() {
        let builder = MsmWfstBuilder::new();
        assert!(builder.query.is_none());
    }

    #[test]
    fn test_builder_with_query() {
        let builder = MsmWfstBuilder::new().query(&[1.0, 2.0, 3.0]);
        assert!(builder.query.is_some());
        assert_eq!(builder.query.unwrap().len(), 3);
    }

    #[test]
    fn test_builder_build_no_query() {
        let builder = MsmWfstBuilder::new().add_target(0, &[1.0, 2.0]);
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_build_no_targets() {
        let builder = MsmWfstBuilder::new().query(&[1.0, 2.0]);
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_build_success() {
        let result = MsmWfstBuilder::new()
            .query(&[1.0, 2.0, 3.0])
            .add_target(0, &[1.0, 2.0, 3.0])
            .build();

        assert!(result.is_ok());
        let wfst = result.unwrap();
        assert_eq!(wfst.query().len(), 3);
    }

    #[test]
    fn test_wfst_basic() {
        let wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0, 3.0])
            .msm_config(MsmConfig::new(1.0))
            .max_cost(10.0)
            .add_target(0, &[1.0, 2.0, 3.0])
            .build()
            .expect("build failed");

        assert!(!wfst.is_empty());
        assert!(wfst.num_states() >= 1);
    }

    #[test]
    fn test_wfst_start_state() {
        let wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0])
            .add_target(0, &[1.0, 2.0])
            .build()
            .expect("build failed");

        let start = Wfst::start(&wfst);
        assert!(wfst.is_valid_state(start));
    }

    #[test]
    fn test_wfst_lazy_expansion() {
        let mut wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0, 3.0])
            .msm_config(MsmConfig::new(1.0))
            .max_cost(10.0)
            .add_target(0, &[1.0, 2.0, 3.0])
            .build()
            .expect("build failed");

        let start = Wfst::start(&wfst);

        // Before expansion
        let initial_computed = wfst.computed_states();

        // Expand
        wfst.expand(start);

        // After expansion, should have cached the state
        assert!(wfst.is_expanded(start));
    }

    #[test]
    fn test_wfst_multiple_targets() {
        let wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0, 3.0])
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
    fn test_wfst_cache_policy() {
        let mut wfst = MsmWfstBuilder::new()
            .query(&[1.0])
            .add_target(0, &[1.0])
            .build()
            .expect("build failed");

        assert!(matches!(
            wfst.cache_policy(),
            lling_llang::wfst::CachePolicy::CacheAll
        ));

        wfst.set_cache_policy(lling_llang::wfst::CachePolicy::Lru { max_states: 100 });
        assert!(matches!(
            wfst.cache_policy(),
            lling_llang::wfst::CachePolicy::Lru { .. }
        ));
    }

    #[test]
    fn test_wfst_clear_cache() {
        let mut wfst = MsmWfstBuilder::new()
            .query(&[1.0, 2.0])
            .add_target(0, &[1.0, 2.0])
            .build()
            .expect("build failed");

        wfst.expand(0);
        assert!(wfst.computed_states() > 0);

        wfst.clear_cache();
        assert_eq!(wfst.computed_states(), 0);
    }
}
