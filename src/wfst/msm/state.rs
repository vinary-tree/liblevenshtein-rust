//! State encoding utilities for MSM WFST.
//!
//! Unlike the string Levenshtein WFST which encodes (dictionary_node, automaton_state)
//! pairs, the MSM WFST encodes (series_index, query_index, target_index) tuples plus
//! value tracking for the C() function.
//!
//! # Encoding Scheme
//!
//! States are encoded as composite IDs:
//! ```text
//! state_id = series_id * (max_query * max_target) + query_index * max_target + target_index
//! ```
//!
//! Additional value tracking is maintained in a separate registry since the C()
//! function's data-dependent costs require remembering the last values.

use lling_llang::wfst::StateId;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Key for MSM state: (series_id, query_index, target_index).
pub type MsmStateKey = (u32, u32, u32);

/// Composite state with value tracking for MSM WFST.
#[derive(Clone, Copy, Debug)]
pub struct MsmCompositeState {
    /// Index of the target series in the index/dictionary.
    pub series_id: u32,

    /// Index in the query series (how many query elements consumed).
    pub query_index: u32,

    /// Index in the target series (how many target elements matched).
    pub target_index: u32,

    /// Last query value consumed (for C() function).
    pub last_query_value: f64,

    /// Last target value consumed (for C() function).
    pub last_target_value: f64,
}

impl MsmCompositeState {
    /// Create a new composite state.
    #[inline]
    pub fn new(
        series_id: u32,
        query_index: u32,
        target_index: u32,
        last_query_value: f64,
        last_target_value: f64,
    ) -> Self {
        Self {
            series_id,
            query_index,
            target_index,
            last_query_value,
            last_target_value,
        }
    }

    /// Create an initial state for the given series.
    #[inline]
    pub fn initial(series_id: u32, first_query_value: f64, first_target_value: f64) -> Self {
        Self {
            series_id,
            query_index: 0,
            target_index: 0,
            last_query_value: first_query_value,
            last_target_value: first_target_value,
        }
    }

    /// Get the key tuple for this state.
    #[inline]
    pub fn key(&self) -> MsmStateKey {
        (self.series_id, self.query_index, self.target_index)
    }

    /// Check if this state has reached the final position for the given lengths.
    #[inline]
    pub fn is_final(&self, query_length: usize, target_length: usize) -> bool {
        self.query_index >= query_length as u32 && self.target_index >= target_length as u32
    }
}

/// Encode a composite MSM state into a single StateId.
///
/// # Arguments
///
/// * `series_id` - Index of the target series
/// * `query_index` - Position in query series
/// * `target_index` - Position in target series
/// * `max_query_len` - Maximum query length (for encoding)
/// * `max_target_len` - Maximum target length (for encoding)
///
/// # Returns
///
/// Encoded state ID.
#[inline]
pub fn encode_msm_state(
    series_id: u32,
    query_index: u32,
    target_index: u32,
    max_query_len: u32,
    max_target_len: u32,
) -> StateId {
    let query_target_space = (max_query_len + 1) * (max_target_len + 1);
    series_id * query_target_space + query_index * (max_target_len + 1) + target_index
}

/// Decode a StateId back into (series_id, query_index, target_index).
///
/// # Arguments
///
/// * `state_id` - The encoded state ID
/// * `max_query_len` - Maximum query length used in encoding
/// * `max_target_len` - Maximum target length used in encoding
///
/// # Returns
///
/// Tuple of (series_id, query_index, target_index).
#[inline]
pub fn decode_msm_state(state_id: StateId, max_query_len: u32, max_target_len: u32) -> MsmStateKey {
    let query_target_space = (max_query_len + 1) * (max_target_len + 1);
    let target_space = max_target_len + 1;

    let series_id = state_id / query_target_space;
    let remainder = state_id % query_target_space;
    let query_index = remainder / target_space;
    let target_index = remainder % target_space;

    (series_id, query_index, target_index)
}

/// Registry for MSM states with value tracking.
///
/// Since the C() function requires tracking last values, we need to maintain
/// a registry that maps StateId to the full composite state including values.
#[derive(Clone)]
pub struct MsmStateRegistry {
    /// Map from StateId to composite state.
    states: FxHashMap<StateId, MsmCompositeState>,

    /// Map from state key to StateId (for deduplication).
    key_to_id: FxHashMap<MsmStateKey, StateId>,

    /// Next available StateId.
    next_id: StateId,

    /// Maximum query length.
    max_query_len: u32,

    /// Maximum target length.
    max_target_len: u32,
}

impl MsmStateRegistry {
    /// Create a new registry with the given size bounds.
    pub fn new(max_query_len: u32, max_target_len: u32) -> Self {
        Self {
            states: FxHashMap::default(),
            key_to_id: FxHashMap::default(),
            next_id: 0,
            max_query_len,
            max_target_len,
        }
    }

    /// Create a registry with estimated capacity.
    pub fn with_capacity(max_query_len: u32, max_target_len: u32, capacity: usize) -> Self {
        Self {
            states: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            key_to_id: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            next_id: 0,
            max_query_len,
            max_target_len,
        }
    }

    /// Get or create a StateId for the given composite state.
    ///
    /// If a state with the same key already exists, returns its ID.
    /// Otherwise, creates a new state and returns the new ID.
    pub fn get_or_create(&mut self, state: MsmCompositeState) -> StateId {
        let key = state.key();

        if let Some(&id) = self.key_to_id.get(&key) {
            // State exists, but we may need to update values if they differ
            // (for data-dependent costs, we keep the better one)
            return id;
        }

        // Create new state
        let id = self.next_id;
        self.next_id += 1;
        self.states.insert(id, state);
        self.key_to_id.insert(key, id);
        id
    }

    /// Get the composite state for a StateId.
    #[inline]
    pub fn get(&self, id: StateId) -> Option<&MsmCompositeState> {
        self.states.get(&id)
    }

    /// Get the StateId for a state key.
    #[inline]
    pub fn get_id(&self, key: &MsmStateKey) -> Option<StateId> {
        self.key_to_id.get(key).copied()
    }

    /// Get the number of registered states.
    #[inline]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if the registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Clear all states from the registry.
    pub fn clear(&mut self) {
        self.states.clear();
        self.key_to_id.clear();
        self.next_id = 0;
    }

    /// Get the maximum query length.
    #[inline]
    pub fn max_query_len(&self) -> u32 {
        self.max_query_len
    }

    /// Get the maximum target length.
    #[inline]
    pub fn max_target_len(&self) -> u32 {
        self.max_target_len
    }
}

/// Estimate the number of states for an MSM WFST.
///
/// The product of series × query positions × target positions gives an
/// upper bound, but subsumption and cost thresholds reduce this significantly.
#[inline]
pub fn estimate_msm_states(num_series: usize, query_len: usize, max_target_len: usize) -> usize {
    // Conservative estimate: not all cells will be explored
    // due to cost thresholds and early termination
    let cells_per_series = (query_len + 1) * (max_target_len + 1);
    num_series * cells_per_series / 2 // Assume ~50% pruning
}

/// MSM transition types for state expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsmTransition {
    /// Move operation: consume one from both series.
    Move,
    /// Merge-like: consume one from query only.
    Merge,
    /// Split-like: consume one from target only.
    Split,
}

impl MsmTransition {
    /// Get all transition types.
    #[inline]
    pub fn all() -> SmallVec<[MsmTransition; 3]> {
        smallvec::smallvec![
            MsmTransition::Move,
            MsmTransition::Merge,
            MsmTransition::Split
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let max_q = 10;
        let max_t = 15;

        for series_id in 0..5 {
            for query_idx in 0..=max_q {
                for target_idx in 0..=max_t {
                    let encoded = encode_msm_state(series_id, query_idx, target_idx, max_q, max_t);
                    let (dec_s, dec_q, dec_t) = decode_msm_state(encoded, max_q, max_t);

                    assert_eq!(dec_s, series_id, "series_id mismatch");
                    assert_eq!(dec_q, query_idx, "query_index mismatch");
                    assert_eq!(dec_t, target_idx, "target_index mismatch");
                }
            }
        }
    }

    #[test]
    fn test_composite_state_creation() {
        let state = MsmCompositeState::new(1, 2, 3, 1.5, 2.5);
        assert_eq!(state.series_id, 1);
        assert_eq!(state.query_index, 2);
        assert_eq!(state.target_index, 3);
        assert!((state.last_query_value - 1.5).abs() < 1e-9);
        assert!((state.last_target_value - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_composite_state_initial() {
        let state = MsmCompositeState::initial(5, 1.0, 2.0);
        assert_eq!(state.series_id, 5);
        assert_eq!(state.query_index, 0);
        assert_eq!(state.target_index, 0);
        assert!((state.last_query_value - 1.0).abs() < 1e-9);
        assert!((state.last_target_value - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_composite_state_is_final() {
        let state = MsmCompositeState::new(0, 3, 4, 0.0, 0.0);
        assert!(state.is_final(3, 4));
        assert!(state.is_final(2, 3)); // Past end
        assert!(!state.is_final(4, 4)); // Not at end of query
        assert!(!state.is_final(3, 5)); // Not at end of target
    }

    #[test]
    fn test_registry_creation() {
        let registry = MsmStateRegistry::new(10, 15);
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.max_query_len(), 10);
        assert_eq!(registry.max_target_len(), 15);
    }

    #[test]
    fn test_registry_get_or_create() {
        let mut registry = MsmStateRegistry::new(10, 15);

        let state1 = MsmCompositeState::new(0, 1, 2, 1.0, 2.0);
        let id1 = registry.get_or_create(state1);
        assert_eq!(id1, 0);
        assert_eq!(registry.len(), 1);

        // Same key should return same ID
        let state2 = MsmCompositeState::new(0, 1, 2, 1.5, 2.5);
        let id2 = registry.get_or_create(state2);
        assert_eq!(id2, id1);
        assert_eq!(registry.len(), 1);

        // Different key should create new state
        let state3 = MsmCompositeState::new(0, 2, 2, 1.0, 2.0);
        let id3 = registry.get_or_create(state3);
        assert_ne!(id3, id1);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_registry_get() {
        let mut registry = MsmStateRegistry::new(10, 15);

        let state = MsmCompositeState::new(1, 2, 3, 1.5, 2.5);
        let id = registry.get_or_create(state);

        let retrieved = registry.get(id);
        assert!(retrieved.is_some());
        let retrieved = retrieved.expect("expected Some retrieved in test");
        assert_eq!(retrieved.series_id, 1);
        assert_eq!(retrieved.query_index, 2);
        assert_eq!(retrieved.target_index, 3);
    }

    #[test]
    fn test_registry_clear() {
        let mut registry = MsmStateRegistry::new(10, 15);
        registry.get_or_create(MsmCompositeState::new(0, 1, 2, 0.0, 0.0));
        registry.get_or_create(MsmCompositeState::new(0, 2, 3, 0.0, 0.0));

        assert_eq!(registry.len(), 2);

        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_estimate_msm_states() {
        let estimate = estimate_msm_states(100, 10, 15);
        assert!(estimate > 0);
        // Should be reasonable: 100 series * 11 query pos * 16 target pos / 2 = ~8800
        assert!(estimate < 100 * 11 * 16);
    }

    #[test]
    fn test_msm_transition_all() {
        let all = MsmTransition::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&MsmTransition::Move));
        assert!(all.contains(&MsmTransition::Merge));
        assert!(all.contains(&MsmTransition::Split));
    }
}
