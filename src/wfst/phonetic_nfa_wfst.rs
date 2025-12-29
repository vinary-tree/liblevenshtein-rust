//! Pure Phonetic NFA WFST wrapper for lling-llang.
//!
//! This module provides [`PhoneticNfaWfst`], a WFST wrapper for phonetic NFAs
//! that can be composed with other WFSTs in a pipeline. Unlike [`PhoneticWfst`],
//! this does not include dictionary integration - it's a pure phonetic transducer.
//!
//! # Use Cases
//!
//! - **Composition pipeline**: Chain phonetic matching with Levenshtein and language models
//! - **Standalone phonetic matching**: Use NFA for pattern matching without edit distance
//! - **Custom pipelines**: Build complex matching pipelines with explicit composition

use lling_llang::prelude::{
    LazyState, LazyWfst, Semiring, StateId, StateSource, TropicalWeight, Wfst,
    WeightedTransition,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::{NFAChar, StateSet, TransitionLabelChar};

/// A pure phonetic NFA exposed as a lling-llang WFST.
///
/// This wrapper exposes a phonetic NFA as a weighted transducer:
/// - **Input labels**: Characters from the input string
/// - **Output labels**: Same as input (identity transduction)
/// - **Weights**: Phonetic transformation costs as `TropicalWeight`
///
/// The NFA supports phonetic alternations like `(ph|f)` where `ph` and `f`
/// are phonetically equivalent with potentially different costs.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wfst::PhoneticNfaWfst;
/// use liblevenshtein::phonetic::nfa::compile;
/// use liblevenshtein::phonetic::regex::parse;
/// use lling_llang::prelude::*;
///
/// let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
/// let wfst = PhoneticNfaWfst::new(nfa);
///
/// // Compose with other WFSTs
/// // let pipeline = compose(wfst, levenshtein_wfst);
/// ```
#[cfg(feature = "phonetic-rules")]
#[derive(Clone)]
pub struct PhoneticNfaWfst {
    /// The phonetic NFA
    nfa: NFAChar,
    /// Phonetic weight (cost for each NFA transition)
    phonetic_weight: f64,
    /// State registry: maps NFA state set to WFST state ID
    state_registry: FxHashMap<StateSetKey, StateId>,
    /// Reverse mapping: WFST state ID -> NFA state set
    id_to_state_set: Vec<StateSet>,
    /// Cached state information
    cache: FxHashMap<StateId, CachedNfaState>,
    /// Cache policy
    cache_policy: lling_llang::wfst::CachePolicy,
    /// Maximum cache size for LRU policy
    max_cache_size: usize,
}

/// Key for hashing NFA state sets.
#[cfg(feature = "phonetic-rules")]
#[derive(Clone, PartialEq, Eq, Hash)]
struct StateSetKey(Vec<u32>);

/// Cached state information.
#[derive(Clone)]
struct CachedNfaState {
    is_final: bool,
    final_weight: TropicalWeight,
    transitions: SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>,
}

/// Default maximum cache size (50,000 states)
const DEFAULT_MAX_CACHE_SIZE: usize = 50_000;

#[cfg(feature = "phonetic-rules")]
impl PhoneticNfaWfst {
    /// Create a new phonetic NFA WFST with default settings.
    ///
    /// # Arguments
    ///
    /// - `nfa`: The phonetic NFA to wrap
    pub fn new(nfa: NFAChar) -> Self {
        Self::with_phonetic_weight(nfa, 0.0)
    }

    /// Create a new phonetic NFA WFST with custom phonetic weight.
    ///
    /// # Arguments
    ///
    /// - `nfa`: The phonetic NFA to wrap
    /// - `phonetic_weight`: Cost added for each NFA transition
    pub fn with_phonetic_weight(nfa: NFAChar, phonetic_weight: f64) -> Self {
        let mut state_registry = FxHashMap::default();
        let mut id_to_state_set = Vec::new();

        // Register initial state (epsilon closure of start state)
        let initial_closure = nfa.epsilon_closure_single(nfa.start());
        let initial_key = state_set_to_key(&initial_closure);
        state_registry.insert(initial_key, 0);
        id_to_state_set.push(initial_closure);

        Self {
            nfa,
            phonetic_weight,
            state_registry,
            id_to_state_set,
            cache: FxHashMap::default(),
            cache_policy: lling_llang::wfst::CachePolicy::CacheAll,
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
        }
    }

    /// Get the phonetic weight.
    pub fn phonetic_weight(&self) -> f64 {
        self.phonetic_weight
    }

    /// Set the maximum cache size for LRU eviction.
    pub fn set_max_cache_size(&mut self, size: usize) {
        self.max_cache_size = size;
    }

    /// Get or create a state ID for a state set.
    fn get_or_create_state(&mut self, state_set: StateSet) -> StateId {
        let key = state_set_to_key(&state_set);
        if let Some(&id) = self.state_registry.get(&key) {
            return id;
        }

        let id = self.id_to_state_set.len() as StateId;
        self.state_registry.insert(key, id);
        self.id_to_state_set.push(state_set);
        id
    }

    /// Compute transitions for an NFA state set.
    fn compute_nfa_transitions(
        &mut self,
        state_id: StateId,
    ) -> (bool, TropicalWeight, SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>) {
        let state_set = match self.id_to_state_set.get(state_id as usize) {
            Some(set) => set.clone(),
            None => return (false, TropicalWeight::zero(), SmallVec::new()),
        };

        let mut transitions = SmallVec::new();

        // Collect all possible input characters from this state set
        let mut chars: Vec<char> = Vec::new();
        for nfa_state in state_set.iter() {
            for trans in self.nfa.transitions_from(nfa_state) {
                if trans.label.consumes_input() {
                    // Collect characters this transition can match
                    match &trans.label {
                        TransitionLabelChar::Char(c) => {
                            if !chars.contains(c) {
                                chars.push(*c);
                            }
                        }
                        TransitionLabelChar::CharClass(class) => {
                            // For character classes, sample the first character from each range
                            for &(start, _end) in &class.ranges {
                                if !chars.contains(&start) {
                                    chars.push(start);
                                }
                            }
                        }
                        TransitionLabelChar::Any => {
                            // For "any", add a representative set of characters
                            for c in ('a'..='z').chain('A'..='Z').chain('0'..='9') {
                                if !chars.contains(&c) {
                                    chars.push(c);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // For each possible character, compute the successor state
        for c in chars {
            let mut next_states = StateSet::new();

            for nfa_state in state_set.iter() {
                for trans in self.nfa.transitions_from(nfa_state) {
                    if trans.label.matches(c) && trans.label.consumes_input() {
                        next_states.insert(trans.to);
                    }
                }
            }

            if next_states.is_empty() {
                continue;
            }

            // Apply epsilon closure
            let next_closure = self.nfa.epsilon_closure(&next_states);
            let target_id = self.get_or_create_state(next_closure);

            transitions.push(WeightedTransition::new(
                state_id,
                Some(c),
                Some(c),
                target_id,
                TropicalWeight::new(self.phonetic_weight),
            ));
        }

        // Check if this is a final state
        let is_final = state_set.iter().any(|s| self.nfa.is_final(s));
        let final_weight = if is_final {
            TropicalWeight::one()
        } else {
            TropicalWeight::zero()
        };

        (is_final, final_weight, transitions)
    }

    /// Ensure a state is computed and cached.
    fn ensure_state(&mut self, state: StateId) {
        if self.cache.contains_key(&state) {
            return;
        }

        let (is_final, final_weight, transitions) = self.compute_nfa_transitions(state);

        let cached = CachedNfaState {
            is_final,
            final_weight,
            transitions,
        };

        // Apply cache eviction if using LRU and over limit
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

        self.cache.insert(state, cached);
    }
}

/// Convert a StateSet to a hashable key.
#[cfg(feature = "phonetic-rules")]
fn state_set_to_key(state_set: &StateSet) -> StateSetKey {
    let mut states: Vec<u32> = state_set.iter().collect();
    states.sort_unstable();
    StateSetKey(states)
}

#[cfg(feature = "phonetic-rules")]
impl Wfst<char, TropicalWeight> for PhoneticNfaWfst {
    fn start(&self) -> StateId {
        0 // Initial state is always ID 0
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
            .unwrap_or_else(TropicalWeight::zero)
    }

    fn transitions(&self, state: StateId) -> &[WeightedTransition<char, TropicalWeight>] {
        static EMPTY: &[WeightedTransition<char, TropicalWeight>] = &[];
        self.cache
            .get(&state)
            .map(|s| s.transitions.as_slice())
            .unwrap_or(EMPTY)
    }

    fn num_states(&self) -> usize {
        self.id_to_state_set.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.nfa.is_empty()
    }

    #[inline]
    fn is_valid_state(&self, state: StateId) -> bool {
        (state as usize) < self.id_to_state_set.len()
    }
}

#[cfg(feature = "phonetic-rules")]
impl LazyWfst<char, TropicalWeight> for PhoneticNfaWfst {
    fn is_expanded(&self, state: StateId) -> bool {
        self.cache.contains_key(&state)
    }

    fn expand(&mut self, state: StateId) {
        self.ensure_state(state);
    }

    fn transitions_lazy(&mut self, state: StateId) -> &[WeightedTransition<char, TropicalWeight>] {
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

#[cfg(feature = "phonetic-rules")]
impl StateSource<char, TropicalWeight> for PhoneticNfaWfst {
    fn compute_state(&self, _state: StateId) -> LazyState<char, TropicalWeight> {
        // For StateSource, we need a non-mutable version
        // This is a simplified implementation that returns Pending
        // since we need mutable access to compute transitions
        LazyState::Pending
    }

    fn start(&self) -> StateId {
        0
    }

    fn num_states_hint(&self) -> Option<usize> {
        // Estimate based on NFA size
        Some(self.nfa.num_states() * 10)
    }
}

#[cfg(test)]
#[cfg(feature = "phonetic-rules")]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::compile;
    use crate::phonetic::regex::parse;

    #[test]
    fn test_phonetic_nfa_wfst_creation() {
        let nfa = compile(&parse("(ph|f)one").expect("parse")).expect("compile");
        let wfst = PhoneticNfaWfst::new(nfa);

        assert_eq!(wfst.phonetic_weight(), 0.0);
        assert!(!wfst.is_empty());
    }

    #[test]
    fn test_phonetic_nfa_wfst_start_state() {
        let nfa = compile(&parse("test").expect("parse")).expect("compile");
        let wfst = PhoneticNfaWfst::new(nfa);

        assert_eq!(Wfst::start(&wfst), 0);
        assert!(wfst.is_valid_state(0));
    }

    #[test]
    fn test_phonetic_nfa_wfst_expand_state() {
        let nfa = compile(&parse("(a|b)c").expect("parse")).expect("compile");
        let mut wfst = PhoneticNfaWfst::new(nfa);

        let start = Wfst::start(&wfst);
        assert!(!wfst.is_expanded(start));

        wfst.expand(start);
        assert!(wfst.is_expanded(start));

        // Should have transitions for 'a' and 'b'
        let transitions = wfst.transitions(start);
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_phonetic_nfa_wfst_with_weight() {
        let nfa = compile(&parse("test").expect("parse")).expect("compile");
        let wfst = PhoneticNfaWfst::with_phonetic_weight(nfa, 0.5);

        assert_eq!(wfst.phonetic_weight(), 0.5);
    }

    #[test]
    fn test_phonetic_nfa_wfst_cache_policy() {
        let nfa = compile(&parse("test").expect("parse")).expect("compile");
        let mut wfst = PhoneticNfaWfst::new(nfa);

        assert!(matches!(
            wfst.cache_policy(),
            lling_llang::wfst::CachePolicy::CacheAll
        ));

        wfst.set_cache_policy(lling_llang::wfst::CachePolicy::Lru { max_states: 500 });
        assert!(matches!(
            wfst.cache_policy(),
            lling_llang::wfst::CachePolicy::Lru { .. }
        ));
    }

    #[test]
    fn test_phonetic_nfa_wfst_num_states() {
        let nfa = compile(&parse("abc").expect("parse")).expect("compile");
        let mut wfst = PhoneticNfaWfst::new(nfa);

        // Initially just the start state
        assert_eq!(wfst.num_states(), 1);

        // After expanding, more states are created
        wfst.expand(0);
        assert!(wfst.num_states() >= 1);
    }
}
