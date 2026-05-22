//! PhoneticWfst wrapper for lling-llang Wfst trait.
//!
//! This module provides [`PhoneticWfst`], a wrapper that exposes a phonetic
//! transducer (NFA × Levenshtein × Dictionary) as a lling-llang WFST.
//!
//! # Key Benefits
//!
//! - **Sound-alike matching**: Matches phonetically similar words (ph ↔ f)
//! - **Edit tolerance**: Combined with Levenshtein for typo tolerance
//! - **Dictionary integration**: Efficiently traverses dictionary structure
//! - **WFST composition**: Can be composed with language models

use lling_llang::prelude::{
    LazyState, LazyWfst, Semiring, StateId, StateSource, TropicalWeight, WeightedTransition, Wfst,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use libdictenstein::{Dictionary, DictionaryNode};

use super::state_encoding;

#[cfg(feature = "phonetic-rules")]
use super::phonetic_state_source::PhoneticStateSource;
#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::NFAChar;

/// A phonetic transducer exposed as a lling-llang WFST.
///
/// This wrapper presents the product of a phonetic NFA, Levenshtein automaton,
/// and dictionary as a weighted finite state transducer with:
/// - **Input labels**: Dictionary characters
/// - **Output labels**: Dictionary characters
/// - **Weights**: Combined phonetic + edit distance as `TropicalWeight`
///
/// # Type Parameters
///
/// - `D`: Dictionary type implementing [`Dictionary`] with `char` units
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wfst::PhoneticWfst;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
/// use liblevenshtein::phonetic::nfa::compile;
/// use liblevenshtein::phonetic::regex::parse;
/// use lling_llang::prelude::*;
///
/// let dict = DynamicDawgChar::from_terms(vec!["phone", "fone", "bone"]);
/// let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
/// let wfst = PhoneticWfst::new(&dict, nfa, 2);
///
/// // Use with lling-llang's composition
/// // let composed = compose(wfst, language_model);
/// ```
#[cfg(feature = "phonetic-rules")]
#[derive(Clone)]
pub struct PhoneticWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// The state source for computing transitions
    state_source: PhoneticStateSource<D>,
    /// Cached states (state_id -> computed state info)
    cache: FxHashMap<StateId, CachedState>,
    /// Maximum edit distance
    max_distance: u8,
    /// Phonetic weight
    phonetic_weight: f64,
    /// Maximum product states for state encoding
    max_product_states: u32,
    /// Cache policy
    cache_policy: lling_llang::wfst::CachePolicy,
    /// Maximum cache size for LRU policy
    max_cache_size: usize,
}

/// Cached state information for a single WFST state.
#[derive(Clone)]
struct CachedState {
    is_final: bool,
    final_weight: TropicalWeight,
    transitions: SmallVec<[WeightedTransition<char, TropicalWeight>; 4]>,
}

/// Default maximum cache size for LRU policy (100,000 states)
const DEFAULT_MAX_CACHE_SIZE: usize = 100_000;

#[cfg(feature = "phonetic-rules")]
impl<D> PhoneticWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Create a new phonetic WFST for the given NFA pattern and max distance.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The dictionary to search
    /// - `nfa`: The phonetic NFA pattern (e.g., compiled from "(ph|f)one")
    /// - `max_distance`: Maximum edit distance for matches
    ///
    /// # Returns
    ///
    /// A new `PhoneticWfst` ready for composition or traversal.
    pub fn new(dictionary: &D, nfa: NFAChar, max_distance: u8) -> Self {
        Self::with_phonetic_weight(dictionary, nfa, max_distance, 0.0)
    }

    /// Create a new phonetic WFST with a custom phonetic weight.
    ///
    /// # Arguments
    ///
    /// - `dictionary`: The dictionary to search
    /// - `nfa`: The phonetic NFA pattern
    /// - `max_distance`: Maximum edit distance for matches
    /// - `phonetic_weight`: Cost added for phonetic transformations
    pub fn with_phonetic_weight(
        dictionary: &D,
        nfa: NFAChar,
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        let state_source = PhoneticStateSource::with_phonetic_weight(
            dictionary,
            nfa,
            max_distance,
            phonetic_weight,
        );

        // Estimate max product states
        let max_product_states = ((max_distance as u32 + 1) * 1000).max(10_000);

        Self {
            state_source,
            cache: FxHashMap::default(),
            max_distance,
            phonetic_weight,
            max_product_states,
            cache_policy: lling_llang::wfst::CachePolicy::CacheAll,
            max_cache_size: DEFAULT_MAX_CACHE_SIZE,
        }
    }

    /// Get the maximum edit distance.
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Get the phonetic weight.
    pub fn phonetic_weight(&self) -> f64 {
        self.phonetic_weight
    }

    /// Set the maximum cache size for LRU eviction.
    pub fn set_max_cache_size(&mut self, size: usize) {
        self.max_cache_size = size;
    }

    /// Ensure a state is computed and cached.
    fn ensure_state(&mut self, state: StateId) {
        if self.cache.contains_key(&state) {
            return;
        }

        // Use the state source to compute the state
        let lazy_state = self.state_source.compute_state(state);

        // Convert LazyState to CachedState via pattern matching
        let cached = match lazy_state {
            LazyState::Computed {
                is_final,
                final_weight,
                transitions,
            } => CachedState {
                is_final,
                final_weight,
                transitions,
            },
            LazyState::Pending => CachedState {
                is_final: false,
                final_weight: TropicalWeight::zero(),
                transitions: SmallVec::new(),
            },
        };

        // Apply cache eviction if using LRU and over limit
        if let lling_llang::wfst::CachePolicy::Lru { max_states } = self.cache_policy {
            let limit = if max_states > 0 {
                max_states
            } else {
                self.max_cache_size
            };
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

#[cfg(feature = "phonetic-rules")]
impl<D> Wfst<char, TropicalWeight> for PhoneticWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    fn start(&self) -> StateId {
        self.state_source.start()
    }

    fn is_final(&self, state: StateId) -> bool {
        self.cache.get(&state).map(|s| s.is_final).unwrap_or(false)
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
        self.cache.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        false
    }

    #[inline]
    fn is_valid_state(&self, state: StateId) -> bool {
        let (dict_node, product_state) = state_encoding::decode(state, self.max_product_states);
        product_state < self.max_product_states || dict_node == 0
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D> LazyWfst<char, TropicalWeight> for PhoneticWfst<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
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

/// Builder for PhoneticWfst with pattern string support.
///
/// This provides a convenient API for creating phonetic WFSTs from
/// pattern strings without manually compiling the NFA.
#[cfg(feature = "phonetic-rules")]
pub struct PhoneticWfstBuilder<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    dictionary: D,
    max_distance: u8,
    phonetic_weight: f64,
}

#[cfg(feature = "phonetic-rules")]
impl<D> PhoneticWfstBuilder<D>
where
    D: Dictionary + Clone + Send + Sync,
    D::Node: Send + Sync,
    <D::Node as DictionaryNode>::Unit: Into<char> + TryFrom<char> + Copy + Send + Sync,
{
    /// Create a new builder for the given dictionary.
    pub fn new(dictionary: D, max_distance: u8) -> Self {
        Self {
            dictionary,
            max_distance,
            phonetic_weight: 0.0,
        }
    }

    /// Set the phonetic weight.
    pub fn phonetic_weight(mut self, weight: f64) -> Self {
        self.phonetic_weight = weight;
        self
    }

    /// Build a PhoneticWfst from a pattern string.
    ///
    /// # Arguments
    ///
    /// - `pattern`: A phonetic regex pattern (e.g., "(ph|f)one")
    ///
    /// # Returns
    ///
    /// A `Result` containing the `PhoneticWfst` or an error if parsing fails.
    pub fn build_from_pattern(self, pattern: &str) -> Result<PhoneticWfst<D>, String> {
        use crate::phonetic::nfa::compiler::compile;
        use crate::phonetic::regex::parse;

        let ast = parse(pattern).map_err(|e| format!("Parse error: {:?}", e))?;
        let nfa = compile(&ast).map_err(|e| format!("Compile error: {:?}", e))?;

        Ok(PhoneticWfst::with_phonetic_weight(
            &self.dictionary,
            nfa,
            self.max_distance,
            self.phonetic_weight,
        ))
    }
}

#[cfg(test)]
#[cfg(feature = "phonetic-rules")]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::compile;
    use crate::phonetic::regex::parse;
    use libdictenstein::dynamic_dawg_char::DynamicDawgChar;

    #[test]
    fn test_phonetic_wfst_creation() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "fone", "help"]);
        let nfa = compile(&parse("(ph|f)one").expect("parse")).expect("compile");
        let wfst = PhoneticWfst::new(&dict, nfa, 2);

        assert_eq!(wfst.max_distance(), 2);
        assert_eq!(wfst.phonetic_weight(), 0.0);
    }

    #[test]
    fn test_phonetic_wfst_start_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "help"]);
        let nfa = compile(&parse("(ph|f)one").expect("parse")).expect("compile");
        let wfst = PhoneticWfst::new(&dict, nfa, 2);

        let start = wfst.start();
        let (dict_node, product_state) = state_encoding::decode(start, wfst.max_product_states);
        assert_eq!(dict_node, 0);
        assert_eq!(product_state, 0);
    }

    #[test]
    fn test_phonetic_wfst_expand_state() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "fone"]);
        let nfa = compile(&parse("(ph|f)one").expect("parse")).expect("compile");
        let mut wfst = PhoneticWfst::new(&dict, nfa, 2);

        let start = wfst.start();
        assert!(!wfst.is_expanded(start));

        wfst.expand(start);
        assert!(wfst.is_expanded(start));
        assert!(wfst.computed_states() >= 1);
    }

    #[test]
    fn test_phonetic_wfst_with_weight() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone"]);
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");
        let wfst = PhoneticWfst::with_phonetic_weight(&dict, nfa, 2, 0.5);

        assert_eq!(wfst.phonetic_weight(), 0.5);
    }

    #[test]
    fn test_phonetic_wfst_cache_policy() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["test"]);
        let nfa = compile(&parse("test").expect("parse")).expect("compile");
        let mut wfst = PhoneticWfst::new(&dict, nfa, 1);

        assert!(matches!(
            wfst.cache_policy(),
            lling_llang::wfst::CachePolicy::CacheAll
        ));

        wfst.set_cache_policy(lling_llang::wfst::CachePolicy::Lru { max_states: 1000 });
        assert!(matches!(
            wfst.cache_policy(),
            lling_llang::wfst::CachePolicy::Lru { .. }
        ));
    }

    #[test]
    fn test_phonetic_wfst_builder() {
        let dict = DynamicDawgChar::<()>::from_terms(vec!["phone", "fone"]);
        let builder = PhoneticWfstBuilder::new(dict, 2).phonetic_weight(0.1);

        let wfst = builder.build_from_pattern("(ph|f)one").expect("build");
        assert_eq!(wfst.max_distance(), 2);
        assert_eq!(wfst.phonetic_weight(), 0.1);
    }
}
