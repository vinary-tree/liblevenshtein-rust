//! Universal Levenshtein Automaton
//!
//! Implements the complete Universal Levenshtein Automaton A^∀,χ_n from Mitankin's thesis.
//!
//! # Theory Background
//!
//! ## Definition 15: Universal Levenshtein Automaton (Page 30)
//!
//! ```text
//! A^∀,χ_n = ⟨Σ^∀_n, Q^∀,χ_n, I^∀,χ, F^∀,χ_n, δ^∀,χ_n⟩
//! ```
//!
//! Where:
//! - **Σ^∀_n**: Bit vectors of length ≤ 2n + 2
//! - **Q^∀,χ_n**: State space (I^χ_states ∪ M^χ_states)
//! - **I^∀,χ**: Initial state {I#0}
//! - **F^∀,χ_n**: Final states M^χ_states (states with M-type positions)
//! - **δ^∀,χ_n**: Transition function
//!
//! ## Acceptance Condition (Page 48)
//!
//! Given a word w and input x, the automaton accepts if:
//! 1. Encode the pair as h_n(w, x) = β(x₁, s_n(w,1))...β(x_t, s_n(w,t))
//! 2. Process the bit vector sequence: δ^∀,χ_n*(I^∀,χ, h_n(w, x))
//! 3. Check if the resulting state is in F^∀,χ_n (contains M-type positions)
//!
//! ## Key Properties
//!
//! - **Parameter-free**: Same automaton works for any word w
//! - **Deterministic**: δ^∀,χ_n is a function (not a relation)
//! - **Finite**: State space is finite for fixed n
//! - **Universal**: Simulates A^D,χ_n(w) for any word w

use crate::transducer::universal::{
    CharacteristicVector, PositionVariant, UniversalPosition, UniversalState,
};
use crate::transducer::{SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use libdictenstein::CharUnit;

/// Universal Levenshtein Automaton A^∀,χ_n
///
/// A parameter-free automaton that recognizes the Levenshtein neighborhood
/// L^χ_{Lev}(n, w) for any word w without modification.
///
/// # Type Parameters
///
/// - `V`: Position variant (Standard, Transposition, or MergeAndSplit)
/// - `P`: Zero-cost substitution policy (defaults to [`Unrestricted`])
///
/// The default [`Unrestricted`] policy is a zero-sized type, so there is
/// zero memory or performance overhead for the default case.
///
/// A configured policy participates in characteristic-vector construction for
/// every online transition. Policy direction is dictionary unit first, query
/// unit second, consistently with the other transducer engines.
///
/// # Examples
///
/// ```rust
/// use liblevenshtein::transducer::universal::{UniversalAutomaton, Standard};
///
/// // Create automaton for maximum distance n=2
/// let automaton = UniversalAutomaton::<Standard>::new(2);
///
/// // Check if "test" accepts "text" (distance 1)
/// assert!(automaton.accepts("test", "text"));
///
/// // Check if "test" accepts "hello" (distance > 2)
/// assert!(!automaton.accepts("test", "hello"));
/// ```
#[derive(Debug, Clone)]
pub struct UniversalAutomaton<V: PositionVariant, P: SubstitutionPolicy = Unrestricted> {
    /// Maximum edit distance n
    max_distance: u8,
    /// Substitution equivalence used by policy-aware characteristic vectors.
    policy: P,
    /// Position behavior is selected through monomorphization.
    _variant: std::marker::PhantomData<V>,
}

/// Stable online execution state for one fixed dictionary word.
///
/// Each call to [`advance`](Self::advance) consumes one input scalar and keeps
/// only the fixed word, the current canonical universal antichain, and a scalar
/// position counter. Retained memory is independent of the consumed input
/// prefix and no recursion is used.
#[derive(Debug, Clone)]
pub struct UniversalOnlineAutomaton<
    V: PositionVariant,
    P: SubstitutionPolicy = Unrestricted,
    U: CharUnit = char,
> {
    max_distance: u8,
    word: Vec<U>,
    state: Option<UniversalState<V>>,
    input_length: usize,
    policy: P,
}

impl<V, P, U> UniversalOnlineAutomaton<V, P, U>
where
    V: PositionVariant,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    U: CharUnit,
{
    /// Consume one input unit. `false` means the exact universal frontier is
    /// dead; subsequent calls remain dead and allocate nothing.
    pub fn advance(&mut self, input: U) -> bool {
        let Some(position) = self.input_length.checked_add(1) else {
            self.state = None;
            return false;
        };
        self.input_length = position;
        if position
            > self
                .word
                .len()
                .saturating_add(usize::from(self.max_distance))
        {
            self.state = None;
            return false;
        }

        let Some(source) = self.state.take() else {
            return false;
        };
        let (false_prefix, relevant) =
            relevant_word_window(&self.word, position, self.max_distance);
        let characteristic = CharacteristicVector::from_padded_units_with_policy(
            input,
            false_prefix,
            relevant,
            &self.policy,
        );
        self.state = source.transition(&characteristic, position);
        self.state.is_some()
    }

    /// Whether the consumed prefix is currently an accepted complete input.
    pub fn is_accepting(&self) -> bool {
        self.state.as_ref().is_some_and(|state| {
            universal_state_is_accepting(
                state,
                self.word.len(),
                self.input_length,
                self.max_distance,
            )
        })
    }

    /// Number of input scalars consumed so far.
    pub fn input_length(&self) -> usize {
        self.input_length
    }

    /// Fixed word length retained by this online machine.
    pub fn word_length(&self) -> usize {
        self.word.len()
    }

    /// Current canonical universal state, or `None` after the frontier dies.
    pub fn state(&self) -> Option<&UniversalState<V>> {
        self.state.as_ref()
    }

    /// Substitution policy retained by this online machine.
    pub fn policy(&self) -> &P {
        &self.policy
    }
}

fn relevant_word_window<U: CharUnit>(
    word: &[U],
    position: usize,
    max_distance: u8,
) -> (usize, &[U]) {
    let distance = usize::from(max_distance);
    let false_prefix = distance.saturating_add(1).saturating_sub(position);
    let start = position.saturating_sub(distance).max(1);
    let end = position
        .saturating_add(distance)
        .saturating_add(1)
        .min(word.len());
    let relevant = if start <= end {
        &word[start - 1..end]
    } else {
        &[]
    };
    (false_prefix, relevant)
}

fn universal_state_is_accepting<V: PositionVariant>(
    state: &UniversalState<V>,
    word_len: usize,
    input_len: usize,
    max_distance: u8,
) -> bool {
    let distance = i128::from(max_distance);
    let word_len = word_len as i128;
    let input_len = input_len as i128;

    state.positions().any(|position| {
        if position.is_m_type() {
            position.offset() <= 0 && position.errors() <= max_distance
        } else {
            let current_word_pos = input_len + i128::from(position.offset());
            if current_word_pos < 0 {
                return false;
            }
            let remaining_chars = word_len - current_word_pos;
            let remaining_errors = distance - i128::from(position.errors());
            remaining_chars >= 0 && remaining_chars <= remaining_errors
        }
    })
}

// Backward-compatible constructors for Unrestricted policy
impl<V: PositionVariant> UniversalAutomaton<V, Unrestricted> {
    /// Create a new Universal Levenshtein Automaton for maximum distance n
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n (typically 1, 2, or 3)
    ///
    /// # Returns
    ///
    /// A new `UniversalAutomaton` instance with unrestricted substitutions
    ///
    /// # Examples
    ///
    /// ```rust
    /// use liblevenshtein::transducer::universal::{Standard, UniversalAutomaton};
    ///
    /// let automaton = UniversalAutomaton::<Standard>::new(2);
    /// assert_eq!(automaton.max_distance(), 2);
    /// ```
    #[must_use]
    pub fn new(max_distance: u8) -> Self {
        Self {
            max_distance,
            policy: Unrestricted,
            _variant: std::marker::PhantomData,
        }
    }
}

// Generic methods (work with any policy)
impl<V: PositionVariant, P: SubstitutionPolicy> UniversalAutomaton<V, P> {
    /// Construct a Universal Levenshtein Automaton with a zero-cost
    /// substitution policy.
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n (typically 1, 2, or 3)
    /// - `policy`: Policy applied to dictionary/query unit pairs
    ///
    /// # Returns
    ///
    /// A new `UniversalAutomaton` instance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use liblevenshtein::transducer::substitution_policy::RestrictedChar;
    /// use liblevenshtein::transducer::universal::{Standard, UniversalAutomaton};
    /// use liblevenshtein::transducer::SubstitutionSetChar;
    ///
    /// let mut policy_set = SubstitutionSetChar::new();
    /// policy_set.allow('é', 'e');
    /// let policy = RestrictedChar::new(&policy_set);
    /// let automaton = UniversalAutomaton::<Standard, _>::with_policy(0, policy);
    /// assert!(automaton.accepts("café", "cafe"));
    /// ```
    #[must_use]
    pub fn with_policy(max_distance: u8, policy: P) -> Self {
        Self {
            max_distance,
            policy,
            _variant: std::marker::PhantomData,
        }
    }

    /// Get the maximum edit distance n
    #[must_use]
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Substitution policy used by policy-aware characteristic vectors.
    #[must_use]
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Create the initial state I^∀,χ = {I#0}
    ///
    /// From thesis page 38: The initial state contains a single I-type position
    /// with offset 0 and 0 errors.
    ///
    /// # Returns
    ///
    /// Initial state {I#0}
    fn initial_state(&self) -> UniversalState<V> {
        let mut state = UniversalState::new(self.max_distance);
        // I#0: I-type position with offset 0, errors 0
        if let Ok(pos) = UniversalPosition::new_i(0, 0, self.max_distance) {
            state.add_position(pos);
        }
        state
    }

    #[cfg(test)]
    fn is_accepting(&self, state: &UniversalState<V>, word_len: usize, input_len: usize) -> bool {
        universal_state_is_accepting(state, word_len, input_len, self.max_distance)
    }

    /// Bind this parameter-free automaton to one fixed word for stable online
    /// input processing.
    pub fn online(&self, word: &str) -> UniversalOnlineAutomaton<V, P>
    where
        P: SubstitutionPolicyFor<char>,
    {
        self.online_owned_units(word.chars().collect())
    }

    /// Bind this automaton to an arbitrary fixed unit sequence for stable
    /// online processing.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::universal::{Standard, UniversalAutomaton};
    ///
    /// let automaton = UniversalAutomaton::<Standard>::new(1);
    /// let mut online = automaton.online_units(&[10_u64, 20, 30]);
    /// for unit in [10_u64, 25, 30] {
    ///     assert!(online.advance(unit));
    /// }
    /// assert!(online.is_accepting());
    /// ```
    pub fn online_units<U>(&self, word: &[U]) -> UniversalOnlineAutomaton<V, P, U>
    where
        U: CharUnit,
        P: SubstitutionPolicyFor<U>,
    {
        self.online_owned_units(word.to_vec())
    }

    #[inline]
    fn online_owned_units<U>(&self, word: Vec<U>) -> UniversalOnlineAutomaton<V, P, U>
    where
        U: CharUnit,
        P: SubstitutionPolicyFor<U>,
    {
        UniversalOnlineAutomaton {
            max_distance: self.max_distance,
            word,
            state: Some(self.initial_state()),
            input_length: 0,
            policy: self.policy.clone(),
        }
    }

    /// Bind this automaton to one fixed byte sequence for stable online
    /// processing.
    ///
    /// Unlike [`online`](Self::online), this method does not require UTF-8.
    pub fn online_bytes(&self, word: &[u8]) -> UniversalOnlineAutomaton<V, P, u8>
    where
        P: SubstitutionPolicyFor<u8>,
    {
        self.online_units(word)
    }

    /// Check if word w accepts input x within the maximum distance
    ///
    /// From thesis page 51-52: Encodes the pair (w, x) as h_n(w, x) and
    /// processes it through the automaton.
    ///
    /// # Arguments
    ///
    /// - `word`: Dictionary word w
    /// - `input`: Input string x to match against
    ///
    /// # Returns
    ///
    /// `true` if Lev(w, x) ≤ n, `false` otherwise
    ///
    /// # Algorithm
    ///
    /// 1. Start with initial state I^∀,χ = {I#0}
    /// 2. For each character x_i in input:
    ///    - Compute relevant subword s_n(w, i)
    ///    - Compute characteristic vector β(x_i, s_n(w, i))
    ///    - Apply transition: state := δ^∀,χ_n(state, β)
    /// 3. Check if final state is in F^∀,χ_n (contains M-type positions)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use liblevenshtein::transducer::universal::{Standard, UniversalAutomaton};
    ///
    /// let automaton = UniversalAutomaton::<Standard>::new(2);
    ///
    /// // Distance 1: one substitution
    /// assert!(automaton.accepts("test", "text"));
    ///
    /// // Distance 0: identical
    /// assert!(automaton.accepts("test", "test"));
    ///
    /// // Distance 3: too far
    /// assert!(!automaton.accepts("test", "hello"));
    /// ```
    pub fn accepts(&self, word: &str, input: &str) -> bool
    where
        P: SubstitutionPolicyFor<char>,
    {
        let mut online = self.online(word);
        for input_char in input.chars() {
            if !online.advance(input_char) {
                return false;
            }
        }
        online.is_accepting()
    }

    /// Check whether two arbitrary unit sequences are within the configured
    /// distance under this substitution policy.
    ///
    /// This is the unit-generic counterpart to [`accepts`](Self::accepts).
    pub fn accepts_units<U>(&self, word: &[U], input: &[U]) -> bool
    where
        U: CharUnit,
        P: SubstitutionPolicyFor<U>,
    {
        let mut online = self.online_units(word);
        for input_unit in input.iter().copied() {
            if !online.advance(input_unit) {
                return false;
            }
        }
        online.is_accepting()
    }

    /// Check whether two byte sequences are within the configured distance
    /// under this substitution policy.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::universal::{Standard, UniversalAutomaton};
    ///
    /// let automaton = UniversalAutomaton::<Standard>::new(1);
    /// assert!(automaton.accepts_bytes(&[0xff, 0x00], &[0xfe, 0x00]));
    /// ```
    pub fn accepts_bytes(&self, word: &[u8], input: &[u8]) -> bool
    where
        P: SubstitutionPolicyFor<u8>,
    {
        self.accepts_units(word, input)
    }

    /// Compute relevant subword s_n(w, i)
    ///
    /// From thesis page 51:
    /// ```text
    /// s_n(w, i) = w_{i-n}w_{i-n+1}...w_v
    /// where v = min(|w|, i + n + 1)
    /// ```
    ///
    /// Pad with '$' for positions before start of word.
    ///
    /// # Arguments
    ///
    /// - `word`: Dictionary word w
    /// - `position`: Position i (1-indexed)
    ///
    /// # Returns
    ///
    /// Relevant subword around position i
    #[cfg(test)]
    fn relevant_subword(&self, word: &str, position: usize) -> String {
        let word_chars: Vec<char> = word.chars().collect();
        self.relevant_subword_from_chars(&word_chars, position)
    }

    #[cfg(test)]
    fn relevant_subword_from_chars(&self, word_chars: &[char], position: usize) -> String {
        let n = self.max_distance as usize;
        let word_len = word_chars.len();

        // From thesis page 51: s_n(w, i) = w_{i-n}...w_v where v = min(|w|, i + n + 1).
        // Positions are 1-indexed in the thesis, while slices are 0-indexed.
        let pad_count = n.saturating_add(1).saturating_sub(position);
        let start = position.saturating_sub(n).max(1);
        let end = position.saturating_add(n).saturating_add(1).min(word_len);
        let word_char_count = if start <= end { end - start + 1 } else { 0 };

        let mut result = String::with_capacity(pad_count.saturating_add(word_char_count));
        result.extend(std::iter::repeat_n('$', pad_count));

        if start <= end {
            result.extend(word_chars[start - 1..end].iter().copied());
        }

        result
    }

    /// Process a bit vector sequence and return the final state
    ///
    /// This is useful for debugging and testing intermediate states.
    ///
    /// # Arguments
    ///
    /// - `bit_vectors`: Sequence of characteristic vectors
    ///
    /// # Returns
    ///
    /// Final state after processing all bit vectors, or None if any transition fails
    pub fn process(&self, bit_vectors: &[CharacteristicVector]) -> Option<UniversalState<V>> {
        let mut state = self.initial_state();

        for (i, bv) in bit_vectors.iter().enumerate() {
            state = state.transition(bv, i.saturating_add(1))?;
        }

        Some(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::universal::{MergeAndSplit, Standard, Transposition};
    use crate::transducer::{
        OwnedRestricted, OwnedRestrictedChar, Restricted, RestrictedChar, SubstitutionSet,
        SubstitutionSetChar,
    };
    use proptest::prelude::*;

    fn zero_cost_levenshtein<U, F>(word: &[U], input: &[U], mut is_equivalent: F) -> usize
    where
        U: Copy + Eq,
        F: FnMut(U, U) -> bool,
    {
        let mut previous: Vec<usize> = (0..=input.len()).collect();
        let mut current = vec![0; input.len().saturating_add(1)];

        for (word_index, dict_unit) in word.iter().copied().enumerate() {
            current[0] = word_index.saturating_add(1);
            for (input_index, query_unit) in input.iter().copied().enumerate() {
                let substitution_cost = usize::from(!is_equivalent(dict_unit, query_unit));
                current[input_index.saturating_add(1)] = previous[input_index]
                    .saturating_add(substitution_cost)
                    .min(previous[input_index.saturating_add(1)].saturating_add(1))
                    .min(current[input_index].saturating_add(1));
            }
            std::mem::swap(&mut previous, &mut current);
        }

        previous[input.len()]
    }

    fn assert_unicode_policy_semantics<V: PositionVariant>() {
        let mut substitutions = SubstitutionSetChar::new();
        substitutions.allow('é', 'e');
        let automaton =
            UniversalAutomaton::<V, _>::with_policy(0, RestrictedChar::new(&substitutions));

        assert!(automaton.accepts("café", "cafe"));
        assert!(automaton.accepts("café", "café"));
        assert!(!automaton.accepts("cafe", "café"));
        assert!(!automaton.accepts("café", "cafx"));

        let mut online = automaton.online("café");
        for query_unit in "cafe".chars() {
            assert!(online.advance(query_unit));
        }
        assert!(online.is_accepting());
        assert_eq!(online.word_length(), 4);
        assert_eq!(online.input_length(), 4);
    }

    #[test]
    fn unicode_policy_is_directional_for_every_universal_variant() {
        assert_unicode_policy_semantics::<Standard>();
        assert_unicode_policy_semantics::<Transposition>();
        assert_unicode_policy_semantics::<MergeAndSplit>();
    }

    #[test]
    fn byte_policy_drives_batch_and_online_universal_matching() {
        let mut substitutions = SubstitutionSet::new();
        substitutions.allow_byte(b'k', b'c');
        let automaton =
            UniversalAutomaton::<Standard, _>::with_policy(0, Restricted::new(&substitutions));

        assert!(automaton.accepts_bytes(b"kit", b"cit"));
        assert!(automaton.accepts_units(b"kit", b"cit"));
        assert!(!automaton.accepts_bytes(b"cit", b"kit"));
        assert!(!automaton.accepts_bytes(b"kit", b"sit"));

        let mut online = automaton.online_bytes(b"kit");
        for query_unit in b"cit" {
            assert!(online.advance(*query_unit));
        }
        assert!(online.is_accepting());
        assert!(std::ptr::eq(online.policy().set(), &substitutions));
    }

    #[test]
    fn owned_byte_and_unicode_policies_survive_constructor_scope() {
        let byte_automaton = {
            let mut substitutions = SubstitutionSet::new();
            substitutions.allow_byte(0xff, 0xfe);
            UniversalAutomaton::<Standard, _>::with_policy(0, OwnedRestricted::new(substitutions))
        };
        assert!(byte_automaton.accepts_bytes(&[0xff], &[0xfe]));
        assert!(!byte_automaton.accepts_bytes(&[0xfe], &[0xff]));

        let unicode_automaton = {
            let mut substitutions = SubstitutionSetChar::new();
            substitutions.allow('Ω', 'ω');
            UniversalAutomaton::<Standard, _>::with_policy(
                0,
                OwnedRestrictedChar::new(substitutions),
            )
        };
        assert!(unicode_automaton.accepts("Ω", "ω"));
        assert!(!unicode_automaton.accepts("ω", "Ω"));
    }

    #[derive(Clone)]
    struct TokenEquivalence;

    impl SubstitutionPolicy for TokenEquivalence {
        fn is_allowed(&self, _dict_char: u8, _query_char: u8) -> bool {
            false
        }
    }

    impl SubstitutionPolicyFor<u64> for TokenEquivalence {
        fn is_allowed_for(&self, dict_unit: u64, query_unit: u64) -> bool {
            dict_unit == 22 && query_unit == 99
        }
    }

    #[test]
    fn custom_u64_policy_uses_the_same_unit_generic_encoder() {
        let automaton = UniversalAutomaton::<Standard, _>::with_policy(0, TokenEquivalence);
        assert!(automaton.accepts_units(&[11_u64, 22, 33], &[11, 99, 33]));
        assert!(!automaton.accepts_units(&[11_u64, 99, 33], &[11, 22, 33]));
    }

    #[test]
    fn policy_equivalence_composes_with_ordinary_edits() {
        let mut substitutions = SubstitutionSetChar::new();
        substitutions.allow('é', 'e');
        let automaton =
            UniversalAutomaton::<Standard, _>::with_policy(1, RestrictedChar::new(&substitutions));

        assert!(automaton.accepts("café", "xcafe"));
        assert!(automaton.accepts("xcafé", "cafe"));
        assert!(!automaton.accepts("café", "xxcafe"));
    }

    #[test]
    fn unrestricted_unit_api_covers_u64_tokens_without_policy_overhead() {
        let automaton = UniversalAutomaton::<Standard>::new(0);
        assert!(automaton.accepts_units(&[11_u64, 22, 33], &[11, 22, 33]));
        assert!(!automaton.accepts_units(&[11_u64, 22, 33], &[11, 99, 33]));
    }

    proptest! {
        #[test]
        fn unicode_policy_matches_directional_dynamic_programming_oracle(
            word in prop::collection::vec(prop_oneof![Just('a'), Just('c'), Just('e'), Just('é'), Just('x')], 0..9),
            input in prop::collection::vec(prop_oneof![Just('a'), Just('c'), Just('e'), Just('é'), Just('x')], 0..9),
            max_distance in 0_u8..=3,
        ) {
            let mut substitutions = SubstitutionSetChar::new();
            substitutions.allow('é', 'e');
            substitutions.allow('c', 'x');
            let policy = RestrictedChar::new(&substitutions);
            let automaton = UniversalAutomaton::<Standard, _>::with_policy(max_distance, policy);
            let expected_distance = zero_cost_levenshtein(&word, &input, |dict_unit, query_unit| {
                dict_unit == query_unit || substitutions.contains(dict_unit, query_unit)
            });

            prop_assert_eq!(
                automaton.accepts_units(&word, &input),
                expected_distance <= usize::from(max_distance)
            );
        }

        #[test]
        fn byte_policy_matches_directional_dynamic_programming_oracle(
            word in prop::collection::vec(prop_oneof![Just(b'a'), Just(b'c'), Just(b'k'), Just(b'x')], 0..9),
            input in prop::collection::vec(prop_oneof![Just(b'a'), Just(b'c'), Just(b'k'), Just(b'x')], 0..9),
            max_distance in 0_u8..=3,
        ) {
            let mut substitutions = SubstitutionSet::new();
            substitutions.allow_byte(b'k', b'c');
            substitutions.allow_byte(b'x', b'a');
            let policy = Restricted::new(&substitutions);
            let automaton = UniversalAutomaton::<Standard, _>::with_policy(max_distance, policy);
            let expected_distance = zero_cost_levenshtein(&word, &input, |dict_unit, query_unit| {
                dict_unit == query_unit || substitutions.contains(dict_unit, query_unit)
            });

            prop_assert_eq!(
                automaton.accepts_units(&word, &input),
                expected_distance <= usize::from(max_distance)
            );
        }
    }

    #[test]
    fn online_universal_state_is_prefix_independent_and_batch_equivalent() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut online = automaton.online("test");
        for character in "text".chars() {
            assert!(online.advance(character));
        }
        assert!(online.is_accepting());
        assert_eq!(online.word_length(), 4);
        assert_eq!(online.input_length(), 4);
        assert_eq!(automaton.accepts("test", "text"), online.is_accepting());

        for _ in 0..100_000 {
            online.advance('x');
        }
        assert!(online.state().is_none());
        assert_eq!(online.word_length(), 4);
    }

    // =========================================================================
    // Basic Automaton Creation Tests
    // =========================================================================

    #[test]
    fn test_new_automaton() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        assert_eq!(automaton.max_distance(), 2);
    }

    #[test]
    fn test_initial_state() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let state = automaton.initial_state();

        // Initial state should have exactly one position: I#0
        let positions: Vec<_> = state.positions().collect();
        assert_eq!(positions.len(), 1);
        assert!(positions[0].is_i_type());
        assert_eq!(positions[0].offset(), 0);
        assert_eq!(positions[0].errors(), 0);
    }

    #[test]
    fn test_is_accepting_i_type_at_word_end() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        // I + 0#0 after processing 4 chars of 4-char word
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        // Should be accepting: word_len=4, input_len=4, offset=0, errors=0
        // current_word_pos = 4 + 0 = 4, remaining = 4 - 4 = 0 ≤ (2 - 0) = 2 ✓
        assert!(automaton.is_accepting(&state, 4, 4));
    }

    #[test]
    fn test_is_accepting_i_type_before_word_end() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        // I + 0#0 after processing 2 chars of 4-char word
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        // Should be accepting: word_len=4, input_len=2, offset=0, errors=0
        // current_word_pos = 2 + 0 = 2, remaining = 4 - 2 = 2 ≤ (2 - 0) = 2 ✓
        assert!(automaton.is_accepting(&state, 4, 2));
    }

    #[test]
    fn test_is_accepting_m_type_state() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        // M + 0#0 (past word end with 0 errors)
        state.add_position(
            UniversalPosition::new_m(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_m with valid args"),
        );

        // M-type with offset ≤ 0 and errors ≤ n is accepting
        assert!(automaton.is_accepting(&state, 4, 5));
    }

    #[test]
    fn test_is_accepting_mixed_state() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );
        state.add_position(
            UniversalPosition::new_m(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_m with valid args"),
        );

        // State with at least one accepting position (M-type) is accepting
        assert!(automaton.is_accepting(&state, 4, 5));
    }

    #[test]
    fn test_not_accepting_too_many_remaining_chars() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        let mut state = UniversalState::new(2);
        // I + 0#0 after processing 0 chars of 4-char word
        state.add_position(
            UniversalPosition::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args"),
        );

        // Should NOT be accepting: remaining = 4 - 0 = 4 > (2 - 0) = 2
        assert!(!automaton.is_accepting(&state, 4, 0));
    }

    // =========================================================================
    // Relevant Subword Tests
    // =========================================================================

    #[test]
    fn test_relevant_subword_start() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Position 1, n=2: window is [i-n, v] = [-1, 4] inclusive
        // v = min(|w|, i+n+1) = min(4, 1+2+1) = 4
        // Should be: $, $, w_1, w_2, w_3, w_4 = $$test (6 chars, 2n+2)
        let subword = automaton.relevant_subword("test", 1);
        assert_eq!(subword, "$$test");
    }

    #[test]
    fn test_relevant_subword_middle() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Position 3: window is [1, 6)
        // Should be: w[0], w[1], w[2], w[3]
        let subword = automaton.relevant_subword("test", 3);
        assert_eq!(subword, "test");
    }

    #[test]
    fn test_relevant_subword_end() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Position 4: window is [2, 7) but word ends at 4
        // Should be: w[1], w[2], w[3]
        let subword = automaton.relevant_subword("test", 4);
        assert_eq!(subword, "est");
    }

    #[test]
    fn test_relevant_subword_n1() {
        let automaton = UniversalAutomaton::<Standard>::new(1);
        // Position 2, n=1: window is [i-n, v] = [1, 4] inclusive
        // v = min(|w|, i+n+1) = min(4, 2+1+1) = 4
        // Should be: w_1, w_2, w_3, w_4 = test (4 chars, 2n+2 = 4)
        let subword = automaton.relevant_subword("test", 2);
        assert_eq!(subword, "test");
    }

    #[test]
    fn test_relevant_subword_unicode_character_positions() {
        let automaton = UniversalAutomaton::<Standard>::new(1);
        let subword = automaton.relevant_subword("éaß", 2);
        assert_eq!(subword, "éaß");
    }

    #[test]
    fn relevant_subword_at_saturated_position_is_empty() {
        let automaton = UniversalAutomaton::<Standard>::new(u8::MAX);
        let subword = automaton.relevant_subword("abc", usize::MAX);
        assert_eq!(subword, "");
    }

    // =========================================================================
    // Acceptance Tests
    // =========================================================================

    #[test]
    fn test_accepts_identical() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 0
        assert!(automaton.accepts("test", "test"));
    }

    #[test]
    fn test_accepts_substitution() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 1: one substitution (s → x)
        assert!(automaton.accepts("test", "text"));
    }

    #[test]
    fn test_accepts_insertion() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 1: one insertion (added 'a')
        assert!(automaton.accepts("test", "teast"));
    }

    #[test]
    fn test_accepts_deletion() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 1: one deletion (removed 's')
        assert!(automaton.accepts("test", "tet"));
    }

    #[test]
    fn test_rejects_too_far() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 5: too many differences
        assert!(!automaton.accepts("test", "hello"));
    }

    #[test]
    fn test_accepts_empty_to_empty() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 0
        assert!(automaton.accepts("", ""));
    }

    #[test]
    fn test_accepts_empty_word() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 2: insert two characters
        assert!(automaton.accepts("", "ab"));
    }

    #[test]
    fn test_rejects_empty_word_too_far() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 3: too many insertions
        assert!(!automaton.accepts("", "abc"));
    }

    #[test]
    fn test_accepts_to_empty() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 2: delete two characters
        assert!(automaton.accepts("ab", ""));
    }

    #[test]
    fn test_rejects_to_empty_too_far() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 3: too many deletions
        assert!(!automaton.accepts("abc", ""));
    }

    #[test]
    fn test_accepts_multiple_edits() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 2: two substitutions
        assert!(automaton.accepts("test", "best"));
        assert!(automaton.accepts("test", "tent"));
    }

    #[test]
    fn test_accepts_n1() {
        let automaton = UniversalAutomaton::<Standard>::new(1);
        // Distance 1: one substitution (t→x at position 2)
        assert!(automaton.accepts("test", "text"));
        // Distance 1: one substitution (t→b at position 0)
        assert!(automaton.accepts("test", "best"));
        // Distance 2: should reject (two substitutions needed)
        assert!(!automaton.accepts("test", "bear"));
    }

    #[test]
    fn test_accepts_unicode_by_character_distance() {
        let automaton = UniversalAutomaton::<Standard>::new(1);
        assert!(automaton.accepts("é", ""));
        assert!(automaton.accepts("", "é"));
        assert!(automaton.accepts("café", "cafe"));
        assert!(!automaton.accepts("éø", ""));
    }

    #[test]
    fn test_accepts_longer_words() {
        let automaton = UniversalAutomaton::<Standard>::new(2);
        // Distance 1
        assert!(automaton.accepts("algorithm", "algorythm"));
        // Distance 2
        assert!(automaton.accepts("algorithm", "algarithm"));
    }

    // =========================================================================
    // Transposition Variant Tests
    // =========================================================================

    #[test]
    fn test_transposition_adjacent_swap_start() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Swap first two characters: "test" → "etst"
        assert!(automaton.accepts("test", "etst"));
    }

    #[test]
    fn test_transposition_adjacent_swap_middle() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Swap middle characters: "test" → "tset"
        assert!(automaton.accepts("test", "tset"));
    }

    #[test]
    fn test_transposition_adjacent_swap_end() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Swap last two characters: "test" → "tets"
        assert!(automaton.accepts("test", "tets"));
    }

    #[test]
    fn test_transposition_with_standard_operations() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(2);

        // Transposition + deletion: "test" → "tset" → "set" (distance 2)
        assert!(automaton.accepts("test", "set"));

        // Transposition + insertion: "test" → "tset" → "taset" (distance 2)
        assert!(automaton.accepts("test", "taset"));
    }

    #[test]
    fn test_transposition_longer_words() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // "algorithm" → "lagorithm" (swap 'a' and 'l')
        assert!(automaton.accepts("algorithm", "lagorithm"));

        // "algorithm" → "aglorithm" (swap 'l' and 'g')
        assert!(automaton.accepts("algorithm", "aglorithm"));
    }

    #[test]
    fn test_transposition_rejects_non_adjacent() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Cannot swap non-adjacent chars with distance 1
        // "test" → "stet" requires swapping 't' and 's' (positions 0 and 2)
        // This needs 2 operations, so should reject
        assert!(!automaton.accepts("test", "stet"));
    }

    #[test]
    fn test_transposition_empty_and_single_char() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Empty word
        assert!(automaton.accepts("", ""));

        // Single character - transposition mode still supports standard operations
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "b")); // Accepts via substitution (transposition includes standard ops)
    }

    #[test]
    fn test_transposition_two_chars() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // Two characters - single transposition
        assert!(automaton.accepts("ab", "ba"));
        assert!(automaton.accepts("xy", "yx"));
    }

    #[test]
    fn test_transposition_distance_zero() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(0);

        // Distance 0 - only exact matches
        assert!(automaton.accepts("test", "test"));
        assert!(!automaton.accepts("test", "etst")); // Would need transposition
    }

    #[test]
    fn test_transposition_vs_standard() {
        use crate::transducer::universal::Transposition;

        // With transposition: "test" → "etst" = distance 1
        let trans_automaton = UniversalAutomaton::<Transposition>::new(1);
        assert!(trans_automaton.accepts("test", "etst"));

        // With standard: "test" → "etst" requires 2 operations
        // (delete 't', insert 'e' OR substitute twice)
        let std_automaton = UniversalAutomaton::<Standard>::new(1);
        assert!(!std_automaton.accepts("test", "etst"));
    }

    #[test]
    fn test_transposition_multiple_swaps() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(2);

        // Two transpositions: "abcd" → "bacd" → "badc"
        assert!(automaton.accepts("abcd", "badc"));
    }

    #[test]
    fn test_transposition_with_repeated_chars() {
        use crate::transducer::universal::Transposition;
        let automaton = UniversalAutomaton::<Transposition>::new(1);

        // "abcd" → "bacd" (swap first two adjacent chars)
        assert!(automaton.accepts("abcd", "bacd"));

        // "aabb" → "abab" (swap middle two adjacent chars)
        assert!(automaton.accepts("aabb", "abab"));

        // "aabc" → "aacb" (swap last two adjacent chars)
        assert!(automaton.accepts("aabc", "aacb"));
    }

    // ============================================================================
    // MERGE AND SPLIT TESTS
    // ============================================================================

    #[test]
    fn test_merge_and_split_distance_zero() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(0);

        // Distance 0 should only accept identical strings
        assert!(automaton.accepts("", ""));
        assert!(automaton.accepts("hello", "hello"));
        assert!(!automaton.accepts("hello", "helo"));
        assert!(!automaton.accepts("hello", "helllo"));
    }

    #[test]
    fn test_merge_simple() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge: "ab" → "a" (merge two input chars 'ab' into one word char 'a')
        // Input has "ab", word has "a" - merge consumes 2 input chars for 1 word char
        assert!(automaton.accepts("ab", "a"));

        // Merge at different positions
        assert!(automaton.accepts("abc", "ac")); // merge 'ab' → 'a'
        assert!(automaton.accepts("xab", "xa")); // merge 'ab' → 'a' at end
        assert!(automaton.accepts("xaby", "xay")); // merge 'ab' → 'a' in middle
    }

    #[test]
    fn test_split_simple() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Split: "a" → "ab" (split one input char 'a' into two word chars 'ab')
        // Input has "a", word has "ab" - split expands 1 input char to 2 word chars
        assert!(automaton.accepts("a", "ab"));

        // Split at different positions
        assert!(automaton.accepts("ac", "abc")); // split 'a' → 'ab'
        assert!(automaton.accepts("xa", "xab")); // split 'a' → 'ab' at end
        assert!(automaton.accepts("xay", "xaby")); // split 'a' → 'ab' in middle

        // Additional split tests
        assert!(automaton.accepts("b", "bc")); // split 'b' → 'bc'
        assert!(automaton.accepts("t", "te")); // split 't' → 'te'
    }

    #[test]
    fn test_merge_and_split_longer_words() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge in longer words
        assert!(automaton.accepts("algorithm", "algorihm")); // merge 'it' → 'i'
        assert!(automaton.accepts("banana", "banna")); // merge 'an' → 'n'

        // Split in longer words
        assert!(automaton.accepts("algorithim", "algorithm")); // split 'i' → 'it'
        assert!(automaton.accepts("banna", "banana")); // split 'n' → 'an'
    }

    #[test]
    fn test_merge_and_split_with_standard_operations() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge/split mode should include ALL standard operations

        // Standard insertion
        assert!(automaton.accepts("test", "teest"));

        // Standard deletion
        assert!(automaton.accepts("test", "tst"));

        // Standard substitution
        assert!(automaton.accepts("test", "best"));
    }

    #[test]
    fn test_merge_and_split_empty_and_single_char() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Empty word
        assert!(automaton.accepts("", ""));

        // Single character - merge/split mode still supports standard operations
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "b")); // substitution
        assert!(automaton.accepts("a", "")); // deletion
        assert!(automaton.accepts("", "a")); // insertion

        // Single char to two chars (split)
        assert!(automaton.accepts("a", "ab"));

        // Two chars to one char (merge)
        assert!(automaton.accepts("ab", "a"));
    }

    #[test]
    fn test_merge_at_start() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge at the very start of the word
        assert!(automaton.accepts("abcd", "acd")); // merge 'ab' → 'a'
        assert!(automaton.accepts("test", "est")); // merge 'te' → 'e'
    }

    #[test]
    fn test_merge_at_end() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge at the very end of the word
        assert!(automaton.accepts("test", "tes")); // merge 'st' → 's'
        assert!(automaton.accepts("abcd", "abc")); // merge 'cd' → 'c'
    }

    #[test]
    fn test_split_at_start() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Split at the very start of the word
        assert!(automaton.accepts("acd", "abcd")); // split 'a' → 'ab'
        assert!(automaton.accepts("est", "test")); // split 'e' → 'te'
    }

    #[test]
    fn test_split_at_end() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Split at the very end of the word
        assert!(automaton.accepts("tes", "test")); // split 's' → 'st'
        assert!(automaton.accepts("abc", "abcd")); // split 'c' → 'cd'
    }

    #[test]
    fn test_merge_and_split_multiple_operations() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(2);

        // Multiple merge operations
        assert!(automaton.accepts("abcd", "ac")); // merge 'ab' → 'a', merge 'cd' → 'c'

        // Multiple split operations
        assert!(automaton.accepts("ac", "abcd")); // split 'a' → 'ab', split 'c' → 'cd'

        // Mix of operations
        assert!(automaton.accepts("abc", "abbc")); // split 'b' → 'bb'
        assert!(automaton.accepts("abbc", "abc")); // merge 'bb' → 'b'
    }

    #[test]
    fn test_merge_and_split_vs_standard() {
        use crate::transducer::universal::{MergeAndSplit, Standard};
        let merge_split_automaton = UniversalAutomaton::<MergeAndSplit>::new(1);
        let standard_automaton = UniversalAutomaton::<Standard>::new(1);

        // Standard operations should work in both
        assert_eq!(
            standard_automaton.accepts("test", "best"),
            merge_split_automaton.accepts("test", "best")
        );
        assert_eq!(
            standard_automaton.accepts("test", "tst"),
            merge_split_automaton.accepts("test", "tst")
        );

        // Verify that merge/split automaton does support these operations
        // Note: We can't easily demonstrate that standard DOESN'T support merge/split
        // because "ab" → "a" can be achieved with deletion, and "a" → "ab" with insertion.
        // The key difference is efficiency: merge/split does it in 1 operation,
        // while standard needs 2 operations.

        // With distance 1, merge/split can do:
        assert!(merge_split_automaton.accepts("ab", "a")); // merge in 1 op
        assert!(merge_split_automaton.accepts("a", "ab")); // split in 1 op
        assert!(merge_split_automaton.accepts("abc", "ac")); // merge 'ab' → 'a'
        assert!(merge_split_automaton.accepts("ac", "abc")); // split 'a' → 'ab'

        // Standard automaton can also accept these, but via different paths
        // (e.g., deletion + substitution for "ab" → "a", insertion + substitution for "a" → "ab")
        // So these tests just verify merge/split works correctly
    }

    #[test]
    fn test_merge_and_split_with_repeated_chars() {
        use crate::transducer::universal::MergeAndSplit;
        let automaton = UniversalAutomaton::<MergeAndSplit>::new(1);

        // Merge with repeated characters
        assert!(automaton.accepts("aab", "ab")); // merge 'aa' → 'a'
        assert!(automaton.accepts("aabb", "abb")); // merge 'aa' → 'a'
        assert!(automaton.accepts("abbb", "abb")); // merge 'bb' → 'b'

        // Split with repeated characters
        assert!(automaton.accepts("ab", "aab")); // split 'a' → 'aa'
        assert!(automaton.accepts("abb", "aabb")); // split 'a' → 'aa'
        assert!(automaton.accepts("abb", "abbb")); // split 'b' → 'bb'
    }
}
