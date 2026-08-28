//! Exact runtime-configurable generalized alignment.
//!
//! [`GeneralizedAutomaton`] decides whether an `OperationSet` alignment fits an
//! integer budget. Decimal operation weights are converted through one exact
//! [`CostScale`], and a sparse topological alignment graph evaluates every
//! configured source/target consumption rule.
//!
//! # Design Philosophy
//!
//! `UniversalAutomaton` remains the compile-time specialized choice for its
//! fixed variants. This engine is the correctness oracle for runtime presets,
//! phonetic restrictions, fractional weights, and multi-scalar operations.
//!
//! # Operation Support
//!
//! The public Boolean API fails closed on invalid cost domains, arithmetic
//! overflow, or the alignment-state resource ceiling. Use
//! [`GeneralizedAutomaton::try_accepts`] to distinguish those errors.
//!
//! # Theory Background
//!
//! ## Acceptance condition
//!
//! An alignment cell `(i, j)` has consumed `i` source scalars and `j` target
//! scalars. An operation advances by its declared pair and adds its exact
//! scaled cost. Acceptance means reaching both string ends within budget.
//!
//! # Examples
//!
//! ```rust
//! use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
//!
//! // Create automaton for maximum distance n=2
//! let automaton = GeneralizedAutomaton::new(2);
//!
//! // Check if "test" accepts "text" (distance 1)
//! assert!(automaton.accepts("test", "text"));
//!
//! // Check if "test" accepts "hello" (distance > 2)
//! assert!(!automaton.accepts("test", "hello"));
//! ```

use crate::cost::{CostScale, ScaleError};
use crate::transducer::{OperationSet, OperationSetValidationError, OperationType};
use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt;

/// Maximum number of reachable alignment cells explored by one generalized query.
pub const MAX_GENERALIZED_ALIGNMENT_STATES: usize = 1_000_000;

/// Failure while preparing or evaluating a generalized automaton.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeneralizedAutomatonError {
    /// The complete operation set violates a structural or resource invariant.
    OperationSet(OperationSetValidationError),
    /// An operation weight cannot be represented by the exact scale.
    Scale(ScaleError),
    /// Checked alignment arithmetic overflowed.
    ArithmeticOverflow,
    /// The reachable alignment graph exceeded the public resource ceiling.
    ResourceLimit {
        /// Number of states observed when evaluation stopped.
        observed: usize,
        /// Configured hard ceiling.
        limit: usize,
    },
}

impl fmt::Display for GeneralizedAutomatonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OperationSet(error) => write!(f, "invalid generalized operation set: {error}"),
            Self::Scale(error) => write!(f, "invalid generalized-operation cost: {error}"),
            Self::ArithmeticOverflow => f.write_str("generalized-alignment arithmetic overflowed"),
            Self::ResourceLimit { observed, limit } => write!(
                f,
                "generalized alignment reached {observed} states (limit {limit})"
            ),
        }
    }
}

impl std::error::Error for GeneralizedAutomatonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OperationSet(error) => Some(error),
            Self::Scale(error) => Some(error),
            Self::ArithmeticOverflow | Self::ResourceLimit { .. } => None,
        }
    }
}

impl From<OperationSetValidationError> for GeneralizedAutomatonError {
    fn from(value: OperationSetValidationError) -> Self {
        Self::OperationSet(value)
    }
}

impl From<ScaleError> for GeneralizedAutomatonError {
    fn from(error: ScaleError) -> Self {
        Self::Scale(error)
    }
}

/// Exact runtime-configurable generalized alignment engine.
///
/// Supported operation sets include:
/// - Standard operations (match, substitute, insert, delete)
/// - Adjacent transposition
/// - Multi-character operations (phonetic corrections, merge/split)
/// - Exact decimal-weighted operations
///
/// # Examples
///
/// ```rust
/// use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
/// use liblevenshtein::transducer::OperationSet;
///
/// // Standard Levenshtein
/// let automaton = GeneralizedAutomaton::new(2);
///
/// // With transposition
/// let automaton = GeneralizedAutomaton::with_operations(
///     2,
///     OperationSet::with_transposition()
/// );
///
/// // Check if "test" accepts "text" (distance 1)
/// assert!(automaton.accepts("test", "text"));
/// ```
#[derive(Debug, Clone)]
pub struct GeneralizedAutomaton {
    /// Maximum edit distance n
    max_distance: u8,

    /// Set of operations defining the edit distance metric
    operations: OperationSet,

    /// Exact decimal scale shared by every operation and the public budget.
    cost_scale: Result<CostScale, ScaleError>,
}

impl GeneralizedAutomaton {
    /// Create a new Generalized Levenshtein Automaton with standard operations
    ///
    /// Uses the standard operation set (match, substitute, insert, delete).
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n (typically 1, 2, or 3)
    ///
    /// # Returns
    ///
    /// A new `GeneralizedAutomaton` instance with standard operations
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let automaton = GeneralizedAutomaton::new(2);
    /// ```
    #[must_use]
    pub fn new(max_distance: u8) -> Self {
        let operations = OperationSet::standard();
        let cost_scale = CostScale::for_operations(&operations);
        Self {
            max_distance,
            operations,
            cost_scale,
        }
    }

    /// Create a new Generalized Levenshtein Automaton with custom operations
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n (typically 1, 2, or 3)
    /// - `operations`: Set of operations defining the edit distance metric
    ///
    /// # Returns
    ///
    /// A new `GeneralizedAutomaton` instance with custom operations
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Automaton with transposition support
    /// let automaton = GeneralizedAutomaton::with_operations(
    ///     2,
    ///     OperationSet::with_transposition()
    /// );
    ///
    /// // Automaton with custom phonetic operations
    /// use liblevenshtein::transducer::{OperationSetBuilder, OperationType, SubstitutionSet};
    ///
    /// let mut phonetic = SubstitutionSet::new();
    /// phonetic.allow_str("ph", "f");
    /// let ops = OperationSetBuilder::new()
    ///     .with_standard_ops()
    ///     .with_operation(OperationType::with_restriction(
    ///         2, 1, 0.15, phonetic, "ph_to_f"
    ///     ))
    ///     .build();
    /// let automaton = GeneralizedAutomaton::with_operations(2, ops);
    /// ```
    #[must_use]
    pub fn with_operations(max_distance: u8, operations: OperationSet) -> Self {
        let cost_scale = CostScale::for_operations(&operations);
        Self {
            max_distance,
            operations,
            cost_scale,
        }
    }

    /// Create an automaton, rejecting an invalid operation-cost domain eagerly.
    pub fn try_with_operations(
        max_distance: u8,
        operations: OperationSet,
    ) -> Result<Self, GeneralizedAutomatonError> {
        operations.validate()?;
        let cost_scale = CostScale::for_operations(&operations)?;
        Ok(Self {
            max_distance,
            operations,
            cost_scale: Ok(cost_scale),
        })
    }

    /// Get the maximum edit distance n
    #[must_use]
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Return the exact scale derived from the complete operation set.
    pub fn cost_scale(&self) -> Result<CostScale, GeneralizedAutomatonError> {
        self.cost_scale.clone().map_err(Into::into)
    }

    /// Check whether an exact configured-operation alignment fits the budget.
    ///
    /// # Arguments
    ///
    /// - `word`: Dictionary word w
    /// - `input`: Input string x to match against
    ///
    /// # Returns
    ///
    /// `true` if the least configured-operation cost is at most the integer
    /// budget. Invalid costs, overflow, and resource exhaustion fail closed.
    ///
    /// # Algorithm
    ///
    /// 1. Derive one exact decimal scale for the operation set.
    /// 2. Traverse reachable alignment cells in topological order.
    /// 3. Relax every applicable operation whose exact cost remains in budget.
    /// 4. Accept only when both strings are completely consumed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let automaton = GeneralizedAutomaton::new(2);
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
    pub fn accepts(&self, word: &str, input: &str) -> bool {
        self.try_accepts(word, input).unwrap_or(false)
    }

    /// Fallible acceptance with exact operation-driven costs.
    ///
    /// Evaluation is a shortest-path computation over the acyclic alignment
    /// grid. Only cells reachable within the scaled budget are materialized.
    pub fn try_accepts(&self, word: &str, input: &str) -> Result<bool, GeneralizedAutomatonError> {
        Ok(self.scaled_distance(word, input)?.is_some())
    }

    /// Return the least exact scaled cost when it is inside the configured budget.
    pub fn scaled_distance(
        &self,
        word: &str,
        input: &str,
    ) -> Result<Option<usize>, GeneralizedAutomatonError> {
        self.scaled_distance_with_limit(word, input, MAX_GENERALIZED_ALIGNMENT_STATES)
    }

    fn scaled_distance_with_limit(
        &self,
        word: &str,
        input: &str,
        state_limit: usize,
    ) -> Result<Option<usize>, GeneralizedAutomatonError> {
        self.operations.validate()?;
        let scale = self.cost_scale()?;
        let budget = scale.scale_budget(self.max_distance)?;
        let word_offsets = char_byte_offsets(word);
        let input_offsets = char_byte_offsets(input);
        let word_len = word_offsets.len() - 1;
        let input_len = input_offsets.len() - 1;

        let weighted_operations = self
            .operations
            .iter()
            .map(|operation| Ok((operation, scale.to_scaled(operation.weight())?)))
            .collect::<Result<Vec<_>, ScaleError>>()?;

        let mut pending = BTreeMap::new();
        pending.insert((0_usize, 0_usize), 0_usize);
        let mut discovered = 1_usize;
        if discovered > state_limit {
            return Err(GeneralizedAutomatonError::ResourceLimit {
                observed: discovered,
                limit: state_limit,
            });
        }

        while let Some(((word_pos, input_pos), accumulated)) = pending.pop_first() {
            if word_pos == word_len && input_pos == input_len {
                return Ok(Some(accumulated));
            }

            for (operation, step) in &weighted_operations {
                if operation.consume_x() == 0 && operation.consume_y() == 0 {
                    continue;
                }
                let Some(next_word) = word_pos.checked_add(operation.consume_x()) else {
                    continue;
                };
                let Some(next_input) = input_pos.checked_add(operation.consume_y()) else {
                    continue;
                };
                if next_word > word_len || next_input > input_len {
                    continue;
                }

                let word_slice = &word[word_offsets[word_pos]..word_offsets[next_word]];
                let input_slice = &input[input_offsets[input_pos]..input_offsets[next_input]];
                if !operation_applies(operation, word_slice, input_slice) {
                    continue;
                }
                let Some(next_cost) = accumulated.checked_add(*step) else {
                    continue;
                };
                if next_cost > budget {
                    continue;
                }

                match pending.entry((next_word, next_input)) {
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() = (*entry.get()).min(next_cost);
                    }
                    Entry::Vacant(entry) => {
                        discovered = discovered
                            .checked_add(1)
                            .ok_or(GeneralizedAutomatonError::ArithmeticOverflow)?;
                        if discovered > state_limit {
                            return Err(GeneralizedAutomatonError::ResourceLimit {
                                observed: discovered,
                                limit: state_limit,
                            });
                        }
                        entry.insert(next_cost);
                    }
                }
            }
        }

        Ok(None)
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
}

fn char_byte_offsets(value: &str) -> Vec<usize> {
    let mut offsets = value
        .char_indices()
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    offsets.push(value.len());
    offsets
}

fn operation_applies(operation: &OperationType, word: &str, input: &str) -> bool {
    operation.applies_to_slices(word, input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let automaton = GeneralizedAutomaton::new(2);
        assert_eq!(automaton.max_distance(), 2);
    }

    #[test]
    fn reachable_alignment_state_ceiling_fails_closed() {
        let automaton = GeneralizedAutomaton::new(0);
        assert_eq!(
            automaton.scaled_distance_with_limit("a", "a", 1),
            Err(GeneralizedAutomatonError::ResourceLimit {
                observed: 2,
                limit: 1,
            })
        );
    }

    #[test]
    fn alignment_state_ceiling_counts_the_initial_materialized_cell() {
        let automaton = GeneralizedAutomaton::new(0);
        assert_eq!(
            automaton.scaled_distance_with_limit("", "", 0),
            Err(GeneralizedAutomatonError::ResourceLimit {
                observed: 1,
                limit: 0,
            })
        );
    }

    #[test]
    fn test_relevant_subword_unicode_character_positions() {
        let automaton = GeneralizedAutomaton::new(1);
        let subword = automaton.relevant_subword("éaß", 2);
        assert_eq!(subword, "éaß");
    }

    #[test]
    fn relevant_subword_at_saturated_position_is_empty() {
        let automaton = GeneralizedAutomaton::new(u8::MAX);
        let subword = automaton.relevant_subword("abc", usize::MAX);
        assert_eq!(subword, "");
    }

    #[test]
    fn test_accepts_identical() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("test", "test"));
        assert!(automaton.accepts("", ""));
        assert!(automaton.accepts("hello", "hello"));
    }

    #[test]
    fn test_accepts_one_substitution() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("test", "text")); // t->x
        assert!(automaton.accepts("hello", "hallo")); // e->a
    }

    #[test]
    fn test_accepts_one_insertion() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("test", "tests")); // +s at end
        assert!(automaton.accepts("test", "ttest")); // +t at start
    }

    #[test]
    fn test_accepts_one_deletion() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("tests", "test")); // -s at end
        assert!(automaton.accepts("ttest", "test")); // -t at start
    }

    #[test]
    fn test_rejects_too_far() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(!automaton.accepts("test", "hello")); // distance 4
        assert!(!automaton.accepts("abc", "xyz")); // distance 3
    }

    #[test]
    fn test_empty_input() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("", "")); // distance 0
        assert!(automaton.accepts("ab", "")); // distance 2
        assert!(!automaton.accepts("abc", "")); // distance 3 > 2
    }

    #[test]
    fn test_empty_word() {
        let automaton = GeneralizedAutomaton::new(2);
        assert!(automaton.accepts("", "ab")); // distance 2
        assert!(!automaton.accepts("", "abc")); // distance 3 > 2
    }

    #[test]
    fn test_max_distance_zero() {
        let automaton = GeneralizedAutomaton::new(0);
        assert!(automaton.accepts("test", "test")); // exact match
        assert!(!automaton.accepts("test", "text")); // any difference rejected
    }

    #[test]
    fn test_max_distance_one() {
        let automaton = GeneralizedAutomaton::new(1);
        assert!(automaton.accepts("test", "text")); // 1 substitution
        assert!(automaton.accepts("test", "tst")); // 1 deletion
        assert!(!automaton.accepts("test", "tx")); // distance 2
    }

    #[test]
    fn test_accepts_unicode_by_character_distance() {
        let automaton = GeneralizedAutomaton::new(1);
        assert!(automaton.accepts("é", ""));
        assert!(automaton.accepts("", "é"));
        assert!(automaton.accepts("café", "cafe"));
        assert!(!automaton.accepts("éø", ""));
    }

    // Phase 2d.3: Transposition tests

    #[test]
    fn test_transposition_distance_zero() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(0, ops);

        assert!(automaton.accepts("test", "test")); // Exact match
        assert!(!automaton.accepts("test", "tset")); // Requires 1 error
        assert!(!automaton.accepts("test", "etst")); // Requires 1 error
    }

    #[test]
    fn test_transposition_adjacent_swap_middle() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops.clone());

        // "test" → "tset" (swap 'e' and 's')
        assert!(automaton.accepts("test", "tset"));
    }

    #[test]
    fn test_transposition_adjacent_swap_start() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "test" → "etst" (swap 't' and 'e')
        assert!(automaton.accepts("test", "etst"));
    }

    #[test]
    fn test_transposition_adjacent_swap_end() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "test" → "tets" (swap 's' and 't')
        assert!(automaton.accepts("test", "tets"));
    }

    #[test]
    fn test_transposition_longer_words() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "algorithm" → "lagorithm" (swap 'a' and 'l')
        assert!(automaton.accepts("algorithm", "lagorithm"));

        // "programming" → "porgramming" (swap 'r' and 'o')
        assert!(automaton.accepts("programming", "porgramming"));
    }

    #[test]
    fn test_transposition_rejects_non_adjacent() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "test" → "tsta" (non-adjacent swap) requires 2 errors
        // (Cannot swap 'e' and 's' if they're not adjacent)
        assert!(!automaton.accepts("test", "tsta"));

        // "abc" → "cba" (swap non-adjacent) requires 2 errors
        assert!(!automaton.accepts("abc", "cba"));
    }

    #[test]
    fn test_transposition_multiple_swaps() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // "abcd" → "badc" (two adjacent swaps)
        assert!(automaton.accepts("abcd", "badc"));

        // "test" → "etst" → "etts" (two swaps)
        assert!(automaton.accepts("test", "etts"));
    }

    #[test]
    fn test_transposition_with_standard_operations() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Combine transposition with substitution
        // "test" → "txst" (substitute e→x) → "tsxt" (transpose)
        assert!(automaton.accepts("test", "tsxt"));

        // Transpose + deletion
        // "test" → "tset" (transpose) → "set" (delete 't')
        assert!(automaton.accepts("test", "set"));
    }

    #[test]
    fn test_transposition_empty_and_single() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops.clone());

        // Empty strings
        assert!(automaton.accepts("", ""));

        // Single character (no transposition possible, but substitution works)
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "b")); // Accepted via substitution (not transposition)

        // Verify distance 0 rejects difference
        let strict_automaton = GeneralizedAutomaton::with_operations(0, ops);
        assert!(!strict_automaton.accepts("a", "b")); // No errors allowed
    }

    #[test]
    fn test_transposition_two_char_words() {
        let ops = crate::transducer::OperationSet::with_transposition();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Two characters - simple swap
        assert!(automaton.accepts("ab", "ba"));
        assert!(automaton.accepts("xy", "yx"));

        // Should also work with identical chars
        assert!(automaton.accepts("aa", "aa"));
    }

    // Phase 2d.4: Merge tests

    #[test]
    fn test_merge_simple() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "a" → "ab" (merge two input chars "ab" into one word char "a")
        // Note: Direction is word → input, so we're checking if input "ab" matches word "a"
        assert!(automaton.accepts("a", "ab"));
    }

    #[test]
    fn test_merge_at_start() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "ac" → "abc" (merge "ab" into "a", then match "c")
        assert!(automaton.accepts("ac", "abc"));

        // "test" → "teest" (merge "te" into "t", then match "est")
        assert!(automaton.accepts("test", "teest"));
    }

    #[test]
    fn test_merge_at_end() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "xa" → "xab" (match "x", merge "ab" into "a")
        assert!(automaton.accepts("xa", "xab"));

        // "testa" → "testab" (match "test", merge "ab" into "a")
        assert!(automaton.accepts("testa", "testab"));
    }

    #[test]
    fn test_merge_middle() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "cat" → "cabt" (match "c", merge "ab" into "a", match "t")
        assert!(automaton.accepts("cat", "cabt"));
    }

    #[test]
    fn test_merge_distance_zero() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(0, ops);

        // Distance 0: no merge allowed
        assert!(automaton.accepts("test", "test"));
        assert!(!automaton.accepts("test", "teest"));
        assert!(!automaton.accepts("a", "ab"));
    }

    #[test]
    fn test_merge_with_standard_operations() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Merge + substitution
        // "test" → "texst" (merge "te" into "t", substitute e→x, match "st")
        assert!(automaton.accepts("test", "texst"));

        // Merge + deletion
        // "test" → "teest" (merge) → "eest" (delete first t)
        assert!(automaton.accepts("test", "eest"));
    }

    #[test]
    fn test_merge_empty_and_edge_cases() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Empty strings
        assert!(automaton.accepts("", ""));

        // Single character (merge requires 2 input chars)
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "ab")); // Merge "ab" → "a"

        // Two char word
        assert!(automaton.accepts("ab", "ab"));
        assert!(automaton.accepts("ab", "abb")); // Merge "ab" → "a", match "b"
    }

    // Phase 2d.5: Split tests

    #[test]
    fn test_split_simple() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "ab" → "a" (split one input char "a" into two word chars "ab")
        // Note: Direction is word → input, so we're checking if input "a" matches word "ab"
        assert!(automaton.accepts("ab", "a"));
    }

    #[test]
    fn test_split_at_start() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "abc" → "ac" (split "a" into "ab", then match "c")
        assert!(automaton.accepts("abc", "ac"));

        // "acd" → "acd" or "acd" → "ad" (split "a" into "ab", match "c", delete "d")
        assert!(automaton.accepts("abcd", "acd"));
    }

    #[test]
    fn test_split_at_end() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "xab" → "xa" (match "x", split "a" into "ab")
        assert!(automaton.accepts("xab", "xa"));

        // "testab" → "testa" (match "test", split "a" into "ab")
        assert!(automaton.accepts("testab", "testa"));
    }

    #[test]
    fn test_split_middle() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "cat" → "caat" (match 'c', split word 'a' into input "aa", match 't')
        assert!(automaton.accepts("cat", "caat"));
    }

    #[test]
    fn test_split_distance_zero() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(0, ops);

        // Distance 0: no split allowed
        assert!(automaton.accepts("test", "test"));
        assert!(!automaton.accepts("ttest", "test"));
        assert!(!automaton.accepts("ab", "a"));
    }

    #[test]
    fn test_split_with_standard_operations() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Split + substitution
        // "test" → "txst" (split "t" into "te", substitute e→x, match "st")
        assert!(automaton.accepts("test", "txst"));

        // Split + deletion
        // "test" → "ttest" (split) → "test" (delete extra t)
        assert!(automaton.accepts("test", "test"));
    }

    #[test]
    fn test_split_empty_and_edge_cases() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Empty strings
        assert!(automaton.accepts("", ""));

        // Single character (split produces 2 chars from 1)
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("ab", "a")); // Split "a" → "ab"

        // Two char word
        assert!(automaton.accepts("ab", "ab"));
        assert!(automaton.accepts("abb", "ab")); // Split "a" → "ab", match "b"
    }

    #[test]
    fn test_split_and_merge_combined() {
        let ops = crate::transducer::OperationSet::with_merge_split();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Both split and merge in same string
        // "abc" → "ac" (split "a" → "ab") → "abc" (merge "bc" → "c")
        assert!(automaton.accepts("abc", "ac"));

        // Complex: "test" → "ttest" (split "t") → "test" (merge "tt" → "t")
        assert!(automaton.accepts("test", "test"));
    }

    // ============================================================================
    // Phase 2d.6: Integration Tests - Mixed Operations and Complex Cases
    // ============================================================================

    #[test]
    fn test_all_multichar_operations_combined() {
        // Create operation set with all multi-character operations
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(3, ops);

        // Combines transpose, merge, and split
        // "abc" → "abbc" (split 'b') → "acbc" (transpose "bb" → "cb")
        assert!(automaton.accepts("abc", "acbc"));

        // Complex transformation using all operation types
        // "hello" → "heello" (split 'e') → "hello" (merge "ee" → "e")
        assert!(automaton.accepts("hello", "hello"));
    }

    #[test]
    fn test_multichar_with_distance_constraints() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        // Distance 1: only one multi-char operation allowed
        let automaton1 = GeneralizedAutomaton::with_operations(1, ops.clone());
        assert!(automaton1.accepts("ab", "ba")); // transpose
        assert!(automaton1.accepts("a", "aa")); // split
        assert!(!automaton1.accepts("ab", "bba")); // would need transpose + split (2 errors)

        // Distance 2: two operations allowed
        let automaton2 = GeneralizedAutomaton::with_operations(2, ops.clone());
        assert!(automaton2.accepts("abc", "baca")); // transpose + split

        // Distance 3: three operations allowed
        let automaton3 = GeneralizedAutomaton::with_operations(3, ops);
        assert!(automaton3.accepts("abc", "bbcaa")); // multiple splits and transpose
    }

    #[test]
    fn test_multichar_operations_at_string_boundaries() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Transpose at start
        assert!(automaton.accepts("ab", "ba"));

        // Transpose at end
        assert!(automaton.accepts("cab", "cba"));

        // Split at start
        assert!(automaton.accepts("abc", "aabc"));

        // Split at end
        assert!(automaton.accepts("abc", "abcc"));

        // Merge at start (input has merge at start)
        assert!(automaton.accepts("abc", "bc")); // delete or merge "ab" → "b" + keep c

        // Merge at end
        assert!(automaton.accepts("abc", "ab")); // delete 'c' or merge "bc" → "b"
    }

    #[test]
    fn test_repeated_multichar_operations() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        // Distance 2: two transposes
        let automaton2 = GeneralizedAutomaton::with_operations(2, ops.clone());
        assert!(automaton2.accepts("abcd", "badc")); // transpose "ab" and "cd"

        // Distance 2: two splits
        assert!(automaton2.accepts("ab", "aabb")); // split 'a' and 'b'

        // Distance 3: three operations
        let automaton3 = GeneralizedAutomaton::with_operations(3, ops);
        assert!(automaton3.accepts("abc", "aabbcc")); // split all three chars
    }

    #[test]
    fn test_multichar_with_standard_operations_complex() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(3, ops);

        // Transpose + insert + delete
        assert!(automaton.accepts("abc", "bac")); // transpose (1 error)
        assert!(automaton.accepts("abc", "bacd")); // transpose + insert (2 errors)
        assert!(automaton.accepts("abcd", "bac")); // transpose + delete (2 errors)

        // Split + substitute
        assert!(automaton.accepts("abc", "aabc")); // split 'a' (1 error)
        assert!(automaton.accepts("abc", "aabx")); // split 'a' + substitute c→x (2 errors)

        // Merge + insert
        assert!(automaton.accepts("abc", "bc")); // delete 'a' (1 error)
        assert!(automaton.accepts("abc", "bcd")); // delete 'a' + insert 'd' (2 errors)
    }

    #[test]
    fn test_multichar_edge_cases() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Empty strings (already covered but let's be explicit)
        assert!(automaton.accepts("", ""));
        assert!(automaton.accepts("a", "")); // delete
        assert!(automaton.accepts("", "a")); // insert

        // Single character (no transpose possible)
        assert!(automaton.accepts("a", "a"));
        assert!(automaton.accepts("a", "aa")); // split

        // Two characters (minimal transpose)
        assert!(automaton.accepts("ab", "ba")); // transpose
        assert!(automaton.accepts("ab", "aab")); // split first
        assert!(automaton.accepts("ab", "abb")); // split second

        // Identical strings (no operations needed)
        assert!(automaton.accepts("test", "test"));
        assert!(automaton.accepts("hello", "hello"));
    }

    #[test]
    fn test_multichar_pathological_cases() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(5, ops);

        // All same character
        assert!(automaton.accepts("aaaa", "aaaa"));
        assert!(automaton.accepts("aaaa", "aaaaaaaa")); // split all (4 errors)
        assert!(automaton.accepts("aaaa", "aa")); // merge/delete (2 errors)

        // Alternating pattern
        assert!(automaton.accepts("abab", "baba")); // transpose pairs
        assert!(automaton.accepts("abab", "aabbab")); // split 'a's

        // Reversed string (requires multiple transposes)
        assert!(automaton.accepts("abc", "cba")); // transpose + transpose
    }

    #[test]
    fn test_multichar_operations_respect_invariants() {
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        // Distance 0: no operations allowed
        let automaton0 = GeneralizedAutomaton::with_operations(0, ops.clone());
        assert!(automaton0.accepts("test", "test"));
        assert!(!automaton0.accepts("test", "tset")); // no transpose
        assert!(!automaton0.accepts("test", "teest")); // no split
        assert!(!automaton0.accepts("test", "tes")); // no delete/merge

        // Distance 1: exactly one operation
        let automaton1 = GeneralizedAutomaton::with_operations(1, ops.clone());
        assert!(automaton1.accepts("test", "tset")); // transpose
        assert!(automaton1.accepts("test", "teest")); // split
        assert!(!automaton1.accepts("test", "tseest")); // transpose + split (2 errors)

        // Verify operations don't succeed beyond max distance
        assert!(!automaton1.accepts("abc", "aabbcc")); // would need 3 splits
    }

    #[test]
    fn test_multichar_subsumption_correctness() {
        // This test verifies that subsumption works correctly with intermediate states
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // These should all work and produce minimal state sets
        assert!(automaton.accepts("ab", "ba")); // transpose
        assert!(automaton.accepts("abc", "bac")); // transpose
        assert!(automaton.accepts("test", "tset")); // transpose

        // Verify split subsumption
        assert!(automaton.accepts("a", "aa")); // split
        assert!(automaton.accepts("ab", "aab")); // split first
        assert!(automaton.accepts("ab", "aabb")); // split both

        // Complex cases that test subsumption during multi-char operations
        assert!(automaton.accepts("abcd", "bacd")); // transpose at start
        assert!(automaton.accepts("abcd", "abdc")); // transpose at end
    }

    #[test]
    fn test_multichar_operation_ordering() {
        // Verify that different orderings of operations work correctly
        let ops = crate::transducer::OperationSetBuilder::new()
            .with_standard_ops()
            .with_transposition()
            .with_merge()
            .with_split()
            .build();

        let automaton = GeneralizedAutomaton::with_operations(3, ops);

        // transpose then split
        assert!(automaton.accepts("abc", "baac")); // transpose "ab", then split 'a'

        // "ab" → "ba" (transpose) → "baa" (split second 'a')
        assert!(automaton.accepts("ab", "baa"));

        // Historical mixed-operation acceptance case.
        assert!(automaton.accepts("abc", "abba"));

        // The point is: regardless of the order operations are discovered,
        // the same transformations should be accepted
        assert!(automaton.accepts("abc", "aabc")); // split or insert
        assert!(automaton.accepts("abc", "bac")); // transpose
        assert!(automaton.accepts("abc", "ab")); // delete
    }

    // ============================================================================
    // Phase 3: Phonetic Operations Tests
    // ============================================================================

    #[test]
    fn test_phonetic_simple_digraph_accepts() {
        // Simple phonetic operation smoke test.
        let ops = crate::transducer::phonetic::consonant_digraphs();
        assert!(!ops.operations().is_empty());

        let automaton = GeneralizedAutomaton::with_operations(1, ops);
        let subword = automaton.relevant_subword("ph", 1);

        assert!(subword.contains("ph") || subword.contains('p'));
        assert!(
            automaton.accepts("ph", "f"),
            "Expected 'ph' → 'f' to be accepted"
        );
    }

    #[test]
    fn test_phonetic_digraph_2to1_ch_to_k() {
        // Test phonetic operation: "ch" → "k" (⟨2,1,0.15⟩)
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "church" can match "kurk" via "ch"→"k" digraph operations
        assert!(automaton.accepts("church", "kurk"));

        // "chair" can match "kair" via "ch"→"k"
        assert!(automaton.accepts("chair", "kair"));

        // Distance 0: exact match only (with standard match operation)
        let ops0 = crate::transducer::OperationSet::default();
        let automaton0 = GeneralizedAutomaton::with_operations(0, ops0);
        assert!(automaton0.accepts("church", "church")); // exact match
        assert!(!automaton0.accepts("church", "kurk")); // requires phonetic ops
    }

    #[test]
    fn test_phonetic_digraph_2to1_ph_to_f() {
        // Test phonetic operation: "ph" → "f" (⟨2,1,0.15⟩)
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "phone" can match "fone" via "ph"→"f"
        assert!(automaton.accepts("phone", "fone"));

        // "graph" can match "graf" via "ph"→"f"
        assert!(automaton.accepts("graph", "graf"));
    }

    #[test]
    fn test_phonetic_digraph_2to1_sh_to_s() {
        // Test phonetic operation: "sh" → "s" (⟨2,1,0.15⟩)
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "ship" can match "sip" via "sh"→"s"
        assert!(automaton.accepts("ship", "sip"));

        // "wash" can match "was" via "sh"→"s"
        assert!(automaton.accepts("wash", "was"));
    }

    #[test]
    fn test_phonetic_digraph_2to1_th_to_t() {
        // Test phonetic operation: "th" → "t" (⟨2,1,0.15⟩)
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "think" can match "tink" via "th"→"t"
        assert!(automaton.accepts("think", "tink"));

        // "bath" can match "bat" via "th"→"t"
        assert!(automaton.accepts("bath", "bat"));
    }

    #[test]
    fn test_phonetic_digraph_multiple_in_word() {
        // Test multiple phonetic operations in same word
        // Need standard ops for matching non-phonetic characters
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // "church" has two "ch" digraphs, both can convert: "kurk"
        // But we only have distance 1 per "ch"→"k", so we need distance 2
        assert!(automaton.accepts("church", "kurc")); // one ch→k
        assert!(automaton.accepts("church", "churk")); // one ch→k
    }

    #[test]
    fn test_phonetic_with_standard_ops() {
        // Test phonetic operations combined with standard operations
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();

        // Add all phonetic operations
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // "phone" → "fone" (ph→f) + extra char
        assert!(automaton.accepts("phone", "fones")); // ph→f + insert 's'

        // "chair" → "kair" (ch→k) + substitute
        assert!(automaton.accepts("chair", "kair")); // ch→k
        assert!(automaton.accepts("chair", "kair")); // ch→k
    }

    #[test]
    fn test_phonetic_distance_constraints() {
        // Verify phonetic operations respect distance limits
        // Need standard ops for matching non-phonetic characters and deletions
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        // Distance 1: allows one phonetic operation
        let automaton1 = GeneralizedAutomaton::with_operations(1, ops.clone());
        assert!(automaton1.accepts("phone", "fone")); // one ph→f
        assert!(!automaton1.accepts("phone", "fo")); // would need ph→f + delete (2 ops)

        // The exact cost is 0.15 + 2×1.0 = 2.15, so an integer budget of 2
        // still rejects and budget 3 accepts.
        let automaton2 = GeneralizedAutomaton::with_operations(2, ops.clone());
        assert!(!automaton2.accepts("phone", "fo"));
        let automaton3 = GeneralizedAutomaton::with_operations(3, ops);
        assert!(automaton3.accepts("phone", "fo"));
    }

    // ==================== Cross-Validation Tests ====================
    // Compare Generalized Automaton with Universal Automaton
    // to ensure correctness of phonetic operations

    #[test]
    fn test_cross_validate_standard_operations() {
        // Test that generalized automaton matches universal automaton
        // for standard Levenshtein operations
        use crate::transducer::universal::Standard;
        use crate::transducer::universal::UniversalAutomaton;

        let test_cases = vec![
            ("kitten", "sitting", 3, true),
            ("kitten", "sitting", 2, false),
            ("saturday", "sunday", 3, true),
            ("saturday", "sunday", 2, false),
            ("test", "test", 0, true),
            ("test", "tast", 1, true),
            ("", "", 0, true),
            ("a", "b", 1, true),
            ("abc", "def", 3, true),
        ];

        for (word, input, distance, expected) in test_cases {
            let gen_auto = GeneralizedAutomaton::new(distance);
            let univ_auto = UniversalAutomaton::<Standard>::new(distance);

            let gen_result = gen_auto.accepts(word, input);
            let univ_result = univ_auto.accepts(word, input);

            assert_eq!(
                gen_result, univ_result,
                "Mismatch for ('{}', '{}', {}): gen={}, univ={}",
                word, input, distance, gen_result, univ_result
            );
            assert_eq!(
                gen_result, expected,
                "Expected {} for ('{}', '{}', {}), got {}",
                expected, word, input, distance, gen_result
            );
        }
    }

    #[test]
    fn test_cross_validate_phonetic_merge_simple() {
        // Cross-validate phonetic merge operations
        // Generalized automaton should accept same strings as manual validation
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // These should all be accepted
        let accept_cases = vec![
            ("phone", "fone"),  // ph→f
            ("graph", "graf"),  // ph→f
            ("ship", "sip"),    // sh→s
            ("think", "tink"),  // th→t
            ("church", "kurk"), // both ch→k, exact cost 0.30
            ("chair", "kair"),  // ch→k
        ];

        for (word, input) in accept_cases {
            assert!(
                automaton.accepts(word, input),
                "Should accept ('{}', '{}')",
                word,
                input
            );
        }

        // These should all be rejected (require > distance 1)
        let reject_cases = vec![
            ("phone", "fo"),   // ph→f + delete (needs distance 2)
            ("church", "urk"), // ch→k + delete (needs distance 2)
        ];

        for (word, input) in reject_cases {
            assert!(
                !automaton.accepts(word, input),
                "Should reject ('{}', '{}') at distance 1",
                word,
                input
            );
        }
    }

    #[test]
    fn test_cross_validate_fractional_weights() {
        // Verify fractional weights (0.15) accumulate exactly rather than
        // truncating to free operations.
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        // With distance 1, multiple fractional-weight operations should work
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "church" → "kurk" requires 2× ch→k operations
        // Each operation has weight 0.15; total cost 0.30 is inside budget 1.
        assert!(
            automaton.accepts("church", "kurk"),
            "Two phonetic operations (2×0.15=0.30) should work at distance 1"
        );

        // "church" → "kurks" costs 2×0.15 + 1.0 = 1.30 and must not
        // inherit the old truncation-based acceptance at budget 1.
        assert!(
            !automaton.accepts("church", "kurks"),
            "Two phonetic + one standard operation costs 1.30"
        );

        let automaton2 = GeneralizedAutomaton::with_operations(2, {
            let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
            let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
            for op in phonetic_ops.operations() {
                builder = builder.with_operation(op.clone());
            }
            builder.build()
        });
        assert!(automaton2.accepts("church", "kurks"));

        // But two standard operations should fail
        assert!(
            !automaton.accepts("church", "korks"),
            "Two phonetic + two standard operations should fail at distance 1"
        );
    }

    // Phase 3b: Phonetic split ⟨1,2⟩ tests
    #[test]
    fn test_phonetic_split_k_to_ch() {
        // Test "k"→"ch" split operation
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops.clone());

        // "ark" → "arch" via k→ch split
        assert!(
            automaton.accepts("ark", "arch"),
            "Split k→ch should work at distance 1"
        );

        // "back" → "bach" via k→ch split
        assert!(
            automaton.accepts("back", "bach"),
            "Split k→ch at word end should work"
        );

        // "kan" → "chan" via k→ch split at start
        assert!(
            automaton.accepts("kan", "chan"),
            "Split k→ch at word start should work"
        );
    }

    #[test]
    fn test_phonetic_split_f_to_ph() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "graf" → "graph" via f→ph split
        assert!(
            automaton.accepts("graf", "graph"),
            "Split f→ph should work at distance 1"
        );

        // "foto" → "photo" via f→ph split at start
        assert!(
            automaton.accepts("foto", "photo"),
            "Split f→ph at word start should work"
        );
    }

    #[test]
    fn test_phonetic_split_s_to_sh() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "sip" → "ship" via s→sh split
        assert!(
            automaton.accepts("sip", "ship"),
            "Split s→sh should work at distance 1"
        );

        // "sell" → "shell" via s→sh split at start
        assert!(
            automaton.accepts("sell", "shell"),
            "Split s→sh at word start should work"
        );
    }

    #[test]
    fn test_phonetic_split_t_to_th() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "bat" → "bath" via t→th split
        assert!(
            automaton.accepts("bat", "bath"),
            "Split t→th should work at distance 1"
        );

        // "tin" → "thin" via t→th split at start
        assert!(
            automaton.accepts("tin", "thin"),
            "Split t→th at word start should work"
        );
    }

    #[test]
    fn test_phonetic_split_multiple() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        // Multiple splits require higher distance
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // A single split costs the exact fractional weight 0.15.
        assert!(
            automaton.accepts("kair", "chair"),
            "Single k→ch split should work at distance 1"
        );

        // Two splits cost 0.30 exactly and therefore fit distance 1.
        assert!(
            automaton.accepts("kat", "chath"),
            "Two splits (k→ch, t→th) with fractional weights should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_split_with_standard_ops() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(2, ops);

        // Split + standard operations
        // "graf" → "grape" via f→ph split (0 errors) + e→a substitute (1 error) = 1 total
        assert!(
            automaton.accepts("graf", "graphe"),
            "Split f→ph + insert 'e' should work at distance 1"
        );

        // Multiple standard operations
        // "bak" → "batch" via k→ch split (0) + insert 't' (1) = 1 total
        assert!(
            automaton.accepts("bak", "batch"),
            "Split k→ch + insert should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_split_distance_constraints() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(0, ops.clone());

        // At distance 0, no operations should work
        assert!(
            !automaton.accepts("ark", "arch"),
            "Split should not work at distance 0"
        );

        // At distance 1, fractional-weight split should work
        let automaton = GeneralizedAutomaton::with_operations(1, ops);
        assert!(
            automaton.accepts("ark", "arch"),
            "Split should work at distance 1"
        );
    }

    // Phase 3b: Phonetic transpose ⟨2,2⟩ tests
    #[test]
    fn test_phonetic_transpose_qu_to_kw() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "queen" → "kween" via qu→kw transpose
        assert!(
            automaton.accepts("queen", "kween"),
            "Transpose qu→kw should work at distance 1"
        );

        // "quick" → "kwick" via qu→kw transpose
        assert!(
            automaton.accepts("quick", "kwick"),
            "Transpose qu→kw at word start should work"
        );

        // "quit" → "kwit" via qu→kw transpose
        assert!(
            automaton.accepts("quit", "kwit"),
            "Transpose qu→kw should work"
        );
    }

    #[test]
    fn test_phonetic_transpose_kw_to_qu() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // "kween" → "queen" via kw→qu transpose (reverse)
        assert!(
            automaton.accepts("kween", "queen"),
            "Transpose kw→qu should work at distance 1"
        );

        // "kwik" → "quik" via kw→qu transpose
        assert!(
            automaton.accepts("kwik", "quik"),
            "Transpose kw→qu should work"
        );
    }

    #[test]
    fn test_phonetic_transpose_multiple() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();

        // Multiple transposes with fractional weights
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Single transpose at distance 1
        assert!(
            automaton.accepts("queen", "kween"),
            "Single transpose should work at distance 1"
        );

        // "ququ" → "kwkw" uses two transposes for exact total cost 0.30.
        assert!(
            automaton.accepts("ququ", "kwkw"),
            "Two transposes with fractional weights should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_transpose_with_standard_ops() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops.clone());

        // Transpose + standard operations
        // "queen" → "kweens" costs 0.15 + 1.0 = 1.15.
        assert!(
            !automaton.accepts("queen", "kweens"),
            "Transpose + insert must exceed distance 1"
        );
        let automaton2 = GeneralizedAutomaton::with_operations(2, ops);
        assert!(automaton2.accepts("queen", "kweens"));
    }

    #[test]
    fn test_phonetic_transpose_distance_constraints() {
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(0, ops.clone());

        // At distance 0, no operations should work
        assert!(
            !automaton.accepts("queen", "kween"),
            "Transpose should not work at distance 0"
        );

        // At distance 1, fractional-weight transpose should work
        let automaton = GeneralizedAutomaton::with_operations(1, ops);
        assert!(
            automaton.accepts("queen", "kween"),
            "Transpose should work at distance 1"
        );
    }

    #[test]
    fn test_phonetic_mixed_merge_split_transpose() {
        // Test combining all phonetic operation types
        let phonetic_ops = crate::transducer::phonetic::consonant_digraphs();
        let mut builder = crate::transducer::OperationSetBuilder::new().with_standard_ops();
        for op in phonetic_ops.operations() {
            builder = builder.with_operation(op.clone());
        }
        let ops = builder.build();
        let automaton = GeneralizedAutomaton::with_operations(1, ops);

        // Merge: "phone" → "fone" (ph→f)
        assert!(
            automaton.accepts("phone", "fone"),
            "Merge operation should work"
        );

        // Split: "graf" → "graph" (f→ph)
        assert!(
            automaton.accepts("graf", "graph"),
            "Split operation should work"
        );

        // Transpose: "queen" → "kween" (qu→kw)
        assert!(
            automaton.accepts("queen", "kween"),
            "Transpose operation should work"
        );

        // Each configured phonetic rule has exact cost 0.15 and fits budget 1.
    }
}
