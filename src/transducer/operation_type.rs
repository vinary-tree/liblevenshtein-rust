//! Generalized operation types for Levenshtein automata.
//!
//! This module implements the operation type system from "Efficient Computation of Substring Equivalence Classes with Applications to Polynomial-Time Tree Isomorphism"
//! (TCS 2011, Section 3, pages 2341-2342).
//!
//! ## Overview
//!
//! An **operation type** defines a single edit operation in a generalized Levenshtein
//! distance metric. Unlike the hardcoded Standard/Transposition/MergeAndSplit algorithms,
//! operation types are data-driven and composable.
//!
//! ## Operation Type Definition
//!
//! An operation type `t = ⟨t^x, t^y, t^w⟩` consists of:
//! - `t^x`: Number of characters consumed from the first word (dictionary)
//! - `t^y`: Number of characters consumed from the second word (query)
//! - `t^w`: Operation weight/cost (float)
//!
//! ## Standard Levenshtein Operations
//!
//! ```text
//! Match:         ⟨1, 1, 0.0⟩  // Consume 1 from each, no cost
//! Substitution:  ⟨1, 1, 1.0⟩  // Consume 1 from each, cost 1
//! Insertion:     ⟨0, 1, 1.0⟩  // Consume only from query, cost 1
//! Deletion:      ⟨1, 0, 1.0⟩  // Consume only from dictionary, cost 1
//! Transposition: ⟨2, 2, 1.0⟩  // Consume 2 from each, cost 1
//! ```
//!
//! ## Extended Operations
//!
//! ### Phonetic Corrections
//! ```text
//! ph→f digraph:  ⟨2, 1, 0.15⟩  // "ph" in dict matches "f" in query
//! Silent e:      ⟨1, 0, 0.1⟩   // Final "e" deletion, low cost
//! ```
//!
//! ### Weighted OCR Corrections
//! ```text
//! O↔0 confusion: ⟨1, 1, 0.2⟩  // Common OCR error, low cost
//! l↔I confusion: ⟨1, 1, 0.3⟩  // Less common, higher cost
//! ```
//!
//! ## Restricted Operations
//!
//! A **restricted operation** `op = ⟨op^x, op^y, op^r, op^w⟩` adds:
//! - `op^r ⊆ Σ^{op^x} × Σ^{op^y}`: Set of allowed character pair substitutions
//!
//! Only character pairs in `op^r` can use this operation.
//!
//! ### Example: Keyboard Proximity
//! ```rust
//! use liblevenshtein::transducer::{OperationType, SubstitutionSet};
//!
//! let mut proximity = SubstitutionSet::new();
//! proximity.allow_str("q", "w");  // Adjacent on QWERTY
//! proximity.allow_str("w", "q");
//! proximity.allow_str("w", "e");
//! proximity.allow_str("e", "w");
//!
//! let op = OperationType::with_restriction(
//!     1, 1, 0.3,  // Low cost for adjacent keys
//!     proximity,
//!     "keyboard_proximity"
//! );
//! ```
//!
//! ## Theoretical Constraints
//!
//! From TCS 2011 Theorem 8.2 (Bounded Diagonal Property):
//!
//! 1. **Bounded consumption**: `t^x + t^y ≤ c` for some constant `c`
//! 2. **Zero-weight must be length-preserving**: If `t^w = 0`, then `t^x = t^y`
//!
//! These constraints ensure:
//! - Finite state space for universal automata
//! - Bounded subsumption check complexity
//! - SmallVec optimization (inline size = 8 is mathematically sufficient)
//!
//! ## Performance
//!
//! - **Operation matching**: O(1) for unrestricted, O(|op^r|) for restricted
//! - **Memory**: 16-40 bytes per operation depending on restriction set size
//! - **Cache-friendly**: Small operations stored inline (SmallVec)

use crate::transducer::substitution_set::SubstitutionSet;
use std::borrow::Cow;
use std::fmt;

/// Maximum UTF-8 byte length accepted for a persisted operation name.
pub const MAX_OPERATION_NAME_BYTES: usize = 1_024;

/// Semantic predicate controlling where an operation may apply.
///
/// This discriminator is deliberately independent of [`OperationType::name`]:
/// diagnostic renaming must never change the accepted language.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperationApplicability {
    /// Apply to every source/target slice of the declared dimensions.
    Any,
    /// Apply only when the complete source and target slices are equal.
    Equal,
    /// Apply only to an adjacent two-scalar reversal.
    AdjacentTranspose,
    /// Apply only to pairs explicitly listed in the restriction set.
    Listed(SubstitutionSet),
}

/// A generalized operation type for Levenshtein automata.
///
/// Represents an edit operation `t = ⟨t^x, t^y, t^w⟩` from TCS 2011.
///
/// # Examples
///
/// ## Standard Operations
///
/// ```rust
/// # use liblevenshtein::transducer::OperationType;
/// // Match: ⟨1, 1, 0.0⟩
/// let match_op = OperationType::new(1, 1, 0.0, "match");
///
/// // Substitution: ⟨1, 1, 1.0⟩
/// let subst_op = OperationType::new(1, 1, 1.0, "substitute");
///
/// // Insertion: ⟨0, 1, 1.0⟩
/// let insert_op = OperationType::new(0, 1, 1.0, "insert");
///
/// // Deletion: ⟨1, 0, 1.0⟩
/// let delete_op = OperationType::new(1, 0, 1.0, "delete");
/// ```
///
/// ## Multi-Character Operations
///
/// ```rust
/// # use liblevenshtein::transducer::{OperationType, SubstitutionSet};
/// // Phonetic: ph→f digraph
/// let mut phonetic = SubstitutionSet::new();
/// phonetic.allow_str("ph", "f");
///
/// let ph_to_f = OperationType::with_restriction(
///     2, 1, 0.15,
///     phonetic,
///     "consonant_digraph_ph"
/// );
/// ```
///
/// ## Weighted Operations
///
/// ```rust
/// # use liblevenshtein::transducer::OperationType;
/// // OCR confusion: O↔0 (common, low cost)
/// let ocr_op = OperationType::new(1, 1, 0.2, "ocr_o_zero");
/// ```
#[derive(Clone, Debug)]
pub struct OperationType {
    /// Number of characters consumed from first word (dictionary).
    /// Corresponds to `t^x` in TCS 2011.
    consume_x: usize,

    /// Number of characters consumed from second word (query).
    /// Corresponds to `t^y` in TCS 2011.
    consume_y: usize,

    /// Operation weight/cost (≥ 0.0).
    /// Corresponds to `t^w` in TCS 2011.
    /// - 0.0: Free operation (must have consume_x == consume_y by Theorem 8.2)
    /// - 1.0: Standard Levenshtein cost
    /// - Other values: Custom costs for weighted operations
    weight: f64,

    /// Explicit semantic applicability predicate.
    applicability: OperationApplicability,

    /// Human-readable name for debugging and profiling.
    /// Examples: "match", "substitute", "ph_to_f", "silent_e"
    name: Cow<'static, str>,
}

impl OperationType {
    /// Create a new unrestricted operation type.
    ///
    /// # Parameters
    ///
    /// - `consume_x`: Characters consumed from dictionary word (≥ 0)
    /// - `consume_y`: Characters consumed from query word (≥ 0)
    /// - `weight`: Operation cost (≥ 0.0)
    /// - `name`: Human-readable identifier
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `weight < 0.0` (negative costs are invalid)
    /// - `weight == 0.0` but `consume_x != consume_y` (violates Theorem 8.2)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::OperationType;
    /// // Standard match operation
    /// let match_op = OperationType::new(1, 1, 0.0, "match");
    ///
    /// // Custom weighted operation
    /// let custom = OperationType::new(2, 1, 0.5, "custom_digraph");
    /// ```
    #[inline]
    pub fn new(consume_x: usize, consume_y: usize, weight: f64, name: &'static str) -> Self {
        assert!(weight >= 0.0, "Operation weight must be non-negative");

        // Theorem 8.2: Zero-weight operations must be length-preserving
        if weight == 0.0 {
            assert_eq!(
                consume_x, consume_y,
                "Zero-weight operation must preserve length (consume_x == consume_y)"
            );
        }

        Self {
            consume_x,
            consume_y,
            weight,
            applicability: if weight == 0.0 {
                OperationApplicability::Equal
            } else {
                OperationApplicability::Any
            },
            name: Cow::Borrowed(name),
        }
    }

    /// Create an unrestricted operation with an owned diagnostic name.
    ///
    /// This is the non-leaking construction path used by deserialization and
    /// by applications that load operation definitions at runtime.
    #[inline]
    pub fn new_owned(
        consume_x: usize,
        consume_y: usize,
        weight: f64,
        name: impl Into<String>,
    ) -> Self {
        assert!(weight >= 0.0, "Operation weight must be non-negative");
        if weight == 0.0 {
            assert_eq!(
                consume_x, consume_y,
                "Zero-weight operation must preserve length (consume_x == consume_y)"
            );
        }
        Self {
            consume_x,
            consume_y,
            weight,
            applicability: if weight == 0.0 {
                OperationApplicability::Equal
            } else {
                OperationApplicability::Any
            },
            name: Cow::Owned(name.into()),
        }
    }

    /// Create a restricted operation type with character pair constraints.
    ///
    /// The operation only applies when the consumed characters match a pair
    /// in the restriction set.
    ///
    /// # Parameters
    ///
    /// - `consume_x`: Characters consumed from dictionary word
    /// - `consume_y`: Characters consumed from query word
    /// - `weight`: Operation cost
    /// - `restriction`: Allowed character pair substitutions
    /// - `name`: Human-readable identifier
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationType, SubstitutionSet};
    /// // Phonetic ph→f
    /// let mut phonetic = SubstitutionSet::new();
    /// phonetic.allow_str("ph", "f");
    ///
    /// let op = OperationType::with_restriction(
    ///     2, 1, 0.15,
    ///     phonetic,
    ///     "ph_to_f"
    /// );
    /// ```
    #[inline]
    pub fn with_restriction(
        consume_x: usize,
        consume_y: usize,
        weight: f64,
        restriction: SubstitutionSet,
        name: &'static str,
    ) -> Self {
        assert!(weight >= 0.0, "Operation weight must be non-negative");
        if weight == 0.0 {
            assert_eq!(
                consume_x, consume_y,
                "Zero-weight operation must preserve length (consume_x == consume_y)"
            );
        }
        Self {
            consume_x,
            consume_y,
            weight,
            applicability: OperationApplicability::Listed(restriction),
            name: Cow::Borrowed(name),
        }
    }

    /// Create a restricted operation with an owned diagnostic name.
    #[inline]
    pub fn with_owned_restriction(
        consume_x: usize,
        consume_y: usize,
        weight: f64,
        restriction: SubstitutionSet,
        name: impl Into<String>,
    ) -> Self {
        assert!(weight >= 0.0, "Operation weight must be non-negative");
        if weight == 0.0 {
            assert_eq!(
                consume_x, consume_y,
                "Zero-weight operation must preserve length (consume_x == consume_y)"
            );
        }
        Self {
            consume_x,
            consume_y,
            weight,
            applicability: OperationApplicability::Listed(restriction),
            name: Cow::Owned(name.into()),
        }
    }

    /// Create an adjacent-transposition operation.
    ///
    /// The operation consumes exactly two scalar values from each side and
    /// applies only when the target slice reverses the source slice.
    #[inline]
    pub fn adjacent_transposition(weight: f64, name: &'static str) -> Self {
        assert!(weight.is_finite() && weight >= 0.0);
        Self {
            consume_x: 2,
            consume_y: 2,
            weight,
            applicability: OperationApplicability::AdjacentTranspose,
            name: Cow::Borrowed(name),
        }
    }

    /// Create an operation with an explicit semantic applicability predicate.
    #[inline]
    pub fn with_applicability(
        consume_x: usize,
        consume_y: usize,
        weight: f64,
        applicability: OperationApplicability,
        name: &'static str,
    ) -> Self {
        assert!(weight >= 0.0, "Operation weight must be non-negative");
        if weight == 0.0 {
            assert_eq!(
                consume_x, consume_y,
                "Zero-weight operation must preserve length (consume_x == consume_y)"
            );
        }
        Self {
            consume_x,
            consume_y,
            weight,
            applicability,
            name: Cow::Borrowed(name),
        }
    }

    /// Create an operation with explicit applicability and an owned name.
    ///
    /// This is the non-leaking construction path for schema-driven and other
    /// runtime-loaded operation definitions. Semantic compatibility between
    /// the applicability discriminator and the declared arity is checked by
    /// [`OperationSet::validate`](super::OperationSet::validate).
    #[inline]
    pub fn with_owned_applicability(
        consume_x: usize,
        consume_y: usize,
        weight: f64,
        applicability: OperationApplicability,
        name: impl Into<String>,
    ) -> Self {
        assert!(weight >= 0.0, "Operation weight must be non-negative");
        if weight == 0.0 {
            assert_eq!(
                consume_x, consume_y,
                "Zero-weight operation must preserve length (consume_x == consume_y)"
            );
        }
        Self {
            consume_x,
            consume_y,
            weight,
            applicability,
            name: Cow::Owned(name.into()),
        }
    }

    /// Get the number of characters consumed from the dictionary word.
    #[inline]
    pub fn consume_x(&self) -> usize {
        self.consume_x
    }

    /// Get the number of characters consumed from the query word.
    #[inline]
    pub fn consume_y(&self) -> usize {
        self.consume_y
    }

    /// Get the operation weight/cost.
    #[inline]
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Get the operation name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Check if this operation is restricted to specific character pairs.
    #[inline]
    pub fn is_restricted(&self) -> bool {
        matches!(self.applicability, OperationApplicability::Listed(_))
    }

    /// Get the restriction set, if any.
    #[inline]
    pub fn restriction(&self) -> Option<&SubstitutionSet> {
        match &self.applicability {
            OperationApplicability::Listed(restriction) => Some(restriction),
            _ => None,
        }
    }

    /// Return the explicit semantic applicability predicate.
    #[inline]
    pub fn applicability(&self) -> &OperationApplicability {
        &self.applicability
    }

    /// Check if this operation is a match (zero cost, length-preserving).
    #[inline]
    pub fn is_match(&self) -> bool {
        self.weight == 0.0 && matches!(self.applicability, OperationApplicability::Equal)
    }

    /// Check if this operation is an insertion (consumes only from query).
    #[inline]
    pub fn is_insertion(&self) -> bool {
        self.consume_x == 0 && self.consume_y > 0
    }

    /// Check if this operation is a deletion (consumes only from dictionary).
    #[inline]
    pub fn is_deletion(&self) -> bool {
        self.consume_x > 0 && self.consume_y == 0
    }

    /// Check if this is a non-equality single-scalar replacement.
    #[inline]
    pub fn is_substitution(&self) -> bool {
        self.consume_x == 1 && self.consume_y == 1 && !self.is_match()
    }

    /// Check if this operation can apply to the given character pair.
    ///
    /// For unrestricted operations, always returns true.
    /// For restricted operations, checks if the pair is in the restriction set.
    ///
    /// # Parameters
    ///
    /// - `dict_chars`: Characters from dictionary word (length = consume_x)
    /// - `query_chars`: Characters from query word (length = consume_y)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationType, SubstitutionSet};
    /// // Unrestricted match
    /// let match_op = OperationType::new(1, 1, 0.0, "match");
    /// assert!(match_op.can_apply(b"a", b"a"));
    /// assert!(!match_op.can_apply(b"a", b"b"));  // Characters must match for match op
    ///
    /// // Restricted phonetic operation
    /// let mut phonetic = SubstitutionSet::new();
    /// phonetic.allow_str("ph", "f");
    /// let ph_op = OperationType::with_restriction(2, 1, 0.15, phonetic, "ph_to_f");
    ///
    /// assert!(ph_op.can_apply(b"ph", b"f"));
    /// assert!(!ph_op.can_apply(b"ph", b"g"));
    /// ```
    #[inline]
    pub fn can_apply(&self, dict_chars: &[u8], query_chars: &[u8]) -> bool {
        if let (Ok(dict), Ok(query)) = (
            std::str::from_utf8(dict_chars),
            std::str::from_utf8(query_chars),
        ) {
            return self.can_apply_str(dict, query);
        }

        // Length check
        if dict_chars.len() != self.consume_x || query_chars.len() != self.consume_y {
            return false;
        }

        match &self.applicability {
            OperationApplicability::Any => true,
            OperationApplicability::Equal => dict_chars == query_chars,
            OperationApplicability::AdjacentTranspose => {
                dict_chars.len() == 2
                    && query_chars.len() == 2
                    && dict_chars[0] == query_chars[1]
                    && dict_chars[1] == query_chars[0]
            }
            OperationApplicability::Listed(set) => set.contains_str(dict_chars, query_chars),
        }
    }

    /// Check whether this operation applies to two Unicode string slices.
    ///
    /// `consume_x` and `consume_y` count Unicode scalar values, not UTF-8
    /// bytes. This method performs that dimensional check before delegating
    /// restricted-pair lookup to the byte-oriented storage. It is therefore
    /// the correct entry point for the `char`-based generalized automaton.
    #[inline]
    pub fn can_apply_str(&self, dict: &str, query: &str) -> bool {
        if dict.chars().count() != self.consume_x || query.chars().count() != self.consume_y {
            return false;
        }

        match &self.applicability {
            OperationApplicability::Any => true,
            OperationApplicability::Equal => dict == query,
            OperationApplicability::AdjacentTranspose => {
                let mut source = dict.chars();
                let mut target = query.chars();
                matches!(
                    (source.next(), source.next(), source.next(), target.next(), target.next(), target.next()),
                    (Some(source_first), Some(source_second), None, Some(target_first), Some(target_second), None)
                        if source_first == target_second && source_second == target_first
                )
            }
            OperationApplicability::Listed(set) => {
                set.contains_str(dict.as_bytes(), query.as_bytes())
            }
        }
    }

    /// Check complete alignment semantics for character slices.
    ///
    /// The built-in unrestricted `transpose` operation denotes an adjacent
    /// reversal rather than an arbitrary two-for-two substitution. All other
    /// operations use their declared restriction, or apply to every slice of
    /// the declared scalar dimensions when unrestricted.
    #[inline]
    pub fn applies_to_slices(&self, dict: &str, query: &str) -> bool {
        self.can_apply_str(dict, query)
    }

    /// Check if this operation could potentially apply to the given source characters.
    ///
    /// Unlike [`can_apply()`](Self::can_apply), this only checks the source (dictionary)
    /// characters and ignores the target (query) characters. This is useful for speculative
    /// state entry when we don't yet know the target characters (e.g., entering a split state).
    ///
    /// # Theoretical Justification
    ///
    /// This method satisfies the **locality property** from weighted-levenshtein-automata
    /// research: the decision uses only O(n) lookahead in the word. Checking if a source
    /// exists in a bounded restriction set is O(|restriction|) which is bounded by O(n).
    ///
    /// # Parameters
    ///
    /// - `dict_chars`: Characters from dictionary word
    ///
    /// # Returns
    ///
    /// `true` if the operation COULD apply to these source characters (given the right target),
    /// `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationType, SubstitutionSet};
    /// // Phonetic split operation: k→ch
    /// let mut phonetic = SubstitutionSet::new();
    /// phonetic.allow_str("k", "ch");
    /// phonetic.allow_str("t", "th");
    /// let split_op = OperationType::with_restriction(1, 2, 0.15, phonetic, "split");
    ///
    /// assert!(split_op.can_apply_to_source(b"k"));  // k can be split (→ch)
    /// assert!(split_op.can_apply_to_source(b"t"));  // t can be split (→th)
    /// assert!(!split_op.can_apply_to_source(b"a")); // a cannot be split
    /// ```
    ///
    /// # Use Case: Split State Entry
    ///
    /// When deciding whether to enter a split state for phonetic operations:
    ///
    /// ```rust,ignore
    /// // Check if THIS word character can be split (without knowing target yet)
    /// let can_enter_split = split_ops.iter().any(|op| {
    ///     op.can_apply_to_source(word_char.as_bytes())
    /// });
    /// ```
    #[inline]
    pub fn can_apply_to_source(&self, dict_chars: &[u8]) -> bool {
        // Valid UTF-8 follows the public scalar-value consumption contract.
        // Retain byte-counting only for the low-level invalid-UTF-8
        // compatibility path, matching `can_apply`.
        let dimension_matches = std::str::from_utf8(dict_chars).map_or_else(
            |_| dict_chars.len() == self.consume_x,
            |dict| dict.chars().count() == self.consume_x,
        );
        if !dimension_matches {
            return false;
        }

        match &self.applicability {
            OperationApplicability::Listed(set) => set.has_source(dict_chars),
            OperationApplicability::Any
            | OperationApplicability::Equal
            | OperationApplicability::AdjacentTranspose => true,
        }
    }

    /// Check if this operation could apply to the given source AND the target starts with the given character.
    ///
    /// This is used for phonetic split operations to validate that when entering a split state,
    /// the current input character matches the first character of the operation's target sequence.
    ///
    /// # Theoretical Justification
    ///
    /// This maintains the **locality property** - the decision uses only:
    /// 1. O(1) current input character
    /// 2. O(n) word position check (via restriction set)
    ///
    /// Both are within the allowed O(n) lookahead constraint.
    ///
    /// # Parameters
    ///
    /// - `dict_chars`: Characters from dictionary word (source)
    /// - `first_target_char`: The expected first character of the target sequence
    ///
    /// # Returns
    ///
    /// `true` if this operation could apply to the source AND has a target starting with `first_target_char`,
    /// `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use liblevenshtein::transducer::{OperationType, SubstitutionSet};
    /// // Phonetic split operation: k→ch, t→th
    /// let mut phonetic = SubstitutionSet::new();
    /// phonetic.allow_str("k", "ch");
    /// phonetic.allow_str("t", "th");
    /// let split_op = OperationType::with_restriction(1, 2, 0.15, phonetic, "split");
    ///
    /// // k→ch: first target char is 'c'
    /// assert!(split_op.matches_first_target_char(b"k", 'c'));
    /// assert!(!split_op.matches_first_target_char(b"k", 't'));
    /// assert!(!split_op.matches_first_target_char(b"k", 'a'));
    ///
    /// // t→th: first target char is 't'
    /// assert!(split_op.matches_first_target_char(b"t", 't'));
    /// assert!(!split_op.matches_first_target_char(b"t", 'c'));
    /// ```
    ///
    /// # Use Case: Split State Entry Validation
    ///
    /// When deciding whether to enter a split state, verify both:
    /// 1. The word character can be split (via `can_apply_to_source`)
    /// 2. The current input character matches the first target character
    ///
    /// ```rust,ignore
    /// // Only enter split state if input char matches first char of split target
    /// let can_enter_split = split_ops.iter().any(|op| {
    ///     op.can_apply_to_source(word_char.as_bytes())
    ///         && op.matches_first_target_char(word_char.as_bytes(), input_char)
    /// });
    /// ```
    #[inline]
    pub fn matches_first_target_char(&self, dict_chars: &[u8], first_target_char: char) -> bool {
        // Keep source feasibility consistent with `can_apply_to_source`: valid
        // UTF-8 counts scalar values, while invalid UTF-8 uses raw bytes.
        let dimension_matches = std::str::from_utf8(dict_chars).map_or_else(
            |_| dict_chars.len() == self.consume_x,
            |dict| dict.chars().count() == self.consume_x,
        );
        if !dimension_matches {
            return false;
        }

        match &self.applicability {
            OperationApplicability::Listed(set) => {
                set.has_target_starting_with(dict_chars, first_target_char)
            }
            OperationApplicability::Any
            | OperationApplicability::Equal
            | OperationApplicability::AdjacentTranspose => false,
        }
    }
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}⟨{}, {}, {}⟩",
            self.name, self.consume_x, self.consume_y, self.weight
        )?;

        if self.is_restricted() {
            write!(f, " [restricted]")?;
        }

        Ok(())
    }
}

impl PartialEq for OperationType {
    fn eq(&self, other: &Self) -> bool {
        self.consume_x == other.consume_x
            && self.consume_y == other.consume_y
            && self.weight == other.weight
            && self.applicability == other.applicability
            && self.name == other.name
    }
}

impl Eq for OperationType {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_operations() {
        let match_op = OperationType::new(1, 1, 0.0, "match");
        assert!(match_op.is_match());
        assert!(!match_op.is_restricted());
        assert_eq!(match_op.consume_x(), 1);
        assert_eq!(match_op.consume_y(), 1);
        assert_eq!(match_op.weight(), 0.0);

        let subst_op = OperationType::new(1, 1, 1.0, "substitute");
        assert!(subst_op.is_substitution());
        assert!(!subst_op.is_match());

        let insert_op = OperationType::new(0, 1, 1.0, "insert");
        assert!(insert_op.is_insertion());

        let delete_op = OperationType::new(1, 0, 1.0, "delete");
        assert!(delete_op.is_deletion());
    }

    #[test]
    fn test_operation_type_equality_uses_exact_weight_semantics() {
        let zero_cost = OperationType::new(1, 1, 0.0, "match");
        let tiny_cost = OperationType::new(1, 1, f64::MIN_POSITIVE, "match");

        assert!(zero_cost.is_match());
        assert!(!tiny_cost.is_match());
        assert_ne!(zero_cost, tiny_cost);

        let negative_zero = OperationType::new(1, 1, -0.0, "match");
        assert_eq!(zero_cost, negative_zero);
    }

    #[test]
    #[should_panic(expected = "Zero-weight operation must preserve length")]
    fn test_zero_weight_must_preserve_length() {
        OperationType::new(2, 1, 0.0, "invalid");
    }

    #[test]
    #[should_panic(expected = "Operation weight must be non-negative")]
    fn test_negative_weight_panics() {
        OperationType::new(1, 1, -0.5, "invalid");
    }

    #[test]
    fn test_match_requires_equality() {
        let match_op = OperationType::new(1, 1, 0.0, "match");
        assert!(match_op.can_apply(b"a", b"a"));
        assert!(!match_op.can_apply(b"a", b"b"));
    }

    #[test]
    fn test_unrestricted_operations() {
        let subst = OperationType::new(1, 1, 1.0, "substitute");
        // Unrestricted substitution works for any non-matching pair
        assert!(subst.can_apply(b"a", b"b"));
        assert!(subst.can_apply(b"x", b"y"));
    }

    #[test]
    fn test_restricted_operation() {
        let mut phonetic = SubstitutionSet::new();
        phonetic.allow('f', 'p');
        phonetic.allow('p', 'h');

        let restricted = OperationType::with_restriction(1, 1, 0.3, phonetic, "phonetic");

        assert!(restricted.is_restricted());
        assert!(restricted.can_apply(b"f", b"p"));
        assert!(restricted.can_apply(b"p", b"h"));
        assert!(!restricted.can_apply(b"a", b"b"));
    }

    #[test]
    fn test_restricted_operation_utf8_strings() {
        let mut rules = SubstitutionSet::new();
        rules.allow_str("α", "β");
        rules.allow_str("sch", "š");

        let greek = OperationType::with_restriction(
            "α".chars().count(),
            "β".chars().count(),
            0.2,
            rules.clone(),
            "greek",
        );
        assert!(greek.can_apply("α".as_bytes(), "β".as_bytes()));
        assert!(greek.can_apply_str("α", "β"));
        assert!(greek.can_apply_to_source("α".as_bytes()));
        assert!(!greek.can_apply("α".as_bytes(), "γ".as_bytes()));

        let palatal = OperationType::with_restriction(
            "sch".chars().count(),
            "š".chars().count(),
            0.2,
            rules,
            "palatal",
        );
        assert!(palatal.can_apply(b"sch", "š".as_bytes()));
        assert!(palatal.can_apply_to_source(b"sch"));
        assert!(palatal.matches_first_target_char(b"sch", 'š'));
        assert!(!palatal.matches_first_target_char(b"sch", 's'));
    }

    #[test]
    fn alignment_slice_semantics_distinguish_transpose_from_two_for_two_substitution() {
        let transpose = OperationType::adjacent_transposition(1.0, "diagnostic label");
        let unrestricted = OperationType::new(2, 2, 1.0, "two_for_two");
        let misleading_name = OperationType::new(2, 2, 1.0, "transpose");

        assert!(transpose.applies_to_slices("ab", "ba"));
        assert!(!transpose.applies_to_slices("ab", "cd"));
        assert!(unrestricted.applies_to_slices("ab", "cd"));
        assert!(misleading_name.applies_to_slices("ab", "cd"));
        assert!(transpose.applies_to_slices("éß", "ßé"));
    }

    #[test]
    fn zero_cost_listed_applicability_is_not_replaced_by_equality() {
        let mut listed = SubstitutionSet::new();
        listed.allow_str("a", "b");
        let operation = OperationType::with_restriction(1, 1, 0.0, listed, "free_alias");

        assert!(operation.can_apply_str("a", "b"));
        assert!(!operation.can_apply_str("a", "a"));
        assert!(!operation.is_match());
    }

    #[test]
    fn test_display() {
        let op = OperationType::new(2, 1, 0.15, "ph_to_f");
        let display = format!("{}", op);
        assert!(display.contains("ph_to_f"));
        assert!(display.contains("2"));
        assert!(display.contains("1"));
        assert!(display.contains("0.15"));
    }

    #[test]
    fn test_can_apply_to_source_unrestricted() {
        let subst = OperationType::new(1, 1, 1.0, "substitution");

        // Unrestricted operation can apply to any source of correct length
        assert!(subst.can_apply_to_source(b"a"));
        assert!(subst.can_apply_to_source(b"x"));
        assert!(subst.can_apply_to_source(b"z"));

        // Wrong length should fail
        assert!(!subst.can_apply_to_source(b"ab"));
        assert!(!subst.can_apply_to_source(b""));
    }

    #[test]
    fn test_can_apply_to_source_restricted() {
        // Phonetic split operation: k→ch, t→th
        let mut phonetic = SubstitutionSet::new();
        phonetic.allow_str("k", "ch");
        phonetic.allow_str("t", "th");

        let split_op = OperationType::with_restriction(1, 2, 0.15, phonetic, "split");

        // Sources that exist in restriction set
        assert!(split_op.can_apply_to_source(b"k")); // k→ch
        assert!(split_op.can_apply_to_source(b"t")); // t→th

        // Sources that don't exist in restriction set
        assert!(!split_op.can_apply_to_source(b"a"));
        assert!(!split_op.can_apply_to_source(b"e"));
        assert!(!split_op.can_apply_to_source(b"s"));

        // Wrong length
        assert!(!split_op.can_apply_to_source(b"ch")); // Target, not source, and wrong length
    }

    #[test]
    fn test_can_apply_to_source_multi_char() {
        // Merge operation: ch→k, sh→s
        let mut phonetic = SubstitutionSet::new();
        phonetic.allow_str("ch", "k");
        phonetic.allow_str("sh", "s");

        let merge_op = OperationType::with_restriction(2, 1, 0.15, phonetic, "merge");

        // Multi-char sources that exist
        assert!(merge_op.can_apply_to_source(b"ch")); // ch→k
        assert!(merge_op.can_apply_to_source(b"sh")); // sh→s

        // Sources that don't exist
        assert!(!merge_op.can_apply_to_source(b"ph"));
        assert!(!merge_op.can_apply_to_source(b"th"));

        // Wrong length
        assert!(!merge_op.can_apply_to_source(b"c"));
        assert!(!merge_op.can_apply_to_source(b"s"));
    }

    #[test]
    fn test_can_apply_to_source_vs_can_apply() {
        let mut phonetic = SubstitutionSet::new();
        phonetic.allow_str("k", "ch");

        let split_op = OperationType::with_restriction(1, 2, 0.15, phonetic, "split");

        // can_apply_to_source only checks source
        assert!(split_op.can_apply_to_source(b"k"));

        // can_apply checks both source AND target
        assert!(split_op.can_apply(b"k", b"ch"));
        assert!(!split_op.can_apply(b"k", b"sh")); // Wrong target

        // Source doesn't exist
        assert!(!split_op.can_apply_to_source(b"a"));
        assert!(!split_op.can_apply(b"a", b"ch"));
    }
}
