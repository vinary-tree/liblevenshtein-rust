//! Generalized Position Types
//!
//! Implements universal positions adapted for runtime-configurable operations.
//! Based on Definition 15 from Mitankin's thesis (pages 30-33), but without
//! compile-time variant specialization.
//!
//! # Theory Background
//!
//! Generalized positions use parameters I (non-final) and M (final) with runtime operations:
//! - `I + t#k`: Non-final position, offset t from start, scaled cost k consumed
//! - `M + t#k`: Final position, offset t from end, scaled cost k consumed
//!
//! Unlike UniversalPosition, GeneralizedPosition does not use compile-time variants
//! (Standard, Transposition, MergeAndSplit). Instead, operations are determined at
//! runtime via the OperationSet parameter.
//!
//! ## Invariants (from Definition 15)
//!
//! The source-compatible public constructors use a unit-cost budget and
//! preserve the classical invariants. **I-type (non-final)** positions satisfy:
//! ```text
//! I^ε_s = {I + t#k | |t| ≤ k ∧ -n ≤ t ≤ n ∧ 0 ≤ k ≤ n}
//! ```
//!
//! **M-type (final)**:
//! ```text
//! M^ε_s = {M + t#k | k ≥ -t - n ∧ -2n ≤ t ≤ 0 ∧ 0 ≤ k ≤ n}
//! ```
//!
//! Streaming transitions carry an exact denominator `d` internally, store
//! `k` as a `usize` numerator, and compare it with the scaled budget `K = n*d`.
//! For a non-classical operation lattice, only the budget and bounded-offset
//! constraints are universal; cheap multi-scalar rules invalidate the unit
//! theorem `|t| ≤ k`. Subsumption is correspondingly conservative.
//!
//! # Examples
//!
//! ```rust
//! use liblevenshtein::transducer::generalized::GeneralizedPosition;
//!
//! // Create I-type position: I + 0#0 (initial state)
//! let initial = GeneralizedPosition::new_i(0, 0, 2)
//!     .expect("0#0 satisfies the generalized I-position invariant");
//! assert!(initial.is_non_final());
//!
//! // Create M-type position: M + (-1)#1 (one error, one char before end)
//! let final_pos = GeneralizedPosition::new_m(-1, 1, 2)
//!     .expect("-1#1 satisfies the generalized M-position invariant");
//! assert!(final_pos.is_final());
//! ```

use std::fmt;

/// Error type for invalid position creation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionError {
    /// I-type position violates the applicable offset or scaled-budget invariant.
    InvalidIPosition {
        /// The offset value that violated the invariant
        offset: i32,
        /// The exact scaled cost that violated the invariant.
        errors: usize,
        /// Legacy field name; contains the maximum scaled cost.
        max_distance: usize,
    },

    /// M-type position violates the applicable offset or scaled-budget invariant.
    InvalidMPosition {
        /// The offset value that violated the invariant
        offset: i32,
        /// The exact scaled cost that violated the invariant.
        errors: usize,
        /// Legacy field name; contains the maximum scaled cost.
        max_distance: usize,
    },
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PositionError::InvalidIPosition {
                offset,
                errors,
                max_distance,
            } => {
                write!(
                    f,
                    "invalid I-position: I + {}#{} with scaled budget {}",
                    offset, errors, max_distance
                )
            }
            PositionError::InvalidMPosition {
                offset,
                errors,
                max_distance,
            } => {
                write!(
                    f,
                    "invalid M-position: M + {}#{} with scaled budget {}",
                    offset, errors, max_distance
                )
            }
        }
    }
}

impl std::error::Error for PositionError {}

/// Generalized position with parameter (I or M)
///
/// From Definition 15 (thesis pages 30-33), adapted for runtime operations.
///
/// Unlike `UniversalPosition<V>`, this type does not use compile-time variant
/// specialization. Operations are determined at runtime via `OperationSet`.
///
/// # Notation Mapping
///
/// Theory → Rust:
/// - `I + t#k` → `INonFinal { offset: t, errors: k }`
/// - `M + t#k` → `MFinal { offset: t, errors: k }`
/// - `I + t#k_t` → `ITransposing { offset: t, errors: k }` (transposing state)
/// - `M + t#k_t` → `MTransposing { offset: t, errors: k }` (transposing state)
/// - `I + t#k_s` → `ISplitting { offset: t, errors: k }` (splitting state)
/// - `M + t#k_s` → `MSplitting { offset: t, errors: k }` (splitting state)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GeneralizedPosition {
    /// I-type (non-final): I + offset#errors
    ///
    /// Represents position relative to start of word.
    ///
    /// Invariant: |offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ n
    INonFinal {
        /// Offset from parameter I (range: -n to n)
        offset: i32,

        /// Number of errors consumed (range: 0 to n)
        errors: usize,
    },

    /// M-type (final): M + offset#errors
    ///
    /// Represents position relative to end of word.
    ///
    /// Invariant: errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ n
    MFinal {
        /// Offset from parameter M (range: -2n to 0)
        offset: i32,

        /// Number of errors consumed (range: 0 to n)
        errors: usize,
    },

    /// I-type transposing state: I + offset#errors_t
    ///
    /// Intermediate state during transposition operation ⟨2,2,1⟩.
    /// Represents a position in the middle of a two-step adjacent character swap.
    ///
    /// Same invariants as INonFinal: |offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ n
    ITransposing {
        /// Offset from parameter I (range: -n to n)
        offset: i32,

        /// Number of errors consumed (range: 0 to n)
        errors: usize,
    },

    /// M-type transposing state: M + offset#errors_t
    ///
    /// Intermediate state during transposition operation ⟨2,2,1⟩ at final positions.
    ///
    /// Same invariants as MFinal: errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ n
    MTransposing {
        /// Offset from parameter M (range: -2n to 0)
        offset: i32,

        /// Number of errors consumed (range: 0 to n)
        errors: usize,
    },

    /// I-type splitting state: I + offset#errors_s
    ///
    /// Intermediate state during split operation ⟨1,2,1⟩.
    /// Represents a position after matching the first word character of a split.
    ///
    /// Same invariants as INonFinal: |offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ n
    ISplitting {
        /// Offset from parameter I (range: -n to n)
        offset: i32,

        /// Number of errors consumed (range: 0 to n)
        errors: usize,

        /// Phase 3b: Character read when entering the split state (first of 2-char input sequence)
        /// This is needed because when completing the split, we need to pair this with the
        /// current input character to form the 2-character sequence for phonetic validation.
        entry_char: char,
    },

    /// M-type splitting state: M + offset#errors_s
    ///
    /// Intermediate state during split operation ⟨1,2,1⟩ at final positions.
    ///
    /// Same invariants as MFinal: errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ n
    MSplitting {
        /// Offset from parameter M (range: -2n to 0)
        offset: i32,

        /// Number of errors consumed (range: 0 to n)
        errors: usize,

        /// Phase 3b: Character read when entering the split state (first of 2-char input sequence)
        entry_char: char,
    },
}

// Custom Ord implementation to sort by (errors, offset) for efficient subsumption checks
impl PartialOrd for GeneralizedPosition {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GeneralizedPosition {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        use GeneralizedPosition::*;

        // Sort by (variant_priority, errors, offset) to enable early termination in subsumption checks
        // Positions with fewer errors come first, making it easy to skip positions
        // that cannot participate in subsumption relationships
        //
        // Variant priority order (lower comes first):
        // I-types (0-2): INonFinal < ITransposing < ISplitting
        // M-types (3-5): MFinal < MTransposing < MSplitting

        let variant_priority = |pos: &GeneralizedPosition| match pos {
            INonFinal { .. } => 0,
            ITransposing { .. } => 1,
            ISplitting { .. } => 2,
            MFinal { .. } => 3,
            MTransposing { .. } => 4,
            MSplitting { .. } => 5,
        };

        // Splitting variants carry an `entry_char`; use it as a final tiebreaker so
        // `Ord` stays consistent with the derived `Eq`/`Hash`. Without it, two
        // ISplitting/MSplitting positions with equal (offset, errors) but distinct
        // `entry_char` would yield `cmp() == Equal` yet `eq() == false`, violating the
        // std Ord/Eq contract that `binary_search` and ordered containers rely on.
        let entry_char_key = |pos: &GeneralizedPosition| -> Option<char> {
            match pos {
                ISplitting { entry_char, .. } | MSplitting { entry_char, .. } => Some(*entry_char),
                _ => None,
            }
        };

        let p1 = variant_priority(self);
        let p2 = variant_priority(other);

        match p1.cmp(&p2) {
            Ordering::Equal => {
                // Same variant type, sort by (errors, offset, entry_char)
                let e1 = self.errors();
                let e2 = other.errors();
                match e1.cmp(&e2) {
                    Ordering::Equal => self
                        .offset()
                        .cmp(&other.offset())
                        .then_with(|| entry_char_key(self).cmp(&entry_char_key(other))),
                    other => other,
                }
            }
            other => other,
        }
    }
}

impl GeneralizedPosition {
    /// Create new I-type (non-final) position with invariant validation
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter I (must satisfy |t| ≤ k and -n ≤ t ≤ n)
    /// - `errors`: Number of errors k consumed (must satisfy 0 ≤ k ≤ n)
    /// - `max_distance`: Maximum edit distance n
    ///
    /// # Invariant
    ///
    /// `|offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ n`
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::generalized::GeneralizedPosition;
    ///
    /// let initial = GeneralizedPosition::new_i(0, 0, 2)
    ///     .expect("0#0 satisfies the generalized I-position invariant");
    /// let ahead = GeneralizedPosition::new_i(1, 1, 2)
    ///     .expect("1#1 satisfies the generalized I-position invariant");
    /// let behind = GeneralizedPosition::new_i(-2, 2, 2)
    ///     .expect("-2#2 satisfies the generalized I-position invariant");
    /// assert_eq!((initial.offset(), ahead.offset(), behind.offset()), (0, 1, -2));
    /// ```
    pub fn new_i(offset: i32, errors: u8, max_distance: u8) -> Result<Self, PositionError> {
        Self::new_i_scaled(
            offset,
            usize::from(errors),
            usize::from(max_distance),
            max_distance,
            1,
        )
    }

    pub(crate) fn new_i_scaled(
        offset: i32,
        errors: usize,
        max_cost: usize,
        max_distance: u8,
        scale_denominator: u32,
    ) -> Result<Self, PositionError> {
        let n = i32::from(max_distance);
        let unit_reachable = usize::try_from(offset.unsigned_abs())
            .ok()
            .and_then(|distance| distance.checked_mul(scale_denominator as usize))
            .is_some_and(|required| required <= errors);
        let invariant_satisfied = errors <= max_cost
            && offset >= -n
            && offset <= n
            && (scale_denominator != 1 || unit_reachable);

        if invariant_satisfied {
            Ok(GeneralizedPosition::INonFinal { offset, errors })
        } else {
            Err(PositionError::InvalidIPosition {
                offset,
                errors,
                max_distance: max_cost,
            })
        }
    }

    /// Create new M-type (final) position with invariant validation
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter M (must satisfy -2n ≤ t ≤ 0 and k ≥ -t - n)
    /// - `errors`: Number of errors k consumed (must satisfy 0 ≤ k ≤ n)
    /// - `max_distance`: Maximum edit distance n
    ///
    /// # Invariant
    ///
    /// `errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ n`
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::generalized::GeneralizedPosition;
    ///
    /// let at_end = GeneralizedPosition::new_m(0, 0, 2)
    ///     .expect("0#0 satisfies the generalized M-position invariant");
    /// let before_end = GeneralizedPosition::new_m(-1, 1, 2)
    ///     .expect("-1#1 satisfies the generalized M-position invariant");
    /// let farthest = GeneralizedPosition::new_m(-4, 2, 2)
    ///     .expect("-4#2 satisfies the generalized M-position invariant");
    /// assert_eq!((at_end.offset(), before_end.offset(), farthest.offset()), (0, -1, -4));
    /// ```
    pub fn new_m(offset: i32, errors: u8, max_distance: u8) -> Result<Self, PositionError> {
        Self::new_m_scaled(
            offset,
            usize::from(errors),
            usize::from(max_distance),
            max_distance,
            1,
        )
    }

    pub(crate) fn new_m_scaled(
        offset: i32,
        errors: usize,
        max_cost: usize,
        max_distance: u8,
        scale_denominator: u32,
    ) -> Result<Self, PositionError> {
        let n = i32::from(max_distance);
        let unit_reachable = i64::try_from(errors)
            .ok()
            .is_some_and(|cost| cost >= i64::from(-offset - n));
        let invariant_satisfied = errors <= max_cost
            && offset >= -2 * n
            && offset <= 0
            && (scale_denominator != 1 || unit_reachable);

        if invariant_satisfied {
            Ok(GeneralizedPosition::MFinal { offset, errors })
        } else {
            Err(PositionError::InvalidMPosition {
                offset,
                errors,
                max_distance: max_cost,
            })
        }
    }

    /// Create new I-type transposing position with invariant validation
    ///
    /// Transposing positions are intermediate states during transposition ⟨2,2,1⟩.
    /// They use the same invariants as INonFinal positions.
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter I (must satisfy |t| ≤ k and -n ≤ t ≤ n)
    /// - `errors`: Number of errors k consumed (must satisfy 0 ≤ k ≤ n)
    /// - `max_distance`: Maximum edit distance n
    ///
    /// # Invariant
    ///
    /// `|offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ n` (same as INonFinal)
    pub fn new_i_transposing(
        offset: i32,
        errors: u8,
        max_distance: u8,
    ) -> Result<Self, PositionError> {
        Self::new_i_transposing_scaled(
            offset,
            usize::from(errors),
            usize::from(max_distance),
            max_distance,
            1,
        )
    }

    pub(crate) fn new_i_transposing_scaled(
        offset: i32,
        errors: usize,
        max_cost: usize,
        max_distance: u8,
        scale_denominator: u32,
    ) -> Result<Self, PositionError> {
        let n = i32::from(max_distance);
        let unit_reachable = usize::try_from(offset.unsigned_abs())
            .ok()
            .and_then(|distance| distance.checked_mul(scale_denominator as usize))
            .is_some_and(|required| required <= errors);
        let invariant_satisfied = errors <= max_cost
            && offset >= -n
            && offset <= n
            && (scale_denominator != 1 || unit_reachable);

        if invariant_satisfied {
            Ok(GeneralizedPosition::ITransposing { offset, errors })
        } else {
            Err(PositionError::InvalidIPosition {
                offset,
                errors,
                max_distance: max_cost,
            })
        }
    }

    /// Create new M-type transposing position with invariant validation
    ///
    /// Transposing positions are intermediate states during transposition ⟨2,2,1⟩.
    /// They use the same invariants as MFinal positions.
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter M (must satisfy -2n ≤ t ≤ 0 and k ≥ -t - n)
    /// - `errors`: Number of errors k consumed (must satisfy 0 ≤ k ≤ n)
    /// - `max_distance`: Maximum edit distance n
    ///
    /// # Invariant
    ///
    /// `errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ n` (same as MFinal)
    pub fn new_m_transposing(
        offset: i32,
        errors: u8,
        max_distance: u8,
    ) -> Result<Self, PositionError> {
        Self::new_m_transposing_scaled(
            offset,
            usize::from(errors),
            usize::from(max_distance),
            max_distance,
            1,
        )
    }

    pub(crate) fn new_m_transposing_scaled(
        offset: i32,
        errors: usize,
        max_cost: usize,
        max_distance: u8,
        scale_denominator: u32,
    ) -> Result<Self, PositionError> {
        let n = i32::from(max_distance);
        let unit_reachable = i64::try_from(errors)
            .ok()
            .is_some_and(|cost| cost >= i64::from(-offset - n));
        let invariant_satisfied = errors <= max_cost
            && offset >= -2 * n
            && offset <= 0
            && (scale_denominator != 1 || unit_reachable);

        if invariant_satisfied {
            Ok(GeneralizedPosition::MTransposing { offset, errors })
        } else {
            Err(PositionError::InvalidMPosition {
                offset,
                errors,
                max_distance: max_cost,
            })
        }
    }

    /// Create new I-type splitting position with invariant validation
    ///
    /// Splitting positions are intermediate states during split ⟨1,2,1⟩.
    /// They use the same invariants as INonFinal positions.
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter I (must satisfy |t| ≤ k and -n ≤ t ≤ n)
    /// - `errors`: Number of errors k consumed (must satisfy 0 ≤ k ≤ n)
    /// - `max_distance`: Maximum edit distance n
    /// - `entry_char`: Character read when entering the split state (first of 2-char sequence)
    ///
    /// # Invariant
    ///
    /// `|offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ n` (same as INonFinal)
    pub fn new_i_splitting(
        offset: i32,
        errors: u8,
        max_distance: u8,
        entry_char: char,
    ) -> Result<Self, PositionError> {
        Self::new_i_splitting_scaled(
            offset,
            usize::from(errors),
            usize::from(max_distance),
            max_distance,
            1,
            entry_char,
        )
    }

    pub(crate) fn new_i_splitting_scaled(
        offset: i32,
        errors: usize,
        max_cost: usize,
        max_distance: u8,
        scale_denominator: u32,
        entry_char: char,
    ) -> Result<Self, PositionError> {
        let n = i32::from(max_distance);
        let reachable_with_intermediate = usize::try_from(offset.unsigned_abs())
            .ok()
            .and_then(|distance| {
                distance
                    .saturating_sub(1)
                    .checked_mul(scale_denominator as usize)
            })
            .is_some_and(|required| required <= errors);
        let invariant_satisfied = errors <= max_cost
            && offset >= -n
            && offset <= n
            && (scale_denominator != 1 || reachable_with_intermediate);

        if invariant_satisfied {
            Ok(GeneralizedPosition::ISplitting {
                offset,
                errors,
                entry_char,
            })
        } else {
            Err(PositionError::InvalidIPosition {
                offset,
                errors,
                max_distance: max_cost,
            })
        }
    }

    /// Create new M-type splitting position with invariant validation
    ///
    /// Splitting positions are intermediate states during split ⟨1,2,1⟩.
    /// They use the same invariants as MFinal positions.
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter M (must satisfy -2n ≤ t ≤ 0 and k ≥ -t - n)
    /// - `errors`: Number of errors k consumed (must satisfy 0 ≤ k ≤ n)
    /// - `max_distance`: Maximum edit distance n
    /// - `entry_char`: Character read when entering the split state (first of 2-char sequence)
    ///
    /// # Invariant
    ///
    /// `errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ n` (same as MFinal)
    pub fn new_m_splitting(
        offset: i32,
        errors: u8,
        max_distance: u8,
        entry_char: char,
    ) -> Result<Self, PositionError> {
        Self::new_m_splitting_scaled(
            offset,
            usize::from(errors),
            usize::from(max_distance),
            max_distance,
            1,
            entry_char,
        )
    }

    pub(crate) fn new_m_splitting_scaled(
        offset: i32,
        errors: usize,
        max_cost: usize,
        max_distance: u8,
        scale_denominator: u32,
        entry_char: char,
    ) -> Result<Self, PositionError> {
        let n = i32::from(max_distance);
        let reachable_with_intermediate = i64::try_from(errors)
            .ok()
            .and_then(|cost| cost.checked_add(i64::from(scale_denominator)))
            .is_some_and(|cost| cost >= i64::from(-offset - n));
        let invariant_satisfied = errors <= max_cost
            && offset >= -2 * n
            && offset <= 0
            && (scale_denominator != 1 || reachable_with_intermediate);

        if invariant_satisfied {
            Ok(GeneralizedPosition::MSplitting {
                offset,
                errors,
                entry_char,
            })
        } else {
            Err(PositionError::InvalidMPosition {
                offset,
                errors,
                max_distance: max_cost,
            })
        }
    }

    /// Get the offset value
    pub fn offset(&self) -> i32 {
        match self {
            GeneralizedPosition::INonFinal { offset, .. }
            | GeneralizedPosition::MFinal { offset, .. }
            | GeneralizedPosition::ITransposing { offset, .. }
            | GeneralizedPosition::MTransposing { offset, .. }
            | GeneralizedPosition::ISplitting { offset, .. }
            | GeneralizedPosition::MSplitting { offset, .. } => *offset,
        }
    }

    /// Get the exact scaled cost accumulated by this position.
    pub fn errors(&self) -> usize {
        match self {
            GeneralizedPosition::INonFinal { errors, .. }
            | GeneralizedPosition::MFinal { errors, .. }
            | GeneralizedPosition::ITransposing { errors, .. }
            | GeneralizedPosition::MTransposing { errors, .. }
            | GeneralizedPosition::ISplitting { errors, .. }
            | GeneralizedPosition::MSplitting { errors, .. } => *errors,
        }
    }

    pub(crate) fn checked_rescale(&self, multiplier: usize) -> Option<Self> {
        let errors = self.errors().checked_mul(multiplier)?;
        Some(match self {
            Self::INonFinal { offset, .. } => Self::INonFinal {
                offset: *offset,
                errors,
            },
            Self::MFinal { offset, .. } => Self::MFinal {
                offset: *offset,
                errors,
            },
            Self::ITransposing { offset, .. } => Self::ITransposing {
                offset: *offset,
                errors,
            },
            Self::MTransposing { offset, .. } => Self::MTransposing {
                offset: *offset,
                errors,
            },
            Self::ISplitting {
                offset, entry_char, ..
            } => Self::ISplitting {
                offset: *offset,
                errors,
                entry_char: *entry_char,
            },
            Self::MSplitting {
                offset, entry_char, ..
            } => Self::MSplitting {
                offset: *offset,
                errors,
                entry_char: *entry_char,
            },
        })
    }

    /// Check if this is an I-type (non-final) position
    ///
    /// Returns true for INonFinal, ITransposing, and ISplitting variants.
    pub fn is_non_final(&self) -> bool {
        matches!(
            self,
            GeneralizedPosition::INonFinal { .. }
                | GeneralizedPosition::ITransposing { .. }
                | GeneralizedPosition::ISplitting { .. }
        )
    }

    /// Check if this is an M-type (final) position
    ///
    /// Returns true for MFinal, MTransposing, and MSplitting variants.
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            GeneralizedPosition::MFinal { .. }
                | GeneralizedPosition::MTransposing { .. }
                | GeneralizedPosition::MSplitting { .. }
        )
    }
}

impl fmt::Display for GeneralizedPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeneralizedPosition::INonFinal { offset, errors } => {
                write!(f, "I + {}#{}", offset, errors)
            }
            GeneralizedPosition::MFinal { offset, errors } => {
                write!(f, "M + {}#{}", offset, errors)
            }
            GeneralizedPosition::ITransposing { offset, errors } => {
                write!(f, "I + {}#{}_t", offset, errors)
            }
            GeneralizedPosition::MTransposing { offset, errors } => {
                write!(f, "M + {}#{}_t", offset, errors)
            }
            GeneralizedPosition::ISplitting { offset, errors, .. } => {
                write!(f, "I + {}#{}_s", offset, errors)
            }
            GeneralizedPosition::MSplitting { offset, errors, .. } => {
                write!(f, "M + {}#{}_s", offset, errors)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_i_valid() {
        // I + 0#0 (initial state)
        let pos = GeneralizedPosition::new_i(0, 0, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        assert_eq!(pos.offset(), 0);
        assert_eq!(pos.errors(), 0);
        assert!(pos.is_non_final());

        // I + 1#1
        let pos = GeneralizedPosition::new_i(1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        assert_eq!(pos.offset(), 1);
        assert_eq!(pos.errors(), 1);

        // I + (-2)#2
        let pos = GeneralizedPosition::new_i(-2, 2, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        assert_eq!(pos.offset(), -2);
        assert_eq!(pos.errors(), 2);
    }

    #[test]
    fn test_new_i_invalid() {
        // |offset| > errors
        assert!(GeneralizedPosition::new_i(2, 1, 2).is_err());

        // offset > n
        assert!(GeneralizedPosition::new_i(3, 3, 2).is_err());

        // offset < -n
        assert!(GeneralizedPosition::new_i(-3, 3, 2).is_err());

        // errors > n
        assert!(GeneralizedPosition::new_i(0, 3, 2).is_err());
    }

    #[test]
    fn test_new_m_valid() {
        // M + 0#0
        let pos = GeneralizedPosition::new_m(0, 0, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        assert_eq!(pos.offset(), 0);
        assert_eq!(pos.errors(), 0);
        assert!(pos.is_final());

        // M + (-1)#1
        let pos = GeneralizedPosition::new_m(-1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        assert_eq!(pos.offset(), -1);
        assert_eq!(pos.errors(), 1);

        // M + (-4)#2
        let pos = GeneralizedPosition::new_m(-4, 2, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        assert_eq!(pos.offset(), -4);
        assert_eq!(pos.errors(), 2);
    }

    #[test]
    fn test_new_m_invalid() {
        // errors < -offset - n
        assert!(GeneralizedPosition::new_m(-4, 1, 2).is_err());

        // offset > 0
        assert!(GeneralizedPosition::new_m(1, 1, 2).is_err());

        // offset < -2n
        assert!(GeneralizedPosition::new_m(-5, 2, 2).is_err());

        // errors > n
        assert!(GeneralizedPosition::new_m(-1, 3, 2).is_err());
    }

    #[test]
    fn test_ordering() {
        let pos1 = GeneralizedPosition::new_i(0, 0, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos2 = GeneralizedPosition::new_i(1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos3 = GeneralizedPosition::new_i(-1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let pos4 = GeneralizedPosition::new_m(0, 0, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");

        // Sort by errors first
        assert!(pos1 < pos2);
        assert!(pos1 < pos3);

        // Within same error level, sort by offset
        assert!(pos3 < pos2);

        // I-type comes before M-type
        assert!(pos1 < pos4);
        assert!(pos2 < pos4);
    }

    #[test]
    fn test_display() {
        let pos1 = GeneralizedPosition::new_i(1, 2, 3)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        assert_eq!(format!("{}", pos1), "I + 1#2");

        let pos2 = GeneralizedPosition::new_m(-1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        assert_eq!(format!("{}", pos2), "M + -1#1");
    }

    // Tests for new position variants (Phase 2d.1)

    #[test]
    fn test_new_i_transposing_valid() {
        let pos = GeneralizedPosition::new_i_transposing(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        assert_eq!(pos.offset(), 0);
        assert_eq!(pos.errors(), 1);
        assert!(pos.is_non_final());
        assert!(!pos.is_final());

        // Test with negative offset
        let pos = GeneralizedPosition::new_i_transposing(-1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        assert_eq!(pos.offset(), -1);
        assert_eq!(pos.errors(), 1);
    }

    #[test]
    fn test_new_i_transposing_invalid() {
        // Same invariants as INonFinal
        assert!(GeneralizedPosition::new_i_transposing(3, 1, 2).is_err()); // offset > n
        assert!(GeneralizedPosition::new_i_transposing(2, 1, 2).is_err()); // |offset| > errors
    }

    #[test]
    fn test_new_m_transposing_valid() {
        let pos = GeneralizedPosition::new_m_transposing(0, 0, 2)
            .expect("test fixture: GeneralizedPosition::new_m_transposing with valid args");
        assert_eq!(pos.offset(), 0);
        assert_eq!(pos.errors(), 0);
        assert!(!pos.is_non_final());
        assert!(pos.is_final());

        let pos = GeneralizedPosition::new_m_transposing(-1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m_transposing with valid args");
        assert_eq!(pos.offset(), -1);
        assert_eq!(pos.errors(), 1);
    }

    #[test]
    fn test_new_m_transposing_invalid() {
        // Same invariants as MFinal
        assert!(GeneralizedPosition::new_m_transposing(-4, 1, 2).is_err()); // errors < -offset - n
        assert!(GeneralizedPosition::new_m_transposing(1, 1, 2).is_err()); // offset > 0
    }

    #[test]
    fn test_new_i_splitting_valid() {
        let pos = GeneralizedPosition::new_i_splitting(0, 1, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");
        assert_eq!(pos.offset(), 0);
        assert_eq!(pos.errors(), 1);
        assert!(pos.is_non_final());
        assert!(!pos.is_final());

        let pos = GeneralizedPosition::new_i_splitting(-2, 2, 2, 'b')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");
        assert_eq!(pos.offset(), -2);
        assert_eq!(pos.errors(), 2);
    }

    #[test]
    fn test_new_i_splitting_invalid() {
        // Same invariants as INonFinal
        assert!(GeneralizedPosition::new_i_splitting(3, 1, 2, 'a').is_err()); // offset > n
        assert!(GeneralizedPosition::new_i_splitting(-3, 2, 2, 'a').is_err()); // offset < -n
    }

    #[test]
    fn test_new_m_splitting_valid() {
        let pos = GeneralizedPosition::new_m_splitting(0, 0, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_m_splitting with valid args");
        assert_eq!(pos.offset(), 0);
        assert_eq!(pos.errors(), 0);
        assert!(!pos.is_non_final());
        assert!(pos.is_final());

        let pos = GeneralizedPosition::new_m_splitting(-2, 2, 2, 'b')
            .expect("test fixture: GeneralizedPosition::new_m_splitting with valid args");
        assert_eq!(pos.offset(), -2);
        assert_eq!(pos.errors(), 2);
    }

    #[test]
    fn test_new_m_splitting_invalid() {
        // Same invariants as MFinal
        assert!(GeneralizedPosition::new_m_splitting(-5, 2, 2, 'a').is_err()); // offset < -2n
        assert!(GeneralizedPosition::new_m_splitting(1, 1, 2, 'a').is_err()); // offset > 0
    }

    #[test]
    fn test_display_transposing() {
        let pos = GeneralizedPosition::new_i_transposing(1, 2, 3)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        assert_eq!(format!("{}", pos), "I + 1#2_t");

        let pos = GeneralizedPosition::new_m_transposing(-1, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m_transposing with valid args");
        assert_eq!(format!("{}", pos), "M + -1#1_t");
    }

    #[test]
    fn test_display_splitting() {
        let pos = GeneralizedPosition::new_i_splitting(0, 1, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");
        assert_eq!(format!("{}", pos), "I + 0#1_s");

        let pos = GeneralizedPosition::new_m_splitting(-2, 2, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_m_splitting with valid args");
        assert_eq!(format!("{}", pos), "M + -2#2_s");
    }

    #[test]
    fn test_ordering_with_new_variants() {
        let i_normal = GeneralizedPosition::new_i(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i with valid args");
        let i_transposing = GeneralizedPosition::new_i_transposing(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_i_transposing with valid args");
        let i_splitting = GeneralizedPosition::new_i_splitting(0, 1, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_i_splitting with valid args");
        let m_normal = GeneralizedPosition::new_m(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m with valid args");
        let m_transposing = GeneralizedPosition::new_m_transposing(0, 1, 2)
            .expect("test fixture: GeneralizedPosition::new_m_transposing with valid args");
        let m_splitting = GeneralizedPosition::new_m_splitting(0, 1, 2, 'a')
            .expect("test fixture: GeneralizedPosition::new_m_splitting with valid args");

        // I-types come before M-types
        assert!(i_normal < m_normal);
        assert!(i_transposing < m_transposing);
        assert!(i_splitting < m_splitting);

        // Within I-types: INonFinal < ITransposing < ISplitting
        assert!(i_normal < i_transposing);
        assert!(i_transposing < i_splitting);

        // Within M-types: MFinal < MTransposing < MSplitting
        assert!(m_normal < m_transposing);
        assert!(m_transposing < m_splitting);
    }

    /// Regression: `Ord` must agree with the derived `Eq`/`Hash`, including the
    /// `entry_char` carried by the Splitting variants. Before the fix, two
    /// ISplitting/MSplitting positions differing only in `entry_char` compared
    /// `Equal` yet were `!=`, violating the std Ord/Eq contract that
    /// `binary_search` and ordered containers depend on.
    #[test]
    fn ord_is_consistent_with_eq_for_splitting_entry_char() {
        use std::cmp::Ordering;

        let a = GeneralizedPosition::new_i_splitting(0, 1, 2, 'a')
            .expect("test fixture: valid ISplitting");
        let b = GeneralizedPosition::new_i_splitting(0, 1, 2, 'b')
            .expect("test fixture: valid ISplitting");

        // Differ only in entry_char: must be unequal AND cmp must not be Equal.
        assert_ne!(a, b);
        assert_ne!(a.cmp(&b), Ordering::Equal);
        assert_eq!(a.cmp(&b), Ordering::Less); // 'a' < 'b'
        assert_eq!(b.cmp(&a), Ordering::Greater);

        // Identical entry_char: equal AND cmp == Equal.
        let a2 = GeneralizedPosition::new_i_splitting(0, 1, 2, 'a')
            .expect("test fixture: valid ISplitting");
        assert_eq!(a, a2);
        assert_eq!(a.cmp(&a2), Ordering::Equal);

        // Exhaustive contract check across a mixed set: (cmp == Equal) iff (eq).
        let positions = [
            GeneralizedPosition::new_i(0, 1, 2).expect("test fixture: valid INonFinal"),
            GeneralizedPosition::new_i_splitting(0, 1, 2, 'a')
                .expect("test fixture: valid ISplitting"),
            GeneralizedPosition::new_i_splitting(0, 1, 2, 'b')
                .expect("test fixture: valid ISplitting"),
            GeneralizedPosition::new_m_splitting(-2, 2, 2, 'a')
                .expect("test fixture: valid MSplitting"),
            GeneralizedPosition::new_m_splitting(-2, 2, 2, 'b')
                .expect("test fixture: valid MSplitting"),
        ];
        for x in &positions {
            for y in &positions {
                assert_eq!(
                    x.cmp(y) == Ordering::Equal,
                    x == y,
                    "Ord/Eq contract violated for {x:?} vs {y:?}"
                );
            }
        }
    }
}
