//! Universal Position Types
//!
//! Implements universal positions from Mitankin's thesis (Definition 15, pages 30-33).
//!
//! # Theory Background
//!
//! Universal positions use parameters I (non-final) and M (final) instead of concrete word indices:
//! - `I + t#k`: Non-final position, offset t from start, k errors consumed
//! - `M + t#k`: Final position, offset t from end, k errors consumed
//!
//! ## Invariants (from Definition 15)
//!
//! **I-type (non-final)**:
//! ```text
//! I^ε_s = {I + t#k | |t| ≤ k ∧ -n ≤ t ≤ n ∧ 0 ≤ k ≤ n}
//! ```
//!
//! **M-type (final)**:
//! ```text
//! M^ε_s = {M + t#k | k ≥ -t - n ∧ -2n ≤ t ≤ 0 ∧ 0 ≤ k ≤ n}
//! ```
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::transducer::universal::{UniversalPosition, Standard};
//!
//! // Create I-type position: I + 0#0 (initial state)
//! let initial = UniversalPosition::<Standard>::new_i(0, 0, 2)?;
//!
//! // Create M-type position: M + (-1)#1 (one error, one char before end)
//! let final_pos = UniversalPosition::<Standard>::new_m(-1, 1, 2)?;
//! ```

use std::fmt;

/// Error type for invalid position creation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionError {
    /// I-type position violates invariant |t| ≤ k ∧ -n ≤ t ≤ n ∧ 0 ≤ k ≤ n
    InvalidIPosition {
        /// The offset value that violated the invariant
        offset: i32,
        /// The error count that violated the invariant
        errors: u8,
        /// The maximum distance n
        max_distance: u8,
    },

    /// M-type position violates invariant k ≥ -t - n ∧ -2n ≤ t ≤ 0 ∧ 0 ≤ k ≤ n
    InvalidMPosition {
        /// The offset value that violated the invariant
        offset: i32,
        /// The error count that violated the invariant
        errors: u8,
        /// The maximum distance n
        max_distance: u8,
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
                    "Invalid I-position: I + {}#{} with n={}. \
                     Invariant: |t| ≤ k ∧ -n ≤ t ≤ n ∧ 0 ≤ k ≤ n",
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
                    "Invalid M-position: M + {}#{} with n={}. \
                     Invariant: k ≥ -t - n ∧ -2n ≤ t ≤ 0 ∧ 0 ≤ k ≤ n",
                    offset, errors, max_distance
                )
            }
        }
    }
}

impl std::error::Error for PositionError {}

/// Trait for position variants (Standard, Transposition, MergeAndSplit)
///
/// This trait distinguishes between the three distance variants from the thesis:
/// - χ = ε (Standard Levenshtein)
/// - χ = t (with Transposition)
/// - χ = ms (with Merge/Split)
pub trait PositionVariant: Clone + fmt::Debug + PartialEq + Eq + std::hash::Hash {
    /// State type for this variant
    ///
    /// - Standard: `()` (zero-sized, no state needed)
    /// - Transposition: `TranspositionState` (Usual or Transposing)
    /// - MergeAndSplit: `MergeSplitState` (Usual or Splitting)
    type State: Clone + fmt::Debug + PartialEq + Eq + std::hash::Hash + Default;

    /// Human-readable variant name
    fn variant_name() -> &'static str;

    /// Compute successors for I-type positions with this variant
    ///
    /// Each variant implements variant-specific successor generation logic.
    fn compute_i_successors(
        offset: i32,
        errors: u8,
        variant_state: &Self::State,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>>;

    /// Compute successors for M-type positions with this variant
    ///
    /// Each variant implements variant-specific successor generation logic.
    fn compute_m_successors(
        offset: i32,
        errors: u8,
        variant_state: &Self::State,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>>;
}

/// Standard Levenshtein distance variant (χ = ε)
///
/// Operations: insertion, deletion, substitution
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Standard;

impl PositionVariant for Standard {
    type State = ();

    fn variant_name() -> &'static str {
        "Standard"
    }

    fn compute_i_successors(
        offset: i32,
        errors: u8,
        _variant_state: &Self::State,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>> {
        UniversalPosition::<Self>::successors_i_type_standard(
            offset,
            errors,
            bit_vector,
            max_distance,
        )
    }

    fn compute_m_successors(
        offset: i32,
        errors: u8,
        _variant_state: &Self::State,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>> {
        UniversalPosition::<Self>::successors_m_type_standard(
            offset,
            errors,
            bit_vector,
            max_distance,
        )
    }
}

/// Transposition state for tracking transposition operations
///
/// From Mitankin's thesis (Definition 7, Page 16):
/// - **Usual**: Regular position i#e
/// - **Transposing**: Transposition state i#e_t (waiting to complete swap)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum TranspositionState {
    /// Regular position (usual type)
    #[default]
    Usual,

    /// Transposition state (waiting to complete swap)
    /// Corresponds to i#e_t notation in thesis
    Transposing,
}

/// Transposition variant (χ = t)
///
/// Operations: insertion, deletion, substitution, **transposition**
///
/// Note: d^t_L does NOT satisfy triangle inequality (see thesis page 3)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Transposition;

impl PositionVariant for Transposition {
    type State = TranspositionState;

    fn variant_name() -> &'static str {
        "Transposition"
    }

    fn compute_i_successors(
        offset: i32,
        errors: u8,
        variant_state: &Self::State,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>> {
        match variant_state {
            TranspositionState::Usual => {
                // Get standard successors for usual state
                let mut successors = UniversalPosition::<Self>::successors_i_type_standard(
                    offset,
                    errors,
                    bit_vector,
                    max_distance,
                );

                // Add transposition initiation: δ^D,t_e(i#e, b) includes {i#(e+1)_t} if b[1] = 1 ∧ e < n
                // b[1] refers to position n+offset+1 in the bit vector (next position after current)
                // Cross-validated with lazy automaton: creates i#(e+1)_t (same i, +1 error)
                let next_match_index = (max_distance as i32 + offset + 1) as usize;
                if next_match_index < bit_vector.len()
                    && bit_vector.is_match(next_match_index)
                    && errors < max_distance
                {
                    // Enter transposition state: i#(e+1)_t (same word position i, one more error)
                    // Universal offset for i#(e+1): offset = i - (e+1) = (i-e) - 1 = offset - 1
                    if let Ok(trans) = UniversalPosition::new_i_with_state(
                        offset - 1,
                        errors + 1,
                        max_distance,
                        TranspositionState::Transposing,
                    ) {
                        successors.push(trans);
                    }
                }

                successors
            }
            TranspositionState::Transposing => {
                // In transposition state: δ^D,t_e(i#e_t, b) = {(i+2)#(e-1)} if b[0] = 1, else ∅
                // Cross-validated with lazy automaton: checks cv[0] and creates (i+2)#e
                let match_index = (max_distance as i32 + offset) as usize;

                if match_index < bit_vector.len() && bit_vector.is_match(match_index) {
                    // Complete transposition: i#(e+1)_t → (i+2)#e
                    // At input k: current word position i = offset + k
                    // Lazy creates: (i+2)#(e-1) (absolute position i+2)
                    // At next input k+1: need offset' such that offset' + (k+1) = i+2 = (offset+k)+2
                    // Therefore: offset' = (offset+k+2) - (k+1) = offset + 1
                    if let Ok(succ) = UniversalPosition::new_i_with_state(
                        offset + 1,
                        errors - 1,
                        max_distance,
                        TranspositionState::Usual,
                    ) {
                        vec![succ]
                    } else {
                        vec![]
                    }
                } else {
                    // Transposition failed
                    vec![]
                }
            }
        }
    }

    fn compute_m_successors(
        offset: i32,
        errors: u8,
        variant_state: &Self::State,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>> {
        match variant_state {
            TranspositionState::Usual => {
                // Get standard successors for usual state
                let mut successors = UniversalPosition::<Self>::successors_m_type_standard(
                    offset,
                    errors,
                    bit_vector,
                    max_distance,
                );

                // Add transposition initiation: δ^D,t_e(i#e, b) includes {i#(e+1)_t} if b[1] = 1 ∧ e < n
                // Cross-validated with lazy automaton: creates i#(e+1)_t (same i, +1 error)
                let next_match_index = (max_distance as i32 + offset + 1) as usize;
                if next_match_index < bit_vector.len()
                    && bit_vector.is_match(next_match_index)
                    && errors < max_distance
                {
                    // Enter transposition state: i#(e+1)_t (same word position i, one more error)
                    // Universal offset for i#(e+1): offset = i - (e+1) = (i-e) - 1 = offset - 1
                    if let Ok(trans) = UniversalPosition::new_m_with_state(
                        offset - 1,
                        errors + 1,
                        max_distance,
                        TranspositionState::Transposing,
                    ) {
                        successors.push(trans);
                    }
                }

                successors
            }
            TranspositionState::Transposing => {
                // In transposition state: δ^D,t_e(i#e_t, b) = {(i+2)#(e-1)} if b[0] = 1, else ∅
                // Cross-validated with lazy automaton: checks cv[0] and creates (i+2)#e
                let match_index = (max_distance as i32 + offset) as usize;

                if match_index < bit_vector.len() && bit_vector.is_match(match_index) {
                    // Complete transposition: i#(e+1)_t → (i+2)#e
                    // At input k: current word position i = offset + k
                    // Lazy creates: (i+2)#(e-1) (absolute position i+2)
                    // At next input k+1: need offset' such that offset' + (k+1) = i+2 = (offset+k)+2
                    // Therefore: offset' = (offset+k+2) - (k+1) = offset + 1
                    if let Ok(succ) = UniversalPosition::new_m_with_state(
                        offset + 1,
                        errors - 1,
                        max_distance,
                        TranspositionState::Usual,
                    ) {
                        vec![succ]
                    } else {
                        vec![]
                    }
                } else {
                    // Transposition failed
                    vec![]
                }
            }
        }
    }
}

/// Merge/Split state for tracking merge and split operations
///
/// From Mitankin's thesis (Definition 7, Page 16):
/// - **Usual**: Regular position i#e
/// - **Splitting**: Split state i#e_s (waiting to emit second character)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum MergeSplitState {
    /// Regular position (usual type)
    #[default]
    Usual,

    /// Split state (waiting to emit second character)
    /// Corresponds to i#e_s notation in thesis
    Splitting,
}

/// Merge/Split variant (χ = ms)
///
/// Operations: insertion, deletion, substitution, **merge** (2→1), **split** (1→2)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MergeAndSplit;

impl PositionVariant for MergeAndSplit {
    type State = MergeSplitState;

    fn variant_name() -> &'static str {
        "MergeAndSplit"
    }

    fn compute_i_successors(
        offset: i32,
        errors: u8,
        variant_state: &Self::State,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>> {
        let is_splitting = matches!(variant_state, MergeSplitState::Splitting);

        // Start with standard operations (always included - merge/split is ADDITIVE)
        let mut successors = UniversalPosition::<Self>::successors_i_type_standard(
            offset,
            errors,
            bit_vector,
            max_distance,
        );

        // Bit vector indices
        let match_index = (max_distance as i32 + offset) as usize;
        let next_match_index = (max_distance as i32 + offset + 1) as usize;

        if is_splitting {
            // Split Completion: i#(e+1)_s → (i+1)#e
            // At input k: current word position i = offset + k
            // Lazy creates: (i+1)#e (absolute position i+1)
            // At next input k+1: need offset' such that offset' + (k+1) = i+1 = (offset+k)+1
            // Therefore: offset' = (offset+k+1) - (k+1) = offset + 0
            if match_index < bit_vector.len() && bit_vector.is_match(match_index) {
                if let Ok(succ) = UniversalPosition::new_i_with_state(
                    offset,     // offset + 0
                    errors - 1, // Complete split: decrement error back
                    max_distance,
                    MergeSplitState::Usual,
                ) {
                    successors.push(succ);
                }
            }
        } else {
            // Not splitting - can enter merge or split

            // Merge Operation: i#e → (i+2)#(e+1)
            // Merge: consume 2 input chars, 1 word char
            // At input k: current word position i = offset + k
            // Lazy creates: (i+2)#(e+1) (absolute position i+2)
            // At next input k+1: need offset' such that offset' + (k+1) = i+2 = (offset+k)+2
            // Therefore: offset' = (offset+k+2) - (k+1) = offset + 1
            if next_match_index < bit_vector.len()
                && bit_vector.is_match(next_match_index)
                && errors < max_distance
            {
                if let Ok(merge) = UniversalPosition::new_i_with_state(
                    offset + 1,
                    errors + 1,
                    max_distance,
                    MergeSplitState::Usual,
                ) {
                    successors.push(merge);
                }
            }

            // Split Entry: i#e → i#(e+1)_s
            // Split: one input char becomes two word chars
            // At input k: current word position i = offset + k
            // Lazy creates: i#(e+1)_s (same absolute position i, entering split state)
            // At next input k+1: need offset' such that offset' + (k+1) = i = offset+k
            // Therefore: offset' = (offset+k) - (k+1) = offset - 1
            if match_index < bit_vector.len()
                && bit_vector.is_match(match_index)
                && errors < max_distance
            {
                if let Ok(split) = UniversalPosition::new_i_with_state(
                    offset - 1,
                    errors + 1,
                    max_distance,
                    MergeSplitState::Splitting,
                ) {
                    successors.push(split);
                }
            }
        }

        successors
    }

    fn compute_m_successors(
        offset: i32,
        errors: u8,
        variant_state: &Self::State,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<UniversalPosition<Self>> {
        let is_splitting = matches!(variant_state, MergeSplitState::Splitting);

        // Start with standard operations (always included - merge/split is ADDITIVE)
        let mut successors = UniversalPosition::<Self>::successors_m_type_standard(
            offset,
            errors,
            bit_vector,
            max_distance,
        );

        // Bit vector indices
        let match_index = (max_distance as i32 + offset) as usize;
        let next_match_index = (max_distance as i32 + offset + 1) as usize;

        if is_splitting {
            // Split Completion: i#(e+1)_s → (i+1)#e
            // Same offset calculation as I-type
            if match_index < bit_vector.len() && bit_vector.is_match(match_index) {
                if let Ok(succ) = UniversalPosition::new_m_with_state(
                    offset,     // offset + 0
                    errors - 1, // Complete split: decrement error back
                    max_distance,
                    MergeSplitState::Usual,
                ) {
                    successors.push(succ);
                }
            }
        } else {
            // Not splitting - can enter merge or split

            // Merge Operation: i#e → (i+2)#(e+1)
            // Same offset calculation as I-type
            if next_match_index < bit_vector.len()
                && bit_vector.is_match(next_match_index)
                && errors < max_distance
            {
                if let Ok(merge) = UniversalPosition::new_m_with_state(
                    offset + 1,
                    errors + 1,
                    max_distance,
                    MergeSplitState::Usual,
                ) {
                    successors.push(merge);
                }
            }

            // Split Entry: i#e → i#(e+1)_s
            // Same offset calculation as I-type
            if match_index < bit_vector.len()
                && bit_vector.is_match(match_index)
                && errors < max_distance
            {
                if let Ok(split) = UniversalPosition::new_m_with_state(
                    offset - 1,
                    errors + 1,
                    max_distance,
                    MergeSplitState::Splitting,
                ) {
                    successors.push(split);
                }
            }
        }

        successors
    }
}

/// Universal position with parameter (I or M)
///
/// From Definition 15 (thesis pages 30-33).
///
/// # Type Parameters
///
/// - `V`: Position variant (Standard, Transposition, or MergeAndSplit)
///
/// # Notation Mapping
///
/// Theory → Rust:
/// - `I + t#k` → `INonFinal { offset: t, errors: k, variant_state: V::State::default() }`
/// - `M + t#k` → `MFinal { offset: t, errors: k, variant_state: V::State::default() }`
/// - `I + t#k_t` → `INonFinal { .., variant_state: TranspositionState::Transposing }`
/// - `I + t#k_s` → `INonFinal { .., variant_state: MergeSplitState::Splitting }`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UniversalPosition<V: PositionVariant> {
    /// I-type (non-final): I + offset#errors
    ///
    /// Represents position relative to start of word.
    ///
    /// Invariant: |offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ n
    INonFinal {
        /// Offset from parameter I (range: -n to n)
        offset: i32,

        /// Number of errors consumed (range: 0 to n)
        errors: u8,

        /// Variant-specific state (e.g., transposition or split state)
        ///
        /// - Standard: `()` (zero-sized)
        /// - Transposition: `TranspositionState` (Usual or Transposing)
        /// - MergeAndSplit: `MergeSplitState` (Usual or Splitting)
        variant_state: V::State,
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
        errors: u8,

        /// Variant-specific state (e.g., transposition or split state)
        variant_state: V::State,
    },
}

// Custom Ord implementation to sort by (errors, offset) for efficient subsumption checks
impl<V: PositionVariant> PartialOrd for UniversalPosition<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<V: PositionVariant> Ord for UniversalPosition<V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        use UniversalPosition::*;

        // Sort by (errors, offset) to enable early termination in subsumption checks
        // Positions with fewer errors come first, making it easy to skip positions
        // that cannot participate in subsumption relationships
        match (self, other) {
            (
                INonFinal {
                    errors: e1,
                    offset: o1,
                    ..
                },
                INonFinal {
                    errors: e2,
                    offset: o2,
                    ..
                },
            )
            | (
                MFinal {
                    errors: e1,
                    offset: o1,
                    ..
                },
                MFinal {
                    errors: e2,
                    offset: o2,
                    ..
                },
            ) => match e1.cmp(e2) {
                Ordering::Equal => o1.cmp(o2),
                other => other,
            },
            // I-type comes before M-type
            (INonFinal { .. }, MFinal { .. }) => Ordering::Less,
            (MFinal { .. }, INonFinal { .. }) => Ordering::Greater,
        }
    }
}

impl<V: PositionVariant> UniversalPosition<V> {
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
    /// ```ignore
    /// let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)?;  // I + 0#0
    /// let pos = UniversalPosition::<Standard>::new_i(1, 1, 2)?;  // I + 1#1
    /// let pos = UniversalPosition::<Standard>::new_i(-2, 2, 2)?; // I + (-2)#2
    /// ```
    pub fn new_i(offset: i32, errors: u8, max_distance: u8) -> Result<Self, PositionError> {
        let n = max_distance as i32;

        // Check invariant: |offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ max_distance
        if offset.abs() as u8 > errors || offset < -n || offset > n || errors > max_distance {
            return Err(PositionError::InvalidIPosition {
                offset,
                errors,
                max_distance,
            });
        }

        Ok(Self::INonFinal {
            offset,
            errors,
            variant_state: V::State::default(),
        })
    }

    /// Create new M-type (final) position with invariant validation
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter M (must satisfy -2n ≤ t ≤ 0)
    /// - `errors`: Number of errors k consumed (must satisfy k ≥ -t - n and 0 ≤ k ≤ n)
    /// - `max_distance`: Maximum edit distance n
    ///
    /// # Invariant
    ///
    /// `errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ n`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pos = UniversalPosition::<Standard>::new_m(0, 0, 2)?;   // M + 0#0
    /// let pos = UniversalPosition::<Standard>::new_m(-1, 1, 2)?;  // M + (-1)#1
    /// let pos = UniversalPosition::<Standard>::new_m(-2, 0, 2)?;  // M + (-2)#0
    /// ```
    pub fn new_m(offset: i32, errors: u8, max_distance: u8) -> Result<Self, PositionError> {
        let n = max_distance as i32;

        // Check invariant: errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ max_distance
        if (errors as i32) < -offset - n || offset < -2 * n || offset > 0 || errors > max_distance {
            return Err(PositionError::InvalidMPosition {
                offset,
                errors,
                max_distance,
            });
        }

        Ok(Self::MFinal {
            offset,
            errors,
            variant_state: V::State::default(),
        })
    }

    /// Create new I-type position with custom variant state
    ///
    /// This allows creating positions in specific variant states (e.g., transposition state).
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter I
    /// - `errors`: Number of errors k consumed
    /// - `max_distance`: Maximum edit distance n
    /// - `variant_state`: Variant-specific state (e.g., TranspositionState::Transposing)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create transposition state position
    /// let pos = UniversalPosition::<Transposition>::new_i_with_state(
    ///     1, 1, 2, TranspositionState::Transposing
    /// )?;
    /// ```
    pub fn new_i_with_state(
        offset: i32,
        errors: u8,
        max_distance: u8,
        variant_state: V::State,
    ) -> Result<Self, PositionError> {
        let n = max_distance as i32;

        // Check invariant: |offset| ≤ errors ∧ -n ≤ offset ≤ n ∧ 0 ≤ errors ≤ max_distance
        if offset.abs() as u8 > errors || offset < -n || offset > n || errors > max_distance {
            return Err(PositionError::InvalidIPosition {
                offset,
                errors,
                max_distance,
            });
        }

        Ok(Self::INonFinal {
            offset,
            errors,
            variant_state,
        })
    }

    /// Create new M-type position with custom variant state
    ///
    /// # Arguments
    ///
    /// - `offset`: Offset t from parameter M
    /// - `errors`: Number of errors k consumed
    /// - `max_distance`: Maximum edit distance n
    /// - `variant_state`: Variant-specific state
    pub fn new_m_with_state(
        offset: i32,
        errors: u8,
        max_distance: u8,
        variant_state: V::State,
    ) -> Result<Self, PositionError> {
        let n = max_distance as i32;

        // Check invariant: errors ≥ -offset - n ∧ -2n ≤ offset ≤ 0 ∧ 0 ≤ errors ≤ max_distance
        if (errors as i32) < -offset - n || offset < -2 * n || offset > 0 || errors > max_distance {
            return Err(PositionError::InvalidMPosition {
                offset,
                errors,
                max_distance,
            });
        }

        Ok(Self::MFinal {
            offset,
            errors,
            variant_state,
        })
    }

    /// Get the offset value
    pub fn offset(&self) -> i32 {
        match self {
            Self::INonFinal { offset, .. } | Self::MFinal { offset, .. } => *offset,
        }
    }

    /// Get the variant state
    pub fn variant_state(&self) -> &V::State {
        match self {
            Self::INonFinal { variant_state, .. } | Self::MFinal { variant_state, .. } => {
                variant_state
            }
        }
    }

    /// Get the error count
    pub fn errors(&self) -> u8 {
        match self {
            Self::INonFinal { errors, .. } | Self::MFinal { errors, .. } => *errors,
        }
    }

    /// Check if this is an I-type (non-final) position
    pub fn is_i_type(&self) -> bool {
        matches!(self, Self::INonFinal { .. })
    }

    /// Check if this is an M-type (final) position
    pub fn is_m_type(&self) -> bool {
        matches!(self, Self::MFinal { .. })
    }

    /// Compute successor positions (generic dispatcher)
    ///
    /// This method provides a generic interface for successor computation that works
    /// with any variant type. For concrete types (Standard, Transposition, MergeAndSplit),
    /// specialized implementations with variant-specific logic will be used when the
    /// type is known at compile time.
    ///
    /// # Note
    ///
    /// This generic version delegates to the standard successor methods. Specialized
    /// implementations (see `impl UniversalPosition<Standard>`, etc.) provide
    /// variant-specific behavior when the concrete type is known.
    pub fn successors(
        &self,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<Self> {
        // Use trait-based dispatch for variant-specific successor logic
        match self {
            Self::INonFinal {
                offset,
                errors,
                variant_state,
            } => V::compute_i_successors(*offset, *errors, variant_state, bit_vector, max_distance),
            Self::MFinal {
                offset,
                errors,
                variant_state,
            } => V::compute_m_successors(*offset, *errors, variant_state, bit_vector, max_distance),
        }
    }

    /// Compute successors for I-type positions with Standard variant (χ = ε)
    ///
    /// Uses I^ε conversion: I^ε({i#e}) = {I + (i-1)#e}
    ///
    /// # Theory (Definition 7, page 14; Definition 18, page 46)
    ///
    /// ```text
    /// δ^D,ε_e(i#e, b) = {
    ///     {i+1#e}                              if 1 < b (match)
    ///     {i#e+1, i+1#e+1}                     if b = 0^k & e < n (all zeros)
    ///     {i#e+1, i+1#e+1, i+j#e+j-1}         if 0 < b & j = μz[bz = 1]
    ///     {i#e+1}                              if b = ε & e < n (empty)
    ///     ∅                                    otherwise
    /// }
    /// ```
    ///
    /// **Operations**:
    /// - **Match** (1 < b): Advance position without error
    /// - **Delete** (i#e+1): Delete from input, advance error
    /// - **Insert** (i+1#e+1): Insert into input, advance position and error
    /// - **Skip to match** (i+j#e+j-1): Delete j-1 chars to reach match at position j
    fn successors_i_type_standard(
        offset: i32,
        errors: u8,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<Self> {
        let mut successors = Vec::new();

        // Universal position I + t#e uses offset t directly for δ^D,ε_e
        // After δ^D,ε_e, we convert back using I^ε({i#e}) = {I + (i-1)#e}

        // Case 1: Check if there's a match at the current word position
        // For I+offset#errors at input k, the concrete word position is i = offset + k
        // In bit vector s_n(w,k) (which starts at k-n), position i corresponds to index: i - (k-n) = offset + n
        let match_index = (max_distance as i32 + offset) as usize;

        if match_index < bit_vector.len() {
            if bit_vector.is_match(match_index) {
                // MATCH at current word position
                // δ^D,ε_e(t#e, b) where b[n+t] = 1
                // Result: (t+1)#e → I^ε → I+t#e
                if let Ok(succ) = Self::new_i(offset, errors, max_distance) {
                    successors.push(succ);
                }
                return successors;
            } else {
                // NO MATCH at current word position - generate error successors
                if errors < max_distance {
                    // DELETE: skip word character
                    // δ^D,ε_e: t#(e+1) → I^ε → I+(t-1)#(e+1)
                    if let Ok(succ) = Self::new_i(offset - 1, errors + 1, max_distance) {
                        successors.push(succ);
                    }

                    // SUBSTITUTE: advance word and input, add error
                    // δ^D,ε_e: (t+1)#(e+1) → I^ε → I+t#(e+1)
                    if let Ok(succ) = Self::new_i(offset, errors + 1, max_distance) {
                        successors.push(succ);
                    }

                    // SKIP-TO-MATCH: if there's a match later in window
                    // Skip j positions (deleting j characters), consuming j errors
                    for idx in (match_index + 1)..bit_vector.len() {
                        if bit_vector.is_match(idx) {
                            let skip_distance = (idx - match_index) as i32;
                            let new_offset = offset + skip_distance;
                            let new_errors = errors + skip_distance as u8;

                            if new_errors <= max_distance {
                                if let Ok(succ) = Self::new_i(new_offset, new_errors, max_distance)
                                {
                                    successors.push(succ);
                                }
                            }
                            break;
                        }
                    }
                }
                return successors;
            }
        }

        // Position beyond visible window - use generic error transitions
        if errors >= max_distance {
            return successors;
        }

        // Case: b = ε (empty bit vector)
        if bit_vector.is_empty() {
            if let Ok(succ) = Self::new_i(offset - 1, errors + 1, max_distance) {
                successors.push(succ);
            }
            return successors;
        }

        // Remaining cases: generic error transitions for positions outside window
        // Delete
        if let Ok(succ) = Self::new_i(offset - 1, errors + 1, max_distance) {
            successors.push(succ);
        }

        // Insert/Substitute
        if let Ok(succ) = Self::new_i(offset, errors + 1, max_distance) {
            successors.push(succ);
        }

        successors
    }

    /// Compute successors for M-type positions with Standard variant (χ = ε)
    ///
    /// Uses M^ε conversion: M^ε({i#e}) = {M + i#e}
    ///
    /// # Theory (Definition 7, page 14; Definition 18, page 46)
    ///
    /// Same elementary transition as I-type, but converts results using M^ε instead of I^ε.
    /// M^ε does NOT subtract 1 from the concrete position.
    fn successors_m_type_standard(
        offset: i32,
        errors: u8,
        bit_vector: &crate::transducer::universal::CharacteristicVector,
        max_distance: u8,
    ) -> Vec<Self> {
        let mut successors = Vec::new();

        // Universal position M + t#e uses offset t directly for δ^D,ε_e
        // After δ^D,ε_e, we convert back using M^ε({i#e}) = {M + i#e}

        // Case 1: 1 < b (match at position 1 - bit vector starts with 1)
        if bit_vector.starts_with_one() {
            // δ^D,ε_e(t#e, "1...") = {(t+1)#e}
            // M^ε({(t+1)#e}) = {M + (t+1)#e}
            // Offset increases by 1!
            if let Ok(succ) = Self::new_m(offset + 1, errors, max_distance) {
                successors.push(succ);
            }
            return successors;
        }

        // Check if we can still consume errors
        if errors >= max_distance {
            return successors; // No more errors available
        }

        // Case 2: b = ε (empty bit vector)
        if bit_vector.is_empty() {
            // δ^D,ε_e(t#e, ε) = {t#(e+1)}
            // M^ε({t#(e+1)}) = {M + t#(e+1)}
            if let Ok(succ) = Self::new_m(offset, errors + 1, max_distance) {
                successors.push(succ);
            }
            return successors;
        }

        // Case 3: b = 0^k (all zeros - no matches at all)
        if bit_vector.is_all_zeros() {
            // δ^D,ε_e(t#e, "0...") = {t#(e+1), (t+1)#(e+1)}

            // Delete: t#(e+1) → M^ε = M + t#(e+1)
            if let Ok(succ) = Self::new_m(offset, errors + 1, max_distance) {
                successors.push(succ);
            }

            // Insert: (t+1)#(e+1) → M^ε = M + (t+1)#(e+1)
            if let Ok(succ) = Self::new_m(offset + 1, errors + 1, max_distance) {
                successors.push(succ);
            }

            return successors;
        }

        // Case 4: 0 < b (starts with 0, but has 1 somewhere)

        // Delete: t#(e+1) → M^ε = M + t#(e+1)
        if let Ok(succ) = Self::new_m(offset, errors + 1, max_distance) {
            successors.push(succ);
        }

        // Insert: (t+1)#(e+1) → M^ε = M + (t+1)#(e+1)
        if let Ok(succ) = Self::new_m(offset + 1, errors + 1, max_distance) {
            successors.push(succ);
        }

        // Skip to match at position j (0-indexed in bit vector, so j+1 in thesis notation)
        // δ^D,ε_e(t#e, "00...1...") performs j deletions to reach j+1, then matches
        // Concrete result: (t+j+1)#(e+j)
        // M^ε({(t+j+1)#(e+j)}) = {M + (t+j+1)#(e+j)}
        if let Some(j) = bit_vector.first_match() {
            let j = j as i32;
            let new_offset = offset + j + 1;
            let new_errors = errors + j as u8;

            if new_errors <= max_distance {
                if let Ok(succ) = Self::new_m(new_offset, new_errors, max_distance) {
                    successors.push(succ);
                }
            }
        }

        successors
    }
}

impl<V: PositionVariant> fmt::Display for UniversalPosition<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::INonFinal { offset, errors, .. } => {
                write!(f, "I + {}#{}", offset, errors)
            }
            Self::MFinal { offset, errors, .. } => {
                write!(f, "M + {}#{}", offset, errors)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // I-type Position Tests
    // =========================================================================

    #[test]
    fn test_i_position_initial_state() {
        // Initial state: I + 0#0
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert_eq!(pos.offset(), 0);
        assert_eq!(pos.errors(), 0);
        assert!(pos.is_i_type());
        assert!(!pos.is_m_type());
        assert_eq!(format!("{}", pos), "I + 0#0");
    }

    #[test]
    fn test_i_position_positive_offset() {
        // Valid: |2| = 2 ≤ 2
        let pos = UniversalPosition::<Standard>::new_i(2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert_eq!(pos.offset(), 2);
        assert_eq!(pos.errors(), 2);
    }

    #[test]
    fn test_i_position_negative_offset() {
        // Valid: |-2| = 2 ≤ 2
        let pos = UniversalPosition::<Standard>::new_i(-2, 2, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert_eq!(pos.offset(), -2);
        assert_eq!(pos.errors(), 2);
    }

    #[test]
    fn test_i_position_boundary_max_n() {
        // Valid: |3| = 3 ≤ 3, offset at boundary
        let pos = UniversalPosition::<Standard>::new_i(3, 3, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert_eq!(pos.offset(), 3);
        assert_eq!(pos.errors(), 3);
    }

    #[test]
    fn test_i_position_boundary_min_n() {
        // Valid: |-3| = 3 ≤ 3, negative offset at boundary
        let pos = UniversalPosition::<Standard>::new_i(-3, 3, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        assert_eq!(pos.offset(), -3);
        assert_eq!(pos.errors(), 3);
    }

    #[test]
    fn test_i_position_violates_offset_abs_constraint() {
        // Invalid: |3| = 3 > 2
        let result = UniversalPosition::<Standard>::new_i(3, 2, 3);
        assert!(matches!(
            result,
            Err(PositionError::InvalidIPosition { .. })
        ));
    }

    #[test]
    fn test_i_position_violates_offset_too_large() {
        // Invalid: offset = 4 > n = 3
        let result = UniversalPosition::<Standard>::new_i(4, 3, 3);
        assert!(matches!(
            result,
            Err(PositionError::InvalidIPosition { .. })
        ));
    }

    #[test]
    fn test_i_position_violates_offset_too_negative() {
        // Invalid: offset = -4 < -n = -3
        let result = UniversalPosition::<Standard>::new_i(-4, 3, 3);
        assert!(matches!(
            result,
            Err(PositionError::InvalidIPosition { .. })
        ));
    }

    #[test]
    fn test_i_position_violates_errors_too_large() {
        // Invalid: errors = 4 > n = 3
        let result = UniversalPosition::<Standard>::new_i(0, 4, 3);
        assert!(matches!(
            result,
            Err(PositionError::InvalidIPosition { .. })
        ));
    }

    // =========================================================================
    // M-type Position Tests
    // =========================================================================

    #[test]
    fn test_m_position_final_exact() {
        // Final position: M + 0#0 (end of word, no errors)
        let pos = UniversalPosition::<Standard>::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert_eq!(pos.offset(), 0);
        assert_eq!(pos.errors(), 0);
        assert!(!pos.is_i_type());
        assert!(pos.is_m_type());
        assert_eq!(format!("{}", pos), "M + 0#0");
    }

    #[test]
    fn test_m_position_one_before_end() {
        // M + (-1)#1: one char before end, one error
        // Check: k ≥ -t - n ⇒ 1 ≥ -(-1) - 2 = 1 - 2 = -1 ✓
        let pos = UniversalPosition::<Standard>::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert_eq!(pos.offset(), -1);
        assert_eq!(pos.errors(), 1);
    }

    #[test]
    fn test_m_position_boundary_min_offset() {
        // M + (-4)#0: offset at -2n boundary with n=2
        // Check: k ≥ -t - n ⇒ 0 ≥ -(-4) - 2 = 4 - 2 = 2 ✗
        // Actually need k ≥ 2, so k=2 is minimum
        let pos = UniversalPosition::<Standard>::new_m(-4, 2, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        assert_eq!(pos.offset(), -4);
        assert_eq!(pos.errors(), 2);
    }

    #[test]
    fn test_m_position_violates_offset_positive() {
        // Invalid: offset = 1 > 0 (must be ≤ 0)
        let result = UniversalPosition::<Standard>::new_m(1, 0, 2);
        assert!(matches!(
            result,
            Err(PositionError::InvalidMPosition { .. })
        ));
    }

    #[test]
    fn test_m_position_violates_offset_too_negative() {
        // Invalid: offset = -5 < -2n = -4 (n=2)
        let result = UniversalPosition::<Standard>::new_m(-5, 2, 2);
        assert!(matches!(
            result,
            Err(PositionError::InvalidMPosition { .. })
        ));
    }

    #[test]
    fn test_m_position_violates_errors_constraint() {
        // M + (-3)#0: Check k ≥ -t - n ⇒ 0 ≥ -(-3) - 2 = 3 - 2 = 1 ✗
        let result = UniversalPosition::<Standard>::new_m(-3, 0, 2);
        assert!(matches!(
            result,
            Err(PositionError::InvalidMPosition { .. })
        ));
    }

    #[test]
    fn test_m_position_violates_errors_too_large() {
        // Invalid: errors = 3 > n = 2
        let result = UniversalPosition::<Standard>::new_m(-1, 3, 2);
        assert!(matches!(
            result,
            Err(PositionError::InvalidMPosition { .. })
        ));
    }

    // =========================================================================
    // Variant Tests
    // =========================================================================

    #[test]
    fn test_position_variants_different_types() {
        // Create positions for each variant
        let std_pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let trans_pos = UniversalPosition::<Transposition>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let ms_pos = UniversalPosition::<MergeAndSplit>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        // All have same offset/errors but different phantom types
        assert_eq!(std_pos.offset(), trans_pos.offset());
        assert_eq!(std_pos.errors(), trans_pos.errors());
        assert_eq!(std_pos.offset(), ms_pos.offset());
    }

    #[test]
    fn test_variant_names() {
        assert_eq!(Standard::variant_name(), "Standard");
        assert_eq!(Transposition::variant_name(), "Transposition");
        assert_eq!(MergeAndSplit::variant_name(), "MergeAndSplit");
    }

    // =========================================================================
    // Equality and Hashing Tests
    // =========================================================================

    #[test]
    fn test_position_equality() {
        let pos1 = UniversalPosition::<Standard>::new_i(1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos3 = UniversalPosition::<Standard>::new_i(1, 2, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        assert_eq!(pos1, pos2);
        assert_ne!(pos1, pos3);
    }

    #[test]
    fn test_position_clone() {
        let pos1 = UniversalPosition::<Standard>::new_i(1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = pos1.clone();

        assert_eq!(pos1, pos2);
        assert_eq!(pos1.offset(), pos2.offset());
        assert_eq!(pos1.errors(), pos2.errors());
    }

    // =========================================================================
    // Display Tests
    // =========================================================================

    #[test]
    fn test_display_i_positions() {
        let pos1 = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos2 = UniversalPosition::<Standard>::new_i(2, 2, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let pos3 = UniversalPosition::<Standard>::new_i(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");

        assert_eq!(format!("{}", pos1), "I + 0#0");
        assert_eq!(format!("{}", pos2), "I + 2#2");
        assert_eq!(format!("{}", pos3), "I + -1#1");
    }

    #[test]
    fn test_display_m_positions() {
        let pos1 = UniversalPosition::<Standard>::new_m(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos2 = UniversalPosition::<Standard>::new_m(-2, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let pos3 = UniversalPosition::<Standard>::new_m(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");

        assert_eq!(format!("{}", pos1), "M + 0#0");
        assert_eq!(format!("{}", pos2), "M + -2#0");
        assert_eq!(format!("{}", pos3), "M + -1#1");
    }

    // =========================================================================
    // Error Display Tests
    // =========================================================================

    #[test]
    fn test_position_error_display() {
        let err = PositionError::InvalidIPosition {
            offset: 3,
            errors: 1,
            max_distance: 2,
        };
        let display = format!("{}", err);
        assert!(display.contains("Invalid I-position"));
        assert!(display.contains("3#1"));
        assert!(display.contains("n=2"));

        let err = PositionError::InvalidMPosition {
            offset: -5,
            errors: 0,
            max_distance: 2,
        };
        let display = format!("{}", err);
        assert!(display.contains("Invalid M-position"));
        assert!(display.contains("-5#0"));
        assert!(display.contains("n=2"));
    }

    // =========================================================================
    // Successor Function Tests
    // =========================================================================

    use crate::transducer::universal::CharacteristicVector;

    #[test]
    fn test_successors_match() {
        // Match case: 1 < b (bit vector starts with 1 at the correct window position)
        // Testing position I+0#0 at input position 1, word "abc", max_distance 2
        // Windowed subword s_2(abc, 1) = "$$abc" (positions -1,0,1,2,3 → padding + abc)
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('a', "$$abc"); // [false, false, true, false, false]

        let succs = pos.successors(&bv, 2);

        // Match at index 2 (offset 0 + max_distance 2): I+0#0 → I+0#0
        assert_eq!(succs.len(), 1, "Expected 1 successor for match");
        assert_eq!(
            succs[0],
            UniversalPosition::<Standard>::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        );
    }

    #[test]
    fn test_successors_all_zeros() {
        // All zeros case: no matches anywhere
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('x', "abc"); // "000"

        let succs = pos.successors(&bv, 2);

        // Should have two successors: delete and insert
        // Delete: t#(e+1) → I + (t-1)#(e+1) = I + (-1)#1
        // Insert: (t+1)#(e+1) → I + t#(e+1) = I + 0#1
        assert_eq!(succs.len(), 2);

        // Delete: I + (-1)#1
        assert!(succs.contains(
            &UniversalPosition::<Standard>::new_i(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        ));

        // Insert: I + 0#1
        assert!(succs.contains(
            &UniversalPosition::<Standard>::new_i(0, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        ));
    }

    #[test]
    fn test_successors_match_later() {
        // Match later: 0 < b (starts with 0, has 1 later in the window)
        // Testing position I+0#0 at input position 1, word "abc", max_distance 2
        // Windowed subword s_2(abc, 1) = "$$abc"
        // Bit vector for 'b' in "$$abc" = [false, false, false, true, false]
        //                                   $     $     a     b     c
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('b', "$$abc");

        let succs = pos.successors(&bv, 2);

        // Should have three successors: delete, substitute, skip-to-match
        // Delete/substitute operations, plus skip to the 'b' at index 3
        assert_eq!(succs.len(), 3, "Expected 3 successors");

        // Delete: I + (-1)#1
        assert!(succs.contains(
            &UniversalPosition::<Standard>::new_i(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        ));

        // Substitute: I + 0#1
        assert!(succs.contains(
            &UniversalPosition::<Standard>::new_i(0, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        ));

        // Skip to match at index 3: I + 1#1
        assert!(succs.contains(
            &UniversalPosition::<Standard>::new_i(1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        ));
    }

    #[test]
    fn test_successors_empty_bit_vector() {
        // Empty bit vector
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('a', ""); // ""

        let succs = pos.successors(&bv, 2);

        // Should have one successor: delete only
        // Delete: t#(e+1) → I + (t-1)#(e+1) = I + (-1)#1
        assert_eq!(succs.len(), 1);
        assert_eq!(
            succs[0],
            UniversalPosition::<Standard>::new_i(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        );
    }

    #[test]
    fn test_successors_max_errors_reached() {
        // Already at max errors
        let pos = UniversalPosition::<Standard>::new_i(0, 2, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('x', "abc"); // "000"

        let succs = pos.successors(&bv, 2);

        // Should have no successors (can't consume more errors)
        assert_eq!(succs.len(), 0);
    }

    #[test]
    fn test_successors_match_at_max_errors() {
        // At max errors but there's a match
        // Windowed subword s_2(abc, 1) = "$$abc"
        let pos = UniversalPosition::<Standard>::new_i(0, 2, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('a', "$$abc"); // [false, false, true, false, false]

        let succs = pos.successors(&bv, 2);

        // Should still have the match successor at index 2
        assert_eq!(
            succs.len(),
            1,
            "Expected 1 successor for match at max errors"
        );
        assert_eq!(
            succs[0],
            UniversalPosition::<Standard>::new_i(0, 2, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        );
    }

    #[test]
    fn test_successors_negative_offset() {
        // Test with negative offset
        // Position I+(-1)#1 at input position 1 checks word position 0 (1 + (-1) = 0)
        // In windowed subword "$$abc", word position 0 is at index 2-1=1 (the second '$')
        // But we want 'a', so we need word position 1, which is at index 2
        // Wait, let me recalculate: offset=-1 means word_pos = k + offset = 1 + (-1) = 0
        // In window starting at k-n = 1-2 = -1, position 0 is at index 0 - (-1) = 1
        // Hmm, for offset=-1: match_index = max_distance + offset = 2 + (-1) = 1
        // Window "$$abc": index 1 is the second '$', not 'a'
        // So this test expects NO match, not a match! Let me check the original test...
        // Oh wait, it expects a match. So the original test was wrong. Let me fix it properly.
        // For this to match 'a', we need the window where index (n + offset) = (2 + (-1)) = 1 contains 'a'
        // That would be "$aabc" or similar
        let pos = UniversalPosition::<Standard>::new_i(-1, 1, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('$', "$$abc"); // Match the padding character at index 1

        let succs = pos.successors(&bv, 2);

        // Match at index (2 + (-1)) = 1: offset stays same
        assert_eq!(
            succs.len(),
            1,
            "Expected 1 successor for match with negative offset"
        );
        assert_eq!(
            succs[0],
            UniversalPosition::<Standard>::new_i(-1, 1, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        );
    }

    #[test]
    fn test_successors_skip_multiple() {
        // Skip to match at word position 2 (0-indexed)
        // Windowed subword s_3(abcd, 1) = "$$$abcd" (positions -2,-1,0,1,2,3,4)
        // Bit vector for 'c' in "$$$abcd" = [false, false, false, false, false, true, false]
        //                                     $     $     $     a     b     c     d
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 3)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('c', "$$$abcd");

        let succs = pos.successors(&bv, 3);

        // Should have: delete, substitute, skip-to-match
        assert_eq!(succs.len(), 3, "Expected 3 successors");

        // Delete: I + (-1)#1
        assert!(succs.contains(
            &UniversalPosition::<Standard>::new_i(-1, 1, 3)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        ));

        // Substitute: I + 0#1
        assert!(succs.contains(
            &UniversalPosition::<Standard>::new_i(0, 1, 3)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        ));

        // Skip to match at index 5: offset=2, errors=2
        assert!(succs.contains(
            &UniversalPosition::<Standard>::new_i(2, 2, 3)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        ));
    }

    #[test]
    fn test_successors_invariant_violation_filtered() {
        // Test that invalid successors are filtered out
        let pos = UniversalPosition::<Standard>::new_i(2, 2, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('x', "abc"); // "000"

        let succs = pos.successors(&bv, 2);

        // Delete would be 2#3 (invalid: errors > max_distance)
        // Insert would be 3#3 (invalid: offset=3 > n=2 and errors > max_distance)
        // So we should get no valid successors
        assert_eq!(succs.len(), 0);
    }

    #[test]
    fn test_successors_m_type_position() {
        // Test M-type position successor generation
        // M-type positions use M^ε conversion: M^ε({i#e}) = {M + i#e}
        let pos = UniversalPosition::<Standard>::new_m(-1, 0, 2)
            .expect("test fixture: UniversalPosition::new_m with valid args");
        let bv = CharacteristicVector::new('a', "abc"); // "100"

        let succs = pos.successors(&bv, 2);

        // Match: δ^D,ε_e((-1)#0, "1...") = {0#0}
        // M^ε({0#0}) = {M + 0#0}
        assert_eq!(succs.len(), 1);
        assert_eq!(
            succs[0],
            UniversalPosition::<Standard>::new_m(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_m with valid args")
        );
    }

    #[test]
    fn test_successors_boundary_offset() {
        // Test at boundary: offset = n = 2
        // Position I+2#2 at input position 1 checks word position 3 (1 + 2 = 3)
        // match_index = n + offset = 2 + 2 = 4
        // Windowed subword s_2(abc, 1) = "$$abc" (length 5)
        // Need character at index 4, which is 'c'
        let pos = UniversalPosition::<Standard>::new_i(2, 2, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('c', "$$abc"); // [false, false, false, false, true]

        let succs = pos.successors(&bv, 2);

        // Match at index 4: offset stays same
        assert_eq!(
            succs.len(),
            1,
            "Expected 1 successor for boundary offset match"
        );
        assert_eq!(
            succs[0],
            UniversalPosition::<Standard>::new_i(2, 2, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        );
    }

    #[test]
    fn test_successors_multiple_matches() {
        // Multiple matches in bit vector (should only return first match)
        // Windowed subword s_2(aba, 1) = "$$aba"
        // Bit vector for 'a' in "$$aba" = [false, false, true, false, true]
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('a', "$$aba");

        let succs = pos.successors(&bv, 2);

        // Match at index 2 (first 'a'): I + 0#0 → I + 0#0
        assert_eq!(succs.len(), 1, "Expected 1 successor for first match");
        assert_eq!(
            succs[0],
            UniversalPosition::<Standard>::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        );
    }

    #[test]
    fn test_successors_preserves_variant_type() {
        // Ensure Standard variant is preserved
        // Windowed subword s_2(a, 1) = "$$a"
        let pos = UniversalPosition::<Standard>::new_i(0, 0, 2)
            .expect("test fixture: UniversalPosition::new_i with valid args");
        let bv = CharacteristicVector::new('a', "$$a"); // [false, false, true]

        let succs = pos.successors(&bv, 2);

        assert_eq!(succs.len(), 1, "Expected 1 successor");
        // Check that result is also Standard variant: I + 0#0 → I + 0#0
        assert_eq!(
            succs[0],
            UniversalPosition::<Standard>::new_i(0, 0, 2)
                .expect("test fixture: UniversalPosition::new_i with valid args")
        );
    }
}
