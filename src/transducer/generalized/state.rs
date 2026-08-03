//! Generalized State Type
//!
//! Implements universal states adapted for runtime-configurable operations.
//! Based on Definition 15 from Mitankin's thesis (pages 38-39), but without
//! compile-time variant specialization.
//!
//! # Operation Support
//!
//! Every state carries the exact [`CostScale`] derived from its operation set.
//! Position costs are `usize` values in that scale; changing to another exact
//! operation scale rescales the complete state through a checked least common
//! denominator. Multi-scalar operation costs are charged when their final
//! target scalar arrives, so restrictions are known before cost is committed.
//!
//! # Theory Background
//!
//! Universal states are sets of universal positions that maintain the anti-chain property:
//! no position subsumes another. This enables efficient state minimization.
//!
//! ## Anti-chain Property
//!
//! For all positions p₁, p₂ in state Q:
//! - p₁ ⊀^χ_s p₂ (p₁ does not subsume p₂)
//! - p₂ ⊀^χ_s p₁ (p₂ does not subsume p₁)
//!
//! This is maintained by the ⊔ (join) operator when adding positions.

use smallvec::SmallVec;
use std::fmt;

use super::position::GeneralizedPosition;
use super::subsumption::subsumes_scaled;
use crate::cost::{CostScale, ScaleError};
use crate::transducer::universal::bit_vector::CharacteristicVector;

/// Failure while evaluating the bounded universal-state compatibility API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeneralizedStateError {
    /// An operation weight cannot be represented by an exact common scale.
    Scale(ScaleError),
    /// The universal-state encoding has no intermediate position for this
    /// source/target consumption pair.
    UnsupportedOperationArity {
        /// Human-readable operation identifier.
        name: Box<str>,
        /// Number of source scalars consumed.
        consume_x: usize,
        /// Number of target scalars consumed.
        consume_y: usize,
    },
}

impl From<ScaleError> for GeneralizedStateError {
    fn from(error: ScaleError) -> Self {
        Self::Scale(error)
    }
}

impl fmt::Display for GeneralizedStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scale(error) => write!(formatter, "invalid operation cost scale: {error}"),
            Self::UnsupportedOperationArity {
                name,
                consume_x,
                consume_y,
            } => write!(
                formatter,
                "operation {name:?} consumes ({consume_x}, {consume_y}), which the bounded universal-state encoding cannot represent"
            ),
        }
    }
}

impl std::error::Error for GeneralizedStateError {}

#[inline]
fn state_supports_operation(operation: &crate::transducer::OperationType) -> bool {
    matches!(
        (operation.consume_x(), operation.consume_y()),
        (1, 1) | (1, 0) | (0, 1) | (2, 2) | (2, 1) | (1, 2)
    )
}

fn is_classical_levenshtein_lattice(operations: &crate::transducer::OperationSet) -> bool {
    if operations.len() != 4 || operations.iter().any(|operation| operation.is_restricted()) {
        return false;
    }

    let count = |consume_x, consume_y, weight: f64| {
        operations
            .iter()
            .filter(|operation| {
                operation.consume_x() == consume_x
                    && operation.consume_y() == consume_y
                    && operation.weight().to_bits() == weight.to_bits()
            })
            .count()
    };

    count(1, 1, 0.0) == 1 && count(1, 1, 1.0) == 1 && count(0, 1, 1.0) == 1 && count(1, 0, 1.0) == 1
}

#[inline]
fn bounded_index(index: i32, len: usize) -> Option<usize> {
    let index = usize::try_from(index).ok()?;
    (index < len).then_some(index)
}

#[inline]
fn bounded_pair_start(index: i32, len: usize) -> Option<usize> {
    let index = bounded_index(index, len)?;
    (index + 1 < len).then_some(index)
}

#[inline]
fn word_position_index(input_position: usize, offset: i32) -> Option<usize> {
    let input_position = i64::try_from(input_position).ok()?;
    let position = input_position
        .checked_add(i64::from(offset))?
        .checked_sub(2)?;
    usize::try_from(position).ok()
}

#[inline]
fn add_weight_to_errors(
    errors: usize,
    weight: f64,
    scale: CostScale,
    max_cost: usize,
) -> Option<usize> {
    let weight_errors = scale.to_scaled(weight).ok()?;
    errors
        .checked_add(weight_errors)
        .filter(|&new_errors| new_errors <= max_cost)
}

#[inline]
fn add_distance_to_errors(
    errors: usize,
    distance: usize,
    scale: CostScale,
    max_cost: usize,
) -> Option<usize> {
    let distance_cost = distance.checked_mul(scale.denominator() as usize)?;
    errors
        .checked_add(distance_cost)
        .filter(|&new_errors| new_errors <= max_cost)
}

/// Input context for one generalized automaton transition.
#[derive(Clone, Copy)]
pub struct GeneralizedTransitionInput<'a> {
    /// Set of operations defining the edit distance metric.
    pub operations: &'a crate::transducer::OperationSet,
    /// Characteristic vector β(a, w) encoding matches for the current input.
    pub bit_vector: &'a CharacteristicVector,
    /// Full dictionary word currently being traversed.
    pub full_word: &'a str,
    /// Optional precomputed full-word characters for hot phonetic operations.
    pub word_chars: Option<&'a [char]>,
    /// Current word suffix/window.
    pub word_slice: &'a str,
    /// Current input character.
    pub input_char: char,
    /// One-indexed input position used by split-position calculations.
    pub input_position: usize,
}

impl<'a> GeneralizedTransitionInput<'a> {
    /// Create transition input context.
    pub const fn new(
        operations: &'a crate::transducer::OperationSet,
        bit_vector: &'a CharacteristicVector,
        full_word: &'a str,
        word_chars: Option<&'a [char]>,
        word_slice: &'a str,
        input_char: char,
        input_position: usize,
    ) -> Self {
        Self {
            operations,
            bit_vector,
            full_word,
            word_chars,
            word_slice,
            input_char,
            input_position,
        }
    }
}

/// Generalized state maintaining anti-chain property
///
///  state is a set of generalized positions where no position subsumes another.
///
/// # Invariant
///
/// For all p₁, p₂ ∈ positions: p₁ ⊀^χ_s p₂ ∧ p₂ ⊀^χ_s p₁
///
/// This invariant is maintained by `add_position()` using the ⊔ operator.
///
/// # SmallVec Optimization
///
/// Uses SmallVec with inline size of 8 to avoid heap allocations for typical states.
/// See universal/state.rs for theoretical justification via bounded diagonal property.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneralizedState {
    /// Set of positions (anti-chain), maintained in sorted order
    /// SmallVec avoids heap allocation for states with ≤8 positions
    positions: SmallVec<[GeneralizedPosition; 8]>,

    /// Maximum edit distance n in public unit-cost budget units.
    max_distance: u8,

    /// Exact fixed-point scale used by every position cost.
    cost_scale: CostScale,

    /// Maximum budget represented in [`Self::cost_scale`] units.
    max_cost: usize,

    /// Whether the classical unit-cost offset/slack subsumption theorem is
    /// justified by the complete operation set seen so far.
    use_classical_subsumption: bool,

    /// Previous input scalar for two-step transpose validation.
    previous_input_char: Option<char>,
}

impl GeneralizedState {
    /// Create new empty state
    ///
    /// # Arguments
    ///
    /// - `max_distance`: Maximum edit distance n
    pub fn new(max_distance: u8) -> Self {
        Self::with_scale(
            max_distance,
            CostScale::new(1).expect("one is a valid denominator"),
            true,
        )
    }

    fn with_scale(
        max_distance: u8,
        cost_scale: CostScale,
        use_classical_subsumption: bool,
    ) -> Self {
        let max_cost = cost_scale
            .scale_budget(max_distance)
            .expect("u8 budget times u32 denominator fits usize on supported targets");
        Self {
            positions: SmallVec::new(),
            max_distance,
            cost_scale,
            max_cost,
            use_classical_subsumption,
            previous_input_char: None,
        }
    }

    /// Create initial state {I + 0#0}
    ///
    /// From thesis page 38: Initial state I^∀,χ = {I + 0#0}
    pub fn initial(max_distance: u8) -> Self {
        let mut state = Self::new(max_distance);
        // I + 0#0 always satisfies invariant, so unwrap is safe
        let initial_pos = GeneralizedPosition::new_i_scaled(
            0,
            0,
            state.max_cost,
            max_distance,
            state.cost_scale.denominator(),
        )
        .expect("I + 0#0 should always be valid");
        state.positions.push(initial_pos);
        state
    }

    /// Return the exact scale carried by this state.
    pub const fn cost_scale(&self) -> CostScale {
        self.cost_scale
    }

    /// Return the maximum budget in scaled integer units.
    pub const fn max_scaled_cost(&self) -> usize {
        self.max_cost
    }

    #[inline]
    fn new_i_position(&self, offset: i32, cost: usize) -> Option<GeneralizedPosition> {
        GeneralizedPosition::new_i_scaled(
            offset,
            cost,
            self.max_cost,
            self.max_distance,
            self.cost_scale.denominator(),
        )
        .ok()
    }

    #[inline]
    fn new_m_position(&self, offset: i32, cost: usize) -> Option<GeneralizedPosition> {
        GeneralizedPosition::new_m_scaled(
            offset,
            cost,
            self.max_cost,
            self.max_distance,
            self.cost_scale.denominator(),
        )
        .ok()
    }

    #[inline]
    fn new_i_transposing_position(&self, offset: i32, cost: usize) -> Option<GeneralizedPosition> {
        GeneralizedPosition::new_i_transposing_scaled(
            offset,
            cost,
            self.max_cost,
            self.max_distance,
            self.cost_scale.denominator(),
        )
        .ok()
    }

    #[inline]
    fn new_m_transposing_position(&self, offset: i32, cost: usize) -> Option<GeneralizedPosition> {
        GeneralizedPosition::new_m_transposing_scaled(
            offset,
            cost,
            self.max_cost,
            self.max_distance,
            self.cost_scale.denominator(),
        )
        .ok()
    }

    #[inline]
    fn new_i_splitting_position(
        &self,
        offset: i32,
        cost: usize,
        entry_char: char,
    ) -> Option<GeneralizedPosition> {
        GeneralizedPosition::new_i_splitting_scaled(
            offset,
            cost,
            self.max_cost,
            self.max_distance,
            self.cost_scale.denominator(),
            entry_char,
        )
        .ok()
    }

    #[inline]
    fn new_m_splitting_position(
        &self,
        offset: i32,
        cost: usize,
        entry_char: char,
    ) -> Option<GeneralizedPosition> {
        GeneralizedPosition::new_m_splitting_scaled(
            offset,
            cost,
            self.max_cost,
            self.max_distance,
            self.cost_scale.denominator(),
            entry_char,
        )
        .ok()
    }

    /// Add position, maintaining anti-chain property (⊔ operator)
    ///
    /// Implements the subsumption closure from the thesis:
    /// 1. Remove all positions subsumed by the new position (worse positions)
    /// 2. Add new position if it's not subsumed by any existing position
    ///
    /// This maintains the invariant ∀p₁,p₂ ∈ Q (p₁ ⊀^χ_s p₂).
    pub fn add_position(&mut self, pos: GeneralizedPosition) {
        // Check if this position is subsumed by an existing one
        for existing in &self.positions {
            if existing == &pos {
                return;
            }
            if subsumes_scaled(
                existing,
                &pos,
                self.max_distance,
                self.use_classical_subsumption,
            ) {
                return; // Already covered by existing position
            }
        }

        // Remove any positions that this new position subsumes
        self.positions.retain(|p| {
            !subsumes_scaled(&pos, p, self.max_distance, self.use_classical_subsumption)
        });

        // Insert in sorted position (binary search)
        match self.positions.binary_search(&pos) {
            Ok(_) => {}
            Err(insert_pos) => self.positions.insert(insert_pos, pos),
        }
    }

    /// Check if state is empty
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Get number of positions in state
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Get iterator over positions
    pub fn positions(&self) -> impl Iterator<Item = &GeneralizedPosition> {
        self.positions.iter()
    }

    /// Check if state is final
    ///
    /// A state is final if it contains an M-type position with offset ≤ 0.
    pub fn is_final(&self) -> bool {
        self.positions.iter().any(|pos| match pos {
            GeneralizedPosition::MFinal { offset, .. } => *offset <= 0,
            _ => false,
        })
    }

    /// Compute transition to successor state (δ^∀,χ_n)
    ///
    /// Supports runtime-configurable operations via `OperationSet`.
    ///
    /// # Returns
    ///
    /// Successor state, or `None` if no successors exist (undefined transition)
    pub fn transition(&self, input: GeneralizedTransitionInput<'_>) -> Option<Self> {
        self.try_transition(input).ok().flatten()
    }

    /// Compute a successor while reporting invalid scales or operation arities
    /// that this bounded universal-state representation cannot encode.
    ///
    /// [`crate::transducer::generalized::GeneralizedAutomaton`] is the operation-complete
    /// API for arbitrary non-zero consumption pairs.
    pub fn try_transition(
        &self,
        input: GeneralizedTransitionInput<'_>,
    ) -> Result<Option<Self>, GeneralizedStateError> {
        if let Some(operation) = input
            .operations
            .iter()
            .find(|operation| !state_supports_operation(operation))
        {
            return Err(GeneralizedStateError::UnsupportedOperationArity {
                name: operation.name().into(),
                consume_x: operation.consume_x(),
                consume_y: operation.consume_y(),
            });
        }
        let operation_scale = CostScale::for_operations(input.operations)?;
        let common_scale = self.cost_scale.common(operation_scale)?;
        let mut normalized = self.rescaled(common_scale)?;
        normalized.use_classical_subsumption &= is_classical_levenshtein_lattice(input.operations);
        Ok(normalized.transition_scaled(input))
    }

    fn rescaled(&self, target: CostScale) -> Result<Self, ScaleError> {
        if target == self.cost_scale {
            return Ok(self.clone());
        }
        let multiplier = target.denominator() as usize / self.cost_scale.denominator() as usize;
        let positions = self
            .positions
            .iter()
            .map(|position| {
                position
                    .checked_rescale(multiplier)
                    .ok_or(ScaleError::CostOverflow)
            })
            .collect::<Result<SmallVec<[GeneralizedPosition; 8]>, _>>()?;
        Ok(Self {
            positions,
            max_distance: self.max_distance,
            cost_scale: target,
            max_cost: target.scale_budget(self.max_distance)?,
            use_classical_subsumption: self.use_classical_subsumption,
            previous_input_char: self.previous_input_char,
        })
    }

    fn transition_scaled(&self, input: GeneralizedTransitionInput<'_>) -> Option<Self> {
        // Special case: empty state has no successors
        if self.is_empty() {
            return None;
        }

        // Create new state for successors
        let mut next_state = Self::with_scale(
            self.max_distance,
            self.cost_scale,
            self.use_classical_subsumption,
        );

        // For each position in current state
        for pos in &self.positions {
            // Compute successors using runtime-configurable operations
            // Phase 3b/4: Pass full_word, word_slice, and input_position for phonetic operations
            // H2 Optimization: Pass word_chars to eliminate repeated char().collect() calls
            let successors = self.successors(pos, &input);

            // Add all successors to next state
            for succ in successors {
                next_state.add_position(succ);
            }
        }

        // Return None if no successors (undefined transition)
        if next_state.is_empty() {
            None
        } else {
            // Phase 3b: Store current char for next iteration (needed for split/transpose)
            next_state.previous_input_char = Some(input.input_char);
            Some(next_state)
        }
    }

    /// Compute successors for a position using runtime-configurable operations
    ///
    /// The input context supplies the full word, active word slice, current
    /// scalar, and input position needed to validate restricted intermediates.
    fn successors(
        &self,
        pos: &GeneralizedPosition,
        input: &GeneralizedTransitionInput<'_>,
    ) -> Vec<GeneralizedPosition> {
        match pos {
            GeneralizedPosition::INonFinal { offset, errors } => {
                self.successors_i_type(*offset, *errors, input)
            }
            GeneralizedPosition::MFinal { offset, errors } => {
                self.successors_m_type(*offset, *errors, input)
            }
            // Phase 2d: Multi-character operation intermediate states
            GeneralizedPosition::ITransposing { offset, errors } => {
                // Complete transposition for I-type positions
                // Phase 3b: Pass full_word, word_slice, input_char for phonetic validation
                self.successors_i_transposing(*offset, *errors, input)
            }
            GeneralizedPosition::MTransposing { offset, errors } => {
                // Complete transposition for M-type positions
                // Phase 3b: Pass full_word, word_slice, input_char for phonetic validation
                self.successors_m_transposing(*offset, *errors, input)
            }
            // Phase 2d.5: Splitting positions
            GeneralizedPosition::ISplitting {
                offset,
                errors,
                entry_char,
            } => {
                // Complete split for I-type positions
                // Phase 3b/4: Pass full_word, word_slice, input_char, input_position for phonetic validation and word_pos calc
                self.successors_i_splitting(*offset, *errors, *entry_char, input)
            }
            GeneralizedPosition::MSplitting {
                offset,
                errors,
                entry_char,
            } => {
                // Complete split for M-type positions
                // Phase 3b/4: Pass full_word, word_slice, input_char, input_position for phonetic validation and word_pos calc
                self.successors_m_splitting(*offset, *errors, *entry_char, input)
            }
        }
    }

    /// Compute successors for I-type positions with runtime-configurable operations
    ///
    /// Based on Universal automaton's δ^D,ε_e with I^ε conversion.
    ///
    /// # I^ε Conversion
    ///
    /// Universal positions use δ^D,ε_e which operates on raw offsets,
    /// then converts via I^ε({i#e}) = {I + (i-1)#e}.
    ///
    /// This means:
    /// - MATCH: t+1#e → I^ε → I+t#e (offset stays same)
    /// - DELETE: t#e+1 → I^ε → I+(t-1)#(e+1) (offset decreases)
    /// - INSERT: (t+1)#(e+1) → I^ε → I+t#(e+1) (offset stays same)
    /// - SUBSTITUTE: (t+1)#(e+1) → I^ε → I+t#(e+1) (offset stays same)
    fn successors_i_type(
        &self,
        offset: i32,
        errors: usize,
        input: &GeneralizedTransitionInput<'_>,
    ) -> Vec<GeneralizedPosition> {
        let operations = input.operations;
        let bit_vector = input.bit_vector;
        let word_slice = input.word_slice;
        let input_char = input.input_char;
        let mut successors = Vec::new();
        let n = self.max_distance as i32;

        // H2 Optimization: Collect word_slice characters once instead of repeatedly
        let word_slice_chars: Vec<char> = word_slice.chars().collect();

        // H1 Optimization: Pre-encode input_char once to avoid repeated String allocations
        let mut input_char_buf = [0u8; 4];
        let input_char_bytes = input_char.encode_utf8(&mut input_char_buf).as_bytes();

        // Bit vector index for current position: offset + n
        let match_index_i32 = offset + n;

        // Case 1: Position within visible window
        if let Some(match_index) = bounded_index(match_index_i32, bit_vector.len()) {
            let has_match = bit_vector.is_match(match_index);

            // Iterate over all operations (handle standard single-character operations)
            for op in operations.operations() {
                // Skip multi-char operations in this loop (handled separately below)
                if op.consume_x() > 1 || op.consume_y() > 1 {
                    continue;
                }

                // Classify operation type and generate successors
                if op.is_match() {
                    // Match operation: ⟨1, 1, 0.0⟩
                    if has_match {
                        // Phase 3: For match, check can_apply() with actual characters
                        if match_index < word_slice_chars.len() {
                            // H1 Optimization: Use stack buffer instead of heap allocation
                            let mut word_char_buf = [0u8; 4];
                            let word_char_bytes = word_slice_chars[match_index]
                                .encode_utf8(&mut word_char_buf)
                                .as_bytes();
                            if op.can_apply(word_char_bytes, input_char_bytes) {
                                // δ^D,ε_e: (t+1)#e → I^ε → I+t#e
                                if let Some(succ) = self.new_i_position(offset, errors) {
                                    successors.push(succ);
                                    // Phase 3b: Don't return early - allow multi-character operations to compete
                                }
                            }
                        }
                    }
                } else if op.is_deletion() {
                    // Delete operation: ⟨1, 0, w⟩
                    // Phase 3: For deletion, check can_apply() with word character and empty input
                    if errors < self.max_cost {
                        if let Some(new_errors) = add_weight_to_errors(
                            errors,
                            op.weight(),
                            self.cost_scale,
                            self.max_cost,
                        ) {
                            // H2 Optimization: Using word_slice_chars from method beginning
                            if match_index < word_slice_chars.len() {
                                // H1 Optimization: Use stack buffer instead of heap allocation
                                let mut word_char_buf = [0u8; 4];
                                let word_char_bytes = word_slice_chars[match_index]
                                    .encode_utf8(&mut word_char_buf)
                                    .as_bytes();
                                if op.can_apply(word_char_bytes, &[]) {
                                    // δ^D,ε_e: t#(e+w) → I^ε → I+(t-1)#(e+w)
                                    if let Some(succ) = self.new_i_position(offset - 1, new_errors)
                                    {
                                        successors.push(succ);
                                    }
                                }
                            }
                        }
                    }
                } else if op.is_insertion() {
                    // Insert ⟨0, 1, w⟩
                    // Phase 3: For insertion, check can_apply() with empty word and input character
                    if errors < self.max_cost {
                        if let Some(new_errors) = add_weight_to_errors(
                            errors,
                            op.weight(),
                            self.cost_scale,
                            self.max_cost,
                        ) {
                            // H1 Optimization: Use pre-encoded input_char_bytes (no allocation)
                            if op.can_apply(&[], input_char_bytes) {
                                // δ^D,ε_e: (t+1)#(e+w) → I^ε → I+t#(e+w)
                                if let Some(succ) = self.new_i_position(offset, new_errors) {
                                    successors.push(succ);
                                }
                            }
                        }
                    }
                } else if op.is_substitution() {
                    // Substitute ⟨1, 1, w⟩
                    // Phase 3: For substitution, check can_apply() with word and input characters
                    if errors < self.max_cost {
                        if let Some(new_errors) = add_weight_to_errors(
                            errors,
                            op.weight(),
                            self.cost_scale,
                            self.max_cost,
                        ) {
                            // H2 Optimization: Using word_slice_chars from method beginning
                            if match_index < word_slice_chars.len() {
                                // H1 Optimization: Use stack buffer instead of heap allocation
                                let mut word_char_buf = [0u8; 4];
                                let word_char_bytes = word_slice_chars[match_index]
                                    .encode_utf8(&mut word_char_buf)
                                    .as_bytes();
                                if op.can_apply(word_char_bytes, input_char_bytes) {
                                    // δ^D,ε_e: (t+1)#(e+w) → I^ε → I+t#(e+w)
                                    if let Some(succ) = self.new_i_position(offset, new_errors) {
                                        successors.push(succ);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Phase 2d/3b: Multi-character operations - TRANSPOSITION ⟨2,2,1⟩
            // Phase 3b: Support phonetic ⟨2,2⟩ operations (e.g., "qu"↔"kw")
            let transpose_ops: Vec<_> = operations
                .operations()
                .iter()
                .filter(|op| op.consume_x() == 2 && op.consume_y() == 2)
                .collect();

            if transpose_ops.iter().any(|op| {
                add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost).is_some()
            }) {
                // H2 Optimization: Using word_slice_chars from method beginning
                let next_match_index_i32 = match_index_i32 + 1;

                // Check if we have enough word characters for transpose
                if let Some(next_match_index) =
                    bounded_index(next_match_index_i32, word_slice_chars.len())
                {
                    if word_slice_chars[next_match_index] != '$' {
                        // Both target characters are needed to decide a general
                        // two-for-two restriction, so entry is speculative and
                        // completion performs the exact applicability check.
                        if let Some(trans) = self.new_i_transposing_position(offset - 1, errors) {
                            successors.push(trans);
                        }
                    }
                }
            }

            // Phase 2d/3: Multi-character operations - MERGE ⟨2,1⟩
            // Merge: consume 2 word chars, match 1 input char (direct operation)
            // Phase 3: Supports phonetic operations like "ch"→"k", "ph"→"f"
            if errors < self.max_cost {
                // H2 Optimization: Using word_slice_chars from method beginning

                // Check if we have enough word characters (need 2 consecutive chars)
                // Skip padding chars '$'
                if let Some(match_index) =
                    bounded_pair_start(match_index_i32, word_slice_chars.len())
                {
                    if word_slice_chars[match_index] != '$'
                        && word_slice_chars[match_index + 1] != '$'
                    {
                        // H1 Optimization: Encode 2 word characters using stack buffers
                        let mut word_2chars_buf = [0u8; 8]; // Max 4 bytes per char, 2 chars = 8 bytes
                        let mut word_2chars_len = 0usize;
                        {
                            let char1_bytes = word_slice_chars[match_index]
                                .encode_utf8(&mut word_2chars_buf[0..4]);
                            word_2chars_len += char1_bytes.len();
                            let char2_bytes = word_slice_chars[match_index + 1].encode_utf8(
                                &mut word_2chars_buf[word_2chars_len..word_2chars_len + 4],
                            );
                            word_2chars_len += char2_bytes.len();
                        }
                        let word_2chars_bytes = &word_2chars_buf[..word_2chars_len];

                        // Check all ⟨2,1⟩ operations
                        for op in operations.operations() {
                            if op.consume_x() == 2 && op.consume_y() == 1 {
                                // Phase 3: Use can_apply() for phonetic operations
                                // Don't check bit_vector - phonetic ops don't require char matches
                                if op.can_apply(word_2chars_bytes, input_char_bytes) {
                                    if let Some(new_errors) = add_weight_to_errors(
                                        errors,
                                        op.weight(),
                                        self.cost_scale,
                                        self.max_cost,
                                    ) {
                                        if let Some(merge) =
                                            self.new_i_position(offset + 1, new_errors)
                                        {
                                            successors.push(merge);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Phase 2d/3b: Multi-character operations - SPLIT ⟨1,2,1⟩
            // Split: consume 1 word char, match 2 input chars (two-step operation)
            // Phase 3b: Support phonetic ⟨1,2⟩ operations (e.g., "k"→"ch")
            let split_ops: Vec<_> = operations
                .operations()
                .iter()
                .filter(|op| op.consume_x() == 1 && op.consume_y() == 2)
                .collect();

            let can_enter_split = split_ops.iter().any(|op| {
                add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost).is_some()
            });

            if !split_ops.is_empty() && can_enter_split {
                // H2 Optimization: Using word_slice_chars from method beginning

                // Check if we can enter split state
                if let Some(match_index) = bounded_index(match_index_i32, word_slice_chars.len()) {
                    if word_slice_chars[match_index] != '$' {
                        // The second target scalar is not available until the
                        // next transition. Enter speculatively; completion checks
                        // every restriction and charges the selected cost.
                        if let Some(split) =
                            self.new_i_splitting_position(offset, errors, input_char)
                        {
                            successors.push(split);
                        }
                    }
                }
            }

            // The skip-to-match shortcut is a compact form of repeated
            // unrestricted unit deletions. It is valid only for the certified
            // classical operation lattice; other operation sets must advance
            // exclusively through their configured rules.
            if self.use_classical_subsumption && !has_match && errors < self.max_cost {
                for idx in (match_index + 1)..bit_vector.len() {
                    if bit_vector.is_match(idx) {
                        let skip_distance = idx - match_index;
                        if let Some(new_errors) = add_distance_to_errors(
                            errors,
                            skip_distance,
                            self.cost_scale,
                            self.max_cost,
                        ) {
                            let Some(new_offset) = i32::try_from(skip_distance)
                                .ok()
                                .and_then(|distance| offset.checked_add(distance))
                            else {
                                break;
                            };
                            if let Some(succ) = self.new_i_position(new_offset, new_errors) {
                                successors.push(succ);
                            }
                        }
                        break;
                    }
                }
            }

            return successors;
        }

        // Outside the dictionary window, only an operation that consumes the
        // current target scalar without consuming a source scalar can advance
        // this transition. Never synthesize a unit insertion/deletion that is
        // absent from the configured operation set.
        for operation in operations.operations() {
            if operation.consume_x() == 0
                && operation.consume_y() == 1
                && operation.can_apply(&[], input_char_bytes)
            {
                if let Some(new_errors) =
                    add_weight_to_errors(errors, operation.weight(), self.cost_scale, self.max_cost)
                {
                    if let Some(succ) = self.new_i_position(offset, new_errors) {
                        successors.push(succ);
                    }
                }
            }
        }

        successors
    }

    /// Compute successors for M-type positions with runtime-configurable operations
    ///
    /// Similar logic to I-type, but positions are relative to end of word.
    fn successors_m_type(
        &self,
        offset: i32,
        errors: usize,
        input: &GeneralizedTransitionInput<'_>,
    ) -> Vec<GeneralizedPosition> {
        let operations = input.operations;
        let bit_vector = input.bit_vector;
        let word_slice = input.word_slice;
        let input_char = input.input_char;
        let mut successors = Vec::new();

        // H2 Optimization: Collect word_slice characters once instead of repeatedly
        let word_slice_chars: Vec<char> = word_slice.chars().collect();

        // H1 Optimization: Pre-encode input_char once to avoid repeated String allocations
        let mut input_char_buf = [0u8; 4];
        let input_char_bytes = input_char.encode_utf8(&mut input_char_buf).as_bytes();

        // For M-type, bit vector index is computed differently
        // M + offset#errors at input k corresponds to word position m + offset
        // where m is the word length (not known here, so we use simplified logic)
        let bit_index_i32 = offset + bit_vector.len() as i32;
        let has_match = bounded_index(bit_index_i32, bit_vector.len())
            .is_some_and(|idx| bit_vector.is_match(idx));

        // Iterate over single-character operations; multi-character
        // operations are handled by the dedicated sections below.
        for op in operations.operations() {
            if op.consume_x() > 1 || op.consume_y() > 1 {
                continue;
            }

            // Classify operation type and generate successors
            if op.is_match() && has_match {
                // Match operation: ⟨1, 1, 0.0⟩
                // Phase 3: Check can_apply() with actual characters
                // H2 Optimization: Using word_slice_chars from method beginning
                if let Some(bit_index) = bounded_index(bit_index_i32, word_slice_chars.len()) {
                    // H1 Optimization: Use stack buffer instead of heap allocation
                    let mut word_char_buf = [0u8; 4];
                    let word_char_bytes = word_slice_chars[bit_index]
                        .encode_utf8(&mut word_char_buf)
                        .as_bytes();
                    if op.can_apply(word_char_bytes, input_char_bytes) {
                        let new_offset = offset + 1;
                        // Phase 4: M-type invariant is -2n ≤ offset ≤ 0
                        // If new_offset > 0, create I-type instead (I-type allows -n ≤ offset ≤ n)
                        if new_offset > 0 {
                            if let Some(succ) = self.new_i_position(new_offset, errors) {
                                successors.push(succ);
                            }
                        } else {
                            if let Some(succ) = self.new_m_position(new_offset, errors) {
                                successors.push(succ);
                            }
                        }
                    }
                }
            } else if op.is_deletion() && errors < self.max_cost {
                // Delete operation: ⟨1, 0, w⟩
                // Phase 3: Check can_apply() with word character and empty input
                if let Some(new_errors) =
                    add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost)
                {
                    // H2 Optimization: Using word_slice_chars from method beginning
                    if let Some(bit_index) = bounded_index(bit_index_i32, word_slice_chars.len()) {
                        // H1 Optimization: Use stack buffer instead of heap allocation
                        let mut word_char_buf = [0u8; 4];
                        let word_char_bytes = word_slice_chars[bit_index]
                            .encode_utf8(&mut word_char_buf)
                            .as_bytes();
                        if op.can_apply(word_char_bytes, &[]) {
                            if let Some(succ) = self.new_m_position(offset, new_errors) {
                                successors.push(succ);
                            }
                        }
                    }
                }
            } else if op.is_insertion() && errors < self.max_cost {
                // Insert ⟨0, 1, w⟩
                // Phase 3: Check can_apply() with empty word and input character
                if let Some(new_errors) =
                    add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost)
                {
                    // H1 Optimization: Use pre-encoded input_char_bytes (no allocation)
                    if op.can_apply(&[], input_char_bytes) {
                        let new_offset = offset + 1;
                        // Phase 4: M-type invariant is -2n ≤ offset ≤ 0
                        // If new_offset > 0, create I-type instead (I-type allows -n ≤ offset ≤ n)
                        if new_offset > 0 {
                            if let Some(succ) = self.new_i_position(new_offset, new_errors) {
                                successors.push(succ);
                            }
                        } else {
                            if let Some(succ) = self.new_m_position(new_offset, new_errors) {
                                successors.push(succ);
                            }
                        }
                    }
                }
            } else if op.is_substitution() && errors < self.max_cost {
                // Substitute ⟨1, 1, w⟩
                // Phase 3: Check can_apply() with word and input characters
                if let Some(new_errors) =
                    add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost)
                {
                    // H2 Optimization: Using word_slice_chars from method beginning
                    if let Some(bit_index) = bounded_index(bit_index_i32, word_slice_chars.len()) {
                        // H1 Optimization: Use stack buffer instead of heap allocation
                        let mut word_char_buf = [0u8; 4];
                        let word_char_bytes = word_slice_chars[bit_index]
                            .encode_utf8(&mut word_char_buf)
                            .as_bytes();
                        if op.can_apply(word_char_bytes, input_char_bytes) {
                            let new_offset = offset + 1;
                            // Phase 4: M-type invariant is -2n ≤ offset ≤ 0
                            // If new_offset > 0, create I-type instead (I-type allows -n ≤ offset ≤ n)
                            if new_offset > 0 {
                                if let Some(succ) = self.new_i_position(new_offset, new_errors) {
                                    successors.push(succ);
                                }
                            } else {
                                if let Some(succ) = self.new_m_position(new_offset, new_errors) {
                                    successors.push(succ);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Phase 2d/3b: Multi-character operations - TRANSPOSITION ⟨2,2,1⟩
        // Phase 3b: Support phonetic ⟨2,2⟩ operations (e.g., "qu"↔"kw")
        let transpose_ops: Vec<_> = operations
            .operations()
            .iter()
            .filter(|op| op.consume_x() == 2 && op.consume_y() == 2)
            .collect();

        if transpose_ops.iter().any(|op| {
            add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost).is_some()
        }) {
            let next_match_index_i32 = bit_index_i32 + 1;
            // H2 Optimization: Using word_slice_chars from method beginning

            // Check if we have enough word characters for transpose
            if let Some(next_match_index) =
                bounded_index(next_match_index_i32, word_slice_chars.len())
            {
                if word_slice_chars[next_match_index] != '$' {
                    if let Some(trans) = self.new_m_transposing_position(offset - 1, errors) {
                        successors.push(trans);
                    }
                }
            }
        }

        // Phase 2d/3: Multi-character operations - MERGE ⟨2,1⟩
        // Merge: consume 2 word chars, match 1 input char (direct operation)
        // Phase 3: Supports phonetic operations like "ch"→"k", "ph"→"f"
        if errors < self.max_cost {
            // H1 Optimization: Reuse word_slice_chars collected at method start (no redundant collection)
            let next_match_index_i32 = bit_index_i32 + 1;

            if let Some(next_match_index) =
                bounded_pair_start(next_match_index_i32, word_slice_chars.len())
            {
                // Check if we have enough word characters (need 2 consecutive chars)
                // Skip padding chars '$'
                if word_slice_chars[next_match_index] != '$'
                    && word_slice_chars[next_match_index + 1] != '$'
                {
                    // H1 Optimization: Encode 2 word characters using stack buffers
                    let mut word_2chars_buf = [0u8; 8]; // Max 4 bytes per char, 2 chars = 8 bytes
                    let mut word_2chars_len = 0usize;
                    {
                        let char1_bytes = word_slice_chars[next_match_index]
                            .encode_utf8(&mut word_2chars_buf[0..4]);
                        word_2chars_len += char1_bytes.len();
                        let char2_bytes = word_slice_chars[next_match_index + 1].encode_utf8(
                            &mut word_2chars_buf[word_2chars_len..word_2chars_len + 4],
                        );
                        word_2chars_len += char2_bytes.len();
                    }
                    let word_2chars_bytes = &word_2chars_buf[..word_2chars_len];

                    // Check all ⟨2,1⟩ operations
                    for op in operations.operations() {
                        if op.consume_x() == 2 && op.consume_y() == 1 {
                            // Phase 3: Use can_apply() for phonetic operations
                            if op.can_apply(word_2chars_bytes, input_char_bytes) {
                                if let Some(new_errors) = add_weight_to_errors(
                                    errors,
                                    op.weight(),
                                    self.cost_scale,
                                    self.max_cost,
                                ) {
                                    // Direct transition: offset+1, errors+weight
                                    if let Some(merge) = self.new_m_position(offset + 1, new_errors)
                                    {
                                        successors.push(merge);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Phase 2d/3b: Multi-character operations - SPLIT ⟨1,2,1⟩
        // Split: consume 1 input char, match 2 word chars (two-step operation)
        // Phase 3b: Support phonetic ⟨1,2⟩ operations (e.g., "k"→"ch")
        let split_ops: Vec<_> = operations
            .operations()
            .iter()
            .filter(|op| op.consume_x() == 1 && op.consume_y() == 2)
            .collect();

        let can_enter_split = split_ops.iter().any(|op| {
            add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost).is_some()
        });

        if !split_ops.is_empty() && can_enter_split {
            let next_match_index_i32 = bit_index_i32;
            // H2 Optimization: Using word_slice_chars from method beginning

            // Check if we can enter split state
            if let Some(next_match_index) =
                bounded_index(next_match_index_i32, word_slice_chars.len())
            {
                if word_slice_chars[next_match_index] != '$' {
                    if let Some(split) = self.new_m_splitting_position(offset, errors, input_char) {
                        successors.push(split);
                    }
                }
            }
        }

        successors
    }

    /// Compute successors for I-type transposing positions
    ///
    /// Complete the transposition operation: consume the second input character,
    /// match against current word position, and return to usual state.
    ///
    /// # Transposition Complete Logic
    ///
    /// From transposing state I+(offset)#(errors)_t:
    /// - Check bit_vector[offset + n] (current position)
    /// - If the complete slices satisfy a configured rule, create an ordinary
    ///   position and add that rule's exact scaled cost.
    fn successors_i_transposing(
        &self,
        offset: i32,
        errors: usize,
        input: &GeneralizedTransitionInput<'_>,
    ) -> Vec<GeneralizedPosition> {
        let operations = input.operations;
        let word_slice = input.word_slice;
        let input_char = input.input_char;
        let mut successors = Vec::new();
        let n = self.max_distance as i32;
        let match_index_i32 = offset + n;

        // H2 Optimization: Collect word_slice characters once
        let word_slice_chars: Vec<char> = word_slice.chars().collect();

        // Phase 3b: Complete transpose with phonetic validation
        // Extract 2 word characters that are being transposed
        let Some(match_index) = bounded_pair_start(match_index_i32, word_slice_chars.len()) else {
            return successors;
        };
        if word_slice_chars[match_index] == '$' || word_slice_chars[match_index + 1] == '$' {
            return successors;
        }
        let word_2chars: String = word_slice_chars[match_index..match_index + 2]
            .iter()
            .collect();

        // Get both input characters (previous + current)
        let prev_char = self.previous_input_char.unwrap_or('\0');
        let curr_char = input_char;
        let input_2chars = format!("{}{}", prev_char, curr_char);

        // Complete every applicable two-for-two operation and charge its exact
        // cost now that both target characters are available.
        for op in operations.operations() {
            if op.consume_x() == 2
                && op.consume_y() == 2
                && op.applies_to_slices(&word_2chars, &input_2chars)
            {
                if let Some(new_errors) =
                    add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost)
                {
                    if let Some(succ) = self.new_i_position(offset + 1, new_errors) {
                        successors.push(succ);
                    }
                }
            }
        }

        successors
    }

    /// Compute successors for M-type transposing positions
    ///
    /// Complete the transposition operation for M-type (final) positions.
    ///
    /// # Transposition Complete Logic
    ///
    /// From transposing state M+(offset)#(errors)_t:
    /// - Check bit_vector at appropriate index
    /// - If the complete slices satisfy a configured rule, create an ordinary
    ///   M-position and add that rule's exact scaled cost.
    fn successors_m_transposing(
        &self,
        offset: i32,
        errors: usize,
        input: &GeneralizedTransitionInput<'_>,
    ) -> Vec<GeneralizedPosition> {
        let operations = input.operations;
        let bit_vector = input.bit_vector;
        let word_slice = input.word_slice;
        let input_char = input.input_char;
        let mut successors = Vec::new();
        let bit_index_i32 = offset + bit_vector.len() as i32;

        // H2 Optimization: Collect word_slice characters once
        let word_slice_chars: Vec<char> = word_slice.chars().collect();

        // Phase 3b: Complete transpose with phonetic validation
        // Extract 2 word characters that are being transposed
        let Some(next_match_index) = bounded_pair_start(bit_index_i32, word_slice_chars.len())
        else {
            return successors;
        };
        if word_slice_chars[next_match_index] == '$'
            || word_slice_chars[next_match_index + 1] == '$'
        {
            return successors;
        }
        let word_2chars: String = word_slice_chars[next_match_index..next_match_index + 2]
            .iter()
            .collect();

        // Get both input characters (previous + current)
        let prev_char = self.previous_input_char.unwrap_or('\0');
        let curr_char = input_char;
        let input_2chars = format!("{}{}", prev_char, curr_char);

        for op in operations.operations() {
            if op.consume_x() == 2
                && op.consume_y() == 2
                && op.applies_to_slices(&word_2chars, &input_2chars)
            {
                if let Some(new_errors) =
                    add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost)
                {
                    if let Some(succ) = self.new_m_position(offset + 1, new_errors) {
                        successors.push(succ);
                    }
                }
            }
        }

        successors
    }

    /// Compute successors for I-type splitting positions
    ///
    /// Complete the split operation: consume the second input character,
    /// match against current word position, and return to usual state.
    ///
    /// # Split Complete Logic
    ///
    /// From splitting state I+(offset)#(errors)_s:
    /// - Check bit_vector[offset + n] (current position for second word char)
    /// - If the complete slices satisfy a configured rule, create an ordinary
    ///   position at the completed coordinate and add the exact scaled cost.
    ///
    /// # Phase 4: Formal Verification Fix
    ///
    /// When subword is empty, uses formally proven formula to calculate word position:
    /// `word_pos = input_position + offset` (from SubwordOperations.v)
    fn successors_i_splitting(
        &self,
        offset: i32,
        errors: usize,
        entry_char: char,
        input: &GeneralizedTransitionInput<'_>,
    ) -> Vec<GeneralizedPosition> {
        let operations = input.operations;
        let full_word = input.full_word;
        let word_chars = input.word_chars;
        let word_slice = input.word_slice;
        let input_char = input.input_char;
        let input_position = input.input_position;
        let mut successors = Vec::new();
        let n = self.max_distance as i32;
        let match_index_i32 = offset + n;

        // H2 Optimization: Collect word_slice characters once
        let word_slice_chars: Vec<char> = word_slice.chars().collect();

        // Phase 3b: Complete split with phonetic validation
        // Extract word character that was split

        // Phase 3b fix: Handle negative match_index or empty word_slice by using full_word
        let word_1char = if match_index_i32 < 0 || word_slice_chars.is_empty() {
            // H2 Optimization: Use pre-computed word_chars if available, else collect on-demand
            // Phase 4 FIX: input_position is 1-indexed (thesis notation)
            // Split entered at (input_position-1), word_pos (1-indexed) = (input_position-1) + offset
            // Convert to 0-indexed: word_pos = ((input_position-1) + offset) - 1 = input_position + offset - 2
            let Some(word_pos) = word_position_index(input_position, offset) else {
                return successors;
            };

            // Use pre-computed word_chars if available, else collect on-demand for callers without context.
            let owned_word_chars: Vec<char>;
            let full_word_chars: &[char] = match word_chars {
                Some(chars) => chars,
                None => {
                    owned_word_chars = full_word.chars().collect();
                    &owned_word_chars
                }
            };

            if word_pos < full_word_chars.len() && full_word_chars[word_pos] != '$' {
                full_word_chars[word_pos].to_string()
            } else {
                // Past word end - no character to validate
                return successors;
            }
        } else {
            // Normal case: extract from subword
            // Phase 4: With offset unchanged, subword has slid forward by 1
            // The character we entered the split on is now at match_index-1
            let Some(match_index) = usize::try_from(match_index_i32).ok() else {
                return successors;
            };
            let adjusted_index = if match_index > 0 { match_index - 1 } else { 0 };

            if adjusted_index >= word_slice_chars.len() || word_slice_chars[adjusted_index] == '$' {
                return successors;
            }
            word_slice_chars[adjusted_index].to_string()
        };

        // Get both input characters (entry_char + current)
        // Use entry_char (the char read when entering this split state) instead of state-level previous_input_char
        let prev_char = entry_char;
        let curr_char = input_char;
        let input_2chars = format!("{}{}", prev_char, curr_char);

        // Phase 3b: Check PHONETIC split operations FIRST ⟨1,2⟩ (more specific)
        for op in operations.operations() {
            if op.consume_x() == 1
                && op.consume_y() == 2
                && op.applies_to_slices(&word_1char, &input_2chars)
            {
                let Some(new_errors) =
                    add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost)
                else {
                    continue;
                };
                // Phase 4: offset UNCHANGED on completion (per PhoneticOperations.v)
                // Advancement happens via sliding subword window, not offset changes
                let new_offset = offset; // Phase 4 FIX: unchanged (was offset+1)

                // Check if we've reached or passed the end of the word
                // If so, create M-type position; otherwise I-type
                // Phase 4: input_position is 1-indexed, convert to 0-indexed word position
                let full_word_len =
                    word_chars.map_or_else(|| full_word.chars().count(), |chars| chars.len());
                let Some(result_word_pos) = word_position_index(input_position, new_offset) else {
                    return successors;
                };
                // After consuming the character in the split, we advance by 1
                let Some(next_word_pos) = result_word_pos.checked_add(1) else {
                    return successors;
                };

                if next_word_pos >= full_word_len {
                    // Past word end -> M-type position
                    let m_offset = if next_word_pos == full_word_len {
                        // Exactly at word end -> M+0
                        0
                    } else {
                        // Strictly past word end -> calculate offset
                        let result_offset = new_offset + 1;
                        result_offset - (full_word_len as i32 - n)
                    };

                    if let Some(succ) = self.new_m_position(m_offset, new_errors) {
                        successors.push(succ);
                    } else {
                        // Fallback: try creating I-type instead with unchanged offset
                        // This handles the case where we're exactly at word end but M-type invariant can't be satisfied
                        if let Some(succ) = self.new_i_position(new_offset, new_errors) {
                            successors.push(succ);
                        }
                    }
                } else {
                    // Still within word -> I-type position
                    if let Some(succ) = self.new_i_position(new_offset, new_errors) {
                        successors.push(succ);
                    }
                }
            }
        }

        successors
    }

    /// Compute successors for M-type splitting positions
    ///
    /// Complete the split operation for M-type (final) positions.
    ///
    /// # Split Complete Logic
    ///
    /// From splitting state M+(offset)#(errors)_s:
    /// - Check bit_vector at appropriate index
    /// - If the complete slices satisfy a configured rule, create an ordinary
    ///   M-position and add the exact scaled cost.
    fn successors_m_splitting(
        &self,
        offset: i32,
        errors: usize,
        entry_char: char,
        input: &GeneralizedTransitionInput<'_>,
    ) -> Vec<GeneralizedPosition> {
        let operations = input.operations;
        let bit_vector = input.bit_vector;
        let full_word = input.full_word;
        let word_chars = input.word_chars;
        let word_slice = input.word_slice;
        let input_char = input.input_char;
        let input_position = input.input_position;
        let mut successors = Vec::new();
        let bit_index_i32 = offset + bit_vector.len() as i32;

        // H2 Optimization: Collect word_slice characters once
        let word_slice_chars: Vec<char> = word_slice.chars().collect();

        // Phase 3b: Complete split with phonetic validation
        // Extract word character that was split
        let next_match_index_i32 = bit_index_i32;

        // Phase 3b fix: Handle negative or out-of-bounds index by using full_word
        let word_1char = if next_match_index_i32 < 0 || word_slice_chars.is_empty() {
            // H2 Optimization: Use pre-computed word_slice_chars instead of collecting

            // Phase 4 FIX: input_position is 1-indexed, convert to 0-indexed word position
            let Some(word_pos) = word_position_index(input_position, offset) else {
                return successors;
            };

            // Use pre-computed word_chars if available, else collect on-demand for callers without context.
            let owned_word_chars: Vec<char>;
            let full_word_chars: &[char] = match word_chars {
                Some(chars) => chars,
                None => {
                    owned_word_chars = full_word.chars().collect();
                    &owned_word_chars
                }
            };

            if word_pos < full_word_chars.len() && full_word_chars[word_pos] != '$' {
                full_word_chars[word_pos].to_string()
            } else {
                // Past word end - no character to validate
                return successors;
            }
        } else {
            // Normal case: extract from subword
            // Phase 4: With offset unchanged, subword has slid forward by 1
            // The character we entered the split on is now at next_match_index-1
            let Some(next_match_index) = usize::try_from(next_match_index_i32).ok() else {
                return successors;
            };
            let adjusted_index = if next_match_index > 0 {
                next_match_index - 1
            } else {
                0
            };

            if adjusted_index >= word_slice_chars.len() || word_slice_chars[adjusted_index] == '$' {
                return successors;
            }
            word_slice_chars[adjusted_index].to_string()
        };

        // Get both input characters (entry_char + current)
        // Use entry_char (the char read when entering this split state) instead of state-level previous_input_char
        let prev_char = entry_char;
        let curr_char = input_char;
        let input_2chars = format!("{}{}", prev_char, curr_char);

        // Phase 3b: Check PHONETIC split operations FIRST ⟨1,2⟩ (more specific)
        for op in operations.operations() {
            if op.consume_x() == 1
                && op.consume_y() == 2
                && op.applies_to_slices(&word_1char, &input_2chars)
            {
                let Some(new_errors) =
                    add_weight_to_errors(errors, op.weight(), self.cost_scale, self.max_cost)
                else {
                    continue;
                };
                // Phase 4: offset UNCHANGED on completion (per PhoneticOperations.v)
                let new_offset = offset; // Phase 4 FIX: unchanged (was offset+1)
                if let Some(succ) = self.new_m_position(new_offset, new_errors) {
                    successors.push(succ);
                }
            }
        }

        successors
    }
}

impl fmt::Display for GeneralizedState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, pos) in self.positions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", pos)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let state = GeneralizedState::initial(2);
        assert_eq!(state.len(), 1);
        assert!(!state.is_final());
        assert!(!state.is_empty());
    }

    #[test]
    fn test_add_position_no_subsumption() {
        let mut state = GeneralizedState::new(2);
        // Add positions that don't subsume each other
        // I + 0#1 does not subsume I + (-1)#1 (same errors, different offsets)
        // Valid positions: |0| ≤ 1 ✓ and |-1| ≤ 1 ✓
        state.add_position(
            GeneralizedPosition::new_i(0, 1, 2)
                .expect("test fixture: GeneralizedPosition::new_i with valid args"),
        );
        state.add_position(
            GeneralizedPosition::new_i(-1, 1, 2)
                .expect("test fixture: GeneralizedPosition::new_i with valid args"),
        );
        assert_eq!(state.len(), 2);
    }

    #[test]
    fn add_position_preserves_set_semantics_for_equal_positions() {
        let mut state = GeneralizedState::new(2);
        let position = GeneralizedPosition::new_i(0, 1, 2).expect("test fixture: valid I-position");

        state.add_position(position.clone());
        state.add_position(position);

        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_final_state() {
        let mut state = GeneralizedState::new(2);
        state.add_position(
            GeneralizedPosition::new_m(0, 0, 2)
                .expect("test fixture: GeneralizedPosition::new_m with valid args"),
        );
        assert!(state.is_final());
    }

    #[test]
    fn test_display() {
        let mut state = GeneralizedState::new(2);
        state.add_position(
            GeneralizedPosition::new_i(0, 1, 2)
                .expect("test fixture: GeneralizedPosition::new_i with valid args"),
        );
        state.add_position(
            GeneralizedPosition::new_i(-1, 1, 2)
                .expect("test fixture: GeneralizedPosition::new_i with valid args"),
        );
        let display = format!("{}", state);
        assert!(display.contains("I + 0#1") || display.contains("I + -1#1"));
        assert!(display.contains("I + -1#1") || display.contains("I + 0#1"));
    }

    #[test]
    fn test_m_transposing_negative_pair_start_returns_empty_successor_state() {
        let max_distance = 2;
        let mut state = GeneralizedState::new(max_distance);
        state.previous_input_char = Some('a');
        state.add_position(
            GeneralizedPosition::new_m_transposing(-2, 1, max_distance)
                .expect("test fixture: valid M-transposing position"),
        );

        let operations = crate::transducer::OperationSet::with_transposition();
        let cv = CharacteristicVector::new('b', "a");
        let word_chars: Vec<char> = "a".chars().collect();

        let next = state.transition(GeneralizedTransitionInput::new(
            &operations,
            &cv,
            "a",
            Some(&word_chars),
            "a",
            'b',
            1,
        ));

        assert!(next.is_none());
    }

    #[test]
    fn test_m_splitting_negative_word_position_returns_empty_successor_state() {
        let max_distance = 2;
        let mut state = GeneralizedState::new(max_distance);
        state.add_position(
            GeneralizedPosition::new_m_splitting(-2, 1, max_distance, 'a')
                .expect("test fixture: valid M-splitting position"),
        );

        let operations = crate::transducer::OperationSet::with_merge_split();
        let cv = CharacteristicVector::new('b', "a");
        let word_chars: Vec<char> = "a".chars().collect();

        let next = state.transition(GeneralizedTransitionInput::new(
            &operations,
            &cv,
            "a",
            Some(&word_chars),
            "",
            'b',
            1,
        ));

        assert!(next.is_none());
    }

    #[test]
    fn test_i_weighted_operation_over_budget_does_not_wrap_errors() {
        let max_distance = 2;
        let mut state = GeneralizedState::new(max_distance);
        state.add_position(
            GeneralizedPosition::new_i(0, 1, max_distance).expect("test fixture: valid I-position"),
        );

        let operations = crate::transducer::OperationSetBuilder::new()
            .with_operation(crate::transducer::OperationType::new(
                1,
                1,
                255.0,
                "heavy_substitute",
            ))
            .build();
        let cv = CharacteristicVector::new('x', "$$a");

        let next = state.transition(GeneralizedTransitionInput::new(
            &operations,
            &cv,
            "a",
            None,
            "$$a",
            'x',
            1,
        ));

        assert!(next.is_none());
    }

    #[test]
    fn direct_transition_represents_fractional_weight_exactly() {
        use crate::transducer::{OperationSetBuilder, OperationType};

        let operations = OperationSetBuilder::new()
            .with_operation(OperationType::new(1, 1, 0.15, "fractional"))
            .build();
        let state = GeneralizedState::initial(1);
        let vector = CharacteristicVector::new('b', "$a");
        let input = GeneralizedTransitionInput::new(&operations, &vector, "a", None, "$a", 'b', 1);

        let next = state
            .try_transition(input)
            .expect("finite decimal scale")
            .expect("fractional substitution is in budget");
        assert_eq!(next.cost_scale().denominator(), 20);
        assert_eq!(next.max_scaled_cost(), 20);
        assert!(next.positions().any(|position| position.errors() == 3));
        assert!(next.positions().all(|position| position.errors() != 0));
    }

    #[test]
    fn direct_transition_selects_the_least_applicable_scaled_cost_independent_of_order() {
        use crate::transducer::{OperationSetBuilder, OperationType, SubstitutionSet};

        let mut rule = SubstitutionSet::new();
        rule.allow_str("ph", "f");
        let operations = OperationSetBuilder::new()
            .with_operation(OperationType::with_restriction(
                2,
                1,
                0.8,
                rule.clone(),
                "heavy_ph",
            ))
            .with_operation(OperationType::with_restriction(2, 1, 0.2, rule, "light_ph"))
            .build();
        let state = GeneralizedState::initial(1);
        let vector = CharacteristicVector::new('f', "$ph");
        let input =
            GeneralizedTransitionInput::new(&operations, &vector, "ph", None, "$ph", 'f', 1);

        let next = state
            .try_transition(input)
            .expect("finite decimal scale")
            .expect("restricted merge applies");
        assert_eq!(next.cost_scale().denominator(), 5);
        assert_eq!(
            next.positions().map(GeneralizedPosition::errors).min(),
            Some(1)
        );
    }

    #[test]
    fn direct_transition_reports_non_finite_cost_scale() {
        use crate::transducer::{OperationSetBuilder, OperationType};

        let operations = OperationSetBuilder::new()
            .with_operation(OperationType::new(1, 1, f64::INFINITY, "invalid"))
            .build();
        let state = GeneralizedState::initial(1);
        let vector = CharacteristicVector::new('b', "$a");
        let input = GeneralizedTransitionInput::new(&operations, &vector, "a", None, "$a", 'b', 1);

        assert_eq!(
            state.try_transition(input),
            Err(GeneralizedStateError::Scale(ScaleError::NonFiniteWeight))
        );
    }

    #[test]
    fn direct_transition_reports_unrepresentable_operation_arity() {
        use crate::transducer::{OperationSetBuilder, OperationType};

        let operations = OperationSetBuilder::new()
            .with_operation(OperationType::new(3, 1, 1.0, "three_to_one"))
            .build();
        let state = GeneralizedState::initial(3);
        let vector = CharacteristicVector::new('a', "$abc");
        let input =
            GeneralizedTransitionInput::new(&operations, &vector, "abc", None, "$abc", 'a', 1);

        assert_eq!(
            state.try_transition(input),
            Err(GeneralizedStateError::UnsupportedOperationArity {
                name: "three_to_one".into(),
                consume_x: 3,
                consume_y: 1,
            })
        );
        assert!(state.transition(input).is_none());
    }

    #[test]
    fn only_the_exact_standard_lattice_enables_classical_subsumption() {
        use crate::transducer::{OperationSet, OperationSetBuilder, OperationType};

        assert!(is_classical_levenshtein_lattice(&OperationSet::standard()));
        assert!(!is_classical_levenshtein_lattice(
            &OperationSet::with_transposition()
        ));
        let integer_weighted = OperationSetBuilder::new()
            .with_standard_ops()
            .with_operation(OperationType::new(1, 1, 2.0, "heavy"))
            .build();
        assert!(!is_classical_levenshtein_lattice(&integer_weighted));
    }

    #[test]
    fn hamming_state_does_not_synthesize_skip_to_match_deletions() {
        use crate::transducer::OperationSetBuilder;

        let operations = OperationSetBuilder::new()
            .with_match()
            .with_substitution()
            .build();
        let state = GeneralizedState::initial(2);
        let vector = CharacteristicVector::new('a', "$$ba");
        let input =
            GeneralizedTransitionInput::new(&operations, &vector, "ba", None, "$$ba", 'a', 1);

        let next = state
            .try_transition(input)
            .expect("Hamming costs are exactly representable")
            .expect("substitution remains available");
        assert!(next.positions().any(|position| position.offset() == 0));
        assert!(next.positions().all(|position| position.offset() != 1));
    }

    #[test]
    fn test_i_skip_to_far_match_does_not_wrap_errors() {
        let max_distance = 2;
        let mut state = GeneralizedState::new(max_distance);
        state.add_position(
            GeneralizedPosition::new_i(0, 1, max_distance).expect("test fixture: valid I-position"),
        );

        let operations = crate::transducer::OperationSet::standard();
        let word = format!("{}a", "x".repeat(257));
        let cv = CharacteristicVector::new('a', &word);

        let next = state
            .transition(GeneralizedTransitionInput::new(
                &operations,
                &cv,
                &word,
                None,
                &word,
                'a',
                1,
            ))
            .expect("standard delete/substitute successors remain in budget");

        assert!(next
            .positions()
            .all(|pos| pos.errors() <= usize::from(max_distance)));
        let delete = GeneralizedPosition::new_i(-1, 2, max_distance)
            .expect("test fixture: valid I-position");
        let substitute =
            GeneralizedPosition::new_i(0, 2, max_distance).expect("test fixture: valid I-position");
        assert!(next.positions().any(|pos| pos == &delete));
        assert!(next.positions().any(|pos| pos == &substitute));
    }
}
