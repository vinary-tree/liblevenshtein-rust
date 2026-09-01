//! Exact unit-cost correction to a multi-kind Dyck language.
//!
//! Opening kind `r` is token `r`; its matching closer is `kinds + r`.
//! The interval dynamic program is exact for insertion, deletion, and
//! substitution cost one. It uses $`\mathcal{O}(kn^3)`$ time and $`\mathcal{O}(n^2)`$
//! memory, and records deterministic backpointers for a minimum-cost witness.

use std::error::Error;
use std::fmt::{self, Display};

/// Default work ceiling for exact correction.
pub const DYCK_CORRECTION_MAX_WORK: usize = 100_000_000;

/// Error reported before or during exact Dyck correction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DyckCorrectionError {
    /// At least one bracket kind is required for correction.
    NoBracketKinds,
    /// `2 * kinds` cannot be represented by the token alphabet.
    AlphabetOverflow {
        /// Configured bracket-kind count.
        kinds: usize,
    },
    /// An input token is not an opener or closer in the configured alphabet.
    UnknownToken {
        /// Zero-based input offset.
        index: usize,
        /// Invalid token.
        token: u64,
        /// Configured bracket-kind count.
        kinds: usize,
    },
    /// The cubic dynamic program exceeds the configured work policy.
    WorkLimit {
        /// Configured bracket kinds.
        kinds: usize,
        /// Input token count.
        input_len: usize,
        /// Saturating estimate $`k(n+1)^3`$.
        estimated: usize,
        /// Configured ceiling.
        limit: usize,
    },
    /// The quadratic interval table cannot be represented by `usize`.
    TableSizeOverflow {
        /// Input token count.
        input_len: usize,
    },
}

impl Display for DyckCorrectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBracketKinds => {
                formatter.write_str("exact Dyck correction requires at least one bracket kind")
            }
            Self::AlphabetOverflow { kinds } => {
                write!(formatter, "two-sided bracket alphabet overflows for {kinds} kinds")
            }
            Self::UnknownToken {
                index,
                token,
                kinds,
            } => write!(
                formatter,
                "bracket token {token} at input offset {index} is outside the alphabet for {kinds} kinds"
            ),
            Self::WorkLimit {
                kinds,
                input_len,
                estimated,
                limit,
            } => write!(
                formatter,
                "exact Dyck correction for kinds={kinds}, input_len={input_len} estimates {estimated} work units (limit {limit})"
            ),
            Self::TableSizeOverflow { input_len } => write!(
                formatter,
                "exact Dyck correction table size overflows for input_len={input_len}"
            ),
        }
    }
}

impl Error for DyckCorrectionError {}

/// One alignment step in an exact correction witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DyckEdit {
    /// Preserve one input token.
    Keep {
        /// Original input offset.
        input_index: usize,
        /// Preserved token.
        token: u64,
    },
    /// Delete one input token.
    Delete {
        /// Original input offset.
        input_index: usize,
        /// Deleted token.
        token: u64,
    },
    /// Replace one input token.
    Substitute {
        /// Original input offset.
        input_index: usize,
        /// Original token.
        from: u64,
        /// Corrected token.
        to: u64,
    },
    /// Insert a token at an original-input boundary.
    Insert {
        /// Boundary before this original input offset; `input.len()` is the end.
        input_index: usize,
        /// Inserted token.
        token: u64,
    },
}

impl DyckEdit {
    /// Unit Levenshtein cost of this witness step.
    pub const fn cost(&self) -> usize {
        match self {
            Self::Keep { .. } => 0,
            Self::Delete { .. } | Self::Substitute { .. } | Self::Insert { .. } => 1,
        }
    }
}

/// Exact minimum correction and a replayable alignment witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DyckCorrection {
    /// Exact unit-cost Levenshtein distance to the configured Dyck language.
    pub distance: usize,
    /// One deterministic minimum-cost balanced word.
    pub corrected: Vec<u64>,
    /// Alignment from the original input to [`Self::corrected`].
    pub edits: Vec<DyckEdit>,
}

impl DyckCorrection {
    /// Replay the witness, returning `None` if it does not match `input`.
    pub fn replay(&self, input: &[u64]) -> Option<Vec<u64>> {
        let mut cursor = 0usize;
        let mut output = Vec::with_capacity(self.corrected.len());
        for edit in &self.edits {
            match *edit {
                DyckEdit::Insert { input_index, token } => {
                    if input_index != cursor {
                        return None;
                    }
                    output.push(token);
                }
                DyckEdit::Keep { input_index, token } => {
                    if input_index != cursor || input.get(cursor) != Some(&token) {
                        return None;
                    }
                    output.push(token);
                    cursor += 1;
                }
                DyckEdit::Delete { input_index, token } => {
                    if input_index != cursor || input.get(cursor) != Some(&token) {
                        return None;
                    }
                    cursor += 1;
                }
                DyckEdit::Substitute {
                    input_index,
                    from,
                    to,
                } => {
                    if input_index != cursor || input.get(cursor) != Some(&from) {
                        return None;
                    }
                    output.push(to);
                    cursor += 1;
                }
            }
        }
        (cursor == input.len()
            && output == self.corrected
            && self.edits.iter().map(DyckEdit::cost).sum::<usize>() == self.distance)
            .then_some(output)
    }
}

/// Exact corrector for a fixed number of bracket kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DyckCorrector {
    kinds: usize,
    max_work: usize,
}

impl DyckCorrector {
    /// Construct a corrector with the default cubic-work ceiling.
    pub const fn new(kinds: usize) -> Self {
        Self {
            kinds,
            max_work: DYCK_CORRECTION_MAX_WORK,
        }
    }

    /// Construct a corrector with an explicit $`k(n+1)^3`$ work ceiling.
    pub const fn with_max_work(kinds: usize, max_work: usize) -> Self {
        Self { kinds, max_work }
    }

    /// Configured bracket-kind count.
    pub const fn kinds(self) -> usize {
        self.kinds
    }

    /// Compute exact distance and one deterministic minimum-cost witness.
    pub fn correct(self, input: &[u64]) -> Result<DyckCorrection, DyckCorrectionError> {
        let alphabet_end = self.validate_input(input)?;
        let side = input
            .len()
            .checked_add(1)
            .ok_or(DyckCorrectionError::TableSizeOverflow {
                input_len: input.len(),
            })?;
        let table_len = side
            .checked_mul(side)
            .ok_or(DyckCorrectionError::TableSizeOverflow {
                input_len: input.len(),
            })?;
        let mut costs = vec![usize::MAX; table_len];
        let mut choices = vec![None; table_len];
        let at = |left: usize, right: usize| left * side + right;
        for boundary in 0..=input.len() {
            costs[at(boundary, boundary)] = 0;
        }

        for length in 1..=input.len() {
            for left in 0..=input.len() - length {
                let right = left + length;
                let mut best_cost = usize::MAX;
                let mut best_choice = None;
                let mut consider = |cost: usize, choice: Choice| {
                    if cost < best_cost {
                        best_cost = cost;
                        best_choice = Some(choice);
                    }
                };

                // Tie order prefers a typed pair consuming both endpoints,
                // then insertion of a missing opener, then insertion of a
                // missing closer, and finally deletion.
                for kind in 0..self.kinds {
                    let open = kind as u64;
                    let close = (self.kinds + kind) as u64;
                    for close_index in left + 1..right {
                        consider(
                            replacement_cost(input[left], open)
                                + costs[at(left + 1, close_index)]
                                + replacement_cost(input[close_index], close)
                                + costs[at(close_index + 1, right)],
                            Choice::PairFromFirst { kind, close_index },
                        );
                    }
                }
                for kind in 0..self.kinds {
                    let close = (self.kinds + kind) as u64;
                    for close_index in left..right {
                        consider(
                            1 + costs[at(left, close_index)]
                                + replacement_cost(input[close_index], close)
                                + costs[at(close_index + 1, right)],
                            Choice::PairFromInsertedOpen { kind, close_index },
                        );
                    }
                }
                for kind in 0..self.kinds {
                    consider(
                        replacement_cost(input[left], kind as u64) + 1 + costs[at(left + 1, right)],
                        Choice::PairWithInsertedClose { kind },
                    );
                }
                consider(1 + costs[at(left + 1, right)], Choice::DeleteFirst);

                costs[at(left, right)] = best_cost;
                choices[at(left, right)] = best_choice;
            }
        }

        let mut corrected = Vec::new();
        let mut edits = Vec::new();
        reconstruct(
            input,
            self.kinds,
            side,
            &choices,
            0,
            input.len(),
            &mut corrected,
            &mut edits,
        );
        debug_assert!(corrected.iter().all(|token| *token < alphabet_end));
        let correction = DyckCorrection {
            distance: costs[at(0, input.len())],
            corrected,
            edits,
        };
        debug_assert!(correction.replay(input).is_some());
        Ok(correction)
    }

    fn validate_input(self, input: &[u64]) -> Result<u64, DyckCorrectionError> {
        if self.kinds == 0 {
            return Err(DyckCorrectionError::NoBracketKinds);
        }
        let alphabet_end_usize = self
            .kinds
            .checked_mul(2)
            .ok_or(DyckCorrectionError::AlphabetOverflow { kinds: self.kinds })?;
        let alphabet_end = u64::try_from(alphabet_end_usize)
            .map_err(|_| DyckCorrectionError::AlphabetOverflow { kinds: self.kinds })?;
        for (index, &token) in input.iter().enumerate() {
            if token >= alphabet_end {
                return Err(DyckCorrectionError::UnknownToken {
                    index,
                    token,
                    kinds: self.kinds,
                });
            }
        }
        let side = input.len().saturating_add(1);
        let estimated = self
            .kinds
            .saturating_mul(side)
            .saturating_mul(side)
            .saturating_mul(side);
        if estimated > self.max_work {
            return Err(DyckCorrectionError::WorkLimit {
                kinds: self.kinds,
                input_len: input.len(),
                estimated,
                limit: self.max_work,
            });
        }
        Ok(alphabet_end)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    DeleteFirst,
    PairWithInsertedClose { kind: usize },
    PairFromFirst { kind: usize, close_index: usize },
    PairFromInsertedOpen { kind: usize, close_index: usize },
}

const fn replacement_cost(actual: u64, expected: u64) -> usize {
    if actual == expected {
        0
    } else {
        1
    }
}

fn consume_as(
    input: &[u64],
    index: usize,
    token: u64,
    output: &mut Vec<u64>,
    edits: &mut Vec<DyckEdit>,
) {
    let actual = input[index];
    output.push(token);
    if actual == token {
        edits.push(DyckEdit::Keep {
            input_index: index,
            token,
        });
    } else {
        edits.push(DyckEdit::Substitute {
            input_index: index,
            from: actual,
            to: token,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct(
    input: &[u64],
    kinds: usize,
    side: usize,
    choices: &[Option<Choice>],
    left: usize,
    right: usize,
    output: &mut Vec<u64>,
    edits: &mut Vec<DyckEdit>,
) {
    if left == right {
        return;
    }
    let choice = choices[left * side + right].expect("non-empty interval has a backpointer");
    match choice {
        Choice::DeleteFirst => {
            edits.push(DyckEdit::Delete {
                input_index: left,
                token: input[left],
            });
            reconstruct(input, kinds, side, choices, left + 1, right, output, edits);
        }
        Choice::PairWithInsertedClose { kind } => {
            consume_as(input, left, kind as u64, output, edits);
            let close = (kinds + kind) as u64;
            output.push(close);
            edits.push(DyckEdit::Insert {
                input_index: left + 1,
                token: close,
            });
            reconstruct(input, kinds, side, choices, left + 1, right, output, edits);
        }
        Choice::PairFromFirst { kind, close_index } => {
            consume_as(input, left, kind as u64, output, edits);
            reconstruct(
                input,
                kinds,
                side,
                choices,
                left + 1,
                close_index,
                output,
                edits,
            );
            consume_as(input, close_index, (kinds + kind) as u64, output, edits);
            reconstruct(
                input,
                kinds,
                side,
                choices,
                close_index + 1,
                right,
                output,
                edits,
            );
        }
        Choice::PairFromInsertedOpen { kind, close_index } => {
            let open = kind as u64;
            output.push(open);
            edits.push(DyckEdit::Insert {
                input_index: left,
                token: open,
            });
            reconstruct(
                input,
                kinds,
                side,
                choices,
                left,
                close_index,
                output,
                edits,
            );
            consume_as(input, close_index, (kinds + kind) as u64, output, edits);
            reconstruct(
                input,
                kinds,
                side,
                choices,
                close_index + 1,
                right,
                output,
                edits,
            );
        }
    }
}

/// Test whether `tokens` is a kind-sensitive balanced word.
pub fn is_dyck_word(tokens: &[u64], kinds: usize) -> Result<bool, DyckCorrectionError> {
    let corrector = DyckCorrector::with_max_work(kinds, usize::MAX);
    let alphabet_end = corrector.validate_input(tokens)?;
    let closing_start = kinds as u64;
    let mut stack = Vec::new();
    for &token in tokens {
        debug_assert!(token < alphabet_end);
        if token < closing_start {
            stack.push(token as usize);
        } else if stack.pop() != Some((token - closing_start) as usize) {
            return Ok(false);
        }
    }
    Ok(stack.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrects_cross_kind_and_missing_delimiters_exactly() {
        let mismatched = DyckCorrector::new(2).correct(&[0, 3]).unwrap();
        assert_eq!(mismatched.distance, 1);
        assert_eq!(mismatched.corrected, vec![0, 2]);
        assert_eq!(mismatched.replay(&[0, 3]), Some(vec![0, 2]));

        let missing_close = DyckCorrector::new(2).correct(&[1]).unwrap();
        assert_eq!(missing_close.distance, 1);
        assert_eq!(missing_close.corrected, vec![1, 3]);

        let missing_open = DyckCorrector::new(2).correct(&[3]).unwrap();
        assert_eq!(missing_open.distance, 1);
        assert_eq!(missing_open.corrected, vec![1, 3]);
    }

    #[test]
    fn preserves_balanced_nesting_and_rejects_bad_inputs() {
        let input = [0, 1, 3, 2, 1, 3];
        let correction = DyckCorrector::new(2).correct(&input).unwrap();
        assert_eq!(correction.distance, 0);
        assert_eq!(correction.corrected, input);
        assert_eq!(is_dyck_word(&correction.corrected, 2), Ok(true));

        assert_eq!(
            DyckCorrector::new(2).correct(&[4]),
            Err(DyckCorrectionError::UnknownToken {
                index: 0,
                token: 4,
                kinds: 2,
            })
        );
        assert_eq!(
            DyckCorrector::new(0).correct(&[]),
            Err(DyckCorrectionError::NoBracketKinds)
        );
    }

    #[test]
    fn work_guard_precedes_table_allocation() {
        assert!(matches!(
            DyckCorrector::with_max_work(2, 1).correct(&[0, 2]),
            Err(DyckCorrectionError::WorkLimit { .. })
        ));
    }
}
