//! Bounded-depth bracket languages and an admissible Dyck-distance bound.
//!
//! Opening kind `r` is encoded as `r`; its closing token is `kinds + r`.
//! Thus a `kinds = 3` alphabet uses openings `0,1,2` and closings `3,4,5`.

use super::{SmallDfa, SmallDfaError, SMALL_DFA_MAX_STATES};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Display};

/// Hard public resource ceiling for generated bounded-depth bracket DFAs.
pub const BRACKET_DFA_MAX_STATES: usize = 4_096;

/// Error from bracket-language construction or projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BracketError {
    /// The exponential stack-state language exceeds its public resource limit.
    StateLimit {
        /// Number of bracket kinds.
        kinds: usize,
        /// Maximum permitted stack depth.
        max_depth: usize,
        /// Exact state count, or `usize::MAX` when arithmetic saturated.
        requested: usize,
        /// Enforced public maximum.
        maximum: usize,
    },
    /// The requested language fits the public resource policy but not the
    /// compact bit-set representation used by [`SmallDfa`].
    SmallDfaLimit {
        /// Number of states required by the bracket language.
        requested: usize,
        /// Current representational maximum.
        maximum: usize,
    },
    /// A token is neither an opening nor a closing token for this alphabet.
    UnknownToken {
        /// Invalid token.
        token: u64,
        /// Number of configured bracket kinds.
        kinds: usize,
    },
    /// Checked construction failed after preflight validation.
    Dfa(SmallDfaError),
}

impl Display for BracketError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StateLimit {
                kinds,
                max_depth,
                requested,
                maximum,
            } => write!(
                formatter,
                "bounded bracket DFA needs {requested} stack states for kinds={kinds}, depth={max_depth}; the exponential sum 1 + k + ... + k^D must not exceed {maximum}"
            ),
            Self::SmallDfaLimit { requested, maximum } => write!(
                formatter,
                "bounded bracket DFA needs {requested} states, but SmallDfa's compact frontier currently supports {maximum}"
            ),
            Self::UnknownToken { token, kinds } => write!(
                formatter,
                "bracket token {token} is outside the opening/closing alphabet for {kinds} kinds"
            ),
            Self::Dfa(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for BracketError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dfa(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SmallDfaError> for BracketError {
    fn from(error: SmallDfaError) -> Self {
        Self::Dfa(error)
    }
}

/// Count stack words of length at most `max_depth` over `kinds` symbols.
fn bracket_state_count(kinds: usize, max_depth: usize) -> usize {
    let mut total = 1usize;
    let mut level = 1usize;
    for _ in 0..max_depth {
        level = level.saturating_mul(kinds);
        total = total.saturating_add(level);
    }
    total
}

/// Construct the exact bounded-depth, kind-sensitive Dyck language.
///
/// State is the complete bracket stack, so the state count is the exponential
/// geometric sum `1 + kinds + ... + kinds^max_depth`. Construction preflights
/// the 4,096-state public resource policy before allocating any state table.
pub fn balanced_depth_dfa(kinds: usize, max_depth: usize) -> Result<SmallDfa<u64>, BracketError> {
    let requested = bracket_state_count(kinds, max_depth);
    if requested > BRACKET_DFA_MAX_STATES {
        return Err(BracketError::StateLimit {
            kinds,
            max_depth,
            requested,
            maximum: BRACKET_DFA_MAX_STATES,
        });
    }
    if requested > SMALL_DFA_MAX_STATES {
        return Err(BracketError::SmallDfaLimit {
            requested,
            maximum: SMALL_DFA_MAX_STATES,
        });
    }

    let mut dfa = SmallDfa::new();
    dfa.set_accepting(0, true)?;

    let mut stacks = vec![Vec::<usize>::new()];
    let mut level_start = 0usize;
    let mut level_end = 1usize;
    for _depth in 1..=max_depth {
        for parent in level_start..level_end {
            for kind in 0..kinds {
                let mut stack = stacks[parent].clone();
                stack.push(kind);
                stacks.push(stack);
                dfa.add_state(false)?;
            }
        }
        level_start = level_end;
        level_end = stacks.len();
    }

    let ids: HashMap<_, _> = stacks
        .iter()
        .cloned()
        .enumerate()
        .map(|(id, stack)| (stack, id as u32))
        .collect();
    for (id, stack) in stacks.iter().enumerate() {
        if stack.len() < max_depth {
            for kind in 0..kinds {
                let mut pushed = stack.clone();
                pushed.push(kind);
                dfa.add_transition(id as u32, kind as u64, ids[&pushed])?;
            }
        }
        if let Some(&kind) = stack.last() {
            let popped = &stack[..stack.len() - 1];
            dfa.add_transition(id as u32, (kinds + kind) as u64, ids[popped])?;
        }
    }
    Ok(dfa)
}

/// Lower-bound kind-sensitive Dyck correction by erasing bracket kinds.
///
/// The scan computes exact unit-cost Levenshtein distance from the projected
/// one-kind word to the one-kind Dyck language. If `o` unmatched opens and `c`
/// unmatched closes remain, the distance is `ceil(o/2) + ceil(c/2)`: one
/// substitution can repair two same-direction unmatched brackets, while an
/// insertion or deletion repairs one. Because kind erasure is length-preserving
/// and maps every kind-sensitive balanced word to a one-kind balanced word,
/// this value cannot exceed distance to the original language.
pub fn balance_lower_bound(tokens: &[u64], kinds: usize) -> Result<usize, BracketError> {
    let closing_start = kinds as u64;
    let alphabet_end = kinds.saturating_mul(2) as u64;
    let mut unmatched_opens = 0usize;
    let mut unmatched_closes = 0usize;

    for &token in tokens {
        if token < closing_start {
            unmatched_opens = unmatched_opens.saturating_add(1);
        } else if token < alphabet_end {
            if unmatched_opens == 0 {
                unmatched_closes = unmatched_closes.saturating_add(1);
            } else {
                unmatched_opens -= 1;
            }
        } else {
            return Err(BracketError::UnknownToken { token, kinds });
        }
    }

    Ok(unmatched_opens.div_ceil(2) + unmatched_closes.div_ceil(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::language::LanguageProduct;

    #[test]
    fn bounded_language_tracks_kind_and_depth() {
        let dfa = balanced_depth_dfa(2, 3).unwrap();
        assert_eq!(dfa.state_count(), 15);
        let product = LanguageProduct::new(dfa, 0);
        assert_eq!(product.distance_to_language([0u64, 1, 3, 2]), Some(0));
        assert_eq!(product.distance_to_language([0u64, 1, 2, 3]), None);
        assert_eq!(
            product.distance_to_language([0u64, 0, 0, 0, 2, 2, 2, 2]),
            None
        );
    }

    #[test]
    fn bounded_language_uses_the_full_public_state_capacity() {
        let dfa = balanced_depth_dfa(2, 5).unwrap();
        assert_eq!(dfa.state_count(), 63);
        let product = LanguageProduct::new(dfa, 0);
        assert_eq!(
            product.distance_to_language([0_u64, 1, 0, 1, 0, 2, 3, 2, 3, 2]),
            Some(0)
        );
    }

    #[test]
    fn exponential_guard_precedes_allocation() {
        let error = balanced_depth_dfa(3, 10).unwrap_err();
        assert_eq!(
            error,
            BracketError::StateLimit {
                kinds: 3,
                max_depth: 10,
                requested: 88_573,
                maximum: BRACKET_DFA_MAX_STATES,
            }
        );
        let message = error.to_string();
        assert!(message.contains("exponential"));
        assert!(message.contains("88573"));
    }

    #[test]
    fn projection_bound_handles_substitution_pairs_and_bad_tokens() {
        assert_eq!(balance_lower_bound(&[], 3), Ok(0));
        assert_eq!(balance_lower_bound(&[0, 0, 0], 3), Ok(2));
        assert_eq!(balance_lower_bound(&[3, 0], 3), Ok(2));
        assert_eq!(balance_lower_bound(&[0, 1, 4, 3], 3), Ok(0));
        assert!(matches!(
            balance_lower_bound(&[6], 3),
            Err(BracketError::UnknownToken { token: 6, kinds: 3 })
        ));
    }
}
