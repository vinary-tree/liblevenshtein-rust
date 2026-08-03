//! Ergonomic distance selection over the ordered query iterator.
//!
//! Match modes do not add a pruning rule. A dictionary prefix whose current
//! distance is below a requested minimum may still extend to a term inside the
//! requested range, so the minimum is applied only to completed candidates.

use super::OrderedCandidate;
use std::error::Error;
use std::fmt::{self, Display};
use std::iter::FusedIterator;

/// Completed-candidate distance selection for [`Transducer::query_mode`].
///
/// [`Transducer::query_mode`]: super::Transducer::query_mode
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MatchMode {
    /// Yield every candidate at distance at most the supplied maximum.
    Within(usize),
    /// Yield candidates at exactly this distance.
    Exact(usize),
    /// Yield candidates in the inclusive distance interval.
    Range {
        /// Inclusive minimum distance.
        min_distance: usize,
        /// Inclusive maximum distance.
        max_distance: usize,
    },
}

impl MatchMode {
    pub(crate) fn bounds(self) -> Result<(usize, usize), MatchModeError> {
        match self {
            Self::Within(maximum) => Ok((0, maximum)),
            Self::Exact(distance) => Ok((distance, distance)),
            Self::Range {
                min_distance,
                max_distance,
            } if min_distance <= max_distance => Ok((min_distance, max_distance)),
            Self::Range {
                min_distance,
                max_distance,
            } => Err(MatchModeError::InvalidRange {
                min_distance,
                max_distance,
            }),
        }
    }
}

/// Invalid [`MatchMode`] configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchModeError {
    /// The inclusive lower bound exceeded the inclusive upper bound.
    InvalidRange {
        /// Requested inclusive lower bound.
        min_distance: usize,
        /// Requested inclusive upper bound.
        max_distance: usize,
    },
}

impl Display for MatchModeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange {
                min_distance,
                max_distance,
            } => write!(
                formatter,
                "match-mode minimum distance {min_distance} exceeds maximum distance {max_distance}"
            ),
        }
    }
}

impl Error for MatchModeError {}

/// Ordered candidates filtered by a validated [`MatchMode`].
///
/// The wrapped iterator already stops at the mode's maximum. This adapter
/// skips completed candidates below the minimum; it does not prune their
/// prefixes or change the underlying automaton traversal.
pub struct MatchModeQueryIterator<I> {
    inner: I,
    min_distance: usize,
    max_distance: usize,
    exhausted: bool,
}

impl<I> MatchModeQueryIterator<I> {
    pub(crate) fn try_new(inner: I, mode: MatchMode) -> Result<Self, MatchModeError> {
        let (min_distance, max_distance) = mode.bounds()?;
        Ok(Self {
            inner,
            min_distance,
            max_distance,
            exhausted: false,
        })
    }

    /// Consume the adapter and recover the underlying ordered iterator.
    pub fn into_inner(self) -> I {
        self.inner
    }
}

impl<I> Iterator for MatchModeQueryIterator<I>
where
    I: Iterator<Item = OrderedCandidate>,
{
    type Item = OrderedCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        loop {
            let candidate = self.inner.next()?;
            if candidate.distance > self.max_distance {
                self.exhausted = true;
                return None;
            }
            if candidate.distance >= self.min_distance {
                return Some(candidate);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.inner.size_hint();
        (0, upper)
    }
}

impl<I> FusedIterator for MatchModeQueryIterator<I> where I: FusedIterator<Item = OrderedCandidate> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_filter_only_completed_ordered_candidates() {
        let candidates = || {
            vec![
                OrderedCandidate {
                    distance: 0,
                    term: "cat".into(),
                },
                OrderedCandidate {
                    distance: 1,
                    term: "bat".into(),
                },
                OrderedCandidate {
                    distance: 1,
                    term: "cot".into(),
                },
                OrderedCandidate {
                    distance: 2,
                    term: "coat".into(),
                },
            ]
            .into_iter()
        };

        let exact: Vec<_> = MatchModeQueryIterator::try_new(candidates(), MatchMode::Exact(1))
            .unwrap()
            .collect();
        assert_eq!(
            exact
                .iter()
                .map(|item| item.term.as_str())
                .collect::<Vec<_>>(),
            ["bat", "cot"]
        );

        let range: Vec<_> = MatchModeQueryIterator::try_new(
            candidates(),
            MatchMode::Range {
                min_distance: 1,
                max_distance: 2,
            },
        )
        .unwrap()
        .collect();
        assert_eq!(range.len(), 3);
    }

    #[test]
    fn inverted_range_is_rejected() {
        assert_eq!(
            MatchModeQueryIterator::try_new(
                std::iter::empty::<OrderedCandidate>(),
                MatchMode::Range {
                    min_distance: 2,
                    max_distance: 1,
                },
            )
            .err(),
            Some(MatchModeError::InvalidRange {
                min_distance: 2,
                max_distance: 1,
            })
        );
    }
}
