//! Lazy, bounded automata for online temporal-distance transitions.
//!
//! The string transducer constructs only the Levenshtein states reached while
//! intersecting the query automaton with a dictionary.  This module provides
//! the corresponding temporal spine: canonical weighted position antichains,
//! compact query-local state identifiers, an observation-bounded transition
//! cache, and two-generation online execution.
//!
//! A temporal automaton has a **fixed finite query**. It may consume a target
//! stream of unknown length one unit at a time without retaining the target
//! prefix. Exact bilateral distance between two unbounded histories is not
//! claimed; callers needing two evolving series must use a bounded rolling
//! query window.
//!
//! Kernel-specific dominance is accepted only when it is witnessed by a
//! zero-target-consumption path. Approximate floating-point equality is never
//! state identity and never authorizes pruning.

mod arena;
mod cache;
mod column;
mod cost;
mod erp;
mod state;
mod timestamped_twed;

pub use arena::{TemporalArenaLimits, TemporalStateId};
pub use column::{ElasticOnlineAutomaton, ElasticOnlineObservation};
pub(crate) use erp::ErpFrontierMachine;
pub use erp::{ErpOnlineAutomaton, ErpOnlineObservation, OnlineAutomatonLimits};
pub use timestamped_twed::{TimestampedTwedOnlineAutomaton, TimestampedTwedOnlineObservation};

use super::bounded::{IncompleteReason, ResourceUsage, TemporalValidationError};
use thiserror::Error;

/// Construction failure for a bounded temporal automaton.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TemporalAutomatonError {
    /// The query, cutoff, or kernel configuration is outside its typed domain.
    #[error(transparent)]
    Validation(#[from] TemporalValidationError),
    /// The requested fixed machine state does not fit its configured ceiling.
    #[error("temporal automaton resource construction failed: {0:?}")]
    Resource(IncompleteReason),
}

/// Result of consuming one online target unit.
///
/// `Incomplete` leaves the automaton at the preceding target prefix, so the
/// caller may retry the same unit with a differently provisioned machine. It
/// is never an observation about the distance or about absence of a match.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub enum OnlineStepOutcome<T> {
    /// The target unit was consumed and the current prefix observation is exact
    /// within the configured cutoff.
    Advanced {
        /// Observation after the committed transition.
        value: T,
        /// Work and peak storage charged by this one transition.
        usage: ResourceUsage,
    },
    /// The target unit was not consumed.
    Incomplete {
        /// Fail-closed stop reason.
        reason: IncompleteReason,
        /// Work and peak storage observed before the transactional stop.
        usage: ResourceUsage,
    },
}

impl<T> OnlineStepOutcome<T> {
    /// Whether the supplied target unit was committed.
    #[inline]
    pub fn advanced(&self) -> bool {
        matches!(self, Self::Advanced { .. })
    }

    /// Per-transition resource accounting.
    #[inline]
    pub fn usage(&self) -> ResourceUsage {
        match self {
            Self::Advanced { usage, .. } | Self::Incomplete { usage, .. } => *usage,
        }
    }
}
