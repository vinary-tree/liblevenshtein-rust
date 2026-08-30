use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;

use super::state::{CanonicalTemporalState, TemporalPosition};
use crate::time_series::bounded::{IncompleteReason, ResourceKind};

/// Compact identifier for one canonical query-local temporal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemporalStateId(pub(crate) u32);

impl TemporalStateId {
    #[inline]
    pub(crate) fn index(self) -> usize {
        usize::try_from(self.0).expect("u32 state id fits usize")
    }
}

/// Hard arena ceilings for one bounded dictionary-product query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemporalArenaLimits {
    /// Maximum canonical states interned by the query.
    pub max_states: usize,
    /// Maximum positions retained across all canonical states.
    pub max_positions: usize,
}

impl Default for TemporalArenaLimits {
    fn default() -> Self {
        Self {
            max_states: 1_000_000,
            max_positions: 8_000_000,
        }
    }
}

/// Exact collision-checked state interner.
pub(crate) struct TemporalStateArena<C> {
    limits: TemporalArenaLimits,
    position_count: usize,
    states: Vec<CanonicalTemporalState<C>>,
    fingerprints: FxHashMap<u64, SmallVec<[TemporalStateId; 1]>>,
}

impl<C> TemporalStateArena<C>
where
    C: Clone + Eq + Hash,
{
    pub(crate) fn new(limits: TemporalArenaLimits) -> Self {
        Self {
            limits,
            position_count: 0,
            states: Vec::new(),
            fingerprints: FxHashMap::default(),
        }
    }

    #[inline]
    fn fingerprint(context: &C, positions: &[TemporalPosition]) -> u64 {
        let mut hasher = FxHasher::default();
        context.hash(&mut hasher);
        positions.hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) fn intern(
        &mut self,
        context: C,
        positions: Vec<TemporalPosition>,
    ) -> Result<TemporalStateId, IncompleteReason> {
        debug_assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
        let fingerprint = Self::fingerprint(&context, &positions);
        if let Some(candidates) = self.fingerprints.get(&fingerprint) {
            for candidate in candidates {
                let state = &self.states[candidate.index()];
                if state.context == context && state.positions == positions {
                    return Ok(*candidate);
                }
            }
        }

        let requested_states =
            self.states
                .len()
                .checked_add(1)
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::QueueEntries,
                })?;
        if requested_states > self.limits.max_states {
            return Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::QueueEntries,
                limit: self.limits.max_states,
                requested: requested_states,
            });
        }
        let requested_positions = self.position_count.checked_add(positions.len()).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        if requested_positions > self.limits.max_positions {
            return Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: self.limits.max_positions,
                requested: requested_positions,
            });
        }
        let raw_id =
            u32::try_from(self.states.len()).map_err(|_| IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::QueueEntries,
            })?;

        self.states
            .try_reserve(1)
            .map_err(|_| IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: self.limits.max_positions,
                requested: requested_positions,
            })?;
        self.fingerprints
            .try_reserve(1)
            .map_err(|_| IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: self.limits.max_positions,
                requested: requested_positions,
            })?;

        let id = TemporalStateId(raw_id);
        self.states
            .push(CanonicalTemporalState { context, positions });
        self.fingerprints.entry(fingerprint).or_default().push(id);
        self.position_count = requested_positions;
        Ok(id)
    }

    #[inline]
    pub(crate) fn get(&self, id: TemporalStateId) -> &CanonicalTemporalState<C> {
        &self.states[id.index()]
    }

    #[inline]
    pub(crate) fn state_count(&self) -> usize {
        self.states.len()
    }

    #[inline]
    pub(crate) fn position_count(&self) -> usize {
        self.position_count
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::automaton::cost::CanonicalCost;

    fn position(index: u32, cost: f64) -> TemporalPosition {
        TemporalPosition {
            query_index: index,
            cost: CanonicalCost::new(cost).expect("test cost is canonical"),
        }
    }

    #[test]
    fn equal_states_share_one_compact_id() {
        let mut arena = TemporalStateArena::new(TemporalArenaLimits::default());
        let first = arena
            .intern((), vec![position(0, 0.0), position(2, 1.0)])
            .expect("small state fits arena");
        let second = arena
            .intern((), vec![position(0, -0.0), position(2, 1.0)])
            .expect("equal state reuses arena entry");
        assert_eq!(first, second);
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn arena_limit_fails_before_mutation() {
        let mut arena = TemporalStateArena::new(TemporalArenaLimits {
            max_states: 1,
            max_positions: 1,
        });
        arena
            .intern((), vec![position(0, 0.0)])
            .expect("first state fits exact limit");
        let error = arena
            .intern((), vec![position(1, 0.0)])
            .expect_err("second distinct state exceeds exact limit");
        assert!(matches!(
            error,
            IncompleteReason::BudgetExceeded {
                resource: ResourceKind::QueueEntries,
                ..
            }
        ));
        assert_eq!(arena.len(), 1);
    }
}
