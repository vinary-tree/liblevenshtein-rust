use std::collections::VecDeque;
use std::hash::Hash;

use rustc_hash::FxHashMap;

use super::arena::TemporalStateId;
use crate::time_series::bounded::{IncompleteReason, ResourceKind};

/// Bounded FIFO transition cache. Eviction changes work only, never results.
pub(crate) struct BoundedTransitionCache<L> {
    max_entries: usize,
    cells: FxHashMap<(TemporalStateId, L), Option<TemporalStateId>>,
    order: VecDeque<(TemporalStateId, L)>,
}

impl<L> BoundedTransitionCache<L>
where
    L: Copy + Eq + Hash,
{
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            cells: FxHashMap::default(),
            order: VecDeque::new(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, source: TemporalStateId, label: L) -> Option<Option<TemporalStateId>> {
        self.cells.get(&(source, label)).copied()
    }

    pub(crate) fn insert(
        &mut self,
        source: TemporalStateId,
        label: L,
        target: Option<TemporalStateId>,
    ) -> Result<(), IncompleteReason> {
        if self.max_entries == 0 {
            return Ok(());
        }
        let key = (source, label);
        if let Some(existing) = self.cells.get_mut(&key) {
            *existing = target;
            return Ok(());
        }
        if self.cells.len() == self.max_entries {
            if let Some(evicted) = self.order.pop_front() {
                self.cells.remove(&evicted);
            }
        }
        self.cells
            .try_reserve(1)
            .map_err(|_| IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: self.max_entries,
                requested: self.cells.len().saturating_add(1),
            })?;
        self.order
            .try_reserve(1)
            .map_err(|_| IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: self.max_entries,
                requested: self.order.len().saturating_add(1),
            })?;
        self.cells.insert(key, target);
        self.order.push_back(key);
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.cells.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_never_exceeds_its_retained_entry_limit() {
        let mut cache = BoundedTransitionCache::new(2);
        for state in 0..100u32 {
            cache
                .insert(TemporalStateId(state), state, None)
                .expect("two-entry cache remains allocatable");
            assert!(cache.len() <= 2);
        }
    }
}
