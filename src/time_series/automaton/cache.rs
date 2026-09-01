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
        let at_capacity = self.cells.len() == self.max_entries;
        if !at_capacity {
            self.cells
                .try_reserve(1)
                .map_err(|_| IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: self.cells.len().saturating_add(1),
                })?;
            self.order
                .try_reserve(1)
                .map_err(|_| IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: self.order.len().saturating_add(1),
                })?;
        }
        if at_capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.cells.remove(&evicted);
            }
        }
        self.cells.insert(key, target);
        self.order.push_back(key);
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.cells.len()
    }

    pub(crate) fn retained_bytes(&self) -> Option<usize> {
        let cell_bytes = self.cells.capacity().checked_mul(std::mem::size_of::<(
            (TemporalStateId, L),
            Option<TemporalStateId>,
        )>())?;
        let order_bytes = self
            .order
            .capacity()
            .checked_mul(std::mem::size_of::<(TemporalStateId, L)>())?;
        cell_bytes.checked_add(order_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_mutation_gate_cache_returns_exact_live_and_dead_successors() {
        let mut cache = BoundedTransitionCache::new(4);
        let source = TemporalStateId(7);
        let target = TemporalStateId(11);

        assert_eq!(cache.get(source, 3_u8), None);
        cache
            .insert(source, 3_u8, Some(target))
            .expect("bounded live transition fits");
        cache
            .insert(source, 4_u8, None)
            .expect("bounded dead transition fits");

        assert_eq!(cache.get(source, 3_u8), Some(Some(target)));
        assert_eq!(cache.get(source, 4_u8), Some(None));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn updating_an_observation_is_exact_without_growing_the_cache() {
        let mut cache = BoundedTransitionCache::new(1);
        let source = TemporalStateId(2);
        cache
            .insert(source, 9_u8, None)
            .expect("first observation fits");
        cache
            .insert(source, 9_u8, Some(TemporalStateId(5)))
            .expect("existing observation updates in place");

        assert_eq!(cache.get(source, 9_u8), Some(Some(TemporalStateId(5))));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn fifo_eviction_changes_retention_but_not_the_surviving_answers() {
        let mut cache = BoundedTransitionCache::new(2);
        cache
            .insert(TemporalStateId(0), 1_u8, Some(TemporalStateId(1)))
            .expect("first observation fits");
        cache
            .insert(TemporalStateId(0), 2_u8, Some(TemporalStateId(2)))
            .expect("second observation fits");
        cache
            .insert(TemporalStateId(0), 3_u8, Some(TemporalStateId(3)))
            .expect("third observation evicts atomically");

        assert_eq!(cache.get(TemporalStateId(0), 1_u8), None);
        assert_eq!(
            cache.get(TemporalStateId(0), 2_u8),
            Some(Some(TemporalStateId(2)))
        );
        assert_eq!(
            cache.get(TemporalStateId(0), 3_u8),
            Some(Some(TemporalStateId(3)))
        );
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn zero_capacity_cache_is_an_exact_no_retention_mode() {
        let mut cache = BoundedTransitionCache::new(0);
        cache
            .insert(TemporalStateId(0), 1_u8, Some(TemporalStateId(1)))
            .expect("disabled cache performs no allocation");
        assert_eq!(cache.get(TemporalStateId(0), 1_u8), None);
        assert_eq!(cache.len(), 0);
    }

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
