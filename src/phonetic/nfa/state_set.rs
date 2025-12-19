//! Efficient state set representation for NFA simulation.
//!
//! This module provides a `StateSet` type that uses a dense bitset for small NFAs
//! (<=256 states) and falls back to `FxHashSet` for larger NFAs.
//!
//! # Performance Characteristics
//!
//! For small NFAs (typical case in phonetic matching):
//! - O(1) insert, contains, remove
//! - O(4) iteration over set elements (scan 4×u64 words)
//! - 32 bytes fixed memory (4×u64)
//! - Cache-friendly: all operations touch contiguous memory
//!
//! For large NFAs:
//! - Falls back to `FxHashSet<StateId>` behavior
//!
//! # Design
//!
//! We use a 256-bit bitset represented as `[u64; 4]`. Each bit position corresponds
//! to a state ID. This provides:
//! - Membership testing: single bitwise AND
//! - Insertion: single bitwise OR
//! - Union: four bitwise ORs
//! - Iteration: scan each word, extract set bits

use super::types::StateId;
use rustc_hash::FxHashSet;

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

/// Maximum state ID supported by the bitset representation.
const BITSET_MAX_STATE: StateId = 255;

/// Number of u64 words in the bitset.
const WORD_COUNT: usize = 4;

/// Efficient state set for NFA simulation.
///
/// Uses a 256-bit bitset for NFAs with ≤256 states, falling back to `FxHashSet`
/// for larger automata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct StateSet {
    /// Bitset representation for small state sets (states 0-255).
    /// bits[0] covers states 0-63, bits[1] covers 64-127, etc.
    bits: [u64; WORD_COUNT],
    /// Count of states in the bitset portion.
    bit_count: u32,
    /// Fallback for states >= 256.
    #[cfg_attr(feature = "serialization", serde(skip))]
    overflow: Option<Box<FxHashSet<StateId>>>,
}

impl StateSet {
    /// Create a new empty state set.
    #[inline]
    pub fn new() -> Self {
        Self {
            bits: [0; WORD_COUNT],
            bit_count: 0,
            overflow: None,
        }
    }

    /// Create a state set with preallocated capacity.
    ///
    /// For states ≤255, this is a no-op since the bitset is fixed-size.
    /// For larger state spaces, preallocates the overflow hash set.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity > BITSET_MAX_STATE as usize + 1 {
            Self {
                bits: [0; WORD_COUNT],
                bit_count: 0,
                overflow: Some(Box::new(FxHashSet::with_capacity_and_hasher(
                    capacity - 256,
                    Default::default(),
                ))),
            }
        } else {
            Self::new()
        }
    }

    /// Insert a state into the set.
    ///
    /// Returns `true` if the state was newly inserted.
    #[inline]
    pub fn insert(&mut self, state: StateId) -> bool {
        if state <= BITSET_MAX_STATE {
            let word_idx = (state / 64) as usize;
            let bit_idx = state % 64;
            let mask = 1u64 << bit_idx;
            let was_set = (self.bits[word_idx] & mask) != 0;
            if !was_set {
                self.bits[word_idx] |= mask;
                self.bit_count += 1;
                true
            } else {
                false
            }
        } else {
            // Overflow to hash set
            let overflow = self.overflow.get_or_insert_with(|| {
                Box::new(FxHashSet::default())
            });
            overflow.insert(state)
        }
    }

    /// Check if a state is in the set.
    #[inline]
    pub fn contains(&self, state: &StateId) -> bool {
        let state = *state;
        if state <= BITSET_MAX_STATE {
            let word_idx = (state / 64) as usize;
            let bit_idx = state % 64;
            (self.bits[word_idx] & (1u64 << bit_idx)) != 0
        } else {
            self.overflow.as_ref().map_or(false, |o| o.contains(&state))
        }
    }

    /// Remove a state from the set.
    ///
    /// Returns `true` if the state was present.
    #[inline]
    pub fn remove(&mut self, state: &StateId) -> bool {
        let state = *state;
        if state <= BITSET_MAX_STATE {
            let word_idx = (state / 64) as usize;
            let bit_idx = state % 64;
            let mask = 1u64 << bit_idx;
            let was_set = (self.bits[word_idx] & mask) != 0;
            if was_set {
                self.bits[word_idx] &= !mask;
                self.bit_count -= 1;
                true
            } else {
                false
            }
        } else {
            self.overflow.as_mut().map_or(false, |o| o.remove(&state))
        }
    }

    /// Check if the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bit_count == 0 && self.overflow.as_ref().map_or(true, |o| o.is_empty())
    }

    /// Get the number of states in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.bit_count as usize + self.overflow.as_ref().map_or(0, |o| o.len())
    }

    /// Clear all states from the set.
    #[inline]
    pub fn clear(&mut self) {
        self.bits = [0; WORD_COUNT];
        self.bit_count = 0;
        if let Some(ref mut overflow) = self.overflow {
            overflow.clear();
        }
    }

    /// Extend this set with states from another set.
    #[inline]
    pub fn extend(&mut self, other: &StateSet) {
        // Optimized bitwise OR for the bitset portion
        for i in 0..WORD_COUNT {
            let old = self.bits[i];
            let new = old | other.bits[i];
            if old != new {
                self.bits[i] = new;
                // Update count: count newly set bits
                self.bit_count += (new ^ old).count_ones();
            }
        }

        // Handle overflow
        if let Some(ref other_overflow) = other.overflow {
            let overflow = self.overflow.get_or_insert_with(|| {
                Box::new(FxHashSet::default())
            });
            overflow.extend(other_overflow.iter().copied());
        }
    }

    /// Extend this set with states from an iterator.
    #[inline]
    pub fn extend_iter<I: IntoIterator<Item = StateId>>(&mut self, iter: I) {
        for state in iter {
            self.insert(state);
        }
    }

    /// Check if this set is a subset of another.
    #[inline]
    pub fn is_subset(&self, other: &StateSet) -> bool {
        // For bitset portion: (self & other) == self
        for i in 0..WORD_COUNT {
            if (self.bits[i] & other.bits[i]) != self.bits[i] {
                return false;
            }
        }

        // Check overflow
        match (&self.overflow, &other.overflow) {
            (None, _) => true,
            (Some(self_o), None) => self_o.is_empty(),
            (Some(self_o), Some(other_o)) => self_o.is_subset(other_o),
        }
    }

    /// Reserve capacity for additional states.
    ///
    /// This is a no-op for the bitset portion since it's fixed-size.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        // Only relevant for overflow hash set
        if let Some(ref mut overflow) = self.overflow {
            overflow.reserve(additional);
        }
    }

    /// Drain all states from the set, returning them as a Vec.
    pub fn drain(&mut self) -> Vec<StateId> {
        let mut result = Vec::with_capacity(self.len());

        // Drain bitset
        for word_idx in 0..WORD_COUNT {
            let mut word = self.bits[word_idx];
            while word != 0 {
                let bit_idx = word.trailing_zeros();
                let state = (word_idx as StateId) * 64 + bit_idx as StateId;
                result.push(state);
                word &= word - 1; // Clear lowest set bit
            }
            self.bits[word_idx] = 0;
        }
        self.bit_count = 0;

        // Drain overflow
        if let Some(ref mut overflow) = self.overflow {
            result.extend(overflow.drain());
        }

        result
    }

    /// Iterate over all states in the set.
    #[inline]
    pub fn iter(&self) -> StateSetIter<'_> {
        StateSetIter {
            bits: &self.bits,
            word_idx: 0,
            current_word: self.bits[0],
            overflow_iter: self.overflow.as_ref().map(|o| o.iter()),
        }
    }
}

impl Default for StateSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<StateId> for StateSet {
    fn from_iter<I: IntoIterator<Item = StateId>>(iter: I) -> Self {
        let mut set = Self::new();
        for state in iter {
            set.insert(state);
        }
        set
    }
}

impl<'a> IntoIterator for &'a StateSet {
    type Item = StateId;
    type IntoIter = StateSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over states in a StateSet.
pub struct StateSetIter<'a> {
    bits: &'a [u64; WORD_COUNT],
    word_idx: usize,
    current_word: u64,
    overflow_iter: Option<std::collections::hash_set::Iter<'a, StateId>>,
}

impl<'a> Iterator for StateSetIter<'a> {
    type Item = StateId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // First, drain the current word
        loop {
            if self.current_word != 0 {
                let bit_idx = self.current_word.trailing_zeros();
                let state = (self.word_idx as StateId) * 64 + bit_idx as StateId;
                self.current_word &= self.current_word - 1; // Clear lowest set bit
                return Some(state);
            }

            // Move to next word
            self.word_idx += 1;
            if self.word_idx >= WORD_COUNT {
                break;
            }
            self.current_word = self.bits[self.word_idx];
        }

        // Fall through to overflow iterator
        self.overflow_iter.as_mut().and_then(|it| it.next().copied())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // This is a lower bound; we don't track exact count during iteration
        let overflow_count = self.overflow_iter.as_ref().map_or(0, |it| it.size_hint().0);
        (overflow_count, None)
    }
}

// ============================================================================
// Conversion Traits
// ============================================================================

impl From<FxHashSet<StateId>> for StateSet {
    fn from(set: FxHashSet<StateId>) -> Self {
        let mut result = Self::new();
        for state in set {
            result.insert(state);
        }
        result
    }
}

impl From<StateSet> for FxHashSet<StateId> {
    fn from(set: StateSet) -> Self {
        set.iter().collect()
    }
}

impl From<&StateSet> for FxHashSet<StateId> {
    fn from(set: &StateSet) -> Self {
        set.iter().collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_set_basic() {
        let mut set = StateSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        assert!(set.insert(0));
        assert!(!set.insert(0)); // Already present
        assert!(set.contains(&0));
        assert!(!set.contains(&1));
        assert_eq!(set.len(), 1);

        assert!(set.insert(1));
        assert!(set.contains(&1));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_state_set_remove() {
        let mut set = StateSet::new();
        set.insert(5);
        set.insert(10);
        assert_eq!(set.len(), 2);

        assert!(set.remove(&5));
        assert!(!set.contains(&5));
        assert_eq!(set.len(), 1);

        assert!(!set.remove(&5)); // Already removed
    }

    #[test]
    fn test_state_set_word_boundaries() {
        let mut set = StateSet::new();

        // Test across word boundaries
        set.insert(63);  // Last bit of word 0
        set.insert(64);  // First bit of word 1
        set.insert(127); // Last bit of word 1
        set.insert(128); // First bit of word 2
        set.insert(255); // Last bit of word 3

        assert!(set.contains(&63));
        assert!(set.contains(&64));
        assert!(set.contains(&127));
        assert!(set.contains(&128));
        assert!(set.contains(&255));
        assert!(!set.contains(&0));
        assert!(!set.contains(&65));
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_state_set_overflow() {
        let mut set = StateSet::new();

        // Insert states beyond bitset capacity
        set.insert(255);
        set.insert(256);
        set.insert(1000);

        assert!(set.contains(&255));
        assert!(set.contains(&256));
        assert!(set.contains(&1000));
        assert!(!set.contains(&500));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_state_set_extend() {
        let mut set1 = StateSet::new();
        set1.insert(0);
        set1.insert(5);
        set1.insert(100);

        let mut set2 = StateSet::new();
        set2.insert(5);
        set2.insert(10);
        set2.insert(200);

        set1.extend(&set2);

        assert!(set1.contains(&0));
        assert!(set1.contains(&5));
        assert!(set1.contains(&10));
        assert!(set1.contains(&100));
        assert!(set1.contains(&200));
        assert_eq!(set1.len(), 5);
    }

    #[test]
    fn test_state_set_is_subset() {
        let mut set1 = StateSet::new();
        set1.insert(1);
        set1.insert(5);

        let mut set2 = StateSet::new();
        set2.insert(1);
        set2.insert(5);
        set2.insert(10);

        assert!(set1.is_subset(&set2));
        assert!(!set2.is_subset(&set1));

        // Empty set is subset of everything
        let empty = StateSet::new();
        assert!(empty.is_subset(&set1));
        assert!(empty.is_subset(&set2));
    }

    #[test]
    fn test_state_set_iter() {
        let mut set = StateSet::new();
        set.insert(0);
        set.insert(5);
        set.insert(63);
        set.insert(64);
        set.insert(200);

        let mut collected: Vec<StateId> = set.iter().collect();
        collected.sort();
        assert_eq!(collected, vec![0, 5, 63, 64, 200]);
    }

    #[test]
    fn test_state_set_iter_empty() {
        let set = StateSet::new();
        let collected: Vec<StateId> = set.iter().collect();
        assert!(collected.is_empty());
    }

    #[test]
    fn test_state_set_drain() {
        let mut set = StateSet::new();
        set.insert(1);
        set.insert(10);
        set.insert(100);

        let mut drained = set.drain();
        drained.sort();
        assert_eq!(drained, vec![1, 10, 100]);
        assert!(set.is_empty());
    }

    #[test]
    fn test_state_set_from_iter() {
        let states = vec![1, 5, 10, 100, 200];
        let set: StateSet = states.into_iter().collect();

        assert_eq!(set.len(), 5);
        assert!(set.contains(&1));
        assert!(set.contains(&100));
    }

    #[test]
    fn test_state_set_clear() {
        let mut set = StateSet::new();
        set.insert(1);
        set.insert(256); // Overflow

        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(!set.contains(&1));
        assert!(!set.contains(&256));
    }

    #[test]
    fn test_state_set_conversion() {
        let mut fx_set = FxHashSet::default();
        fx_set.insert(1);
        fx_set.insert(50);
        fx_set.insert(200);

        let state_set = StateSet::from(fx_set.clone());
        assert_eq!(state_set.len(), 3);
        assert!(state_set.contains(&1));
        assert!(state_set.contains(&50));
        assert!(state_set.contains(&200));

        let back: FxHashSet<StateId> = state_set.into();
        assert_eq!(back, fx_set);
    }
}
