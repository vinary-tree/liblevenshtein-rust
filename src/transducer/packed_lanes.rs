//! Shared one-word layout for bounded edit-cost frontiers.
//!
//! Each lane stores query positions reached at one exact edit cost.  Keeping
//! the lanes exact avoids copying every cheaper state into every more
//! expensive lane.  Consumers may still obtain the cumulative union used by
//! positional antichain accounting through [`PackedEditLaneLayout::lane_union`].

const MAX_PACKED_DISTANCE: usize = 3;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedEditLaneLayout {
    query_length: usize,
    max_distance: usize,
    lane_width: usize,
    lane_mask: u64,
    end_bit: u64,
    lane_starts: u64,
    active_mask: u64,
    nonterminal_bits: u64,
    terminal_bits: u64,
    #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
    top_lane_mask: u64,
    deletion_source_masks: [u64; MAX_PACKED_DISTANCE],
}

impl PackedEditLaneLayout {
    #[cfg(test)]
    pub(crate) const MAX_DISTANCE: usize = MAX_PACKED_DISTANCE;

    pub(crate) fn eligible(query_length: usize, max_distance: usize) -> bool {
        max_distance <= MAX_PACKED_DISTANCE
            && query_length
                .checked_add(1)
                .and_then(|width| width.checked_mul(max_distance.saturating_add(1)))
                .is_some_and(|bits| bits <= u64::BITS as usize)
    }

    pub(crate) fn new(query_length: usize, max_distance: usize, prefix_mode: bool) -> Option<Self> {
        if !Self::eligible(query_length, max_distance) {
            return None;
        }
        let lane_width = query_length + 1;
        let lane_mask = low_bits(lane_width);
        let end_bit = 1u64 << query_length;
        let mut lane_starts = 0u64;
        for edit in 0..=max_distance {
            lane_starts |= 1u64 << (edit * lane_width);
        }
        let active_mask = low_bits(lane_width * (max_distance + 1));
        let nonterminal_bits = lane_starts * (lane_mask >> 1);
        let terminal_bits = if prefix_mode {
            lane_starts * end_bit
        } else {
            0
        };
        #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
        let top_lane_mask = lane_mask << (max_distance * lane_width);

        let mut deletion_source_masks = [0u64; MAX_PACKED_DISTANCE];
        for deletion_count in 1..=max_distance {
            if query_length < deletion_count {
                continue;
            }
            let source_positions = low_bits(query_length - deletion_count + 1);
            let mut source_lane_starts = 0u64;
            for edit in 0..=max_distance - deletion_count {
                source_lane_starts |= 1u64 << (edit * lane_width);
            }
            deletion_source_masks[deletion_count - 1] = source_lane_starts * source_positions;
        }

        Some(Self {
            query_length,
            max_distance,
            lane_width,
            lane_mask,
            end_bit,
            lane_starts,
            active_mask,
            nonterminal_bits,
            terminal_bits,
            #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
            top_lane_mask,
            deletion_source_masks,
        })
    }

    #[inline(always)]
    pub(crate) fn query_length(self) -> usize {
        self.query_length
    }

    #[inline(always)]
    pub(crate) fn max_distance(self) -> usize {
        self.max_distance
    }

    #[inline(always)]
    pub(crate) fn lane_width(self) -> usize {
        self.lane_width
    }

    #[inline(always)]
    pub(crate) fn lane_mask(self) -> u64 {
        self.lane_mask
    }

    #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
    #[inline(always)]
    pub(crate) fn end_bit(self) -> u64 {
        self.end_bit
    }

    #[inline(always)]
    pub(crate) fn lane_starts(self) -> u64 {
        self.lane_starts
    }

    #[inline(always)]
    pub(crate) fn active_mask(self) -> u64 {
        self.active_mask
    }

    /// Bits in exact-cost lanes that may still spend one edit.
    #[inline(always)]
    pub(crate) fn error_source_bits(self) -> u64 {
        low_bits(self.max_distance * self.lane_width)
    }

    /// Query positions from which a two-unit operation can consume both units.
    #[inline(always)]
    pub(crate) fn two_unit_source_bits(self) -> u64 {
        if self.query_length < 2 {
            return 0;
        }
        self.lane_starts * low_bits(self.query_length - 1)
    }

    #[inline(always)]
    pub(crate) fn nonterminal_bits(self) -> u64 {
        self.nonterminal_bits
    }

    #[inline(always)]
    pub(crate) fn terminal_bits(self) -> u64 {
        self.terminal_bits
    }

    #[cfg(any(test, feature = "perf-instrumentation", feature = "resource-profiling"))]
    #[inline(always)]
    pub(crate) fn top_lane_mask(self) -> u64 {
        self.top_lane_mask
    }

    pub(crate) fn exact_seed(self) -> u64 {
        let mut packed = 0u64;
        for edit in 0..=self.max_distance.min(self.query_length) {
            packed |= 1u64 << (edit * self.lane_width + edit);
        }
        packed
    }

    #[cfg(any(feature = "perf-instrumentation", feature = "resource-profiling"))]
    pub(crate) fn cumulative_seed(self) -> u64 {
        let mut packed = 0u64;
        for edit in 0..=self.max_distance {
            let reachable = low_bits(edit.min(self.query_length) + 1);
            packed |= reachable << (edit * self.lane_width);
        }
        packed
    }

    /// Close exact-cost lanes under one through three query deletions.
    ///
    /// Every shifted term reads from the same pre-closure frontier.  A
    /// `k`-deletion term therefore lands exactly `k` lanes and `k` query
    /// positions above its source without the serial cumulative-lane chain.
    #[inline(always)]
    pub(crate) fn close_exact_deletions(self, initial: u64) -> u64 {
        let stride = self.lane_width + 1;
        let target = match self.max_distance {
            0 => initial,
            1 => initial | ((initial & self.deletion_source_masks[0]) << stride),
            2 => {
                initial
                    | ((initial & self.deletion_source_masks[0]) << stride)
                    | ((initial & self.deletion_source_masks[1]) << (2 * stride))
            }
            3 => {
                initial
                    | ((initial & self.deletion_source_masks[0]) << stride)
                    | ((initial & self.deletion_source_masks[1]) << (2 * stride))
                    | ((initial & self.deletion_source_masks[2]) << (3 * stride))
            }
            _ => unreachable!("packed edit distance exceeds three"),
        };
        target & self.active_mask
    }

    #[inline(always)]
    pub(crate) fn lane(self, packed: u64, edit: usize) -> u64 {
        if self.lane_width == u64::BITS as usize {
            debug_assert_eq!(edit, 0);
            packed
        } else {
            (packed >> (edit * self.lane_width)) & self.lane_mask
        }
    }

    /// Visit every set `(edit cost, query position)` pair in an exact-cost
    /// packed frontier.
    ///
    /// This is deliberately the single decoding seam shared by Standard,
    /// OSA, and MergeSplit. Normal query traversal never decodes packed
    /// frontiers; the operation exists for cold compatibility boundaries such
    /// as converting a partially consumed ordered iterator to its legacy
    /// positional prefix representation.
    #[inline]
    pub(crate) fn for_each_set_position(self, packed: u64, mut visit: impl FnMut(usize, usize)) {
        for edit in 0..=self.max_distance {
            let mut positions = self.lane(packed, edit);
            while positions != 0 {
                let position = positions.trailing_zeros() as usize;
                visit(edit, position);
                positions &= positions - 1;
            }
        }
    }

    #[inline(always)]
    pub(crate) fn lane_union(self, packed: u64) -> u64 {
        let mut union = 0u64;
        for edit in 0..=self.max_distance {
            union |= self.lane(packed, edit);
        }
        union
    }

    pub(crate) fn complete_distance(self, frontier: u64) -> Option<usize> {
        (0..=self.max_distance).find(|&edit| self.lane(frontier, edit) & self.end_bit != 0)
    }

    pub(crate) fn min_distance(self, frontier: u64) -> Option<usize> {
        (0..=self.max_distance).find(|&edit| self.lane(frontier, edit) != 0)
    }

    pub(crate) fn max_consumed(self, frontier: u64) -> usize {
        let positions = self.lane_union(frontier);
        if positions == 0 {
            0
        } else {
            (u64::BITS - 1 - positions.leading_zeros()) as usize
        }
    }

    #[cfg(feature = "perf-instrumentation")]
    pub(crate) fn active_len(self, frontier: u64) -> usize {
        self.lane_union(frontier).count_ones() as usize
    }
}

#[inline]
fn low_bits(count: usize) -> u64 {
    if count == u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << count) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_seed_contains_only_the_empty_term_deletion_diagonal() {
        let layout = PackedEditLaneLayout::new(5, 3, false).unwrap();
        assert_eq!(layout.lane(layout.exact_seed(), 0), 0b000001);
        assert_eq!(layout.lane(layout.exact_seed(), 1), 0b000010);
        assert_eq!(layout.lane(layout.exact_seed(), 2), 0b000100);
        assert_eq!(layout.lane(layout.exact_seed(), 3), 0b001000);
    }

    #[test]
    fn deletion_closure_never_crosses_query_or_lane_boundaries() {
        let layout = PackedEditLaneLayout::new(3, 3, false).unwrap();
        let source = 1u64;
        let closed = layout.close_exact_deletions(source);
        assert_eq!(layout.lane(closed, 0), 0b0001);
        assert_eq!(layout.lane(closed, 1), 0b0010);
        assert_eq!(layout.lane(closed, 2), 0b0100);
        assert_eq!(layout.lane(closed, 3), 0b1000);

        let terminal = layout.end_bit();
        assert_eq!(layout.close_exact_deletions(terminal), terminal);
    }
}
