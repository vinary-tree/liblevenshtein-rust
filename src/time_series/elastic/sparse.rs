//! Shared row scheduler for output-sensitive temporal transitions.

/// Charge one row before evaluating it.
///
/// The returned `requested` value is the first disallowed unit. On failure
/// `work` is unchanged, so callers can clear speculative output and pause
/// without ever executing more recurrence work than the configured ceiling.
#[inline]
pub(crate) fn charge_work(work: &mut usize, limit: usize) -> Result<(), usize> {
    let requested = work.checked_add(1).unwrap_or(usize::MAX);
    if requested > limit {
        Err(requested)
    } else {
        *work = requested;
        Ok(())
    }
}

/// Sorted, duplicate-free merge of every active predecessor row and its
/// immediate successor. These are exactly the rows with a horizontal or
/// diagonal seed before a kernel performs its vertical epsilon closure.
pub(crate) struct NeighborSeedRows<'a> {
    active: &'a [usize],
    same: usize,
    successor: usize,
    max_row: usize,
    last: Option<usize>,
}

impl<'a> NeighborSeedRows<'a> {
    #[inline]
    pub(crate) fn new(active: &'a [usize], max_row: usize) -> Self {
        debug_assert!(active.windows(2).all(|pair| pair[0] < pair[1]));
        Self {
            active,
            same: 0,
            successor: 0,
            max_row,
            last: None,
        }
    }

    fn same_row(&self) -> Option<usize> {
        self.active
            .get(self.same)
            .copied()
            .filter(|row| *row <= self.max_row)
    }

    fn successor_row(&mut self) -> Option<usize> {
        loop {
            let row = *self.active.get(self.successor)?;
            match row.checked_add(1) {
                Some(successor) if successor <= self.max_row => return Some(successor),
                Some(_) => self.successor += 1,
                None => return None,
            }
        }
    }
}

impl Iterator for NeighborSeedRows<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let same = self.same_row();
            let successor = self.successor_row();
            let row = match (same, successor) {
                (Some(left), Some(right)) if left <= right => {
                    self.same += 1;
                    left
                }
                (Some(_), Some(right)) => {
                    self.successor += 1;
                    right
                }
                (Some(left), None) => {
                    self.same += 1;
                    left
                }
                (None, Some(right)) => {
                    self.successor += 1;
                    right
                }
                (None, None) => return None,
            };
            if self.last == Some(row) {
                continue;
            }
            self.last = Some(row);
            return Some(row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_seed_rows_are_sorted_unique_and_bounded() {
        assert_eq!(
            NeighborSeedRows::new(&[0, 2, 3, 7], 7).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 7]
        );
        assert!(NeighborSeedRows::new(&[], 10).next().is_none());
    }

    #[test]
    fn work_is_charged_before_the_first_disallowed_row() {
        let mut work = 0;
        assert_eq!(charge_work(&mut work, 1), Ok(()));
        assert_eq!(work, 1);
        assert_eq!(charge_work(&mut work, 1), Err(2));
        assert_eq!(work, 1);
    }
}
