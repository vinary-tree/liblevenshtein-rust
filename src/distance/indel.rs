//! Insertion/deletion distance and its thresholded dynamic program.
//!
//! Substitution is absent: replacing one symbol therefore costs one deletion
//! plus one insertion. Equivalently, for strings $`x`$ and $`y`$,
//!
//! ```math
//! d_{\mathrm{indel}}(x,y)=|x|+|y|-2\,\operatorname{LCS}(x,y),
//! ```
//!
//! where `LCS` is longest-common-subsequence length.

use smallvec::SmallVec;

/// Compute insertion/deletion distance over Unicode scalar values.
pub fn indel_distance(source: &str, target: &str) -> usize {
    if source == target {
        return 0;
    }
    let source: SmallVec<[char; 32]> = source.chars().collect();
    let target: SmallVec<[char; 32]> = target.chars().collect();
    indel_distance_units(&source, &target)
}

fn indel_distance_units<T: Eq>(source: &[T], target: &[T]) -> usize {
    let (rows, columns) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    let mut previous: Vec<usize> = (0..=columns.len()).collect();
    let mut current = vec![0usize; columns.len() + 1];

    for (row, row_unit) in rows.iter().enumerate() {
        current[0] = row + 1;
        for (column, column_unit) in columns.iter().enumerate() {
            let j = column + 1;
            current[j] = previous[j]
                .saturating_add(1)
                .min(current[j - 1].saturating_add(1));
            if row_unit == column_unit {
                current[j] = current[j].min(previous[j - 1]);
            }
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[columns.len()]
}

/// Compute insertion/deletion distance up to `max_distance`.
///
/// Returns `None` exactly when the distance exceeds the threshold. Only cells
/// within `max_distance` diagonals of the main diagonal can belong to an
/// affordable path, so the dynamic program evaluates a band of width at most
/// `2 * max_distance + 1`.
pub fn indel_distance_bounded(source: &str, target: &str, max_distance: usize) -> Option<usize> {
    if source == target {
        return Some(0);
    }
    if max_distance == usize::MAX {
        return Some(indel_distance(source, target));
    }

    let source: SmallVec<[char; 32]> = source.chars().collect();
    let target: SmallVec<[char; 32]> = target.chars().collect();
    if source.is_empty() {
        return (target.len() <= max_distance).then_some(target.len());
    }
    if target.is_empty() {
        return (source.len() <= max_distance).then_some(source.len());
    }
    if source.len().abs_diff(target.len()) > max_distance {
        return None;
    }

    bounded_indel_units(&source, &target, max_distance)
}

fn bounded_indel_units<T: Eq>(source: &[T], target: &[T], maximum: usize) -> Option<usize> {
    let cap = maximum + 1;
    let mut previous = vec![cap; target.len() + 1];
    let mut current = vec![cap; target.len() + 1];
    for (column, cell) in previous
        .iter_mut()
        .take(target.len().min(maximum) + 1)
        .enumerate()
    {
        *cell = column;
    }

    for (row_index, row_unit) in source.iter().enumerate() {
        let row = row_index + 1;
        let start = row.saturating_sub(maximum).max(1);
        let end = target.len().min(row.saturating_add(maximum));
        if start > end {
            return None;
        }

        current.fill(cap);
        if row <= maximum {
            current[0] = row;
        }
        let mut row_minimum = current[0];
        for column in start..=end {
            let mut best = previous[column]
                .saturating_add(1)
                .min(cap)
                .min(current[column - 1].saturating_add(1).min(cap));
            if row_unit == &target[column - 1] {
                best = best.min(previous[column - 1]);
            }
            current[column] = best;
            row_minimum = row_minimum.min(best);
        }
        if row_minimum > maximum {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }

    (previous[target.len()] <= maximum).then_some(previous[target.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examples_and_boundaries() {
        assert_eq!(indel_distance("", ""), 0);
        assert_eq!(indel_distance("", "abc"), 3);
        assert_eq!(indel_distance("abc", ""), 3);
        assert_eq!(indel_distance("abc", "abc"), 0);
        assert_eq!(indel_distance("abc", "bca"), 2);
        assert_eq!(indel_distance("a", "b"), 2);
        assert_eq!(indel_distance("café", "cafe"), 2);
    }

    #[test]
    fn bounded_result_is_exact_at_the_boundary() {
        assert_eq!(indel_distance_bounded("abc", "bca", 1), None);
        assert_eq!(indel_distance_bounded("abc", "bca", 2), Some(2));
        assert_eq!(indel_distance_bounded("a", "b", 2), Some(2));
        assert_eq!(indel_distance_bounded("", "abc", 3), Some(3));
    }
}
