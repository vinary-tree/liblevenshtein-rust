//! Generic unit-sequence edit-distance kernels.
//!
//! These kernels define the domain-independent semantics shared by Rust and
//! the native language-binding ABI. A unit can be a Unicode scalar (`char`),
//! an arbitrary byte (`u8`), or an application token (`u64`); equality is the
//! only operation required by the standard, optimal-string-alignment, and
//! merge/split families. Unrestricted Damerau--Levenshtein additionally needs
//! hashing to track the last row in which each unit occurred.

use std::collections::HashMap;
use std::hash::Hash;

#[inline]
fn common_affix_cores<'a, U: Eq>(source: &'a [U], target: &'a [U]) -> (&'a [U], &'a [U]) {
    let prefix = source
        .iter()
        .zip(target)
        .take_while(|(left, right)| left == right)
        .count();
    let remaining = source.len().min(target.len()).saturating_sub(prefix);
    let suffix = source[prefix..]
        .iter()
        .rev()
        .zip(target[prefix..].iter().rev())
        .take(remaining)
        .take_while(|(left, right)| left == right)
        .count();

    (
        &source[prefix..source.len() - suffix],
        &target[prefix..target.len() - suffix],
    )
}

/// Compute standard Levenshtein distance over arbitrary equality-comparable
/// units.
///
/// The implementation stores two rows and chooses the shorter input as the
/// column dimension, so auxiliary memory is linear in the shorter sequence.
pub fn standard_distance_units<U: Eq>(source: &[U], target: &[U]) -> usize {
    let (rows, columns) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    if columns.is_empty() {
        return rows.len();
    }

    let mut previous: Vec<usize> = (0..=columns.len()).collect();
    let mut current = vec![0; columns.len() + 1];
    for (row_index, row_unit) in rows.iter().enumerate() {
        current[0] = row_index + 1;
        for (column_index, column_unit) in columns.iter().enumerate() {
            let column = column_index + 1;
            let substitution_cost = usize::from(row_unit != column_unit);
            current[column] = previous[column]
                .saturating_add(1)
                .min(current[column - 1].saturating_add(1))
                .min(previous[column - 1].saturating_add(substitution_cost));
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[columns.len()]
}

/// Compute standard Levenshtein distance within an inclusive bound.
///
/// Only the diagonal band capable of reaching a result no greater than
/// `maximum` is evaluated. `None` means the exact distance is above the bound.
pub fn standard_distance_units_bounded<U: Eq>(
    source: &[U],
    target: &[U],
    maximum: usize,
) -> Option<usize> {
    if source == target {
        return Some(0);
    }
    let (source, target) = common_affix_cores(source, target);
    if source.len().abs_diff(target.len()) > maximum {
        return None;
    }
    if source.is_empty() {
        return (target.len() <= maximum).then_some(target.len());
    }
    if target.is_empty() {
        return (source.len() <= maximum).then_some(source.len());
    }
    if maximum == 0 {
        return None;
    }
    if maximum == usize::MAX {
        return Some(standard_distance_units(source, target));
    }

    let (rows, columns) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    let cap = maximum + 1;
    let mut previous = vec![cap; columns.len() + 1];
    let mut current = vec![cap; columns.len() + 1];
    for (column, cell) in previous
        .iter_mut()
        .take(columns.len().min(maximum) + 1)
        .enumerate()
    {
        *cell = column;
    }

    for row in 1..=rows.len() {
        let start = row.saturating_sub(maximum).max(1);
        let end = columns.len().min(row.saturating_add(maximum));
        if start > end {
            return None;
        }
        current.fill(cap);
        current[0] = if row <= maximum { row } else { cap };
        let mut row_minimum = current[0];
        for column in start..=end {
            let substitution_cost = usize::from(rows[row - 1] != columns[column - 1]);
            current[column] = previous[column]
                .saturating_add(1)
                .min(cap)
                .min(current[column - 1].saturating_add(1).min(cap))
                .min(
                    previous[column - 1]
                        .saturating_add(substitution_cost)
                        .min(cap),
                );
            row_minimum = row_minimum.min(current[column]);
        }
        if row_minimum > maximum {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[columns.len()] <= maximum).then_some(previous[columns.len()])
}

/// Compute optimal-string-alignment (restricted Damerau) distance over units.
pub fn transposition_distance_units<U: Eq>(source: &[U], target: &[U]) -> usize {
    let (rows, columns) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    if columns.is_empty() {
        return rows.len();
    }

    let mut two_ago = vec![0usize; columns.len() + 1];
    let mut previous: Vec<usize> = (0..=columns.len()).collect();
    let mut current = vec![0; columns.len() + 1];
    for row in 1..=rows.len() {
        current[0] = row;
        for column in 1..=columns.len() {
            let substitution_cost = usize::from(rows[row - 1] != columns[column - 1]);
            let mut distance = previous[column]
                .saturating_add(1)
                .min(current[column - 1].saturating_add(1))
                .min(previous[column - 1].saturating_add(substitution_cost));
            if row > 1
                && column > 1
                && rows[row - 1] == columns[column - 2]
                && rows[row - 2] == columns[column - 1]
            {
                distance = distance.min(two_ago[column - 2].saturating_add(1));
            }
            current[column] = distance;
        }
        std::mem::swap(&mut two_ago, &mut previous);
        std::mem::swap(&mut previous, &mut current);
    }
    previous[columns.len()]
}

/// Compute optimal-string-alignment distance within an inclusive bound.
pub fn transposition_distance_units_bounded<U: Eq>(
    source: &[U],
    target: &[U],
    maximum: usize,
) -> Option<usize> {
    if source == target {
        return Some(0);
    }
    let (source, target) = common_affix_cores(source, target);
    if source.len().abs_diff(target.len()) > maximum {
        return None;
    }
    if source.is_empty() {
        return (target.len() <= maximum).then_some(target.len());
    }
    if target.is_empty() {
        return (source.len() <= maximum).then_some(source.len());
    }
    if maximum == 0 {
        return None;
    }
    if maximum == usize::MAX {
        return Some(transposition_distance_units(source, target));
    }

    let (rows, columns) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    let cap = maximum + 1;
    let mut two_ago = vec![cap; columns.len() + 1];
    let mut previous = vec![cap; columns.len() + 1];
    let mut current = vec![cap; columns.len() + 1];
    for (column, cell) in previous
        .iter_mut()
        .take(columns.len().min(maximum) + 1)
        .enumerate()
    {
        *cell = column;
    }

    for row in 1..=rows.len() {
        let start = row.saturating_sub(maximum).max(1);
        let end = columns.len().min(row.saturating_add(maximum));
        if start > end {
            return None;
        }
        current.fill(cap);
        current[0] = if row <= maximum { row } else { cap };
        let mut row_minimum = current[0];
        for column in start..=end {
            let substitution_cost = usize::from(rows[row - 1] != columns[column - 1]);
            let mut distance = previous[column]
                .saturating_add(1)
                .min(cap)
                .min(current[column - 1].saturating_add(1).min(cap))
                .min(
                    previous[column - 1]
                        .saturating_add(substitution_cost)
                        .min(cap),
                );
            if row > 1
                && column > 1
                && rows[row - 1] == columns[column - 2]
                && rows[row - 2] == columns[column - 1]
            {
                distance = distance.min(two_ago[column - 2].saturating_add(1).min(cap));
            }
            current[column] = distance;
            row_minimum = row_minimum.min(distance);
        }
        if row_minimum > maximum {
            return None;
        }
        std::mem::swap(&mut two_ago, &mut previous);
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[columns.len()] <= maximum).then_some(previous[columns.len()])
}

/// Compute unrestricted Damerau--Levenshtein distance over hashable units.
pub fn damerau_levenshtein_distance_units<U: Copy + Eq + Hash>(
    source: &[U],
    target: &[U],
) -> usize {
    if source.is_empty() {
        return target.len();
    }
    if target.is_empty() {
        return source.len();
    }

    let sentinel = source.len().saturating_add(target.len());
    let columns = target.len() + 2;
    let mut matrix = vec![0usize; (source.len() + 2).saturating_mul(columns)];
    let index = |row: usize, column: usize| row * columns + column;
    matrix[index(0, 0)] = sentinel;
    for row in 0..=source.len() {
        matrix[index(row + 1, 0)] = sentinel;
        matrix[index(row + 1, 1)] = row;
    }
    for column in 0..=target.len() {
        matrix[index(0, column + 1)] = sentinel;
        matrix[index(1, column + 1)] = column;
    }

    let mut last_row = HashMap::<U, usize>::new();
    for row in 1..=source.len() {
        let mut last_match_column = 0usize;
        for column in 1..=target.len() {
            let transposition_row = last_row.get(&target[column - 1]).copied().unwrap_or(0);
            let transposition_column = last_match_column;
            let substitution_cost = usize::from(source[row - 1] != target[column - 1]);
            if substitution_cost == 0 {
                last_match_column = column;
            }
            let substitution = matrix[index(row, column)].saturating_add(substitution_cost);
            let insertion = matrix[index(row + 1, column)].saturating_add(1);
            let deletion = matrix[index(row, column + 1)].saturating_add(1);
            let transposition = matrix[index(transposition_row, transposition_column)]
                .saturating_add(row - transposition_row - 1)
                .saturating_add(1)
                .saturating_add(column - transposition_column - 1);
            matrix[index(row + 1, column + 1)] =
                substitution.min(insertion).min(deletion).min(transposition);
        }
        last_row.insert(source[row - 1], row);
    }
    matrix[index(source.len() + 1, target.len() + 1)]
}

/// Compute unrestricted Damerau--Levenshtein distance within an inclusive
/// bound.
///
/// The exact recurrence is retained because unrestricted transpositions can
/// reference non-adjacent historical rows. The length lower bound still avoids
/// allocation whenever the answer cannot fit.
pub fn damerau_levenshtein_distance_units_bounded<U: Copy + Eq + Hash>(
    source: &[U],
    target: &[U],
    maximum: usize,
) -> Option<usize> {
    if source.len().abs_diff(target.len()) > maximum {
        return None;
    }
    let distance = damerau_levenshtein_distance_units(source, target);
    (distance <= maximum).then_some(distance)
}

/// Compute Levenshtein distance extended with generic one-to-two split and
/// two-to-one merge operations, each with unit cost.
pub fn merge_and_split_distance_units<U: Eq>(source: &[U], target: &[U]) -> usize {
    let (source, target) = common_affix_cores(source, target);
    let (rows, columns) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    if columns.is_empty() {
        return rows.len();
    }

    let mut two_ago = vec![0usize; columns.len() + 1];
    let mut previous: Vec<usize> = (0..=columns.len()).collect();
    let mut current = vec![0; columns.len() + 1];
    for row in 1..=rows.len() {
        current[0] = row;
        for column in 1..=columns.len() {
            let substitution_cost = usize::from(rows[row - 1] != columns[column - 1]);
            let mut distance = previous[column]
                .saturating_add(1)
                .min(current[column - 1].saturating_add(1))
                .min(previous[column - 1].saturating_add(substitution_cost));
            if row >= 2 {
                distance = distance.min(two_ago[column - 1].saturating_add(1));
            }
            if column >= 2 {
                distance = distance.min(previous[column - 2].saturating_add(1));
            }
            current[column] = distance;
        }
        std::mem::swap(&mut two_ago, &mut previous);
        std::mem::swap(&mut previous, &mut current);
    }
    previous[columns.len()]
}

/// Compute merge/split distance within an inclusive bound.
///
/// The band remains exact because each unit-cost operation changes the prefix
/// length difference by at most one.
pub fn merge_and_split_distance_units_bounded<U: Eq>(
    source: &[U],
    target: &[U],
    maximum: usize,
) -> Option<usize> {
    if source == target {
        return Some(0);
    }
    let (source, target) = common_affix_cores(source, target);
    if source.len().abs_diff(target.len()) > maximum {
        return None;
    }
    if source.is_empty() {
        return (target.len() <= maximum).then_some(target.len());
    }
    if target.is_empty() {
        return (source.len() <= maximum).then_some(source.len());
    }
    if maximum == 0 {
        return None;
    }
    if maximum == usize::MAX {
        return Some(merge_and_split_distance_units(source, target));
    }

    let (rows, columns) = if source.len() >= target.len() {
        (source, target)
    } else {
        (target, source)
    };
    let cap = maximum + 1;
    let mut two_ago = vec![cap; columns.len() + 1];
    let mut previous = vec![cap; columns.len() + 1];
    let mut current = vec![cap; columns.len() + 1];
    for (column, cell) in previous
        .iter_mut()
        .take(columns.len().min(maximum) + 1)
        .enumerate()
    {
        *cell = column;
    }
    for row in 1..=rows.len() {
        let start = row.saturating_sub(maximum).max(1);
        let end = columns.len().min(row.saturating_add(maximum));
        if start > end {
            return None;
        }
        current.fill(cap);
        current[0] = if row <= maximum { row } else { cap };
        for column in start..=end {
            let substitution_cost = usize::from(rows[row - 1] != columns[column - 1]);
            let mut distance = previous[column]
                .saturating_add(1)
                .min(cap)
                .min(current[column - 1].saturating_add(1).min(cap))
                .min(
                    previous[column - 1]
                        .saturating_add(substitution_cost)
                        .min(cap),
                );
            if row >= 2 {
                distance = distance.min(two_ago[column - 1].saturating_add(1).min(cap));
            }
            if column >= 2 {
                distance = distance.min(previous[column - 2].saturating_add(1).min(cap));
            }
            current[column] = distance;
        }
        std::mem::swap(&mut two_ago, &mut previous);
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[columns.len()] <= maximum).then_some(previous[columns.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_families_are_domain_generic() {
        let bytes = (b"CA".as_slice(), b"ABC".as_slice());
        let tokens = ([10_u64, 20], [20_u64, 10]);
        assert_eq!(standard_distance_units(bytes.0, bytes.1), 3);
        assert_eq!(transposition_distance_units(&tokens.0, &tokens.1), 1);
        assert_eq!(damerau_levenshtein_distance_units(bytes.0, bytes.1), 2);
        assert_eq!(merge_and_split_distance_units(b"m", b"rn"), 1);
        assert_eq!(merge_and_split_distance_units(&[1_u64, 2], &[9_u64]), 1);
    }

    #[test]
    fn bounded_results_equal_exact_results_or_none() {
        let corpus: [&[u8]; 8] = [b"", b"a", b"ab", b"ba", b"abc", b"CA", b"ABC", b"kitten"];
        for source in corpus {
            for target in corpus {
                for maximum in 0..=4 {
                    let expected = |distance| (distance <= maximum).then_some(distance);
                    assert_eq!(
                        standard_distance_units_bounded(source, target, maximum),
                        expected(standard_distance_units(source, target))
                    );
                    assert_eq!(
                        transposition_distance_units_bounded(source, target, maximum),
                        expected(transposition_distance_units(source, target))
                    );
                    assert_eq!(
                        damerau_levenshtein_distance_units_bounded(source, target, maximum),
                        expected(damerau_levenshtein_distance_units(source, target))
                    );
                    assert_eq!(
                        merge_and_split_distance_units_bounded(source, target, maximum),
                        expected(merge_and_split_distance_units(source, target))
                    );
                }
            }
        }
    }
}
