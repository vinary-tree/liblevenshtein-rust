//! Quadratic Gotoh reference implementation for affine gap costs.

use crate::transducer::AffineGapParams;

const UNREACHABLE: usize = usize::MAX;

#[inline]
fn add(cost: usize, increment: usize) -> usize {
    if cost == UNREACHABLE {
        UNREACHABLE
    } else {
        cost.checked_add(increment).unwrap_or(UNREACHABLE)
    }
}

#[inline]
fn gap_open(params: AffineGapParams) -> usize {
    params
        .gap_open()
        .checked_add(params.gap_extend())
        .unwrap_or(UNREACHABLE)
}

/// Compute exact scaled affine-gap distance between two unit sequences.
///
/// This is the deliberately direct `O(nm)` three-matrix recurrence used as an
/// oracle for the lazy automaton. A gap containing `k` units costs
/// `gap_open + k * gap_extend`. Arithmetic overflow makes that route
/// unreachable rather than wrapping.
pub fn affine_gap_distance_units<T: Eq>(
    source: &[T],
    target: &[T],
    params: AffineGapParams,
) -> Option<usize> {
    let rows = source.len().checked_add(1)?;
    let columns = target.len().checked_add(1)?;
    let cells = rows.checked_mul(columns)?;
    let mut matched = vec![UNREACHABLE; cells];
    let mut source_gap = vec![UNREACHABLE; cells];
    let mut target_gap = vec![UNREACHABLE; cells];
    let index = |i: usize, j: usize| i * columns + j;
    let open = gap_open(params);

    matched[index(0, 0)] = 0;
    for i in 1..rows {
        source_gap[index(i, 0)] = params
            .gap_open()
            .checked_add(i.checked_mul(params.gap_extend())?)
            .unwrap_or(UNREACHABLE);
    }
    for j in 1..columns {
        target_gap[index(0, j)] = params
            .gap_open()
            .checked_add(j.checked_mul(params.gap_extend())?)
            .unwrap_or(UNREACHABLE);
    }

    for i in 1..rows {
        for j in 1..columns {
            let diagonal = index(i - 1, j - 1);
            let substitution =
                usize::from(source[i - 1] != target[j - 1]).checked_mul(params.substitution())?;
            matched[index(i, j)] = add(
                matched[diagonal]
                    .min(source_gap[diagonal])
                    .min(target_gap[diagonal]),
                substitution,
            );

            let above = index(i - 1, j);
            source_gap[index(i, j)] = add(matched[above], open)
                .min(add(source_gap[above], params.gap_extend()))
                .min(add(target_gap[above], open));

            let left = index(i, j - 1);
            target_gap[index(i, j)] = add(matched[left], open)
                .min(add(target_gap[left], params.gap_extend()))
                .min(add(source_gap[left], open));
        }
    }

    let end = index(source.len(), target.len());
    let distance = matched[end].min(source_gap[end]).min(target_gap[end]);
    (distance != UNREACHABLE).then_some(distance)
}

/// Compute exact scaled affine-gap distance between Unicode scalar sequences.
pub fn affine_gap_distance(source: &str, target: &str, params: AffineGapParams) -> Option<usize> {
    let source: Vec<_> = source.chars().collect();
    let target: Vec<_> = target.chars().collect();
    affine_gap_distance_units(&source, &target, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::CostScale;

    fn params(open: usize, extend: usize, substitution: usize) -> AffineGapParams {
        AffineGapParams::from_scaled(
            CostScale::new(1).expect("unit scale"),
            open,
            extend,
            substitution,
        )
    }

    #[test]
    fn a_gap_run_pays_open_once() {
        let params = params(3, 2, 10);
        assert_eq!(affine_gap_distance("a", "abcd", params), Some(9));
        assert_eq!(affine_gap_distance("abcd", "a", params), Some(9));
    }

    #[test]
    fn zero_open_degenerates_to_levenshtein() {
        let params = params(0, 1, 1);
        assert_eq!(affine_gap_distance("kitten", "sitting", params), Some(3));
    }

    #[test]
    fn boundary_cases_are_total() {
        let params = params(2, 1, 1);
        assert_eq!(affine_gap_distance("", "", params), Some(0));
        assert_eq!(affine_gap_distance("", "abc", params), Some(5));
        assert_eq!(affine_gap_distance("abc", "", params), Some(5));
    }
}
