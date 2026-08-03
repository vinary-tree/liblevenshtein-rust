//! Discrete Fréchet distance for real-valued sequences.
//!
//! A *coupling* walks monotonically through both sequences, advancing the
//! first sequence, the second sequence, or both at every step. Its cost is the
//! longest link encountered. Discrete Fréchet distance is the minimum coupling
//! cost, giving the familiar dog-and-leash interpretation while respecting
//! sample order.
//!
//! This kernel is the generic elastic walker's first minimax consumer. Path
//! costs use [`crate::cost::BottleneckCost`]: extending a path combines its
//! accumulated cost with a new link using `max`, while alternative predecessor
//! paths are still selected using `min`. Quantized trie edges replace
//! `|query[i] - target[j]|` with its exact minimum over the target bin.
//!
//! The recurrence is Table 1 of Eiter and Mannila, *Computing Discrete Fréchet
//! Distance*, Technical Report CD-TR 94/64, TU Vienna, 1994
//! ([author-hosted report](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/files/cdtr9464.pdf)).

use super::super::elastic::interval::interval_dist;
use super::super::elastic::{Cost, ElasticKernel, ElasticTransducer, MetricElasticKernel};
use crate::cost::{BottleneckCost, CostMonoid};

#[inline]
fn series_is_finite(series: &[f64]) -> bool {
    series.iter().all(|value| value.is_finite())
}

/// Configuration and elastic kernel for one-dimensional discrete Fréchet.
///
/// The scalar absolute-distance model has no runtime parameter. A named unit
/// struct nevertheless makes constructors and future point-metric variants
/// explicit at the public API boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FrechetConfig;

impl FrechetConfig {
    /// Construct the scalar absolute-distance kernel.
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    /// Compute exact discrete Fréchet distance.
    ///
    /// The implementation uses `O(mn)` time and `O(min(m,n))` space. Two empty
    /// sequences have distance zero; exactly one empty sequence has distance
    /// positive infinity because no endpoint-covering coupling exists.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        self.distance_with_cutoff(x, y, f64::INFINITY)
            .unwrap_or(f64::INFINITY)
    }

    /// Compute exact distance when it does not exceed `max_cost`.
    ///
    /// A completed row whose minimum exceeds the cutoff cannot recover: every
    /// later path extends one of those cells and bottleneck accumulation uses
    /// `max`. This is the executable counterpart of the K2 inflation law.
    pub fn distance_with_cutoff(&self, x: &[f64], y: &[f64], max_cost: f64) -> Option<f64> {
        if max_cost.is_nan() || max_cost < 0.0 || !series_is_finite(x) || !series_is_finite(y) {
            return None;
        }

        match (x.is_empty(), y.is_empty()) {
            (true, true) => {
                return BottleneckCost::within(BottleneckCost::ZERO, max_cost)
                    .then_some(BottleneckCost::ZERO);
            }
            (true, false) | (false, true) => {
                return BottleneckCost::within(BottleneckCost::TOP, max_cost)
                    .then_some(BottleneckCost::TOP);
            }
            (false, false) => {}
        }

        // The recurrence is symmetric, so allocate on the shorter axis.
        if y.len() > x.len() {
            return self.distance_with_cutoff(y, x, max_cost);
        }

        let mut previous = vec![BottleneckCost::TOP; y.len()];
        let mut current = vec![BottleneckCost::TOP; y.len()];

        previous[0] = (x[0] - y[0]).abs();
        for column in 1..y.len() {
            previous[column] =
                BottleneckCost::combine(previous[column - 1], (x[0] - y[column]).abs());
        }
        if !BottleneckCost::within(
            previous.iter().copied().fold(BottleneckCost::TOP, f64::min),
            max_cost,
        ) {
            return None;
        }

        for x_value in &x[1..] {
            current[0] = BottleneckCost::combine(previous[0], (*x_value - y[0]).abs());
            let mut row_min = current[0];
            for column in 1..y.len() {
                let predecessor = previous[column - 1]
                    .min(previous[column])
                    .min(current[column - 1]);
                current[column] =
                    BottleneckCost::combine(predecessor, (*x_value - y[column]).abs());
                row_min = row_min.min(current[column]);
            }
            if !BottleneckCost::within(row_min, max_cost) {
                return None;
            }
            std::mem::swap(&mut previous, &mut current);
        }

        let exact = previous[y.len() - 1];
        BottleneckCost::within(exact, max_cost).then_some(exact)
    }

    /// Combined pinned-endpoint and one-sided-Hausdorff candidate bound.
    #[inline]
    pub fn candidate_lower_bound(&self, query: &[f64], candidate: &[f64]) -> f64 {
        frechet_candidate_lower_bound(query, candidate)
    }
}

/// Kernel name for APIs that distinguish configuration from policy.
pub type FrechetKernel = FrechetConfig;

/// Exact discrete Fréchet index backed by the generic elastic trie walker.
pub type FrechetTransducer<V = usize> = ElasticTransducer<FrechetKernel, V>;

/// Lower bound from the two endpoint links forced into every coupling.
pub fn frechet_endpoint_lower_bound(x: &[f64], y: &[f64]) -> f64 {
    if !series_is_finite(x) || !series_is_finite(y) {
        return BottleneckCost::TOP;
    }
    match (x.first(), x.last(), y.first(), y.last()) {
        (None, None, None, None) => BottleneckCost::ZERO,
        (Some(x_first), Some(x_last), Some(y_first), Some(y_last)) => {
            (*x_first - *y_first).abs().max((*x_last - *y_last).abs())
        }
        _ => BottleneckCost::TOP,
    }
}

/// One-sided Hausdorff lower bound from `x` to `y`.
///
/// Every sample of `x` occurs in every coupling and is paired with some sample
/// of `y`, so `max_x min_y |x-y|` cannot exceed the coupling bottleneck. Sorting
/// `y` yields `O((m+n) log n)` time without changing either input.
pub fn frechet_one_sided_hausdorff_lower_bound(x: &[f64], y: &[f64]) -> f64 {
    if !series_is_finite(x) || !series_is_finite(y) {
        return BottleneckCost::TOP;
    }
    if x.is_empty() {
        return BottleneckCost::ZERO;
    }
    if y.is_empty() {
        return BottleneckCost::TOP;
    }

    let mut sorted = y.to_vec();
    sorted.sort_by(f64::total_cmp);
    x.iter().fold(BottleneckCost::ZERO, |bound, value| {
        let nearest = match sorted.binary_search_by(|probe| probe.total_cmp(value)) {
            Ok(_) => 0.0,
            Err(index) => {
                let below = index
                    .checked_sub(1)
                    .map_or(BottleneckCost::TOP, |i| (*value - sorted[i]).abs());
                let above = sorted
                    .get(index)
                    .map_or(BottleneckCost::TOP, |candidate| (*value - *candidate).abs());
                below.min(above)
            }
        };
        bound.max(nearest)
    })
}

/// Candidate bound used by exact trie search.
///
/// The endpoint and one-sided Hausdorff bounds are independently admissible;
/// taking their maximum is therefore admissible and at least as tight as each
/// component.
pub fn frechet_candidate_lower_bound(x: &[f64], y: &[f64]) -> f64 {
    frechet_endpoint_lower_bound(x, y).max(frechet_one_sided_hausdorff_lower_bound(x, y))
}

impl ElasticKernel for FrechetConfig {
    const IS_METRIC: bool = true;

    type Monoid = BottleneckCost;
    type Carry = ();
    type QueryPlan = ();

    #[inline]
    fn supports_interval_query(&self, query: &[f64]) -> bool {
        series_is_finite(query)
    }

    #[inline]
    fn column_len(&self, query_len: usize) -> Option<usize> {
        (query_len > 0).then_some(query_len)
    }

    #[inline]
    fn final_row(&self, query_len: usize) -> usize {
        query_len.saturating_sub(1)
    }

    fn step_column(
        &self,
        previous: &[Cost<Self>],
        query: &[f64],
        current_interval: (f64, f64),
        _previous_carry: Option<Self::Carry>,
        depth: usize,
        _plan: &Self::QueryPlan,
        column: &mut Vec<Cost<Self>>,
    ) -> (Cost<Self>, Self::Carry) {
        if query.is_empty() || depth == 0 {
            column.clear();
            return (BottleneckCost::TOP, ());
        }

        column.resize(query.len(), BottleneckCost::TOP);
        let (low, high) = current_interval;
        let first_link = interval_dist(query[0], low, high);
        column[0] = if depth == 1 {
            first_link
        } else {
            BottleneckCost::combine(
                previous.first().copied().unwrap_or(BottleneckCost::TOP),
                first_link,
            )
        };
        let mut lower_bound = column[0];

        for row in 1..query.len() {
            let predecessor = if depth == 1 {
                column[row - 1]
            } else {
                previous
                    .get(row - 1)
                    .copied()
                    .unwrap_or(BottleneckCost::TOP)
                    .min(previous.get(row).copied().unwrap_or(BottleneckCost::TOP))
                    .min(column[row - 1])
            };
            column[row] =
                BottleneckCost::combine(predecessor, interval_dist(query[row], low, high));
            lower_bound = lower_bound.min(column[row]);
        }

        (lower_bound, ())
    }

    #[inline]
    fn exact_with_cutoff(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: Cost<Self>,
    ) -> Option<Cost<Self>> {
        self.distance_with_cutoff(query, candidate, cutoff)
    }

    #[inline]
    fn candidate_lower_bound(
        &self,
        query: &[f64],
        candidate: &[f64],
        _plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        FrechetConfig::candidate_lower_bound(self, query, candidate)
    }

    #[inline]
    fn plan(&self, _query: &[f64]) -> Self::QueryPlan {}

    #[inline]
    fn empty_pair_cost(&self) -> Cost<Self> {
        BottleneckCost::ZERO
    }

    #[inline]
    fn empty_vs_nonempty_cost(&self, _nonempty: &[f64]) -> Cost<Self> {
        BottleneckCost::TOP
    }
}

impl MetricElasticKernel for FrechetConfig {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn reference_distance(x: &[f64], y: &[f64]) -> f64 {
        if !series_is_finite(x) || !series_is_finite(y) {
            return f64::INFINITY;
        }
        match (x.is_empty(), y.is_empty()) {
            (true, true) => return 0.0,
            (true, false) | (false, true) => return f64::INFINITY,
            (false, false) => {}
        }

        let mut matrix = vec![vec![f64::INFINITY; y.len()]; x.len()];
        matrix[0][0] = (x[0] - y[0]).abs();
        for i in 1..x.len() {
            matrix[i][0] = matrix[i - 1][0].max((x[i] - y[0]).abs());
        }
        for j in 1..y.len() {
            matrix[0][j] = matrix[0][j - 1].max((x[0] - y[j]).abs());
        }
        for i in 1..x.len() {
            for j in 1..y.len() {
                matrix[i][j] = (x[i] - y[j]).abs().max(
                    matrix[i - 1][j]
                        .min(matrix[i - 1][j - 1])
                        .min(matrix[i][j - 1]),
                );
            }
        }
        matrix[x.len() - 1][y.len() - 1]
    }

    fn scalar_column(query: &[f64], target: &[f64]) -> Vec<f64> {
        assert!(!query.is_empty() && !target.is_empty());
        let mut previous = vec![f64::INFINITY; query.len()];
        for (depth, target_value) in target.iter().enumerate() {
            let mut column = vec![f64::INFINITY; query.len()];
            column[0] = if depth == 0 {
                (query[0] - *target_value).abs()
            } else {
                previous[0].max((query[0] - *target_value).abs())
            };
            for row in 1..query.len() {
                let predecessor = if depth == 0 {
                    column[row - 1]
                } else {
                    previous[row - 1].min(previous[row]).min(column[row - 1])
                };
                column[row] = predecessor.max((query[row] - *target_value).abs());
            }
            previous = column;
        }
        previous
    }

    fn run_length_normal_form(series: &[f64]) -> Vec<f64> {
        let mut normal = Vec::with_capacity(series.len());
        for value in series {
            if normal.last() != Some(value) {
                normal.push(*value);
            }
        }
        normal
    }

    #[test]
    fn eiter_mannila_table_one_branches_and_boundaries() {
        let frechet = FrechetConfig::new();
        assert_eq!(frechet.distance(&[], &[]), 0.0);
        assert_eq!(frechet.distance(&[], &[1.0]), f64::INFINITY);
        assert_eq!(frechet.distance(&[1.0], &[]), f64::INFINITY);
        assert_eq!(frechet.distance(&[0.0], &[3.0]), 3.0);
        assert_eq!(frechet.distance(&[0.0], &[1.0, 3.0]), 3.0);
        assert_eq!(frechet.distance(&[0.0, 2.0], &[1.0]), 1.0);
        assert_eq!(frechet.distance(&[0.0, 1.0, 2.0], &[0.0, 2.0]), 1.0);
        assert_eq!(frechet.distance(&[1.0, 1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn cutoff_and_invalid_domain_are_total() {
        let frechet = FrechetConfig::new();
        assert_eq!(frechet.distance_with_cutoff(&[0.0], &[2.0], 2.0), Some(2.0));
        assert_eq!(frechet.distance_with_cutoff(&[0.0], &[2.0], 1.0), None);
        assert_eq!(frechet.distance_with_cutoff(&[f64::NAN], &[2.0], 2.0), None);
        assert_eq!(frechet.distance_with_cutoff(&[0.0], &[2.0], f64::NAN), None);
        assert_eq!(
            frechet.distance_with_cutoff(&[], &[2.0], f64::INFINITY),
            Some(f64::INFINITY)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn optimized_dp_matches_independent_full_matrix(
            x in prop::collection::vec(-20i16..=20, 0..9),
            y in prop::collection::vec(-20i16..=20, 0..9),
            cutoff in 0u16..=60,
        ) {
            let x: Vec<f64> = x.into_iter().map(f64::from).collect();
            let y: Vec<f64> = y.into_iter().map(f64::from).collect();
            let actual = FrechetConfig::new().distance(&x, &y);
            prop_assert_eq!(actual, reference_distance(&x, &y));
            let cutoff = f64::from(cutoff);
            let expected = BottleneckCost::within(actual, cutoff).then_some(actual);
            prop_assert_eq!(FrechetConfig::new().distance_with_cutoff(&x, &y, cutoff), expected);
        }

        #[test]
        fn interval_columns_are_admissible_and_point_bins_are_exact(
            query in prop::collection::vec(-12i16..=12, 1..8),
            target in prop::collection::vec(-12i16..=12, 1..8),
            radii in prop::collection::vec(0u8..=4, 1..8),
        ) {
            let query: Vec<f64> = query.into_iter().map(f64::from).collect();
            let target: Vec<f64> = target.into_iter().map(f64::from).collect();
            let kernel = FrechetConfig::new();
            let mut previous = vec![BottleneckCost::TOP; query.len()];
            let mut point_previous = vec![BottleneckCost::TOP; query.len()];

            for (index, target_value) in target.iter().enumerate() {
                let radius = f64::from(radii[index % radii.len()]);
                let interval = (*target_value - radius, *target_value + radius);
                let mut relaxed = Vec::new();
                kernel.step_column(&previous, &query, interval, None, index + 1, &(), &mut relaxed);

                for realization in [interval.0, *target_value, interval.1] {
                    let mut concrete_target = target[..index].to_vec();
                    concrete_target.push(realization);
                    let exact = scalar_column(&query, &concrete_target);
                    for row in 0..query.len() {
                        prop_assert!(relaxed[row] <= exact[row] + BottleneckCost::EPSILON);
                    }
                }

                let mut point = Vec::new();
                kernel.step_column(
                    &point_previous,
                    &query,
                    (*target_value, *target_value),
                    None,
                    index + 1,
                    &(),
                    &mut point,
                );
                prop_assert_eq!(&point, &scalar_column(&query, &target[..=index]));
                previous = relaxed;
                point_previous = point;
            }
        }

        #[test]
        fn interval_leaf_is_exact_on_an_integer_box(
            value in -20i16..=20,
            low in -20i16..=20,
            width in 0u8..=8,
        ) {
            let high = low + i16::from(width);
            let expected = (low..=high)
                .map(|concrete| (f64::from(value) - f64::from(concrete)).abs())
                .fold(f64::INFINITY, f64::min);
            prop_assert_eq!(
                interval_dist(f64::from(value), f64::from(low), f64::from(high)),
                expected,
            );
        }

        #[test]
        fn bounds_metric_laws_and_run_length_identity(
            a in prop::collection::vec(-8i8..=8, 1..8),
            b in prop::collection::vec(-8i8..=8, 1..8),
            c in prop::collection::vec(-8i8..=8, 1..8),
        ) {
            let a: Vec<f64> = a.into_iter().map(f64::from).collect();
            let b: Vec<f64> = b.into_iter().map(f64::from).collect();
            let c: Vec<f64> = c.into_iter().map(f64::from).collect();
            let metric = FrechetConfig::new();
            let ab = metric.distance(&a, &b);
            let ba = metric.distance(&b, &a);
            let bc = metric.distance(&b, &c);
            let ac = metric.distance(&a, &c);

            prop_assert!(ab >= 0.0);
            prop_assert_eq!(ab, ba);
            prop_assert!(ac <= ab + bc + BottleneckCost::EPSILON);
            prop_assert_eq!(ab == 0.0, run_length_normal_form(&a) == run_length_normal_form(&b));
            prop_assert!(frechet_endpoint_lower_bound(&a, &b) <= ab + BottleneckCost::EPSILON);
            prop_assert!(frechet_one_sided_hausdorff_lower_bound(&a, &b) <= ab + BottleneckCost::EPSILON);
            prop_assert!(frechet_candidate_lower_bound(&a, &b) <= ab + BottleneckCost::EPSILON);
        }
    }
}
