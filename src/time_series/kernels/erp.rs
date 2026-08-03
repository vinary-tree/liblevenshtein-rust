//! Edit distance with Real Penalty (ERP).
//!
//! ERP aligns two real-valued sequences with three operations: match two
//! samples at cost `|x - y|`, delete `x` at cost `|x - g|`, or insert `y` at
//! cost `|y - g|`. The finite scalar `g` is the *gap value*. Unlike a constant
//! edit penalty, the price of a gap therefore depends on the sample.
//!
//! The implementation provides both the exact two-row dynamic program and the
//! interval-relaxed column transition required by
//! [`crate::time_series::elastic::ElasticTransducer`]. At a quantized target
//! interval `[lo, hi]`, the two target-dependent costs use the exact box minima
//! `dist(x, [lo, hi])` and `dist(g, [lo, hi])`.
//!
//! Chen and Ng introduced ERP in *On the Marriage of Lp-norms and Edit
//! Distance*, VLDB 2004, DOI
//! [10.1016/B978-012088469-8.50070-X](https://doi.org/10.1016/B978-012088469-8.50070-X).

use super::super::elastic::interval::interval_dist;
use super::super::elastic::{Cost, ElasticKernel, ElasticTransducer, MetricElasticKernel};
use crate::cost::{CostMonoid, WeightedCost};

const DEFAULT_ERP_GAP: f64 = 0.0;

#[inline]
fn normalize_gap(g: f64) -> f64 {
    if g.is_finite() {
        g
    } else {
        DEFAULT_ERP_GAP
    }
}

#[inline]
fn series_is_finite(series: &[f64]) -> bool {
    series.iter().all(|value| value.is_finite())
}

#[inline]
fn gap_mass(series: &[f64], g: f64) -> f64 {
    series.iter().fold(0.0, |mass, value| {
        WeightedCost::combine(mass, (*value - g).abs())
    })
}

/// Configuration and elastic kernel for Edit distance with Real Penalty.
///
/// `g` is normalized to a finite value by [`Self::new`] and at every public
/// computation boundary. Directly constructed configurations with a non-finite
/// field therefore behave as `g = 0`, rather than contaminating queues with
/// NaN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErpConfig {
    /// Real-valued gap symbol.
    pub g: f64,
}

impl Default for ErpConfig {
    #[inline]
    fn default() -> Self {
        Self::new(DEFAULT_ERP_GAP)
    }
}

impl ErpConfig {
    /// Construct ERP with finite gap value `g`.
    ///
    /// Non-finite inputs use the conventional zero gap.
    #[inline]
    pub fn new(g: f64) -> Self {
        Self {
            g: normalize_gap(g),
        }
    }

    /// Construct ERP with the conventional gap value `g = 0`.
    #[inline]
    pub fn default_gap() -> Self {
        Self::default()
    }

    /// Effective finite gap value.
    #[inline]
    pub fn gap_value(&self) -> f64 {
        normalize_gap(self.g)
    }

    /// Return a configuration whose public field is normalized.
    #[inline]
    pub fn normalized(self) -> Self {
        Self::new(self.g)
    }

    /// Compute exact ERP distance in `O(mn)` time and `O(min(m,n))` space.
    ///
    /// Returns positive infinity when either sequence contains a non-finite
    /// sample. Empty sequences are lawful: their distance is the other
    /// sequence's total gap mass.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        self.distance_with_cutoff(x, y, f64::INFINITY)
            .unwrap_or(f64::INFINITY)
    }

    /// Compute exact ERP distance if it is at most `max_cost`.
    ///
    /// A row-minimum cutoff abandons work without changing the result: every
    /// path to the final cell crosses every completed row and all later edge
    /// costs are non-negative.
    pub fn distance_with_cutoff(&self, x: &[f64], y: &[f64], max_cost: f64) -> Option<f64> {
        if max_cost.is_nan() || max_cost < 0.0 || !series_is_finite(x) || !series_is_finite(y) {
            return None;
        }

        // Keep the allocated row on the shorter axis. ERP is symmetric.
        if y.len() > x.len() {
            return self.distance_with_cutoff(y, x, max_cost);
        }

        let g = self.gap_value();
        let row_len = y.len().checked_add(1)?;
        let mut previous = vec![WeightedCost::TOP; row_len];
        let mut current = vec![WeightedCost::TOP; row_len];
        previous[0] = WeightedCost::ZERO;
        for (j, value) in y.iter().enumerate() {
            previous[j + 1] = WeightedCost::combine(previous[j], (*value - g).abs());
        }

        for x_value in x {
            let delete_cost = (*x_value - g).abs();
            current[0] = WeightedCost::combine(previous[0], delete_cost);
            let mut row_min = current[0];

            for (j, y_value) in y.iter().enumerate() {
                let column = j + 1;
                let substitute =
                    WeightedCost::combine(previous[column - 1], (*x_value - *y_value).abs());
                let delete = WeightedCost::combine(previous[column], delete_cost);
                let insert = WeightedCost::combine(current[column - 1], (*y_value - g).abs());
                current[column] = substitute.min(delete).min(insert);
                row_min = row_min.min(current[column]);
            }

            if !WeightedCost::within(row_min, max_cost) {
                return None;
            }
            std::mem::swap(&mut previous, &mut current);
        }

        let exact = previous[y.len()];
        WeightedCost::within(exact, max_cost).then_some(exact)
    }

    /// Potential-based lower bound for one candidate pair.
    #[inline]
    pub fn candidate_lower_bound(&self, x: &[f64], y: &[f64]) -> f64 {
        if !series_is_finite(x) || !series_is_finite(y) {
            return WeightedCost::TOP;
        }
        erp_gap_mass_lower_bound(x, y, self.gap_value())
    }
}

/// ERP kernel name for APIs that distinguish configuration from policy.
pub type ErpKernel = ErpConfig;

/// Exact ERP index backed by the generic elastic trie walker.
pub type ErpTransducer<V = usize> = ElasticTransducer<ErpKernel, V>;

/// Lower-bound ERP by the difference in total gap mass.
///
/// Every ERP edit changes the potential `sum |value - g|` by no more than its
/// own cost. The reverse triangle inequality therefore gives
/// `|mass(x) - mass(y)| <= ERP(x,y)`.
#[inline]
pub fn erp_gap_mass_lower_bound(x: &[f64], y: &[f64], g: f64) -> f64 {
    let g = normalize_gap(g);
    let x_mass = gap_mass(x, g);
    let y_mass = gap_mass(y, g);
    if x_mass == WeightedCost::TOP || y_mass == WeightedCost::TOP {
        // Infinity minus infinity is undefined and cannot be used to prune.
        WeightedCost::ZERO
    } else {
        (x_mass - y_mass).abs()
    }
}

impl ElasticKernel for ErpConfig {
    const IS_METRIC: bool = true;

    type Monoid = WeightedCost;
    type Carry = ();
    type QueryPlan = ();

    #[inline]
    fn normalized(self) -> Self {
        ErpConfig::normalized(self)
    }

    #[inline]
    fn supports_interval_query(&self, query: &[f64]) -> bool {
        series_is_finite(query)
    }

    #[inline]
    fn column_len(&self, query_len: usize) -> Option<usize> {
        query_len.checked_add(1)
    }

    #[inline]
    fn final_row(&self, query_len: usize) -> usize {
        query_len
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
        let Some(column_len) = query.len().checked_add(1) else {
            column.clear();
            return (WeightedCost::TOP, ());
        };
        column.resize(column_len, WeightedCost::TOP);

        let g = self.gap_value();
        let (lo, hi) = current_interval;
        let target_gap = interval_dist(g, lo, hi);

        // At depth one the generic walker supplies an all-TOP sentinel. Build
        // ERP's exact empty-target boundary lazily; later depths use the
        // preceding interval column.
        let root_boundary = |row: usize| {
            query[..row].iter().fold(WeightedCost::ZERO, |cost, value| {
                WeightedCost::combine(cost, (*value - g).abs())
            })
        };
        let previous_cell = |row: usize| {
            if depth == 1 {
                root_boundary(row)
            } else {
                previous.get(row).copied().unwrap_or(WeightedCost::TOP)
            }
        };

        column[0] = WeightedCost::combine(previous_cell(0), target_gap);
        let mut lower_bound = column[0];
        for (row_index, query_value) in query.iter().enumerate() {
            let row = row_index + 1;
            let substitute =
                WeightedCost::combine(previous_cell(row - 1), interval_dist(*query_value, lo, hi));
            let delete = WeightedCost::combine(column[row - 1], (*query_value - g).abs());
            let insert = WeightedCost::combine(previous_cell(row), target_gap);
            column[row] = substitute.min(delete).min(insert);
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
        ErpConfig::candidate_lower_bound(self, query, candidate)
    }

    #[inline]
    fn plan(&self, _query: &[f64]) -> Self::QueryPlan {}

    #[inline]
    fn empty_pair_cost(&self) -> Cost<Self> {
        WeightedCost::ZERO
    }

    #[inline]
    fn empty_vs_nonempty_cost(&self, nonempty: &[f64]) -> Cost<Self> {
        if series_is_finite(nonempty) {
            gap_mass(nonempty, self.gap_value())
        } else {
            WeightedCost::TOP
        }
    }
}

impl MetricElasticKernel for ErpConfig {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn reference_distance(x: &[f64], y: &[f64], g: f64) -> f64 {
        if !series_is_finite(x) || !series_is_finite(y) {
            return f64::INFINITY;
        }
        let rows = x.len() + 1;
        let columns = y.len() + 1;
        let mut matrix = vec![vec![0.0; columns]; rows];
        for i in 1..rows {
            matrix[i][0] = matrix[i - 1][0] + (x[i - 1] - g).abs();
        }
        for j in 1..columns {
            matrix[0][j] = matrix[0][j - 1] + (y[j - 1] - g).abs();
        }
        for i in 1..rows {
            for j in 1..columns {
                matrix[i][j] = (matrix[i - 1][j - 1] + (x[i - 1] - y[j - 1]).abs())
                    .min(matrix[i - 1][j] + (x[i - 1] - g).abs())
                    .min(matrix[i][j - 1] + (y[j - 1] - g).abs());
            }
        }
        matrix[x.len()][y.len()]
    }

    fn scalar_column(query: &[f64], target: &[f64], g: f64) -> Vec<f64> {
        let mut previous = Vec::with_capacity(query.len() + 1);
        previous.push(0.0);
        for value in query {
            previous.push(previous.last().copied().unwrap_or(0.0) + (*value - g).abs());
        }
        for target_value in target {
            let mut column = vec![0.0; query.len() + 1];
            column[0] = previous[0] + (*target_value - g).abs();
            for i in 1..=query.len() {
                column[i] = (previous[i - 1] + (query[i - 1] - *target_value).abs())
                    .min(column[i - 1] + (query[i - 1] - g).abs())
                    .min(previous[i] + (*target_value - g).abs());
            }
            previous = column;
        }
        previous
    }

    fn quotient_normal_form(series: &[f64], g: f64) -> Vec<f64> {
        series.iter().copied().filter(|value| *value != g).collect()
    }

    #[test]
    fn worked_and_boundary_examples() {
        let erp = ErpConfig::new(0.0);
        assert_eq!(erp.distance(&[], &[]), 0.0);
        assert_eq!(erp.distance(&[1.0, -2.0], &[]), 3.0);
        assert_eq!(erp.distance(&[], &[1.0, -2.0]), 3.0);
        assert_eq!(erp.distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(erp.distance(&[1.0, 2.0], &[1.0, 3.0]), 1.0);
        assert_eq!(erp.distance(&[1.0, 0.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn chen_ng_vldb_worked_example() {
        let erp = ErpConfig::new(0.0);
        let q = [0.0];
        let r = [1.0, 2.0];
        let s = [2.0, 3.0, 3.0];

        assert_eq!(erp.distance(&q, &r), 3.0);
        assert_eq!(erp.distance(&r, &s), 5.0);
        assert_eq!(erp.distance(&q, &s), 8.0);
        assert!(erp.distance(&q, &s) <= erp.distance(&q, &r) + erp.distance(&r, &s));
        assert_eq!(erp.distance(&[3.0], &r), 2.0);
        assert_eq!(erp.distance(&[3.0], &s), 5.0);
    }

    #[test]
    fn cutoff_and_invalid_domain_are_total() {
        let erp = ErpConfig::new(0.0);
        assert_eq!(erp.distance_with_cutoff(&[1.0], &[3.0], 2.0), Some(2.0));
        assert_eq!(erp.distance_with_cutoff(&[1.0], &[3.0], 1.0), None);
        assert_eq!(
            erp.distance_with_cutoff(&[f64::NAN], &[3.0], f64::INFINITY),
            None
        );
        assert_eq!(erp.distance_with_cutoff(&[1.0], &[3.0], f64::NAN), None);
        assert_eq!(ErpConfig { g: f64::NAN }.gap_value(), 0.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn optimized_dp_matches_independent_full_matrix(
            x in prop::collection::vec(-20i16..=20, 0..9),
            y in prop::collection::vec(-20i16..=20, 0..9),
            g in -10i16..=10,
            cutoff in 0u16..=120,
        ) {
            let x: Vec<f64> = x.into_iter().map(f64::from).collect();
            let y: Vec<f64> = y.into_iter().map(f64::from).collect();
            let g = f64::from(g);
            let actual = ErpConfig::new(g).distance(&x, &y);
            prop_assert_eq!(actual, reference_distance(&x, &y, g));
            let cutoff = f64::from(cutoff);
            let expected_cutoff = (actual <= cutoff + WeightedCost::EPSILON).then_some(actual);
            prop_assert_eq!(ErpConfig::new(g).distance_with_cutoff(&x, &y, cutoff), expected_cutoff);
        }

        #[test]
        fn interval_columns_are_admissible_and_point_bins_are_exact(
            query in prop::collection::vec(-12i16..=12, 1..8),
            target in prop::collection::vec(-12i16..=12, 1..8),
            radii in prop::collection::vec(0u8..=4, 1..8),
            g in -6i16..=6,
        ) {
            let query: Vec<f64> = query.into_iter().map(f64::from).collect();
            let target: Vec<f64> = target.into_iter().map(f64::from).collect();
            let g = f64::from(g);
            let kernel = ErpConfig::new(g);
            let mut relaxed_previous = vec![f64::INFINITY; query.len() + 1];
            let mut point_previous = vec![f64::INFINITY; query.len() + 1];
            for (depth_index, value) in target.iter().enumerate() {
                let radius = f64::from(radii[depth_index % radii.len()]);
                let mut relaxed = Vec::new();
                let mut point = Vec::new();
                kernel.step_column(
                    &relaxed_previous,
                    &query,
                    (*value - radius, *value + radius),
                    (depth_index > 0).then_some(()),
                    depth_index + 1,
                    &(),
                    &mut relaxed,
                );
                kernel.step_column(
                    &point_previous,
                    &query,
                    (*value, *value),
                    (depth_index > 0).then_some(()),
                    depth_index + 1,
                    &(),
                    &mut point,
                );
                let center = scalar_column(&query, &target[..=depth_index], g);
                let low_target: Vec<_> = target[..=depth_index]
                    .iter()
                    .enumerate()
                    .map(|(index, center)| {
                        *center - f64::from(radii[index % radii.len()])
                    })
                    .collect();
                let high_target: Vec<_> = target[..=depth_index]
                    .iter()
                    .enumerate()
                    .map(|(index, center)| {
                        *center + f64::from(radii[index % radii.len()])
                    })
                    .collect();
                let low = scalar_column(&query, &low_target, g);
                let high = scalar_column(&query, &high_target, g);
                prop_assert_eq!(&point, &center);
                for row in 0..relaxed.len() {
                    prop_assert!(relaxed[row] <= center[row] + WeightedCost::EPSILON);
                    prop_assert!(relaxed[row] <= low[row] + WeightedCost::EPSILON);
                    prop_assert!(relaxed[row] <= high[row] + WeightedCost::EPSILON);
                }
                relaxed_previous = relaxed;
                point_previous = point;
            }
        }

        #[test]
        fn box_costs_are_leaf_exact(
            scalar in -20i16..=20,
            lo in -20i16..=20,
            width in 0u8..=20,
        ) {
            let scalar = f64::from(scalar);
            let lo = f64::from(lo);
            let hi = lo + f64::from(width);
            let clamp = scalar.clamp(lo, hi);
            let brute = [lo, hi, clamp]
                .into_iter()
                .map(|value| (scalar - value).abs())
                .fold(f64::INFINITY, f64::min);
            prop_assert_eq!(interval_dist(scalar, lo, hi), brute);
        }

        #[test]
        fn gap_mass_bound_and_metric_quotient_laws_hold(
            x in prop::collection::vec(-8i8..=8, 0..7),
            y in prop::collection::vec(-8i8..=8, 0..7),
            z in prop::collection::vec(-8i8..=8, 0..7),
            g in -4i8..=4,
        ) {
            let x: Vec<f64> = x.into_iter().map(f64::from).collect();
            let y: Vec<f64> = y.into_iter().map(f64::from).collect();
            let z: Vec<f64> = z.into_iter().map(f64::from).collect();
            let g = f64::from(g);
            let erp = ErpConfig::new(g);
            let dxy = erp.distance(&x, &y);
            let dyx = erp.distance(&y, &x);
            let dxz = erp.distance(&x, &z);
            let dzy = erp.distance(&z, &y);

            prop_assert!(dxy >= 0.0);
            prop_assert_eq!(dxy, dyx);
            prop_assert!(dxy <= dxz + dzy + WeightedCost::EPSILON);
            prop_assert!(erp.candidate_lower_bound(&x, &y) <= dxy + WeightedCost::EPSILON);
            prop_assert_eq!(dxy == 0.0, quotient_normal_form(&x, g) == quotient_normal_form(&y, g));
        }
    }
}
