//! Symmetrically banded Dynamic Time Warping (DTW).
//!
//! DTW is shipped here as an exact, explicitly **non-metric** similarity
//! measure. The public constructor requires a Sakoe–Chiba half-width `band`;
//! there is no default or unbanded constructor. Internally the recurrence sums
//! squared point deviations to match LB_Keogh's proof. Public distances are
//! square roots, while the generic walker remains in squared-cost units.
//!
//! The band is both a semantic constraint and a pruning/resource boundary. A
//! cell `(i,j)` is live only when `|i-j| <= band`; all other cells are `TOP`.
//! The query plan contains monotonic-deque envelopes used for candidate
//! LB_Keogh and for an incremental prefix gate before each trie column.

use std::hash::Hash;

use super::super::elastic::interval::interval_dist;
use super::super::elastic::{Cost, ElasticKernel, ElasticSearchStats, ElasticTransducer};
use super::keogh::{interval_prefix_step, keogh_envelopes, lb_keogh_squared, KeoghPlan};
use crate::cost::{CostMonoid, WeightedCost};
use crate::time_series::encoding::QuantizationConfig;

#[inline]
fn series_is_finite(series: &[f64]) -> bool {
    series.iter().all(|value| value.is_finite())
}

#[inline]
fn square(value: f64) -> f64 {
    value * value
}

#[inline]
fn squared_cutoff(cutoff: f64) -> f64 {
    if cutoff == f64::INFINITY {
        f64::INFINITY
    } else {
        square(cutoff)
    }
}

/// Required configuration and elastic kernel for banded DTW.
///
/// `band` is the inclusive Sakoe–Chiba half-width. It is public for inspection
/// and serialization-oriented callers, but construction always requires an
/// explicit value through [`Self::new`]. This type intentionally has no
/// `Default` implementation.
///
/// ```compile_fail
/// use liblevenshtein::time_series::DtwConfig;
///
/// // A search window is never selected implicitly.
/// let _configuration = DtwConfig::default();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DtwConfig {
    /// Inclusive half-width `$`|i-j| <= band`$`.
    pub band: usize,
}

impl DtwConfig {
    /// Construct banded DTW with an explicit Sakoe–Chiba half-width.
    #[inline]
    pub const fn new(band: usize) -> Self {
        Self { band }
    }

    /// Exact root-distance DTW in `O(m * min(n, 2*band+1))` time.
    ///
    /// Returns positive infinity when the band cannot connect both endpoints,
    /// exactly one sequence is empty, a sample is non-finite, or squared
    /// accumulation exceeds finite `f64`.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        self.distance_squared(x, y).sqrt()
    }

    /// Exact squared DTW, the native cost used by the elastic kernel.
    pub fn distance_squared(&self, x: &[f64], y: &[f64]) -> f64 {
        self.distance_squared_with_cutoff(x, y, f64::INFINITY)
            .unwrap_or(f64::INFINITY)
    }

    /// Exact root-distance DTW if it is at most `max_distance`.
    pub fn distance_with_cutoff(&self, x: &[f64], y: &[f64], max_distance: f64) -> Option<f64> {
        if max_distance.is_nan() || max_distance < 0.0 {
            return None;
        }
        self.distance_squared_with_cutoff(x, y, squared_cutoff(max_distance))
            .map(f64::sqrt)
    }

    /// Exact squared DTW if it is at most `max_squared_cost`.
    pub fn distance_squared_with_cutoff(
        &self,
        x: &[f64],
        y: &[f64],
        max_squared_cost: f64,
    ) -> Option<f64> {
        if max_squared_cost.is_nan()
            || max_squared_cost < 0.0
            || !series_is_finite(x)
            || !series_is_finite(y)
        {
            return None;
        }
        match (x.is_empty(), y.is_empty()) {
            (true, true) => {
                return WeightedCost::within(WeightedCost::ZERO, max_squared_cost)
                    .then_some(WeightedCost::ZERO);
            }
            (true, false) | (false, true) => {
                return WeightedCost::within(WeightedCost::TOP, max_squared_cost)
                    .then_some(WeightedCost::TOP);
            }
            (false, false) => {}
        }
        if x.len().abs_diff(y.len()) > self.band {
            return WeightedCost::within(WeightedCost::TOP, max_squared_cost)
                .then_some(WeightedCost::TOP);
        }

        // The symmetric band and squared point metric permit transposition.
        if y.len() > x.len() {
            return self.distance_squared_with_cutoff(y, x, max_squared_cost);
        }

        let mut previous = vec![WeightedCost::TOP; y.len() + 1];
        let mut current = vec![WeightedCost::TOP; y.len() + 1];
        previous[0] = WeightedCost::ZERO;

        for (x_index, x_value) in x.iter().enumerate() {
            current.fill(WeightedCost::TOP);
            let row = x_index + 1;
            let start = row.saturating_sub(self.band).max(1);
            let end = row.saturating_add(self.band).min(y.len());
            let mut row_min = WeightedCost::TOP;
            if start <= end {
                for column in start..=end {
                    let predecessor = previous[column - 1]
                        .min(previous[column])
                        .min(current[column - 1]);
                    let local = square(*x_value - y[column - 1]);
                    current[column] = WeightedCost::combine(predecessor, local);
                    row_min = row_min.min(current[column]);
                }
            }
            if !WeightedCost::within(row_min, max_squared_cost) {
                return None;
            }
            std::mem::swap(&mut previous, &mut current);
        }

        let exact = previous[y.len()];
        WeightedCost::within(exact, max_squared_cost).then_some(exact)
    }
}

/// Kernel name for APIs that distinguish configuration from policy.
pub type DtwKernel = DtwConfig;

impl ElasticKernel for DtwConfig {
    const IS_METRIC: bool = false;

    type Monoid = WeightedCost;
    type Carry = f64;
    type QueryPlan = KeoghPlan;

    #[inline]
    fn supports_interval_query(&self, query: &[f64]) -> bool {
        !query.is_empty() && series_is_finite(query)
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
        previous_carry: Option<Self::Carry>,
        depth: usize,
        plan: &Self::QueryPlan,
        column: &mut Vec<Cost<Self>>,
    ) -> (Cost<Self>, Self::Carry) {
        let Some(column_len) = query.len().checked_add(1) else {
            column.clear();
            return (WeightedCost::TOP, WeightedCost::TOP);
        };
        column.resize(column_len, WeightedCost::TOP);
        column.fill(WeightedCost::TOP);
        if depth == 0 {
            return (WeightedCost::TOP, previous_carry.unwrap_or(0.0));
        }

        let target_index = depth - 1;
        let prefix_bound = interval_prefix_step(
            previous_carry.unwrap_or(WeightedCost::ZERO),
            current_interval,
            target_index,
            self.band,
            plan,
        );
        let start = depth.saturating_sub(self.band).max(1);
        let end = depth.saturating_add(self.band).min(query.len());
        let previous_cell = |row: usize| {
            if depth == 1 && row == 0 {
                WeightedCost::ZERO
            } else {
                previous.get(row).copied().unwrap_or(WeightedCost::TOP)
            }
        };

        let mut column_bound = WeightedCost::TOP;
        if start <= end {
            for row in start..=end {
                let predecessor = previous_cell(row - 1)
                    .min(previous_cell(row))
                    .min(column[row - 1]);
                let deviation =
                    interval_dist(query[row - 1], current_interval.0, current_interval.1);
                column[row] = WeightedCost::combine(predecessor, square(deviation));
                column_bound = column_bound.min(column[row]);
            }
        }

        (column_bound.max(prefix_bound), prefix_bound)
    }

    #[inline]
    fn prefix_lower_bound(
        &self,
        _query: &[f64],
        current_interval: (f64, f64),
        previous_carry: Option<Self::Carry>,
        depth: usize,
        plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        let Some(target_index) = depth.checked_sub(1) else {
            return WeightedCost::ZERO;
        };
        interval_prefix_step(
            previous_carry.unwrap_or(WeightedCost::ZERO),
            current_interval,
            target_index,
            self.band,
            plan,
        )
    }

    #[inline]
    fn exact_with_cutoff(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: Cost<Self>,
    ) -> Option<Cost<Self>> {
        self.distance_squared_with_cutoff(query, candidate, cutoff)
    }

    #[inline]
    fn candidate_lower_bound(
        &self,
        _query: &[f64],
        candidate: &[f64],
        plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        lb_keogh_squared(candidate, self.band, plan)
    }

    #[inline]
    fn plan(&self, query: &[f64]) -> Self::QueryPlan {
        keogh_envelopes(query, self.band).unwrap_or_default()
    }

    #[inline]
    fn empty_pair_cost(&self) -> Cost<Self> {
        WeightedCost::ZERO
    }

    #[inline]
    fn empty_vs_nonempty_cost(&self, _nonempty: &[f64]) -> Cost<Self> {
        WeightedCost::TOP
    }
}

/// Exact banded-DTW index with root-distance public scores.
///
/// [`ElasticTransducer`] operates on squared costs so its interval columns and
/// LB_Keogh share one additive domain. This wrapper squares public thresholds
/// and square-roots exact results at the API boundary.
#[derive(Debug)]
pub struct DtwTransducer<V: Eq + Hash + Clone = usize> {
    inner: ElasticTransducer<DtwKernel, V>,
}

impl<V> DtwTransducer<V>
where
    V: Eq + Hash + Clone,
{
    /// Create an empty exact banded-DTW index.
    pub fn new(quantization: QuantizationConfig, config: DtwConfig) -> Self {
        Self {
            inner: ElasticTransducer::new(quantization, config),
        }
    }

    /// Required band configuration.
    #[inline]
    pub fn config(&self) -> &DtwConfig {
        self.inner.kernel()
    }

    /// Quantization configuration used by the trie.
    #[inline]
    pub fn quant_config(&self) -> &QuantizationConfig {
        self.inner.quant_config()
    }

    /// Number of indexed references.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether no references are indexed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Insert or replace an identifier's full-precision series.
    #[inline]
    pub fn insert(&mut self, value: V, series: &[f64]) -> bool {
        self.inner.insert(value, series)
    }

    /// Remove an identifier.
    #[inline]
    pub fn remove(&mut self, value: V) -> bool {
        self.inner.remove(value)
    }

    /// Retrieve an indexed full-precision original.
    #[inline]
    pub fn get_original(&self, value: &V) -> Option<&[f64]> {
        self.inner.get_original(value)
    }

    /// Exact range search in root-distance units.
    pub fn search_range(&self, query: &[f64], max_distance: f64) -> Vec<(V, f64)> {
        if max_distance.is_nan() || max_distance < 0.0 {
            return Vec::new();
        }
        self.inner
            .search_range(query, squared_cutoff(max_distance))
            .into_iter()
            .map(|(value, squared)| (value, squared.sqrt()))
            .collect()
    }

    /// Exact k-nearest-neighbour search in root-distance units.
    pub fn search_knn(&self, query: &[f64], k: usize, initial_threshold: f64) -> Vec<(V, f64)> {
        self.inner
            .search_knn(query, k, squared_cutoff(initial_threshold))
            .into_iter()
            .map(|(value, squared)| (value, squared.sqrt()))
            .collect()
    }

    /// Exact k-nearest-neighbour search with observational pruning counters.
    ///
    /// Distances use the public square-root scale; counters describe the
    /// squared-cost traversal and do not influence its decisions.
    pub fn search_knn_with_stats(
        &self,
        query: &[f64],
        k: usize,
        initial_threshold: f64,
    ) -> (Vec<(V, f64)>, ElasticSearchStats) {
        let (results, stats) =
            self.inner
                .search_knn_with_stats(query, k, squared_cutoff(initial_threshold));
        (
            results
                .into_iter()
                .map(|(value, squared)| (value, squared.sqrt()))
                .collect(),
            stats,
        )
    }
}

impl DtwTransducer<usize> {
    /// Build an index assigning each reference its slice index.
    pub fn from_series(
        quantization: QuantizationConfig,
        config: DtwConfig,
        series: &[Vec<f64>],
    ) -> Self {
        Self {
            inner: ElasticTransducer::from_series(quantization, config, series),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn reference_squared(x: &[f64], y: &[f64], band: usize) -> f64 {
        if !series_is_finite(x) || !series_is_finite(y) {
            return f64::INFINITY;
        }
        match (x.is_empty(), y.is_empty()) {
            (true, true) => return 0.0,
            (true, false) | (false, true) => return f64::INFINITY,
            (false, false) => {}
        }
        let mut matrix = vec![vec![f64::INFINITY; y.len() + 1]; x.len() + 1];
        matrix[0][0] = 0.0;
        for i in 1..=x.len() {
            for j in 1..=y.len() {
                if i.abs_diff(j) <= band {
                    matrix[i][j] = square(x[i - 1] - y[j - 1])
                        + matrix[i - 1][j]
                            .min(matrix[i - 1][j - 1])
                            .min(matrix[i][j - 1]);
                }
            }
        }
        matrix[x.len()][y.len()]
    }

    fn scalar_column(query: &[f64], target: &[f64], band: usize) -> Vec<f64> {
        let mut previous = vec![f64::INFINITY; query.len() + 1];
        previous[0] = 0.0;
        for (target_index, target_value) in target.iter().enumerate() {
            let depth = target_index + 1;
            let mut column = vec![f64::INFINITY; query.len() + 1];
            let start = depth.saturating_sub(band).max(1);
            let end = depth.saturating_add(band).min(query.len());
            if start <= end {
                for row in start..=end {
                    column[row] = square(query[row - 1] - *target_value)
                        + previous[row - 1].min(previous[row]).min(column[row - 1]);
                }
            }
            previous = column;
        }
        previous
    }

    #[test]
    fn boundaries_band_and_root_units_are_explicit() {
        let dtw = DtwConfig::new(1);
        assert_eq!(dtw.distance(&[], &[]), 0.0);
        assert_eq!(dtw.distance(&[], &[1.0]), f64::INFINITY);
        assert_eq!(dtw.distance(&[0.0, 1.0], &[0.0, 2.0]), 1.0);
        assert_eq!(dtw.distance(&[0.0], &[0.0, 0.0, 0.0]), f64::INFINITY);
        assert_eq!(DtwConfig::new(2).distance(&[0.0], &[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(dtw.distance_squared(&[0.0, 1.0], &[0.0, 2.0]), 1.0);
        assert_eq!(dtw.distance_with_cutoff(&[0.0], &[2.0], 2.0), Some(2.0));
        assert_eq!(dtw.distance_with_cutoff(&[0.0], &[2.0], 1.0), None);
        assert_eq!(dtw.distance_with_cutoff(&[0.0], &[2.0], -1.0), None);
        assert_eq!(dtw.distance(&[f64::NAN], &[0.0]), f64::INFINITY);
    }

    #[test]
    fn triangle_inequality_has_a_concrete_counterexample() {
        let dtw = DtwConfig::new(1);
        let x = [0.0];
        let y = [1.0];
        let z = [1.0, 1.0];

        assert_eq!(dtw.distance(&x, &y), 1.0);
        assert_eq!(dtw.distance(&y, &z), 0.0);
        assert_eq!(dtw.distance_squared(&x, &z), 2.0);
        assert!(dtw.distance(&x, &z) > dtw.distance(&x, &y) + dtw.distance(&y, &z));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn optimized_banded_dp_matches_independent_full_matrix(
            x in prop::collection::vec(-20i16..=20, 0..10),
            y in prop::collection::vec(-20i16..=20, 0..10),
            band in 0usize..10,
            cutoff in 0u16..=100,
        ) {
            let x: Vec<f64> = x.into_iter().map(f64::from).collect();
            let y: Vec<f64> = y.into_iter().map(f64::from).collect();
            let config = DtwConfig::new(band);
            let actual = config.distance_squared(&x, &y);
            prop_assert_eq!(actual, reference_squared(&x, &y, band));
            let cutoff = f64::from(cutoff);
            let expected = WeightedCost::within(actual, square(cutoff)).then_some(actual.sqrt());
            prop_assert_eq!(config.distance_with_cutoff(&x, &y, cutoff), expected);
        }

        #[test]
        fn interval_columns_are_admissible_and_point_bins_are_exact(
            query in prop::collection::vec(-12i16..=12, 1..9),
            target in prop::collection::vec(-12i16..=12, 1..9),
            radii in prop::collection::vec(0u8..=4, 1..9),
            band in 0usize..10,
        ) {
            let query: Vec<f64> = query.into_iter().map(f64::from).collect();
            let target: Vec<f64> = target.into_iter().map(f64::from).collect();
            let kernel = DtwConfig::new(band);
            let plan = kernel.plan(&query);
            let mut relaxed_previous = vec![f64::INFINITY; query.len() + 1];
            let mut point_previous = vec![f64::INFINITY; query.len() + 1];
            let mut relaxed_carry = None;
            let mut point_carry = None;

            for (depth_index, value) in target.iter().enumerate() {
                let radius = f64::from(radii[depth_index % radii.len()]);
                let mut relaxed = Vec::new();
                let mut point = Vec::new();
                let (_, next_relaxed_carry) = kernel.step_column(
                    &relaxed_previous,
                    &query,
                    (*value - radius, *value + radius),
                    relaxed_carry,
                    depth_index + 1,
                    &plan,
                    &mut relaxed,
                );
                let (_, next_point_carry) = kernel.step_column(
                    &point_previous,
                    &query,
                    (*value, *value),
                    point_carry,
                    depth_index + 1,
                    &plan,
                    &mut point,
                );

                let center = scalar_column(&query, &target[..=depth_index], band);
                let low_target: Vec<_> = target[..=depth_index]
                    .iter()
                    .enumerate()
                    .map(|(index, center)| *center - f64::from(radii[index % radii.len()]))
                    .collect();
                let high_target: Vec<_> = target[..=depth_index]
                    .iter()
                    .enumerate()
                    .map(|(index, center)| *center + f64::from(radii[index % radii.len()]))
                    .collect();
                let low = scalar_column(&query, &low_target, band);
                let high = scalar_column(&query, &high_target, band);

                prop_assert_eq!(&point, &center);
                for row in 0..relaxed.len() {
                    prop_assert!(relaxed[row] <= center[row] + WeightedCost::EPSILON);
                    prop_assert!(relaxed[row] <= low[row] + WeightedCost::EPSILON);
                    prop_assert!(relaxed[row] <= high[row] + WeightedCost::EPSILON);
                }
                relaxed_previous = relaxed;
                point_previous = point;
                relaxed_carry = Some(next_relaxed_carry);
                point_carry = Some(next_point_carry);
            }
        }

        #[test]
        fn interval_square_cost_is_leaf_exact(
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
                .map(|value| square(scalar - value))
                .fold(f64::INFINITY, f64::min);
            prop_assert_eq!(square(interval_dist(scalar, lo, hi)), brute);
        }

        #[test]
        fn keogh_candidate_and_incremental_prefix_bounds_are_admissible(
            query in prop::collection::vec(-10i16..=10, 1..9),
            candidate in prop::collection::vec(-10i16..=10, 1..9),
            band in 0usize..10,
        ) {
            let query: Vec<f64> = query.into_iter().map(f64::from).collect();
            let candidate: Vec<f64> = candidate.into_iter().map(f64::from).collect();
            let kernel = DtwConfig::new(band);
            let exact = kernel.distance_squared(&query, &candidate);
            let plan = kernel.plan(&query);
            let candidate_bound = kernel.candidate_lower_bound(&query, &candidate, &plan);
            prop_assert!(candidate_bound <= exact + WeightedCost::EPSILON);

            let mut prefix_bound = WeightedCost::ZERO;
            for (index, value) in candidate.iter().enumerate() {
                prefix_bound = interval_prefix_step(
                    prefix_bound,
                    (*value, *value),
                    index,
                    band,
                    &plan,
                );
                prop_assert!(prefix_bound <= exact + WeightedCost::EPSILON);
            }
        }

        #[test]
        fn symmetry_and_non_negativity_hold_without_assuming_metricity(
            x in prop::collection::vec(-10i16..=10, 0..9),
            y in prop::collection::vec(-10i16..=10, 0..9),
            band in 0usize..10,
        ) {
            let x: Vec<f64> = x.into_iter().map(f64::from).collect();
            let y: Vec<f64> = y.into_iter().map(f64::from).collect();
            let dtw = DtwConfig::new(band);
            let dxy = dtw.distance(&x, &y);
            let dyx = dtw.distance(&y, &x);
            prop_assert_eq!(dxy, dyx);
            prop_assert!(dxy >= 0.0);
        }
    }
}
