//! Opt-in approximate MSM kNN via compact feature ranking plus exact reranking.
//!
//! [`ApproxMsmIndex`] is intentionally separate from [`super::MsmTransducer`]:
//! it does not provide exact recall guarantees. It ranks all indexed series by a
//! low-dimensional Piecewise Aggregate Approximation (PAA) feature vector, keeps
//! a bounded candidate pool, then computes exact MSM only for that pool.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::msm::MsmConfig;

/// Configuration for [`ApproxMsmIndex`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApproxMsmConfig {
    /// Number of PAA segments per indexed series.
    pub segments: usize,
    /// Maximum number of feature-ranked candidates to exact-rerank.
    pub candidate_limit: usize,
    /// Exact MSM configuration used for reranking.
    pub msm: MsmConfig,
}

impl ApproxMsmConfig {
    /// Create an approximate MSM configuration.
    ///
    /// The effective candidate count is `max(candidate_limit, k)` and is capped
    /// by index size. A `candidate_limit` of zero therefore uses the requested
    /// `k` as the candidate count. A `segments` value of zero disables PAA value
    /// features and ranks candidates by length only before exact reranking.
    pub fn new(segments: usize, candidate_limit: usize, msm: MsmConfig) -> Self {
        Self {
            segments,
            candidate_limit,
            msm: msm.normalized(),
        }
    }
}

impl Default for ApproxMsmConfig {
    fn default() -> Self {
        Self::new(16, 128, MsmConfig::default_cost())
    }
}

#[derive(Debug, Clone)]
struct ApproxEntry<V> {
    value: V,
    series: Vec<f64>,
    features: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct RankedCandidate {
    idx: usize,
    score: f64,
}

#[derive(Debug, Clone, Copy)]
struct ExactCandidate {
    idx: usize,
    distance: f64,
    sequence: usize,
}

impl PartialEq for ExactCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.sequence == other.sequence
    }
}

impl Eq for ExactCandidate {}

impl PartialOrd for ExactCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExactCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

impl PartialEq for RankedCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for RankedCandidate {}

impl PartialOrd for RankedCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

/// Approximate MSM nearest-neighbor index.
///
/// Query latency is reduced by avoiding exact MSM evaluation for every indexed
/// series. Recall depends on whether the PAA candidate generator includes the
/// true neighbors in its candidate pool.
#[derive(Debug, Clone)]
pub struct ApproxMsmIndex<V = usize> {
    config: ApproxMsmConfig,
    entries: Vec<ApproxEntry<V>>,
}

impl<V: Clone> ApproxMsmIndex<V> {
    /// Create an empty approximate index.
    pub fn new(config: ApproxMsmConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
        }
    }

    /// Build an index from `(value, series)` pairs.
    pub fn from_entries<I, S>(config: ApproxMsmConfig, entries: I) -> Self
    where
        I: IntoIterator<Item = (V, S)>,
        S: AsRef<[f64]>,
    {
        let entries = entries.into_iter();
        let mut index = Self::new(config);
        let (lower_bound, upper_bound) = entries.size_hint();
        index.entries.reserve(upper_bound.unwrap_or(lower_bound));

        for (value, series) in entries {
            index.insert(value, series.as_ref());
        }
        index
    }

    /// Insert one series.
    pub fn insert(&mut self, value: V, series: &[f64]) {
        self.entries.push(ApproxEntry {
            value,
            series: series.to_vec(),
            features: paa_features(series, self.config.segments),
        });
    }

    /// Number of indexed series.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Approximate search: feature-rank candidates, then exact-rerank them.
    pub fn search_knn(&self, query: &[f64], k: usize) -> Vec<(V, f64)> {
        if k == 0 || self.entries.is_empty() {
            return Vec::new();
        }

        let candidate_count = self.config.candidate_limit.max(k).min(self.entries.len());
        if candidate_count == self.entries.len() {
            self.exact_rerank(0..self.entries.len(), query, k)
        } else {
            let query_features = paa_features(query, self.config.segments);
            let ranked_candidates =
                self.top_feature_candidates(&query_features, query.len(), candidate_count);
            self.exact_rerank(
                ranked_candidates.into_iter().map(|candidate| candidate.idx),
                query,
                k,
            )
        }
    }

    /// The immutable configuration.
    #[inline]
    pub fn config(&self) -> &ApproxMsmConfig {
        &self.config
    }

    fn top_feature_candidates(
        &self,
        query_features: &[f64],
        query_len: usize,
        candidate_count: usize,
    ) -> Vec<RankedCandidate> {
        debug_assert!(candidate_count > 0);
        debug_assert!(candidate_count < self.entries.len());

        let mut heap = BinaryHeap::with_capacity(candidate_count);

        for (idx, entry) in self.entries.iter().enumerate() {
            let candidate = RankedCandidate {
                idx,
                score: feature_score(
                    &entry.features,
                    query_features,
                    entry.series.len(),
                    query_len,
                    self.config.msm.split_merge_cost(),
                ),
            };

            if heap.len() < candidate_count {
                heap.push(candidate);
            } else if heap
                .peek()
                .is_some_and(|worst| candidate.cmp(worst) == Ordering::Less)
            {
                heap.pop();
                heap.push(candidate);
            }
        }

        heap.into_sorted_vec()
    }

    fn exact_rerank<I>(&self, candidate_indices: I, query: &[f64], k: usize) -> Vec<(V, f64)>
    where
        I: IntoIterator<Item = usize>,
    {
        let result_count = k.min(self.entries.len());
        if result_count == 0 {
            return Vec::new();
        }

        let mut best: BinaryHeap<ExactCandidate> = BinaryHeap::with_capacity(result_count);

        for (sequence, idx) in candidate_indices.into_iter().enumerate() {
            let entry = &self.entries[idx];
            let cutoff = best
                .peek()
                .filter(|_| best.len() >= result_count)
                .map_or(f64::INFINITY, |worst| worst.distance);
            let Some(distance) = self
                .config
                .msm
                .distance_with_cutoff(query, &entry.series, cutoff)
            else {
                continue;
            };
            if !distance.is_finite() {
                continue;
            }

            if best.len() < result_count {
                best.push(ExactCandidate {
                    idx,
                    distance,
                    sequence,
                });
            } else if best
                .peek()
                .is_some_and(|worst| distance.total_cmp(&worst.distance) == Ordering::Less)
            {
                best.pop();
                best.push(ExactCandidate {
                    idx,
                    distance,
                    sequence,
                });
            }
        }

        best.into_sorted_vec()
            .into_iter()
            .map(|candidate| {
                let entry = &self.entries[candidate.idx];
                (entry.value.clone(), candidate.distance)
            })
            .collect()
    }
}

impl ApproxMsmIndex<usize> {
    /// Build an index from a slice of series, assigning each id by position.
    pub fn from_series(config: ApproxMsmConfig, series: &[Vec<f64>]) -> Self {
        Self::from_entries(config, series.iter().enumerate())
    }
}

/// Compute a fixed-size Piecewise Aggregate Approximation feature vector.
pub fn paa_features(series: &[f64], segments: usize) -> Vec<f64> {
    if segments == 0 {
        return Vec::new();
    }
    if series.is_empty() {
        return vec![0.0; segments];
    }

    let n = series.len();
    (0..segments)
        .map(|segment| {
            let (start, end) =
                paa_segment_bounds(segment, n, segments).unwrap_or_else(|| (n - 1, n));
            let width = end - start;
            series[start..end].iter().sum::<f64>() / width as f64
        })
        .collect()
}

fn paa_segment_bounds(segment: usize, n: usize, segments: usize) -> Option<(usize, usize)> {
    if n == 0 || segments == 0 {
        return None;
    }

    let start = segment.checked_mul(n)?.checked_div(segments)?.min(n - 1);
    let segment_end = segment.checked_add(1)?;
    let numerator = segment_end.checked_mul(n)?;
    let rounding = segments.checked_sub(1)?;
    let rounded_end = numerator.checked_add(rounding)?.checked_div(segments)?;
    let min_end = start.checked_add(1)?;
    let end = rounded_end.max(min_end).min(n);
    Some((start, end))
}

fn feature_score(
    lhs: &[f64],
    rhs: &[f64],
    lhs_len: usize,
    rhs_len: usize,
    split_merge_cost: f64,
) -> f64 {
    let value_score = lhs
        .iter()
        .zip(rhs)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f64>();
    let length_delta = lhs_len.abs_diff(rhs_len) as f64 * split_merge_cost;
    value_score + length_delta * length_delta
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;

    #[derive(Debug)]
    struct CloneCountedValue {
        id: usize,
        clone_count: Rc<Cell<usize>>,
    }

    impl Clone for CloneCountedValue {
        fn clone(&self) -> Self {
            self.clone_count
                .set(clone_count_successor(self.clone_count.get()).unwrap_or(usize::MAX));
            Self {
                id: self.id,
                clone_count: Rc::clone(&self.clone_count),
            }
        }
    }

    fn clone_count_successor(count: usize) -> Option<usize> {
        count.checked_add(1)
    }

    fn brute_knn(
        series: &[Vec<f64>],
        query: &[f64],
        msm: MsmConfig,
        k: usize,
    ) -> Vec<(usize, f64)> {
        let mut out: Vec<(usize, f64)> = series
            .iter()
            .enumerate()
            .map(|(idx, s)| (idx, msm.distance(query, s)))
            .filter(|(_, d)| d.is_finite())
            .collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out.truncate(k);
        out
    }

    #[test]
    fn paa_segment_bounds_match_ceiling_partitions() {
        assert_eq!(paa_segment_bounds(0, 10, 4), Some((0, 3)));
        assert_eq!(paa_segment_bounds(1, 10, 4), Some((2, 5)));
        assert_eq!(paa_segment_bounds(2, 10, 4), Some((5, 8)));
        assert_eq!(paa_segment_bounds(3, 10, 4), Some((7, 10)));
    }

    #[test]
    fn paa_segment_bounds_check_overflow() {
        assert_eq!(paa_segment_bounds(0, 0, 4), None);
        assert_eq!(paa_segment_bounds(0, 4, 0), None);
        assert_eq!(paa_segment_bounds(usize::MAX, 1, usize::MAX), None);
        assert_eq!(
            paa_segment_bounds((usize::MAX / 2) + 1, 3, usize::MAX),
            None
        );
    }

    #[test]
    fn clone_count_successor_checks_overflow() {
        assert_eq!(clone_count_successor(0), Some(1));
        assert_eq!(clone_count_successor(usize::MAX), None);
    }

    #[test]
    fn paa_feature_count_is_stable() {
        let features = paa_features(&[1.0, 2.0, 3.0], 8);
        assert_eq!(features.len(), 8);
        assert!(features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn paa_zero_segments_is_empty_feature_vector() {
        assert!(paa_features(&[1.0, 2.0, 3.0], 0).is_empty());
        assert!(paa_features(&[], 0).is_empty());
    }

    #[test]
    fn zero_segments_and_candidate_limit_are_total_configurations() {
        let series = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![10.0, 10.0],
        ];
        let msm = MsmConfig::new(1.0);
        let config = ApproxMsmConfig::new(0, 0, msm);
        let index = ApproxMsmIndex::from_series(config, &series);

        let got = index.search_knn(&[1.0, 2.0, 3.0, 4.0], 2);

        assert_eq!(index.config().segments, 0);
        assert_eq!(index.config().candidate_limit, 0);
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|(_, distance)| distance.is_finite()));
    }

    #[test]
    fn config_normalizes_invalid_msm_cost() {
        let negative = ApproxMsmConfig::new(4, 8, MsmConfig { c: -3.0 });
        let infinite = ApproxMsmConfig::new(4, 8, MsmConfig { c: f64::INFINITY });

        assert_eq!(negative.msm.c, 0.0);
        assert_eq!(infinite.msm, MsmConfig::default());
    }

    #[test]
    fn exact_rerank_clones_only_returned_values() {
        let clone_count = Rc::new(Cell::new(0));
        let values = [100.0, 90.0, 80.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let entries = values.into_iter().enumerate().map(|(id, value)| {
            (
                CloneCountedValue {
                    id,
                    clone_count: Rc::clone(&clone_count),
                },
                vec![value],
            )
        });
        let config = ApproxMsmConfig::new(1, values.len(), MsmConfig::new(1.0));
        let index = ApproxMsmIndex::from_entries(config, entries);

        let got = index.search_knn(&[0.0], 3);

        let ids: Vec<_> = got.iter().map(|(value, _)| value.id).collect();
        assert_eq!(ids, vec![3, 4, 5]);
        assert_eq!(clone_count.get(), got.len());
    }

    #[test]
    fn full_candidate_pool_matches_brute_force() {
        let series = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.1, 2.1, 2.9, 4.1],
            vec![10.0, 11.0, 12.0, 13.0],
            vec![3.0, 3.0, 3.0, 3.0],
        ];
        let msm = MsmConfig::new(1.0);
        let index =
            ApproxMsmIndex::from_series(ApproxMsmConfig::new(4, series.len(), msm), &series);
        let query = vec![1.0, 2.0, 3.0, 4.0];

        assert_eq!(
            index.search_knn(&query, 3),
            brute_knn(&series, &query, msm, 3)
        );
    }

    #[test]
    fn exact_duplicate_is_recovered_from_small_candidate_pool() {
        let series = vec![
            vec![20.0, 20.0, 20.0, 20.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![80.0, 80.0, 80.0, 80.0],
        ];
        let msm = MsmConfig::new(1.0);
        let index = ApproxMsmIndex::from_series(ApproxMsmConfig::new(4, 1, msm), &series);

        let got = index.search_knn(&series[1], 1);
        assert_eq!(got, vec![(1, 0.0)]);
    }

    #[test]
    fn huge_k_is_capped_to_index_len() {
        let series = vec![vec![3.0], vec![1.0], vec![2.0]];
        let msm = MsmConfig::new(1.0);
        let index = ApproxMsmIndex::from_series(ApproxMsmConfig::new(1, 0, msm), &series);

        let got = index.search_knn(&[1.0], usize::MAX);

        assert_eq!(got, vec![(1, 0.0), (2, 1.0), (0, 2.0)]);
    }
}
