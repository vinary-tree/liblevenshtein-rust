//! Opt-in approximate MSM kNN via compact feature ranking plus exact reranking.
//!
//! [`ApproxMsmIndex`] is intentionally separate from [`super::MsmTransducer`]:
//! it does not provide exact recall guarantees. It ranks all indexed series by a
//! low-dimensional Piecewise Aggregate Approximation (PAA) feature vector, keeps
//! a bounded candidate pool, then computes exact MSM only for that pool.

use std::cmp::Ordering;

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
    /// `segments` and `candidate_limit` must both be positive. The effective
    /// candidate count is `max(candidate_limit, k)` and is capped by index size.
    pub fn new(segments: usize, candidate_limit: usize, msm: MsmConfig) -> Self {
        assert!(segments > 0, "PAA segment count must be positive");
        assert!(candidate_limit > 0, "candidate limit must be positive");
        Self {
            segments,
            candidate_limit,
            msm,
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
        let mut index = Self::new(config);
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
        let query_features = paa_features(query, self.config.segments);
        let mut ranked: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                (
                    idx,
                    feature_score(
                        &entry.features,
                        &query_features,
                        entry.series.len(),
                        query.len(),
                        self.config.msm.c,
                    ),
                )
            })
            .collect();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut exact: Vec<(V, f64)> = ranked
            .into_iter()
            .take(candidate_count)
            .filter_map(|(idx, _)| {
                let entry = &self.entries[idx];
                let distance = self.config.msm.distance(query, &entry.series);
                distance
                    .is_finite()
                    .then(|| (entry.value.clone(), distance))
            })
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        exact.truncate(k);
        exact
    }

    /// The immutable configuration.
    #[inline]
    pub fn config(&self) -> &ApproxMsmConfig {
        &self.config
    }
}

impl ApproxMsmIndex<usize> {
    /// Build an index from a slice of series, assigning each id by position.
    pub fn from_series(config: ApproxMsmConfig, series: &[Vec<f64>]) -> Self {
        Self::from_entries(config, series.iter().enumerate().map(|(idx, s)| (idx, s)))
    }
}

/// Compute a fixed-size Piecewise Aggregate Approximation feature vector.
pub fn paa_features(series: &[f64], segments: usize) -> Vec<f64> {
    assert!(segments > 0, "PAA segment count must be positive");
    if series.is_empty() {
        return vec![0.0; segments];
    }

    let n = series.len();
    (0..segments)
        .map(|segment| {
            let start = segment * n / segments;
            let mut end = ((segment + 1) * n).div_ceil(segments);
            let start = start.min(n - 1);
            end = end.max(start + 1).min(n);
            let width = end - start;
            series[start..end].iter().sum::<f64>() / width as f64
        })
        .collect()
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
    use super::*;

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
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out.truncate(k);
        out
    }

    #[test]
    fn paa_feature_count_is_stable() {
        let features = paa_features(&[1.0, 2.0, 3.0], 8);
        assert_eq!(features.len(), 8);
        assert!(features.iter().all(|v| v.is_finite()));
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
}
