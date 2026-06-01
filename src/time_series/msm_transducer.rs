//! Exact MSM-bounded retrieval over a trie of quantized reference series.
//!
//! [`MsmTransducer`] indexes reference time series as quantized byte sequences in
//! a [`DynamicDawg`] (sharing prefixes, exactly like [`super::TimeSeriesIndex`]),
//! but answers similarity queries by walking the trie with an **interval-relaxed
//! MSM dynamic program** ([`super::msm_interval`]) instead of the lossy
//! Levenshtein approximate filter used by [`super::HybridSearchIndex`].
//!
//! # Guarantee
//!
//! [`MsmTransducer::search_range`] returns *exactly* the set
//! `{ id : MSM(query, reference_id) ≤ τ }` — no false negatives and no false
//! positives — and [`MsmTransducer::search_knn`] returns exactly the `k`
//! nearest references by MSM distance. This is the soundness the approximate
//! quantized-edit-distance filter cannot provide (it can drop a true MSM-near
//! neighbor whose quantized form is several bin-edits away).
//!
//! # How it stays exact while pruning
//!
//! Descending one trie edge consumes one target element, known only up to its
//! quantization bin `[lo, hi]`. Each trie node at depth `d` therefore carries
//! the MSM DP column for the query against the first `d` target elements,
//! computed with [`super::msm_interval::step_interval_column`] using
//! *admissible lower-bound* per-element costs. Two consequences make the walk
//! exact:
//!
//! * **Sound pruning (no false negatives).** The minimum live cell of a node's
//!   column ([`super::msm_interval::column_lower_bound`]) lower-bounds the true
//!   MSM distance of *every* reference reachable below it, because any DP path
//!   to a deeper final must cross this column and all later MSM costs are
//!   non-negative. So a subtree whose bound exceeds `τ` (or the running k-th
//!   best) contains no admissible match and is safely skipped.
//! * **Exact verification (no false positives).** At each final node whose
//!   column lower bound is within threshold, the candidate is re-scored against
//!   the stored **full-precision** original with [`MsmConfig::distance`]; only
//!   genuine matches are emitted.
//!
//! See `msm_interval`'s property tests for the admissibility proofs and the
//! tests at the bottom of this file for the end-to-end exactness gates.

use std::cmp::Ordering;
use std::collections::HashMap;

use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::{Dictionary, DictionaryNode};

use super::encoding::QuantizationConfig;
use super::msm::MsmConfig;
use super::msm_interval::{column_lower_bound, step_interval_column, COST_EPSILON};

/// An exact MSM similarity index over quantized reference series.
///
/// `V` is the identifier associated with each reference (default `usize`).
/// Multiple references that quantize to the same byte key are all retained and
/// individually verified, so quantization collisions never silently drop a
/// result.
#[derive(Debug)]
pub struct MsmTransducer<V: Eq + std::hash::Hash + Clone = usize> {
    /// Prefix-sharing trie over the u8-quantized reference sequences. The
    /// stored value is unused; final resolution goes through `key_to_values`
    /// so that quantization collisions (distinct ids, identical key) are all
    /// recovered.
    dawg: DynamicDawg<usize>,
    /// Quantization configuration (defines the per-bin intervals).
    quant: QuantizationConfig,
    /// MSM configuration for the exact verification step.
    msm: MsmConfig,
    /// Quantized key → all reference ids sharing that key.
    key_to_values: HashMap<Vec<u8>, Vec<V>>,
    /// Reference id → full-precision original series (for exact verification).
    originals: HashMap<V, Vec<f64>>,
}

impl<V: Eq + std::hash::Hash + Clone> MsmTransducer<V> {
    /// Create an empty transducer.
    ///
    /// `quant.num_bins` must be ≤ 256 (byte-quantized trie), matching
    /// [`super::TimeSeriesIndex`]/[`super::HybridSearchIndex`].
    pub fn new(quant: QuantizationConfig, msm: MsmConfig) -> Self {
        assert!(
            quant.num_bins <= 256,
            "MsmTransducer uses a byte-quantized trie; num_bins ({}) must be <= 256",
            quant.num_bins
        );
        Self {
            dawg: DynamicDawg::new(),
            quant,
            msm,
            key_to_values: HashMap::new(),
            originals: HashMap::new(),
        }
    }

    /// Insert a reference series under identifier `value`.
    ///
    /// Returns `true` if `value` was not previously present.
    pub fn insert(&mut self, value: V, series: &[f64]) -> bool {
        let key = self.quant.encode_u8(series);
        // The stored value is unused; resolution is via `key_to_values`.
        let _ = self
            .dawg
            .insert_bytes_with_value(&key, self.originals.len());
        self.key_to_values
            .entry(key)
            .or_default()
            .push(value.clone());
        self.originals.insert(value, series.to_vec()).is_none()
    }

    /// Number of indexed references.
    #[inline]
    pub fn len(&self) -> usize {
        self.originals.len()
    }

    /// Whether the index holds no references.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.originals.is_empty()
    }

    /// The quantization configuration.
    #[inline]
    pub fn quant_config(&self) -> &QuantizationConfig {
        &self.quant
    }

    /// The MSM configuration.
    #[inline]
    pub fn msm_config(&self) -> &MsmConfig {
        &self.msm
    }

    /// Retrieve the original series for a reference id, if present.
    #[inline]
    pub fn get_original(&self, value: &V) -> Option<&[f64]> {
        self.originals.get(value).map(|v| v.as_slice())
    }

    /// Exact range search: return every reference whose MSM distance to `query`
    /// is `≤ tau`, paired with that exact distance, sorted ascending.
    ///
    /// One prefix-amortized trie traversal with admissible interval pruning,
    /// followed by exact full-precision verification at survivors. Returns no
    /// false negatives and no false positives.
    pub fn search_range(&self, query: &[f64], tau: f64) -> Vec<(V, f64)> {
        let m = query.len();
        if m == 0 || self.is_empty() {
            return Vec::new();
        }
        let root = self.dawg.root();
        let root_col = vec![f64::INFINITY; m + 1];
        let mut out: Vec<(V, f64)> = Vec::new();
        let mut path: Vec<u8> = Vec::with_capacity(32);
        self.walk_range(&root, 0, &root_col, None, &mut path, query, tau, &mut out);

        // Defensive dedup by id (keep the smallest distance), then sort.
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let mut seen: HashMap<V, ()> = HashMap::with_capacity(out.len());
        out.retain(|(v, _)| seen.insert(v.clone(), ()).is_none());
        out
    }

    /// Recursive interval-pruned trie walk used by [`Self::search_range`].
    ///
    /// `node` sits at depth `depth` (target elements consumed); `col` is its
    /// interval DP column; `last_interval` is the bin interval of the element
    /// consumed to reach `node` (`None` at the root). Generic over the node type
    /// so the same logic applies to any byte-unit dictionary backend.
    #[allow(clippy::too_many_arguments)]
    fn walk_range<N: DictionaryNode<Unit = u8>>(
        &self,
        node: &N,
        depth: usize,
        col: &[f64],
        last_interval: Option<(f64, f64)>,
        path: &mut Vec<u8>,
        query: &[f64],
        tau: f64,
        out: &mut Vec<(V, f64)>,
    ) {
        let m = query.len();

        // A final node at depth >= 1 marks a complete reference of length
        // `depth`; col[m] is the admissible lower bound on its MSM distance.
        if depth >= 1 && node.is_final() && col[m] <= tau + COST_EPSILON {
            if let Some(ids) = self.key_to_values.get(path) {
                for id in ids {
                    if let Some(orig) = self.originals.get(id) {
                        let exact = self.msm.distance(query, orig);
                        if exact <= tau + COST_EPSILON {
                            out.push((id.clone(), exact));
                        }
                    }
                }
            }
        }

        for (unit, child) in node.edges() {
            let (lo, hi) = self.quant.bin_bounds(unit as u32);
            let new_col = step_interval_column(col, query, (lo, hi), last_interval, self.msm.c);
            if column_lower_bound(&new_col) <= tau + COST_EPSILON {
                path.push(unit);
                self.walk_range(
                    &child,
                    depth + 1,
                    &new_col,
                    Some((lo, hi)),
                    path,
                    query,
                    tau,
                    out,
                );
                path.pop();
            }
        }
    }

    /// Exact k-nearest-neighbor search by MSM distance.
    ///
    /// Layers exact [`Self::search_range`] with geometric threshold growth
    /// (the idiom used by [`super::HybridSearchIndex::search_knn`]): start at
    /// `initial_threshold`, doubling until at least `k` exact matches are found
    /// or the search space is exhausted. Because each range pass is exact, the
    /// `k` returned are exactly the `k` smallest MSM distances.
    pub fn search_knn(&self, query: &[f64], k: usize, initial_threshold: f64) -> Vec<(V, f64)> {
        if k == 0 || self.is_empty() || query.is_empty() {
            return Vec::new();
        }
        let mut threshold = if initial_threshold > 0.0 {
            initial_threshold
        } else {
            1.0
        };
        loop {
            let results = self.search_range(query, threshold);
            if results.len() >= k {
                return results.into_iter().take(k).collect();
            }
            if threshold >= 1e10 {
                // Search space exhausted; return everything found.
                return results;
            }
            threshold *= 2.0;
        }
    }
}

impl MsmTransducer<usize> {
    /// Build a transducer from a slice of reference series, assigning each the
    /// id equal to its index. Convenience mirror of
    /// [`super::TimeSeriesIndex::from_series`].
    pub fn from_series(quant: QuantizationConfig, msm: MsmConfig, series: &[Vec<f64>]) -> Self {
        let mut idx = Self::new(quant, msm);
        for (i, s) in series.iter().enumerate() {
            idx.insert(i, s);
        }
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Brute-force reference set: every series within `tau`, sorted ascending.
    fn brute_range(
        series: &[Vec<f64>],
        query: &[f64],
        msm: &MsmConfig,
        tau: f64,
    ) -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = series
            .iter()
            .enumerate()
            .map(|(i, s)| (i, msm.distance(query, s)))
            .filter(|(_, d)| *d <= tau + 1e-9)
            .collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        v
    }

    #[test]
    fn range_matches_brute_force_small() {
        let series = vec![
            vec![10.0, 20.0, 30.0],
            vec![11.0, 21.0, 29.0],
            vec![50.0, 60.0, 70.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![90.0, 10.0, 50.0],
        ];
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let idx = MsmTransducer::from_series(quant.clone(), msm, &series);

        let query = vec![12.0, 22.0, 31.0];
        for &tau in &[0.0, 2.0, 5.0, 10.0, 50.0, 1000.0] {
            let got = idx.search_range(&query, tau);
            let want = brute_range(&series, &query, &msm, tau);
            let got_ids: Vec<usize> = got.iter().map(|(i, _)| *i).collect();
            let want_ids: Vec<usize> = want.iter().map(|(i, _)| *i).collect();
            assert_eq!(got_ids, want_ids, "tau={tau}: id sets differ");
            for ((gi, gd), (wi, wd)) in got.iter().zip(want.iter()) {
                assert_eq!(gi, wi);
                assert!((gd - wd).abs() < 1e-9, "distance mismatch at id {gi}");
            }
        }
    }

    #[test]
    fn knn_matches_brute_force_small() {
        let series = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![5.0, 6.0, 7.0],
            vec![1.1, 2.1, 2.9],
            vec![9.0, 8.0, 7.0],
        ];
        let quant = QuantizationConfig::for_u8(0.0, 10.0);
        let msm = MsmConfig::new(0.5);
        let idx = MsmTransducer::from_series(quant.clone(), msm, &series);
        let query = vec![1.2, 2.2, 3.1];

        for k in 1..=series.len() {
            let got = idx.search_knn(&query, k, 0.5);
            let mut want = brute_range(&series, &query, &msm, f64::INFINITY);
            want.truncate(k);
            let got_d: Vec<f64> = got.iter().map(|(_, d)| *d).collect();
            let want_d: Vec<f64> = want.iter().map(|(_, d)| *d).collect();
            assert_eq!(got_d.len(), want_d.len(), "k={k} count");
            for (g, w) in got_d.iter().zip(want_d.iter()) {
                assert!((g - w).abs() < 1e-9, "k={k}: {g} != {w}");
            }
        }
    }

    #[test]
    fn collisions_are_all_recovered() {
        // Two distinct ids whose series quantize identically must both be
        // returned (no silent drop on key collision).
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);
        idx.insert(7, &[10.0, 20.0, 30.0]);
        idx.insert(9, &[10.2, 20.2, 30.2]); // same bins as id 7 at 256 bins over [0,100]
        let got = idx.search_range(&[10.1, 20.1, 30.1], 5.0);
        let ids: std::collections::HashSet<u32> = got.iter().map(|(v, _)| *v).collect();
        assert!(
            ids.contains(&7) && ids.contains(&9),
            "both colliding ids must appear: {ids:?}"
        );
    }

    proptest! {
        // The exactness gate (A6): range search equals brute force for random
        // reference sets, queries, c, and thresholds.
        #[test]
        fn prop_range_exact(
            series in prop::collection::vec(
                prop::collection::vec(0.0f64..100.0, 1..7),
                1..15,
            ),
            query in prop::collection::vec(0.0f64..100.0, 1..7),
            c in 0.1f64..3.0,
            tau in 0.0f64..40.0,
        ) {
            let quant = QuantizationConfig::for_u8(0.0, 100.0);
            let msm = MsmConfig::new(c);
            let idx = MsmTransducer::from_series(quant, msm, &series);

            let got = idx.search_range(&query, tau);
            let want = brute_range(&series, &query, &msm, tau);

            // Same set of ids (dedup brute by best-distance id, since duplicate
            // identical series get distinct indices but identical distance).
            let got_ids: std::collections::HashSet<usize> = got.iter().map(|(i, _)| *i).collect();
            let want_ids: std::collections::HashSet<usize> = want.iter().map(|(i, _)| *i).collect();
            prop_assert_eq!(got_ids, want_ids);

            // Every emitted distance is the exact MSM distance.
            for (i, d) in &got {
                let exact = msm.distance(&query, &series[*i]);
                prop_assert!((d - exact).abs() < 1e-9);
            }
        }

        // k-NN exactness: returned distances equal the k smallest brute-force
        // distances (compared as a multiset of distances, robust to ties).
        #[test]
        fn prop_knn_exact(
            series in prop::collection::vec(
                prop::collection::vec(0.0f64..50.0, 1..6),
                1..12,
            ),
            query in prop::collection::vec(0.0f64..50.0, 1..6),
            c in 0.2f64..2.0,
            k in 1usize..6,
        ) {
            let quant = QuantizationConfig::for_u8(0.0, 50.0);
            let msm = MsmConfig::new(c);
            let idx = MsmTransducer::from_series(quant, msm, &series);

            let got = idx.search_knn(&query, k, 1.0);
            let mut want = brute_range(&series, &query, &msm, f64::INFINITY);
            want.truncate(k);

            prop_assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                // Distances must match position-for-position (both sorted asc).
                prop_assert!((g.1 - w.1).abs() < 1e-9, "got {} want {}", g.1, w.1);
            }
        }
    }
}
