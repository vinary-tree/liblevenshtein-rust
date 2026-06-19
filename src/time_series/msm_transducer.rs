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
//!   the stored **full-precision** original with
//!   [`MsmConfig::distance_with_cutoff`]; only genuine matches are emitted.
//!
//! See `msm_interval`'s property tests for the admissibility proofs and the
//! tests at the bottom of this file for the end-to-end exactness gates.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::{Dictionary, DictionaryNode, MappedDictionaryNode};

use super::encoding::QuantizationConfig;
use super::msm::MsmConfig;
use super::msm_interval::{column_lower_bound, step_interval_column_into, COST_EPSILON};

struct KnnQueueNode<N> {
    lower_bound: f64,
    sequence: usize,
    depth: usize,
    node: N,
    column: Vec<f64>,
    last_interval: Option<(f64, f64)>,
}

impl<N> PartialEq for KnnQueueNode<N> {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound.to_bits() == other.lower_bound.to_bits() && self.sequence == other.sequence
    }
}

impl<N> Eq for KnnQueueNode<N> {}

impl<N> PartialOrd for KnnQueueNode<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<N> Ord for KnnQueueNode<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .lower_bound
            .total_cmp(&self.lower_bound)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// An exact MSM similarity index over quantized reference series.
///
/// `V` is the identifier associated with each reference (default `usize`).
/// Multiple references that quantize to the same byte key are all retained and
/// individually verified, so quantization collisions never silently drop a
/// result.
#[derive(Debug)]
pub struct MsmTransducer<V: Eq + std::hash::Hash + Clone = usize> {
    /// Prefix-sharing trie over the u8-quantized reference sequences. The
    /// stored value is the final bucket id for all references that quantize to
    /// that byte key.
    dawg: DynamicDawg<usize>,
    /// Quantization configuration (defines the per-bin intervals).
    quant: QuantizationConfig,
    /// MSM configuration for the exact verification step.
    msm: MsmConfig,
    /// Precomputed quantization-bin intervals for hot trie-edge traversal.
    bin_bounds: Vec<(f64, f64)>,
    /// Final bucket id → all reference ids sharing that quantized key.
    buckets: Vec<Vec<V>>,
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
        let bin_bounds = (0..quant.num_bins)
            .map(|bin| quant.bin_bounds(bin))
            .collect();
        Self {
            dawg: DynamicDawg::new(),
            quant,
            msm,
            bin_bounds,
            buckets: Vec::new(),
            originals: HashMap::new(),
        }
    }

    /// Insert a reference series under identifier `value`.
    ///
    /// Returns `true` if `value` was not previously present.
    pub fn insert(&mut self, value: V, series: &[f64]) -> bool {
        let key = self.quant.encode_u8(series);
        let bucket_id = match self.dawg.get_bytes_value(&key) {
            Some(existing) => existing,
            None => {
                let bucket_id = self.buckets.len();
                self.buckets.push(Vec::new());
                let inserted = self.dawg.insert_bytes_with_value(&key, bucket_id);
                debug_assert!(inserted, "new MSM bucket key should insert exactly once");
                bucket_id
            }
        };
        self.buckets[bucket_id].push(value.clone());
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
        if self.is_empty() {
            return Vec::new();
        }
        if m == 0 {
            return self.search_empty_query(tau);
        }
        let root = self.dawg.root();
        let mut columns = vec![vec![f64::INFINITY; m + 1]];
        let mut out: Vec<(V, f64)> = Vec::new();
        self.walk_range(&root, 0, None, query, tau, &mut out, &mut columns);

        // Defensive dedup by id (keep the smallest distance), then sort.
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let mut seen: HashMap<V, ()> = HashMap::with_capacity(out.len());
        out.retain(|(v, _)| seen.insert(v.clone(), ()).is_none());
        out
    }

    fn search_empty_query(&self, tau: f64) -> Vec<(V, f64)> {
        let mut out: Vec<(V, f64)> = self
            .originals
            .iter()
            .filter_map(|(id, original)| {
                self.msm
                    .distance_with_cutoff(&[], original, tau)
                    .map(|exact| (id.clone(), exact))
            })
            .collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out
    }

    #[inline]
    fn bin_bounds_for(&self, unit: u8) -> (f64, f64) {
        self.bin_bounds
            .get(unit as usize)
            .copied()
            .unwrap_or_else(|| self.quant.bin_bounds(unit as u32))
    }

    /// Recursive interval-pruned trie walk used by [`Self::search_range`].
    ///
    /// `node` sits at depth `depth` (target elements consumed); `col` is its
    /// interval DP column; `last_interval` is the bin interval of the element
    /// consumed to reach `node` (`None` at the root). Generic over the node type
    /// so the same logic applies to any byte-unit dictionary backend.
    #[allow(clippy::too_many_arguments)]
    fn walk_range<N>(
        &self,
        node: &N,
        depth: usize,
        last_interval: Option<(f64, f64)>,
        query: &[f64],
        tau: f64,
        out: &mut Vec<(V, f64)>,
        columns: &mut Vec<Vec<f64>>,
    ) where
        N: DictionaryNode<Unit = u8> + MappedDictionaryNode<Value = usize>,
    {
        let m = query.len();

        // A final node at depth >= 1 marks a complete reference of length
        // `depth`; col[m] is the admissible lower bound on its MSM distance.
        let col_at_query_end = columns[depth][m];
        if depth >= 1 && node.is_final() && col_at_query_end <= tau + COST_EPSILON {
            if let Some(bucket_id) = node.value() {
                let ids = &self.buckets[bucket_id];
                for id in ids {
                    if let Some(orig) = self.originals.get(id) {
                        if let Some(exact) = self.msm.distance_with_cutoff(query, orig, tau) {
                            if exact <= tau + COST_EPSILON {
                                out.push((id.clone(), exact));
                            }
                        }
                    }
                }
            }
        }

        for (unit, child) in node.edges() {
            let child_depth = depth + 1;
            if columns.len() <= child_depth {
                columns.resize_with(child_depth + 1, Vec::new);
            }
            let (lo, hi) = self.bin_bounds_for(unit);
            let child_lower_bound = {
                let (prev_columns, child_columns) = columns.split_at_mut(child_depth);
                step_interval_column_into(
                    &prev_columns[depth],
                    query,
                    (lo, hi),
                    last_interval,
                    self.msm.c,
                    &mut child_columns[0],
                );
                column_lower_bound(&child_columns[0])
            };
            if child_lower_bound <= tau + COST_EPSILON {
                self.walk_range(
                    &child,
                    child_depth,
                    Some((lo, hi)),
                    query,
                    tau,
                    out,
                    columns,
                );
            }
        }
    }

    /// Exact k-nearest-neighbor search by MSM distance.
    ///
    /// Uses a single best-first trie traversal keyed by the admissible
    /// interval-column lower bound. Once `k` exact candidates have been found,
    /// every queued subtree whose lower bound exceeds the current kth exact
    /// distance is safely pruned. `initial_threshold` is retained for API
    /// compatibility; exactness and result ordering do not depend on it.
    pub fn search_knn(&self, query: &[f64], k: usize, initial_threshold: f64) -> Vec<(V, f64)> {
        let _ = initial_threshold;
        if k == 0 || self.is_empty() {
            return Vec::new();
        }
        if query.is_empty() {
            return self.search_empty_query(0.0).into_iter().take(k).collect();
        }

        let m = query.len();
        let mut best: Vec<(V, f64)> = Vec::with_capacity(k.min(self.len()));
        let mut kth_distance = f64::INFINITY;
        let mut sequence = 0usize;
        let mut queue = BinaryHeap::new();
        queue.push(KnnQueueNode {
            lower_bound: 0.0,
            sequence,
            depth: 0,
            node: self.dawg.root(),
            column: vec![f64::INFINITY; m + 1],
            last_interval: None,
        });

        while let Some(current) = queue.pop() {
            if best.len() >= k && current.lower_bound > kth_distance + COST_EPSILON {
                break;
            }

            if current.depth >= 1
                && current.node.is_final()
                && current.column[m] <= kth_distance + COST_EPSILON
            {
                if let Some(bucket_id) = current.node.value() {
                    let ids = &self.buckets[bucket_id];
                    for id in ids {
                        if let Some(orig) = self.originals.get(id) {
                            if let Some(exact) =
                                self.msm.distance_with_cutoff(query, orig, kth_distance)
                            {
                                Self::insert_knn_result(&mut best, id.clone(), exact, k);
                                kth_distance = Self::knn_cutoff(&best, k);
                            }
                        }
                    }
                }
            }

            for (unit, child) in current.node.edges() {
                let (lo, hi) = self.bin_bounds_for(unit);
                let mut child_column = Vec::with_capacity(m + 1);
                step_interval_column_into(
                    &current.column,
                    query,
                    (lo, hi),
                    current.last_interval,
                    self.msm.c,
                    &mut child_column,
                );
                let lower_bound = column_lower_bound(&child_column);
                if best.len() < k || lower_bound <= kth_distance + COST_EPSILON {
                    sequence = sequence.wrapping_add(1);
                    queue.push(KnnQueueNode {
                        lower_bound,
                        sequence,
                        depth: current.depth + 1,
                        node: child,
                        column: child_column,
                        last_interval: Some((lo, hi)),
                    });
                }
            }
        }

        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        best.truncate(k);
        best
    }

    fn insert_knn_result(best: &mut Vec<(V, f64)>, id: V, distance: f64, k: usize) {
        if let Some((_, existing)) = best.iter_mut().find(|(existing_id, _)| existing_id == &id) {
            if distance < *existing {
                *existing = distance;
            }
        } else {
            best.push((id, distance));
        }
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        if best.len() > k {
            best.truncate(k);
        }
    }

    fn knn_cutoff(best: &[(V, f64)], k: usize) -> f64 {
        if best.len() >= k {
            best[k - 1].1
        } else {
            f64::INFINITY
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
