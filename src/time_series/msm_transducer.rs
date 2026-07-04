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
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};

use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::{Dictionary, DictionaryNode, MappedDictionaryNode};

use super::encoding::QuantizationConfig;
use super::msm::{series_values_are_finite, MsmConfig};
use super::msm_interval::{
    interval_column_len, step_interval_column_into_with_bound, COST_EPSILON,
};

const DEFAULT_RESULT_BUFFER_CAPACITY: usize = 64;

type BucketLocation = (usize, usize);

#[inline]
fn take_sequence(counter: &mut usize) -> Option<usize> {
    let sequence = *counter;
    *counter = sequence.checked_add(1)?;
    Some(sequence)
}

#[inline]
fn next_sequence(counter: &mut usize) -> Option<usize> {
    *counter = counter.checked_add(1)?;
    Some(*counter)
}

#[inline]
fn byte_unit_index(unit: u8) -> usize {
    usize::from(unit)
}

#[inline]
fn byte_unit_bin(unit: u8) -> u32 {
    u32::from(unit)
}

#[derive(Debug)]
struct StoredSeries {
    series: Vec<f64>,
    bucket_location: BucketLocation,
}

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

#[derive(Clone)]
struct KnnBestResult<V> {
    distance: f64,
    sequence: usize,
    value: V,
}

impl<V> PartialEq for KnnBestResult<V> {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.sequence == other.sequence
    }
}

impl<V> Eq for KnnBestResult<V> {}

impl<V> PartialOrd for KnnBestResult<V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<V> Ord for KnnBestResult<V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.sequence.cmp(&other.sequence))
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
    /// Reference id → original series and bucket slot (for exact verification and O(1) upserts).
    originals: HashMap<V, StoredSeries>,
}

struct RangeWalkContext<'a, V> {
    query: &'a [f64],
    tau: f64,
    out: &'a mut Vec<(V, f64)>,
    columns: &'a mut Vec<Vec<f64>>,
    column_width: usize,
}

impl<V: Eq + std::hash::Hash + Clone> MsmTransducer<V> {
    /// Create an empty transducer.
    ///
    /// Uses a byte-quantized trie, matching [`super::TimeSeriesIndex`] and
    /// [`super::HybridSearchIndex`]. Quantizers wider than 256 bins are
    /// coarsened to a byte-compatible 256-bin config over the same value range.
    pub fn new(quant: QuantizationConfig, msm: MsmConfig) -> Self {
        let quant = quant.into_u8_compatible();
        let msm = msm.normalized();
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
                self.buckets.push(Vec::with_capacity(1));
                let inserted = self.dawg.insert_bytes_with_value(&key, bucket_id);
                debug_assert!(inserted, "new MSM bucket key should insert exactly once");
                bucket_id
            }
        };
        let is_new = !self.originals.contains_key(&value);

        let bucket_location = match self
            .originals
            .get(&value)
            .map(|stored| stored.bucket_location)
        {
            Some((old_bucket_id, old_slot)) if old_bucket_id == bucket_id => (bucket_id, old_slot),
            Some((old_bucket_id, old_slot)) => {
                if self.remove_from_bucket(&value, old_bucket_id, old_slot) {
                    self.release_empty_bucket_storage(old_bucket_id);
                }
                self.push_to_bucket(value.clone(), bucket_id)
            }
            None => self.push_to_bucket(value.clone(), bucket_id),
        };

        self.originals.insert(
            value,
            StoredSeries {
                series: series.to_vec(),
                bucket_location,
            },
        );
        is_new
    }

    fn push_to_bucket(&mut self, value: V, bucket_id: usize) -> BucketLocation {
        let slot = self.buckets[bucket_id].len();
        self.buckets[bucket_id].push(value);
        (bucket_id, slot)
    }

    fn remove_from_bucket(&mut self, value: &V, bucket_id: usize, slot: usize) -> bool {
        let (moved, became_empty) = {
            let bucket = &mut self.buckets[bucket_id];
            debug_assert!(bucket.get(slot).is_some_and(|stored| stored == value));
            let removed = bucket.swap_remove(slot);
            debug_assert!(&removed == value);
            (bucket.get(slot).cloned(), bucket.is_empty())
        };
        if let Some(moved) = moved {
            if let Some(stored) = self.originals.get_mut(&moved) {
                stored.bucket_location = (bucket_id, slot);
            }
        }
        became_empty
    }

    fn release_empty_bucket_storage(&mut self, bucket_id: usize) {
        debug_assert!(self.buckets[bucket_id].is_empty());
        self.buckets[bucket_id] = Vec::new();
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
        self.originals
            .get(value)
            .map(|stored| stored.series.as_slice())
    }

    /// Remove a reference id from the index.
    ///
    /// This drops the stored full-precision series and removes the id from its
    /// quantized candidate bucket. Empty buckets remain addressable from the
    /// DAWG because this index uses arbitrary byte keys and `DynamicDawg` only
    /// exposes string-key deletion, but their backing storage is released and
    /// they produce no candidates during exact search.
    ///
    /// # Returns
    ///
    /// `true` if the reference id was present and removed, `false` otherwise.
    pub fn remove(&mut self, value: V) -> bool {
        let Some(stored) = self.originals.remove(&value) else {
            return false;
        };

        let (bucket_id, slot) = stored.bucket_location;
        if self.remove_from_bucket(&value, bucket_id, slot) {
            self.release_empty_bucket_storage(bucket_id);
        }
        true
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
        if !series_values_are_finite(query) {
            return self.search_non_finite_query(tau);
        }
        let root = self.dawg.root();
        let Some(column_width) = interval_column_len(m) else {
            return Vec::new();
        };
        let mut columns = vec![vec![f64::INFINITY; column_width]];
        let mut out: Vec<(V, f64)> =
            Vec::with_capacity(self.len().min(DEFAULT_RESULT_BUFFER_CAPACITY));
        let mut ctx = RangeWalkContext {
            query,
            tau,
            out: &mut out,
            columns: &mut columns,
            column_width,
        };
        self.walk_range(&root, 0, None, &mut ctx);

        Self::finish_range_results(out)
    }

    fn search_empty_query(&self, tau: f64) -> Vec<(V, f64)> {
        if tau.is_nan() || tau < -COST_EPSILON {
            return Vec::new();
        }

        if tau.is_finite() {
            return self.empty_series_results(None);
        }

        if !tau.is_sign_positive() {
            return Vec::new();
        }

        // Enumerate ids via the deterministic `buckets` store (a `Vec` with
        // stable slot order) rather than the `originals` `HashMap` (whose
        // `RandomState` randomizes iteration order per process). `sort_by` is a
        // stable sort, so equal-distance ties keep this reproducible bucket
        // order instead of emerging in a per-run-random order.
        let mut out: Vec<(V, f64)> = Vec::with_capacity(self.len());
        for id in self.ids_in_bucket_order() {
            let distance = match self.originals.get(id) {
                Some(stored) if stored.series.is_empty() => 0.0,
                Some(_) => f64::INFINITY,
                None => continue,
            };
            out.push((id.clone(), distance));
        }
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out
    }

    fn search_non_finite_query(&self, tau: f64) -> Vec<(V, f64)> {
        if !tau.is_infinite() || !tau.is_sign_positive() {
            return Vec::new();
        }

        // Every id is returned at distance +∞, so there is no distance to sort
        // by; iterate the deterministic `buckets` store (not the randomized
        // `originals` `HashMap`) so the tie order is stable across runs.
        let mut out: Vec<(V, f64)> = Vec::with_capacity(self.len());
        out.extend(
            self.ids_in_bucket_order()
                .cloned()
                .map(|id| (id, f64::INFINITY)),
        );
        out
    }

    /// Iterate every indexed reference id in deterministic bucket/slot order.
    ///
    /// `buckets` is a `Vec<Vec<V>>` whose slot order is fixed by insertion, so
    /// this yields a reproducible sequence across process runs — unlike
    /// iterating the `originals` `HashMap`, whose default `RandomState` seeds a
    /// different iteration order per process. Every inserted id lives in exactly
    /// one bucket slot (see `push_to_bucket`/`remove_from_bucket`), so this
    /// visits each id exactly once, matching `originals.keys()` as a set.
    fn ids_in_bucket_order(&self) -> impl Iterator<Item = &V> {
        self.buckets.iter().flatten()
    }

    fn empty_series_results(&self, limit: Option<usize>) -> Vec<(V, f64)> {
        let root = self.dawg.root();
        let Some(bucket_id) = root.is_final().then(|| root.value()).flatten() else {
            return Vec::new();
        };

        let ids = &self.buckets[bucket_id];
        let result_len = limit.map_or(ids.len(), |limit| ids.len().min(limit));
        let mut out = Vec::with_capacity(result_len);
        out.extend(ids.iter().take(result_len).map(|id| (id.clone(), 0.0)));
        out
    }

    #[inline]
    fn bin_bounds_for(&self, unit: u8) -> (f64, f64) {
        self.bin_bounds
            .get(byte_unit_index(unit))
            .copied()
            .unwrap_or_else(|| self.quant.bin_bounds(byte_unit_bin(unit)))
    }

    /// Recursive interval-pruned trie walk used by [`Self::search_range`].
    ///
    /// `node` sits at depth `depth` (target elements consumed); `col` is its
    /// interval DP column; `last_interval` is the bin interval of the element
    /// consumed to reach `node` (`None` at the root). Generic over the node type
    /// so the same logic applies to any byte-unit dictionary backend.
    fn walk_range<N>(
        &self,
        node: &N,
        depth: usize,
        last_interval: Option<(f64, f64)>,
        ctx: &mut RangeWalkContext<'_, V>,
    ) where
        N: DictionaryNode<Unit = u8> + MappedDictionaryNode<Value = usize>,
    {
        let m = ctx.query.len();
        let split_merge_cost = self.msm.split_merge_cost();

        // A final node marks a complete reference of length `depth`. The root
        // final represents an empty reference; for non-empty queries its column
        // is all +∞, so it is admitted only by a +∞ threshold.
        let col_at_query_end = ctx.columns[depth][m];
        if node.is_final() && col_at_query_end <= ctx.tau + COST_EPSILON {
            if let Some(bucket_id) = node.value() {
                let ids = &self.buckets[bucket_id];
                for id in ids {
                    if let Some(stored) = self.originals.get(id) {
                        if let Some(exact) =
                            self.msm
                                .distance_with_cutoff(ctx.query, &stored.series, ctx.tau)
                        {
                            if exact <= ctx.tau + COST_EPSILON {
                                ctx.out.push((id.clone(), exact));
                            }
                        }
                    }
                }
            }
        }

        for (unit, child) in node.edges() {
            let Some(child_depth) = depth.checked_add(1) else {
                return;
            };
            if ctx.columns.len() <= child_depth {
                let Some(required_depth_count) = child_depth.checked_add(1) else {
                    return;
                };
                let column_width = ctx.column_width;
                ctx.columns
                    .resize_with(required_depth_count, || Vec::with_capacity(column_width));
            }
            let (lo, hi) = self.bin_bounds_for(unit);
            let child_lower_bound = {
                let (prev_columns, child_columns) = ctx.columns.split_at_mut(child_depth);
                step_interval_column_into_with_bound(
                    &prev_columns[depth],
                    ctx.query,
                    (lo, hi),
                    last_interval,
                    split_merge_cost,
                    &mut child_columns[0],
                )
            };
            if child_lower_bound <= ctx.tau + COST_EPSILON {
                self.walk_range(&child, child_depth, Some((lo, hi)), ctx);
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
            return self.empty_series_results(Some(k));
        }
        if !series_values_are_finite(query) {
            return Vec::new();
        }

        let m = query.len();
        let Some(column_width) = interval_column_len(m) else {
            return Vec::new();
        };
        let mut best: BinaryHeap<KnnBestResult<V>> = BinaryHeap::with_capacity(k.min(self.len()));
        let mut kth_distance = f64::INFINITY;
        let mut sequence = 0usize;
        let mut result_sequence = 0usize;
        let split_merge_cost = self.msm.split_merge_cost();
        let mut queue = BinaryHeap::with_capacity(1);
        queue.push(KnnQueueNode {
            lower_bound: 0.0,
            sequence,
            depth: 0,
            node: self.dawg.root(),
            column: vec![f64::INFINITY; column_width],
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
                        if let Some(stored) = self.originals.get(id) {
                            if let Some(exact) =
                                self.msm
                                    .distance_with_cutoff(query, &stored.series, kth_distance)
                            {
                                let Some(candidate_sequence) = take_sequence(&mut result_sequence)
                                else {
                                    continue;
                                };
                                if Self::push_knn_result(
                                    &mut best,
                                    id.clone(),
                                    exact,
                                    k,
                                    candidate_sequence,
                                ) {
                                    kth_distance = Self::knn_cutoff(&best, k);
                                }
                            }
                        }
                    }
                }
            }

            for (unit, child) in current.node.edges() {
                let (lo, hi) = self.bin_bounds_for(unit);
                let mut child_column = Vec::with_capacity(column_width);
                let lower_bound = step_interval_column_into_with_bound(
                    &current.column,
                    query,
                    (lo, hi),
                    current.last_interval,
                    split_merge_cost,
                    &mut child_column,
                );
                if best.len() < k || lower_bound <= kth_distance + COST_EPSILON {
                    let Some(child_depth) = current.depth.checked_add(1) else {
                        continue;
                    };
                    let Some(child_sequence) = next_sequence(&mut sequence) else {
                        continue;
                    };
                    queue.push(KnnQueueNode {
                        lower_bound,
                        sequence: child_sequence,
                        depth: child_depth,
                        node: child,
                        column: child_column,
                        last_interval: Some((lo, hi)),
                    });
                }
            }
        }

        Self::finish_knn_results(best)
    }

    fn push_knn_result(
        best: &mut BinaryHeap<KnnBestResult<V>>,
        id: V,
        distance: f64,
        k: usize,
        sequence: usize,
    ) -> bool {
        if k == 0 || !distance.is_finite() {
            return false;
        }

        if best.len() == k {
            let Some(worst) = best.peek() else {
                return false;
            };
            if distance.total_cmp(&worst.distance) != Ordering::Less {
                return false;
            }
            best.pop();
        }

        best.push(KnnBestResult {
            distance,
            sequence,
            value: id,
        });
        true
    }

    fn knn_cutoff(best: &BinaryHeap<KnnBestResult<V>>, k: usize) -> f64 {
        if best.len() >= k {
            best.peek().map_or(f64::INFINITY, |entry| entry.distance)
        } else {
            f64::INFINITY
        }
    }

    fn finish_range_results(results: Vec<(V, f64)>) -> Vec<(V, f64)> {
        let mut best_by_id: HashMap<V, (f64, usize)> = HashMap::with_capacity(results.len());
        for (sequence, (value, distance)) in results.into_iter().enumerate() {
            match best_by_id.entry(value) {
                Entry::Vacant(entry) => {
                    entry.insert((distance, sequence));
                }
                Entry::Occupied(mut entry) => {
                    let (best_distance, best_sequence) = entry.get_mut();
                    if distance.total_cmp(best_distance) == Ordering::Less {
                        *best_distance = distance;
                        *best_sequence = sequence;
                    }
                }
            }
        }

        let mut results: Vec<(V, f64, usize)> = best_by_id
            .into_iter()
            .map(|(value, (distance, sequence))| (value, distance, sequence))
            .collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)));
        results
            .into_iter()
            .map(|(value, distance, _)| (value, distance))
            .collect()
    }

    fn finish_knn_results(best: BinaryHeap<KnnBestResult<V>>) -> Vec<(V, f64)> {
        let mut best = best.into_vec();
        best.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.sequence.cmp(&b.sequence))
        });
        best.into_iter()
            .map(|entry| (entry.value, entry.distance))
            .collect()
    }
}

impl MsmTransducer<usize> {
    /// Build a transducer from a slice of reference series, assigning each the
    /// id equal to its index. Convenience mirror of
    /// [`super::TimeSeriesIndex::from_series`].
    pub fn from_series(quant: QuantizationConfig, msm: MsmConfig, series: &[Vec<f64>]) -> Self {
        let mut idx = Self::new(quant, msm);
        idx.originals.reserve(series.len());
        idx.buckets.reserve(series.len());
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
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
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
    fn msm_transducer_empty_and_nonfinite_query_order_is_deterministic() {
        // Several references at *equal* MSM distance (every non-empty reference
        // is at +∞ for an empty or non-finite query). The result order must
        // come from the deterministic `buckets` store, never the randomized
        // `originals` HashMap. Independent index instances seed independent
        // HashMap `RandomState`s, so a HashMap-iterating path would order these
        // ties differently between instances; a bucket-iterating path is stable.
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);

        let build = || {
            let mut idx: MsmTransducer<usize> = MsmTransducer::new(quant.clone(), msm);
            for id in 0..16usize {
                let base = id as f64;
                idx.insert(id, &[1.0 + base, 2.0 + base, 3.0 + base]);
            }
            idx
        };

        let empty_query: Vec<f64> = Vec::new();
        let nonfinite_query = [f64::NAN, 1.0, 2.0];

        let reference = build();
        let empty_expected = reference.search_range(&empty_query, f64::INFINITY);
        let nonfinite_expected = reference.search_range(&nonfinite_query, f64::INFINITY);

        // Membership sanity: every id present, all at distance +∞.
        assert_eq!(empty_expected.len(), 16);
        assert!(empty_expected.iter().all(|(_, d)| d.is_infinite()));
        assert_eq!(nonfinite_expected.len(), 16);
        assert!(nonfinite_expected.iter().all(|(_, d)| d.is_infinite()));

        // Determinism across independent instances (fresh RandomState each time).
        for _ in 0..8 {
            let idx = build();
            assert_eq!(
                idx.search_range(&empty_query, f64::INFINITY),
                empty_expected
            );
            assert_eq!(
                idx.search_range(&nonfinite_query, f64::INFINITY),
                nonfinite_expected
            );
        }
    }

    #[test]
    fn constructor_normalizes_invalid_msm_cost() {
        let quant = QuantizationConfig::for_u8(0.0, 10.0);
        let negative: MsmTransducer<usize> =
            MsmTransducer::new(quant.clone(), MsmConfig { c: -1.0 });
        let infinite: MsmTransducer<usize> =
            MsmTransducer::new(quant, MsmConfig { c: f64::INFINITY });

        assert_eq!(negative.msm_config().c, 0.0);
        assert_eq!(*infinite.msm_config(), MsmConfig::default());
    }

    #[test]
    fn bin_bounds_for_uses_explicit_byte_unit_conversions() {
        let quant = QuantizationConfig::for_u8(0.0, 255.0);
        let zero_bounds = quant.bin_bounds(0);
        let max_bounds = quant.bin_bounds(u32::from(u8::MAX));
        let idx: MsmTransducer<usize> = MsmTransducer::new(quant, MsmConfig::new(1.0));

        assert_eq!(byte_unit_index(0), 0);
        assert_eq!(byte_unit_bin(0), 0);
        assert_eq!(byte_unit_index(u8::MAX), usize::from(u8::MAX));
        assert_eq!(byte_unit_bin(u8::MAX), u32::from(u8::MAX));
        assert_eq!(idx.bin_bounds_for(0), zero_bounds);
        assert_eq!(idx.bin_bounds_for(u8::MAX), max_bounds);
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
    fn knn_sequence_helpers_do_not_wrap_at_usize_boundary() {
        let mut result_sequence = usize::MAX - 1;
        assert_eq!(take_sequence(&mut result_sequence), Some(usize::MAX - 1));
        assert_eq!(result_sequence, usize::MAX);
        assert_eq!(take_sequence(&mut result_sequence), None);
        assert_eq!(result_sequence, usize::MAX);

        let mut queue_sequence = usize::MAX - 1;
        assert_eq!(next_sequence(&mut queue_sequence), Some(usize::MAX));
        assert_eq!(queue_sequence, usize::MAX);
        assert_eq!(next_sequence(&mut queue_sequence), None);
        assert_eq!(queue_sequence, usize::MAX);
    }

    #[test]
    fn traversal_column_width_uses_checked_interval_width() {
        assert_eq!(interval_column_len(0), Some(1));
        assert_eq!(interval_column_len(8), Some(9));
        assert_eq!(interval_column_len(usize::MAX), None);
    }

    #[test]
    fn knn_result_heap_maintains_bounded_top_k() {
        let mut best = BinaryHeap::new();
        let mut sequence = 0usize;

        assert!(MsmTransducer::push_knn_result(
            &mut best, 1usize, 5.0, 2, sequence
        ));
        sequence += 1;
        assert!(MsmTransducer::push_knn_result(
            &mut best, 2usize, 3.0, 2, sequence
        ));
        sequence += 1;
        assert!(MsmTransducer::push_knn_result(
            &mut best, 3usize, 4.0, 2, sequence
        ));
        sequence += 1;

        assert_eq!(
            MsmTransducer::finish_knn_results(best.clone()),
            vec![(2, 3.0), (3, 4.0)]
        );

        assert!(!MsmTransducer::push_knn_result(
            &mut best, 2usize, 7.0, 2, sequence
        ));
        sequence += 1;
        assert_eq!(
            MsmTransducer::finish_knn_results(best.clone()),
            vec![(2, 3.0), (3, 4.0)]
        );

        assert!(MsmTransducer::push_knn_result(
            &mut best, 3usize, 2.0, 2, sequence
        ));
        sequence += 1;
        assert_eq!(
            MsmTransducer::finish_knn_results(best.clone()),
            vec![(3, 2.0), (2, 3.0)]
        );

        assert!(!MsmTransducer::push_knn_result(
            &mut best, 4usize, 9.0, 2, sequence
        ));
        assert_eq!(
            MsmTransducer::finish_knn_results(best),
            vec![(3, 2.0), (2, 3.0)]
        );
    }

    #[test]
    fn range_result_finisher_keeps_best_distance_per_id_without_reordering_ties() {
        let results = vec![(7usize, 4.0), (9, 2.0), (7, 1.5), (11, 1.5), (9, 3.0)];

        assert_eq!(
            MsmTransducer::finish_range_results(results),
            vec![(7, 1.5), (11, 1.5), (9, 2.0)]
        );
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

    #[test]
    fn reinserting_same_id_in_same_bucket_updates_original_without_duplicate_membership() {
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);

        assert!(idx.insert(7, &[10.0, 20.0, 30.0]));
        assert!(!idx.insert(7, &[10.01, 20.01, 30.01]));

        assert_eq!(idx.len(), 1);
        assert_eq!(idx.buckets.iter().map(Vec::len).sum::<usize>(), 1);
        let stored = idx.originals.get(&7).unwrap();
        assert_eq!(
            idx.buckets[stored.bucket_location.0][stored.bucket_location.1],
            7
        );
        assert_eq!(idx.get_original(&7), Some(&[10.01, 20.01, 30.01][..]));

        let got = idx.search_range(&[10.01, 20.01, 30.01], 0.0);
        assert_eq!(got, vec![(7, 0.0)]);
    }

    #[test]
    fn reinserting_same_id_in_different_bucket_relocates_membership() {
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);

        assert!(idx.insert(7, &[10.0]));
        assert!(idx.insert(9, &[10.01]));
        assert!(!idx.insert(7, &[90.0]));

        assert_eq!(idx.len(), 2);
        assert_eq!(idx.buckets.iter().map(Vec::len).sum::<usize>(), 2);
        assert_eq!(idx.get_original(&7), Some(&[90.0][..]));
        assert_eq!(idx.get_original(&9), Some(&[10.01][..]));

        let old_bucket = idx.quant.encode_u8(&[10.01]);
        let old_bucket_id = idx.dawg.get_bytes_value(&old_bucket).unwrap();
        assert_eq!(idx.buckets[old_bucket_id], vec![9]);
        assert_eq!(
            idx.originals.get(&9).map(|stored| stored.bucket_location),
            Some((old_bucket_id, 0))
        );

        assert_eq!(idx.search_range(&[10.01], 0.0), vec![(9, 0.0)]);
        assert_eq!(idx.search_range(&[90.0], 0.0), vec![(7, 0.0)]);
    }

    #[test]
    fn remove_value_clears_membership_and_releases_empty_bucket_storage() {
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);

        assert!(idx.insert(7, &[10.0, 20.0]));
        let bucket_id = idx.originals.get(&7).unwrap().bucket_location.0;
        assert_eq!(idx.search_range(&[10.0, 20.0], 0.0), vec![(7, 0.0)]);

        assert!(idx.remove(7));
        assert!(!idx.remove(7));

        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        assert_eq!(idx.get_original(&7), None);
        assert!(idx.buckets[bucket_id].is_empty());
        assert_eq!(idx.buckets[bucket_id].capacity(), 0);
        assert!(idx.search_range(&[10.0, 20.0], 0.0).is_empty());
        assert!(idx.search_knn(&[10.0, 20.0], 1, 1.0).is_empty());
    }

    #[test]
    fn remove_value_preserves_swapped_bucket_location() {
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);

        assert!(idx.insert(7, &[10.0]));
        assert!(idx.insert(9, &[10.01]));

        let shared_bucket = idx.originals.get(&7).unwrap().bucket_location.0;
        assert_eq!(
            idx.originals.get(&9).unwrap().bucket_location.0,
            shared_bucket
        );

        assert!(idx.remove(7));

        assert_eq!(idx.buckets[shared_bucket], vec![9]);
        assert_eq!(
            idx.originals.get(&9).map(|stored| stored.bucket_location),
            Some((shared_bucket, 0))
        );
        assert_eq!(idx.search_range(&[10.01], 0.0), vec![(9, 0.0)]);

        assert!(!idx.insert(9, &[90.0]));
        assert!(idx.buckets[shared_bucket].is_empty());
        assert_eq!(idx.buckets[shared_bucket].capacity(), 0);
        assert!(idx.search_range(&[10.01], 0.0).is_empty());
        assert_eq!(idx.search_range(&[90.0], 0.0), vec![(9, 0.0)]);
    }

    #[test]
    fn large_quantizer_is_coarsened_without_losing_exactness() {
        let series = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.2, 2.2, 3.2],
            vec![9.0, 9.0, 9.0],
        ];
        let quant = QuantizationConfig::for_u16(0.0, 10.0);
        let msm = MsmConfig::new(1.0);
        let idx = MsmTransducer::from_series(quant, msm, &series);
        let query = vec![1.1, 2.1, 3.1];

        assert_eq!(idx.quant_config().num_bins, 256);
        assert_eq!(
            idx.search_range(&query, 1.0),
            brute_range(&series, &query, &msm, 1.0)
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
