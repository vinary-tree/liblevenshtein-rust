//! Opt-in approximate MSM kNN via compact feature ranking plus exact reranking.
//!
//! [`ApproxMsmIndex`] is intentionally separate from [`super::MsmTransducer`]:
//! it does not provide exact recall guarantees. It ranks all indexed series by a
//! low-dimensional Piecewise Aggregate Approximation (PAA) feature vector, keeps
//! a bounded candidate pool, then computes exact MSM only for that pool.
//!
//! Use [`ApproxMsmIndex::search_knn_bounded`] at evidence-bearing boundaries.
//! Its result type makes exhaustive exact reranking, heuristic advice, request
//! rejection, and resource failure disjoint. In particular, an empty
//! [`ApproxMsmSearchOutcome::Advisory`] result is not evidence that the index
//! contains no neighbor. The legacy [`ApproxMsmIndex::search_knn`] convenience
//! method returns an untagged vector and is therefore advisory only.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::mem::size_of;

use super::bounded::{
    IncompleteReason, Operand, ResourceKind, ResourceLedger, ResourceLimits, ResourceUsage,
    TemporalValidationError,
};
use super::msm::MsmConfig;

/// Exact-coverage accounting for an approximate MSM query.
///
/// `exact_reranked` counts candidates for which exact cutoff-aware MSM reached
/// a mathematically decisive result. Recall is proved only when that count is
/// equal to `indexed_entries`; inspecting every PAA feature is not a substitute
/// for exact reranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApproxMsmCoverage {
    /// Number of entries in the captured immutable index view.
    pub indexed_entries: usize,
    /// Number of entries admitted by the candidate generator.
    pub candidate_entries: usize,
    /// Number of admitted entries decided by exact MSM reranking.
    pub exact_reranked: usize,
}

impl ApproxMsmCoverage {
    /// Whether every indexed entry received an exact MSM decision.
    #[inline]
    pub fn proves_recall(&self) -> bool {
        self.exact_reranked == self.indexed_entries
            && self.candidate_entries == self.indexed_entries
    }
}

/// One neighbor emitted by strict approximate MSM search.
///
/// The value is borrowed from the index so the bounded query never invokes an
/// unconstrained user-defined [`Clone`] implementation. `distance` is always a
/// finite exact MSM distance, even when the containing search is advisory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApproxMsmNeighbor<'a, V> {
    /// Stable insertion position used for deterministic tie-breaking.
    pub index: usize,
    /// Value associated with the indexed series.
    pub value: &'a V,
    /// Exact full-precision MSM distance.
    pub distance: f64,
}

/// Exact candidates emitted so far together with their coverage certificate.
#[derive(Debug, Clone, PartialEq)]
pub struct ApproxMsmSearchResult<'a, V> {
    /// Neighbors ordered by `(distance, insertion position)`.
    pub neighbors: Vec<ApproxMsmNeighbor<'a, V>>,
    /// Exact-reranking coverage established by this result.
    pub coverage: ApproxMsmCoverage,
}

/// Strict outcome of bounded approximate MSM search.
///
/// Only [`Self::Exhaustive`] proves kNN recall or absence. [`Self::Advisory`]
/// contains exact distances for every emitted neighbor but makes no recall
/// claim. [`Self::Incomplete`] cannot be confused with either successful state
/// and records allocation, arithmetic, numeric, or resource failure explicitly.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub enum ApproxMsmSearchOutcome<'a, V> {
    /// Every indexed entry was decided by exact MSM reranking.
    Exhaustive {
        /// Complete exact kNN result.
        result: ApproxMsmSearchResult<'a, V>,
        /// Charged cumulative and peak resources.
        usage: ResourceUsage,
    },
    /// A heuristic candidate subset was exactly reranked.
    Advisory {
        /// Exact neighbors within the selected subset.
        result: ApproxMsmSearchResult<'a, V>,
        /// Charged cumulative and peak resources.
        usage: ResourceUsage,
    },
    /// The bounded operation did not finish its admitted candidate pool.
    Incomplete {
        /// Exact subset available before the failure, when any exists.
        partial: Option<ApproxMsmSearchResult<'a, V>>,
        /// Fail-closed reason the operation stopped.
        reason: IncompleteReason,
        /// Charged cumulative and peak resources at the stop point.
        usage: ResourceUsage,
    },
}

impl<V> ApproxMsmSearchOutcome<'_, V> {
    /// Whether this outcome proves exact kNN recall over the captured index.
    #[inline]
    pub fn proves_recall(&self) -> bool {
        matches!(self, Self::Exhaustive { result, .. } if result.coverage.proves_recall())
    }

    /// Resource accounting for every outcome variant.
    #[inline]
    pub fn usage(&self) -> ResourceUsage {
        match self {
            Self::Exhaustive { usage, .. }
            | Self::Advisory { usage, .. }
            | Self::Incomplete { usage, .. } => *usage,
        }
    }
}

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
    /// Construct a strict configuration without normalizing invalid MSM cost.
    pub fn try_new(
        segments: usize,
        candidate_limit: usize,
        msm: MsmConfig,
    ) -> Result<Self, TemporalValidationError> {
        validate_config(msm)?;
        Ok(Self {
            segments,
            candidate_limit,
            msm,
        })
    }

    /// Create an approximate MSM configuration.
    ///
    /// The effective candidate count is `max(candidate_limit, k)` and is capped
    /// by index size. A `candidate_limit` of zero therefore uses the requested
    /// `k` as the candidate count. A `segments` value of zero disables PAA value
    /// features and ranks candidates by length only before exact reranking.
    /// This legacy constructor normalizes an invalid MSM cost. Evidence-bearing
    /// callers should use [`Self::try_new`] so configuration errors remain
    /// explicit.
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
    finite_invariant: bool,
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
}

impl PartialEq for ExactCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.idx == other.idx
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
            .then_with(|| self.idx.cmp(&other.idx))
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

impl<V> ApproxMsmIndex<V> {
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
        let features = paa_features(series, self.config.segments);
        let finite_invariant = series.iter().all(|sample| sample.is_finite())
            && features.iter().all(|feature| feature.is_finite());
        self.entries.push(ApproxEntry {
            value,
            series: series.to_vec(),
            features,
            finite_invariant,
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

    /// Legacy advisory search: feature-rank candidates, then exact-rerank them.
    ///
    /// This method predates tagged resource outcomes. An empty vector can mean
    /// no selected candidate had a finite MSM alignment, invalid numeric input,
    /// or allocation/resource failure outside the process contract. It never
    /// proves absence or exact recall, even when its configured candidate pool
    /// happens to cover the current index. Use [`Self::search_knn_bounded`] for
    /// evidence-bearing code.
    pub fn search_knn(&self, query: &[f64], k: usize) -> Vec<(V, f64)>
    where
        V: Clone,
    {
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

    /// Strict bounded approximate kNN with explicit evidence status.
    ///
    /// PAA is used only to choose a candidate set. Every emitted distance is
    /// then recomputed by the exact two-row MSM recurrence over the stored
    /// full-precision series. If the candidate set contains all $`N`$ indexed
    /// entries, the result is [`ApproxMsmSearchOutcome::Exhaustive`]; otherwise
    /// it is [`ApproxMsmSearchOutcome::Advisory`], including when its neighbor
    /// vector is empty. No heuristic score is exposed as an MSM distance.
    ///
    /// All allocation sizes and arithmetic work are checked before exact
    /// reranking starts. The recurrence is iterative and reuses two rows, so its
    /// call-stack use is constant and its retained DP memory is
    /// $`\mathcal{O}(\min(|Q|, \max_i |X_i|))`$. Ties are resolved by stable
    /// insertion position after exact distance.
    pub fn search_knn_bounded<'a>(
        &'a self,
        query: &[f64],
        k: usize,
        limits: ResourceLimits,
    ) -> Result<ApproxMsmSearchOutcome<'a, V>, TemporalValidationError> {
        validate_config(self.config.msm)?;

        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;

        let mut validation_work = query.len();
        for entry in &self.entries {
            if entry.series.len() > limits.max_series_len
                || entry.features.len() != self.config.segments
                || !entry.finite_invariant
            {
                return Ok(incomplete_outcome(
                    ledger,
                    IncompleteReason::InvalidStoredData,
                    None,
                ));
            }
            validation_work = match validation_work.checked_add(1) {
                Some(work) => work,
                None => {
                    return Ok(incomplete_outcome(
                        ledger,
                        IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::WorkUnits,
                        },
                        None,
                    ));
                }
            };
        }
        if let Err(reason) = ledger.charge(ResourceKind::WorkUnits, validation_work) {
            return Ok(incomplete_outcome(ledger, reason, None));
        }
        let indexed_entries = self.entries.len();
        if k == 0 || indexed_entries == 0 {
            let coverage = ApproxMsmCoverage {
                indexed_entries,
                candidate_entries: 0,
                exact_reranked: 0,
            };
            let result = ApproxMsmSearchResult {
                neighbors: Vec::new(),
                coverage,
            };
            return Ok(if coverage.proves_recall() {
                ApproxMsmSearchOutcome::Exhaustive {
                    result,
                    usage: ledger.usage(),
                }
            } else {
                ApproxMsmSearchOutcome::Advisory {
                    result,
                    usage: ledger.usage(),
                }
            });
        }

        let candidate_entries = self.config.candidate_limit.max(k).min(indexed_entries);
        let exhaustive_pool = candidate_entries == indexed_entries;

        let ranked = if exhaustive_pool {
            None
        } else {
            let ranking_work =
                match query
                    .len()
                    .checked_add(self.config.segments)
                    .and_then(|work| {
                        indexed_entries
                            .checked_mul(self.config.segments)
                            .and_then(|scores| work.checked_add(scores))
                    }) {
                    Some(work) => work,
                    None => {
                        return Ok(incomplete_outcome(
                            ledger,
                            IncompleteReason::ArithmeticOverflow {
                                resource: ResourceKind::WorkUnits,
                            },
                            None,
                        ));
                    }
                };
            if let Err(reason) = ledger.charge(ResourceKind::WorkUnits, ranking_work) {
                return Ok(incomplete_outcome(ledger, reason, None));
            }

            let ranking_scratch =
                match checked_bytes::<f64>(self.config.segments).and_then(|bytes| {
                    checked_bytes::<RankedCandidate>(candidate_entries)
                        .and_then(|heap| bytes.checked_add(heap))
                }) {
                    Some(bytes) => bytes,
                    None => {
                        return Ok(incomplete_outcome(
                            ledger,
                            IncompleteReason::ArithmeticOverflow {
                                resource: ResourceKind::ScratchBytes,
                            },
                            None,
                        ));
                    }
                };
            if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, ranking_scratch) {
                return Ok(incomplete_outcome(ledger, reason, None));
            }

            let query_features = match paa_features_strict(query, self.config.segments) {
                Ok(features) => features,
                Err(reason) => return Ok(incomplete_outcome(ledger, reason, None)),
            };
            let mut heap = match binary_heap_with_capacity(candidate_entries) {
                Ok(heap) => heap,
                Err(reason) => return Ok(incomplete_outcome(ledger, reason, None)),
            };
            let actual_ranking_scratch =
                checked_bytes::<f64>(query_features.capacity()).and_then(|bytes| {
                    checked_bytes::<RankedCandidate>(heap.capacity())
                        .and_then(|heap_bytes| bytes.checked_add(heap_bytes))
                });
            let Some(actual_ranking_scratch) = actual_ranking_scratch else {
                return Ok(incomplete_outcome(
                    ledger,
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    },
                    None,
                ));
            };
            if let Err(reason) =
                ledger.observe_peak(ResourceKind::ScratchBytes, actual_ranking_scratch)
            {
                return Ok(incomplete_outcome(ledger, reason, None));
            }
            for (idx, entry) in self.entries.iter().enumerate() {
                let score = match feature_score_strict(
                    &entry.features,
                    &query_features,
                    entry.series.len(),
                    query.len(),
                    self.config.msm.c,
                ) {
                    Ok(score) => score,
                    Err(reason) => return Ok(incomplete_outcome(ledger, reason, None)),
                };
                let candidate = RankedCandidate { idx, score };
                if heap.len() < candidate_entries {
                    heap.push(candidate);
                } else if heap
                    .peek()
                    .is_some_and(|worst| candidate.cmp(worst) == Ordering::Less)
                {
                    heap.pop();
                    heap.push(candidate);
                }
            }
            Some(heap.into_sorted_vec())
        };

        let selected_index = |sequence: usize| {
            ranked
                .as_ref()
                .map_or(sequence, |candidates| candidates[sequence].idx)
        };
        let mut dp_cells = 0usize;
        let mut max_row_len = 0usize;
        for sequence in 0..candidate_entries {
            let entry = &self.entries[selected_index(sequence)];
            let cells = if query.is_empty() || entry.series.is_empty() {
                0
            } else {
                match query.len().checked_mul(entry.series.len()) {
                    Some(cells) => cells,
                    None => {
                        return Ok(incomplete_outcome(
                            ledger,
                            IncompleteReason::ArithmeticOverflow {
                                resource: ResourceKind::DpCells,
                            },
                            None,
                        ));
                    }
                }
            };
            dp_cells = match dp_cells.checked_add(cells) {
                Some(cells) => cells,
                None => {
                    return Ok(incomplete_outcome(
                        ledger,
                        IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::DpCells,
                        },
                        None,
                    ));
                }
            };
            if cells != 0 {
                let Some(row_len) = query.len().min(entry.series.len()).checked_add(1) else {
                    return Ok(incomplete_outcome(
                        ledger,
                        IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::ScratchBytes,
                        },
                        None,
                    ));
                };
                max_row_len = max_row_len.max(row_len);
            }
        }

        let result_capacity = k.min(candidate_entries);
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::Candidates, candidate_entries),
            (ResourceKind::DpCells, dp_cells),
            (ResourceKind::WorkUnits, dp_cells),
            (ResourceKind::Results, result_capacity),
        ]) {
            return Ok(incomplete_outcome(ledger, reason, None));
        }

        let selected_bytes = match ranked.as_ref() {
            Some(candidates) => match checked_bytes::<RankedCandidate>(candidates.capacity()) {
                Some(bytes) => bytes,
                None => {
                    return Ok(incomplete_outcome(
                        ledger,
                        IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::ScratchBytes,
                        },
                        None,
                    ));
                }
            },
            None => 0,
        };
        let scoring_scratch = checked_bytes::<ExactCandidate>(result_capacity)
            .and_then(|best| {
                checked_bytes::<ApproxMsmNeighbor<'a, V>>(result_capacity)
                    .and_then(|output| best.checked_add(output))
            })
            .and_then(|bytes| {
                checked_bytes::<f64>(max_row_len)
                    .and_then(|row| row.checked_mul(2))
                    .and_then(|rows| bytes.checked_add(rows))
            })
            .and_then(|bytes| bytes.checked_add(selected_bytes));
        let Some(scoring_scratch) = scoring_scratch else {
            return Ok(incomplete_outcome(
                ledger,
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
                None,
            ));
        };
        if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, scoring_scratch) {
            return Ok(incomplete_outcome(ledger, reason, None));
        }

        let mut best: BinaryHeap<ExactCandidate> = match binary_heap_with_capacity(result_capacity)
        {
            Ok(heap) => heap,
            Err(reason) => return Ok(incomplete_outcome(ledger, reason, None)),
        };
        let mut output = match vec_with_capacity::<ApproxMsmNeighbor<'a, V>>(result_capacity) {
            Ok(output) => output,
            Err(reason) => return Ok(incomplete_outcome(ledger, reason, None)),
        };
        let mut prev = match vec_with_capacity::<f64>(max_row_len) {
            Ok(row) => row,
            Err(reason) => return Ok(incomplete_outcome(ledger, reason, None)),
        };
        let mut curr = match vec_with_capacity::<f64>(max_row_len) {
            Ok(row) => row,
            Err(reason) => return Ok(incomplete_outcome(ledger, reason, None)),
        };
        let actual_scoring_scratch = checked_bytes::<ExactCandidate>(best.capacity())
            .and_then(|bytes| {
                checked_bytes::<ApproxMsmNeighbor<'a, V>>(output.capacity())
                    .and_then(|output_bytes| bytes.checked_add(output_bytes))
            })
            .and_then(|bytes| {
                checked_bytes::<f64>(prev.capacity())
                    .and_then(|prev_bytes| bytes.checked_add(prev_bytes))
            })
            .and_then(|bytes| {
                checked_bytes::<f64>(curr.capacity())
                    .and_then(|curr_bytes| bytes.checked_add(curr_bytes))
            })
            .and_then(|bytes| bytes.checked_add(selected_bytes));
        let Some(actual_scoring_scratch) = actual_scoring_scratch else {
            return Ok(incomplete_outcome(
                ledger,
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
                None,
            ));
        };
        if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, actual_scoring_scratch)
        {
            return Ok(incomplete_outcome(ledger, reason, None));
        }
        prev.resize(max_row_len, f64::INFINITY);
        curr.resize(max_row_len, f64::INFINITY);

        let mut exact_reranked = 0usize;
        for sequence in 0..candidate_entries {
            let idx = selected_index(sequence);
            let entry = &self.entries[idx];
            let cutoff = best
                .peek()
                .filter(|_| best.len() >= result_capacity)
                .map_or(f64::INFINITY, |worst| worst.distance);
            let distance = match exact_msm_with_cutoff_reusing(
                self.config.msm,
                query,
                &entry.series,
                cutoff,
                &mut prev,
                &mut curr,
            ) {
                Ok(distance) => distance,
                Err(reason) => {
                    append_neighbors(&self.entries, best, &mut output);
                    let partial = (exact_reranked != 0).then_some(ApproxMsmSearchResult {
                        neighbors: output,
                        coverage: ApproxMsmCoverage {
                            indexed_entries,
                            candidate_entries,
                            exact_reranked,
                        },
                    });
                    return Ok(incomplete_outcome(ledger, reason, partial));
                }
            };
            exact_reranked += 1;

            let Some(distance) = distance else {
                continue;
            };
            let candidate = ExactCandidate { idx, distance };
            if best.len() < result_capacity {
                best.push(candidate);
            } else if best
                .peek()
                .is_some_and(|worst| candidate.cmp(worst) == Ordering::Less)
            {
                best.pop();
                best.push(candidate);
            }
        }

        append_neighbors(&self.entries, best, &mut output);
        let coverage = ApproxMsmCoverage {
            indexed_entries,
            candidate_entries,
            exact_reranked,
        };
        let result = ApproxMsmSearchResult {
            neighbors: output,
            coverage,
        };
        Ok(if coverage.proves_recall() {
            ApproxMsmSearchOutcome::Exhaustive {
                result,
                usage: ledger.usage(),
            }
        } else {
            ApproxMsmSearchOutcome::Advisory {
                result,
                usage: ledger.usage(),
            }
        })
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
        V: Clone,
    {
        let result_count = k.min(self.entries.len());
        if result_count == 0 {
            return Vec::new();
        }

        let mut best: BinaryHeap<ExactCandidate> = BinaryHeap::with_capacity(result_count);

        for idx in candidate_indices {
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
                best.push(ExactCandidate { idx, distance });
            } else if best
                .peek()
                .is_some_and(|worst| ExactCandidate { idx, distance }.cmp(worst) == Ordering::Less)
            {
                best.pop();
                best.push(ExactCandidate { idx, distance });
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

fn validate_config(msm: MsmConfig) -> Result<(), TemporalValidationError> {
    if !msm.c.is_finite() || msm.c < 0.0 {
        return Err(TemporalValidationError::InvalidConfiguration(
            "approximate MSM split/merge cost must be finite and nonnegative",
        ));
    }
    Ok(())
}

#[inline]
fn checked_bytes<T>(capacity: usize) -> Option<usize> {
    capacity.checked_mul(size_of::<T>())
}

fn vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, IncompleteReason> {
    let requested = checked_bytes::<T>(capacity).ok_or(IncompleteReason::ArithmeticOverflow {
        resource: ResourceKind::ScratchBytes,
    })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })?;
    Ok(values)
}

fn binary_heap_with_capacity<T>(capacity: usize) -> Result<BinaryHeap<T>, IncompleteReason> {
    let requested = checked_bytes::<T>(capacity).ok_or(IncompleteReason::ArithmeticOverflow {
        resource: ResourceKind::ScratchBytes,
    })?;
    let mut heap = BinaryHeap::new();
    heap.try_reserve_exact(capacity)
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })?;
    Ok(heap)
}

fn paa_features_strict(series: &[f64], segments: usize) -> Result<Vec<f64>, IncompleteReason> {
    let mut features = vec_with_capacity(segments)?;
    if segments == 0 {
        return Ok(features);
    }
    if series.is_empty() {
        features.resize(segments, 0.0);
        return Ok(features);
    }

    let n = series.len();
    for segment in 0..segments {
        let Some((start, end)) = paa_segment_bounds(segment, n, segments) else {
            return Err(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::WorkUnits,
            });
        };
        let mut sum = 0.0;
        for value in &series[start..end] {
            sum += value;
            if !sum.is_finite() {
                return Err(IncompleteReason::NumericOverflow);
            }
        }
        let mean = sum / (end - start) as f64;
        if !mean.is_finite() {
            return Err(IncompleteReason::NumericOverflow);
        }
        features.push(mean);
    }
    Ok(features)
}

fn feature_score_strict(
    lhs: &[f64],
    rhs: &[f64],
    lhs_len: usize,
    rhs_len: usize,
    split_merge_cost: f64,
) -> Result<f64, IncompleteReason> {
    if lhs.len() != rhs.len() {
        return Err(IncompleteReason::InvalidStoredData);
    }
    let mut value_score = 0.0;
    for (left, right) in lhs.iter().zip(rhs) {
        let delta = left - right;
        let squared = delta * delta;
        value_score += squared;
        if !delta.is_finite() || !squared.is_finite() || !value_score.is_finite() {
            return Err(IncompleteReason::NumericOverflow);
        }
    }
    let length_delta = lhs_len.abs_diff(rhs_len) as f64 * split_merge_cost;
    let length_score = length_delta * length_delta;
    let score = value_score + length_score;
    if !length_delta.is_finite() || !length_score.is_finite() || !score.is_finite() {
        return Err(IncompleteReason::NumericOverflow);
    }
    Ok(score)
}

fn exact_msm_with_cutoff_reusing<'rows>(
    config: MsmConfig,
    left: &[f64],
    right: &[f64],
    cutoff: f64,
    mut prev: &'rows mut [f64],
    mut curr: &'rows mut [f64],
) -> Result<Option<f64>, IncompleteReason> {
    let (x, y) = if left.len() >= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    if x.is_empty() {
        return Ok(Some(0.0));
    }
    if y.is_empty() {
        return Ok(None);
    }

    let row_len = y
        .len()
        .checked_add(1)
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })?;
    if row_len > prev.len() || row_len > curr.len() {
        return Err(IncompleteReason::InvalidStoredData);
    }
    prev[..row_len].fill(f64::INFINITY);
    curr[..row_len].fill(f64::INFINITY);
    let effective_cutoff = cutoff;

    prev[1] = (x[0] - y[0]).abs();
    let mut row_min = prev[1];
    for j in 2..=y.len() {
        prev[j] = prev[j - 1] + config.c_func(y[j - 1], x[0], y[j - 2]);
        row_min = row_min.min(prev[j]);
    }
    if row_min > effective_cutoff {
        return Ok(None);
    }

    for i in 2..=x.len() {
        curr[1] = prev[1] + config.c_func(x[i - 1], x[i - 2], y[0]);
        row_min = curr[1];
        for j in 2..=y.len() {
            let move_cost = prev[j - 1] + (x[i - 1] - y[j - 1]).abs();
            let merge_cost = prev[j] + config.c_func(x[i - 1], x[i - 2], y[j - 1]);
            let split_cost = curr[j - 1] + config.c_func(y[j - 1], x[i - 1], y[j - 2]);
            curr[j] = move_cost.min(merge_cost).min(split_cost);
            row_min = row_min.min(curr[j]);
        }
        if row_min > effective_cutoff {
            return Ok(None);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let distance = prev[y.len()];
    if !distance.is_finite() {
        return if effective_cutoff.is_finite() {
            Ok(None)
        } else {
            Err(IncompleteReason::NumericOverflow)
        };
    }
    Ok((distance <= effective_cutoff).then_some(distance))
}

fn append_neighbors<'a, V>(
    entries: &'a [ApproxEntry<V>],
    best: BinaryHeap<ExactCandidate>,
    output: &mut Vec<ApproxMsmNeighbor<'a, V>>,
) {
    for candidate in best.into_sorted_vec() {
        output.push(ApproxMsmNeighbor {
            index: candidate.idx,
            value: &entries[candidate.idx].value,
            distance: candidate.distance,
        });
    }
}

fn incomplete_outcome<'a, V>(
    ledger: ResourceLedger,
    reason: IncompleteReason,
    partial: Option<ApproxMsmSearchResult<'a, V>>,
) -> ApproxMsmSearchOutcome<'a, V> {
    ApproxMsmSearchOutcome::Incomplete {
        partial,
        reason,
        usage: ledger.usage(),
    }
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
    fn impossible_capacity_is_a_tagged_allocation_failure() {
        assert!(matches!(
            vec_with_capacity::<u8>(usize::MAX),
            Err(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: usize::MAX,
            })
        ));
        assert!(matches!(
            vec_with_capacity::<u16>(usize::MAX),
            Err(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })
        ));
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
