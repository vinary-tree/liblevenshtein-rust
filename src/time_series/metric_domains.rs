//! Canonical representatives for temporal pseudometric quotient domains.
//!
//! ERP assigns zero cost to insertion or deletion of its gap value, while
//! discrete Frechet assigns zero cost to consecutive repetitions. Their raw
//! sequence kernels are therefore pseudometrics. This module makes the
//! corresponding quotient spaces concrete by canonicalizing every equivalence
//! class before exposing a metric-labelled distance.

use thiserror::Error;

use std::hash::Hash;

use super::bounded::{OperationOutcome, PageBudget, ResourceLimits, TemporalValidationError};
use super::elastic::{BoundedRangeOutcome, ElasticTransducer};
use super::encoding::QuantizationConfig;
use super::kernels::{ErpConfig, FrechetConfig, MetricTwedConfig};
use super::msm_kernel::MetricMsmKernel;
use super::Operand;

/// Failure to construct or compare a canonical quotient representative.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum MetricDomainError {
    /// The configured ERP gap value was NaN or infinite.
    #[error("ERP quotient gap value must be finite")]
    NonFiniteGap,
    /// A series failed the shared bounded scalar validation contract.
    #[error(transparent)]
    InvalidSeries(#[from] TemporalValidationError),
    /// A Frechet path was empty and therefore had no endpoint-covering coupling.
    #[error("the discrete Frechet metric quotient contains only nonempty paths")]
    EmptyFrechetPath,
    /// ERP representatives were created for different gap-value quotients.
    #[error("ERP quotient representatives use different gap values")]
    ErpGapMismatch,
}

/// Canonical representative of an ERP gap-value equivalence class.
///
/// All occurrences of `gap` are removed. Two raw finite sequences have ERP
/// distance zero exactly when these representatives agree.
#[derive(Clone, Debug, PartialEq)]
pub struct ErpQuotientSeries {
    gap: f64,
    samples: Box<[f64]>,
}

impl ErpQuotientSeries {
    /// Validate and canonicalize one raw series under the quotient for `gap`.
    pub fn try_new(
        raw: &[f64],
        gap: f64,
        limits: ResourceLimits,
    ) -> Result<Self, MetricDomainError> {
        if !gap.is_finite() {
            return Err(MetricDomainError::NonFiniteGap);
        }
        validate_finite(raw, limits)?;
        let samples = raw
            .iter()
            .copied()
            .filter(|sample| *sample != gap)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self { gap, samples })
    }

    /// Return the finite gap value defining this quotient.
    #[inline]
    pub fn gap(&self) -> f64 {
        self.gap
    }

    /// Borrow the unique no-gap representative.
    #[inline]
    pub fn canonical_samples(&self) -> &[f64] {
        &self.samples
    }
}

/// ERP distance restricted to one documented gap-value quotient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricErpConfig {
    gap: f64,
}

mod sealed {
    pub trait AuditedMetricIndex {}
}

/// Closed gate for exact temporal indexes whose metric domain was reviewed.
///
/// This marker is intentionally narrower than an open downstream assertion:
/// it covers validated MSM/unit-grid TWED kernels and the canonical ERP and
/// Fréchet quotient indexes defined here. DTW, raw ERP/Fréchet, zero-stiffness
/// TWED, and zero-cost MSM cannot satisfy it.
pub trait AuditedMetricTimeSeriesIndex: sealed::AuditedMetricIndex {}

impl<V> sealed::AuditedMetricIndex for ElasticTransducer<MetricMsmKernel, V> where
    V: Eq + Hash + Clone
{
}
impl<V> AuditedMetricTimeSeriesIndex for ElasticTransducer<MetricMsmKernel, V> where
    V: Eq + Hash + Clone
{
}
impl<V> sealed::AuditedMetricIndex for ElasticTransducer<MetricTwedConfig, V> where
    V: Eq + Hash + Clone
{
}
impl<V> AuditedMetricTimeSeriesIndex for ElasticTransducer<MetricTwedConfig, V> where
    V: Eq + Hash + Clone
{
}

/// Exact ERP index restricted to canonical representatives of one gap quotient.
#[derive(Debug)]
pub struct MetricErpTransducer<V: Eq + Hash + Clone = usize> {
    metric: MetricErpConfig,
    inner: ElasticTransducer<ErpConfig, V>,
}

impl<V> sealed::AuditedMetricIndex for MetricErpTransducer<V> where V: Eq + Hash + Clone {}
impl<V> AuditedMetricTimeSeriesIndex for MetricErpTransducer<V> where V: Eq + Hash + Clone {}

impl<V> MetricErpTransducer<V>
where
    V: Eq + Hash + Clone,
{
    /// Construct an empty exact index for one fixed gap-value quotient.
    pub fn new(quantizer: QuantizationConfig, metric: MetricErpConfig) -> Self {
        Self {
            inner: ElasticTransducer::new(quantizer, ErpConfig::new(metric.gap)),
            metric,
        }
    }

    /// Insert a canonical representative from this index's exact quotient.
    pub fn insert(
        &mut self,
        value: V,
        representative: &ErpQuotientSeries,
    ) -> Result<bool, MetricDomainError> {
        self.validate_gap(representative)?;
        Ok(self.inner.insert(value, representative.canonical_samples()))
    }

    /// Validate, canonicalize, and insert one raw finite series.
    pub fn insert_raw(
        &mut self,
        value: V,
        raw: &[f64],
        limits: ResourceLimits,
    ) -> Result<bool, MetricDomainError> {
        let representative = self.metric.representative(raw, limits)?;
        self.insert(value, &representative)
    }

    /// Remove one stable identifier.
    pub fn remove(&mut self, value: V) -> bool {
        self.inner.remove(value)
    }

    /// Number of indexed quotient representatives.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Exact range query on a canonical representative from the same quotient.
    pub fn search_range(
        &self,
        query: &ErpQuotientSeries,
        cutoff: f64,
    ) -> Result<Vec<(V, f64)>, MetricDomainError> {
        self.validate_gap(query)?;
        Ok(self.inner.search_range(query.canonical_samples(), cutoff))
    }

    /// Strict, resumable exact range query using the compact sparse product.
    pub fn search_range_bounded(
        &self,
        query: &ErpQuotientSeries,
        cutoff: f64,
        limits: ResourceLimits,
        page: PageBudget,
    ) -> Result<BoundedRangeOutcome<'_, ErpConfig, V>, MetricDomainError> {
        self.validate_gap(query)?;
        Ok(self
            .inner
            .search_range_bounded(query.canonical_samples(), cutoff, limits, page)?)
    }

    /// Strict exact kNN using bounded full-precision verification and `O(k)` state.
    pub fn search_knn_bounded(
        &self,
        query: &ErpQuotientSeries,
        k: usize,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<Vec<(V, f64)>>, MetricDomainError> {
        self.validate_gap(query)?;
        Ok(self
            .inner
            .search_knn_bounded(query.canonical_samples(), k, limits)?)
    }

    /// Borrow the underlying exact non-triangle-pruning engine.
    pub fn as_inner(&self) -> &ElasticTransducer<ErpConfig, V> {
        &self.inner
    }

    fn validate_gap(&self, series: &ErpQuotientSeries) -> Result<(), MetricDomainError> {
        if series.gap.to_bits() != self.metric.gap.to_bits() {
            return Err(MetricDomainError::ErpGapMismatch);
        }
        Ok(())
    }
}

impl MetricErpConfig {
    /// Construct an ERP quotient metric with a finite gap value.
    pub fn try_new(gap: f64) -> Result<Self, MetricDomainError> {
        if !gap.is_finite() {
            return Err(MetricDomainError::NonFiniteGap);
        }
        Ok(Self { gap })
    }

    /// Return the finite gap value defining this quotient.
    #[inline]
    pub fn gap(&self) -> f64 {
        self.gap
    }

    /// Validate and canonicalize a member of this quotient space.
    pub fn representative(
        &self,
        raw: &[f64],
        limits: ResourceLimits,
    ) -> Result<ErpQuotientSeries, MetricDomainError> {
        ErpQuotientSeries::try_new(raw, self.gap, limits)
    }

    /// Compute the finite ERP metric between canonical representatives.
    pub fn distance(
        &self,
        left: &ErpQuotientSeries,
        right: &ErpQuotientSeries,
    ) -> Result<f64, MetricDomainError> {
        if left.gap != self.gap || right.gap != self.gap {
            return Err(MetricDomainError::ErpGapMismatch);
        }
        Ok(ErpConfig::new(self.gap).distance(&left.samples, &right.samples))
    }
}

/// Canonical representative of a consecutive-stutter path equivalence class.
///
/// Each maximal run of identical finite samples is represented once. The type
/// is nonempty because discrete Frechet has no finite endpoint-covering
/// coupling between an empty and nonempty path.
#[derive(Clone, Debug, PartialEq)]
pub struct FrechetStutterClass {
    samples: Box<[f64]>,
}

/// Exact discrete Fréchet index over nonempty, stutter-canonical paths.
#[derive(Debug)]
pub struct MetricFrechetTransducer<V: Eq + Hash + Clone = usize> {
    inner: ElasticTransducer<FrechetConfig, V>,
}

impl<V> sealed::AuditedMetricIndex for MetricFrechetTransducer<V> where V: Eq + Hash + Clone {}
impl<V> AuditedMetricTimeSeriesIndex for MetricFrechetTransducer<V> where V: Eq + Hash + Clone {}

impl<V> MetricFrechetTransducer<V>
where
    V: Eq + Hash + Clone,
{
    /// Construct an empty exact quotient index.
    pub fn new(quantizer: QuantizationConfig) -> Self {
        Self {
            inner: ElasticTransducer::new(quantizer, FrechetConfig::new()),
        }
    }

    /// Insert one nonempty stutter-canonical representative.
    pub fn insert(&mut self, value: V, representative: &FrechetStutterClass) -> bool {
        self.inner.insert(value, representative.canonical_samples())
    }

    /// Validate, canonicalize, and insert one raw finite nonempty path.
    pub fn insert_raw(
        &mut self,
        value: V,
        raw: &[f64],
        limits: ResourceLimits,
    ) -> Result<bool, MetricDomainError> {
        let representative = FrechetStutterClass::try_new(raw, limits)?;
        Ok(self.insert(value, &representative))
    }

    /// Remove one stable identifier.
    pub fn remove(&mut self, value: V) -> bool {
        self.inner.remove(value)
    }

    /// Number of indexed quotient representatives.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Exact range query over the documented metric quotient.
    pub fn search_range(&self, query: &FrechetStutterClass, cutoff: f64) -> Vec<(V, f64)> {
        self.inner.search_range(query.canonical_samples(), cutoff)
    }

    /// Strict, resumable exact range query using the compact sparse product.
    pub fn search_range_bounded(
        &self,
        query: &FrechetStutterClass,
        cutoff: f64,
        limits: ResourceLimits,
        page: PageBudget,
    ) -> Result<BoundedRangeOutcome<'_, FrechetConfig, V>, TemporalValidationError> {
        self.inner
            .search_range_bounded(query.canonical_samples(), cutoff, limits, page)
    }

    /// Strict exact kNN using bounded full-precision verification and `O(k)` state.
    pub fn search_knn_bounded(
        &self,
        query: &FrechetStutterClass,
        k: usize,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<Vec<(V, f64)>>, TemporalValidationError> {
        self.inner
            .search_knn_bounded(query.canonical_samples(), k, limits)
    }

    /// Borrow the underlying exact non-triangle-pruning engine.
    pub fn as_inner(&self) -> &ElasticTransducer<FrechetConfig, V> {
        &self.inner
    }
}

impl FrechetStutterClass {
    /// Validate and run-collapse one nonempty scalar path.
    pub fn try_new(raw: &[f64], limits: ResourceLimits) -> Result<Self, MetricDomainError> {
        validate_finite(raw, limits)?;
        if raw.is_empty() {
            return Err(MetricDomainError::EmptyFrechetPath);
        }
        let mut samples = Vec::with_capacity(raw.len());
        for &sample in raw {
            if samples.last().is_none_or(|previous| *previous != sample) {
                samples.push(sample);
            }
        }
        Ok(Self {
            samples: samples.into_boxed_slice(),
        })
    }

    /// Borrow the unique consecutive-stutter-free representative.
    #[inline]
    pub fn canonical_samples(&self) -> &[f64] {
        &self.samples
    }

    /// Compute the finite discrete Frechet metric on the quotient space.
    #[inline]
    pub fn distance(&self, other: &Self) -> f64 {
        FrechetConfig::new().distance(&self.samples, &other.samples)
    }
}

fn validate_finite(raw: &[f64], limits: ResourceLimits) -> Result<(), MetricDomainError> {
    if raw.len() > limits.max_series_len {
        return Err(TemporalValidationError::SeriesTooLong {
            operand: Operand::Query,
            len: raw.len(),
            limit: limits.max_series_len,
        }
        .into());
    }
    if let Some(index) = raw.iter().position(|sample| !sample.is_finite()) {
        return Err(TemporalValidationError::NonFiniteSample {
            operand: Operand::Query,
            index,
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erp_quotient_removes_exactly_the_gap_class() {
        let metric = MetricErpConfig::try_new(0.0).unwrap();
        let left = metric
            .representative(&[1.0, 0.0, 2.0], ResourceLimits::default())
            .unwrap();
        let right = metric
            .representative(&[1.0, 2.0], ResourceLimits::default())
            .unwrap();
        assert_eq!(left, right);
        assert_eq!(metric.distance(&left, &right), Ok(0.0));
    }

    #[test]
    fn frechet_quotient_collapses_consecutive_stutters_only() {
        let left =
            FrechetStutterClass::try_new(&[1.0, 1.0, 2.0, 2.0, 1.0], ResourceLimits::default())
                .unwrap();
        let right =
            FrechetStutterClass::try_new(&[1.0, 2.0, 1.0], ResourceLimits::default()).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.distance(&right), 0.0);
    }

    #[test]
    fn metric_erp_index_canonicalizes_before_quantization_and_exact_verification() {
        let metric = MetricErpConfig::try_new(0.0).unwrap();
        let mut index = MetricErpTransducer::new(QuantizationConfig::for_u8(-10.0, 10.0), metric);
        assert!(index
            .insert_raw(7_u64, &[1.0, 0.0, 2.0], ResourceLimits::default())
            .unwrap());
        assert!(index
            .insert_raw(11_u64, &[1.0, 2.0], ResourceLimits::default())
            .unwrap());
        let query = metric
            .representative(&[1.0, 0.0, 0.0, 2.0], ResourceLimits::default())
            .unwrap();
        let mut results = index.search_range(&query, 0.0).unwrap();
        results.sort_by_key(|(stable_id, _)| *stable_id);
        assert_eq!(results, vec![(7, 0.0), (11, 0.0)]);
    }

    #[test]
    fn metric_frechet_index_collapses_stutters_before_dictionary_product() {
        let mut index = MetricFrechetTransducer::new(QuantizationConfig::for_u8(-10.0, 10.0));
        assert!(index
            .insert_raw(7_u64, &[1.0, 1.0, 2.0, 2.0], ResourceLimits::default())
            .unwrap());
        assert!(index
            .insert_raw(11_u64, &[1.0, 2.0], ResourceLimits::default())
            .unwrap());
        let query =
            FrechetStutterClass::try_new(&[1.0, 1.0, 1.0, 2.0], ResourceLimits::default()).unwrap();
        let mut results = index.search_range(&query, 0.0);
        results.sort_by_key(|(stable_id, _)| *stable_id);
        assert_eq!(results, vec![(7, 0.0), (11, 0.0)]);
    }
}
