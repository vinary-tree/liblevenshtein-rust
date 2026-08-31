//! Typed vector samples and metric temporal paths.
//!
//! Vector samples are never flattened into scalar sequences. A sealed ground
//! metric compares corresponding points, and discrete Frechet then aligns
//! whole points along monotone paths. Consecutive identical points are
//! canonicalized so the exposed path space is the true metric quotient rather
//! than the raw pseudometric domain.

use std::collections::HashSet;
use std::mem::size_of;
use thiserror::Error;

use super::automaton::{OnlineAutomatonLimits, OnlineStepOutcome};
use super::bounded::{
    ExactDecision, IncompleteReason, NoWitness, OperationOutcome, ResourceKind, ResourceLedger,
    ResourceLimits, ResourceUsage, TemporalValidationError,
};
use super::elastic::sparse::{charge_work, NeighborSeedRows};
use super::timestamped_twed::TimestampUnit;
use crate::cost::{BottleneckCost, CostMonoid, WeightedCost};

mod sealed {
    pub trait Sealed {}
}

/// Audited metric used to compare two equal-dimensional vector samples.
///
/// The trait is sealed: exact temporal metric claims are available only for
/// implementations whose nonnegativity, identity, symmetry, and triangle laws
/// are part of this crate's test and review surface.
pub trait GroundMetric: sealed::Sealed + Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Compute a finite nonnegative point distance, or positive infinity on
    /// floating-point overflow.
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64;

    /// Fixed coordinate count when the metric carries a typed schema.
    #[inline]
    fn required_dimension(&self) -> Option<usize> {
        None
    }
}

/// Exact application-level identity of one channel and its physical unit.
///
/// Identity is deliberately textual and equality is exact. The crate never
/// infers that two differently named units are convertible and never aligns
/// channels by position after their identities disagree.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelIdentity {
    channel: Box<str>,
    unit: Box<str>,
}

impl ChannelIdentity {
    /// Construct a nonempty channel/unit identity.
    pub fn try_new(channel: &str, unit: &str) -> Result<Self, VectorMetricError> {
        if channel.trim().is_empty() {
            return Err(VectorMetricError::InvalidChannelIdentity { field: "channel" });
        }
        if unit.trim().is_empty() {
            return Err(VectorMetricError::InvalidChannelIdentity { field: "unit" });
        }
        Ok(Self {
            channel: try_boxed_str(channel)?,
            unit: try_boxed_str(unit)?,
        })
    }

    /// Fallibly copy this identity without relying on the process-aborting
    /// allocation path of [`Clone`].
    pub fn try_clone(&self) -> Result<Self, VectorMetricError> {
        Ok(Self {
            channel: try_boxed_str(&self.channel)?,
            unit: try_boxed_str(&self.unit)?,
        })
    }

    /// Borrow the exact channel identifier.
    #[inline]
    pub fn channel(&self) -> &str {
        &self.channel
    }

    /// Borrow the exact unit identifier.
    #[inline]
    pub fn unit(&self) -> &str {
        &self.unit
    }
}

/// Provenance fixing all scales to one training fold and estimator revision.
///
/// The values are identity metadata, not labels used to select data. Held-out
/// or pair-local observations cannot alter a constructed metric.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FoldLocalScaleProvenance {
    training_fold_id: Box<str>,
    estimator_revision: Box<str>,
}

impl FoldLocalScaleProvenance {
    /// Construct nonempty, exact fold-local scale provenance.
    pub fn try_new(
        training_fold_id: &str,
        estimator_revision: &str,
    ) -> Result<Self, VectorMetricError> {
        if training_fold_id.trim().is_empty() {
            return Err(VectorMetricError::InvalidScaleProvenance {
                field: "training_fold_id",
            });
        }
        if estimator_revision.trim().is_empty() {
            return Err(VectorMetricError::InvalidScaleProvenance {
                field: "estimator_revision",
            });
        }
        Ok(Self {
            training_fold_id: try_boxed_str(training_fold_id)?,
            estimator_revision: try_boxed_str(estimator_revision)?,
        })
    }

    /// Borrow the immutable training-fold identity.
    #[inline]
    pub fn training_fold_id(&self) -> &str {
        &self.training_fold_id
    }

    /// Borrow the immutable scale-estimator revision.
    #[inline]
    pub fn estimator_revision(&self) -> &str {
        &self.estimator_revision
    }
}

/// One strictly positive term in a fixed weighted channel metric.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricChannel {
    identity: ChannelIdentity,
    scale: f64,
    weight: f64,
}

impl MetricChannel {
    /// Construct one channel with finite `scale > 0` and `weight > 0`.
    pub fn try_new(
        identity: ChannelIdentity,
        scale: f64,
        weight: f64,
    ) -> Result<Self, VectorMetricError> {
        if !scale.is_finite() || scale <= 0.0 {
            let ChannelIdentity { channel, .. } = identity;
            return Err(VectorMetricError::InvalidChannelScale { channel });
        }
        if !weight.is_finite() || weight <= 0.0 {
            let ChannelIdentity { channel, .. } = identity;
            return Err(VectorMetricError::InvalidChannelWeight { channel });
        }
        if !(weight / scale).is_finite() {
            return Err(VectorMetricError::NonFiniteChannelAggregate);
        }
        Ok(Self {
            identity,
            scale,
            weight,
        })
    }

    /// Borrow the typed channel/unit identity.
    #[inline]
    pub fn identity(&self) -> &ChannelIdentity {
        &self.identity
    }

    /// Return the fixed fold-local scale.
    #[inline]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Return the fixed strictly positive channel weight.
    #[inline]
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

/// Ordered channel/unit schema carried by samples, boxes, and scorers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChannelLayout {
    identities: Vec<ChannelIdentity>,
}

impl ChannelLayout {
    fn from_channels(channels: &[MetricChannel]) -> Result<Self, VectorMetricError> {
        let requested = channels
            .len()
            .checked_mul(size_of::<ChannelIdentity>())
            .ok_or(VectorMetricError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        let mut identities = Vec::new();
        identities.try_reserve_exact(channels.len()).map_err(|_| {
            VectorMetricError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested,
            })
        })?;
        for channel in channels {
            identities.push(channel.identity.try_clone()?);
        }
        Ok(Self { identities })
    }

    /// Fallibly copy this complete ordered channel/unit schema.
    pub fn try_clone(&self) -> Result<Self, VectorMetricError> {
        let requested = self
            .identities
            .len()
            .checked_mul(size_of::<ChannelIdentity>())
            .ok_or(VectorMetricError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        let mut identities = Vec::new();
        identities
            .try_reserve_exact(self.identities.len())
            .map_err(|_| {
                VectorMetricError::Resource(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested,
                })
            })?;
        for identity in &self.identities {
            identities.push(identity.try_clone()?);
        }
        Ok(Self { identities })
    }
    /// Number of typed channels.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.identities.len()
    }

    /// Borrow identities in their fixed coordinate order.
    #[inline]
    pub fn identities(&self) -> &[ChannelIdentity] {
        &self.identities
    }
}

/// Fixed positive sum of scaled scalar channel metrics.
///
/// For channel `c`, the point cost is
/// $`w_c |x_c-y_c| / s_c`$. Channel identities, units, positive weights,
/// positive scales, and fold provenance are immutable for the metric's whole
/// lifetime. Missing coordinates and pair-dependent renormalization are not
/// members of this domain.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedChannelMetric {
    channels: Vec<MetricChannel>,
    layout: ChannelLayout,
    provenance: FoldLocalScaleProvenance,
}

impl FixedChannelMetric {
    /// Construct a nonempty fixed metric, rejecting duplicate identities and
    /// any aggregate that cannot remain finite.
    pub fn try_new(
        channels: Vec<MetricChannel>,
        provenance: FoldLocalScaleProvenance,
    ) -> Result<Self, VectorMetricError> {
        if channels.is_empty() {
            return Err(VectorMetricError::ZeroDimension);
        }
        let requested = channels
            .len()
            .checked_mul(size_of::<ChannelIdentity>())
            .ok_or(VectorMetricError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        let mut seen: HashSet<&ChannelIdentity> = HashSet::new();
        seen.try_reserve(channels.len()).map_err(|_| {
            VectorMetricError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested,
            })
        })?;
        let mut coefficient_sum = WeightedCost::ZERO;
        for channel in &channels {
            if !seen.insert(&channel.identity) {
                return Err(VectorMetricError::DuplicateChannelIdentity {
                    channel: try_boxed_str(channel.identity.channel())?,
                    unit: try_boxed_str(channel.identity.unit())?,
                });
            }
            coefficient_sum =
                WeightedCost::combine(coefficient_sum, channel.weight / channel.scale);
        }
        if !coefficient_sum.is_finite() {
            return Err(VectorMetricError::NonFiniteChannelAggregate);
        }
        let layout = ChannelLayout::from_channels(&channels)?;
        Ok(Self {
            channels,
            layout,
            provenance,
        })
    }

    /// Borrow the exact ordered channel/unit schema.
    #[inline]
    pub fn channel_layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Borrow immutable fold-local scale provenance.
    #[inline]
    pub fn scale_provenance(&self) -> &FoldLocalScaleProvenance {
        &self.provenance
    }

    /// Borrow fixed scale and weight definitions in schema order.
    #[inline]
    pub fn channels(&self) -> &[MetricChannel] {
        &self.channels
    }

    /// Compute the fixed channel sum after checking both dimensions.
    pub fn distance_checked(
        &self,
        left: &VectorSample,
        right: &VectorSample,
    ) -> Result<f64, VectorMetricError> {
        self.validate_sample(left)?;
        self.validate_sample(right)?;
        Ok(self.distance_unchecked(left, right))
    }

    /// K1 point-to-box lower bound for an on-demand vector dictionary edge.
    pub fn point_box_lower_bound(
        &self,
        point: &VectorSample,
        bounds: &VectorBox,
    ) -> Result<f64, VectorMetricError> {
        self.validate_sample(point)?;
        self.validate_layout(bounds.layout())?;
        Ok(self
            .channels
            .iter()
            .zip(point.coordinates.iter().zip(bounds.bounds.iter()))
            .fold(WeightedCost::ZERO, |total, (channel, (value, bound))| {
                WeightedCost::combine(
                    total,
                    channel.weight * interval_distance(*value, bound.0, bound.1) / channel.scale,
                )
            }))
    }

    /// K1 box-to-box lower bound used by adjacent-label vector kernels.
    pub fn box_box_lower_bound(
        &self,
        left: &VectorBox,
        right: &VectorBox,
    ) -> Result<f64, VectorMetricError> {
        self.validate_layout(left.layout())?;
        self.validate_layout(right.layout())?;
        Ok(self
            .channels
            .iter()
            .zip(left.bounds.iter().zip(right.bounds.iter()))
            .fold(WeightedCost::ZERO, |total, (channel, (left, right))| {
                WeightedCost::combine(
                    total,
                    channel.weight * interval_gap(*left, *right) / channel.scale,
                )
            }))
    }

    fn validate_sample(&self, sample: &VectorSample) -> Result<(), VectorMetricError> {
        if sample.dimension() != self.layout.dimension() {
            return Err(VectorMetricError::DimensionMismatch {
                expected: self.layout.dimension(),
                observed: sample.dimension(),
            });
        }
        Ok(())
    }

    fn validate_layout(&self, layout: &ChannelLayout) -> Result<(), VectorMetricError> {
        if layout != &self.layout {
            return Err(VectorMetricError::ChannelLayoutMismatch);
        }
        Ok(())
    }

    fn distance_unchecked(&self, left: &VectorSample, right: &VectorSample) -> f64 {
        self.channels.iter().zip(point_pairs(left, right)).fold(
            WeightedCost::ZERO,
            |total, (channel, (left, right))| {
                WeightedCost::combine(total, channel.weight * (left - right).abs() / channel.scale)
            },
        )
    }
}

fn try_boxed_str(value: &str) -> Result<Box<str>, VectorMetricError> {
    let mut storage = String::new();
    storage.try_reserve_exact(value.len()).map_err(|_| {
        VectorMetricError::Resource(IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested: value.len(),
        })
    })?;
    storage.push_str(value);
    Ok(storage.into_boxed_str())
}

impl sealed::Sealed for FixedChannelMetric {}

impl GroundMetric for FixedChannelMetric {
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64 {
        debug_assert_eq!(left.dimension(), self.layout.dimension());
        debug_assert_eq!(right.dimension(), self.layout.dimension());
        self.distance_unchecked(left, right)
    }

    #[inline]
    fn required_dimension(&self) -> Option<usize> {
        Some(self.layout.dimension())
    }
}

/// Axis-aligned interval label with an exact channel/unit schema.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorBox {
    layout: ChannelLayout,
    bounds: Vec<(f64, f64)>,
}

impl VectorBox {
    /// Validate finite nonempty closed coordinate intervals.
    pub fn try_new(
        layout: ChannelLayout,
        bounds: &[(f64, f64)],
    ) -> Result<Self, VectorMetricError> {
        if bounds.len() != layout.dimension() {
            return Err(VectorMetricError::DimensionMismatch {
                expected: layout.dimension(),
                observed: bounds.len(),
            });
        }
        if let Some(index) = bounds
            .iter()
            .position(|(low, high)| !low.is_finite() || !high.is_finite() || low > high)
        {
            return Err(VectorMetricError::InvalidVectorBox { index });
        }
        let requested = bounds.len().checked_mul(size_of::<(f64, f64)>()).ok_or(
            VectorMetricError::Resource(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            }),
        )?;
        let mut storage = Vec::new();
        storage.try_reserve_exact(bounds.len()).map_err(|_| {
            VectorMetricError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested,
            })
        })?;
        storage.extend_from_slice(bounds);
        Ok(Self {
            layout,
            bounds: storage,
        })
    }

    /// Construct a degenerate exact box around one point.
    pub fn from_sample(
        layout: ChannelLayout,
        sample: &VectorSample,
    ) -> Result<Self, VectorMetricError> {
        if sample.dimension() != layout.dimension() {
            return Err(VectorMetricError::DimensionMismatch {
                expected: layout.dimension(),
                observed: sample.dimension(),
            });
        }
        let requested = sample
            .dimension()
            .checked_mul(size_of::<(f64, f64)>())
            .ok_or(VectorMetricError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        let mut bounds = Vec::new();
        bounds.try_reserve_exact(sample.dimension()).map_err(|_| {
            VectorMetricError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested,
            })
        })?;
        bounds.extend(
            sample
                .coordinates()
                .iter()
                .copied()
                .map(|value| (value, value)),
        );
        Self::try_new(layout, &bounds)
    }

    /// Borrow the exact channel layout.
    #[inline]
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Borrow closed bounds in channel order.
    #[inline]
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    /// Whether this box is a coordinatewise subset of `coarse`.
    pub fn refines(&self, coarse: &Self) -> Result<bool, VectorMetricError> {
        if self.layout != coarse.layout {
            return Err(VectorMetricError::ChannelLayoutMismatch);
        }
        Ok(self
            .bounds
            .iter()
            .zip(coarse.bounds.iter())
            .all(|(fine, coarse)| coarse.0 <= fine.0 && fine.1 <= coarse.1))
    }
}

/// Manhattan (`L1`) ground metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct L1GroundMetric;

impl sealed::Sealed for L1GroundMetric {}

impl GroundMetric for L1GroundMetric {
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64 {
        point_pairs(left, right)
            .map(|(a, b)| (a - b).abs())
            .fold(WeightedCost::ZERO, WeightedCost::combine)
    }
}

/// Euclidean (`L2`) ground metric, accumulated with stable `hypot` steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct L2GroundMetric;

impl sealed::Sealed for L2GroundMetric {}

impl GroundMetric for L2GroundMetric {
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64 {
        point_pairs(left, right)
            .map(|(a, b)| a - b)
            .fold(0.0_f64, f64::hypot)
    }
}

/// Chebyshev (`L-infinity`) ground metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LinfGroundMetric;

impl sealed::Sealed for LinfGroundMetric {}

impl GroundMetric for LinfGroundMetric {
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64 {
        point_pairs(left, right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }
}

/// Failure to construct or compare typed vector temporal values.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum VectorMetricError {
    /// A point had zero coordinates.
    #[error("vector samples must have positive dimension")]
    ZeroDimension,
    /// A coordinate was NaN or infinite.
    #[error("coordinate {index} is not finite")]
    NonFiniteCoordinate {
        /// Zero-based coordinate index.
        index: usize,
    },
    /// A channel name or unit identity was empty.
    #[error("channel identity field {field} must be nonempty")]
    InvalidChannelIdentity {
        /// Invalid identity field.
        field: &'static str,
    },
    /// Fold-local scale provenance was incomplete.
    #[error("scale provenance field {field} must be nonempty")]
    InvalidScaleProvenance {
        /// Invalid provenance field.
        field: &'static str,
    },
    /// A channel scale was not finite and strictly positive.
    #[error("channel {channel} scale must be finite and strictly positive")]
    InvalidChannelScale {
        /// Exact channel identifier.
        channel: Box<str>,
    },
    /// A channel weight was not finite and strictly positive.
    #[error("channel {channel} weight must be finite and strictly positive")]
    InvalidChannelWeight {
        /// Exact channel identifier.
        channel: Box<str>,
    },
    /// The metric contained the same typed channel more than once.
    #[error("duplicate channel identity {channel} [{unit}]")]
    DuplicateChannelIdentity {
        /// Exact channel identifier.
        channel: Box<str>,
        /// Exact unit identifier.
        unit: Box<str>,
    },
    /// Fixed positive channel coefficients could not be represented finitely.
    #[error("fixed channel metric aggregate is not finite")]
    NonFiniteChannelAggregate,
    /// A series or interval label used a different typed channel layout.
    #[error("typed channel layouts differ")]
    ChannelLayoutMismatch,
    /// One vector interval was nonfinite, empty, or reversed.
    #[error("vector interval coordinate {index} is not a finite closed interval")]
    InvalidVectorBox {
        /// Zero-based coordinate index.
        index: usize,
    },
    /// A timestamp interval was nonfinite, empty, or reversed.
    #[error("timestamp interval must be finite and closed")]
    InvalidTimestampInterval,
    /// The path contained no samples.
    #[error("a discrete Frechet metric path must be nonempty")]
    EmptyPath,
    /// Timestamped vector TWED is defined here only for nonempty series.
    #[error("timestamped vector TWED metric series must be nonempty")]
    EmptyTimestampedSeries,
    /// Samples within or across paths had different dimensions.
    #[error("vector dimensions differ: expected {expected}, observed {observed}")]
    DimensionMismatch {
        /// Required coordinate count.
        expected: usize,
        /// Observed coordinate count.
        observed: usize,
    },
    /// Value and timestamp arrays had different lengths.
    #[error("vector sample count {samples} differs from timestamp count {timestamps}")]
    TimestampLengthMismatch {
        /// Number of vector samples.
        samples: usize,
        /// Number of timestamps.
        timestamps: usize,
    },
    /// A timestamp or origin was NaN or infinite.
    #[error("timestamp index {index:?} is not finite (None denotes the origin)")]
    NonFiniteTimestamp {
        /// Zero-based timestamp index; `None` denotes the origin.
        index: Option<usize>,
    },
    /// A timestamp was not strictly greater than its predecessor.
    #[error("timestamp {index} is not strictly greater than its predecessor")]
    NonMonotoneTimestamp {
        /// Zero-based timestamp index.
        index: usize,
    },
    /// The first timestamp preceded the physical origin.
    #[error("first timestamp precedes the declared origin")]
    TimestampBeforeOrigin,
    /// Timestamped operands used different canonical units.
    #[error("timestamp units differ")]
    MixedTimestampUnits,
    /// Timestamped operands used different physical origins.
    #[error("timestamp origins differ")]
    MixedTimestampOrigins,
    /// Timestamped vector TWED stiffness was outside its metric domain.
    #[error("timestamped vector TWED stiffness must be finite and strictly positive")]
    InvalidStiffness,
    /// Timestamped vector TWED gap penalty was outside its metric domain.
    #[error("timestamped vector TWED gap penalty must be finite and nonnegative")]
    InvalidGapPenalty,
    /// Fixed online state or scratch allocation exceeded an explicit ceiling.
    #[error("vector temporal automaton resource construction failed: {0:?}")]
    Resource(IncompleteReason),
    /// Shared temporal request validation failed.
    #[error(transparent)]
    InvalidRequest(#[from] TemporalValidationError),
}

/// Exact observation of one committed vector-target prefix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorFrechetOnlineObservation {
    /// Number of target points consumed by the fixed-query machine.
    pub consumed_target_len: usize,
    /// Non-subsumed recurrence rows within the construction cutoff.
    pub active_positions: usize,
    /// Exact complete-prefix distance when it lies within the cutoff.
    pub distance_within_cutoff: Option<f64>,
    /// Smallest finite bottleneck value in the live generation.
    pub minimum_active_cost: Option<f64>,
}

/// Owned, finite, positive-dimensional vector sample.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorSample {
    coordinates: Vec<f64>,
}

impl VectorSample {
    /// Validate and copy one vector sample.
    pub fn try_new(coordinates: &[f64], limits: ResourceLimits) -> Result<Self, VectorMetricError> {
        if coordinates.is_empty() {
            return Err(VectorMetricError::ZeroDimension);
        }
        if coordinates.len() > limits.max_dimension {
            return Err(TemporalValidationError::DimensionTooLarge {
                dimension: coordinates.len(),
                limit: limits.max_dimension,
            }
            .into());
        }
        if let Some(index) = coordinates
            .iter()
            .position(|coordinate| !coordinate.is_finite())
        {
            return Err(VectorMetricError::NonFiniteCoordinate { index });
        }
        let requested =
            coordinates
                .len()
                .checked_mul(size_of::<f64>())
                .ok_or(VectorMetricError::Resource(
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    },
                ))?;
        if requested > limits.max_scratch_bytes {
            return Err(VectorMetricError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested,
                },
            ));
        }
        let mut storage = Vec::new();
        storage.try_reserve_exact(coordinates.len()).map_err(|_| {
            VectorMetricError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested,
            })
        })?;
        storage.extend_from_slice(coordinates);
        Ok(Self {
            coordinates: storage,
        })
    }

    /// Return the fixed coordinate count.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }

    /// Borrow the finite coordinates.
    #[inline]
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }
}

/// Typed vector series whose samples all use one exact channel layout.
///
/// Empty series remain typed because the layout is retained independently of
/// the samples. Construction never pads, drops, or renormalizes channels.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelVectorSeries {
    layout: ChannelLayout,
    samples: Vec<VectorSample>,
}

impl ChannelVectorSeries {
    fn try_new(
        layout: ChannelLayout,
        samples: Vec<VectorSample>,
        limits: ResourceLimits,
    ) -> Result<Self, VectorMetricError> {
        if samples.len() > limits.max_series_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Query,
                len: samples.len(),
                limit: limits.max_series_len,
            }
            .into());
        }
        if layout.dimension() > limits.max_dimension {
            return Err(TemporalValidationError::DimensionTooLarge {
                dimension: layout.dimension(),
                limit: limits.max_dimension,
            }
            .into());
        }
        if let Some(sample) = samples
            .iter()
            .find(|sample| sample.dimension() != layout.dimension())
        {
            return Err(VectorMetricError::DimensionMismatch {
                expected: layout.dimension(),
                observed: sample.dimension(),
            });
        }
        Ok(Self { layout, samples })
    }

    /// Borrow the exact typed layout.
    #[inline]
    pub fn channel_layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Borrow samples without flattening channel boundaries.
    #[inline]
    pub fn samples(&self) -> &[VectorSample] {
        &self.samples
    }
}

/// Canonical ERP metric representative modulo insertion/deletion of the fixed
/// vector gap sample.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorErpSeries(ChannelVectorSeries);

impl VectorErpSeries {
    /// Borrow the canonical gap-free samples.
    #[inline]
    pub fn samples(&self) -> &[VectorSample] {
        self.0.samples()
    }

    /// Borrow the exact typed channel layout.
    #[inline]
    pub fn channel_layout(&self) -> &ChannelLayout {
        self.0.channel_layout()
    }
}

/// Vector ERP metric over the quotient that removes its fixed gap sample.
///
/// Raw vector sequences form only a pseudometric because inserting the exact
/// gap sample costs zero. [`Self::try_series`] selects the canonical gap-free
/// representative before metric laws are claimed.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorErpMetric {
    ground: FixedChannelMetric,
    gap: VectorSample,
}

impl VectorErpMetric {
    /// Bind one typed fixed-channel ground metric to one fixed vector gap.
    pub fn try_new(
        ground: FixedChannelMetric,
        gap: VectorSample,
    ) -> Result<Self, VectorMetricError> {
        ground.validate_sample(&gap)?;
        Ok(Self { ground, gap })
    }

    /// Borrow the fixed ground metric.
    #[inline]
    pub fn ground_metric(&self) -> &FixedChannelMetric {
        &self.ground
    }

    /// Borrow the fixed gap sample.
    #[inline]
    pub fn gap_sample(&self) -> &VectorSample {
        &self.gap
    }

    /// Validate and canonicalize a raw typed vector sequence.
    pub fn try_series(
        &self,
        samples: Vec<VectorSample>,
        limits: ResourceLimits,
    ) -> Result<VectorErpSeries, VectorMetricError> {
        let raw = ChannelVectorSeries::try_new(self.ground.layout.try_clone()?, samples, limits)?;
        let mut canonical = raw.samples;
        canonical.retain(|sample| sample != &self.gap);
        Ok(VectorErpSeries(ChannelVectorSeries {
            layout: raw.layout,
            samples: canonical,
        }))
    }

    /// K1 local match lower bound for a vector dictionary box.
    #[inline]
    pub fn interval_match_lower_bound(
        &self,
        query: &VectorSample,
        candidate: &VectorBox,
    ) -> Result<f64, VectorMetricError> {
        self.ground.point_box_lower_bound(query, candidate)
    }

    /// K1 local target-gap lower bound for a vector dictionary box.
    #[inline]
    pub fn interval_gap_lower_bound(
        &self,
        candidate: &VectorBox,
    ) -> Result<f64, VectorMetricError> {
        self.ground.point_box_lower_bound(&self.gap, candidate)
    }

    /// K4 gap-mass lower bound on a complete candidate pair.
    pub fn candidate_lower_bound(
        &self,
        left: &VectorErpSeries,
        right: &VectorErpSeries,
    ) -> Result<f64, VectorMetricError> {
        self.validate_series(left)?;
        self.validate_series(right)?;
        let left_mass = vector_gap_mass(&self.ground, &self.gap, left.samples());
        let right_mass = vector_gap_mass(&self.ground, &self.gap, right.samples());
        Ok(if left_mass.is_finite() && right_mass.is_finite() {
            (left_mass - right_mass).abs()
        } else {
            WeightedCost::ZERO
        })
    }

    /// Exact stack-safe ERP cutoff decision with pre-evaluation resource
    /// charging and `O(min(m,n))` retained DP storage.
    pub fn distance_bounded(
        &self,
        left: &VectorErpSeries,
        right: &VectorErpSeries,
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, VectorMetricError> {
        self.validate_series(left)?;
        self.validate_series(right)?;
        validate_vector_cutoff(cutoff)?;
        validate_vector_lengths(left.samples().len(), right.samples().len(), limits)?;

        let mut ledger = ResourceLedger::new(limits);
        let cells = matrix_cells(left.samples().len(), right.samples().len());
        let distance_evaluations = left
            .samples()
            .len()
            .checked_mul(right.samples().len())
            .and_then(|inner| inner.checked_mul(2))
            .and_then(|inner| inner.checked_add(left.samples().len()))
            .and_then(|total| total.checked_add(right.samples().len()));
        let work = distance_evaluations
            .and_then(|count| count.checked_mul(self.ground.layout.dimension()));
        let scratch = rolling_scratch_bytes(left.samples().len(), right.samples().len());
        let (Some(cells), Some(work), Some(scratch)) = (cells, work, scratch) else {
            return Ok(incomplete(
                ledger,
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                },
            ));
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, cells),
            (ResourceKind::WorkUnits, work),
            (ResourceKind::ScratchBytes, scratch),
        ]) {
            return Ok(incomplete(ledger, reason));
        }

        let exact = match vector_erp_with_cutoff(self, left.samples(), right.samples(), cutoff) {
            Ok(exact) => exact,
            Err(reason) => return Ok(incomplete(ledger, reason)),
        };
        Ok(exact_score_outcome(exact, cutoff, ledger))
    }

    fn validate_series(&self, series: &VectorErpSeries) -> Result<(), VectorMetricError> {
        self.ground.validate_layout(series.channel_layout())
    }
}

fn vector_gap_mass(
    ground: &FixedChannelMetric,
    gap: &VectorSample,
    samples: &[VectorSample],
) -> f64 {
    samples.iter().fold(WeightedCost::ZERO, |mass, sample| {
        WeightedCost::combine(mass, ground.distance_unchecked(sample, gap))
    })
}

fn vector_erp_with_cutoff(
    config: &VectorErpMetric,
    left: &[VectorSample],
    right: &[VectorSample],
    cutoff: f64,
) -> Result<Option<f64>, IncompleteReason> {
    let (left, right) = if right.len() > left.len() {
        (right, left)
    } else {
        (left, right)
    };
    let width = right
        .len()
        .checked_add(1)
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })?;
    let (mut previous, mut current) = try_cost_rows(width)?;
    previous[0] = WeightedCost::ZERO;
    for (column, sample) in right.iter().enumerate() {
        previous[column + 1] = WeightedCost::combine(
            previous[column],
            config.ground.distance_unchecked(sample, &config.gap),
        );
    }

    for left_sample in left {
        let delete = config.ground.distance_unchecked(left_sample, &config.gap);
        current[0] = WeightedCost::combine(previous[0], delete);
        let mut row_min = current[0];
        for (index, right_sample) in right.iter().enumerate() {
            let column = index + 1;
            let substitute = WeightedCost::combine(
                previous[column - 1],
                config.ground.distance_unchecked(left_sample, right_sample),
            );
            let delete = WeightedCost::combine(previous[column], delete);
            let insert = WeightedCost::combine(
                current[column - 1],
                config.ground.distance_unchecked(right_sample, &config.gap),
            );
            current[column] = substitute.min(delete).min(insert);
            row_min = row_min.min(current[column]);
        }
        if !WeightedCost::within(row_min, cutoff) {
            return Ok(None);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let exact = previous[right.len()];
    Ok(WeightedCost::within(exact, cutoff).then_some(exact))
}

/// Validated nonempty vector samples paired with physical timestamps.
#[derive(Clone, Debug, PartialEq)]
pub struct TimestampedVectorSeries {
    series: ChannelVectorSeries,
    timestamps: Vec<f64>,
    unit: TimestampUnit,
    origin: f64,
}

impl TimestampedVectorSeries {
    /// Borrow typed vector samples.
    #[inline]
    pub fn samples(&self) -> &[VectorSample] {
        self.series.samples()
    }

    /// Borrow finite strictly increasing timestamps.
    #[inline]
    pub fn timestamps(&self) -> &[f64] {
        &self.timestamps
    }

    /// Borrow the exact channel/unit layout.
    #[inline]
    pub fn channel_layout(&self) -> &ChannelLayout {
        self.series.channel_layout()
    }

    /// Return the canonical timestamp unit.
    #[inline]
    pub fn timestamp_unit(&self) -> TimestampUnit {
        self.unit
    }

    /// Return the shared physical origin.
    #[inline]
    pub fn origin(&self) -> f64 {
        self.origin
    }
}

/// Timestamp interval and vector box used by a lazy dictionary product.
#[derive(Clone, Debug, PartialEq)]
pub struct TimestampedVectorBox {
    sample: VectorBox,
    time: (f64, f64),
    unit: TimestampUnit,
}

impl TimestampedVectorBox {
    /// Construct one finite closed physical-time interval and vector box.
    pub fn try_new(
        sample: VectorBox,
        time: (f64, f64),
        unit: TimestampUnit,
    ) -> Result<Self, VectorMetricError> {
        if !time.0.is_finite() || !time.1.is_finite() || time.0 > time.1 {
            return Err(VectorMetricError::InvalidTimestampInterval);
        }
        Ok(Self { sample, time, unit })
    }

    /// Borrow the vector-label box.
    #[inline]
    pub fn sample_box(&self) -> &VectorBox {
        &self.sample
    }

    /// Return the finite closed physical-time interval.
    #[inline]
    pub fn time_interval(&self) -> (f64, f64) {
        self.time
    }

    /// Return the canonical timestamp unit.
    #[inline]
    pub fn timestamp_unit(&self) -> TimestampUnit {
        self.unit
    }

    /// Whether both point and time boxes refine `coarse` without unit changes.
    pub fn refines(&self, coarse: &Self) -> Result<bool, VectorMetricError> {
        if self.unit != coarse.unit {
            return Err(VectorMetricError::MixedTimestampUnits);
        }
        Ok(self.sample.refines(&coarse.sample)?
            && coarse.time.0 <= self.time.0
            && self.time.1 <= coarse.time.1)
    }
}

/// Metric vector TWED over explicit physical timestamps and one fixed point
/// sentinel.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTimestampedTwedMetric {
    ground: FixedChannelMetric,
    sentinel: VectorSample,
    nu: f64,
    lambda: f64,
}

impl VectorTimestampedTwedMetric {
    /// Construct a typed timestamped TWED metric with `nu > 0` and
    /// `lambda >= 0`.
    pub fn try_new(
        ground: FixedChannelMetric,
        sentinel: VectorSample,
        nu: f64,
        lambda: f64,
    ) -> Result<Self, VectorMetricError> {
        ground.validate_sample(&sentinel)?;
        if !nu.is_finite() || nu <= 0.0 {
            return Err(VectorMetricError::InvalidStiffness);
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(VectorMetricError::InvalidGapPenalty);
        }
        Ok(Self {
            ground,
            sentinel,
            nu,
            lambda,
        })
    }

    /// Return strictly positive temporal stiffness.
    #[inline]
    pub fn stiffness(&self) -> f64 {
        self.nu
    }

    /// Return the nonnegative gap penalty.
    #[inline]
    pub fn gap_penalty(&self) -> f64 {
        self.lambda
    }

    /// Borrow the fixed typed point sentinel.
    #[inline]
    pub fn sentinel(&self) -> &VectorSample {
        &self.sentinel
    }

    /// Validate one nonempty vector/time series in this metric's channel
    /// domain.
    #[allow(clippy::too_many_arguments)]
    pub fn try_series(
        &self,
        samples: Vec<VectorSample>,
        timestamps: &[f64],
        unit: TimestampUnit,
        origin: f64,
        limits: ResourceLimits,
    ) -> Result<TimestampedVectorSeries, VectorMetricError> {
        if samples.len() != timestamps.len() {
            return Err(VectorMetricError::TimestampLengthMismatch {
                samples: samples.len(),
                timestamps: timestamps.len(),
            });
        }
        if samples.is_empty() {
            return Err(VectorMetricError::EmptyTimestampedSeries);
        }
        let series =
            ChannelVectorSeries::try_new(self.ground.layout.try_clone()?, samples, limits)?;
        validate_vector_timestamps(timestamps, origin)?;
        let requested =
            timestamps
                .len()
                .checked_mul(size_of::<f64>())
                .ok_or(VectorMetricError::Resource(
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    },
                ))?;
        if requested > limits.max_scratch_bytes {
            return Err(VectorMetricError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested,
                },
            ));
        }
        let mut timestamp_storage = Vec::new();
        timestamp_storage
            .try_reserve_exact(timestamps.len())
            .map_err(|_| {
                VectorMetricError::Resource(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested,
                })
            })?;
        timestamp_storage.extend_from_slice(timestamps);
        Ok(TimestampedVectorSeries {
            series,
            timestamps: timestamp_storage,
            unit,
            origin,
        })
    }

    /// K1 deletion lower bound between consecutive timestamped dictionary
    /// boxes. Ignoring cross-coordinate and monotonic-time correlations can
    /// weaken this value but cannot make it inadmissible.
    pub fn interval_delete_lower_bound(
        &self,
        current: &TimestampedVectorBox,
        previous: &TimestampedVectorBox,
    ) -> Result<f64, VectorMetricError> {
        if current.unit != previous.unit {
            return Err(VectorMetricError::MixedTimestampUnits);
        }
        let spatial = self
            .ground
            .box_box_lower_bound(&current.sample, &previous.sample)?;
        let temporal = self.nu * interval_gap(current.time, previous.time);
        Ok(WeightedCost::combine(
            WeightedCost::combine(spatial, temporal),
            self.lambda,
        ))
    }

    /// K1 match lower bound between two exact query points and two candidate
    /// boxes carrying explicit physical-time intervals.
    #[allow(clippy::too_many_arguments)]
    pub fn interval_match_lower_bound(
        &self,
        query_current: &VectorSample,
        query_previous: &VectorSample,
        query_current_time: f64,
        query_previous_time: f64,
        candidate_current: &TimestampedVectorBox,
        candidate_previous: &TimestampedVectorBox,
    ) -> Result<f64, VectorMetricError> {
        if candidate_current.unit != candidate_previous.unit {
            return Err(VectorMetricError::MixedTimestampUnits);
        }
        if !query_current_time.is_finite() || !query_previous_time.is_finite() {
            return Err(VectorMetricError::NonFiniteTimestamp { index: None });
        }
        let spatial = WeightedCost::combine(
            self.ground
                .point_box_lower_bound(query_current, &candidate_current.sample)?,
            self.ground
                .point_box_lower_bound(query_previous, &candidate_previous.sample)?,
        );
        let temporal = self.nu
            * (interval_distance(
                query_current_time,
                candidate_current.time.0,
                candidate_current.time.1,
            ) + interval_distance(
                query_previous_time,
                candidate_previous.time.0,
                candidate_previous.time.1,
            ));
        Ok(WeightedCost::combine(spatial, temporal))
    }

    /// K4 identity lower bound. Timestamp-aware product implementations may
    /// strengthen it, but zero is coherent for every lawful pair.
    #[inline]
    pub fn candidate_lower_bound(
        &self,
        left: &TimestampedVectorSeries,
        right: &TimestampedVectorSeries,
    ) -> Result<f64, VectorMetricError> {
        self.validate_pair(left, right)?;
        Ok(WeightedCost::ZERO)
    }

    /// Exact stack-safe explicit-time vector TWED cutoff decision.
    pub fn distance_bounded(
        &self,
        left: &TimestampedVectorSeries,
        right: &TimestampedVectorSeries,
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, VectorMetricError> {
        self.validate_pair(left, right)?;
        validate_vector_cutoff(cutoff)?;
        validate_vector_lengths(left.samples().len(), right.samples().len(), limits)?;

        let mut ledger = ResourceLedger::new(limits);
        let cells = matrix_cells(left.samples().len(), right.samples().len());
        let distance_evaluations = left
            .samples()
            .len()
            .checked_mul(right.samples().len())
            .and_then(|inner| inner.checked_mul(3))
            .and_then(|inner| inner.checked_add(left.samples().len()))
            .and_then(|total| total.checked_add(right.samples().len()));
        let work_width = self.ground.layout.dimension().checked_add(1);
        let work = distance_evaluations
            .and_then(|count| work_width.and_then(|width| count.checked_mul(width)));
        let scratch = rolling_scratch_bytes(left.samples().len(), right.samples().len());
        let (Some(cells), Some(work), Some(scratch)) = (cells, work, scratch) else {
            return Ok(incomplete(
                ledger,
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                },
            ));
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, cells),
            (ResourceKind::WorkUnits, work),
            (ResourceKind::ScratchBytes, scratch),
        ]) {
            return Ok(incomplete(ledger, reason));
        }

        let exact = match vector_timestamped_twed_with_cutoff(self, left, right, cutoff) {
            Ok(exact) => exact,
            Err(reason) => return Ok(incomplete(ledger, reason)),
        };
        Ok(exact_score_outcome(exact, cutoff, ledger))
    }

    fn validate_pair(
        &self,
        left: &TimestampedVectorSeries,
        right: &TimestampedVectorSeries,
    ) -> Result<(), VectorMetricError> {
        self.ground.validate_layout(left.channel_layout())?;
        self.ground.validate_layout(right.channel_layout())?;
        if left.unit != right.unit {
            return Err(VectorMetricError::MixedTimestampUnits);
        }
        if left.origin != right.origin {
            return Err(VectorMetricError::MixedTimestampOrigins);
        }
        Ok(())
    }
}

fn vector_timestamped_twed_with_cutoff(
    config: &VectorTimestampedTwedMetric,
    left: &TimestampedVectorSeries,
    right: &TimestampedVectorSeries,
    cutoff: f64,
) -> Result<Option<f64>, IncompleteReason> {
    let (left, right) = if right.samples().len() > left.samples().len() {
        (right, left)
    } else {
        (left, right)
    };
    let width =
        right
            .samples()
            .len()
            .checked_add(1)
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
    let (mut previous, mut current) = try_cost_rows(width)?;
    previous[0] = WeightedCost::ZERO;

    let mut previous_sample = &config.sentinel;
    let mut previous_time = right.origin;
    for (index, sample) in right.samples().iter().enumerate() {
        previous[index + 1] = WeightedCost::combine(
            previous[index],
            vector_twed_delete_cost(
                &config.ground,
                sample,
                previous_sample,
                right.timestamps[index],
                previous_time,
                config.nu,
                config.lambda,
            ),
        );
        previous_sample = sample;
        previous_time = right.timestamps[index];
    }

    let mut left_previous_sample = &config.sentinel;
    let mut left_previous_time = left.origin;
    for (left_index, left_sample) in left.samples().iter().enumerate() {
        let left_time = left.timestamps[left_index];
        let delete_left = vector_twed_delete_cost(
            &config.ground,
            left_sample,
            left_previous_sample,
            left_time,
            left_previous_time,
            config.nu,
            config.lambda,
        );
        current[0] = WeightedCost::combine(previous[0], delete_left);
        let mut row_min = current[0];
        let mut right_previous_sample = &config.sentinel;
        let mut right_previous_time = right.origin;
        for (right_index, right_sample) in right.samples().iter().enumerate() {
            let right_time = right.timestamps[right_index];
            let pair = WeightedCost::combine(
                previous[right_index],
                vector_twed_match_cost(
                    &config.ground,
                    left_sample,
                    left_previous_sample,
                    left_time,
                    left_previous_time,
                    right_sample,
                    right_previous_sample,
                    right_time,
                    right_previous_time,
                    config.nu,
                ),
            );
            let delete_left = WeightedCost::combine(previous[right_index + 1], delete_left);
            let delete_right = WeightedCost::combine(
                current[right_index],
                vector_twed_delete_cost(
                    &config.ground,
                    right_sample,
                    right_previous_sample,
                    right_time,
                    right_previous_time,
                    config.nu,
                    config.lambda,
                ),
            );
            current[right_index + 1] = pair.min(delete_left).min(delete_right);
            row_min = row_min.min(current[right_index + 1]);
            right_previous_sample = right_sample;
            right_previous_time = right_time;
        }
        if !WeightedCost::within(row_min, cutoff) {
            return Ok(None);
        }
        std::mem::swap(&mut previous, &mut current);
        left_previous_sample = left_sample;
        left_previous_time = left_time;
    }
    let exact = previous[right.samples().len()];
    Ok(WeightedCost::within(exact, cutoff).then_some(exact))
}

#[allow(clippy::too_many_arguments)]
fn vector_twed_delete_cost(
    ground: &FixedChannelMetric,
    current: &VectorSample,
    previous: &VectorSample,
    current_time: f64,
    previous_time: f64,
    nu: f64,
    lambda: f64,
) -> f64 {
    WeightedCost::combine(
        WeightedCost::combine(
            ground.distance_unchecked(current, previous),
            nu * (current_time - previous_time),
        ),
        lambda,
    )
}

#[allow(clippy::too_many_arguments)]
fn vector_twed_match_cost(
    ground: &FixedChannelMetric,
    left: &VectorSample,
    left_previous: &VectorSample,
    left_time: f64,
    left_previous_time: f64,
    right: &VectorSample,
    right_previous: &VectorSample,
    right_time: f64,
    right_previous_time: f64,
    nu: f64,
) -> f64 {
    let spatial = WeightedCost::combine(
        ground.distance_unchecked(left, right),
        ground.distance_unchecked(left_previous, right_previous),
    );
    let temporal =
        nu * ((left_time - right_time).abs() + (left_previous_time - right_previous_time).abs());
    WeightedCost::combine(spatial, temporal)
}

/// Exact, explicitly nonmetric vector banded-DTW diagnostic scorer.
///
/// Local costs are the square of the fixed channel metric, accumulated inside
/// a required symmetric Sakoe–Chiba band. Public scores are square roots.
/// The type intentionally implements no metric marker: repetition can give
/// distinct raw series distance zero and DTW violates the triangle law.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorBandedDtwScorer {
    ground: FixedChannelMetric,
    band: usize,
}

impl VectorBandedDtwScorer {
    /// Construct a diagnostic scorer with an explicit inclusive half-band.
    #[inline]
    pub fn new(ground: FixedChannelMetric, band: usize) -> Self {
        Self { ground, band }
    }

    /// Return the required inclusive Sakoe–Chiba half-band.
    #[inline]
    pub fn band(&self) -> usize {
        self.band
    }

    /// Validate a possibly empty raw vector series in the fixed channel
    /// domain. No stutter quotient is applied to diagnostic DTW inputs.
    pub fn try_series(
        &self,
        samples: Vec<VectorSample>,
        limits: ResourceLimits,
    ) -> Result<ChannelVectorSeries, VectorMetricError> {
        ChannelVectorSeries::try_new(self.ground.layout.try_clone()?, samples, limits)
    }

    /// K1 squared local-cost lower bound for an interval vector edge.
    pub fn interval_local_lower_bound_squared(
        &self,
        query: &VectorSample,
        candidate: &VectorBox,
    ) -> Result<f64, VectorMetricError> {
        let distance = self.ground.point_box_lower_bound(query, candidate)?;
        Ok(distance * distance)
    }

    /// K4 identity lower bound, always coherent for a nonnegative scorer.
    pub fn candidate_lower_bound(
        &self,
        left: &ChannelVectorSeries,
        right: &ChannelVectorSeries,
    ) -> Result<f64, VectorMetricError> {
        self.validate_pair(left, right)?;
        Ok(WeightedCost::ZERO)
    }

    /// Exact stack-safe banded-DTW decision with work proportional to live
    /// band cells and two retained DP rows.
    pub fn distance_bounded(
        &self,
        left: &ChannelVectorSeries,
        right: &ChannelVectorSeries,
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, VectorMetricError> {
        self.validate_pair(left, right)?;
        validate_vector_cutoff(cutoff)?;
        validate_vector_lengths(left.samples().len(), right.samples().len(), limits)?;
        if self.band > limits.max_band_width {
            return Err(TemporalValidationError::InvalidConfiguration(
                "vector DTW band exceeds max_band_width",
            )
            .into());
        }

        let mut ledger = ResourceLedger::new(limits);
        let cells = banded_cell_count(left.samples().len(), right.samples().len(), self.band);
        let work = cells.and_then(|count| count.checked_mul(self.ground.layout.dimension()));
        let scratch = rolling_scratch_bytes(left.samples().len(), right.samples().len());
        let (Some(cells), Some(work), Some(scratch)) = (cells, work, scratch) else {
            return Ok(incomplete(
                ledger,
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                },
            ));
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, cells),
            (ResourceKind::WorkUnits, work),
            (ResourceKind::ScratchBytes, scratch),
        ]) {
            return Ok(incomplete(ledger, reason));
        }

        let result =
            match vector_banded_dtw_with_cutoff(self, left.samples(), right.samples(), cutoff) {
                Ok(result) => result,
                Err(reason) => return Ok(incomplete(ledger, reason)),
            };
        let usage = ledger.usage();
        Ok(match result {
            VectorDtwScore::Within(distance) => OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff {
                    distance,
                    witness: NoWitness,
                },
                usage,
            },
            VectorDtwScore::Above => OperationOutcome::Complete {
                value: ExactDecision::AboveCutoff,
                usage,
            },
            VectorDtwScore::NoAlignment => OperationOutcome::Complete {
                value: ExactDecision::NoFiniteAlignment,
                usage,
            },
            VectorDtwScore::Overflow => OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::NumericOverflow,
                continuation: None,
                usage,
            },
        })
    }

    fn validate_pair(
        &self,
        left: &ChannelVectorSeries,
        right: &ChannelVectorSeries,
    ) -> Result<(), VectorMetricError> {
        self.ground.validate_layout(left.channel_layout())?;
        self.ground.validate_layout(right.channel_layout())?;
        Ok(())
    }
}

enum VectorDtwScore {
    Within(f64),
    Above,
    NoAlignment,
    Overflow,
}

fn vector_banded_dtw_with_cutoff(
    config: &VectorBandedDtwScorer,
    left: &[VectorSample],
    right: &[VectorSample],
    cutoff: f64,
) -> Result<VectorDtwScore, IncompleteReason> {
    match (left.is_empty(), right.is_empty()) {
        (true, true) => return Ok(VectorDtwScore::Within(0.0)),
        (true, false) | (false, true) => return Ok(VectorDtwScore::NoAlignment),
        (false, false) => {}
    }
    if left.len().abs_diff(right.len()) > config.band {
        return Ok(VectorDtwScore::NoAlignment);
    }
    let (left, right) = if right.len() > left.len() {
        (right, left)
    } else {
        (left, right)
    };
    let width = right
        .len()
        .checked_add(1)
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })?;
    let (mut previous, mut current) = try_cost_rows(width)?;
    previous[0] = WeightedCost::ZERO;

    for (left_index, left_sample) in left.iter().enumerate() {
        current.fill(WeightedCost::TOP);
        let row = left_index + 1;
        let start = row.saturating_sub(config.band).max(1);
        let end = row.saturating_add(config.band).min(right.len());
        if start > end {
            return Ok(VectorDtwScore::NoAlignment);
        }
        let mut row_min = WeightedCost::TOP;
        for column in start..=end {
            let distance = config
                .ground
                .distance_unchecked(left_sample, &right[column - 1]);
            let local = distance * distance;
            let predecessor = previous[column - 1]
                .min(previous[column])
                .min(current[column - 1]);
            current[column] = WeightedCost::combine(predecessor, local);
            row_min = row_min.min(current[column]);
        }
        // Cutoff membership belongs to the public square-root score domain.
        // Squaring an already-rounded public cutoff can round downward and
        // reject the exact score that produced that cutoff.  A square root per
        // row is conservative, preserves inclusive binary64 membership, and
        // remains negligible beside the row's point-distance evaluations.
        if cutoff.is_finite()
            && (!row_min.is_finite() || !WeightedCost::within(row_min.sqrt(), cutoff))
        {
            return Ok(VectorDtwScore::Above);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let exact = previous[right.len()];
    let public_score = exact.sqrt();
    if exact.is_finite() && WeightedCost::within(public_score, cutoff) {
        Ok(VectorDtwScore::Within(public_score))
    } else if cutoff.is_finite() {
        Ok(VectorDtwScore::Above)
    } else if exact == WeightedCost::TOP {
        Ok(VectorDtwScore::Overflow)
    } else {
        Ok(VectorDtwScore::NoAlignment)
    }
}

/// Audited public decision for vector MSM support.
///
/// Scalar MSM's split/merge cost depends on one-dimensional betweenness. A
/// coordinatewise or norm-only replacement is not canonical and currently has
/// neither a reviewed metric proof nor K1 interval proof. The crate therefore
/// rejects a nominal vector MSM rather than silently flattening channels or
/// advertising an unproved metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VectorMsmSupportDecision {
    /// No canonical vector betweenness relation has been selected and proved.
    UnsupportedNoCanonicalBetweenness,
}

/// Machine-readable vector MSM support decision.
pub const VECTOR_MSM_SUPPORT: VectorMsmSupportDecision =
    VectorMsmSupportDecision::UnsupportedNoCanonicalBetweenness;

/// Canonical nonempty vector path modulo consecutive identical stutters.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorFrechetPath {
    samples: Vec<VectorSample>,
    dimension: usize,
}

impl VectorFrechetPath {
    /// Validate a path and collapse every maximal run of identical points.
    pub fn try_new(
        mut samples: Vec<VectorSample>,
        limits: ResourceLimits,
    ) -> Result<Self, VectorMetricError> {
        if samples.is_empty() {
            return Err(VectorMetricError::EmptyPath);
        }
        if samples.len() > limits.max_series_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Query,
                len: samples.len(),
                limit: limits.max_series_len,
            }
            .into());
        }
        let dimension = samples[0].dimension();
        for sample in &samples {
            if sample.dimension() != dimension {
                return Err(VectorMetricError::DimensionMismatch {
                    expected: dimension,
                    observed: sample.dimension(),
                });
            }
        }
        samples.dedup();
        Ok(Self { samples, dimension })
    }

    /// Validate rows directly without flattening channel boundaries.
    pub fn try_from_rows(
        rows: &[&[f64]],
        limits: ResourceLimits,
    ) -> Result<Self, VectorMetricError> {
        let requested = rows.len().checked_mul(size_of::<VectorSample>()).ok_or(
            VectorMetricError::Resource(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            }),
        )?;
        let mut samples = Vec::new();
        samples.try_reserve_exact(rows.len()).map_err(|_| {
            VectorMetricError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested,
            })
        })?;
        for row in rows {
            samples.push(VectorSample::try_new(row, limits)?);
        }
        Self::try_new(samples, limits)
    }

    /// Return the fixed coordinate count of every point.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow the stutter-free vector samples.
    #[inline]
    pub fn samples(&self) -> &[VectorSample] {
        &self.samples
    }
}

/// Discrete Frechet metric over vector paths and an audited ground metric.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorFrechetMetric<M: GroundMetric> {
    ground: M,
}

impl<M: GroundMetric> VectorFrechetMetric<M> {
    /// Construct a vector-path metric from an audited ground metric.
    #[inline]
    pub fn new(ground: M) -> Self {
        Self { ground }
    }

    /// Borrow the point metric.
    #[inline]
    pub fn ground_metric(&self) -> &M {
        &self.ground
    }

    /// Compute exact vector discrete Frechet under deterministic hard limits.
    pub fn distance_bounded(
        &self,
        left: &VectorFrechetPath,
        right: &VectorFrechetPath,
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, VectorMetricError> {
        if left.dimension != right.dimension {
            return Err(VectorMetricError::DimensionMismatch {
                expected: left.dimension,
                observed: right.dimension,
            });
        }
        if let Some(expected) = self.ground.required_dimension() {
            if left.dimension != expected {
                return Err(VectorMetricError::DimensionMismatch {
                    expected,
                    observed: left.dimension,
                });
            }
        }
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }
        if left.samples.len() > limits.max_series_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Query,
                len: left.samples.len(),
                limit: limits.max_series_len,
            }
            .into());
        }
        if right.samples.len() > limits.max_series_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Candidate,
                len: right.samples.len(),
                limit: limits.max_series_len,
            }
            .into());
        }

        let mut ledger = ResourceLedger::new(limits);
        let cells = left.samples.len().checked_mul(right.samples.len());
        let coordinate_work = cells.and_then(|count| count.checked_mul(left.dimension));
        let scratch = left
            .samples
            .len()
            .min(right.samples.len())
            .checked_mul(2)
            .and_then(|slots| slots.checked_mul(std::mem::size_of::<f64>()));
        let (Some(cells), Some(coordinate_work), Some(scratch)) = (cells, coordinate_work, scratch)
        else {
            return Ok(incomplete(
                ledger,
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                },
            ));
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, cells),
            (ResourceKind::WorkUnits, coordinate_work),
            (ResourceKind::ScratchBytes, scratch),
        ]) {
            return Ok(incomplete(ledger, reason));
        }

        let exact = match vector_frechet_with_cutoff(&self.ground, left, right, cutoff) {
            Ok(exact) => exact,
            Err(reason) => return Ok(incomplete(ledger, reason)),
        };
        let usage = ledger.usage();
        Ok(match exact {
            Some(distance) if distance.is_finite() => OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff {
                    distance,
                    witness: NoWitness,
                },
                usage,
            },
            Some(_) | None if cutoff.is_finite() => OperationOutcome::Complete {
                value: ExactDecision::AboveCutoff,
                usage,
            },
            Some(_) | None => OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::NumericOverflow,
                continuation: None,
                usage,
            },
        })
    }
}

impl VectorFrechetMetric<FixedChannelMetric> {
    /// K1 local bottleneck-link lower bound for a vector dictionary box.
    #[inline]
    pub fn interval_link_lower_bound(
        &self,
        query: &VectorSample,
        candidate: &VectorBox,
    ) -> Result<f64, VectorMetricError> {
        self.ground.point_box_lower_bound(query, candidate)
    }

    /// K4 endpoint lower bound on complete canonical paths.
    pub fn candidate_lower_bound(
        &self,
        left: &VectorFrechetPath,
        right: &VectorFrechetPath,
    ) -> Result<f64, VectorMetricError> {
        let expected = self.ground.layout.dimension();
        if left.dimension != expected {
            return Err(VectorMetricError::DimensionMismatch {
                expected,
                observed: left.dimension,
            });
        }
        if right.dimension != expected {
            return Err(VectorMetricError::DimensionMismatch {
                expected,
                observed: right.dimension,
            });
        }
        let first = self
            .ground
            .distance_unchecked(&left.samples[0], &right.samples[0]);
        let last = self.ground.distance_unchecked(
            left.samples.last().expect("Fréchet paths are nonempty"),
            right.samples.last().expect("Fréchet paths are nonempty"),
        );
        Ok(first.max(last))
    }
}

fn vector_frechet_with_cutoff<M: GroundMetric>(
    ground: &M,
    left: &VectorFrechetPath,
    right: &VectorFrechetPath,
    cutoff: f64,
) -> Result<Option<f64>, IncompleteReason> {
    let (left, right) = if right.samples.len() > left.samples.len() {
        (right, left)
    } else {
        (left, right)
    };
    let (mut previous, mut current) = try_cost_rows(right.samples.len())?;
    previous.fill(BottleneckCost::TOP);
    current.fill(BottleneckCost::TOP);
    previous[0] = ground.distance(&left.samples[0], &right.samples[0]);
    for column in 1..right.samples.len() {
        previous[column] = BottleneckCost::combine(
            previous[column - 1],
            ground.distance(&left.samples[0], &right.samples[column]),
        );
    }
    for left_sample in &left.samples[1..] {
        current[0] =
            BottleneckCost::combine(previous[0], ground.distance(left_sample, &right.samples[0]));
        let mut row_min = current[0];
        for column in 1..right.samples.len() {
            let predecessor = previous[column - 1]
                .min(previous[column])
                .min(current[column - 1]);
            current[column] = BottleneckCost::combine(
                predecessor,
                ground.distance(left_sample, &right.samples[column]),
            );
            row_min = row_min.min(current[column]);
        }
        if !BottleneckCost::within(row_min, cutoff) {
            return Ok(None);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let exact = previous[right.samples.len() - 1];
    Ok(BottleneckCost::within(exact, cutoff).then_some(exact))
}

/// Fixed-query, stack-safe online vector discrete-Fréchet automaton.
///
/// Whole vector points are labels: coordinates are never flattened and an
/// alignment transition cannot cross a channel boundary. The machine keeps
/// two query-width generations and sorted active-row IDs, consumes an unknown
/// number of target points, and retains no target prefix.
pub struct VectorFrechetOnlineAutomaton<M: GroundMetric> {
    query: VectorFrechetPath,
    ground: M,
    cutoff: f64,
    limits: OnlineAutomatonLimits,
    current: Vec<f64>,
    next: Vec<f64>,
    current_active: Vec<usize>,
    next_active: Vec<usize>,
    consumed_target_len: usize,
    scratch_bytes: usize,
}

impl<M: GroundMetric> std::fmt::Debug for VectorFrechetOnlineAutomaton<M> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VectorFrechetOnlineAutomaton")
            .field("query_len", &self.query.samples.len())
            .field("dimension", &self.query.dimension)
            .field("cutoff", &self.cutoff)
            .field("active_positions", &self.current_active.len())
            .field("consumed_target_len", &self.consumed_target_len)
            .field("scratch_bytes", &self.scratch_bytes)
            .finish_non_exhaustive()
    }
}

impl<M: GroundMetric> VectorFrechetOnlineAutomaton<M> {
    /// Construct a bounded online machine for one canonical fixed query.
    pub fn new(
        query: VectorFrechetPath,
        ground: M,
        cutoff: f64,
        limits: OnlineAutomatonLimits,
    ) -> Result<Self, VectorMetricError> {
        if !cutoff.is_finite() || cutoff < 0.0 {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }
        if let Some(expected) = ground.required_dimension() {
            if query.dimension != expected {
                return Err(VectorMetricError::DimensionMismatch {
                    expected,
                    observed: query.dimension,
                });
            }
        }
        if query.samples.len() > limits.max_query_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Query,
                len: query.samples.len(),
                limit: limits.max_query_len,
            }
            .into());
        }
        let width = query.samples.len();
        if width > limits.max_frontier_positions {
            return Err(VectorMetricError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::QueueEntries,
                    limit: limits.max_frontier_positions,
                    requested: width,
                },
            ));
        }
        let query_bytes = width
            .checked_mul(query.dimension)
            .and_then(|coordinates| coordinates.checked_mul(size_of::<f64>()));
        let generation_bytes = width
            .checked_mul(2)
            .and_then(|slots| slots.checked_mul(size_of::<f64>()))
            .and_then(|costs| {
                width
                    .checked_mul(2)
                    .and_then(|slots| slots.checked_mul(size_of::<usize>()))
                    .and_then(|active| costs.checked_add(active))
            });
        let scratch_bytes = query_bytes
            .and_then(|query| generation_bytes.and_then(|state| query.checked_add(state)))
            .ok_or(VectorMetricError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        if scratch_bytes > limits.max_scratch_bytes {
            return Err(VectorMetricError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: scratch_bytes,
                },
            ));
        }

        let mut current = Vec::new();
        reserve_vector_storage(&mut current, width, scratch_bytes)?;
        current.resize(width, BottleneckCost::TOP);
        let mut next = Vec::new();
        reserve_vector_storage(&mut next, width, scratch_bytes)?;
        next.resize(width, BottleneckCost::TOP);
        let mut current_active = Vec::new();
        reserve_vector_indices(&mut current_active, width, scratch_bytes)?;
        let mut next_active = Vec::new();
        reserve_vector_indices(&mut next_active, width, scratch_bytes)?;

        Ok(Self {
            query,
            ground,
            cutoff,
            limits,
            current,
            next,
            current_active,
            next_active,
            consumed_target_len: 0,
            scratch_bytes,
        })
    }

    /// Observe the already committed target prefix.
    pub fn observation(&self) -> VectorFrechetOnlineObservation {
        let distance_within_cutoff = (self.consumed_target_len > 0)
            .then(|| self.current.last().copied())
            .flatten()
            .filter(|cost| valid_online_cost(*cost, self.cutoff));
        let minimum_active_cost = self
            .current_active
            .iter()
            .filter_map(|row| self.current.get(*row).copied())
            .min_by(f64::total_cmp);
        VectorFrechetOnlineObservation {
            consumed_target_len: self.consumed_target_len,
            active_positions: self.current_active.len(),
            distance_within_cutoff,
            minimum_active_cost,
        }
    }

    /// Consume one finite point of the fixed vector dimension.
    pub fn advance(
        &mut self,
        target: &VectorSample,
    ) -> Result<OnlineStepOutcome<VectorFrechetOnlineObservation>, VectorMetricError> {
        if target.dimension() != self.query.dimension {
            return Err(VectorMetricError::DimensionMismatch {
                expected: self.query.dimension,
                observed: target.dimension(),
            });
        }
        let next_depth =
            self.consumed_target_len
                .checked_add(1)
                .ok_or(VectorMetricError::Resource(
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::SeriesLength,
                    },
                ))?;

        while let Some(row) = self.next_active.pop() {
            self.next[row] = BottleneckCost::TOP;
        }
        let mut work = 0usize;
        if next_depth == 1 {
            for row in 0..self.query.samples.len() {
                if let Err(requested) = charge_vector_work(
                    &mut work,
                    self.query.dimension,
                    self.limits.max_step_work_units,
                ) {
                    return Ok(self.incomplete_work(requested, work));
                }
                let predecessor = if row == 0 {
                    BottleneckCost::ZERO
                } else {
                    self.next[row - 1]
                };
                let cost = BottleneckCost::combine(
                    predecessor,
                    self.ground.distance(&self.query.samples[row], target),
                );
                if !valid_online_cost(cost, self.cutoff) {
                    break;
                }
                self.next[row] = cost;
                self.next_active.push(row);
            }
        } else if self.current_active.is_empty() {
            if charge_work(&mut work, self.limits.max_step_work_units).is_err() {
                return Ok(self.incomplete_work(1, 0));
            }
        } else {
            let last_row = self.query.samples.len() - 1;
            let mut seeds = NeighborSeedRows::new(&self.current_active, last_row);
            let mut next_seed = seeds.next();
            while let Some(start) = next_seed {
                let mut row = start;
                loop {
                    while next_seed == Some(row) {
                        next_seed = seeds.next();
                    }
                    if let Err(requested) = charge_vector_work(
                        &mut work,
                        self.query.dimension,
                        self.limits.max_step_work_units,
                    ) {
                        return Ok(self.incomplete_work(requested, work));
                    }
                    let diagonal = row
                        .checked_sub(1)
                        .and_then(|index| self.current.get(index))
                        .copied()
                        .unwrap_or(BottleneckCost::TOP);
                    let same = self.current[row];
                    let vertical = row
                        .checked_sub(1)
                        .and_then(|index| self.next.get(index))
                        .copied()
                        .unwrap_or(BottleneckCost::TOP);
                    let cost = BottleneckCost::combine(
                        diagonal.min(same).min(vertical),
                        self.ground.distance(&self.query.samples[row], target),
                    );
                    if !cost.is_finite() && cost != BottleneckCost::TOP {
                        return Ok(self.incomplete_numeric(work));
                    }
                    if valid_online_cost(cost, self.cutoff) {
                        self.next[row] = cost;
                        self.next_active.push(row);
                        if row < last_row {
                            row += 1;
                            continue;
                        }
                    }
                    break;
                }
            }
        }

        std::mem::swap(&mut self.current, &mut self.next);
        std::mem::swap(&mut self.current_active, &mut self.next_active);
        self.consumed_target_len = next_depth;
        Ok(OnlineStepOutcome::Advanced {
            value: self.observation(),
            usage: self.step_usage(work),
        })
    }

    /// Fixed retained logical bytes; independent of target-prefix length.
    #[inline]
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }

    fn incomplete_work(
        &mut self,
        requested: usize,
        completed: usize,
    ) -> OnlineStepOutcome<VectorFrechetOnlineObservation> {
        self.clear_speculative();
        OnlineStepOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                limit: self.limits.max_step_work_units,
                requested,
            },
            usage: self.step_usage(completed),
        }
    }

    fn incomplete_numeric(
        &mut self,
        completed: usize,
    ) -> OnlineStepOutcome<VectorFrechetOnlineObservation> {
        self.clear_speculative();
        OnlineStepOutcome::Incomplete {
            reason: IncompleteReason::NumericOverflow,
            usage: self.step_usage(completed),
        }
    }

    fn clear_speculative(&mut self) {
        while let Some(row) = self.next_active.pop() {
            self.next[row] = BottleneckCost::TOP;
        }
    }

    fn step_usage(&self, work: usize) -> ResourceUsage {
        ResourceUsage {
            dp_cells: work.checked_div(self.query.dimension).unwrap_or(0),
            work_units: work,
            scratch_bytes: self.scratch_bytes,
            queue_entries: self.current_active.len(),
            ..ResourceUsage::default()
        }
    }
}

#[inline]
fn charge_vector_work(work: &mut usize, amount: usize, limit: usize) -> Result<(), usize> {
    let requested = work.checked_add(amount).unwrap_or(usize::MAX);
    if requested > limit {
        Err(requested)
    } else {
        *work = requested;
        Ok(())
    }
}

#[inline]
fn valid_online_cost(cost: f64, cutoff: f64) -> bool {
    cost.is_finite() && cost >= 0.0 && cost <= cutoff
}

fn reserve_vector_storage(
    storage: &mut Vec<f64>,
    width: usize,
    requested: usize,
) -> Result<(), VectorMetricError> {
    storage.try_reserve_exact(width).map_err(|_| {
        VectorMetricError::Resource(IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })
    })
}

fn reserve_vector_indices(
    storage: &mut Vec<usize>,
    width: usize,
    requested: usize,
) -> Result<(), VectorMetricError> {
    storage.try_reserve_exact(width).map_err(|_| {
        VectorMetricError::Resource(IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })
    })
}

fn point_pairs<'a>(
    left: &'a VectorSample,
    right: &'a VectorSample,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    debug_assert_eq!(left.dimension(), right.dimension());
    left.coordinates
        .iter()
        .copied()
        .zip(right.coordinates.iter().copied())
}

#[inline]
fn interval_distance(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low - value
    } else if value > high {
        value - high
    } else {
        0.0
    }
}

#[inline]
fn interval_gap(left: (f64, f64), right: (f64, f64)) -> f64 {
    if left.1 < right.0 {
        right.0 - left.1
    } else if right.1 < left.0 {
        left.0 - right.1
    } else {
        0.0
    }
}

fn validate_vector_cutoff(cutoff: f64) -> Result<(), VectorMetricError> {
    if cutoff.is_nan() || cutoff < 0.0 {
        return Err(TemporalValidationError::InvalidCutoff.into());
    }
    Ok(())
}

fn validate_vector_lengths(
    left: usize,
    right: usize,
    limits: ResourceLimits,
) -> Result<(), VectorMetricError> {
    if left > limits.max_series_len {
        return Err(TemporalValidationError::SeriesTooLong {
            operand: super::bounded::Operand::Query,
            len: left,
            limit: limits.max_series_len,
        }
        .into());
    }
    if right > limits.max_series_len {
        return Err(TemporalValidationError::SeriesTooLong {
            operand: super::bounded::Operand::Candidate,
            len: right,
            limit: limits.max_series_len,
        }
        .into());
    }
    Ok(())
}

fn validate_vector_timestamps(timestamps: &[f64], origin: f64) -> Result<(), VectorMetricError> {
    if !origin.is_finite() {
        return Err(VectorMetricError::NonFiniteTimestamp { index: None });
    }
    if let Some(index) = timestamps
        .iter()
        .position(|timestamp| !timestamp.is_finite())
    {
        return Err(VectorMetricError::NonFiniteTimestamp { index: Some(index) });
    }
    if timestamps
        .first()
        .is_some_and(|timestamp| *timestamp < origin)
    {
        return Err(VectorMetricError::TimestampBeforeOrigin);
    }
    if let Some(index) = timestamps
        .windows(2)
        .position(|pair| pair[1] <= pair[0])
        .map(|index| index + 1)
    {
        return Err(VectorMetricError::NonMonotoneTimestamp { index });
    }
    Ok(())
}

fn matrix_cells(left: usize, right: usize) -> Option<usize> {
    left.checked_add(1)?.checked_mul(right.checked_add(1)?)
}

fn rolling_scratch_bytes(left: usize, right: usize) -> Option<usize> {
    left.min(right)
        .checked_add(1)?
        .checked_mul(2)?
        .checked_mul(size_of::<f64>())
}

fn try_cost_rows(width: usize) -> Result<(Vec<f64>, Vec<f64>), IncompleteReason> {
    let requested = width
        .checked_mul(2)
        .and_then(|slots| slots.checked_mul(size_of::<f64>()))
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })?;
    let mut previous = Vec::new();
    previous
        .try_reserve_exact(width)
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })?;
    previous.resize(width, WeightedCost::TOP);
    let mut current = Vec::new();
    current
        .try_reserve_exact(width)
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })?;
    current.resize(width, WeightedCost::TOP);
    Ok((previous, current))
}

fn banded_cell_count(left: usize, right: usize, band: usize) -> Option<usize> {
    let mut cells = 0usize;
    for row in 1..=left {
        let start = row.saturating_sub(band).max(1);
        let end = row.saturating_add(band).min(right);
        if start <= end {
            cells = cells.checked_add(end - start + 1)?;
        }
    }
    Some(cells)
}

fn exact_score_outcome(
    exact: Option<f64>,
    cutoff: f64,
    ledger: ResourceLedger,
) -> OperationOutcome<ExactDecision> {
    let usage = ledger.usage();
    match exact {
        Some(distance) if distance.is_finite() => OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff {
                distance,
                witness: NoWitness,
            },
            usage,
        },
        Some(_) | None if cutoff.is_finite() => OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            usage,
        },
        Some(_) | None => OperationOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::NumericOverflow,
            continuation: None,
            usage,
        },
    }
}

fn incomplete(ledger: ResourceLedger, reason: IncompleteReason) -> OperationOutcome<ExactDecision> {
    OperationOutcome::Incomplete {
        partial: None,
        reason,
        continuation: None,
        usage: ledger.usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(outcome: OperationOutcome<ExactDecision>) -> f64 {
        match outcome {
            OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff { distance, .. },
                ..
            } => distance,
            other => panic!("expected exact completion, got {other:?}"),
        }
    }

    #[test]
    fn audited_ground_metrics_have_expected_values() {
        let limits = ResourceLimits::default();
        let left = VectorSample::try_new(&[0.0, 0.0], limits).unwrap();
        let right = VectorSample::try_new(&[3.0, 4.0], limits).unwrap();
        assert_eq!(L1GroundMetric.distance(&left, &right), 7.0);
        assert_eq!(L2GroundMetric.distance(&left, &right), 5.0);
        assert_eq!(LinfGroundMetric.distance(&left, &right), 4.0);
    }

    #[test]
    fn vector_frechet_uses_points_and_stutter_quotient() {
        let limits = ResourceLimits::default();
        let left =
            VectorFrechetPath::try_from_rows(&[&[0.0, 0.0], &[0.0, 0.0], &[1.0, 1.0]], limits)
                .unwrap();
        let right = VectorFrechetPath::try_from_rows(&[&[0.0, 0.0], &[1.0, 1.0]], limits).unwrap();
        assert_eq!(left, right);
        let distance = VectorFrechetMetric::new(L2GroundMetric)
            .distance_bounded(&left, &right, f64::INFINITY, limits)
            .unwrap();
        assert_eq!(exact(distance), 0.0);
    }

    #[test]
    fn dimension_mismatch_is_never_flattened() {
        let limits = ResourceLimits::default();
        let left = VectorFrechetPath::try_from_rows(&[&[0.0, 1.0]], limits).unwrap();
        let right = VectorFrechetPath::try_from_rows(&[&[0.0, 1.0, 2.0]], limits).unwrap();
        assert!(matches!(
            VectorFrechetMetric::new(L1GroundMetric).distance_bounded(
                &left,
                &right,
                f64::INFINITY,
                limits
            ),
            Err(VectorMetricError::DimensionMismatch { .. })
        ));
    }
}
