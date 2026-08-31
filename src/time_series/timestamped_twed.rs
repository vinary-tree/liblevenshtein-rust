//! Metric Time Warp Edit Distance over explicit physical timestamps.
//!
//! This is deliberately separate from unit-grid TWED. Every series carries a
//! canonical unit and a shared physical origin; timestamps are finite,
//! strictly increasing, and no earlier than that origin. Consequently the
//! stiffness term uses real elapsed time rather than sample indices.

use thiserror::Error;

use super::bounded::{
    ExactDecision, IncompleteReason, NoWitness, Operand, OperationOutcome, ResourceKind,
    ResourceLedger, ResourceLimits, TemporalValidationError,
};
use crate::cost::{CostMonoid, WeightedCost};

/// Canonical physical unit attached to every explicit timestamp array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TimestampUnit {
    /// SI seconds.
    Seconds,
    /// One thousandth of an SI second.
    Milliseconds,
    /// One millionth of an SI second.
    Microseconds,
    /// One billionth of an SI second.
    Nanoseconds,
}

/// Failure to construct or compare explicit-timestamp TWED values.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TimestampedTwedError {
    /// Value and timestamp arrays had different lengths.
    #[error("value count {values} differs from timestamp count {timestamps}")]
    LengthMismatch {
        /// Number of scalar samples.
        values: usize,
        /// Number of timestamps.
        timestamps: usize,
    },
    /// Metric timestamped TWED is defined here only for nonempty series.
    #[error("timestamped TWED metric series must be nonempty")]
    EmptySeries,
    /// A timestamp or origin was NaN or infinite.
    #[error("timestamp index {index:?} is not finite (None denotes the origin)")]
    NonFiniteTimestamp {
        /// Zero-based timestamp index; `None` denotes the origin.
        index: Option<usize>,
    },
    /// A timestamp was not strictly greater than its predecessor.
    #[error("timestamp {index} is not strictly greater than its predecessor")]
    NonMonotoneTimestamp {
        /// Zero-based index of the offending timestamp.
        index: usize,
    },
    /// The first timestamp preceded the declared physical origin.
    #[error("first timestamp precedes the declared origin")]
    TimestampBeforeOrigin,
    /// Two operands used different canonical units.
    #[error("timestamp units differ")]
    MixedUnits,
    /// Two operands used different physical origins.
    #[error("timestamp origins differ")]
    MixedOrigins,
    /// The metric configuration had a nonpositive or nonfinite stiffness.
    #[error("timestamped TWED stiffness must be finite and strictly positive")]
    InvalidStiffness,
    /// The metric configuration had a negative or nonfinite gap penalty.
    #[error("timestamped TWED gap penalty must be finite and nonnegative")]
    InvalidGapPenalty,
    /// A closed scalar-value interval contained NaN or reversed endpoints.
    #[error("timestamped TWED value interval contains NaN or is reversed")]
    InvalidValueInterval,
    /// A closed physical-time interval had a nonfinite or reversed endpoint.
    #[error("timestamped TWED time interval is nonfinite or reversed")]
    InvalidTimestampInterval,
    /// Scalar values failed the shared bounded validation contract.
    #[error(transparent)]
    InvalidSeries(#[from] TemporalValidationError),
    /// A bounded online machine could not construct its fixed state.
    #[error("timestamped TWED online resource construction failed: {0:?}")]
    Resource(IncompleteReason),
}

/// One abstract scalar/time label for a timestamped-TWED dictionary product.
///
/// The two closed intervals denote every concrete point represented by one
/// quantization label. Correlation between value and time is intentionally
/// forgotten; this can weaken a lower bound but cannot make it inadmissible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampedScalarBox {
    value: (f64, f64),
    time: (f64, f64),
    unit: TimestampUnit,
}

impl TimestampedScalarBox {
    /// Validate a NaN-free ordered value interval and a finite ordered
    /// physical-time interval. Value endpoints may be infinite so the extreme
    /// bins of a clamping quantizer remain admissible; timestamps may not.
    pub fn try_new(
        value: (f64, f64),
        time: (f64, f64),
        unit: TimestampUnit,
    ) -> Result<Self, TimestampedTwedError> {
        if !valid_value_interval(value) {
            return Err(TimestampedTwedError::InvalidValueInterval);
        }
        if !valid_time_interval(time) {
            return Err(TimestampedTwedError::InvalidTimestampInterval);
        }
        Ok(Self { value, time, unit })
    }

    /// Construct the exact singleton abstraction of one concrete point.
    pub fn point(
        value: f64,
        timestamp: f64,
        unit: TimestampUnit,
    ) -> Result<Self, TimestampedTwedError> {
        Self::try_new((value, value), (timestamp, timestamp), unit)
    }

    /// Return the closed scalar-value interval.
    #[inline]
    pub fn value_interval(self) -> (f64, f64) {
        self.value
    }

    /// Return the closed physical-time interval.
    #[inline]
    pub fn time_interval(self) -> (f64, f64) {
        self.time
    }

    /// Return the canonical physical-time unit.
    #[inline]
    pub fn timestamp_unit(self) -> TimestampUnit {
        self.unit
    }

    /// Whether this label is a precision refinement of `coarse`.
    pub fn refines(self, coarse: Self) -> Result<bool, TimestampedTwedError> {
        if self.unit != coarse.unit {
            return Err(TimestampedTwedError::MixedUnits);
        }
        Ok(
            interval_contains(coarse.value, self.value)
                && interval_contains(coarse.time, self.time),
        )
    }
}

/// Validated, nonempty scalar samples paired with physical timestamps.
#[derive(Clone, Debug, PartialEq)]
pub struct TimestampedSeries {
    values: Box<[f64]>,
    timestamps: Box<[f64]>,
    unit: TimestampUnit,
    origin: f64,
}

impl TimestampedSeries {
    /// Validate a series whose physical timestamp origin is zero.
    pub fn try_new(
        values: &[f64],
        timestamps: &[f64],
        unit: TimestampUnit,
        limits: ResourceLimits,
    ) -> Result<Self, TimestampedTwedError> {
        Self::try_new_with_origin(values, timestamps, unit, 0.0, limits)
    }

    /// Validate a series with an explicit origin in the same canonical unit.
    pub fn try_new_with_origin(
        values: &[f64],
        timestamps: &[f64],
        unit: TimestampUnit,
        origin: f64,
        limits: ResourceLimits,
    ) -> Result<Self, TimestampedTwedError> {
        if values.len() != timestamps.len() {
            return Err(TimestampedTwedError::LengthMismatch {
                values: values.len(),
                timestamps: timestamps.len(),
            });
        }
        if values.is_empty() {
            return Err(TimestampedTwedError::EmptySeries);
        }
        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, values)?;
        if !origin.is_finite() {
            return Err(TimestampedTwedError::NonFiniteTimestamp { index: None });
        }
        if let Some(index) = timestamps
            .iter()
            .position(|timestamp| !timestamp.is_finite())
        {
            return Err(TimestampedTwedError::NonFiniteTimestamp { index: Some(index) });
        }
        if timestamps[0] < origin {
            return Err(TimestampedTwedError::TimestampBeforeOrigin);
        }
        if let Some(index) = timestamps
            .windows(2)
            .position(|pair| pair[1] <= pair[0])
            .map(|index| index + 1)
        {
            return Err(TimestampedTwedError::NonMonotoneTimestamp { index });
        }
        let retained_bytes = values
            .len()
            .checked_mul(2)
            .and_then(|slots| slots.checked_mul(std::mem::size_of::<f64>()))
            .ok_or(TimestampedTwedError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        ledger
            .observe_peak(ResourceKind::ScratchBytes, retained_bytes)
            .map_err(TimestampedTwedError::Resource)?;
        let mut value_storage = Vec::new();
        value_storage.try_reserve_exact(values.len()).map_err(|_| {
            TimestampedTwedError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: retained_bytes,
            })
        })?;
        let mut timestamp_storage = Vec::new();
        timestamp_storage
            .try_reserve_exact(timestamps.len())
            .map_err(|_| {
                TimestampedTwedError::Resource(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: retained_bytes,
                })
            })?;
        value_storage.extend_from_slice(values);
        timestamp_storage.extend_from_slice(timestamps);
        Ok(Self {
            values: value_storage.into_boxed_slice(),
            timestamps: timestamp_storage.into_boxed_slice(),
            unit,
            origin,
        })
    }

    /// Borrow the validated scalar values.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Borrow the validated strictly increasing timestamps.
    #[inline]
    pub fn timestamps(&self) -> &[f64] {
        &self.timestamps
    }

    /// Return the canonical timestamp unit.
    #[inline]
    pub fn unit(&self) -> TimestampUnit {
        self.unit
    }

    /// Return the physical timestamp of the synthetic zero-valued predecessor.
    #[inline]
    pub fn origin(&self) -> f64 {
        self.origin
    }
}

/// Validated metric TWED configuration for explicit physical timestamps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricTimestampedTwedConfig {
    nu: f64,
    lambda: f64,
}

impl MetricTimestampedTwedConfig {
    /// Construct physical-time TWED with `nu > 0` and `lambda >= 0`.
    pub fn try_new(nu: f64, lambda: f64) -> Result<Self, TimestampedTwedError> {
        if !nu.is_finite() || nu <= 0.0 {
            return Err(TimestampedTwedError::InvalidStiffness);
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(TimestampedTwedError::InvalidGapPenalty);
        }
        Ok(Self { nu, lambda })
    }

    /// Return the strictly positive cost per physical time unit.
    #[inline]
    pub fn stiffness(&self) -> f64 {
        self.nu
    }

    /// Return the nonnegative edit gap penalty.
    #[inline]
    pub fn gap_penalty(&self) -> f64 {
        self.lambda
    }

    /// Admissible local deletion cost for two consecutive abstract candidate
    /// labels. Singleton boxes reproduce the exact TWED deletion term.
    pub fn interval_delete_lower_bound(
        &self,
        current: TimestampedScalarBox,
        previous: TimestampedScalarBox,
    ) -> Result<f64, TimestampedTwedError> {
        if current.unit != previous.unit {
            return Err(TimestampedTwedError::MixedUnits);
        }
        let value = interval_gap(current.value, previous.value);
        let elapsed = interval_gap(current.time, previous.time);
        Ok(WeightedCost::combine(
            WeightedCost::combine(value, self.nu * elapsed),
            self.lambda,
        ))
    }

    /// Admissible local match cost between two exact query points and two
    /// consecutive abstract candidate labels.
    ///
    /// This is the K1 abstraction used by a lazy dictionary product. It is
    /// deliberately correlation-blind, so refinement can only increase the
    /// bound and singleton labels reproduce the concrete recurrence exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn interval_match_lower_bound(
        &self,
        query_current_value: f64,
        query_previous_value: f64,
        query_current_time: f64,
        query_previous_time: f64,
        query_unit: TimestampUnit,
        candidate_current: TimestampedScalarBox,
        candidate_previous: TimestampedScalarBox,
    ) -> Result<f64, TimestampedTwedError> {
        if candidate_current.unit != candidate_previous.unit || candidate_current.unit != query_unit
        {
            return Err(TimestampedTwedError::MixedUnits);
        }
        for (index, value) in [query_current_value, query_previous_value]
            .into_iter()
            .enumerate()
        {
            if !value.is_finite() {
                return Err(TemporalValidationError::NonFiniteSample {
                    operand: Operand::Query,
                    index,
                }
                .into());
            }
        }
        if !query_current_time.is_finite() || !query_previous_time.is_finite() {
            return Err(TimestampedTwedError::NonFiniteTimestamp { index: None });
        }
        if query_current_time <= query_previous_time {
            return Err(TimestampedTwedError::NonMonotoneTimestamp { index: 1 });
        }
        let value = WeightedCost::combine(
            point_interval_distance(query_current_value, candidate_current.value),
            point_interval_distance(query_previous_value, candidate_previous.value),
        );
        let time = self.nu
            * (point_interval_distance(query_current_time, candidate_current.time)
                + point_interval_distance(query_previous_time, candidate_previous.time));
        Ok(WeightedCost::combine(value, time))
    }

    /// Compute an exact fail-closed cutoff decision with deterministic limits.
    pub fn distance_bounded(
        &self,
        left: &TimestampedSeries,
        right: &TimestampedSeries,
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, TimestampedTwedError> {
        if left.unit != right.unit {
            return Err(TimestampedTwedError::MixedUnits);
        }
        if left.origin != right.origin {
            return Err(TimestampedTwedError::MixedOrigins);
        }
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }

        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_series_len(Operand::Query, left.values.len())?;
        ledger.validate_series_len(Operand::Candidate, right.values.len())?;
        let cells = left.values.len().checked_add(1).and_then(|rows| {
            right
                .values
                .len()
                .checked_add(1)
                .and_then(|columns| rows.checked_mul(columns))
        });
        let work = left
            .values
            .len()
            .checked_mul(right.values.len())
            .and_then(|inner| inner.checked_add(left.values.len()))
            .and_then(|total| total.checked_add(right.values.len()));
        let scratch = left
            .values
            .len()
            .min(right.values.len())
            .checked_add(1)
            .and_then(|width| width.checked_mul(2))
            .and_then(|slots| slots.checked_mul(std::mem::size_of::<f64>()));
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

        let exact = match timestamped_distance_with_cutoff(self, left, right, cutoff) {
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

#[inline]
fn valid_value_interval(interval: (f64, f64)) -> bool {
    !interval.0.is_nan() && !interval.1.is_nan() && interval.0 <= interval.1
}

#[inline]
fn valid_time_interval(interval: (f64, f64)) -> bool {
    interval.0.is_finite() && interval.1.is_finite() && interval.0 <= interval.1
}

#[inline]
fn interval_contains(outer: (f64, f64), inner: (f64, f64)) -> bool {
    outer.0 <= inner.0 && inner.1 <= outer.1
}

#[inline]
fn point_interval_distance(point: f64, interval: (f64, f64)) -> f64 {
    if point < interval.0 {
        interval.0 - point
    } else if point > interval.1 {
        point - interval.1
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

fn incomplete(ledger: ResourceLedger, reason: IncompleteReason) -> OperationOutcome<ExactDecision> {
    OperationOutcome::Incomplete {
        partial: None,
        reason,
        continuation: None,
        usage: ledger.usage(),
    }
}

fn timestamped_distance_with_cutoff(
    config: &MetricTimestampedTwedConfig,
    left: &TimestampedSeries,
    right: &TimestampedSeries,
    cutoff: f64,
) -> Result<Option<f64>, IncompleteReason> {
    // Symmetry permits allocating on the shorter axis without recursive
    // argument swapping, preserving the iterative stack-safety contract.
    let (left, right) = if right.values.len() > left.values.len() {
        (right, left)
    } else {
        (left, right)
    };

    let row_len =
        right
            .values
            .len()
            .checked_add(1)
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::DpCells,
            })?;
    let requested = row_len
        .checked_mul(2)
        .and_then(|slots| slots.checked_mul(std::mem::size_of::<f64>()))
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })?;
    let mut previous = Vec::new();
    previous
        .try_reserve_exact(row_len)
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })?;
    previous.resize(row_len, WeightedCost::TOP);
    let mut current = Vec::new();
    current
        .try_reserve_exact(row_len)
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })?;
    current.resize(row_len, WeightedCost::TOP);
    previous[0] = WeightedCost::ZERO;

    let mut previous_value = 0.0;
    let mut previous_time = right.origin;
    for column in 1..row_len {
        let value = right.values[column - 1];
        let time = right.timestamps[column - 1];
        previous[column] = WeightedCost::combine(
            previous[column - 1],
            delete_cost(value, previous_value, time, previous_time, config),
        );
        previous_value = value;
        previous_time = time;
    }

    let mut left_previous_value = 0.0;
    let mut left_previous_time = left.origin;
    for left_index in 0..left.values.len() {
        let left_value = left.values[left_index];
        let left_time = left.timestamps[left_index];
        let left_delete = delete_cost(
            left_value,
            left_previous_value,
            left_time,
            left_previous_time,
            config,
        );
        current[0] = WeightedCost::combine(previous[0], left_delete);
        let mut row_min = current[0];
        let mut right_previous_value = 0.0;
        let mut right_previous_time = right.origin;

        for right_index in 0..right.values.len() {
            let right_value = right.values[right_index];
            let right_time = right.timestamps[right_index];
            let pair = WeightedCost::combine(
                previous[right_index],
                match_cost(
                    left_value,
                    left_previous_value,
                    left_time,
                    left_previous_time,
                    right_value,
                    right_previous_value,
                    right_time,
                    right_previous_time,
                    config,
                ),
            );
            let delete_left = WeightedCost::combine(previous[right_index + 1], left_delete);
            let delete_right = WeightedCost::combine(
                current[right_index],
                delete_cost(
                    right_value,
                    right_previous_value,
                    right_time,
                    right_previous_time,
                    config,
                ),
            );
            current[right_index + 1] = pair.min(delete_left).min(delete_right);
            row_min = row_min.min(current[right_index + 1]);
            right_previous_value = right_value;
            right_previous_time = right_time;
        }
        if !WeightedCost::within(row_min, cutoff) {
            return Ok(None);
        }
        std::mem::swap(&mut previous, &mut current);
        left_previous_value = left_value;
        left_previous_time = left_time;
    }

    let exact = previous[right.values.len()];
    Ok(WeightedCost::within(exact, cutoff).then_some(exact))
}

#[inline]
pub(crate) fn delete_cost(
    value: f64,
    previous_value: f64,
    time: f64,
    previous_time: f64,
    config: &MetricTimestampedTwedConfig,
) -> f64 {
    WeightedCost::combine(
        WeightedCost::combine(
            (value - previous_value).abs(),
            config.nu * (time - previous_time),
        ),
        config.lambda,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn match_cost(
    left_value: f64,
    left_previous_value: f64,
    left_time: f64,
    left_previous_time: f64,
    right_value: f64,
    right_previous_value: f64,
    right_time: f64,
    right_previous_time: f64,
    config: &MetricTimestampedTwedConfig,
) -> f64 {
    let value_cost = WeightedCost::combine(
        (left_value - right_value).abs(),
        (left_previous_value - right_previous_value).abs(),
    );
    let time_cost = config.nu
        * ((left_time - right_time).abs() + (left_previous_time - right_previous_time).abs());
    WeightedCost::combine(value_cost, time_cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::MetricTwedConfig;

    fn completed_distance(outcome: OperationOutcome<ExactDecision>) -> f64 {
        match outcome {
            OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff { distance, .. },
                ..
            } => distance,
            other => panic!("expected a complete exact distance, got {other:?}"),
        }
    }

    #[test]
    fn unit_timestamps_correspond_to_unit_grid_twed() {
        let limits = ResourceLimits::default();
        let left = TimestampedSeries::try_new(
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0, 3.0],
            TimestampUnit::Seconds,
            limits,
        )
        .unwrap();
        let right =
            TimestampedSeries::try_new(&[0.0, 2.0], &[1.0, 2.0], TimestampUnit::Seconds, limits)
                .unwrap();
        let timestamped = MetricTimestampedTwedConfig::try_new(0.5, 1.0).unwrap();
        let unit_grid = MetricTwedConfig::try_new(0.5, 1.0).unwrap();
        assert_eq!(
            completed_distance(
                timestamped
                    .distance_bounded(&left, &right, f64::INFINITY, limits)
                    .unwrap()
            ),
            unit_grid.distance(left.values(), right.values())
        );
    }

    #[test]
    fn physical_displacement_changes_the_distance() {
        let limits = ResourceLimits::default();
        let left =
            TimestampedSeries::try_new(&[1.0, 2.0], &[1.0, 2.0], TimestampUnit::Seconds, limits)
                .unwrap();
        let delayed =
            TimestampedSeries::try_new(&[1.0, 2.0], &[2.0, 4.0], TimestampUnit::Seconds, limits)
                .unwrap();
        let config = MetricTimestampedTwedConfig::try_new(1.0, 0.0).unwrap();
        let distance = completed_distance(
            config
                .distance_bounded(&left, &delayed, f64::INFINITY, limits)
                .unwrap(),
        );
        assert!(distance > 0.0);
    }

    #[test]
    fn invalid_and_mixed_time_domains_fail_closed() {
        let limits = ResourceLimits::default();
        assert!(matches!(
            TimestampedSeries::try_new(&[1.0, 2.0], &[1.0, 1.0], TimestampUnit::Seconds, limits),
            Err(TimestampedTwedError::NonMonotoneTimestamp { index: 1 })
        ));
        assert!(matches!(
            TimestampedSeries::try_new(&[1.0], &[f64::NAN], TimestampUnit::Seconds, limits),
            Err(TimestampedTwedError::NonFiniteTimestamp { index: Some(0) })
        ));
        let seconds =
            TimestampedSeries::try_new(&[1.0], &[1.0], TimestampUnit::Seconds, limits).unwrap();
        let millis =
            TimestampedSeries::try_new(&[1.0], &[1.0], TimestampUnit::Milliseconds, limits)
                .unwrap();
        let config = MetricTimestampedTwedConfig::try_new(1.0, 0.0).unwrap();
        assert_eq!(
            config.distance_bounded(&seconds, &millis, 1.0, limits),
            Err(TimestampedTwedError::MixedUnits)
        );
    }

    #[test]
    fn retained_and_rolling_storage_obey_the_scratch_ceiling() {
        let construction_limits = ResourceLimits {
            max_scratch_bytes: 31,
            ..ResourceLimits::default()
        };
        assert!(matches!(
            TimestampedSeries::try_new(
                &[1.0, 2.0],
                &[1.0, 2.0],
                TimestampUnit::Seconds,
                construction_limits,
            ),
            Err(TimestampedTwedError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: 31,
                    requested: 32,
                }
            ))
        ));

        let series = TimestampedSeries::try_new(
            &[1.0, 2.0],
            &[1.0, 2.0],
            TimestampUnit::Seconds,
            ResourceLimits::default(),
        )
        .unwrap();
        let scoring_limits = ResourceLimits {
            max_scratch_bytes: 47,
            ..ResourceLimits::default()
        };
        let outcome = MetricTimestampedTwedConfig::try_new(1.0, 0.0)
            .unwrap()
            .distance_bounded(&series, &series, f64::INFINITY, scoring_limits)
            .unwrap();
        assert!(matches!(
            outcome,
            OperationOutcome::Incomplete {
                reason: IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: 47,
                    requested: 48,
                },
                ..
            }
        ));
    }
}
