//! Budgeted temporal lower bounds and reusable native Keogh plans.

use super::{
    index::{boundary, slice},
    temporal::{reason_code, LlevTemporalDistanceResult, LlevTemporalLimits},
    LlevStatus,
};
use crate::time_series::{
    combined_lb, erp_gap_mass_lower_bound, euclidean_lb, frechet_candidate_lower_bound,
    frechet_endpoint_lower_bound, frechet_one_sided_hausdorff_lower_bound,
    kernels::try_keogh_envelopes, l1_lb, lb_keogh_squared, length_lb, twed_length_lower_bound,
    IncompleteReason, KeoghPlan, Operand, ResourceKind, ResourceLedger, ResourceLimits,
    TemporalValidationError,
};

/// Public temporal lower-bound selector.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlevTemporalBoundAlgorithm {
    /// ERP gap-mass difference.
    ErpGapMass = 1,
    /// Fréchet mandatory endpoint links.
    FrechetEndpoints = 2,
    /// Fréchet one-sided Hausdorff bound.
    FrechetHausdorff = 3,
    /// Maximum of Fréchet endpoint and Hausdorff bounds.
    FrechetCandidate = 4,
    /// Root-distance Keogh bound.
    Keogh = 5,
    /// Correctness-preserving MSM length bound.
    MsmLength = 6,
    /// Prefix Euclidean MSM heuristic, unsafe for exact pruning.
    MsmEuclideanHeuristic = 7,
    /// Prefix L1 MSM heuristic, unsafe for exact pruning.
    MsmL1Heuristic = 8,
    /// Maximum of length and prefix Euclidean MSM heuristics.
    MsmCombinedHeuristic = 9,
}

impl TryFrom<u32> for LlevTemporalBoundAlgorithm {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::ErpGapMass),
            2 => Ok(Self::FrechetEndpoints),
            3 => Ok(Self::FrechetHausdorff),
            4 => Ok(Self::FrechetCandidate),
            5 => Ok(Self::Keogh),
            6 => Ok(Self::MsmLength),
            7 => Ok(Self::MsmEuclideanHeuristic),
            8 => Ok(Self::MsmL1Heuristic),
            9 => Ok(Self::MsmCombinedHeuristic),
            _ => Err(()),
        }
    }
}

/// Opaque reusable Keogh envelope over one finite nonempty query.
pub struct LlevKeoghPlan {
    plan: KeoghPlan,
    band: usize,
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

fn validation(error: TemporalValidationError) -> (LlevStatus, String) {
    match error {
        TemporalValidationError::SeriesTooLong { .. } => {
            (LlevStatus::LimitExceeded, error.to_string())
        }
        _ => invalid(error.to_string()),
    }
}

fn limits_from_c(raw: LlevTemporalLimits) -> ResourceLimits {
    ResourceLimits {
        max_series_len: raw.max_series_len,
        max_dp_cells: raw.max_dp_cells,
        max_work_units: raw.max_work_units,
        max_scratch_bytes: raw.max_scratch_bytes,
        ..ResourceLimits::default()
    }
}

fn incomplete(
    reason: IncompleteReason,
    output: &mut LlevTemporalDistanceResult,
) -> (LlevStatus, String) {
    output.kind = 3;
    output.reason = reason_code(reason);
    (
        LlevStatus::LimitExceeded,
        format!("temporal bound incomplete: {reason:?}"),
    )
}

fn checked_work(a: usize, b: usize) -> Result<usize, IncompleteReason> {
    a.checked_add(b)
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::WorkUnits,
        })
}

fn checked_scratch(count: usize, bytes: usize) -> Result<usize, IncompleteReason> {
    count
        .checked_mul(bytes)
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })
}

fn charge(
    ledger: &mut ResourceLedger,
    work: usize,
    scratch: usize,
    output: &mut LlevTemporalDistanceResult,
) -> Result<(), (LlevStatus, String)> {
    ledger
        .charge(ResourceKind::WorkUnits, work)
        .and_then(|()| ledger.observe_peak(ResourceKind::ScratchBytes, scratch))
        .map_err(|reason| incomplete(reason, output))?;
    output.work_units = ledger.usage().work_units;
    output.scratch_bytes = ledger.usage().scratch_bytes;
    Ok(())
}

fn finish(
    score: f64,
    no_alignment: bool,
    output: &mut LlevTemporalDistanceResult,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if score.is_finite() {
        output.value = score;
        output.kind = 0;
        Ok(LlevStatus::Ok)
    } else if no_alignment {
        output.kind = 2;
        Ok(LlevStatus::Ok)
    } else {
        Err(incomplete(IncompleteReason::NumericOverflow, output))
    }
}

fn validate_series<'a>(
    left: &'a [f64],
    right: &'a [f64],
    ledger: &ResourceLedger,
) -> Result<(), (LlevStatus, String)> {
    ledger
        .validate_finite_series(Operand::Query, left)
        .map_err(validation)?;
    ledger
        .validate_finite_series(Operand::Candidate, right)
        .map_err(validation)?;
    Ok(())
}

/// Evaluate one native temporal lower bound or explicitly selected heuristic
/// under work and scratch ceilings. `parameter0` is the finite ERP gap or
/// nonnegative MSM split/merge cost; `band` belongs to Keogh. MSM prefix
/// Euclidean, L1, and combined scores are unsafe for exact pruning.
///
/// # Safety
/// Nonempty arrays must be readable and aligned; `limits` and `out_result`
/// must be valid and disjoint from both input arrays.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_lower_bound(
    left: *const f64,
    left_len: usize,
    right: *const f64,
    right_len: usize,
    algorithm: u32,
    parameter0: f64,
    band: usize,
    limits: *const LlevTemporalLimits,
    out_result: *mut LlevTemporalDistanceResult,
) -> LlevStatus {
    boundary(|| {
        let output = out_result.as_mut().ok_or((
            LlevStatus::NullPointer,
            "temporal bound result is null".into(),
        ))?;
        *output = LlevTemporalDistanceResult::default();
        let raw = *limits
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "temporal limits are null".into()))?;
        let algorithm = LlevTemporalBoundAlgorithm::try_from(algorithm)
            .map_err(|()| invalid("unknown temporal lower-bound algorithm"))?;
        let cost_mode = matches!(
            algorithm,
            LlevTemporalBoundAlgorithm::MsmLength
                | LlevTemporalBoundAlgorithm::MsmCombinedHeuristic
        );
        if !parameter0.is_finite()
            || (cost_mode && parameter0 < 0.0)
            || (!cost_mode
                && algorithm != LlevTemporalBoundAlgorithm::ErpGapMass
                && parameter0 != 0.0)
            || (algorithm != LlevTemporalBoundAlgorithm::Keogh && band != 0)
        {
            return Err(invalid("invalid or unused temporal bound parameters"));
        }
        let limits = limits_from_c(raw);
        if band > limits.max_band_width {
            return Err((LlevStatus::LimitExceeded, "Keogh band exceeds limit".into()));
        }
        let left = slice(left, left_len, "left temporal series")?;
        let right = slice(right, right_len, "right temporal series")?;
        let mut ledger = ResourceLedger::new(limits);
        validate_series(left, right, &ledger)?;
        let base =
            checked_work(left.len(), right.len()).map_err(|reason| incomplete(reason, output))?;
        let (work, scratch) = match algorithm {
            LlevTemporalBoundAlgorithm::ErpGapMass
            | LlevTemporalBoundAlgorithm::FrechetEndpoints
            | LlevTemporalBoundAlgorithm::MsmEuclideanHeuristic
            | LlevTemporalBoundAlgorithm::MsmL1Heuristic
            | LlevTemporalBoundAlgorithm::MsmCombinedHeuristic => (base, 0),
            LlevTemporalBoundAlgorithm::MsmLength => (1, 0),
            LlevTemporalBoundAlgorithm::FrechetHausdorff
            | LlevTemporalBoundAlgorithm::FrechetCandidate => {
                let pair_work = left.len().checked_mul(right.len()).ok_or_else(|| {
                    incomplete(
                        IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::WorkUnits,
                        },
                        output,
                    )
                })?;
                // The native one-sided bound sorts the right operand before
                // probing it. Its square is a conservative ceiling for that
                // O(n log n) sort, including short left operands.
                let sort_work = right.len().checked_mul(right.len()).ok_or_else(|| {
                    incomplete(
                        IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::WorkUnits,
                        },
                        output,
                    )
                })?;
                let work = checked_work(base, pair_work)
                    .and_then(|value| checked_work(value, sort_work))
                    .map_err(|reason| incomplete(reason, output))?;
                let scratch = checked_scratch(right.len(), 2 * std::mem::size_of::<f64>())
                    .map_err(|reason| incomplete(reason, output))?;
                (work, scratch)
            }
            LlevTemporalBoundAlgorithm::Keogh => {
                let work = left
                    .len()
                    .checked_mul(8)
                    .and_then(|value| value.checked_add(right.len()))
                    .ok_or_else(|| {
                        incomplete(
                            IncompleteReason::ArithmeticOverflow {
                                resource: ResourceKind::WorkUnits,
                            },
                            output,
                        )
                    })?;
                let scratch = checked_scratch(
                    left.len(),
                    4 * std::mem::size_of::<f64>() + 2 * std::mem::size_of::<usize>(),
                )
                .map_err(|reason| incomplete(reason, output))?;
                (work, scratch)
            }
        };
        charge(&mut ledger, work, scratch, output)?;
        let score = match algorithm {
            LlevTemporalBoundAlgorithm::ErpGapMass => {
                erp_gap_mass_lower_bound(left, right, parameter0)
            }
            LlevTemporalBoundAlgorithm::FrechetEndpoints => {
                frechet_endpoint_lower_bound(left, right)
            }
            LlevTemporalBoundAlgorithm::FrechetHausdorff => {
                frechet_one_sided_hausdorff_lower_bound(left, right)
            }
            LlevTemporalBoundAlgorithm::FrechetCandidate => {
                frechet_candidate_lower_bound(left, right)
            }
            LlevTemporalBoundAlgorithm::Keogh => {
                if let Some(plan) =
                    try_keogh_envelopes(left, band).map_err(|reason| incomplete(reason, output))?
                {
                    lb_keogh_squared(right, band, &plan).sqrt()
                } else if right.is_empty() && left.is_empty() {
                    0.0
                } else {
                    f64::INFINITY
                }
            }
            LlevTemporalBoundAlgorithm::MsmLength => length_lb(left, right, parameter0),
            LlevTemporalBoundAlgorithm::MsmEuclideanHeuristic => euclidean_lb(left, right),
            LlevTemporalBoundAlgorithm::MsmL1Heuristic => l1_lb(left, right),
            LlevTemporalBoundAlgorithm::MsmCombinedHeuristic => {
                combined_lb(left, right, parameter0)
            }
        };
        let no_alignment = left.is_empty() != right.is_empty()
            || (algorithm == LlevTemporalBoundAlgorithm::Keogh
                && left.len().abs_diff(right.len()) > band);
        finish(score, no_alignment, output)
    })
}

/// Compute the native TWED length bound under the same explicit ceilings.
///
/// # Safety
/// `limits` and `out_result` must point to distinct valid storage.
#[no_mangle]
pub unsafe extern "C" fn llev_twed_length_lower_bound(
    left_len: usize,
    right_len: usize,
    gap_penalty: f64,
    limits: *const LlevTemporalLimits,
    out_result: *mut LlevTemporalDistanceResult,
) -> LlevStatus {
    boundary(|| {
        let output = out_result.as_mut().ok_or((
            LlevStatus::NullPointer,
            "TWED lower-bound result is null".into(),
        ))?;
        *output = LlevTemporalDistanceResult::default();
        let raw = *limits
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "temporal limits are null".into()))?;
        if !gap_penalty.is_finite() || gap_penalty < 0.0 {
            return Err(invalid("TWED gap penalty must be finite and nonnegative"));
        }
        let limits = limits_from_c(raw);
        if left_len > limits.max_series_len || right_len > limits.max_series_len {
            return Err((LlevStatus::LimitExceeded, "TWED series length limit".into()));
        }
        let mut ledger = ResourceLedger::new(limits);
        charge(&mut ledger, 1, 0, output)?;
        finish(
            twed_length_lower_bound(left_len, right_len, gap_penalty),
            false,
            output,
        )
    })
}

/// Construct a reusable native Keogh envelope over a nonempty finite query.
///
/// # Safety
/// Nonempty query arrays, `limits`, and `out_plan` must be valid and
/// disjoint; the query is copied into the plan.
#[no_mangle]
pub unsafe extern "C" fn llev_keogh_plan_new(
    query: *const f64,
    query_len: usize,
    band: usize,
    limits: *const LlevTemporalLimits,
    out_plan: *mut *mut LlevKeoghPlan,
) -> LlevStatus {
    boundary(|| {
        let output = out_plan
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "Keogh plan output is null".into()))?;
        *output = std::ptr::null_mut();
        let raw = *limits
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "temporal limits are null".into()))?;
        let limits = limits_from_c(raw);
        if band > limits.max_band_width {
            return Err((LlevStatus::LimitExceeded, "Keogh band exceeds limit".into()));
        }
        let query = slice(query, query_len, "Keogh query")?;
        let mut ledger = ResourceLedger::new(limits);
        ledger
            .validate_finite_series(Operand::Query, query)
            .map_err(validation)?;
        if query.is_empty() {
            return Err(invalid("Keogh plan requires a nonempty query"));
        }
        let work = query.len().checked_mul(8).ok_or((
            LlevStatus::LimitExceeded,
            "Keogh plan work calculation overflow".into(),
        ))?;
        let scratch = query
            .len()
            .checked_mul(4 * std::mem::size_of::<f64>() + 2 * std::mem::size_of::<usize>())
            .ok_or((
                LlevStatus::LimitExceeded,
                "Keogh plan scratch calculation overflow".into(),
            ))?;
        ledger
            .charge(ResourceKind::WorkUnits, work)
            .and_then(|()| ledger.observe_peak(ResourceKind::ScratchBytes, scratch))
            .map_err(|reason| (LlevStatus::LimitExceeded, format!("{reason:?}")))?;
        let plan = try_keogh_envelopes(query, band)
            .map_err(|reason| (LlevStatus::LimitExceeded, format!("{reason:?}")))?
            .ok_or_else(|| invalid("Keogh query has no valid envelope"))?;
        *output = Box::into_raw(Box::new(LlevKeoghPlan { plan, band }));
        Ok(LlevStatus::Ok)
    })
}

/// Read an envelope interval at one target position.
///
/// # Safety
/// `plan` must be live; all three output pointers must be writable and
/// distinct. `out_has` is zero if the position is unreachable.
#[no_mangle]
pub unsafe extern "C" fn llev_keogh_plan_bounds_at(
    plan: *const LlevKeoghPlan,
    target_index: usize,
    out_has: *mut u8,
    out_low: *mut f64,
    out_high: *mut f64,
) -> LlevStatus {
    boundary(|| {
        let has = out_has
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "Keogh has output is null".into()))?;
        let low = out_low
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "Keogh low output is null".into()))?;
        let high = out_high
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "Keogh high output is null".into()))?;
        *has = 0;
        *low = 0.0;
        *high = 0.0;
        let plan = plan
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "Keogh plan is null".into()))?;
        if let Some((lower, upper)) = plan.plan.bounds_at(target_index, plan.band) {
            *has = 1;
            *low = lower;
            *high = upper;
        }
        Ok(LlevStatus::Ok)
    })
}

/// Score a candidate with a retained Keogh plan in squared-cost or root
/// units. Results use the same finite/no-alignment/incomplete tags as the
/// standalone temporal bound operation.
///
/// # Safety
/// The plan must be live; the candidate, limits, and result must be valid and
/// disjoint for this call.
#[no_mangle]
pub unsafe extern "C" fn llev_keogh_plan_score(
    plan: *const LlevKeoghPlan,
    candidate: *const f64,
    candidate_len: usize,
    squared: u8,
    limits: *const LlevTemporalLimits,
    out_result: *mut LlevTemporalDistanceResult,
) -> LlevStatus {
    boundary(|| {
        let output = out_result
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "Keogh result is null".into()))?;
        *output = LlevTemporalDistanceResult::default();
        if squared > 1 {
            return Err(invalid("Keogh squared flag must be zero or one"));
        }
        let plan = plan
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "Keogh plan is null".into()))?;
        let raw = *limits
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "temporal limits are null".into()))?;
        let limits = limits_from_c(raw);
        let candidate = slice(candidate, candidate_len, "Keogh candidate")?;
        let mut ledger = ResourceLedger::new(limits);
        ledger
            .validate_series_len(Operand::Query, plan.plan.query_len())
            .map_err(validation)?;
        ledger
            .validate_finite_series(Operand::Candidate, candidate)
            .map_err(validation)?;
        charge(&mut ledger, candidate.len(), 0, output)?;
        let score = lb_keogh_squared(candidate, plan.band, &plan.plan);
        let no_alignment =
            candidate.is_empty() || plan.plan.query_len().abs_diff(candidate.len()) > plan.band;
        finish(
            if squared == 0 { score.sqrt() } else { score },
            no_alignment,
            output,
        )
    })
}

/// Release one Keogh plan handle.
///
/// # Safety
/// The pointer must be live and freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn llev_keogh_plan_free(plan: *mut LlevKeoghPlan) {
    if !plan.is_null() {
        drop(Box::from_raw(plan));
    }
}
