#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_keogh_plan_bounds_at, llev_keogh_plan_free, llev_keogh_plan_new, llev_keogh_plan_score,
    llev_temporal_lower_bound, llev_twed_length_lower_bound, LlevKeoghPlan, LlevStatus,
    LlevTemporalBoundAlgorithm, LlevTemporalDistanceResult, LlevTemporalLimits,
};
use liblevenshtein::time_series::{
    erp_gap_mass_lower_bound, frechet_candidate_lower_bound, frechet_endpoint_lower_bound,
    frechet_one_sided_hausdorff_lower_bound, keogh_envelopes, lb_keogh, lb_keogh_squared,
    twed_length_lower_bound,
};
use std::ptr;

fn run(
    left: &[f64],
    right: &[f64],
    algorithm: LlevTemporalBoundAlgorithm,
    parameter0: f64,
    band: usize,
    limits: &LlevTemporalLimits,
) -> (LlevStatus, LlevTemporalDistanceResult) {
    let mut result = LlevTemporalDistanceResult::default();
    let status = unsafe {
        llev_temporal_lower_bound(
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
            algorithm as u32,
            parameter0,
            band,
            limits,
            &mut result,
        )
    };
    (status, result)
}

#[test]
fn lower_bound_bridge_matches_all_native_scalar_families() {
    let left = [1.0, 2.0, 3.0];
    let right = [1.0, 2.5, 3.0];
    let limits = LlevTemporalLimits::default();
    let cases = [
        (
            LlevTemporalBoundAlgorithm::ErpGapMass,
            0.0,
            0,
            erp_gap_mass_lower_bound(&left, &right, 0.0),
        ),
        (
            LlevTemporalBoundAlgorithm::FrechetEndpoints,
            0.0,
            0,
            frechet_endpoint_lower_bound(&left, &right),
        ),
        (
            LlevTemporalBoundAlgorithm::FrechetHausdorff,
            0.0,
            0,
            frechet_one_sided_hausdorff_lower_bound(&left, &right),
        ),
        (
            LlevTemporalBoundAlgorithm::FrechetCandidate,
            0.0,
            0,
            frechet_candidate_lower_bound(&left, &right),
        ),
        (
            LlevTemporalBoundAlgorithm::Keogh,
            0.0,
            1,
            lb_keogh(&left, &right, 1),
        ),
    ];
    for (algorithm, parameter0, band, expected) in cases {
        let (status, result) = run(&left, &right, algorithm, parameter0, band, &limits);
        assert_eq!(status, LlevStatus::Ok, "{algorithm:?}");
        assert_eq!(result.kind, 0, "{algorithm:?}");
        assert!((result.value - expected).abs() < 1e-10, "{algorithm:?}");
        assert_eq!(result.dp_cells, 0);
    }

    let mut length = LlevTemporalDistanceResult::default();
    let status = unsafe { llev_twed_length_lower_bound(3, 5, 0.7, &limits, &mut length) };
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(length.kind, 0);
    assert_eq!(length.value, twed_length_lower_bound(3, 5, 0.7));
}

#[test]
fn reusable_keogh_plan_matches_native_envelopes_and_scores() {
    let query = [1.0, 2.0, 3.0];
    let candidate = [1.0, 2.5, 3.0];
    let limits = LlevTemporalLimits::default();
    let mut handle: *mut LlevKeoghPlan = ptr::null_mut();
    assert_eq!(
        unsafe { llev_keogh_plan_new(query.as_ptr(), query.len(), 1, &limits, &mut handle) },
        LlevStatus::Ok
    );
    assert!(!handle.is_null());
    let native = keogh_envelopes(&query, 1).unwrap();
    for index in 0..query.len() {
        let mut has = 0;
        let mut low = 0.0;
        let mut high = 0.0;
        assert_eq!(
            unsafe { llev_keogh_plan_bounds_at(handle, index, &mut has, &mut low, &mut high) },
            LlevStatus::Ok
        );
        assert_eq!(has, 1);
        assert_eq!((low, high), native.bounds_at(index, 1).unwrap());
    }
    for squared in [0, 1] {
        let mut result = LlevTemporalDistanceResult::default();
        let status = unsafe {
            llev_keogh_plan_score(
                handle,
                candidate.as_ptr(),
                candidate.len(),
                squared,
                &limits,
                &mut result,
            )
        };
        assert_eq!(status, LlevStatus::Ok);
        assert_eq!(result.kind, 0);
        let expected = lb_keogh_squared(&candidate, 1, &native);
        let expected = if squared == 0 {
            expected.sqrt()
        } else {
            expected
        };
        assert!((result.value - expected).abs() < 1e-10);
    }
    unsafe { llev_keogh_plan_free(handle) };
}

#[test]
fn lower_bounds_preserve_empty_numeric_and_budget_outcomes() {
    let limits = LlevTemporalLimits::default();
    let (status, one_sided) = run(
        &[],
        &[1.0],
        LlevTemporalBoundAlgorithm::FrechetHausdorff,
        0.0,
        0,
        &limits,
    );
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(one_sided.kind, 0);
    assert_eq!(one_sided.value, 0.0);
    let (status, no_path) = run(
        &[],
        &[1.0],
        LlevTemporalBoundAlgorithm::FrechetEndpoints,
        0.0,
        0,
        &limits,
    );
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(no_path.kind, 2);

    let (status, overflow) = run(
        &[f64::MAX],
        &[-f64::MAX],
        LlevTemporalBoundAlgorithm::FrechetEndpoints,
        0.0,
        0,
        &limits,
    );
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(overflow.kind, 3);
    assert_eq!(overflow.reason, 4);

    let no_work = LlevTemporalLimits {
        max_work_units: 0,
        ..limits
    };
    let (status, incomplete) = run(
        &[1.0],
        &[1.0],
        LlevTemporalBoundAlgorithm::ErpGapMass,
        0.0,
        0,
        &no_work,
    );
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(incomplete.kind, 3);
    assert_eq!(incomplete.reason, 2);

    let no_scratch = LlevTemporalLimits {
        max_scratch_bytes: 0,
        ..limits
    };
    let (status, incomplete) = run(
        &[1.0, 2.0],
        &[1.0, 2.0],
        LlevTemporalBoundAlgorithm::Keogh,
        0.0,
        1,
        &no_scratch,
    );
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(incomplete.kind, 3);
    assert_eq!(incomplete.reason, 3);

    let (status, invalid) = run(
        &[f64::NAN],
        &[1.0],
        LlevTemporalBoundAlgorithm::Keogh,
        0.0,
        1,
        &limits,
    );
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert_eq!(invalid.value, 0.0);
}
