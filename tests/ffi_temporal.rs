#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_temporal_distance, LlevStatus, LlevTemporalAlgorithm, LlevTemporalConfig,
    LlevTemporalDistanceResult, LlevTemporalLimits,
};
use liblevenshtein::time_series::{
    DtwConfig, ErpConfig, FrechetConfig, MsmConfig, OperationOutcome, ResourceLimits,
    SoftDtwAnalysis, SoftDtwConfig, TwedConfig,
};

fn config(
    algorithm: LlevTemporalAlgorithm,
    parameter0: f64,
    parameter1: f64,
    band: usize,
    cutoff: f64,
) -> LlevTemporalConfig {
    LlevTemporalConfig {
        algorithm: algorithm as u32,
        reserved: 0,
        parameter0,
        parameter1,
        band,
        cutoff,
    }
}

fn run(
    left: &[f64],
    right: &[f64],
    config: &LlevTemporalConfig,
    limits: &LlevTemporalLimits,
) -> (LlevStatus, LlevTemporalDistanceResult) {
    let mut result = LlevTemporalDistanceResult::default();
    let status = unsafe {
        llev_temporal_distance(
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
            config,
            limits,
            &mut result,
        )
    };
    (status, result)
}

#[test]
fn scalar_temporal_bridge_agrees_with_each_native_kernel() {
    let left = [1.0, 2.0, 3.0];
    let right = [1.0, 2.5, 3.0];
    let limits = LlevTemporalLimits::default();
    let cases = [
        (
            config(LlevTemporalAlgorithm::Msm, 1.0, 0.0, 0, f64::INFINITY),
            MsmConfig::try_new(1.0).unwrap().distance(&left, &right),
        ),
        (
            config(LlevTemporalAlgorithm::Erp, 0.0, 0.0, 0, f64::INFINITY),
            ErpConfig::new(0.0).distance(&left, &right),
        ),
        (
            config(LlevTemporalAlgorithm::Twed, 1.0, 0.0, 0, f64::INFINITY),
            TwedConfig::new(1.0, 0.0).distance(&left, &right),
        ),
        (
            config(LlevTemporalAlgorithm::Dtw, 0.0, 0.0, 1, f64::INFINITY),
            DtwConfig::new(1).distance(&left, &right),
        ),
        (
            config(LlevTemporalAlgorithm::Frechet, 0.0, 0.0, 0, f64::INFINITY),
            FrechetConfig::new().distance(&left, &right),
        ),
        (
            config(LlevTemporalAlgorithm::SoftDtw, 1.0, 0.0, 0, f64::INFINITY),
            match SoftDtwConfig::try_new(1.0)
                .unwrap()
                .analyze_bounded(&left, &right, ResourceLimits::default())
                .unwrap()
            {
                OperationOutcome::Complete {
                    value: SoftDtwAnalysis::Finite { value },
                    ..
                } => value,
                other => panic!("unexpected soft-DTW outcome: {other:?}"),
            },
        ),
    ];
    for (config, native) in cases {
        let (status, result) = run(&left, &right, &config, &limits);
        assert_eq!(status, LlevStatus::Ok, "algorithm {}", config.algorithm);
        assert_eq!(result.kind, 0, "algorithm {}", config.algorithm);
        assert!(
            (result.value - native).abs() < 1e-10,
            "algorithm {}: {} != {}",
            config.algorithm,
            result.value,
            native
        );
        assert_eq!(result.dp_cells, left.len() * right.len());
    }
}

#[test]
fn cutoff_no_path_and_budget_are_distinct() {
    let limits = LlevTemporalLimits::default();
    let finite = config(LlevTemporalAlgorithm::Erp, 0.0, 0.0, 0, 0.0);
    let (status, above) = run(&[0.0], &[3.0], &finite, &limits);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(above.kind, 1);

    let negative_soft = config(LlevTemporalAlgorithm::SoftDtw, 1.0, 0.0, 0, -0.5);
    let (status, negative) = run(&[0.0, 0.0], &[0.0, 0.0], &negative_soft, &limits);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(negative.kind, 0);
    assert!(negative.value < -0.5);

    for algorithm in [
        LlevTemporalAlgorithm::Msm,
        LlevTemporalAlgorithm::Dtw,
        LlevTemporalAlgorithm::Frechet,
        LlevTemporalAlgorithm::SoftDtw,
    ] {
        let parameter0 = match algorithm {
            LlevTemporalAlgorithm::Msm | LlevTemporalAlgorithm::SoftDtw => 1.0,
            _ => 0.0,
        };
        let config = config(algorithm, parameter0, 0.0, 0, f64::INFINITY);
        let (status, no_path) = run(&[], &[1.0], &config, &limits);
        assert_eq!(status, LlevStatus::Ok);
        assert_eq!(no_path.kind, 2, "algorithm {}", config.algorithm);
    }

    let restricted = LlevTemporalLimits {
        max_dp_cells: 1,
        ..limits
    };
    let (status, incomplete) = run(
        &[1.0, 2.0],
        &[1.0, 2.0],
        &config(LlevTemporalAlgorithm::Msm, 1.0, 0.0, 0, f64::INFINITY),
        &restricted,
    );
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(incomplete.kind, 3);
    assert_eq!(incomplete.reason, 1);
    assert_eq!(incomplete.value, 0.0);

    let zero = LlevTemporalLimits {
        max_dp_cells: 0,
        max_work_units: 0,
        max_scratch_bytes: 0,
        ..limits
    };
    let (status, impossible) = run(
        &[1.0, 2.0],
        &[1.0],
        &config(LlevTemporalAlgorithm::Dtw, 0.0, 0.0, 0, f64::INFINITY),
        &zero,
    );
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(impossible.kind, 2);

    let (status, overflow) = run(
        &[f64::MAX],
        &[-f64::MAX],
        &config(LlevTemporalAlgorithm::Erp, 0.0, 0.0, 0, f64::INFINITY),
        &limits,
    );
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(overflow.kind, 3);
    assert_eq!(overflow.reason, 4);
}

#[test]
fn invalid_inputs_fail_closed_before_scoring() {
    let limits = LlevTemporalLimits::default();
    let mut result = LlevTemporalDistanceResult {
        value: 42.0,
        kind: 99,
        reason: 99,
        dp_cells: 99,
        work_units: 99,
        scratch_bytes: 99,
    };
    let valid = config(LlevTemporalAlgorithm::Msm, 1.0, 0.0, 0, f64::INFINITY);
    let status = unsafe {
        llev_temporal_distance(
            std::ptr::null(),
            1,
            std::ptr::null(),
            0,
            &valid,
            &limits,
            &mut result,
        )
    };
    assert_eq!(status, LlevStatus::NullPointer);
    assert_eq!(result.kind, 0);
    assert_eq!(result.value, 0.0);

    let (status, result) = run(&[f64::NAN], &[1.0], &valid, &limits);
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert_eq!(result.value, 0.0);

    let invalid = config(LlevTemporalAlgorithm::Twed, -1.0, 0.0, 0, f64::INFINITY);
    let (status, result) = run(&[1.0], &[2.0], &invalid, &limits);
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert_eq!(result.kind, 0);
}
