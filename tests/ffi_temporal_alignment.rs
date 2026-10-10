#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_temporal_alignment_free, llev_temporal_alignment_new, llev_temporal_alignment_page,
    llev_temporal_alignment_replay, llev_temporal_distance, llev_timestamped_twed_alignment_new,
    llev_timestamped_twed_alignment_replay, llev_timestamped_twed_distance, LlevStatus,
    LlevTemporalAlignment, LlevTemporalAlignmentLimits, LlevTemporalAlignmentOutcome,
    LlevTemporalAlignmentStep, LlevTemporalConfig, LlevTemporalDistanceResult, LlevTemporalLimits,
    LlevTimestampedSeriesView,
};

fn limits() -> LlevTemporalAlignmentLimits {
    LlevTemporalAlignmentLimits {
        temporal: LlevTemporalLimits::default(),
        max_witness_bytes: 64 * 1024 * 1024,
    }
}

fn config(algorithm: u32, parameter0: f64, parameter1: f64, band: usize) -> LlevTemporalConfig {
    LlevTemporalConfig {
        algorithm,
        reserved: 0,
        parameter0,
        parameter1,
        band,
        cutoff: f64::INFINITY,
    }
}

fn extract(
    query: &[f64],
    candidate: &[f64],
    config: &LlevTemporalConfig,
    limits: &LlevTemporalAlignmentLimits,
) -> (*mut LlevTemporalAlignment, LlevTemporalAlignmentOutcome) {
    let mut handle = std::ptr::null_mut();
    let mut outcome = LlevTemporalAlignmentOutcome::default();
    assert_eq!(
        unsafe {
            llev_temporal_alignment_new(
                query.as_ptr(),
                query.len(),
                candidate.as_ptr(),
                candidate.len(),
                config,
                limits,
                &mut handle,
                &mut outcome,
            )
        },
        LlevStatus::Ok
    );
    (handle, outcome)
}

#[test]
fn scalar_witnesses_page_and_replay_exact_native_scores() {
    let query = [1.0, 2.0, 3.0];
    let candidate = [1.0, 2.5, 3.0];
    for config in [
        config(1, 1.0, 0.0, 0),
        config(2, 0.0, 0.0, 0),
        config(3, 0.5, 1.0, 0),
        config(4, 0.0, 0.0, 1),
        config(5, 0.0, 0.0, 0),
    ] {
        let (handle, outcome) = extract(&query, &candidate, &config, &limits());
        assert_eq!(outcome.kind, 0, "algorithm {}", config.algorithm);
        assert!(!handle.is_null());
        assert!(outcome.step_count > 0);
        assert!(outcome.witness_bytes > 0);
        let mut scalar = LlevTemporalDistanceResult::default();
        assert_eq!(
            unsafe {
                llev_temporal_distance(
                    query.as_ptr(),
                    query.len(),
                    candidate.as_ptr(),
                    candidate.len(),
                    &config,
                    &limits().temporal,
                    &mut scalar,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(outcome.distance, scalar.value);

        let mut steps = vec![LlevTemporalAlignmentStep::default(); outcome.step_count];
        let mut written = 0;
        assert_eq!(
            unsafe {
                llev_temporal_alignment_page(
                    handle,
                    0,
                    steps.as_mut_ptr(),
                    steps.len(),
                    &mut written,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(written, outcome.step_count);
        assert!(steps.iter().all(|step| (1..=3).contains(&step.operation)));
        written = usize::MAX;
        assert_eq!(
            unsafe {
                llev_temporal_alignment_page(
                    handle,
                    outcome.step_count,
                    std::ptr::null_mut(),
                    0,
                    &mut written,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(written, 0);
        let mut replay = 0.0;
        assert_eq!(
            unsafe {
                llev_temporal_alignment_replay(
                    handle,
                    query.as_ptr(),
                    query.len(),
                    candidate.as_ptr(),
                    candidate.len(),
                    &mut replay,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(replay, outcome.distance);
        if config.algorithm == 2 {
            let changed = [9.0, 2.5, 3.0];
            assert_eq!(
                unsafe {
                    llev_temporal_alignment_replay(
                        handle,
                        query.as_ptr(),
                        query.len(),
                        changed.as_ptr(),
                        changed.len(),
                        &mut replay,
                    )
                },
                LlevStatus::InvalidArgument
            );
        }
        unsafe { llev_temporal_alignment_free(handle) };
    }
}

fn view<'a>(values: &'a [f64], timestamps: &'a [f64]) -> LlevTimestampedSeriesView {
    LlevTimestampedSeriesView {
        values: values.as_ptr(),
        timestamps: timestamps.as_ptr(),
        len: values.len(),
        unit: 2,
        reserved: 0,
        origin: 10.0,
    }
}

#[test]
fn timestamped_witness_replays_and_limits_fail_closed() {
    let query = [1.0, 2.0];
    let candidate = [1.0, 2.5];
    let query_times = [10.0, 13.0];
    let candidate_times = [10.0, 14.0];
    let query_view = view(&query, &query_times);
    let candidate_view = view(&candidate, &candidate_times);
    let mut handle = std::ptr::null_mut();
    let mut outcome = LlevTemporalAlignmentOutcome::default();
    assert_eq!(
        unsafe {
            llev_timestamped_twed_alignment_new(
                &query_view,
                &candidate_view,
                0.5,
                1.0,
                f64::INFINITY,
                &limits(),
                &mut handle,
                &mut outcome,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(outcome.kind, 0);
    assert!(!handle.is_null());
    let mut scalar = LlevTemporalDistanceResult::default();
    assert_eq!(
        unsafe {
            llev_timestamped_twed_distance(
                &query_view,
                &candidate_view,
                0.5,
                1.0,
                f64::INFINITY,
                &limits().temporal,
                &mut scalar,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(outcome.distance, scalar.value);
    let mut replay = 0.0;
    assert_eq!(
        unsafe {
            llev_timestamped_twed_alignment_replay(
                handle,
                &query_view,
                &candidate_view,
                &mut replay,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(replay, outcome.distance);
    unsafe { llev_temporal_alignment_free(handle) };

    let restricted = LlevTemporalAlignmentLimits {
        max_witness_bytes: 0,
        ..limits()
    };
    let (handle, outcome) = extract(&query, &candidate, &config(2, 0.0, 0.0, 0), &restricted);
    assert!(handle.is_null());
    assert_eq!(outcome.kind, 3);
    assert_eq!(outcome.reason, 5);

    let mut cutoff_config = config(2, 0.0, 0.0, 0);
    cutoff_config.cutoff = 0.0;
    let (handle, outcome) = extract(&query, &candidate, &cutoff_config, &limits());
    assert!(handle.is_null());
    assert_eq!(outcome.kind, 1);
}
