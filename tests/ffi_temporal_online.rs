#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_temporal_distance, llev_temporal_online_advance, llev_temporal_online_free,
    llev_temporal_online_new, llev_temporal_online_observation, llev_temporal_online_scratch_bytes,
    LlevStatus, LlevTemporalConfig, LlevTemporalDistanceResult, LlevTemporalLimits,
    LlevTemporalOnlineAutomaton, LlevTemporalOnlineLimits, LlevTemporalOnlineObservation,
    LlevTemporalOnlineStep,
};
use std::ptr;

fn limits() -> LlevTemporalOnlineLimits {
    LlevTemporalOnlineLimits {
        max_query_len: 32,
        max_frontier_positions: 64,
        max_step_work_units: 10_000,
        max_scratch_bytes: 1024 * 1024,
    }
}

fn config(algorithm: u32) -> LlevTemporalConfig {
    LlevTemporalConfig {
        algorithm,
        reserved: 0,
        parameter0: match algorithm {
            1 | 3 | 6 => 1.0,
            _ => 0.0,
        },
        parameter1: 0.0,
        band: usize::from(algorithm == 4) * 2,
        cutoff: 100.0,
    }
}

fn new_machine(
    query: &[f64],
    config: &LlevTemporalConfig,
    limits: &LlevTemporalOnlineLimits,
) -> (LlevStatus, *mut LlevTemporalOnlineAutomaton) {
    let mut machine = ptr::null_mut();
    let status = unsafe {
        llev_temporal_online_new(query.as_ptr(), query.len(), config, limits, &mut machine)
    };
    (status, machine)
}

#[test]
fn online_bridge_matches_native_scalar_prefix_scores() {
    let query = [1.0, 2.0, 3.0];
    let target = [1.0, 2.5, 3.0];
    for algorithm in 1..=5 {
        let config = config(algorithm);
        let (status, machine) = new_machine(&query, &config, &limits());
        assert_eq!(status, LlevStatus::Ok, "algorithm {algorithm}");
        assert!(!machine.is_null());
        let mut initial = LlevTemporalOnlineObservation::default();
        assert_eq!(
            unsafe { llev_temporal_online_observation(machine, &mut initial) },
            LlevStatus::Ok
        );
        assert_eq!(initial.consumed_target_len, 0);
        let mut scratch = 0;
        assert_eq!(
            unsafe { llev_temporal_online_scratch_bytes(machine, &mut scratch) },
            LlevStatus::Ok
        );
        assert!(scratch > 0);

        for end in 1..=target.len() {
            let mut step = LlevTemporalOnlineStep::default();
            assert_eq!(
                unsafe { llev_temporal_online_advance(machine, target[end - 1], &mut step) },
                LlevStatus::Ok
            );
            assert_eq!(step.kind, 0);
            assert_eq!(step.reason, 0);
            assert_eq!(step.observation.consumed_target_len, end);
            assert!(step.work_units <= limits().max_step_work_units);
            let mut scalar = LlevTemporalDistanceResult::default();
            assert_eq!(
                unsafe {
                    llev_temporal_distance(
                        query.as_ptr(),
                        query.len(),
                        target.as_ptr(),
                        end,
                        &config,
                        &LlevTemporalLimits::default(),
                        &mut scalar,
                    )
                },
                LlevStatus::Ok
            );
            if scalar.kind == 0 {
                assert_eq!(step.observation.has_distance, 1);
                let expected = if algorithm == 4 {
                    scalar.value * scalar.value
                } else {
                    scalar.value
                };
                assert!(
                    (step.observation.distance_within_cutoff - expected).abs() < 1e-10,
                    "algorithm {algorithm}, prefix {end}"
                );
            }
        }
        unsafe { llev_temporal_online_free(machine) };
    }
}

#[test]
fn online_bridge_preserves_incomplete_and_invalid_steps_transactionally() {
    let query = [1.0, 2.0, 3.0];
    let mut no_work = limits();
    no_work.max_step_work_units = 0;
    let (status, machine) = new_machine(&query, &config(1), &no_work);
    assert_eq!(status, LlevStatus::Ok);
    let mut step = LlevTemporalOnlineStep::default();
    assert_eq!(
        unsafe { llev_temporal_online_advance(machine, 1.0, &mut step) },
        LlevStatus::Ok
    );
    assert_eq!(step.kind, 1);
    assert_eq!(step.reason, 2);
    assert_eq!(step.observation.has_distance, 0);
    let mut observation = LlevTemporalOnlineObservation::default();
    assert_eq!(
        unsafe { llev_temporal_online_observation(machine, &mut observation) },
        LlevStatus::Ok
    );
    assert_eq!(observation.consumed_target_len, 0);
    unsafe { llev_temporal_online_free(machine) };

    let (status, machine) = new_machine(&query, &config(2), &limits());
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(
        unsafe { llev_temporal_online_advance(machine, f64::NAN, &mut step) },
        LlevStatus::InvalidArgument
    );
    assert_eq!(
        unsafe { llev_temporal_online_observation(machine, &mut observation) },
        LlevStatus::Ok
    );
    assert_eq!(observation.consumed_target_len, 0);
    unsafe { llev_temporal_online_free(machine) };
}

#[test]
fn online_bridge_rejects_unavailable_or_over_budget_construction() {
    let query = [1.0, 2.0, 3.0];
    let mut short = limits();
    short.max_query_len = 2;
    let (status, machine) = new_machine(&query, &config(1), &short);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert!(machine.is_null());

    let mut little_scratch = limits();
    little_scratch.max_scratch_bytes = 0;
    let (status, machine) = new_machine(&query, &config(1), &little_scratch);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert!(machine.is_null());

    let (status, machine) = new_machine(&query, &config(6), &limits());
    assert_eq!(status, LlevStatus::Unsupported);
    assert!(machine.is_null());

    let mut unbounded = config(1);
    unbounded.cutoff = f64::INFINITY;
    let (status, machine) = new_machine(&query, &unbounded, &limits());
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert!(machine.is_null());

    unbounded = config(2);
    unbounded.cutoff = f64::INFINITY;
    let (status, machine) = new_machine(&query, &unbounded, &limits());
    assert_eq!(status, LlevStatus::Ok);
    unsafe { llev_temporal_online_free(machine) };
}
