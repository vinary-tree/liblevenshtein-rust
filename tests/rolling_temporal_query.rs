//! Bounded rolling-window stability and resumable exact range composition.

use liblevenshtein::time_series::{
    BoundedRollingWindow, IncompleteReason, MsmConfig, MsmTransducer, OperationOutcome, PageBudget,
    QuantizationConfig, ResourceKind, ResourceLimits, RollingWindowStep,
};

#[test]
fn rolling_windows_have_exact_width_stride_offsets_and_order() {
    let mut rolling =
        BoundedRollingWindow::new(3, 2, ResourceLimits::default()).expect("config is valid");
    let retained = rolling.scratch_bytes();
    let mut observed = Vec::new();
    for sample in 0..9 {
        let RollingWindowStep::Advanced { snapshot, .. } = rolling
            .advance(f64::from(sample))
            .expect("sample is finite")
        else {
            panic!("default limits accept each fixed-width step");
        };
        if let Some(snapshot) = snapshot {
            observed.push((
                snapshot.window_id(),
                snapshot.start_offset(),
                snapshot.end_offset(),
                snapshot.values().to_vec(),
            ));
        }
        assert_eq!(rolling.scratch_bytes(), retained);
    }
    assert_eq!(
        observed,
        vec![
            (0, 0, 3, vec![0.0, 1.0, 2.0]),
            (1, 2, 5, vec![2.0, 3.0, 4.0]),
            (2, 4, 7, vec![4.0, 5.0, 6.0]),
            (3, 6, 9, vec![6.0, 7.0, 8.0]),
        ]
    );
}

#[test]
fn invalid_sample_is_transactional_and_unknown_stream_memory_is_constant() {
    let mut rolling =
        BoundedRollingWindow::new(4, 10_000, ResourceLimits::default()).expect("config is valid");
    assert!(rolling.advance(f64::NAN).is_err());
    assert_eq!(rolling.consumed_samples(), 0);
    let retained = rolling.scratch_bytes();
    for index in 0..100_000usize {
        assert!(matches!(
            rolling
                .advance((index % 7) as f64)
                .expect("sample is finite"),
            RollingWindowStep::Advanced { .. }
        ));
        assert_eq!(rolling.scratch_bytes(), retained);
    }
    assert_eq!(rolling.consumed_samples(), 100_000);
}

#[test]
fn emitted_window_composes_with_resumable_exact_range_search() {
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(-10.0, 10.0),
        MsmConfig::try_new(1.0).expect("positive MSM cost is metric"),
        &[vec![0.0, 1.0, 2.0], vec![5.0, 5.0, 5.0]],
    );
    let mut rolling =
        BoundedRollingWindow::new(3, 1, ResourceLimits::default()).expect("config is valid");
    let mut snapshot = None;
    for value in [0.0, 1.0, 2.0] {
        let RollingWindowStep::Advanced {
            snapshot: emitted, ..
        } = rolling.advance(value).expect("sample is finite")
        else {
            panic!("default limits accept the rolling step");
        };
        snapshot = emitted.or(snapshot);
    }
    let snapshot = snapshot.expect("the first full window is emitted");
    let page = PageBudget {
        max_work_units: 1_000,
        max_results: 1,
    };
    let mut outcome = snapshot
        .search_range_bounded(&index, 0.0, ResourceLimits::default(), page)
        .expect("window and cutoff are valid");
    let paged = loop {
        match outcome {
            OperationOutcome::Complete { value, .. } => break value,
            OperationOutcome::Incomplete {
                continuation: Some(next),
                ..
            } => outcome = next.resume(page),
            other => panic!("default limits must leave a resumable query: {other:?}"),
        }
    };
    assert_eq!(paged, index.search_range(snapshot.values(), 0.0));
    assert_eq!(paged, vec![(0, 0.0)]);
}

#[test]
fn snapshot_ceiling_fails_at_construction_not_as_an_empty_window() {
    let limits = ResourceLimits {
        max_snapshot_bytes: 7,
        ..ResourceLimits::default()
    };
    let error =
        BoundedRollingWindow::new(1, 1, limits).expect_err("one f64 snapshot needs eight bytes");
    assert!(matches!(
        error,
        liblevenshtein::time_series::TemporalAutomatonError::Resource(
            IncompleteReason::BudgetExceeded {
                resource: ResourceKind::SnapshotBytes,
                limit: 7,
                requested: 8,
            }
        )
    ));
}
