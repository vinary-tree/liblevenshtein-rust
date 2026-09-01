//! Resource and exactness boundaries for reusable elastic point workspaces.
//!
//! These tests exercise only public APIs.  In particular, bounded range, kNN,
//! and certificate queries must use the fallible query-plan/workspace path and
//! must never fall back to the legacy allocating exact scorer.

use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use liblevenshtein::cost::{CostMonoid, WeightedCost};
use liblevenshtein::time_series::elastic::{
    Cost, ElasticKernel, ElasticTransducer, QueryPlanStorage,
};
use liblevenshtein::time_series::{
    DtwConfig, ElasticCertificateError, ElasticCertificateLimits, ElasticOnlineAutomaton,
    ErpConfig, ErpTransducer, FrechetConfig, IncompleteReason, MsmConfig, MsmKernel,
    OnlineAutomatonLimits, OnlineStepOutcome, OperationOutcome, PageBudget, QuantizationConfig,
    ResourceKind, ResourceLimits, TemporalAutomatonError, TwedConfig,
};
use proptest::prelude::*;

fn arithmetic_overflow() -> IncompleteReason {
    IncompleteReason::ArithmeticOverflow {
        resource: ResourceKind::ScratchBytes,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn query_plan_storage_matches_a_wider_checked_oracle(
        elements in any::<usize>(),
        retained_per_element in any::<usize>(),
        transient_per_element in any::<usize>(),
    ) {
        let retained = (elements as u128).checked_mul(retained_per_element as u128);
        let transient = (elements as u128).checked_mul(transient_per_element as u128);
        let expected = retained
            .zip(transient)
            .and_then(|(retained, transient)| {
                retained.checked_add(transient).map(|peak| (retained, peak))
            })
            .filter(|(_, peak)| *peak <= usize::MAX as u128);

        match (QueryPlanStorage::checked_per_element(
            elements,
            retained_per_element,
            transient_per_element,
        ), expected) {
            (Ok(storage), Some((retained, peak))) => {
                prop_assert_eq!(storage.retained_bytes(), retained as usize);
                prop_assert_eq!(storage.construction_peak_bytes(), peak as usize);
                prop_assert!(storage.construction_peak_bytes() >= storage.retained_bytes());
            }
            (Err(error), None) => prop_assert_eq!(error, arithmetic_overflow()),
            (actual, expected) => {
                prop_assert!(false, "storage/oracle mismatch: actual={actual:?}, expected={expected:?}");
            }
        }
    }
}

#[test]
fn query_plan_storage_pins_all_three_overflow_sites() {
    assert_eq!(
        QueryPlanStorage::checked_per_element(usize::MAX, 2, 0),
        Err(arithmetic_overflow())
    );
    assert_eq!(
        QueryPlanStorage::checked_per_element(usize::MAX, 0, 2),
        Err(arithmetic_overflow())
    );
    assert_eq!(
        QueryPlanStorage::checked_per_element(1, usize::MAX, 1),
        Err(arithmetic_overflow())
    );

    let normalized = QueryPlanStorage::new(17, 3);
    assert_eq!(normalized.retained_bytes(), 17);
    assert_eq!(normalized.construction_peak_bytes(), 17);
}

fn dtw_workspace_storage(query_len: usize) -> (usize, usize) {
    let kernel = DtwConfig::new(query_len);
    let plan = kernel
        .query_plan_storage(query_len)
        .expect("small DTW plan arithmetic");
    assert_eq!(plan.retained_bytes(), 4 * query_len * size_of::<f64>());
    assert_eq!(
        plan.construction_peak_bytes(),
        plan.retained_bytes() + 2 * query_len * size_of::<usize>()
    );

    let width = query_len + 1;
    let frontier = 2 * width * (size_of::<f64>() + size_of::<usize>());
    let retained = plan.retained_bytes() + frontier;
    (retained, plan.construction_peak_bytes().max(retained))
}

#[test]
fn dtw_online_constructor_accepts_the_exact_peak_and_rejects_one_byte_less() {
    let query = [0.0, 1.0, 2.0, 3.0, 4.0];
    let (workspace_retained, workspace_peak) = dtw_workspace_storage(query.len());
    let query_bytes = query.len() * size_of::<f64>();
    let retained = query_bytes + workspace_retained;
    let peak = query_bytes + workspace_peak;

    let below = OnlineAutomatonLimits {
        max_scratch_bytes: peak - 1,
        ..OnlineAutomatonLimits::default()
    };
    let error = ElasticOnlineAutomaton::new(&query, DtwConfig::new(query.len()), 10_000.0, below)
        .expect_err("one byte below the declared construction peak must fail");
    assert!(matches!(
        error,
        TemporalAutomatonError::Resource(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::ScratchBytes,
            limit,
            requested,
        }) if limit == peak - 1 && requested == peak
    ));

    let exact = OnlineAutomatonLimits {
        max_scratch_bytes: peak,
        ..OnlineAutomatonLimits::default()
    };
    let mut machine =
        ElasticOnlineAutomaton::new(&query, DtwConfig::new(query.len()), 10_000.0, exact)
            .expect("the exact logical peak is sufficient");
    assert_eq!(machine.scratch_bytes(), retained);
    let OnlineStepOutcome::Advanced { usage, .. } = machine.advance(0.0).unwrap() else {
        panic!("the exact-boundary machine must advance");
    };
    assert_eq!(usage.scratch_bytes, peak);
}

#[test]
fn online_usage_keeps_every_resource_field_observable() {
    let mut machine = ElasticOnlineAutomaton::new(
        &[0.0, 1.0],
        FrechetConfig::new(),
        100.0,
        OnlineAutomatonLimits::default(),
    )
    .unwrap();
    let retained = machine.scratch_bytes();
    let OnlineStepOutcome::Advanced { value, usage } = machine.advance(0.5).unwrap() else {
        panic!("small finite transition must advance");
    };
    assert!(usage.dp_cells > 0);
    assert_eq!(usage.work_units, usage.dp_cells);
    assert_eq!(usage.scratch_bytes, retained);
    assert_eq!(usage.queue_entries, value.active_positions);
    assert!(usage.queue_entries > 0);
}

fn dtw_index() -> ElasticTransducer<DtwConfig, u64> {
    let mut index = ElasticTransducer::new(
        QuantizationConfig::uniform(-100.0, 100.0, 1),
        DtwConfig::new(8),
    );
    assert!(index.insert(7, &[0.0, 1.0, 2.0, 3.0]));
    index
}

#[test]
fn every_bounded_dtw_endpoint_fails_closed_below_workspace_peak() {
    let query = [0.0, 1.0, 2.0, 3.0];
    let (workspace_retained, workspace_peak) = dtw_workspace_storage(query.len());
    let limit = workspace_peak - 1;
    let resources = ResourceLimits {
        max_scratch_bytes: limit,
        ..ResourceLimits::default()
    };

    let range_index = dtw_index();
    let range = range_index
        .search_range_bounded(
            &query,
            10_000.0,
            resources,
            PageBudget {
                max_work_units: 1_000_000,
                max_results: 32,
            },
        )
        .unwrap();
    assert!(matches!(
        range,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: actual_limit,
                requested,
            },
            ..
        } if actual_limit == limit && requested == workspace_peak
    ));

    let knn = dtw_index()
        .search_knn_bounded(&query, 1, resources)
        .unwrap();
    assert!(matches!(
        knn,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: actual_limit,
                requested,
            },
            ..
        } if actual_limit == limit && requested == workspace_peak
    ));

    let certificate_limits = ElasticCertificateLimits {
        resources,
        ..ElasticCertificateLimits::default()
    };
    assert!(matches!(
        dtw_index().search_range_with_certificate(&query, 10_000.0, certificate_limits),
        Err(ElasticCertificateError::BudgetExceeded {
            resource: ResourceKind::ScratchBytes,
            limit: actual_limit,
            requested,
        }) if actual_limit == limit && requested == workspace_peak
    ));

    let knn_finalization_peak = workspace_retained
        .checked_add(std::mem::size_of::<(usize, f64)>())
        .unwrap();
    let exact_resources = ResourceLimits {
        max_scratch_bytes: workspace_peak.max(knn_finalization_peak),
        ..ResourceLimits::default()
    };
    assert!(matches!(
        dtw_index()
            .search_knn_bounded(&query, 1, exact_resources)
            .unwrap(),
        OperationOutcome::Complete { .. }
    ));
}

fn assert_alternating_candidate_reuse<K>(kernel: K)
where
    K: ElasticKernel + Clone,
    K::Monoid: CostMonoid<Cost = f64>,
    Cost<K>: PartialEq,
{
    let query = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let candidates = vec![
        (10_u64, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        (11_u64, vec![0.0]),
        (12_u64, vec![0.0, 1.5, 2.5, 3.5, 4.5, 5.5]),
        (13_u64, vec![]),
        (14_u64, vec![0.0, 2.0]),
        (15_u64, query.clone()),
    ];
    let cutoff = 10_000.0;
    let mut index: ElasticTransducer<K, u64> = ElasticTransducer::new(
        QuantizationConfig::uniform(-100.0, 100.0, 1),
        kernel.clone(),
    );
    for (id, candidate) in &candidates {
        assert!(index.insert(*id, candidate));
    }

    let mut expected: Vec<_> = candidates
        .iter()
        .filter_map(|(id, candidate)| {
            kernel
                .exact_with_cutoff(&query, candidate, cutoff)
                .map(|cost| (*id, cost.to_bits()))
        })
        .collect();
    expected.sort_unstable_by_key(|entry| entry.0);

    let range = index
        .search_range_bounded(
            &query,
            cutoff,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 1_000_000,
                max_results: candidates.len(),
            },
        )
        .unwrap();
    let OperationOutcome::Complete { value, .. } = range else {
        panic!("small mixed-length range search must complete: {range:?}");
    };
    let mut actual: Vec<_> = value
        .into_iter()
        .map(|(id, cost)| (id, cost.to_bits()))
        .collect();
    actual.sort_unstable_by_key(|entry| entry.0);
    assert_eq!(actual, expected);

    let knn = index
        .search_knn_bounded(&query, candidates.len(), ResourceLimits::default())
        .unwrap();
    let OperationOutcome::Complete { value, .. } = knn else {
        panic!("small mixed-length kNN search must complete: {knn:?}");
    };
    let mut actual: Vec<_> = value
        .into_iter()
        .map(|(id, cost)| (id, cost.to_bits()))
        .collect();
    actual.sort_unstable_by_key(|entry| entry.0);
    assert_eq!(actual, expected);
}

#[test]
fn all_builtin_workspaces_reuse_one_query_allocation_across_alternating_lengths() {
    assert_alternating_candidate_reuse(MsmKernel::new(MsmConfig::try_new(1.0).unwrap()));
    assert_alternating_candidate_reuse(ErpConfig::new(0.0));
    assert_alternating_candidate_reuse(TwedConfig::new(0.5, 1.0));
    assert_alternating_candidate_reuse(FrechetConfig::new());
    assert_alternating_candidate_reuse(DtwConfig::new(8));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeBehavior {
    FiniteZero,
    NoFiniteAlignment,
    NumericTop,
    PlanFailure,
    PlanTransientPeak,
}

#[derive(Clone, Debug)]
struct ProbeKernel {
    behavior: ProbeBehavior,
    storage_calls: Arc<AtomicUsize>,
    plan_calls: Arc<AtomicUsize>,
}

impl ProbeKernel {
    fn new(behavior: ProbeBehavior) -> Self {
        Self {
            behavior,
            storage_calls: Arc::new(AtomicUsize::new(0)),
            plan_calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl ElasticKernel for ProbeKernel {
    const IS_METRIC: bool = false;

    type Monoid = WeightedCost;
    type Carry = ();
    type QueryPlan = ();

    fn query_plan_storage(&self, _query_len: usize) -> Result<QueryPlanStorage, IncompleteReason> {
        self.storage_calls.fetch_add(1, Ordering::SeqCst);
        Ok(if self.behavior == ProbeBehavior::PlanTransientPeak {
            QueryPlanStorage::new(0, 256)
        } else {
            QueryPlanStorage::EMPTY
        })
    }

    fn alignment_is_structurally_possible(&self, _query_len: usize, _candidate_len: usize) -> bool {
        !matches!(
            self.behavior,
            ProbeBehavior::NoFiniteAlignment | ProbeBehavior::PlanTransientPeak
        )
    }

    fn supports_interval_query(&self, query: &[f64]) -> bool {
        self.behavior != ProbeBehavior::FiniteZero && query.iter().all(|value| value.is_finite())
    }

    fn column_len(&self, query_len: usize) -> Option<usize> {
        query_len.checked_add(1)
    }

    fn final_row(&self, query_len: usize) -> usize {
        query_len
    }

    fn step_column(
        &self,
        _previous: &[f64],
        _query: &[f64],
        _current_interval: (f64, f64),
        _previous_carry: Option<Self::Carry>,
        _depth: usize,
        _plan: &Self::QueryPlan,
        column: &mut Vec<f64>,
    ) -> (f64, Self::Carry) {
        let cost = if self.behavior == ProbeBehavior::FiniteZero {
            0.0
        } else {
            f64::INFINITY
        };
        column.fill(cost);
        (cost, ())
    }

    fn exact_with_cutoff(&self, _query: &[f64], _candidate: &[f64], _cutoff: f64) -> Option<f64> {
        panic!("bounded operations must use the reusable exact-point workspace")
    }

    fn candidate_lower_bound(
        &self,
        _query: &[f64],
        _candidate: &[f64],
        _plan: &Self::QueryPlan,
    ) -> f64 {
        0.0
    }

    fn try_plan(&self, _query: &[f64]) -> Result<Self::QueryPlan, IncompleteReason> {
        self.plan_calls.fetch_add(1, Ordering::SeqCst);
        if self.behavior == ProbeBehavior::PlanFailure {
            Err(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: 17,
            })
        } else {
            Ok(())
        }
    }

    fn empty_pair_cost(&self) -> f64 {
        0.0
    }

    fn empty_vs_nonempty_cost(&self, _nonempty: &[f64]) -> f64 {
        f64::INFINITY
    }
}

fn probe_index(kernel: ProbeKernel) -> ElasticTransducer<ProbeKernel, u64> {
    let mut index = ElasticTransducer::new(QuantizationConfig::uniform(-1.0, 1.0, 1), kernel);
    assert!(index.insert(1, &[0.0]));
    index
}

#[test]
fn bounded_range_charges_the_exact_finalization_permutation_peak() {
    let index = probe_index(ProbeKernel::new(ProbeBehavior::FiniteZero));
    let query = [0.0];
    let width = query.len() + 1;
    let retained_workspace = 2 * width * (size_of::<f64>() + size_of::<usize>());
    let finalization_peak = retained_workspace + size_of::<(usize, usize)>();
    let page = PageBudget {
        max_work_units: 64,
        max_results: 1,
    };

    let limited = index
        .search_range_bounded(
            &query,
            0.0,
            ResourceLimits {
                max_scratch_bytes: finalization_peak - 1,
                ..ResourceLimits::default()
            },
            page,
        )
        .unwrap();
    assert!(matches!(
        limited,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit,
                requested,
            },
            ..
        } if limit == finalization_peak - 1 && requested == finalization_peak
    ));

    let exact = index
        .search_range_bounded(
            &query,
            0.0,
            ResourceLimits {
                max_scratch_bytes: finalization_peak,
                ..ResourceLimits::default()
            },
            page,
        )
        .unwrap();
    assert!(matches!(
        exact,
        OperationOutcome::Complete { value, .. }
            if value.len() == 1 && value[0].0 == 1 && value[0].1 == 0.0
    ));
}

#[test]
fn specialized_erp_range_uses_the_checked_shared_finalizer() {
    let index = ErpTransducer::from_series(
        QuantizationConfig::uniform(-1.0, 1.0, 1),
        ErpConfig::new(0.0),
        &[vec![0.0]],
    );
    let outcome = index
        .search_range_automaton_bounded(
            &[0.0],
            0.0,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 64,
                max_results: 1,
            },
        )
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Complete { value, .. }
            if value.len() == 1 && value[0] == (0, 0.0)
    ));
}

#[test]
fn structurally_impossible_top_is_not_numeric_overflow_but_reachable_top_is() {
    let no_alignment = ProbeKernel::new(ProbeBehavior::NoFiniteAlignment);
    let outcome = probe_index(no_alignment)
        .search_knn_bounded(&[0.0], 1, ResourceLimits::default())
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Complete { value, .. } if value.is_empty()
    ));

    let numeric = ProbeKernel::new(ProbeBehavior::NumericTop);
    let outcome = probe_index(numeric)
        .search_knn_bounded(&[0.0], 1, ResourceLimits::default())
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::NumericOverflow,
            ..
        }
    ));

    // Empty/nonempty scoring reaches `classify_exact` directly. At TOP cutoff
    // a structurally reachable TOP cost is overflow, not above-cutoff or
    // absence; this pins both halves of the classification match guard.
    let numeric = ProbeKernel::new(ProbeBehavior::NumericTop);
    let outcome = probe_index(numeric)
        .search_knn_bounded(&[], 1, ResourceLimits::default())
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::NumericOverflow,
            ..
        }
    ));
}

#[test]
fn workspace_usage_retains_a_query_plan_transient_construction_peak() {
    let kernel = ProbeKernel::new(ProbeBehavior::PlanTransientPeak);
    let outcome = probe_index(kernel)
        .search_knn_bounded(
            &[0.0],
            1,
            ResourceLimits {
                max_scratch_bytes: 256,
                ..ResourceLimits::default()
            },
        )
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Complete { value, usage }
            if value.is_empty() && usage.scratch_bytes == 256
    ));
}

#[test]
fn plan_failure_is_tagged_by_range_knn_and_certificate() {
    let kernel = ProbeKernel::new(ProbeBehavior::PlanFailure);
    let index = probe_index(kernel.clone());
    let expected = IncompleteReason::AllocationFailed {
        resource: ResourceKind::ScratchBytes,
        requested: 17,
    };

    let range = index
        .search_range_bounded(
            &[0.0],
            1.0,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 1_000,
                max_results: 8,
            },
        )
        .unwrap();
    assert!(matches!(
        range,
        OperationOutcome::Incomplete { reason, .. } if reason == expected
    ));

    let knn = index
        .search_knn_bounded(&[0.0], 1, ResourceLimits::default())
        .unwrap();
    assert!(matches!(
        knn,
        OperationOutcome::Incomplete { reason, .. } if reason == expected
    ));

    assert!(matches!(
        index.search_range_with_certificate(&[0.0], 1.0, ElasticCertificateLimits::default(),),
        Err(ElasticCertificateError::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested: 17,
        })
    ));
    assert_eq!(kernel.storage_calls.load(Ordering::SeqCst), 3);
    assert_eq!(kernel.plan_calls.load(Ordering::SeqCst), 3);
}

#[test]
fn scratch_preflight_rejects_before_query_plan_construction() {
    let kernel = ProbeKernel::new(ProbeBehavior::NumericTop);
    let index = probe_index(kernel.clone());
    let outcome = index
        .search_knn_bounded(
            &[0.0],
            1,
            ResourceLimits {
                max_scratch_bytes: 0,
                ..ResourceLimits::default()
            },
        )
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: 0,
                requested,
            },
            ..
        } if requested > 0
    ));
    assert_eq!(kernel.storage_calls.load(Ordering::SeqCst), 1);
    assert_eq!(kernel.plan_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn empty_range_and_knn_do_not_construct_a_query_plan() {
    let kernel = ProbeKernel::new(ProbeBehavior::PlanFailure);
    let index: ElasticTransducer<ProbeKernel, u64> =
        ElasticTransducer::new(QuantizationConfig::uniform(-1.0, 1.0, 1), kernel.clone());

    let range = index
        .search_range_bounded(
            &[0.0],
            1.0,
            ResourceLimits::default(),
            PageBudget {
                max_work_units: 1_000,
                max_results: 8,
            },
        )
        .unwrap();
    assert!(matches!(
        range,
        OperationOutcome::Complete { value, .. } if value.is_empty()
    ));
    let knn = index
        .search_knn_bounded(&[0.0], 1, ResourceLimits::default())
        .unwrap();
    assert!(matches!(
        knn,
        OperationOutcome::Complete { value, .. } if value.is_empty()
    ));
    assert_eq!(kernel.storage_calls.load(Ordering::SeqCst), 0);
    assert_eq!(kernel.plan_calls.load(Ordering::SeqCst), 0);
}
