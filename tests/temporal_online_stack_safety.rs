//! Constrained-stack and stable-retention regression for every production
//! metric/scorer online temporal machine.

use liblevenshtein::cost::CostMonoid;
use liblevenshtein::time_series::{
    DtwConfig, ElasticOnlineAutomaton, ErpConfig, ErpOnlineAutomaton, FrechetConfig,
    L1GroundMetric, MetricTimestampedTwedConfig, MsmConfig, MsmKernel, MsmTransducer,
    OnlineAutomatonLimits, OperationOutcome, PageBudget, QuantizationConfig, ResourceLimits,
    TimestampUnit, TimestampedSeries, TimestampedTwedOnlineAutomaton, TwedConfig,
    VectorFrechetOnlineAutomaton, VectorFrechetPath, VectorSample,
};

const PREFIX_LEN: usize = 100_000;

fn consume_scalar<K>(mut machine: ElasticOnlineAutomaton<K>)
where
    K: liblevenshtein::time_series::elastic::ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    let retained = machine.scratch_bytes();
    for index in 0..PREFIX_LEN {
        assert!(machine
            .advance((index % 3) as f64)
            .expect("generated scalar stream is finite")
            .advanced());
        assert_eq!(machine.scratch_bytes(), retained);
    }
    assert_eq!(machine.observation().consumed_target_len, PREFIX_LEN);
}

#[test]
fn every_online_temporal_machine_is_iterative_and_prefix_stable() {
    std::thread::Builder::new()
        .name("temporal-online-small-stack".into())
        .stack_size(128 * 1024)
        .spawn(|| {
            let limits = OnlineAutomatonLimits::default();
            let query = [0.0, 1.0, 2.0];

            consume_scalar(
                ElasticOnlineAutomaton::new(
                    &query,
                    MsmKernel::new(MsmConfig::try_new(1.0).unwrap()),
                    1.0e12,
                    limits,
                )
                .unwrap(),
            );
            consume_scalar(
                ElasticOnlineAutomaton::new(&query, TwedConfig::new(0.5, 1.0), 1.0e12, limits)
                    .unwrap(),
            );
            consume_scalar(
                ElasticOnlineAutomaton::new(&query, FrechetConfig::new(), 1.0e12, limits).unwrap(),
            );
            consume_scalar(
                ElasticOnlineAutomaton::new(&query, DtwConfig::new(2), 1.0e12, limits).unwrap(),
            );

            let mut erp =
                ErpOnlineAutomaton::new(&query, ErpConfig::new(0.0), 1.0e12, limits).unwrap();
            let erp_retained = erp.scratch_bytes();
            for index in 0..PREFIX_LEN {
                assert!(erp.advance((index % 3) as f64).unwrap().advanced());
                assert_eq!(erp.scratch_bytes(), erp_retained);
            }
            assert_eq!(erp.observation().consumed_target_len, PREFIX_LEN);

            let timestamped_query = TimestampedSeries::try_new(
                &query,
                &[1.0, 2.0, 3.0],
                TimestampUnit::Seconds,
                ResourceLimits::default(),
            )
            .unwrap();
            let mut timestamped = TimestampedTwedOnlineAutomaton::new(
                timestamped_query,
                TimestampUnit::Seconds,
                0.0,
                MetricTimestampedTwedConfig::try_new(0.5, 1.0).unwrap(),
                1.0e12,
                limits,
            )
            .unwrap();
            let timestamped_retained = timestamped.scratch_bytes();
            for index in 0..PREFIX_LEN {
                assert!(timestamped
                    .advance((index % 3) as f64, (index + 1) as f64)
                    .unwrap()
                    .advanced());
                assert_eq!(timestamped.scratch_bytes(), timestamped_retained);
            }
            assert_eq!(timestamped.observation().consumed_target_len, PREFIX_LEN);

            let vector_query = VectorFrechetPath::try_from_rows(
                &[&[0.0, 0.0], &[1.0, 1.0], &[2.0, 2.0]],
                ResourceLimits::default(),
            )
            .unwrap();
            let mut vector =
                VectorFrechetOnlineAutomaton::new(vector_query, L1GroundMetric, 1.0e12, limits)
                    .unwrap();
            let vector_retained = vector.scratch_bytes();
            let vector_labels = (0..15)
                .map(|index| {
                    VectorSample::try_new(
                        &[(index % 3) as f64, (index % 5) as f64],
                        ResourceLimits::default(),
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            for index in 0..PREFIX_LEN {
                assert!(vector
                    .advance(&vector_labels[index % vector_labels.len()])
                    .unwrap()
                    .advanced());
                assert_eq!(vector.scratch_bytes(), vector_retained);
            }
            assert_eq!(vector.observation().consumed_target_len, PREFIX_LEN);
        })
        .expect("small-stack test thread must start")
        .join()
        .expect("online machines must not consume the process call stack");
}

#[test]
fn adversarial_dictionary_depth_uses_the_bounded_heap_stack() {
    const DEPTH: usize = 100_000;
    let index = MsmTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 1.0),
        MsmConfig::try_new(1.0).unwrap(),
        &[vec![0.0; DEPTH]],
    );
    let returned_index = std::thread::Builder::new()
        .name("temporal-product-small-stack".into())
        .stack_size(128 * 1024)
        .spawn(move || {
            {
                let page = PageBudget {
                    max_work_units: 1_000_000,
                    max_results: 1,
                };
                let mut outcome = index
                    .search_range_bounded(&[0.0], f64::INFINITY, ResourceLimits::default(), page)
                    .unwrap();
                loop {
                    match outcome {
                        OperationOutcome::Complete { value, .. } => {
                            assert_eq!(value.len(), 1);
                            assert_eq!(value[0].0, 0);
                            break;
                        }
                        OperationOutcome::Incomplete {
                            continuation: Some(next),
                            ..
                        } => outcome = next.resume(page),
                        other => panic!("deep bounded traversal terminated early: {other:?}"),
                    }
                }
            }
            // Return ownership so dictionary reclamation is tested separately
            // from the deliberately constrained traversal stack.
            index
        })
        .expect("small-stack product thread must start")
        .join()
        .expect("dictionary depth must not consume the process call stack");
    drop(returned_index);
}
