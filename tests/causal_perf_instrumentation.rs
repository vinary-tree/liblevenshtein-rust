#![cfg(feature = "perf-instrumentation")]

use libdictenstein::dynamic_dawg::DynamicDawg;
use liblevenshtein::transducer::{Algorithm, Transducer};
use liblevenshtein::{causal_perf_stats, reset_causal_perf_stats};

#[test]
fn native_query_counters_partition_observed_work() {
    let dictionary = DynamicDawg::<()>::from_terms(["cat", "cot", "coat", "dog"]);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);

    reset_causal_perf_stats();
    let matches: Vec<_> = transducer.query_with_distance("cat", 1).collect();
    let stats = causal_perf_stats();

    assert!(!matches.is_empty());
    assert_eq!(stats.final_checks, stats.dictionary_intersections);
    assert_eq!(stats.edges_enumerated, stats.transition_attempts);
    assert!(stats.transition_accepted <= stats.transition_attempts);
    assert_eq!(stats.matches_materialized, matches.len() as u64);
    assert!(stats.characteristic_vectors >= stats.transition_attempts);
    assert!(stats.characteristic_units >= stats.characteristic_vectors);
    assert!(stats.state_insert_attempts >= stats.state_insert_retained);
    assert!(stats.state_positions_enqueued >= stats.transition_accepted);
    assert_eq!(
        stats.state_bytes_enqueued,
        stats
            .state_positions_enqueued
            .saturating_mul(std::mem::size_of::<liblevenshtein::transducer::Position>() as u64)
    );
    assert!(stats.pool_acquires >= stats.pool_reuses);
    assert_eq!(stats.pool_acquires, stats.pool_reuses + stats.pool_misses);
}
