//! Compile-check for public elastic-time-series patterns in the documentation.
//!
//! This example keeps the README, time-series module documentation, and
//! elastic-measures guide synchronized with the public API. Run it with:
//!
//! ```sh
//! cargo check --example doc_time_series_check --all-features
//! ```

use liblevenshtein::time_series::elastic::ElasticKernel;
use liblevenshtein::time_series::{
    DtwConfig, DtwTransducer, MetricTwedConfig, MetricTwedTransducer, QuantizationConfig,
    TwedConfig,
};

fn main() {
    let references = vec![
        vec![0.0, 1.0, 2.0],
        vec![0.0, 1.0, 1.0, 2.0],
        vec![8.0, 9.0],
    ];
    let index = DtwTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 10.0),
        DtwConfig::new(1),
        &references,
    );

    // Thresholds and returned scores are root distances at the public boundary.
    let exact = index.search_range(&[0.0, 1.0, 2.0], 0.0);
    assert_eq!(exact.len(), 2);
    assert!(exact.iter().all(|(_, distance)| *distance == 0.0));

    let nearest = index.search_knn(&[0.0, 1.0, 2.0], 2, f64::INFINITY);
    assert_eq!(nearest.len(), 2);
    assert!(nearest.iter().all(|(_, distance)| *distance == 0.0));

    // The type-level status is suitable for generic algorithm selection.
    const { assert!(!DtwConfig::IS_METRIC) };

    // The unrestricted TWED family includes the non-metric `nu = 0` case.
    let degenerate = TwedConfig::new(0.0, 0.0);
    assert_eq!(degenerate.distance(&[0.0, 1.0], &[1.0]), 0.0);
    const { assert!(!TwedConfig::IS_METRIC) };

    // Validate `nu > 0` to obtain the type-level metric contract.
    let metric_twed = MetricTwedConfig::try_new(0.5, 1.0).expect("valid metric parameters");
    let twed_index = MetricTwedTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 10.0),
        metric_twed,
        &references,
    );
    assert_eq!(twed_index.search_range(&[0.0, 1.0, 2.0], 0.0).len(), 1);
    const { assert!(MetricTwedConfig::IS_METRIC) };
}
