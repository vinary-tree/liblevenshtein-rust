//! Shared time-series proptest strategies.

use proptest::prelude::*;
use std::ops::Range;

fn time_series_strategy(len: Range<usize>) -> impl Strategy<Value = Vec<f64>> {
    len.clone().prop_flat_map(move |length| {
        prop::collection::vec(-10.0f64..10.0f64, length..=length).prop_map(|deltas| {
            let mut series = Vec::with_capacity(deltas.len());
            let mut current = 0.0;
            for delta in deltas {
                current += delta * 0.1;
                series.push(current);
            }
            series
        })
    })
}

/// Generate short random-walk time series.
pub fn short_time_series_strategy() -> impl Strategy<Value = Vec<f64>> {
    time_series_strategy(2..10)
}
