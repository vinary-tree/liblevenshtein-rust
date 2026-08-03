//! Sakoe–Chiba envelopes and LB_Keogh for banded DTW.
//!
//! A query envelope stores the minimum and maximum query sample reachable from
//! each target position under the symmetric band. Candidate deviation outside
//! that interval is unavoidable, so the sum of squared deviations lower-bounds
//! squared banded DTW. The same geometry applies to a quantized target bin and
//! yields an O(1) incremental prefix bound per trie edge.

use std::collections::VecDeque;

use super::super::elastic::interval::interval_gap;
use crate::cost::{CostMonoid, WeightedCost};

#[inline]
fn square(value: f64) -> f64 {
    value * value
}

/// Query-side LB_Keogh metadata constructed once per search.
///
/// `lower[i]` and `upper[i]` are centered Sakoe–Chiba envelopes for target
/// position `i < query_len`. Suffix extrema answer the few legal positions
/// beyond the query end in O(1), without allocating memory proportional to a
/// caller-supplied band.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeoghPlan {
    lower: Vec<f64>,
    upper: Vec<f64>,
    suffix_lower: Vec<f64>,
    suffix_upper: Vec<f64>,
}

impl KeoghPlan {
    /// Number of query samples represented by the plan.
    #[inline]
    pub fn query_len(&self) -> usize {
        self.lower.len()
    }

    /// Centered lower envelope for positions inside the query.
    #[inline]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Centered upper envelope for positions inside the query.
    #[inline]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// Envelope bounds for one target position, or `None` if the band makes
    /// that position unreachable from every query sample.
    pub fn bounds_at(&self, target_index: usize, band: usize) -> Option<(f64, f64)> {
        let query_len = self.query_len();
        if target_index < query_len {
            return Some((self.lower[target_index], self.upper[target_index]));
        }
        let last_query = query_len.checked_sub(1)?;
        if target_index.abs_diff(last_query) > band {
            return None;
        }
        let first_reachable = target_index.saturating_sub(band);
        Some((
            *self.suffix_lower.get(first_reachable)?,
            *self.suffix_upper.get(first_reachable)?,
        ))
    }
}

/// Build centered Sakoe–Chiba envelopes in linear time.
///
/// Each query index enters and leaves each monotonic deque once. Non-finite
/// samples are rejected because their ordering and squared deviations do not
/// define a lawful lower bound.
pub fn keogh_envelopes(query: &[f64], band: usize) -> Option<KeoghPlan> {
    if query.is_empty() || query.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let mut lower = vec![0.0; query.len()];
    let mut upper = vec![0.0; query.len()];
    let mut minima = VecDeque::with_capacity(query.len());
    let mut maxima = VecDeque::with_capacity(query.len());
    let mut next_to_add = 0usize;

    for center in 0..query.len() {
        let right = center.saturating_add(band).min(query.len() - 1);
        while next_to_add <= right {
            while minima
                .back()
                .is_some_and(|index| query[*index] >= query[next_to_add])
            {
                minima.pop_back();
            }
            while maxima
                .back()
                .is_some_and(|index| query[*index] <= query[next_to_add])
            {
                maxima.pop_back();
            }
            minima.push_back(next_to_add);
            maxima.push_back(next_to_add);
            let Some(next) = next_to_add.checked_add(1) else {
                break;
            };
            next_to_add = next;
        }

        let left = center.saturating_sub(band);
        while minima.front().is_some_and(|index| *index < left) {
            minima.pop_front();
        }
        while maxima.front().is_some_and(|index| *index < left) {
            maxima.pop_front();
        }
        lower[center] = query[*minima.front()?];
        upper[center] = query[*maxima.front()?];
    }

    let mut suffix_lower = vec![0.0; query.len()];
    let mut suffix_upper = vec![0.0; query.len()];
    let last = query.len() - 1;
    suffix_lower[last] = query[last];
    suffix_upper[last] = query[last];
    for index in (0..last).rev() {
        suffix_lower[index] = query[index].min(suffix_lower[index + 1]);
        suffix_upper[index] = query[index].max(suffix_upper[index + 1]);
    }

    Some(KeoghPlan {
        lower,
        upper,
        suffix_lower,
        suffix_upper,
    })
}

/// Full candidate LB_Keogh in squared-cost units.
pub fn lb_keogh_squared(candidate: &[f64], band: usize, plan: &KeoghPlan) -> f64 {
    if candidate.iter().any(|value| !value.is_finite()) {
        return WeightedCost::TOP;
    }
    match (plan.query_len(), candidate.len()) {
        (0, 0) => return WeightedCost::ZERO,
        (0, _) | (_, 0) => return WeightedCost::TOP,
        _ => {}
    }
    if plan.query_len().abs_diff(candidate.len()) > band {
        return WeightedCost::TOP;
    }

    candidate
        .iter()
        .enumerate()
        .fold(WeightedCost::ZERO, |bound, (index, value)| {
            let Some((low, high)) = plan.bounds_at(index, band) else {
                return WeightedCost::TOP;
            };
            let deviation = if *value < low {
                low - *value
            } else if high < *value {
                *value - high
            } else {
                0.0
            };
            WeightedCost::combine(bound, square(deviation))
        })
}

/// Public root-distance LB_Keogh convenience function.
///
/// The kernel accumulates squared costs internally and exposes their square
/// root at the API boundary.
pub fn lb_keogh(query: &[f64], candidate: &[f64], band: usize) -> f64 {
    let Some(plan) = keogh_envelopes(query, band) else {
        return if query.is_empty() && candidate.is_empty() {
            0.0
        } else {
            f64::INFINITY
        };
    };
    lb_keogh_squared(candidate, band, &plan).sqrt()
}

/// Advance interval LB_Keogh for one quantized target edge.
pub(crate) fn interval_prefix_step(
    previous: f64,
    target_interval: (f64, f64),
    target_index: usize,
    band: usize,
    plan: &KeoghPlan,
) -> f64 {
    let Some(query_envelope) = plan.bounds_at(target_index, band) else {
        return WeightedCost::TOP;
    };
    let deviation = interval_gap(target_interval, query_envelope);
    WeightedCost::combine(previous, square(deviation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn naive_envelopes(query: &[f64], band: usize) -> (Vec<f64>, Vec<f64>) {
        let mut lower = Vec::with_capacity(query.len());
        let mut upper = Vec::with_capacity(query.len());
        for center in 0..query.len() {
            let left = center.saturating_sub(band);
            let right = center.saturating_add(band).min(query.len() - 1);
            lower.push(
                query[left..=right]
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
            );
            upper.push(
                query[left..=right]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
            );
        }
        (lower, upper)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn monotonic_deques_match_naive_centered_windows(
            query in prop::collection::vec(-30i16..=30, 1..20),
            band in 0usize..25,
        ) {
            let query: Vec<f64> = query.into_iter().map(f64::from).collect();
            let plan = keogh_envelopes(&query, band).expect("generated query is finite and nonempty");
            let (lower, upper) = naive_envelopes(&query, band);
            prop_assert_eq!(plan.lower(), lower.as_slice());
            prop_assert_eq!(plan.upper(), upper.as_slice());
        }
    }
}
