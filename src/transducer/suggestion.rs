//! Value-aware spelling suggestions and confidence scoring.

/// A fuzzy dictionary match enriched with its stored value and derived score.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion<V> {
    /// Materialized dictionary term.
    pub term: String,
    /// Exact edit distance from the query.
    pub distance: usize,
    /// Generic value stored at the final dictionary node.
    pub value: V,
    /// Score derived by the configured [`SuggestionScorer`].
    pub confidence: f64,
}

/// Derive a confidence score without constraining the dictionary's value type.
pub trait SuggestionScorer<V> {
    /// Score one final-node value. Larger scores rank first within one distance
    /// layer. Non-finite scores are retained as least-confident by the iterator.
    fn confidence(&self, term: &str, distance: usize, value: &V) -> f64;
}

impl<V, F> SuggestionScorer<V> for F
where
    F: Fn(&str, usize, &V) -> f64,
{
    #[inline]
    fn confidence(&self, term: &str, distance: usize, value: &V) -> f64 {
        self(term, distance, value)
    }
}

/// Convert a non-negative stored count to the float domain used for ranking.
pub trait FrequencyValue {
    /// Lossy conversion is intentional: the result is used only for ordering,
    /// while the exact generic value remains in [`Suggestion::value`].
    fn frequency_f64(&self) -> f64;
}

macro_rules! impl_frequency_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl FrequencyValue for $ty {
                #[inline]
                fn frequency_f64(&self) -> f64 {
                    *self as f64
                }
            }
        )+
    };
}

impl_frequency_value!(u8, u16, u32, u64, u128, usize);

/// Rank frequencies on a compressed logarithmic scale.
///
/// Distance remains the iterator's primary key. The score is `ln(1 + f)`, so
/// very common terms lead within a distance layer without allowing frequency
/// to move a more distant match ahead of a closer one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogFrequencyScorer;

impl<V: FrequencyValue> SuggestionScorer<V> for LogFrequencyScorer {
    #[inline]
    fn confidence(&self, _term: &str, _distance: usize, value: &V) -> f64 {
        value.frequency_f64().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_frequency_is_monotone_and_keeps_the_exact_value_external() {
        let scorer = LogFrequencyScorer;
        assert!(scorer.confidence("common", 1, &100u64) > scorer.confidence("rare", 1, &2u64));
        assert_eq!(scorer.confidence("zero", 0, &0usize), 0.0);
    }
}
