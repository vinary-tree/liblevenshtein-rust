//! Alignment-expressible operation-set presets.
//!
//! These are configurations of [`OperationSet`], not new lazy-automaton
//! variants. [`GeneralizedAutomaton`](super::generalized::GeneralizedAutomaton)
//! evaluates them exactly; specialized dictionary walkers remain rejected by
//! the frozen Phase-8 benchmark decision.

use super::{OperationSet, OperationSetBuilder};

impl OperationSet {
    /// Hamming operations: match and substitution only.
    ///
    /// Unequal-length inputs have no complete alignment.
    #[must_use]
    pub fn hamming() -> Self {
        OperationSetBuilder::new()
            .with_match()
            .with_substitution()
            .build()
    }

    /// Insertion/deletion operations: match, insertion, and deletion.
    ///
    /// Substitution is absent, so replacement costs two. The induced distance
    /// is `$`|x|+|y|-2\operatorname{LCS}(x,y)`$`; it is named `indel` because
    /// that value is a distance, whereas LCS itself is a similarity length.
    #[must_use]
    pub fn indel() -> Self {
        OperationSetBuilder::new()
            .with_match()
            .with_insertion()
            .with_deletion()
            .build()
    }

    /// Bounded-skip operations: match plus source deletion.
    ///
    /// `GeneralizedAutomaton::accepts(word, input)` succeeds exactly when
    /// `input` is a subsequence of `word` and the number of skipped source
    /// scalars fits the configured budget. This is structural subsequence
    /// matching only; it does not implement fzf scoring or ranking.
    #[must_use]
    pub fn bounded_skip() -> Self {
        OperationSetBuilder::new()
            .with_match()
            .with_deletion()
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_have_the_exact_declared_operations() {
        let names = |set: OperationSet| {
            set.iter()
                .map(|operation| operation.name().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(OperationSet::hamming()), ["match", "substitute"]);
        assert_eq!(names(OperationSet::indel()), ["match", "insert", "delete"]);
        assert_eq!(names(OperationSet::bounded_skip()), ["match", "delete"]);
    }
}
