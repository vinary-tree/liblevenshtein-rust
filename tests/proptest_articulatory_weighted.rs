//! Property-based tests for the *weighted* articulatory distance (G4).
//!
//! These complement the example-based unit tests in
//! `src/phonetic/feature_distance.rs` and mirror the structure of
//! `tests/proptest_distance_metrics.rs`. They establish the metric-shaped
//! invariants of `articulatory_distance_weighted` for ARBITRARY non-negative
//! per-dimension weights, and pin the two behavioral contracts the HEAD commit
//! claims: default weights reproduce the unweighted functions exactly, and a
//! per-dimension weight increase never decreases the distance.
//!
//! The Coq theory `docs/verification/articulatory/theories/FeatureDistance.v`
//! and `FeatureDistanceWeighted.v` prove the same symmetry / identity /
//! boundedness / monotonicity facts over an idealized rational model; these
//! property tests check that the concrete `f64` implementation agrees.
//!
//! ## Cap interaction (why monotonicity is non-strict here)
//!
//! `feature_set_distance_weighted` caps its result at `1.0`. Once either operand
//! saturates the cap, raising a weight no longer increases the result, so the
//! honest property over the full weight/char space is **monotone
//! non-decreasing** (`>=`). Strict, pre-cap monotonicity is exercised by the
//! hand-picked unit tests in `feature_distance.rs`.

#![cfg(feature = "phonetic-rules")]

use liblevenshtein::phonetic::feature_distance::{
    articulatory_distance, articulatory_distance_weighted, articulatory_edit_distance,
    articulatory_edit_distance_weighted, FeatureDistanceWeights,
};
use proptest::prelude::*;

/// Characters that exercise the consonant path (voicing/place/manner), the vowel
/// path (height/backness/rounding), and the cross-category and unknown branches.
/// Characters absent from the IPA feature table simply yield distance `1.0`,
/// which still satisfies every invariant below.
fn arb_phon_char() -> impl Strategy<Value = char> {
    prop_oneof![
        proptest::char::range('a', 'z'),
        proptest::char::range('A', 'Z'),
        Just('ʃ'),
        Just('ʒ'),
        Just('θ'),
        Just('ð'),
        Just('ŋ'),
        Just('ə'),
        Just('ɛ'),
        Just('ɔ'),
        Just('æ'),
        Just('ø'),
        Just('ɨ'),
    ]
}

/// Non-negative, bounded per-dimension weights. `manner_table_scale` ranges
/// wider (`0..=2`) because it is a multiplier; the others are base distances.
/// The ranges deliberately allow weight combinations that push consonant pairs
/// past the `1.0` cap — exercising the cap is intentional.
fn arb_weights() -> impl Strategy<Value = FeatureDistanceWeights> {
    (
        0.0f64..=1.0, // voicing
        0.0f64..=1.0, // place_step
        0.0f64..=1.0, // manner_default
        0.0f64..=2.0, // manner_table_scale
        0.0f64..=1.0, // vowel_height_step
        0.0f64..=1.0, // vowel_backness_step
        0.0f64..=1.0, // vowel_rounding
    )
        .prop_map(
            |(
                voicing,
                place_step,
                manner_default,
                manner_table_scale,
                vowel_height_step,
                vowel_backness_step,
                vowel_rounding,
            )| FeatureDistanceWeights {
                voicing,
                place_step,
                manner_default,
                manner_table_scale,
                vowel_height_step,
                vowel_backness_step,
                vowel_rounding,
            },
        )
}

fn arb_word() -> impl Strategy<Value = String> {
    "[a-z]{0,8}"
}

/// Return `weights` with a single dimension (selected by `dim`) increased by
/// `delta`. Used to test per-dimension monotonicity.
fn bump(mut weights: FeatureDistanceWeights, dim: usize, delta: f64) -> FeatureDistanceWeights {
    match dim {
        0 => weights.voicing += delta,
        1 => weights.place_step += delta,
        2 => weights.manner_default += delta,
        3 => weights.manner_table_scale += delta,
        4 => weights.vowel_height_step += delta,
        5 => weights.vowel_backness_step += delta,
        _ => weights.vowel_rounding += delta,
    }
    weights
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Symmetry for all weights: each per-dimension term is symmetric, and they
    /// are accumulated in the same order, so the f64 result is bit-identical.
    #[test]
    fn weighted_symmetry_all_weights(a in arb_phon_char(), b in arb_phon_char(), w in arb_weights()) {
        prop_assert_eq!(
            articulatory_distance_weighted(a, b, &w),
            articulatory_distance_weighted(b, a, &w)
        );
    }

    /// Identity: a character against itself is exactly 0 (early return), for any weights.
    #[test]
    fn weighted_identity(a in arb_phon_char(), w in arb_weights()) {
        prop_assert_eq!(articulatory_distance_weighted(a, a, &w), 0.0);
    }

    /// Non-negativity: with non-negative weights every term is non-negative.
    #[test]
    fn weighted_non_negativity(a in arb_phon_char(), b in arb_phon_char(), w in arb_weights()) {
        prop_assert!(articulatory_distance_weighted(a, b, &w) >= 0.0);
    }

    /// Boundedness: the final `.min(1.0)` (and the cross-category / unknown
    /// branches) guarantee the result never exceeds 1.0, for ANY non-negative weights.
    #[test]
    fn weighted_boundedness(a in arb_phon_char(), b in arb_phon_char(), w in arb_weights()) {
        prop_assert!(articulatory_distance_weighted(a, b, &w) <= 1.0);
    }

    /// Default parity: `articulatory_distance` equals the weighted function with
    /// default weights over the full character space (regression guard for the
    /// delegation that keeps the unweighted API behavior-preserving).
    #[test]
    fn weighted_default_parity_all_pairs(a in arb_phon_char(), b in arb_phon_char()) {
        prop_assert_eq!(
            articulatory_distance(a, b),
            articulatory_distance_weighted(a, b, &FeatureDistanceWeights::default())
        );
    }

    /// Per-dimension monotonicity: increasing exactly one weight component never
    /// decreases the distance. Non-strict (`>=`) because of the `1.0` cap.
    #[test]
    fn weighted_per_dimension_monotonicity(
        a in arb_phon_char(),
        b in arb_phon_char(),
        w in arb_weights(),
        dim in 0usize..7,
        delta in 0.0f64..=0.5,
    ) {
        let base = articulatory_distance_weighted(a, b, &w);
        let bumped = articulatory_distance_weighted(a, b, &bump(w, dim, delta));
        prop_assert!(
            bumped >= base - 1e-9,
            "bumping dim {dim} by {delta} decreased distance: {base} -> {bumped}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Weighted edit distance with default weights equals the unweighted edit distance.
    #[test]
    fn weighted_edit_distance_default_parity(s in arb_word(), t in arb_word()) {
        prop_assert_eq!(
            articulatory_edit_distance(&s, &t),
            articulatory_edit_distance_weighted(&s, &t, &FeatureDistanceWeights::default())
        );
    }

    /// Weighted edit distance is symmetric: insertion/deletion costs are equal
    /// (1.0) and the per-cell substitution cost is symmetric, so the DP value is
    /// symmetric up to floating-point evaluation order.
    #[test]
    fn weighted_edit_distance_symmetry(s in arb_word(), t in arb_word(), w in arb_weights()) {
        let d_st = articulatory_edit_distance_weighted(&s, &t, &w);
        let d_ts = articulatory_edit_distance_weighted(&t, &s, &w);
        prop_assert!((d_st - d_ts).abs() < 1e-9, "edit distance asymmetric: {d_st} vs {d_ts}");
    }
}
