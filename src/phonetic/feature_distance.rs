//! Articulatory feature distance computation.
//!
//! Computes phonetic similarity based on IPA feature vectors.
//! Captures intuitions like /p/ being closer to /b/ (voicing only)
//! than to /h/ (place + manner + voicing).
//!
//! # Usage
//!
//! ```rust
//! use liblevenshtein::phonetic::feature_distance::articulatory_distance;
//!
//! // Same sound = 0.0
//! assert_eq!(articulatory_distance('p', 'p'), 0.0);
//!
//! // Voicing difference only = small distance
//! let dist_pb = articulatory_distance('p', 'b');
//! assert!(dist_pb > 0.0 && dist_pb < 0.3);
//!
//! // Different place and manner = larger distance
//! let dist_ph = articulatory_distance('p', 'h');
//! assert!(dist_ph > dist_pb);
//! ```
//!
//! # Distance Matrix
//!
//! The distance computation considers three major dimensions:
//!
//! 1. **Voicing**: Binary feature (voiced vs voiceless)
//!    - Distance: 0.1 if different
//!
//! 2. **Place of articulation**: Location in the vocal tract
//!    - Adjacent places have small distance (e.g., bilabial ↔ labiodental = 0.2)
//!    - Non-adjacent places use shortest path through the chain
//!
//! 3. **Manner of articulation**: How air flows
//!    - Related manners have small distance (e.g., stop ↔ affricate = 0.2)
//!    - Unrelated manners have larger distance

use super::features::{get_features, PhoneticFeature};
use rustc_hash::FxHashSet;

/// Voicing distance (simple binary)
const VOICING_DISTANCE: f64 = 0.1;

/// Place of articulation ordering (front to back)
const PLACE_ORDER: &[PhoneticFeature] = &[
    PhoneticFeature::Bilabial,
    PhoneticFeature::Labiodental,
    PhoneticFeature::Dental,
    PhoneticFeature::Alveolar,
    PhoneticFeature::PostAlveolar,
    PhoneticFeature::Retroflex,
    PhoneticFeature::Palatal,
    PhoneticFeature::Velar,
    PhoneticFeature::Uvular,
    PhoneticFeature::Pharyngeal,
    PhoneticFeature::Epiglottal,
    PhoneticFeature::Glottal,
];

/// Distance per place step (adjacent places)
const PLACE_STEP_DISTANCE: f64 = 0.15;

/// Manner of articulation distances
/// These are based on acoustic and articulatory similarity
const MANNER_DISTANCES: &[(PhoneticFeature, PhoneticFeature, f64)] = &[
    // Stop related
    (PhoneticFeature::Stop, PhoneticFeature::Affricate, 0.2),
    (PhoneticFeature::Stop, PhoneticFeature::Fricative, 0.3),
    (PhoneticFeature::Stop, PhoneticFeature::Nasal, 0.3),
    (PhoneticFeature::Stop, PhoneticFeature::Approximant, 0.4),
    // Fricative related
    (PhoneticFeature::Fricative, PhoneticFeature::Affricate, 0.2),
    (
        PhoneticFeature::Fricative,
        PhoneticFeature::Approximant,
        0.3,
    ),
    // Nasal related
    (PhoneticFeature::Nasal, PhoneticFeature::Approximant, 0.3),
    // Approximant related
    (PhoneticFeature::Approximant, PhoneticFeature::Lateral, 0.1),
    (PhoneticFeature::Lateral, PhoneticFeature::Rhotic, 0.2),
    // Tap/Trill related
    (PhoneticFeature::Tap, PhoneticFeature::Trill, 0.1),
    (PhoneticFeature::Tap, PhoneticFeature::Rhotic, 0.1),
    (PhoneticFeature::Trill, PhoneticFeature::Rhotic, 0.1),
    (PhoneticFeature::Tap, PhoneticFeature::Approximant, 0.2),
    (PhoneticFeature::Trill, PhoneticFeature::Approximant, 0.2),
];

/// Default manner distance when no specific pair is defined
const DEFAULT_MANNER_DISTANCE: f64 = 0.5;

/// Vowel feature distances
const VOWEL_HEIGHT_STEP: f64 = 0.15; // high/mid/low
const VOWEL_BACKNESS_STEP: f64 = 0.15; // front/central/back
const VOWEL_ROUNDING_DIFF: f64 = 0.1; // rounded/unrounded

/// Per-dimension articulatory base-distance weights.
///
/// Each field is the base distance (cost) contributed by one feature dimension.
/// `Default` reproduces the module's built-in IPA constants exactly, so the
/// `*_weighted` functions called with `FeatureDistanceWeights::default()` are
/// identical to their unweighted counterparts (and all existing behavior /
/// tests are preserved). The curated manner-distance *table* keeps its relative
/// structure; `manner_table_scale` scales its magnitude uniformly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureDistanceWeights {
    /// Cost of a voicing difference between two consonants. Default 0.1.
    pub voicing: f64,
    /// Cost per place-of-articulation step (adjacent places). Default 0.15.
    pub place_step: f64,
    /// Fallback cost when a manner pair is not in the curated table (and the
    /// place fallback when place features are absent). Default 0.5.
    pub manner_default: f64,
    /// Multiplier on the curated manner-distance table entries. Default 1.0.
    pub manner_table_scale: f64,
    /// Cost per vowel-height step (high/mid/low). Default 0.15.
    pub vowel_height_step: f64,
    /// Cost per vowel-backness step (front/central/back). Default 0.15.
    pub vowel_backness_step: f64,
    /// Cost of a vowel rounding difference. Default 0.1.
    pub vowel_rounding: f64,
}

impl FeatureDistanceWeights {
    /// The built-in IPA constants as a `const` value (usable in `const fn`
    /// contexts such as `ArticulatoryCosts::new`). Equivalent to `default()`.
    pub const fn standard() -> Self {
        Self {
            voicing: VOICING_DISTANCE,
            place_step: PLACE_STEP_DISTANCE,
            manner_default: DEFAULT_MANNER_DISTANCE,
            manner_table_scale: 1.0,
            vowel_height_step: VOWEL_HEIGHT_STEP,
            vowel_backness_step: VOWEL_BACKNESS_STEP,
            vowel_rounding: VOWEL_ROUNDING_DIFF,
        }
    }
}

impl Default for FeatureDistanceWeights {
    fn default() -> Self {
        Self::standard()
    }
}

/// Compute articulatory distance between two characters.
///
/// Returns a value in [0.0, 1.0] where:
/// - 0.0 = identical sounds
/// - Small values = similar sounds (e.g., voicing pairs)
/// - Larger values = dissimilar sounds
/// - 1.0 = completely different or unknown
///
/// # Arguments
///
/// * `c1` - First character
/// * `c2` - Second character
///
/// # Returns
///
/// Normalized articulatory distance
pub fn articulatory_distance(c1: char, c2: char) -> f64 {
    articulatory_distance_weighted(c1, c2, &FeatureDistanceWeights::default())
}

/// Weighted variant of [`articulatory_distance`] using caller-supplied
/// per-dimension [`FeatureDistanceWeights`].
pub fn articulatory_distance_weighted(c1: char, c2: char, weights: &FeatureDistanceWeights) -> f64 {
    if c1 == c2 {
        return 0.0;
    }

    let f1 = get_features(c1);
    let f2 = get_features(c2);

    // Unknown characters get maximum distance
    if f1.is_empty() || f2.is_empty() {
        return 1.0;
    }

    feature_set_distance_weighted(&f1, &f2, weights)
}

/// Compute distance between two feature sets.
///
/// This is the core distance computation that considers:
/// 1. Whether both are vowels or consonants
/// 2. Voicing differences
/// 3. Place of articulation differences
/// 4. Manner of articulation differences
pub fn feature_set_distance(
    f1: &FxHashSet<PhoneticFeature>,
    f2: &FxHashSet<PhoneticFeature>,
) -> f64 {
    feature_set_distance_weighted(f1, f2, &FeatureDistanceWeights::default())
}

/// Weighted variant of [`feature_set_distance`].
pub fn feature_set_distance_weighted(
    f1: &FxHashSet<PhoneticFeature>,
    f2: &FxHashSet<PhoneticFeature>,
    weights: &FeatureDistanceWeights,
) -> f64 {
    let mut distance = 0.0;

    // Check if same category (vowel vs consonant)
    let v1_vowel = f1.contains(&PhoneticFeature::Vowel);
    let v2_vowel = f2.contains(&PhoneticFeature::Vowel);

    if v1_vowel != v2_vowel {
        // Vowel vs consonant: maximum distance
        return 1.0;
    }

    if v1_vowel {
        // Both vowels: compare vowel features
        distance += vowel_distance(f1, f2, weights);
    } else {
        // Both consonants: compare consonant features
        // Voicing difference
        let v1_voiced = f1.contains(&PhoneticFeature::Voiced);
        let v2_voiced = f2.contains(&PhoneticFeature::Voiced);
        if v1_voiced != v2_voiced {
            distance += weights.voicing;
        }

        // Place difference
        distance += place_distance(f1, f2, weights);

        // Manner difference
        distance += manner_distance(f1, f2, weights);
    }

    distance.min(1.0) // Cap at 1.0
}

/// Compute place of articulation distance.
fn place_distance(
    f1: &FxHashSet<PhoneticFeature>,
    f2: &FxHashSet<PhoneticFeature>,
    weights: &FeatureDistanceWeights,
) -> f64 {
    let pos1 = find_place_position(f1);
    let pos2 = find_place_position(f2);

    match (pos1, pos2) {
        (Some(p1), Some(p2)) => {
            let diff = (p1 as i32 - p2 as i32).unsigned_abs() as f64;
            diff * weights.place_step
        }
        // If one or both don't have place features, use default
        _ => weights.manner_default,
    }
}

/// Find the position in the PLACE_ORDER array for a feature set.
fn find_place_position(features: &FxHashSet<PhoneticFeature>) -> Option<usize> {
    for (i, place) in PLACE_ORDER.iter().enumerate() {
        if features.contains(place) {
            return Some(i);
        }
    }
    None
}

/// Compute manner of articulation distance.
fn manner_distance(
    f1: &FxHashSet<PhoneticFeature>,
    f2: &FxHashSet<PhoneticFeature>,
    weights: &FeatureDistanceWeights,
) -> f64 {
    let manner1 = find_manner(f1);
    let manner2 = find_manner(f2);

    match (manner1, manner2) {
        (Some(m1), Some(m2)) => {
            if m1 == m2 {
                return 0.0;
            }

            // Look up in manner distance table (scaled to preserve structure)
            for &(a, b, dist) in MANNER_DISTANCES {
                if (a == m1 && b == m2) || (a == m2 && b == m1) {
                    return dist * weights.manner_table_scale;
                }
            }

            // Default for unspecified pairs
            weights.manner_default
        }
        // If manner not found, use default
        _ => weights.manner_default,
    }
}

/// Find the primary manner feature.
fn find_manner(features: &FxHashSet<PhoneticFeature>) -> Option<PhoneticFeature> {
    const MANNER_FEATURES: &[PhoneticFeature] = &[
        PhoneticFeature::Stop,
        PhoneticFeature::Fricative,
        PhoneticFeature::Affricate,
        PhoneticFeature::Nasal,
        PhoneticFeature::Approximant,
        PhoneticFeature::Lateral,
        PhoneticFeature::Rhotic,
        PhoneticFeature::Tap,
        PhoneticFeature::Trill,
    ];

    for manner in MANNER_FEATURES {
        if features.contains(manner) {
            return Some(*manner);
        }
    }
    None
}

/// Compute distance between vowels.
fn vowel_distance(
    f1: &FxHashSet<PhoneticFeature>,
    f2: &FxHashSet<PhoneticFeature>,
    weights: &FeatureDistanceWeights,
) -> f64 {
    let mut distance = 0.0;

    // Height difference
    let h1 = vowel_height(f1);
    let h2 = vowel_height(f2);
    distance += (h1 - h2).abs() as f64 * weights.vowel_height_step;

    // Backness difference
    let b1 = vowel_backness(f1);
    let b2 = vowel_backness(f2);
    distance += (b1 - b2).abs() as f64 * weights.vowel_backness_step;

    // Rounding difference
    let r1 = f1.contains(&PhoneticFeature::Rounded);
    let r2 = f2.contains(&PhoneticFeature::Rounded);
    if r1 != r2 {
        distance += weights.vowel_rounding;
    }

    distance
}

/// Get vowel height as numeric value (0=low, 1=mid, 2=high).
fn vowel_height(features: &FxHashSet<PhoneticFeature>) -> i32 {
    if features.contains(&PhoneticFeature::High) {
        2
    } else if features.contains(&PhoneticFeature::Mid) {
        1
    } else if features.contains(&PhoneticFeature::Low) {
        0
    } else {
        1 // Default to mid
    }
}

/// Get vowel backness as numeric value (0=front, 1=central, 2=back).
fn vowel_backness(features: &FxHashSet<PhoneticFeature>) -> i32 {
    if features.contains(&PhoneticFeature::Back) {
        2
    } else if features.contains(&PhoneticFeature::Central) {
        1
    } else if features.contains(&PhoneticFeature::Front) {
        0
    } else {
        1 // Default to central
    }
}

/// Check if substitution is "free" (sounds are very similar).
///
/// Returns true if the articulatory distance is below a threshold,
/// indicating the sounds are phonetically similar enough that
/// confusing them is understandable.
///
/// # Examples
///
/// - p ↔ b (voicing only) → true
/// - t ↔ d (voicing only) → true
/// - a ↔ e (height difference) → false
pub fn is_free_substitution(c1: char, c2: char) -> bool {
    articulatory_distance(c1, c2) < 0.15
}

/// Compute edit distance with articulatory costs.
///
/// This is an alternative to standard Levenshtein distance that uses
/// articulatory distance for substitution costs instead of the binary
/// 0/1 cost model.
///
/// # Arguments
///
/// * `source` - Source string
/// * `target` - Target string
///
/// # Returns
///
/// Weighted edit distance where substitutions are weighted by
/// articulatory similarity.
pub fn articulatory_edit_distance(source: &str, target: &str) -> f64 {
    articulatory_edit_distance_weighted(source, target, &FeatureDistanceWeights::default())
}

fn articulatory_dp_row_len(target_len: usize) -> Option<usize> {
    target_len.checked_add(1)
}

/// Weighted variant of [`articulatory_edit_distance`]: substitution costs use
/// [`articulatory_distance_weighted`] with the supplied weights.
pub fn articulatory_edit_distance_weighted(
    source: &str,
    target: &str,
    weights: &FeatureDistanceWeights,
) -> f64 {
    let source_chars: Vec<char> = source.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();

    let m = source_chars.len();
    let n = target_chars.len();

    if m == 0 {
        return n as f64;
    }
    if n == 0 {
        return m as f64;
    }

    let Some(row_len) = articulatory_dp_row_len(n) else {
        return f64::INFINITY;
    };

    // DP with f64 costs
    let mut prev_row: Vec<f64> = (0..row_len).map(|j| j as f64).collect();
    let mut curr_row = vec![0.0; row_len];

    for i in 1..=m {
        curr_row[0] = i as f64;
        for j in 1..=n {
            let sub_cost =
                articulatory_distance_weighted(source_chars[i - 1], target_chars[j - 1], weights);
            curr_row[j] = (prev_row[j] + 1.0) // deletion
                .min(curr_row[j - 1] + 1.0) // insertion
                .min(prev_row[j - 1] + sub_cost); // substitution
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn articulatory_dp_row_len_checks_overflow() {
        assert_eq!(articulatory_dp_row_len(0), Some(1));
        assert_eq!(articulatory_dp_row_len(4), Some(5));
        assert_eq!(articulatory_dp_row_len(usize::MAX), None);
    }

    #[test]
    fn test_same_character() {
        assert_eq!(articulatory_distance('p', 'p'), 0.0);
        assert_eq!(articulatory_distance('a', 'a'), 0.0);
    }

    #[test]
    fn test_voicing_pairs() {
        // Voicing pairs should have small distance
        let dist_pb = articulatory_distance('p', 'b');
        let dist_td = articulatory_distance('t', 'd');
        let dist_kg = articulatory_distance('k', 'g');
        let dist_fv = articulatory_distance('f', 'v');
        let dist_sz = articulatory_distance('s', 'z');

        // All should be around VOICING_DISTANCE
        assert!(dist_pb > 0.0 && dist_pb < 0.2, "p-b dist: {}", dist_pb);
        assert!(dist_td > 0.0 && dist_td < 0.2, "t-d dist: {}", dist_td);
        assert!(dist_kg > 0.0 && dist_kg < 0.2, "k-g dist: {}", dist_kg);
        assert!(dist_fv > 0.0 && dist_fv < 0.2, "f-v dist: {}", dist_fv);
        assert!(dist_sz > 0.0 && dist_sz < 0.2, "s-z dist: {}", dist_sz);
    }

    #[test]
    fn test_different_place() {
        // Same manner, different place
        let dist_pt = articulatory_distance('p', 't'); // bilabial vs alveolar
        let dist_pk = articulatory_distance('p', 'k'); // bilabial vs velar

        // Different places should be further apart
        assert!(dist_pt > 0.0, "p-t should have positive distance");
        assert!(dist_pk > dist_pt, "p-k should be further than p-t");
    }

    #[test]
    fn test_different_manner() {
        // Same place, different manner
        let dist_ps = articulatory_distance('p', 'f'); // stop vs fricative, bilabial vs labiodental
        let dist_pm = articulatory_distance('p', 'm'); // stop vs nasal, same place

        assert!(dist_ps > 0.0);
        assert!(dist_pm > 0.0);
    }

    #[test]
    fn test_vowel_vs_consonant() {
        // Vowel vs consonant should be maximum distance
        let dist_ap = articulatory_distance('a', 'p');
        assert_eq!(dist_ap, 1.0);
    }

    #[test]
    fn test_vowel_distances() {
        // Similar vowels
        let dist_ae = articulatory_distance('a', 'e'); // different backness
        let dist_ai = articulatory_distance('a', 'i'); // different height + backness

        assert!(dist_ae > 0.0);
        assert!(dist_ai > dist_ae, "a-i should be further than a-e");
    }

    #[test]
    fn test_unknown_character() {
        // Unknown characters get max distance
        let dist = articulatory_distance('p', '£');
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_is_free_substitution() {
        // Voicing pairs should be free substitutions
        assert!(is_free_substitution('p', 'b'));
        assert!(is_free_substitution('t', 'd'));

        // Different sounds should not be free
        assert!(!is_free_substitution('p', 'h'));
        assert!(!is_free_substitution('a', 'u'));
    }

    #[test]
    fn test_articulatory_edit_distance() {
        // Same strings
        assert_eq!(articulatory_edit_distance("test", "test"), 0.0);

        // Empty strings
        assert_eq!(articulatory_edit_distance("", "test"), 4.0);
        assert_eq!(articulatory_edit_distance("test", ""), 4.0);

        // Single substitution with similar sounds
        let dist = articulatory_edit_distance("pat", "bat");
        assert!(
            dist > 0.0 && dist < 1.0,
            "pat-bat should have fractional distance: {}",
            dist
        );

        // Single substitution with different sounds
        let dist2 = articulatory_edit_distance("pat", "hat");
        assert!(dist2 > dist, "pat-hat should be more than pat-bat");
    }

    #[test]
    fn test_symmetry() {
        // Distance should be symmetric
        assert_eq!(
            articulatory_distance('p', 'b'),
            articulatory_distance('b', 'p')
        );
        assert_eq!(
            articulatory_distance('a', 'i'),
            articulatory_distance('i', 'a')
        );
    }

    #[test]
    fn test_ipa_characters() {
        // Test IPA characters
        let dist_sh = articulatory_distance('ʃ', 'ʒ'); // sh vs zh (voicing)
        assert!(dist_sh > 0.0 && dist_sh < 0.2);

        let dist_theta_eth = articulatory_distance('θ', 'ð'); // th voiced/voiceless
        assert!(dist_theta_eth > 0.0 && dist_theta_eth < 0.2);
    }

    #[test]
    fn weighted_default_matches_unweighted() {
        // Default weights reproduce the built-in constants exactly.
        let w = FeatureDistanceWeights::default();
        for (a, b) in [
            ('p', 'b'),
            ('p', 'k'),
            ('a', 'e'),
            ('s', 'z'),
            ('p', 'h'),
            ('t', 'd'),
            ('a', 'i'),
        ] {
            assert_eq!(
                articulatory_distance(a, b),
                articulatory_distance_weighted(a, b, &w),
                "default-weighted must match unweighted for {a}-{b}"
            );
        }
        assert_eq!(
            articulatory_edit_distance("receive", "recieve"),
            articulatory_edit_distance_weighted("receive", "recieve", &w)
        );
    }

    #[test]
    fn weighted_voicing_scales_monotonically() {
        // p↔b differ only in voicing; the distance equals the voicing weight, so
        // raising it raises the distance.
        let low = FeatureDistanceWeights {
            voicing: 0.05,
            ..Default::default()
        };
        let high = FeatureDistanceWeights {
            voicing: 0.5,
            ..Default::default()
        };
        let d_low = articulatory_distance_weighted('p', 'b', &low);
        let d_high = articulatory_distance_weighted('p', 'b', &high);
        assert!(
            d_high > d_low,
            "higher voicing weight must increase p-b distance: {d_low} vs {d_high}"
        );
        assert!((d_low - 0.05).abs() < 1e-9);
        assert!((d_high - 0.5).abs() < 1e-9);
    }

    // --- Per-dimension weight tests ---------------------------------------
    //
    // Each pair below is chosen to ISOLATE one weight dimension while keeping
    // the pre-cap distance well under 1.0, so a weight increase is observable
    // (the final `.min(1.0)` cap would otherwise saturate it). Feature sets
    // (verified in `features.rs`): p=[Stop,Bilabial,Voiceless],
    // b=[Stop,Bilabial,Voiced], t=[Stop,Alveolar,Voiceless],
    // m=[Nasal,Bilabial,Voiced], e=[Vowel,Mid,Front,Unrounded],
    // i=[Vowel,High,Front,Unrounded], ɨ=[Vowel,High,Central,Unrounded],
    // ø=[Vowel,Mid,Front,Rounded]. Place indices: Bilabial=0, Alveolar=3.
    //
    // "Monotone": raising the relevant weight strictly increases the distance.
    // "Isolated": raising an irrelevant weight leaves the distance unchanged.

    #[test]
    fn weighted_voicing_isolated() {
        // p/b differ only in voicing → distance == voicing (0.1 by default).
        let base = FeatureDistanceWeights::default();
        let d = articulatory_distance_weighted('p', 'b', &base);
        assert!((d - 0.1).abs() < 1e-9, "p/b expected 0.1, got {d}");
        // Same place (diff 0), same manner (Stop), not vowels: nothing else applies.
        for w in [
            FeatureDistanceWeights {
                place_step: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                manner_default: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                manner_table_scale: 5.0,
                ..base
            },
            FeatureDistanceWeights {
                vowel_height_step: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                vowel_backness_step: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                vowel_rounding: 0.9,
                ..base
            },
        ] {
            assert_eq!(articulatory_distance_weighted('p', 'b', &w), d);
        }
    }

    #[test]
    fn weighted_place_step_monotonic_and_isolated() {
        // p/t: voiceless stops differing only in place (Bilabial 0 vs Alveolar 3)
        // → distance == 3 * place_step (= 0.45 by default).
        let base = FeatureDistanceWeights::default();
        let d = articulatory_distance_weighted('p', 't', &base);
        assert!((d - 0.45).abs() < 1e-9, "p/t expected 0.45, got {d}");
        let lo = FeatureDistanceWeights {
            place_step: 0.10,
            ..base
        };
        let hi = FeatureDistanceWeights {
            place_step: 0.20,
            ..base
        };
        assert!(
            articulatory_distance_weighted('p', 't', &hi)
                > articulatory_distance_weighted('p', 't', &lo),
            "raising place_step must increase p/t distance"
        );
        for w in [
            FeatureDistanceWeights {
                voicing: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                manner_table_scale: 5.0,
                ..base
            },
            FeatureDistanceWeights {
                vowel_rounding: 0.9,
                ..base
            },
        ] {
            assert_eq!(articulatory_distance_weighted('p', 't', &w), d);
        }
    }

    #[test]
    fn weighted_manner_table_scale_monotonic_and_isolated() {
        // p/m: both Bilabial; Stop↔Nasal is a curated table entry (0.3). They also
        // differ in voicing, so distance = voicing + 0.3*manner_table_scale (= 0.4).
        // The additive structure makes it strictly increasing in manner_table_scale.
        let base = FeatureDistanceWeights::default();
        let d = articulatory_distance_weighted('p', 'm', &base);
        assert!((d - 0.4).abs() < 1e-9, "p/m expected 0.4, got {d}");
        let lo = FeatureDistanceWeights {
            manner_table_scale: 0.5,
            ..base
        };
        let hi = FeatureDistanceWeights {
            manner_table_scale: 2.0,
            ..base
        };
        assert!(
            articulatory_distance_weighted('p', 'm', &hi)
                > articulatory_distance_weighted('p', 'm', &lo),
            "raising manner_table_scale must increase p/m distance"
        );
        for w in [
            FeatureDistanceWeights {
                vowel_height_step: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                vowel_backness_step: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                vowel_rounding: 0.9,
                ..base
            },
        ] {
            assert_eq!(articulatory_distance_weighted('p', 'm', &w), d);
        }
    }

    #[test]
    fn weighted_manner_default_monotonic() {
        // Hand-built consonant feature sets where f1 lacks a manner feature, forcing
        // the `manner_default` fallback in manner_distance. Same place (Bilabial,
        // diff 0), neither voiced → distance is exactly `manner_default`.
        let mut f1 = FxHashSet::default();
        f1.insert(PhoneticFeature::Consonant);
        f1.insert(PhoneticFeature::Bilabial);
        let mut f2 = FxHashSet::default();
        f2.insert(PhoneticFeature::Consonant);
        f2.insert(PhoneticFeature::Bilabial);
        f2.insert(PhoneticFeature::Stop);

        let lo = FeatureDistanceWeights {
            manner_default: 0.3,
            ..Default::default()
        };
        let hi = FeatureDistanceWeights {
            manner_default: 0.7,
            ..Default::default()
        };
        let d_lo = feature_set_distance_weighted(&f1, &f2, &lo);
        let d_hi = feature_set_distance_weighted(&f1, &f2, &hi);
        assert!(
            (d_lo - 0.3).abs() < 1e-9,
            "expected manner_default 0.3, got {d_lo}"
        );
        assert!(
            (d_hi - 0.7).abs() < 1e-9,
            "expected manner_default 0.7, got {d_hi}"
        );
        assert!(
            d_hi > d_lo,
            "raising manner_default must increase the distance"
        );
    }

    #[test]
    fn weighted_vowel_height_monotonic_and_isolated() {
        // e/i: both Front, Unrounded; differ only in height → distance == vowel_height_step.
        let base = FeatureDistanceWeights::default();
        let d = articulatory_distance_weighted('e', 'i', &base);
        assert!((d - 0.15).abs() < 1e-9, "e/i expected 0.15, got {d}");
        let lo = FeatureDistanceWeights {
            vowel_height_step: 0.10,
            ..base
        };
        let hi = FeatureDistanceWeights {
            vowel_height_step: 0.30,
            ..base
        };
        assert!(
            articulatory_distance_weighted('e', 'i', &hi)
                > articulatory_distance_weighted('e', 'i', &lo),
            "raising vowel_height_step must increase e/i distance"
        );
        for w in [
            FeatureDistanceWeights {
                vowel_backness_step: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                vowel_rounding: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                voicing: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                place_step: 0.9,
                ..base
            },
        ] {
            assert_eq!(articulatory_distance_weighted('e', 'i', &w), d);
        }
    }

    #[test]
    fn weighted_vowel_backness_monotonic_and_isolated() {
        // i/ɨ: both High, Unrounded; differ only in backness → distance == vowel_backness_step.
        let base = FeatureDistanceWeights::default();
        let d = articulatory_distance_weighted('i', 'ɨ', &base);
        assert!((d - 0.15).abs() < 1e-9, "i/ɨ expected 0.15, got {d}");
        let lo = FeatureDistanceWeights {
            vowel_backness_step: 0.10,
            ..base
        };
        let hi = FeatureDistanceWeights {
            vowel_backness_step: 0.30,
            ..base
        };
        assert!(
            articulatory_distance_weighted('i', 'ɨ', &hi)
                > articulatory_distance_weighted('i', 'ɨ', &lo),
            "raising vowel_backness_step must increase i/ɨ distance"
        );
        for w in [
            FeatureDistanceWeights {
                vowel_height_step: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                vowel_rounding: 0.9,
                ..base
            },
        ] {
            assert_eq!(articulatory_distance_weighted('i', 'ɨ', &w), d);
        }
    }

    #[test]
    fn weighted_vowel_rounding_monotonic_and_isolated() {
        // e/ø: both Mid, Front; differ only in rounding → distance == vowel_rounding.
        let base = FeatureDistanceWeights::default();
        let d = articulatory_distance_weighted('e', 'ø', &base);
        assert!((d - 0.1).abs() < 1e-9, "e/ø expected 0.1, got {d}");
        let lo = FeatureDistanceWeights {
            vowel_rounding: 0.05,
            ..base
        };
        let hi = FeatureDistanceWeights {
            vowel_rounding: 0.40,
            ..base
        };
        assert!(
            articulatory_distance_weighted('e', 'ø', &hi)
                > articulatory_distance_weighted('e', 'ø', &lo),
            "raising vowel_rounding must increase e/ø distance"
        );
        for w in [
            FeatureDistanceWeights {
                vowel_height_step: 0.9,
                ..base
            },
            FeatureDistanceWeights {
                vowel_backness_step: 0.9,
                ..base
            },
        ] {
            assert_eq!(articulatory_distance_weighted('e', 'ø', &w), d);
        }
    }

    #[test]
    fn weighted_standard_equals_default() {
        assert_eq!(
            FeatureDistanceWeights::standard(),
            FeatureDistanceWeights::default()
        );
        let w = FeatureDistanceWeights::standard();
        assert_eq!(w.voicing, VOICING_DISTANCE);
        assert_eq!(w.place_step, PLACE_STEP_DISTANCE);
        assert_eq!(w.manner_default, DEFAULT_MANNER_DISTANCE);
        assert_eq!(w.manner_table_scale, 1.0);
        assert_eq!(w.vowel_height_step, VOWEL_HEIGHT_STEP);
        assert_eq!(w.vowel_backness_step, VOWEL_BACKNESS_STEP);
        assert_eq!(w.vowel_rounding, VOWEL_ROUNDING_DIFF);
    }
}
