//! Articulatory-aware operation costs for Levenshtein automata.
//!
//! This module provides `ArticulatoryCosts`, a configuration structure for
//! assigning phonetically-informed substitution costs based on articulatory
//! feature distances between characters.
//!
//! # Overview
//!
//! Unlike standard Levenshtein distance where all substitutions cost 1.0,
//! articulatory costs reflect phonetic similarity:
//!
//! - **Similar sounds**: /p/ → /b/ (voicing only) costs less than /p/ → /h/
//! - **Vowel gradation**: /i/ → /e/ (height difference) costs less than /i/ → /o/
//! - **Place of articulation**: Adjacent places (bilabial ↔ labiodental) cost less
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::transducer::ArticulatoryCosts;
//!
//! let costs = ArticulatoryCosts::default();
//!
//! // Similar sounds have low substitution cost
//! let pb_cost = costs.substitution_cost('p', 'b');
//! assert!(pb_cost < 0.5); // Only voicing differs
//!
//! // Distant sounds have high substitution cost
//! let ph_cost = costs.substitution_cost('p', 'h');
//! assert!(ph_cost > 0.7); // Place + manner differ significantly
//! ```
//!
//! # Integration with Automata
//!
//! These costs integrate with the float-weighted Levenshtein automata
//! (`IntersectionF64`, `QueryIteratorF64`) for phonetically-aware fuzzy matching.
//!
//! # Theoretical Basis
//!
//! The articulatory distance model is based on IPA phonetic features:
//! - **Place of articulation**: bilabial → glottal (9 positions)
//! - **Manner of articulation**: stop, fricative, nasal, approximant, etc.
//! - **Voicing**: voiced vs voiceless
//! - **Vowel features**: height, backness, rounding
//!
//! See `src/phonetic/feature_distance.rs` for the underlying distance computation.

use std::fmt;

use super::costs_f64::OperationCostsF64;

/// Weight for articulatory distance in substitution cost blending.
/// 0.0 = use base cost only, 1.0 = use articulatory distance only.
const DEFAULT_ARTICULATION_WEIGHT: f64 = 0.6;

/// Threshold below which substitution is considered "free" (near-identical sounds).
const FREE_SUBSTITUTION_THRESHOLD: f64 = 0.15;

/// Operation costs with articulatory feature awareness.
///
/// Extends `OperationCostsF64` by providing character-specific substitution
/// costs based on phonetic similarity.
///
/// # Fields
///
/// | Field | Description |
/// |-------|-------------|
/// | `base` | Base operation costs (insertion, deletion, transposition, etc.) |
/// | `articulation_weight` | Weight for articulatory distance in substitution (0.0-1.0) |
/// | `free_substitution_threshold` | Distance below which sounds are "free" to substitute |
///
/// # Cost Blending
///
/// The substitution cost is computed as:
///
/// ```text
/// cost = base.substitution * (1 - weight) + art_distance * weight
/// ```
///
/// Where `art_distance` is the articulatory distance from `feature_distance.rs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArticulatoryCosts {
    /// Base operation costs for non-substitution operations.
    pub base: OperationCostsF64,

    /// Weight for articulatory distance in substitution cost (0.0-1.0).
    ///
    /// - 0.0: Use base substitution cost only (ignores phonetics)
    /// - 0.5: Equal blend of base cost and articulatory distance
    /// - 1.0: Use articulatory distance only
    pub articulation_weight: f64,

    /// Distance threshold for "free" substitutions.
    ///
    /// If articulatory distance is below this threshold, the sounds are
    /// considered nearly identical and substitution cost is reduced to near-zero.
    pub free_substitution_threshold: f64,
}

impl ArticulatoryCosts {
    /// Create articulatory costs with default settings.
    ///
    /// Uses standard base costs, 0.6 articulation weight, and 0.15 free threshold.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::transducer::ArticulatoryCosts;
    ///
    /// let costs = ArticulatoryCosts::new();
    /// assert_eq!(costs.articulation_weight, 0.6);
    /// ```
    #[inline]
    pub const fn new() -> Self {
        Self {
            base: OperationCostsF64::standard(),
            articulation_weight: DEFAULT_ARTICULATION_WEIGHT,
            free_substitution_threshold: FREE_SUBSTITUTION_THRESHOLD,
        }
    }

    /// Create articulatory costs with custom articulation weight.
    ///
    /// # Arguments
    ///
    /// * `weight` - Articulation weight (0.0-1.0)
    ///
    /// # Panics
    ///
    /// Panics if weight is not in [0.0, 1.0].
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::transducer::ArticulatoryCosts;
    ///
    /// // Heavily weight articulatory features
    /// let costs = ArticulatoryCosts::with_weight(0.8);
    /// ```
    pub fn with_weight(weight: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&weight),
            "Articulation weight must be in [0.0, 1.0], got {}",
            weight
        );
        Self {
            base: OperationCostsF64::standard(),
            articulation_weight: weight,
            free_substitution_threshold: FREE_SUBSTITUTION_THRESHOLD,
        }
    }

    /// Create articulatory costs with custom base costs.
    ///
    /// # Arguments
    ///
    /// * `base` - Base operation costs
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::transducer::{ArticulatoryCosts, OperationCostsF64};
    ///
    /// let base = OperationCostsF64::typo_friendly();
    /// let costs = ArticulatoryCosts::with_base(base);
    /// ```
    pub fn with_base(base: OperationCostsF64) -> Self {
        Self {
            base,
            articulation_weight: DEFAULT_ARTICULATION_WEIGHT,
            free_substitution_threshold: FREE_SUBSTITUTION_THRESHOLD,
        }
    }

    /// Create fully customized articulatory costs.
    ///
    /// # Arguments
    ///
    /// * `base` - Base operation costs
    /// * `articulation_weight` - Weight for articulatory distance (0.0-1.0)
    /// * `free_threshold` - Threshold for free substitutions
    ///
    /// # Panics
    ///
    /// Panics if articulation_weight is not in [0.0, 1.0] or free_threshold is negative.
    pub fn custom(base: OperationCostsF64, articulation_weight: f64, free_threshold: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&articulation_weight),
            "Articulation weight must be in [0.0, 1.0], got {}",
            articulation_weight
        );
        assert!(
            free_threshold >= 0.0,
            "Free substitution threshold must be non-negative, got {}",
            free_threshold
        );
        Self {
            base,
            articulation_weight,
            free_substitution_threshold: free_threshold,
        }
    }

    /// Compute substitution cost between two characters.
    ///
    /// Blends the base substitution cost with the articulatory distance
    /// according to the articulation weight.
    ///
    /// # Arguments
    ///
    /// * `from` - Source character
    /// * `to` - Target character
    ///
    /// # Returns
    ///
    /// Substitution cost in [0.0, base.substitution]
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::transducer::ArticulatoryCosts;
    ///
    /// let costs = ArticulatoryCosts::default();
    ///
    /// // Same character = free
    /// assert_eq!(costs.substitution_cost('a', 'a'), 0.0);
    ///
    /// // Similar sounds = cheap
    /// let pb = costs.substitution_cost('p', 'b');
    /// let pk = costs.substitution_cost('p', 'k');
    /// assert!(pb < pk); // p→b cheaper than p→k
    /// ```
    #[cfg(feature = "phonetic-rules")]
    pub fn substitution_cost(&self, from: char, to: char) -> f64 {
        if from == to {
            return 0.0;
        }

        let art_dist = crate::phonetic::feature_distance::articulatory_distance(from, to);

        // Check for free substitution (very similar sounds)
        if art_dist < self.free_substitution_threshold {
            return art_dist * 0.1; // Near-zero but not exactly zero
        }

        // Blend base cost with articulatory distance
        self.base.substitution * (1.0 - self.articulation_weight)
            + art_dist * self.articulation_weight
    }

    /// Compute substitution cost between two characters (fallback without phonetic-rules).
    #[cfg(not(feature = "phonetic-rules"))]
    pub fn substitution_cost(&self, from: char, to: char) -> f64 {
        if from == to {
            0.0
        } else {
            self.base.substitution
        }
    }

    /// Check if substitution between two characters is "free" (near-identical sounds).
    ///
    /// Returns `true` if the articulatory distance is below the free threshold.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::transducer::ArticulatoryCosts;
    ///
    /// let costs = ArticulatoryCosts::default();
    ///
    /// // Voicing pairs are often free
    /// assert!(costs.is_free_substitution('p', 'b'));
    /// assert!(costs.is_free_substitution('t', 'd'));
    ///
    /// // Distant sounds are not free
    /// assert!(!costs.is_free_substitution('p', 'h'));
    /// ```
    #[cfg(feature = "phonetic-rules")]
    pub fn is_free_substitution(&self, from: char, to: char) -> bool {
        if from == to {
            return true;
        }
        crate::phonetic::feature_distance::articulatory_distance(from, to)
            < self.free_substitution_threshold
    }

    /// Check if substitution is free (fallback without phonetic-rules).
    #[cfg(not(feature = "phonetic-rules"))]
    pub fn is_free_substitution(&self, from: char, to: char) -> bool {
        from == to
    }

    /// Get the insertion cost.
    #[inline]
    pub fn insertion_cost(&self) -> f64 {
        self.base.insertion
    }

    /// Get the deletion cost.
    #[inline]
    pub fn deletion_cost(&self) -> f64 {
        self.base.deletion
    }

    /// Get the transposition cost.
    #[inline]
    pub fn transposition_cost(&self) -> f64 {
        self.base.transposition
    }

    /// Get the split cost.
    #[inline]
    pub fn split_cost(&self) -> f64 {
        self.base.split
    }

    /// Get the merge cost.
    #[inline]
    pub fn merge_cost(&self) -> f64 {
        self.base.merge
    }

    /// Validate that all costs are properly configured.
    ///
    /// Returns `true` if:
    /// - Base costs are valid
    /// - Articulation weight is in [0.0, 1.0]
    /// - Free threshold is non-negative
    pub fn is_valid(&self) -> bool {
        self.base.is_valid()
            && (0.0..=1.0).contains(&self.articulation_weight)
            && self.free_substitution_threshold >= 0.0
    }

    /// Get the minimum non-zero cost among all operations.
    ///
    /// This is useful for computing lower bounds in pruning strategies.
    /// Note: Substitution minimum depends on character pairs, so this
    /// uses the theoretical minimum (near-zero for free substitutions).
    pub fn min_nonzero_cost(&self) -> f64 {
        // Minimum substitution cost is near-zero for free substitutions
        let min_sub = self.free_substitution_threshold * 0.1;

        let costs = [
            min_sub,
            self.base.insertion,
            self.base.deletion,
            self.base.transposition,
            self.base.split,
            self.base.merge,
        ];

        costs
            .iter()
            .copied()
            .filter(|&c| c > 0.0)
            .min_by(|a, b| a.partial_cmp(b).expect("valid f64"))
            .unwrap_or(0.01) // Fallback for edge cases
    }
}

impl Default for ArticulatoryCosts {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ArticulatoryCosts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArticulatoryCosts(weight={:.2}, threshold={:.2}, base={})",
            self.articulation_weight, self.free_substitution_threshold, self.base
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_default_costs() {
        let costs = ArticulatoryCosts::default();
        assert!((costs.articulation_weight - 0.6).abs() < EPSILON);
        assert!((costs.free_substitution_threshold - 0.15).abs() < EPSILON);
        assert!(costs.is_valid());
    }

    #[test]
    fn test_with_weight() {
        let costs = ArticulatoryCosts::with_weight(0.8);
        assert!((costs.articulation_weight - 0.8).abs() < EPSILON);
        assert!(costs.is_valid());
    }

    #[test]
    fn test_with_base() {
        let base = OperationCostsF64::typo_friendly();
        let costs = ArticulatoryCosts::with_base(base);
        assert!((costs.base.transposition - 0.5).abs() < EPSILON);
        assert!(costs.is_valid());
    }

    #[test]
    fn test_custom() {
        let base = OperationCostsF64::standard();
        let costs = ArticulatoryCosts::custom(base, 0.9, 0.2);
        assert!((costs.articulation_weight - 0.9).abs() < EPSILON);
        assert!((costs.free_substitution_threshold - 0.2).abs() < EPSILON);
        assert!(costs.is_valid());
    }

    #[test]
    #[should_panic(expected = "Articulation weight must be in [0.0, 1.0]")]
    fn test_invalid_weight_panics() {
        ArticulatoryCosts::with_weight(1.5);
    }

    #[test]
    #[should_panic(expected = "Free substitution threshold must be non-negative")]
    fn test_negative_threshold_panics() {
        let base = OperationCostsF64::standard();
        ArticulatoryCosts::custom(base, 0.5, -0.1);
    }

    #[test]
    fn test_same_character_free() {
        let costs = ArticulatoryCosts::default();
        assert!((costs.substitution_cost('a', 'a') - 0.0).abs() < EPSILON);
        assert!((costs.substitution_cost('z', 'z') - 0.0).abs() < EPSILON);
    }

    #[cfg(feature = "phonetic-rules")]
    #[test]
    fn test_voicing_pairs_cheap() {
        let costs = ArticulatoryCosts::default();

        // Voicing pairs should be cheaper than unrelated sounds
        let pb = costs.substitution_cost('p', 'b');
        let ph = costs.substitution_cost('p', 'h');
        assert!(pb < ph, "p→b ({}) should be cheaper than p→h ({})", pb, ph);

        let td = costs.substitution_cost('t', 'd');
        let th = costs.substitution_cost('t', 'h');
        assert!(td < th, "t→d ({}) should be cheaper than t→h ({})", td, th);
    }

    #[cfg(feature = "phonetic-rules")]
    #[test]
    fn test_free_substitution() {
        let costs = ArticulatoryCosts::default();

        // Same character is always free
        assert!(costs.is_free_substitution('a', 'a'));
        assert!(costs.is_free_substitution('p', 'p'));

        // Distant sounds are not free
        assert!(!costs.is_free_substitution('a', 'z'));
        assert!(!costs.is_free_substitution('p', 'h'));
    }

    #[test]
    fn test_other_costs() {
        let costs = ArticulatoryCosts::default();
        assert!((costs.insertion_cost() - 1.0).abs() < EPSILON);
        assert!((costs.deletion_cost() - 1.0).abs() < EPSILON);
        assert!((costs.transposition_cost() - 1.0).abs() < EPSILON);
        assert!((costs.split_cost() - 1.0).abs() < EPSILON);
        assert!((costs.merge_cost() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_min_nonzero_cost() {
        let costs = ArticulatoryCosts::default();
        let min = costs.min_nonzero_cost();
        assert!(min > 0.0);
        assert!(min < 1.0); // Should be the near-zero free substitution minimum
    }

    #[test]
    fn test_display() {
        let costs = ArticulatoryCosts::default();
        let s = format!("{}", costs);
        assert!(s.contains("weight=0.60"));
        assert!(s.contains("threshold=0.15"));
    }
}
