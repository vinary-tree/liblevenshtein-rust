//! Configurable operation costs for float-weighted Levenshtein automata.
//!
//! This module provides `OperationCostsF64`, a configuration structure for
//! assigning custom float costs to each edit operation type.
//!
//! # Overview
//!
//! While the standard Levenshtein automaton uses integer costs (all operations
//! cost 1), many applications benefit from weighted operations:
//!
//! - **Phonetic matching**: Give lower costs to phonetically similar substitutions
//! - **OCR correction**: Common OCR confusions (O↔0, l↔I) get lower costs
//! - **Keyboard proximity**: Adjacent key typos (q↔w) are cheaper than distant ones
//! - **Domain-specific**: Custom weighting for specific use cases
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::transducer::OperationCostsF64;
//!
//! // Standard costs (all 1.0)
//! let standard = OperationCostsF64::standard();
//! assert_eq!(standard.substitution, 1.0);
//!
//! // Custom costs for typo correction
//! let typo_costs = OperationCostsF64 {
//!     match_cost: 0.0,
//!     substitution: 1.5,      // Penalize substitutions more
//!     insertion: 1.0,
//!     deletion: 1.0,
//!     transposition: 0.5,     // Transpositions are common typos
//!     split: 2.0,
//!     merge: 2.0,
//! };
//! ```
//!
//! # Theoretical Constraints
//!
//! From TCS 2011 Theorem 8.2 (Bounded Diagonal Property):
//!
//! - **Match cost must be 0**: Match operations (same character) are free
//! - **All costs must be non-negative**: Negative costs break the metric properties
//!
//! # Difference from MSM
//!
//! This module provides **operation-level** weighting where costs are fixed per
//! operation type. For **transition-level** weighting where costs depend on the
//! actual values (like MSM's `|x_i - y_j|`), see the `time_series` module.

use std::fmt;

/// Float-weighted operation costs for Levenshtein automata.
///
/// Each field specifies the cost for one type of edit operation.
/// All costs must be non-negative, and match cost must be zero.
///
/// # Fields
///
/// | Field | Operation | Description |
/// |-------|-----------|-------------|
/// | `match_cost` | Match | Same character, always 0.0 |
/// | `substitution` | Substitution | Replace one char with another |
/// | `insertion` | Insertion | Insert a char into query |
/// | `deletion` | Deletion | Delete a char from dictionary |
/// | `transposition` | Transposition | Swap adjacent characters |
/// | `split` | Split | One dict char → two query chars |
/// | `merge` | Merge | Two query chars → one dict char |
///
/// # Standard Levenshtein
///
/// For standard Levenshtein distance, use `OperationCostsF64::standard()`:
/// - substitution = insertion = deletion = 1.0
/// - transposition = split = merge = 1.0
/// - match_cost = 0.0
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperationCostsF64 {
    /// Cost for matching characters (must be 0.0 for metric property).
    pub match_cost: f64,

    /// Cost for substituting one character for another.
    pub substitution: f64,

    /// Cost for inserting a character (character in query not in dictionary).
    pub insertion: f64,

    /// Cost for deleting a character (character in dictionary not in query).
    pub deletion: f64,

    /// Cost for transposing adjacent characters (Damerau-Levenshtein).
    pub transposition: f64,

    /// Cost for splitting one dictionary character into two query characters.
    pub split: f64,

    /// Cost for merging two query characters into one dictionary character.
    pub merge: f64,
}

impl OperationCostsF64 {
    /// Create costs for standard Levenshtein distance (all costs = 1.0).
    ///
    /// This is equivalent to the classic Levenshtein distance where
    /// all edit operations have unit cost.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::OperationCostsF64;
    ///
    /// let costs = OperationCostsF64::standard();
    /// assert_eq!(costs.substitution, 1.0);
    /// assert_eq!(costs.insertion, 1.0);
    /// assert_eq!(costs.deletion, 1.0);
    /// ```
    #[inline]
    pub const fn standard() -> Self {
        Self {
            match_cost: 0.0,
            substitution: 1.0,
            insertion: 1.0,
            deletion: 1.0,
            transposition: 1.0,
            split: 1.0,
            merge: 1.0,
        }
    }

    /// Create costs optimized for typo correction.
    ///
    /// Transpositions are cheaper (common typing error like "teh" → "the"),
    /// while substitutions are more expensive.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::OperationCostsF64;
    ///
    /// let costs = OperationCostsF64::typo_friendly();
    /// assert!(costs.transposition < costs.substitution);
    /// ```
    #[inline]
    pub const fn typo_friendly() -> Self {
        Self {
            match_cost: 0.0,
            substitution: 1.2,
            insertion: 1.0,
            deletion: 1.0,
            transposition: 0.5, // Transpositions are common typos
            split: 1.5,
            merge: 1.5,
        }
    }

    /// Create costs optimized for OCR correction.
    ///
    /// All operations have standard cost. Use with a SubstitutionSet
    /// containing OCR-friendly character pairs for best results.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::OperationCostsF64;
    ///
    /// let costs = OperationCostsF64::ocr_friendly();
    /// ```
    #[inline]
    pub const fn ocr_friendly() -> Self {
        Self {
            match_cost: 0.0,
            substitution: 0.8, // OCR errors are common, lower penalty
            insertion: 1.0,
            deletion: 1.0,
            transposition: 1.2, // Less common in OCR
            split: 1.5,
            merge: 1.5,
        }
    }

    /// Create custom costs with all values specified.
    ///
    /// # Arguments
    ///
    /// * `substitution` - Cost for character substitution (≥ 0)
    /// * `insertion` - Cost for character insertion (≥ 0)
    /// * `deletion` - Cost for character deletion (≥ 0)
    /// * `transposition` - Cost for adjacent character swap (≥ 0)
    /// * `split` - Cost for one-to-two character split (≥ 0)
    /// * `merge` - Cost for two-to-one character merge (≥ 0)
    ///
    /// # Panics
    ///
    /// Panics if any cost is negative.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::OperationCostsF64;
    ///
    /// let costs = OperationCostsF64::custom(1.5, 1.0, 1.0, 0.5, 2.0, 2.0);
    /// assert_eq!(costs.substitution, 1.5);
    /// assert_eq!(costs.transposition, 0.5);
    /// ```
    pub fn custom(
        substitution: f64,
        insertion: f64,
        deletion: f64,
        transposition: f64,
        split: f64,
        merge: f64,
    ) -> Self {
        assert!(substitution >= 0.0, "Substitution cost must be non-negative");
        assert!(insertion >= 0.0, "Insertion cost must be non-negative");
        assert!(deletion >= 0.0, "Deletion cost must be non-negative");
        assert!(
            transposition >= 0.0,
            "Transposition cost must be non-negative"
        );
        assert!(split >= 0.0, "Split cost must be non-negative");
        assert!(merge >= 0.0, "Merge cost must be non-negative");

        Self {
            match_cost: 0.0, // Match is always free
            substitution,
            insertion,
            deletion,
            transposition,
            split,
            merge,
        }
    }

    /// Validate that all costs satisfy the metric constraints.
    ///
    /// Returns `true` if:
    /// - Match cost is 0.0
    /// - All other costs are non-negative
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::OperationCostsF64;
    ///
    /// let costs = OperationCostsF64::standard();
    /// assert!(costs.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        self.match_cost == 0.0
            && self.substitution >= 0.0
            && self.insertion >= 0.0
            && self.deletion >= 0.0
            && self.transposition >= 0.0
            && self.split >= 0.0
            && self.merge >= 0.0
    }

    /// Get the minimum non-zero cost among all operations.
    ///
    /// This is useful for computing lower bounds in pruning strategies.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::OperationCostsF64;
    ///
    /// let costs = OperationCostsF64::typo_friendly();
    /// let min = costs.min_nonzero_cost();
    /// assert_eq!(min, 0.5); // transposition cost
    /// ```
    pub fn min_nonzero_cost(&self) -> f64 {
        let costs = [
            self.substitution,
            self.insertion,
            self.deletion,
            self.transposition,
            self.split,
            self.merge,
        ];

        costs
            .iter()
            .copied()
            .filter(|&c| c > 0.0)
            .min_by(|a, b| a.partial_cmp(b).expect("OperationCostsF64: costs are finite (filtered > 0.0)"))
            .unwrap_or(1.0)
    }

    /// Check if these costs are equivalent to standard integer Levenshtein.
    ///
    /// Returns `true` if all non-match costs are exactly 1.0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::transducer::OperationCostsF64;
    ///
    /// assert!(OperationCostsF64::standard().is_standard());
    /// assert!(!OperationCostsF64::typo_friendly().is_standard());
    /// ```
    pub fn is_standard(&self) -> bool {
        const EPSILON: f64 = 1e-10;
        (self.match_cost - 0.0).abs() < EPSILON
            && (self.substitution - 1.0).abs() < EPSILON
            && (self.insertion - 1.0).abs() < EPSILON
            && (self.deletion - 1.0).abs() < EPSILON
            && (self.transposition - 1.0).abs() < EPSILON
            && (self.split - 1.0).abs() < EPSILON
            && (self.merge - 1.0).abs() < EPSILON
    }
}

impl Default for OperationCostsF64 {
    /// Returns standard Levenshtein costs (all operations = 1.0).
    fn default() -> Self {
        Self::standard()
    }
}

impl fmt::Display for OperationCostsF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OperationCosts(sub={:.2}, ins={:.2}, del={:.2}, trans={:.2}, split={:.2}, merge={:.2})",
            self.substitution,
            self.insertion,
            self.deletion,
            self.transposition,
            self.split,
            self.merge
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_standard_costs() {
        let costs = OperationCostsF64::standard();
        assert!((costs.match_cost - 0.0).abs() < EPSILON);
        assert!((costs.substitution - 1.0).abs() < EPSILON);
        assert!((costs.insertion - 1.0).abs() < EPSILON);
        assert!((costs.deletion - 1.0).abs() < EPSILON);
        assert!((costs.transposition - 1.0).abs() < EPSILON);
        assert!((costs.split - 1.0).abs() < EPSILON);
        assert!((costs.merge - 1.0).abs() < EPSILON);
        assert!(costs.is_valid());
        assert!(costs.is_standard());
    }

    #[test]
    fn test_typo_friendly() {
        let costs = OperationCostsF64::typo_friendly();
        assert!(costs.transposition < costs.substitution);
        assert!(costs.is_valid());
        assert!(!costs.is_standard());
    }

    #[test]
    fn test_custom_costs() {
        let costs = OperationCostsF64::custom(1.5, 1.0, 0.8, 0.5, 2.0, 2.0);
        assert!((costs.substitution - 1.5).abs() < EPSILON);
        assert!((costs.deletion - 0.8).abs() < EPSILON);
        assert!((costs.transposition - 0.5).abs() < EPSILON);
        assert!(costs.is_valid());
    }

    #[test]
    #[should_panic(expected = "Substitution cost must be non-negative")]
    fn test_negative_substitution_panics() {
        OperationCostsF64::custom(-0.5, 1.0, 1.0, 1.0, 1.0, 1.0);
    }

    #[test]
    fn test_min_nonzero_cost() {
        let costs = OperationCostsF64::custom(1.5, 1.0, 0.8, 0.3, 2.0, 2.0);
        assert!((costs.min_nonzero_cost() - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_display() {
        let costs = OperationCostsF64::standard();
        let s = format!("{}", costs);
        assert!(s.contains("sub=1.00"));
        assert!(s.contains("ins=1.00"));
    }
}
