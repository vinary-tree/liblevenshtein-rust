//! Syllable condition types for context matching.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Syllable condition for context matching.
///
/// These conditions are evaluated at the position where a pattern matches
/// to determine if the context constraint is satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyllableCondition {
    /// Word has exactly one syllable (e.g., "fly", "ply")
    Monosyllable,
    /// Word has more than one syllable (e.g., "happy", "flying")
    Polysyllable,
    /// Current syllable ends in a vowel (vowel is long)
    OpenSyllable,
    /// Current syllable ends in a consonant (vowel is short)
    ClosedSyllable,
    /// Match position is in the final syllable
    FinalSyllable,
    /// Match position is in the initial syllable
    InitialSyllable,
}

impl SyllableCondition {
    /// Parse a syllable condition from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "monosyllable" => Some(SyllableCondition::Monosyllable),
            "polysyllable" => Some(SyllableCondition::Polysyllable),
            "open_syllable" => Some(SyllableCondition::OpenSyllable),
            "closed_syllable" => Some(SyllableCondition::ClosedSyllable),
            "final_syllable" => Some(SyllableCondition::FinalSyllable),
            "initial_syllable" => Some(SyllableCondition::InitialSyllable),
            _ => None,
        }
    }

    /// Get the string representation of this condition.
    pub fn as_str(&self) -> &'static str {
        match self {
            SyllableCondition::Monosyllable => "monosyllable",
            SyllableCondition::Polysyllable => "polysyllable",
            SyllableCondition::OpenSyllable => "open_syllable",
            SyllableCondition::ClosedSyllable => "closed_syllable",
            SyllableCondition::FinalSyllable => "final_syllable",
            SyllableCondition::InitialSyllable => "initial_syllable",
        }
    }
}

impl fmt::Display for SyllableCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyllableCondition::Monosyllable => write!(f, "monosyllable"),
            SyllableCondition::Polysyllable => write!(f, "polysyllable"),
            SyllableCondition::OpenSyllable => write!(f, "open_syllable"),
            SyllableCondition::ClosedSyllable => write!(f, "closed_syllable"),
            SyllableCondition::FinalSyllable => write!(f, "final_syllable"),
            SyllableCondition::InitialSyllable => write!(f, "initial_syllable"),
        }
    }
}

/// A syllable expression with logical operators.
///
/// Allows combining syllable conditions with AND, OR, NOT.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyllableExpr {
    /// Simple syllable condition
    Cond(SyllableCondition),
    /// Both expressions must be true
    And(Box<SyllableExpr>, Box<SyllableExpr>),
    /// Either expression must be true
    Or(Box<SyllableExpr>, Box<SyllableExpr>),
    /// Expression must be false
    Not(Box<SyllableExpr>),
}

impl SyllableExpr {
    /// Create a simple condition expression.
    pub fn cond(cond: SyllableCondition) -> Self {
        SyllableExpr::Cond(cond)
    }

    /// Create an AND expression.
    pub fn and(left: SyllableExpr, right: SyllableExpr) -> Self {
        SyllableExpr::And(Box::new(left), Box::new(right))
    }

    /// Create an OR expression.
    pub fn or(left: SyllableExpr, right: SyllableExpr) -> Self {
        SyllableExpr::Or(Box::new(left), Box::new(right))
    }

    /// Create a NOT expression.
    pub fn not(inner: SyllableExpr) -> Self {
        SyllableExpr::Not(Box::new(inner))
    }

    /// Get the estimated size/complexity of this syllable expression.
    pub fn size(&self) -> usize {
        match self {
            SyllableExpr::Cond(_) => 1,
            SyllableExpr::And(a, b) | SyllableExpr::Or(a, b) => 1 + a.size() + b.size(),
            SyllableExpr::Not(inner) => 1 + inner.size(),
        }
    }

    /// Check if this is a simple condition (no compound operators).
    pub fn is_simple(&self) -> bool {
        matches!(self, SyllableExpr::Cond(_))
    }

    /// Get the inner condition if this is simple.
    pub fn as_condition(&self) -> Option<SyllableCondition> {
        match self {
            SyllableExpr::Cond(c) => Some(*c),
            _ => None,
        }
    }
}

impl fmt::Display for SyllableExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyllableExpr::Cond(cond) => write!(f, "{}", cond),
            SyllableExpr::And(left, right) => write!(f, "({} & {})", left, right),
            SyllableExpr::Or(left, right) => write!(f, "({} | {})", left, right),
            SyllableExpr::Not(inner) => write!(f, "!{}", inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syllable_condition_from_str() {
        assert_eq!(
            SyllableCondition::from_str("monosyllable"),
            Some(SyllableCondition::Monosyllable)
        );
        assert_eq!(
            SyllableCondition::from_str("polysyllable"),
            Some(SyllableCondition::Polysyllable)
        );
        assert_eq!(
            SyllableCondition::from_str("open_syllable"),
            Some(SyllableCondition::OpenSyllable)
        );
        assert_eq!(
            SyllableCondition::from_str("closed_syllable"),
            Some(SyllableCondition::ClosedSyllable)
        );
        assert_eq!(
            SyllableCondition::from_str("final_syllable"),
            Some(SyllableCondition::FinalSyllable)
        );
        assert_eq!(
            SyllableCondition::from_str("initial_syllable"),
            Some(SyllableCondition::InitialSyllable)
        );
        assert_eq!(SyllableCondition::from_str("invalid"), None);
    }

    #[test]
    fn test_syllable_condition_display() {
        assert_eq!(SyllableCondition::Monosyllable.to_string(), "monosyllable");
        assert_eq!(SyllableCondition::Polysyllable.to_string(), "polysyllable");
        assert_eq!(SyllableCondition::OpenSyllable.to_string(), "open_syllable");
        assert_eq!(
            SyllableCondition::ClosedSyllable.to_string(),
            "closed_syllable"
        );
        assert_eq!(
            SyllableCondition::FinalSyllable.to_string(),
            "final_syllable"
        );
        assert_eq!(
            SyllableCondition::InitialSyllable.to_string(),
            "initial_syllable"
        );
    }

    #[test]
    fn test_syllable_expr_cond() {
        let expr = SyllableExpr::cond(SyllableCondition::Monosyllable);
        assert_eq!(expr.to_string(), "monosyllable");
    }

    #[test]
    fn test_syllable_expr_and() {
        let left = SyllableExpr::cond(SyllableCondition::Monosyllable);
        let right = SyllableExpr::cond(SyllableCondition::OpenSyllable);
        let expr = SyllableExpr::and(left, right);
        assert_eq!(expr.to_string(), "(monosyllable & open_syllable)");
    }

    #[test]
    fn test_syllable_expr_or() {
        let left = SyllableExpr::cond(SyllableCondition::Monosyllable);
        let right = SyllableExpr::cond(SyllableCondition::Polysyllable);
        let expr = SyllableExpr::or(left, right);
        assert_eq!(expr.to_string(), "(monosyllable | polysyllable)");
    }

    #[test]
    fn test_syllable_expr_not() {
        let inner = SyllableExpr::cond(SyllableCondition::FinalSyllable);
        let expr = SyllableExpr::not(inner);
        assert_eq!(expr.to_string(), "!final_syllable");
    }

    #[test]
    fn test_syllable_expr_complex() {
        // (monosyllable | !final_syllable) & open_syllable
        let mono = SyllableExpr::cond(SyllableCondition::Monosyllable);
        let not_final = SyllableExpr::not(SyllableExpr::cond(SyllableCondition::FinalSyllable));
        let left = SyllableExpr::or(mono, not_final);
        let right = SyllableExpr::cond(SyllableCondition::OpenSyllable);
        let expr = SyllableExpr::and(left, right);
        assert_eq!(
            expr.to_string(),
            "((monosyllable | !final_syllable) & open_syllable)"
        );
    }

    #[test]
    fn test_syllable_condition_as_str() {
        assert_eq!(SyllableCondition::Monosyllable.as_str(), "monosyllable");
        assert_eq!(SyllableCondition::Polysyllable.as_str(), "polysyllable");
        assert_eq!(SyllableCondition::OpenSyllable.as_str(), "open_syllable");
        assert_eq!(SyllableCondition::ClosedSyllable.as_str(), "closed_syllable");
        assert_eq!(SyllableCondition::FinalSyllable.as_str(), "final_syllable");
        assert_eq!(SyllableCondition::InitialSyllable.as_str(), "initial_syllable");
    }

    #[test]
    fn test_syllable_expr_is_simple() {
        let simple = SyllableExpr::cond(SyllableCondition::Monosyllable);
        assert!(simple.is_simple());

        let compound = SyllableExpr::and(
            SyllableExpr::cond(SyllableCondition::Monosyllable),
            SyllableExpr::cond(SyllableCondition::OpenSyllable),
        );
        assert!(!compound.is_simple());
    }

    #[test]
    fn test_syllable_expr_as_condition() {
        let simple = SyllableExpr::cond(SyllableCondition::Monosyllable);
        assert_eq!(simple.as_condition(), Some(SyllableCondition::Monosyllable));

        let compound = SyllableExpr::and(
            SyllableExpr::cond(SyllableCondition::Monosyllable),
            SyllableExpr::cond(SyllableCondition::OpenSyllable),
        );
        assert_eq!(compound.as_condition(), None);
    }
}
