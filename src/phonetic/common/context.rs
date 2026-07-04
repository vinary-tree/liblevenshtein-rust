//! Generic context expression types for pattern matching.

use std::fmt;

/// A context expression that matches patterns or word boundaries.
///
/// This type is generic over the pattern type `P`, allowing it to be used
/// with different AST types (e.g., `Regex` or `Expression`).
///
/// # Type Parameters
///
/// * `P` - The pattern type used in `Pattern` variant (e.g., `Regex`, `Expression`, `RegexByte`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextExpr<P> {
    /// Simple pattern-based context
    Pattern(P),
    /// Word boundary marker (`#`)
    WordBoundary,
    /// Both contexts must match (AND)
    And(Box<ContextExpr<P>>, Box<ContextExpr<P>>),
    /// Either context must match (OR)
    Or(Box<ContextExpr<P>>, Box<ContextExpr<P>>),
    /// Context must NOT match
    Not(Box<ContextExpr<P>>),
}

impl<P> ContextExpr<P> {
    /// Create a pattern-based context expression.
    pub fn pattern(pattern: P) -> Self {
        ContextExpr::Pattern(pattern)
    }

    /// Create a word boundary context expression.
    pub fn word_boundary() -> Self {
        ContextExpr::WordBoundary
    }

    /// Create an AND context expression.
    pub fn and(left: ContextExpr<P>, right: ContextExpr<P>) -> Self {
        ContextExpr::And(Box::new(left), Box::new(right))
    }

    /// Create an OR context expression.
    pub fn or(left: ContextExpr<P>, right: ContextExpr<P>) -> Self {
        ContextExpr::Or(Box::new(left), Box::new(right))
    }

    /// Create a NOT context expression.
    pub fn negate(inner: ContextExpr<P>) -> Self {
        ContextExpr::Not(Box::new(inner))
    }
}

impl<P> std::ops::Not for ContextExpr<P> {
    type Output = Self;

    fn not(self) -> Self::Output {
        ContextExpr::Not(Box::new(self))
    }
}

impl<P: fmt::Display> fmt::Display for ContextExpr<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextExpr::Pattern(p) => write!(f, "{}", p),
            ContextExpr::WordBoundary => write!(f, "#"),
            ContextExpr::And(left, right) => write!(f, "({} & {})", left, right),
            ContextExpr::Or(left, right) => write!(f, "({} | {})", left, right),
            ContextExpr::Not(inner) => write!(f, "!{}", inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test pattern type for testing
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestPattern(String);

    impl fmt::Display for TestPattern {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[test]
    fn test_context_expr_pattern() {
        let expr: ContextExpr<TestPattern> = ContextExpr::pattern(TestPattern("abc".to_string()));
        assert_eq!(expr.to_string(), "abc");
    }

    #[test]
    fn test_context_expr_word_boundary() {
        let expr: ContextExpr<TestPattern> = ContextExpr::word_boundary();
        assert_eq!(expr.to_string(), "#");
    }

    #[test]
    fn test_context_expr_and() {
        let left = ContextExpr::pattern(TestPattern("a".to_string()));
        let right = ContextExpr::pattern(TestPattern("b".to_string()));
        let expr = ContextExpr::and(left, right);
        assert_eq!(expr.to_string(), "(a & b)");
    }

    #[test]
    fn test_context_expr_or() {
        let left = ContextExpr::pattern(TestPattern("a".to_string()));
        let right = ContextExpr::pattern(TestPattern("b".to_string()));
        let expr = ContextExpr::or(left, right);
        assert_eq!(expr.to_string(), "(a | b)");
    }

    #[test]
    fn test_context_expr_not() {
        let inner = ContextExpr::pattern(TestPattern("a".to_string()));
        let expr = ContextExpr::negate(inner);
        assert_eq!(expr.to_string(), "!a");
    }

    #[test]
    fn test_context_expr_complex() {
        // (a | #) & !b
        let a = ContextExpr::pattern(TestPattern("a".to_string()));
        let boundary: ContextExpr<TestPattern> = ContextExpr::word_boundary();
        let left = ContextExpr::or(a, boundary);
        let b = ContextExpr::pattern(TestPattern("b".to_string()));
        let right = ContextExpr::negate(b);
        let expr = ContextExpr::and(left, right);
        assert_eq!(expr.to_string(), "((a | #) & !b)");
    }

    #[test]
    fn test_context_expr_equality() {
        let expr1: ContextExpr<TestPattern> = ContextExpr::word_boundary();
        let expr2: ContextExpr<TestPattern> = ContextExpr::word_boundary();
        assert_eq!(expr1, expr2);

        let expr3 = ContextExpr::pattern(TestPattern("a".to_string()));
        let expr4 = ContextExpr::pattern(TestPattern("a".to_string()));
        assert_eq!(expr3, expr4);
    }
}
