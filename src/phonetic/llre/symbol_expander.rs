//! Pattern symbol expansion for LLRE files.
//!
//! This module expands pattern symbol references (`GroupRef` nodes) in a regex AST
//! by inlining the referenced pattern from the symbol table.
//!
//! # Example
//!
//! Given a symbol table with:
//! ```text
//! DIGIT = [0-9]
//! NUMBER = DIGIT+
//! ```
//!
//! And a pattern `(?&NUMBER)\.(?&NUMBER)`, this module expands it to:
//! ```text
//! (?:[0-9]+)\.(?:[0-9]+)
//! ```
//!
//! # Cycle Detection
//!
//! Cyclic pattern references are detected and reported as errors:
//! ```text
//! A = (?&B)
//! B = (?&A)  // Error: cyclic reference
//! ```

use std::collections::HashSet;

use crate::phonetic::regex::ast::Regex;

use super::ast::SymbolTable;
use super::error::{LLreError, LLreErrorKind, LLreResult};

/// Maximum recursion depth for pattern expansion to prevent infinite loops.
const MAX_EXPANSION_DEPTH: usize = 100;

/// Expand all pattern symbol references in a regex AST.
///
/// This replaces `Regex::GroupRef(name)` nodes with the pattern definition
/// from the symbol table, with cycle detection to prevent infinite recursion.
///
/// # Arguments
///
/// * `regex` - The regex AST to expand
/// * `symbol_table` - Symbol table containing pattern definitions
///
/// # Returns
///
/// The expanded regex AST with all pattern symbols replaced by their definitions.
///
/// # Errors
///
/// Returns an error if:
/// - A cyclic pattern reference is detected
/// - The expansion depth exceeds `MAX_EXPANSION_DEPTH`
pub fn expand_pattern_symbols(regex: &Regex, symbol_table: &SymbolTable) -> LLreResult<Regex> {
    let mut visited = HashSet::new();
    expand_recursive(regex, symbol_table, &mut visited, 0)
}

fn expand_recursive(
    regex: &Regex,
    symbol_table: &SymbolTable,
    visited: &mut HashSet<String>,
    depth: usize,
) -> LLreResult<Regex> {
    // Check depth limit
    if depth > MAX_EXPANSION_DEPTH {
        return Err(LLreError::new(LLreErrorKind::RecursionDepthExceeded {
            depth,
            max: MAX_EXPANSION_DEPTH,
        }));
    }

    match regex {
        // Key case: expand GroupRef to the pattern definition
        Regex::GroupRef(name) => {
            // Check if this is a pattern symbol in the symbol table
            if let Some(pattern) = symbol_table.patterns.get(name) {
                // Check for cycles
                if visited.contains(name) {
                    return Err(LLreError::new(LLreErrorKind::CyclicPatternReference {
                        name: name.clone(),
                        chain: visited.iter().cloned().collect(),
                    }));
                }

                // Mark as visited for cycle detection
                visited.insert(name.clone());

                // Recursively expand the referenced pattern
                let expanded = expand_recursive(pattern, symbol_table, visited, depth + 1)?;

                // Remove from visited (allows reuse in non-cyclic contexts)
                visited.remove(name);

                // Wrap in non-capturing group for correct precedence
                // e.g., if PATTERN = a|b, then (?&PATTERN)+ should be (?:a|b)+, not a|b+
                Ok(Regex::NonCapturingGroup(Box::new(expanded)))
            } else {
                // Not a pattern symbol - preserve as-is
                // This allows named group references within the same pattern to work
                Ok(regex.clone())
            }
        }

        // Recursive cases: traverse AST structure
        Regex::Concat(a, b) => {
            let a_exp = expand_recursive(a, symbol_table, visited, depth)?;
            let b_exp = expand_recursive(b, symbol_table, visited, depth)?;
            Ok(Regex::Concat(Box::new(a_exp), Box::new(b_exp)))
        }

        Regex::Alt(a, b) => {
            let a_exp = expand_recursive(a, symbol_table, visited, depth)?;
            let b_exp = expand_recursive(b, symbol_table, visited, depth)?;
            Ok(Regex::Alt(Box::new(a_exp), Box::new(b_exp)))
        }

        Regex::Star(inner) => {
            let inner_exp = expand_recursive(inner, symbol_table, visited, depth)?;
            Ok(Regex::Star(Box::new(inner_exp)))
        }

        Regex::Plus(inner) => {
            let inner_exp = expand_recursive(inner, symbol_table, visited, depth)?;
            Ok(Regex::Plus(Box::new(inner_exp)))
        }

        Regex::Optional(inner) => {
            let inner_exp = expand_recursive(inner, symbol_table, visited, depth)?;
            Ok(Regex::Optional(Box::new(inner_exp)))
        }

        Regex::RepeatExact(inner, n) => {
            let inner_exp = expand_recursive(inner, symbol_table, visited, depth)?;
            Ok(Regex::RepeatExact(Box::new(inner_exp), *n))
        }

        Regex::RepeatRange(inner, min, max) => {
            let inner_exp = expand_recursive(inner, symbol_table, visited, depth)?;
            Ok(Regex::RepeatRange(Box::new(inner_exp), *min, *max))
        }

        Regex::CapturingGroup(num, inner) => {
            let inner_exp = expand_recursive(inner, symbol_table, visited, depth)?;
            Ok(Regex::CapturingGroup(*num, Box::new(inner_exp)))
        }

        Regex::NonCapturingGroup(inner) => {
            let inner_exp = expand_recursive(inner, symbol_table, visited, depth)?;
            Ok(Regex::NonCapturingGroup(Box::new(inner_exp)))
        }

        Regex::NamedGroup(name, inner) => {
            let inner_exp = expand_recursive(inner, symbol_table, visited, depth)?;
            Ok(Regex::NamedGroup(name.clone(), Box::new(inner_exp)))
        }

        Regex::FlagsGroup { flags, inner } => {
            let inner_exp = inner
                .as_ref()
                .map(|i| expand_recursive(i, symbol_table, visited, depth))
                .transpose()?;
            Ok(Regex::FlagsGroup {
                flags: flags.clone(),
                inner: inner_exp.map(Box::new),
            })
        }

        Regex::RewriteRule {
            pattern,
            replacement,
            context,
            weight,
        } => {
            let pattern_exp = expand_recursive(pattern, symbol_table, visited, depth)?;
            let replacement_exp = expand_recursive(replacement, symbol_table, visited, depth)?;
            // Context predicates may also contain pattern references
            let context_exp = context
                .as_ref()
                .map(|ctx| expand_context_predicate(ctx, symbol_table, visited, depth))
                .transpose()?;
            Ok(Regex::RewriteRule {
                pattern: Box::new(pattern_exp),
                replacement: Box::new(replacement_exp),
                context: context_exp.map(Box::new),
                weight: *weight,
            })
        }

        // Base cases: no expansion needed (terminals)
        Regex::Empty
        | Regex::Char(_)
        | Regex::CharClass(_)
        | Regex::Any
        | Regex::WordBoundary
        | Regex::StartOfLine
        | Regex::EndOfLine
        | Regex::StartOfInput
        | Regex::EndOfInput
        | Regex::EndOfInputStrict => Ok(regex.clone()),
    }
}

/// Expand pattern symbols in a context predicate.
fn expand_context_predicate(
    ctx: &crate::phonetic::regex::ast::ContextPredicate,
    symbol_table: &SymbolTable,
    visited: &mut HashSet<String>,
    depth: usize,
) -> LLreResult<crate::phonetic::regex::ast::ContextPredicate> {
    use crate::phonetic::regex::ast::ContextPredicate;

    Ok(ContextPredicate {
        left: ctx
            .left
            .as_ref()
            .map(|e| expand_context_expr(e, symbol_table, visited, depth))
            .transpose()?,
        right: ctx
            .right
            .as_ref()
            .map(|e| expand_context_expr(e, symbol_table, visited, depth))
            .transpose()?,
        syllable: ctx.syllable.clone(),
    })
}

/// Expand pattern symbols in a context expression.
fn expand_context_expr(
    expr: &crate::phonetic::regex::ast::ContextExpr,
    symbol_table: &SymbolTable,
    visited: &mut HashSet<String>,
    depth: usize,
) -> LLreResult<crate::phonetic::regex::ast::ContextExpr> {
    use crate::phonetic::regex::ast::ContextExpr;

    match expr {
        ContextExpr::Pattern(regex) => {
            let expanded = expand_recursive(regex, symbol_table, visited, depth)?;
            Ok(ContextExpr::Pattern(expanded))
        }
        ContextExpr::WordBoundary => Ok(ContextExpr::WordBoundary),
        ContextExpr::And(a, b) => {
            let a_exp = expand_context_expr(a, symbol_table, visited, depth)?;
            let b_exp = expand_context_expr(b, symbol_table, visited, depth)?;
            Ok(ContextExpr::And(Box::new(a_exp), Box::new(b_exp)))
        }
        ContextExpr::Or(a, b) => {
            let a_exp = expand_context_expr(a, symbol_table, visited, depth)?;
            let b_exp = expand_context_expr(b, symbol_table, visited, depth)?;
            Ok(ContextExpr::Or(Box::new(a_exp), Box::new(b_exp)))
        }
        ContextExpr::Not(inner) => {
            let inner_exp = expand_context_expr(inner, symbol_table, visited, depth)?;
            Ok(ContextExpr::Not(Box::new(inner_exp)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::regex::parse;

    #[test]
    fn test_no_expansion_needed() {
        let table = SymbolTable::new();
        let regex = parse("[a-z]+").unwrap();
        let expanded = expand_pattern_symbols(&regex, &table).unwrap();
        assert_eq!(format!("{}", expanded), format!("{}", regex));
    }

    #[test]
    fn test_simple_pattern_expansion() {
        let mut table = SymbolTable::new();
        let digit_pattern = parse("[0-9]").unwrap();
        table.add_pattern("DIGIT", digit_pattern, None);

        // Parse pattern that references DIGIT
        // Note: GroupRef syntax is (?&name)
        let regex = Regex::GroupRef("DIGIT".to_string());
        let expanded = expand_pattern_symbols(&regex, &table).unwrap();

        // Should be wrapped in non-capturing group
        if let Regex::NonCapturingGroup(inner) = &expanded {
            if let Regex::CharClass(_) = inner.as_ref() {
                // Success
            } else {
                panic!("Expected CharClass inside NonCapturingGroup, got {:?}", inner);
            }
        } else {
            panic!("Expected NonCapturingGroup, got {:?}", expanded);
        }
    }

    #[test]
    fn test_nested_pattern_expansion() {
        let mut table = SymbolTable::new();

        // DIGIT = [0-9]
        let digit_pattern = parse("[0-9]").unwrap();
        table.add_pattern("DIGIT", digit_pattern, None);

        // NUMBER = DIGIT+ (using GroupRef)
        let number_pattern = Regex::Plus(Box::new(Regex::GroupRef("DIGIT".to_string())));
        table.add_pattern("NUMBER", number_pattern, None);

        // Pattern: (?&NUMBER)
        let regex = Regex::GroupRef("NUMBER".to_string());
        let expanded = expand_pattern_symbols(&regex, &table).unwrap();

        // Should contain digit characters (the Display impl may expand [0-9] to [0123456789])
        // The inner DIGIT should be expanded to a digit character class wrapped in non-capturing group
        let formatted = format!("{}", expanded);
        // Check for either the compact [0-9] form or the expanded [0123456789] form
        assert!(
            formatted.contains("[0-9]") || formatted.contains("[0123456789]"),
            "Expected expanded pattern to contain digit class, got: {}",
            formatted
        );
    }

    #[test]
    fn test_cycle_detection_direct() {
        let mut table = SymbolTable::new();

        // A = (?&A) - direct self-reference
        let a_pattern = Regex::GroupRef("A".to_string());
        table.add_pattern("A", a_pattern, None);

        let regex = Regex::GroupRef("A".to_string());
        let result = expand_pattern_symbols(&regex, &table);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind,
            LLreErrorKind::CyclicPatternReference { .. }
        ));
    }

    #[test]
    fn test_cycle_detection_indirect() {
        let mut table = SymbolTable::new();

        // A = (?&B)
        let a_pattern = Regex::GroupRef("B".to_string());
        table.add_pattern("A", a_pattern, None);

        // B = (?&A) - indirect cycle through A
        let b_pattern = Regex::GroupRef("A".to_string());
        table.add_pattern("B", b_pattern, None);

        let regex = Regex::GroupRef("A".to_string());
        let result = expand_pattern_symbols(&regex, &table);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind,
            LLreErrorKind::CyclicPatternReference { .. }
        ));
    }

    #[test]
    fn test_undefined_pattern_preserved() {
        let table = SymbolTable::new();

        // Reference to undefined pattern should be preserved
        // (might be a named group reference within the same pattern)
        let regex = Regex::GroupRef("undefined".to_string());
        let expanded = expand_pattern_symbols(&regex, &table).unwrap();

        // Should remain as GroupRef
        assert!(matches!(expanded, Regex::GroupRef(_)));
    }

    #[test]
    fn test_pattern_reuse_non_cyclic() {
        let mut table = SymbolTable::new();

        // DIGIT = [0-9]
        let digit_pattern = parse("[0-9]").unwrap();
        table.add_pattern("DIGIT", digit_pattern, None);

        // Pattern uses DIGIT twice: (?&DIGIT)\.(?&DIGIT)
        // This is NOT a cycle - it's reuse
        let regex = Regex::Concat(
            Box::new(Regex::GroupRef("DIGIT".to_string())),
            Box::new(Regex::Concat(
                Box::new(Regex::Char('.')),
                Box::new(Regex::GroupRef("DIGIT".to_string())),
            )),
        );

        let result = expand_pattern_symbols(&regex, &table);
        assert!(result.is_ok(), "Pattern reuse should not trigger cycle detection");
    }
}
