//! Context and syllable predicate parsing.
//!
//! Implements the sub-grammar for the optional `/ left _ right if SYL`
//! clause of a rewrite rule. Has its own precedence levels for context
//! (OR < AND < NOT) and for syllable predicates (OR < AND < NOT).

use super::super::ast::{ContextExpr, SyllableCondition, SyllableExpr};
use super::super::error::{LLevError, LLevErrorKind, LLevResult};
use super::super::lexer::Token;
use super::Parser;

impl<'a> Parser<'a> {
    /// Parse context specification `/ left? _ right? syllable_clause?`.
    ///
    /// Returns (left_context, right_context, syllable_condition).
    pub(super) fn parse_context_with_syllable(
        &mut self,
    ) -> LLevResult<(
        Option<ContextExpr>,
        Option<ContextExpr>,
        Option<SyllableExpr>,
    )> {
        let mut left = None;
        let mut right = None;

        // Parse left context (before '_')
        if !self.check(&Token::Underscore) {
            let expr = self.parse_context_or()?;
            left = Some(expr);
        }

        // Expect '_'
        self.expect(&Token::Underscore)?;

        // Parse right context (after '_')
        if !self.check_context_end()
            && !self.check(&Token::KeywordIf)
            && !self.next_token_is_weight_suffix()
        {
            let expr = self.parse_context_or()?;
            right = Some(expr);
        }

        // Parse optional syllable clause
        // Switch to TopLevel mode for syllable keywords (identifiers)
        let syllable = if self.check(&Token::KeywordIf) {
            self.advance()?;
            self.lexer.enter_top_level();
            let result = self.parse_syllable_or()?;
            let _ = self.lexer.remaining_input();
            self.lexer.enter_pattern(); // Switch back for any following parsing
            Some(result)
        } else {
            None
        };

        Ok((left, right, syllable))
    }

    // ==================== Context Expression Parsing ====================
    // Precedence (lowest to highest): OR < AND < NOT

    /// Parse context OR expression: `context_and ("|" context_and)*`
    fn parse_context_or(&mut self) -> LLevResult<ContextExpr> {
        let mut left = self.parse_context_and()?;

        while self.check(&Token::Pipe) {
            self.advance()?;
            let right = self.parse_context_and()?;
            left = ContextExpr::Or(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse context AND expression: `context_not ("&" context_not)*`
    fn parse_context_and(&mut self) -> LLevResult<ContextExpr> {
        let mut left = self.parse_context_not()?;

        while self.check(&Token::Ampersand) {
            self.advance()?;
            let right = self.parse_context_not()?;
            left = ContextExpr::And(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse context NOT expression: `"!" context_not | context_primary`
    fn parse_context_not(&mut self) -> LLevResult<ContextExpr> {
        if self.check(&Token::Bang) {
            self.advance()?;
            let inner = self.parse_context_not()?;
            Ok(ContextExpr::Not(Box::new(inner)))
        } else {
            self.parse_context_primary()
        }
    }

    /// Parse context primary: `expression | "(" context_expr ")" | "#"`
    fn parse_context_primary(&mut self) -> LLevResult<ContextExpr> {
        // Check for word boundary
        if self.check(&Token::Hash) {
            self.advance()?;
            return Ok(ContextExpr::WordBoundary);
        }

        // Check for grouped expression
        if self.check(&Token::GroupStart) {
            self.advance()?;
            let expr = self.parse_context_or()?;
            self.expect(&Token::GroupEnd)?;
            return Ok(expr);
        }

        // Otherwise parse as a pattern expression (concatenation)
        let expr = self.parse_concatenation_before_weight_suffix()?;
        Ok(ContextExpr::Pattern(expr))
    }

    // ==================== Syllable Expression Parsing ====================
    // Precedence (lowest to highest): OR < AND < NOT

    /// Parse syllable OR expression: `syllable_and ("|" syllable_and)*`
    fn parse_syllable_or(&mut self) -> LLevResult<SyllableExpr> {
        let mut left = self.parse_syllable_and()?;

        while self.check(&Token::Pipe) {
            self.advance()?;
            let right = self.parse_syllable_and()?;
            left = SyllableExpr::Or(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse syllable AND expression: `syllable_not ("&" syllable_not)*`
    fn parse_syllable_and(&mut self) -> LLevResult<SyllableExpr> {
        let mut left = self.parse_syllable_not()?;

        while self.check(&Token::Ampersand) {
            self.advance()?;
            let right = self.parse_syllable_not()?;
            left = SyllableExpr::And(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse syllable NOT expression: `"!" syllable_not | syllable_primary`
    fn parse_syllable_not(&mut self) -> LLevResult<SyllableExpr> {
        if self.check(&Token::Bang) {
            self.advance()?;
            let inner = self.parse_syllable_not()?;
            Ok(SyllableExpr::Not(Box::new(inner)))
        } else {
            self.parse_syllable_primary()
        }
    }

    /// Parse syllable primary: `syllable_keyword | "(" syllable_expr ")"`
    fn parse_syllable_primary(&mut self) -> LLevResult<SyllableExpr> {
        // Check for grouped expression
        if self.check(&Token::GroupStart) {
            self.advance()?;
            let expr = self.parse_syllable_or()?;
            self.expect(&Token::GroupEnd)?;
            return Ok(expr);
        }

        // Parse syllable keyword
        let cond = self.parse_syllable_keyword()?;
        Ok(SyllableExpr::Cond(cond))
    }

    /// Parse a syllable keyword like "monosyllable", "polysyllable", etc.
    fn parse_syllable_keyword(&mut self) -> LLevResult<SyllableCondition> {
        // We expect an identifier that matches a syllable keyword
        let token = self.advance()?;
        match &token {
            Token::Identifier(name) => match name.as_str() {
                "monosyllable" => Ok(SyllableCondition::Monosyllable),
                "polysyllable" => Ok(SyllableCondition::Polysyllable),
                "open_syllable" => Ok(SyllableCondition::OpenSyllable),
                "closed_syllable" => Ok(SyllableCondition::ClosedSyllable),
                "final_syllable" => Ok(SyllableCondition::FinalSyllable),
                "initial_syllable" => Ok(SyllableCondition::InitialSyllable),
                _ => Err(LLevError::with_position(
                    LLevErrorKind::ExpectedToken {
                        expected: "syllable keyword (monosyllable, polysyllable, open_syllable, closed_syllable, final_syllable, initial_syllable)".into(),
                        found: format!("identifier '{}'", name),
                    },
                    self.lexer.position(),
                )),
            },
            _ => Err(LLevError::with_position(
                LLevErrorKind::ExpectedToken {
                    expected: "syllable keyword".into(),
                    found: format!("{:?}", token),
                },
                self.lexer.position(),
            )),
        }
    }

    /// Check if we're at the end of a context (before syllable clause).
    fn check_context_end(&mut self) -> bool {
        // Use raw lookahead for safety in case we're in a mixed state
        let remaining = self.lexer.remaining_input();
        remaining.is_empty() || remaining.starts_with(';') || remaining.starts_with('\n')
        // Note: '[' is NOT an end marker - it can start a char class in the context
        // Note: 'if' is handled separately for syllable clause
    }
}
