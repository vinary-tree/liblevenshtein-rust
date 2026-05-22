//! Expression / pattern parsing.
//!
//! Implements alternation, concatenation, quantification, primary
//! expressions, character classes (including standalone and nested
//! `[:NAME:]` named classes and feature bundles).

use crate::phonetic::common::utils::negate_char_class;

use super::super::ast::Expression;
use super::super::error::{LLevError, LLevErrorKind, LLevResult};
use super::super::lexer::Token;
use super::Parser;

impl<'a> Parser<'a> {
    /// Parse a single expression (for testing or embedded use).
    pub fn parse_expression(&mut self) -> LLevResult<Expression> {
        self.lexer.enter_pattern();
        self.parse_alternation()
    }

    // ==================== Expression Parsing ====================

    /// Parse alternation: `concat ("|" concat)*`
    pub(super) fn parse_alternation(&mut self) -> LLevResult<Expression> {
        let mut left = self.parse_concatenation()?;

        while self.check(&Token::Pipe) {
            self.advance()?;
            let right = self.parse_concatenation()?;
            left = Expression::Alt(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse concatenation: `quantified*`
    pub(super) fn parse_concatenation(&mut self) -> LLevResult<Expression> {
        let mut terms = Vec::new();

        while self.can_start_primary() {
            let term = self.parse_quantified()?;
            terms.push(term);
        }

        match terms.len() {
            0 => Ok(Expression::Empty),
            1 => Ok(terms.remove(0)),
            _ => {
                // Build left-associative concatenation
                let mut result = terms.remove(0);
                for term in terms {
                    result = Expression::Concat(Box::new(result), Box::new(term));
                }
                Ok(result)
            }
        }
    }

    /// Check if current token can start a primary expression.
    fn can_start_primary(&mut self) -> bool {
        matches!(
            self.lexer.peek().ok(),
            Some(
                Token::Char(_)
                    | Token::CharClassStart
                    | Token::GroupStart
                    | Token::ScopedFlagsStart(_)
                    | Token::Dot
                    | Token::Hash
                    | Token::PhoneticShortcut { .. }
                    | Token::Identifier(_)
                    | Token::String(_)
                    | Token::SymbolRef(_)
            )
        )
    }

    /// Parse quantified: `primary quantifier?`
    fn parse_quantified(&mut self) -> LLevResult<Expression> {
        let base = self.parse_primary()?;

        if self.check(&Token::Star) {
            self.advance()?;
            Ok(Expression::Star(Box::new(base)))
        } else if self.check(&Token::Plus) {
            self.advance()?;
            Ok(Expression::Plus(Box::new(base)))
        } else if self.check(&Token::Question) {
            self.advance()?;
            Ok(Expression::Optional(Box::new(base)))
        } else if self.check(&Token::BraceStart) {
            self.parse_counted_quantifier(base)
        } else {
            Ok(base)
        }
    }

    /// Parse counted quantifier `{n}` or `{n,m}`.
    fn parse_counted_quantifier(&mut self, base: Expression) -> LLevResult<Expression> {
        self.advance()?; // consume '{'

        let min = self.expect_number()?;

        let max = if self.check(&Token::Comma) {
            self.advance()?;
            if self.check(&Token::BraceEnd) {
                None // {n,} means n or more
            } else {
                Some(self.expect_number()?)
            }
        } else {
            Some(min) // {n} means exactly n
        };

        self.expect(&Token::BraceEnd)?;

        // Use RepeatExact for {n}, RepeatRange for {n,m} or {n,}
        if max == Some(min) {
            Ok(Expression::RepeatExact(Box::new(base), min))
        } else {
            Ok(Expression::RepeatRange {
                inner: Box::new(base),
                min,
                max,
            })
        }
    }

    /// Parse primary: char, char_class, group, any, symbol_ref, boundary.
    fn parse_primary(&mut self) -> LLevResult<Expression> {
        let token = self.advance()?;

        match token {
            Token::Char(c) => Ok(Expression::Char(c)),

            Token::String(s) => {
                // String becomes concatenation of chars
                if s.is_empty() {
                    Ok(Expression::Empty)
                } else {
                    let mut chars = s.chars();
                    let first = Expression::Char(chars.next().expect("non-empty"));
                    let result = chars.fold(first, |acc, c| {
                        Expression::Concat(Box::new(acc), Box::new(Expression::Char(c)))
                    });
                    Ok(result)
                }
            }

            Token::CharClassStart => self.parse_char_class(),

            Token::GroupStart => {
                let expr = self.parse_alternation()?;
                self.expect(&Token::GroupEnd)?;
                Ok(expr)
            }

            Token::ScopedFlagsStart(flags) => {
                // Parse scoped flags group: (?c:...) or (?-i:...)
                let inner = self.parse_alternation()?;
                self.expect(&Token::GroupEnd)?;
                Ok(Expression::ScopedFlags {
                    flags,
                    inner: Box::new(inner),
                })
            }

            Token::Dot => Ok(Expression::Any),

            Token::Hash => Ok(Expression::WordBoundary),

            Token::SymbolRef(name) => {
                // Check if it's a defined symbol
                if let Some(expr) = self.symbols.get(&name) {
                    Ok(expr.clone())
                } else {
                    // Error on undefined symbol with suggestion
                    let available: Vec<&str> = self.symbols.keys().map(|s| s.as_str()).collect();
                    Err(LLevError::undefined_symbol_with_suggestion(
                        &name,
                        &available,
                        self.lexer.position(),
                    ))
                }
            }

            Token::PhoneticShortcut {
                class_name,
                negated,
            } => {
                // Expand phonetic shortcut to character class
                // e.g., \v expands to [:vowel:], \V expands to [^:vowel:]
                use crate::phonetic::named_classes::get_chars_only;

                if let Some(chars) = get_chars_only(&class_name) {
                    let final_chars = if negated {
                        negate_char_class(&chars)
                    } else {
                        chars
                    };
                    Ok(Expression::CharClass {
                        chars: final_chars,
                        negated: false, // negation already applied to chars
                    })
                } else {
                    // Should not happen - lexer only emits valid class names
                    Err(LLevError::new(LLevErrorKind::InvalidPattern(format!(
                        "unknown phonetic class '{}'",
                        class_name
                    )))
                    .at_position(self.lexer.position()))
                }
            }

            other => Err(LLevError::new(LLevErrorKind::ExpectedToken {
                expected: "primary expression".to_string(),
                found: format!("{:?}", other),
            })
            .at_position(self.lexer.position())),
        }
    }

    /// Parse character class `[...]` or standalone named class `[:NAME:]`.
    fn parse_char_class(&mut self) -> LLevResult<Expression> {
        self.lexer.enter_char_class();

        // Check for standalone named class syntax: [:NAME:]
        // This is a shorthand for [[:NAME:]] when no additional chars are needed
        if self.check(&Token::Colon) {
            return self.parse_standalone_named_class();
        }

        // Check for negation
        let negated = if self.check(&Token::Caret) {
            self.advance()?;
            true
        } else {
            false
        };

        let mut chars = Vec::new();

        loop {
            let token = self.advance()?;

            match token {
                Token::CharClassEnd => break,

                // Nested character class: [[...]]
                // Token::CharClassStart inside a char class signals nesting
                Token::CharClassStart => {
                    let nested_chars = self.parse_nested_char_class_content()?;
                    chars.extend(nested_chars);
                }

                Token::Char(c) => {
                    // Check for range
                    if self.check(&Token::Dash) {
                        // Check for range
                        self.advance()?;
                        if self.check(&Token::CharClassEnd) {
                            // '-' at end is literal
                            chars.push(c);
                            chars.push('-');
                        } else if let Token::Char(end) = self.advance()? {
                            // Range c-end
                            for ch in c..=end {
                                chars.push(ch);
                            }
                        } else {
                            return Err(LLevError::new(LLevErrorKind::InvalidPattern(format!(
                                "invalid character range starting with '{}'",
                                c
                            )))
                            .at_position(self.lexer.position()));
                        }
                    } else {
                        chars.push(c);
                    }
                }

                Token::Dash => {
                    // '-' at start is literal
                    chars.push('-');
                }

                Token::Colon => {
                    // ':' inside character class is a literal character.
                    // Named class syntax requires brackets: [[:name:]] not [:name:]
                    chars.push(':');
                }

                Token::SymbolRef(name) => {
                    // User-defined symbol inside character class: [$FOO]
                    if let Some(expr) = self.symbols.get(&name) {
                        match expr {
                            Expression::CharClass {
                                chars: ref class_chars,
                                ..
                            } => {
                                chars.extend(class_chars.iter().cloned());
                            }
                            _ => {
                                return Err(LLevError::new(LLevErrorKind::TypeMismatch {
                                    expected: "character class".to_string(),
                                    found: format!("symbol '{}' is not a character class", name),
                                })
                                .at_position(self.lexer.position()));
                            }
                        }
                    } else {
                        // Undefined symbol
                        let defined: Vec<&str> = self.symbols.keys().map(|s| s.as_str()).collect();
                        return Err(LLevError::undefined_symbol_with_suggestion(
                            &name,
                            &defined,
                            self.lexer.position(),
                        ));
                    }
                }

                Token::PhoneticShortcut {
                    class_name,
                    negated,
                } => {
                    // Phonetic shortcut inside character class: [\v] expands vowels, [\V] expands non-vowels
                    use crate::phonetic::named_classes::get_chars_only;

                    if let Some(class_chars) = get_chars_only(&class_name) {
                        let final_chars = if negated {
                            negate_char_class(&class_chars)
                        } else {
                            class_chars
                        };
                        chars.extend(final_chars);
                    } else {
                        return Err(LLevError::new(LLevErrorKind::InvalidPattern(format!(
                            "unknown phonetic class '{}'",
                            class_name
                        )))
                        .at_position(self.lexer.position()));
                    }
                }

                Token::Eof => {
                    return Err(LLevError::new(LLevErrorKind::UnclosedCharClass)
                        .at_position(self.lexer.position()));
                }

                other => {
                    return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                        expected: "character or ']'".to_string(),
                        found: format!("{:?}", other),
                    })
                    .at_position(self.lexer.position()));
                }
            }
        }

        self.lexer.exit_char_class();

        Ok(Expression::CharClass { chars, negated })
    }

    /// Parse standalone named class `[:NAME:]` or fallback to literal chars.
    ///
    /// Called when we've entered char class mode and see a leading colon.
    /// If the pattern is `[:NAME:]`, parses as named class.
    /// If the pattern is `[:chars]` (no closing `:`), treats as literal characters.
    /// Parse a standalone named class or feature bundle: `[:name:]` or `[:feature1 feature2:]`
    ///
    /// Feature bundles allow intersection of phonetic features:
    /// - `[:voiced stop:]` → voiced AND stop = b, d, g
    /// - `[:!nasal stop:]` → NOT nasal AND stop = p, t, k, b, d, g
    /// - `[:high front vowel:]` → high AND front AND vowel
    fn parse_standalone_named_class(&mut self) -> LLevResult<Expression> {
        // Consume the leading ':'
        self.advance()?; // Colon

        // Collect feature terms: Vec<(name, negated)>
        let mut terms: Vec<(String, bool)> = Vec::new();
        let mut current_name = String::new();
        let mut current_negated = false;

        loop {
            match self.advance()? {
                Token::Char(c) if c.is_alphanumeric() || c == '_' => {
                    current_name.push(c);
                }
                Token::Char(' ') | Token::Char('\t') => {
                    // Space separates terms in feature bundles
                    if !current_name.is_empty() {
                        terms.push((current_name.clone(), current_negated));
                        current_name.clear();
                        current_negated = false;
                    }
                    // Skip additional whitespace
                }
                Token::Char('!') => {
                    // Negation prefix for feature term
                    if !current_name.is_empty() {
                        // '!' after name chars - push previous term first
                        terms.push((current_name.clone(), current_negated));
                        current_name.clear();
                    }
                    current_negated = true;
                }
                Token::Colon => {
                    // Found closing colon - push final term if any
                    if !current_name.is_empty() {
                        terms.push((current_name.clone(), current_negated));
                    }
                    break;
                }
                Token::CharClassEnd => {
                    // No closing colon found - treat ':' and collected chars as literals
                    self.lexer.exit_char_class();
                    let mut chars = vec![':'];
                    for (name, neg) in &terms {
                        if *neg {
                            chars.push('!');
                        }
                        chars.extend(name.chars());
                        chars.push(' ');
                    }
                    if current_negated {
                        chars.push('!');
                    }
                    chars.extend(current_name.chars());
                    return Ok(Expression::CharClass {
                        chars,
                        negated: false,
                    });
                }
                Token::Dash => {
                    // Dash in the name area - treat as literals
                    // Need to collect remaining chars until ]
                    let mut chars = vec![':'];
                    for (name, neg) in &terms {
                        if *neg {
                            chars.push('!');
                        }
                        chars.extend(name.chars());
                        chars.push(' ');
                    }
                    if current_negated {
                        chars.push('!');
                    }
                    chars.extend(current_name.chars());
                    chars.push('-');
                    loop {
                        match self.advance()? {
                            Token::CharClassEnd => {
                                self.lexer.exit_char_class();
                                return Ok(Expression::CharClass {
                                    chars,
                                    negated: false,
                                });
                            }
                            Token::Char(c) => chars.push(c),
                            Token::Colon => chars.push(':'),
                            Token::Dash => chars.push('-'),
                            Token::Eof => {
                                return Err(LLevError::new(LLevErrorKind::UnclosedCharClass)
                                    .at_position(self.lexer.position()));
                            }
                            _ => {} // Ignore other tokens in fallback mode
                        }
                    }
                }
                other => {
                    return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                        expected: "identifier, '!', ':', or ']'".to_string(),
                        found: format!("{:?}", other),
                    })
                    .at_position(self.lexer.position()));
                }
            }
        }

        // Expect closing ']'
        match self.advance()? {
            Token::CharClassEnd => {}
            other => {
                return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                    expected: "']' to close named class".to_string(),
                    found: format!("{:?}", other),
                })
                .at_position(self.lexer.position()));
            }
        }

        self.lexer.exit_char_class();

        if terms.is_empty() {
            return Err(LLevError::new(LLevErrorKind::InvalidPattern(
                "empty named character class '[:]'".to_string(),
            ))
            .at_position(self.lexer.position()));
        }

        // Resolve feature bundle
        self.resolve_feature_bundle(&terms)
    }

    /// Resolve a feature bundle to a character class by computing the intersection.
    ///
    /// Each term is (name, negated). The result is the intersection of all terms,
    /// where negated terms are first complemented relative to all phonetic characters.
    fn resolve_feature_bundle(&self, terms: &[(String, bool)]) -> LLevResult<Expression> {
        use crate::phonetic::named_classes::{
            get_chars_only, intersect_char_sets, negate_char_set,
        };

        let mut char_sets: Vec<Vec<char>> = Vec::new();

        for (name, negated) in terms {
            // Try user symbol first, then built-in
            let chars = if let Some(expr) = self.symbols.get(name) {
                match expr {
                    Expression::CharClass { chars, negated: _ } => chars.clone(),
                    _ => {
                        return Err(LLevError::new(LLevErrorKind::TypeMismatch {
                            expected: "character class".to_string(),
                            found: format!("symbol '{}' is not a character class", name),
                        })
                        .at_position(self.lexer.position()));
                    }
                }
            } else if let Some(builtin_chars) = get_chars_only(name) {
                builtin_chars
            } else {
                // Not found - provide suggestions
                let mut defined: Vec<&str> = self.symbols.keys().map(|s| s.as_str()).collect();
                defined.extend(crate::phonetic::named_classes::all_builtin_class_names());
                return Err(LLevError::undefined_symbol_with_suggestion(
                    name,
                    &defined,
                    self.lexer.position(),
                ));
            };

            let final_chars = if *negated {
                negate_char_set(&chars)
            } else {
                chars
            };

            char_sets.push(final_chars);
        }

        let result_chars = intersect_char_sets(&char_sets);

        Ok(Expression::CharClass {
            chars: result_chars,
            negated: false,
        })
    }

    /// Parse nested character class content after '[' has been consumed.
    ///
    /// Handles:
    /// - `[[:NAME:]]` - nested named class
    /// - `[[:feature1 feature2:]]` - nested feature bundle (intersection)
    /// - `[[^:NAME:]]` - negated nested named class
    /// - `[[abc]]` - arbitrary characters inside nested class
    /// - `[[a-z]]` - ranges inside nested class
    /// - `[[$SYMBOL]]` - symbol references inside nested class
    /// - Arbitrary nesting depth with union semantics
    ///
    /// Returns the characters to add to the parent class.
    fn parse_nested_char_class_content(&mut self) -> LLevResult<Vec<char>> {
        let start_pos = self.lexer.position();

        // Check for negation: [^...]
        let is_negated = self.check(&Token::Caret);
        if is_negated {
            self.advance()?; // consume '^'
        }

        let mut nested_chars = Vec::new();

        // Check if first token is ':' - this indicates [:name:] or [:feature bundle:] pattern
        // which is only valid at the START of nested content
        if self.check(&Token::Colon) {
            self.advance()?; // consume ':'

            // Parse feature terms (supports both single name and feature bundles)
            let mut terms: Vec<(String, bool)> = Vec::new();
            let mut current_name = String::new();
            let mut current_negated = false;

            loop {
                match self.advance()? {
                    Token::Char(nc) if nc.is_alphanumeric() || nc == '_' => {
                        current_name.push(nc);
                    }
                    Token::Char(' ') | Token::Char('\t') => {
                        // Space separates terms in feature bundles
                        if !current_name.is_empty() {
                            terms.push((current_name.clone(), current_negated));
                            current_name.clear();
                            current_negated = false;
                        }
                    }
                    Token::Char('!') => {
                        // Negation prefix for feature term
                        if !current_name.is_empty() {
                            terms.push((current_name.clone(), current_negated));
                            current_name.clear();
                        }
                        current_negated = true;
                    }
                    Token::Colon => {
                        // End of feature bundle - push final term if any
                        if !current_name.is_empty() {
                            terms.push((current_name.clone(), current_negated));
                        }
                        break;
                    }
                    other => {
                        return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                            expected: "identifier, '!', or ':'".to_string(),
                            found: format!("{:?}", other),
                        })
                        .at_position(self.lexer.position()));
                    }
                }
            }

            // Expect closing ']' for the named class
            match self.advance()? {
                Token::CharClassEnd => {
                    self.lexer.enter_char_class();
                }
                other => {
                    return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                        expected: "']' to close named class".to_string(),
                        found: format!("{:?}", other),
                    })
                    .at_position(self.lexer.position()));
                }
            }

            if terms.is_empty() {
                return Err(LLevError::new(LLevErrorKind::InvalidPattern(
                    "empty named character class '[:]'".to_string(),
                ))
                .at_position(self.lexer.position()));
            }

            // Resolve feature bundle
            let expr = self.resolve_feature_bundle(&terms)?;
            nested_chars = match expr {
                Expression::CharClass { chars, negated: _ } => chars,
                _ => Vec::new(),
            };

            // Apply outer negation if needed and return early
            return if is_negated {
                Ok(negate_char_class(&nested_chars))
            } else {
                Ok(nested_chars)
            };
        }

        // Parse arbitrary content until we hit ']'
        loop {
            let token = self.advance()?;

            match token {
                Token::CharClassEnd => {
                    // End of nested class - re-enter char class mode for outer class
                    self.lexer.enter_char_class();
                    break;
                }

                Token::Colon => {
                    // ':' inside nested class is a literal character (not at start)
                    nested_chars.push(':');
                }

                // Nested character class: [[...]]
                // Token::CharClassStart signals further nesting
                Token::CharClassStart => {
                    let inner = self.parse_nested_char_class_content()?;
                    nested_chars.extend(inner);
                }

                Token::Char(c) => {
                    // Token::Char('[') is a literal '[' from escape \[
                    // Token::CharClassStart handles nested classes
                    if self.check(&Token::Dash) {
                        // Check for range
                        self.advance()?;
                        if self.check(&Token::CharClassEnd) {
                            // '-' at end is literal
                            nested_chars.push(c);
                            nested_chars.push('-');
                        } else if let Token::Char(end) = self.advance()? {
                            // Range c-end
                            for ch in c..=end {
                                nested_chars.push(ch);
                            }
                        } else {
                            return Err(LLevError::new(LLevErrorKind::InvalidPattern(format!(
                                "invalid character range starting with '{}'",
                                c
                            )))
                            .at_position(self.lexer.position()));
                        }
                    } else {
                        nested_chars.push(c);
                    }
                }

                Token::Dash => {
                    // '-' at start is literal
                    nested_chars.push('-');
                }

                Token::SymbolRef(name) => {
                    // Symbol reference $SYMBOL
                    if let Some(expr) = self.symbols.get(&name) {
                        match expr {
                            Expression::CharClass {
                                chars: ref class_chars,
                                ..
                            } => {
                                nested_chars.extend(class_chars.iter().cloned());
                            }
                            _ => {
                                return Err(LLevError::new(LLevErrorKind::TypeMismatch {
                                    expected: "character class".to_string(),
                                    found: format!("symbol '{}' is not a character class", name),
                                })
                                .at_position(self.lexer.position()));
                            }
                        }
                    } else {
                        let defined: Vec<&str> = self.symbols.keys().map(|s| s.as_str()).collect();
                        return Err(LLevError::undefined_symbol_with_suggestion(
                            &name,
                            &defined,
                            self.lexer.position(),
                        ));
                    }
                }

                Token::PhoneticShortcut {
                    class_name,
                    negated,
                } => {
                    // Phonetic shortcut \v, \V, \c, \C, etc.
                    use crate::phonetic::named_classes::get_chars_only;
                    if let Some(class_chars) = get_chars_only(&class_name) {
                        let final_chars = if negated {
                            negate_char_class(&class_chars)
                        } else {
                            class_chars
                        };
                        nested_chars.extend(final_chars);
                    } else {
                        return Err(LLevError::new(LLevErrorKind::InvalidPattern(format!(
                            "unknown phonetic class '{}'",
                            class_name
                        )))
                        .at_position(self.lexer.position()));
                    }
                }

                Token::Eof => {
                    return Err(LLevError::new(LLevErrorKind::UnclosedCharClass)
                        .at_position(self.lexer.position()));
                }

                other => {
                    return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                        expected: "character or ']'".to_string(),
                        found: format!("{:?}", other),
                    })
                    .at_position(self.lexer.position()));
                }
            }
        }

        // Check for empty nested class
        if nested_chars.is_empty() && !is_negated {
            return Err(LLevError::new(LLevErrorKind::InvalidPattern(
                "empty nested character class".to_string(),
            ))
            .at_position(start_pos));
        }

        // Apply negation if needed
        if is_negated {
            Ok(negate_char_class(&nested_chars))
        } else {
            Ok(nested_chars)
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Parse a single expression from a string.
pub fn parse_expression(input: &str) -> LLevResult<Expression> {
    let mut parser = Parser::new_pattern(input);
    parser.parse_expression()
}
