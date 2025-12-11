//! Recursive descent parser for `.llev` rule files.
//!
//! This parser transforms a stream of tokens into an AST representing
//! the `.llev` file structure including directives, metadata, and rules.
//!
//! # Grammar Overview
//!
//! ```text
//! llev_file       ::= (directive | rule_definition | NEWLINE)*
//! directive       ::= "@" DIRECTIVE_NAME directive_value
//! rule_definition ::= metadata_block? rewrite_rule TERMINATOR?
//! rewrite_rule    ::= pattern ARROW replacement context? weight_suffix?
//! pattern         ::= expression
//! replacement     ::= expression | EMPTY
//! context         ::= "/" left_context? "_" right_context?
//! expression      ::= alternation
//! alternation     ::= concatenation ("|" concatenation)*
//! concatenation   ::= quantified*
//! quantified      ::= primary quantifier?
//! primary         ::= char | char_class | group | any | symbol_ref | boundary
//! ```

use std::collections::HashMap;

use super::ast::{
    ContextAST, ContextExpr, Expression, IncludeDirective, LLevFile, RewriteRuleAST,
    RuleDefinition, RuleMetadata, SyllableCondition, SyllableExpr, SymbolDef,
};
use super::error::{LLevError, LLevErrorKind, LLevResult, Position};
use super::lexer::{Lexer, Token};

/// Check if a name represents a user-defined symbol (all uppercase).
///
/// User-defined symbols use UPPERCASE names to distinguish them from
/// built-in classes which use lowercase (e.g., `[:alpha:]`, `[:vowel:]`).
///
/// A name is considered a user symbol if:
/// - It starts with an uppercase letter
/// - All alphabetic characters are uppercase
/// - Non-alphabetic characters (digits, underscores) are allowed after the first character
///
/// # Examples
/// - `"VOWEL"` → true (user symbol)
/// - `"FRONT_V"` → true (user symbol)
/// - `"V2"` → true (user symbol with digit)
/// - `"alpha"` → false (built-in)
/// - `"Vowel"` → false (mixed case, treated as built-in)
/// - `"_FOO"` → false (must start with a letter)
/// - `"123"` → false (must start with a letter)
fn is_user_symbol_name(name: &str) -> bool {
    let mut chars = name.chars();
    // Must start with an uppercase letter
    match chars.next() {
        Some(first) if first.is_uppercase() => {
            // Remaining characters: alphabetic must be uppercase, non-alphabetic are allowed
            chars.all(|c| c.is_uppercase() || !c.is_alphabetic())
        }
        _ => false,
    }
}

/// Compute complement of a character class using printable ASCII.
///
/// This is used for negated named classes like `[^[:vowel:]]`.
/// Returns all printable ASCII characters (0x20-0x7E) that are NOT in the input set.
fn negate_char_class(chars: &[char]) -> Vec<char> {
    (0x20u8..=0x7Eu8)
        .map(|b| b as char)
        .filter(|c| !chars.contains(c))
        .collect()
}

/// Parser for `.llev` files.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    /// Symbol table for `@define` symbols
    symbols: HashMap<String, Expression>,
}

impl<'a> Parser<'a> {
    /// Create a new parser for the given input string.
    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Lexer::new_file(input),
            symbols: HashMap::new(),
        }
    }

    /// Create a new parser for pattern expressions only (not full files).
    pub fn new_pattern(input: &'a str) -> Self {
        Self {
            lexer: Lexer::new(input),
            symbols: HashMap::new(),
        }
    }

    /// Parse a complete `.llev` file.
    pub fn parse_file(&mut self) -> LLevResult<LLevFile> {
        let mut file = LLevFile::new();

        loop {
            // Use raw input lookahead to avoid consuming characters in TopLevel mode.
            // This is critical because peeking in TopLevel mode tokenizes alphabetic
            // sequences as identifiers, consuming the characters.
            let remaining = self.lexer.remaining_input();

            // Skip empty content
            if remaining.is_empty() {
                break;
            }

            // Check for directive (starts with '@')
            if remaining.starts_with('@') {
                self.parse_directive(&mut file)?;
            } else if remaining.starts_with('\n') {
                // Skip newline
                self.lexer.enter_top_level(); // Ensure we're in TopLevel mode
                self.advance()?;
            } else {
                // Parse a rule definition (handles metadata blocks internally)
                let rule = self.parse_rule_definition()?;
                file.rules.push(rule);
            }
        }

        Ok(file)
    }

    /// Parse a single expression (for testing or embedded use).
    pub fn parse_expression(&mut self) -> LLevResult<Expression> {
        self.lexer.enter_pattern();
        self.parse_alternation()
    }

    // ==================== Directive Parsing ====================

    /// Check if the remaining input starts with a directive.
    /// Uses raw input lookahead to avoid consuming characters.
    fn is_directive_raw(&mut self) -> bool {
        let remaining = self.lexer.remaining_input();
        remaining.starts_with('@')
    }

    /// Parse a directive and add it to the file.
    fn parse_directive(&mut self, file: &mut LLevFile) -> LLevResult<()> {
        let token = self.advance()?;

        match token {
            Token::DirectiveName => {
                let value = self.expect_string()?;
                file.metadata.name = Some(value);
            }
            Token::DirectiveVersion => {
                let value = self.expect_string()?;
                file.metadata.version = Some(value);
            }
            Token::DirectiveAuthor => {
                let value = self.expect_string()?;
                file.metadata.author = Some(value);
            }
            Token::DirectiveDescription => {
                let value = self.expect_string()?;
                file.metadata.description = Some(value);
            }
            Token::DirectiveInclude => {
                let path = self.expect_string()?;
                let pos = self.lexer.position();
                file.includes.push(IncludeDirective::new(path, pos));
            }
            Token::DirectiveDefine => {
                self.parse_define_directive(file)?;
            }
            _ => {
                return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                    expected: "directive".to_string(),
                    found: format!("{:?}", token),
                })
                .at_position(self.lexer.position()));
            }
        }

        // Skip optional terminator
        self.skip_terminator();

        Ok(())
    }

    /// Parse a `@define` directive.
    fn parse_define_directive(&mut self, file: &mut LLevFile) -> LLevResult<()> {
        // Expect identifier
        let name = self.expect_identifier()?;
        let pos = self.lexer.position();

        // User-defined symbols must be UPPERCASE (built-in classes are lowercase)
        if name.chars().any(|c| c.is_ascii_lowercase()) {
            return Err(LLevError::symbol_name_must_be_uppercase(&name, pos));
        }

        // Expect '='
        self.expect(&Token::Equals)?;

        // Parse the expression
        self.lexer.enter_pattern();
        let expr = self.parse_alternation()?;
        self.lexer.enter_top_level();

        // Add to symbol table
        self.symbols.insert(name.clone(), expr.clone());

        // Add to file
        file.symbols.push(SymbolDef::new(name, expr, pos));

        Ok(())
    }

    // ==================== Rule Parsing ====================

    /// Parse a rule definition with optional metadata.
    fn parse_rule_definition(&mut self) -> LLevResult<RuleDefinition> {
        let pos = self.lexer.position();
        let mut metadata = RuleMetadata::default();

        // Check for metadata block using raw lookahead (no peeking in TopLevel mode!)
        // In TopLevel mode, peeking would tokenize "ph" as Identifier("ph"), consuming chars
        if self.is_metadata_block_start()? {
            // Consume '[' in TopLevel mode
            self.lexer.enter_top_level();
            self.advance()?; // consume '['
            self.lexer.enter_metadata();
            self.parse_metadata_block_contents(&mut metadata)?;
            // After metadata, skip any newlines before the rule
            // We need to be in TopLevel mode to properly handle newlines
            self.lexer.enter_top_level();
            self.skip_whitespace_tokens();
        }

        // Parse the rewrite rule - enter pattern mode
        self.lexer.enter_pattern();
        let rule = self.parse_rewrite_rule()?;
        self.lexer.enter_top_level();

        // Skip optional terminator
        self.skip_terminator();

        Ok(RuleDefinition::new(metadata, rule, pos))
    }

    /// Check if the upcoming '[' starts a metadata block (not a char class pattern).
    /// This uses lookahead to check if the pattern is `[ identifier :`.
    fn is_metadata_block_start(&mut self) -> LLevResult<bool> {
        // We need to check if after '[' we have identifier followed by ':'
        // This is tricky - we'll use the lexer's raw input to look ahead

        // For now, use a simpler heuristic: if we're at TopLevel and see '[',
        // check if the character after '[' could start a metadata key (letter)
        // and if there's a ':' before the next ']' or newline

        // Actually, the safest approach is to look at the raw characters
        // Let's check the next few characters after '['
        let pos = self.lexer.current_offset();
        let remaining = self.lexer.remaining_input();

        if !remaining.starts_with('[') {
            return Ok(false);
        }

        // Look for pattern: [ identifier :
        let after_bracket = &remaining[1..];
        let trimmed = after_bracket.trim_start();

        // Check if it starts with a valid metadata key
        let metadata_keys = ["id:", "name:", "weight:", "group:", "enabled:"];
        for key in metadata_keys {
            if trimmed.starts_with(key) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Parse the contents of a metadata block (after '[' has been consumed).
    fn parse_metadata_block_contents(&mut self, metadata: &mut RuleMetadata) -> LLevResult<()> {
        loop {
            // Parse key
            let key = match self.advance()? {
                Token::Identifier(k) => k,
                Token::MetadataEnd => break,
                other => {
                    return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                        expected: "metadata key or ']'".to_string(),
                        found: format!("{:?}", other),
                    })
                    .at_position(self.lexer.position()));
                }
            };

            // Expect ':'
            self.expect(&Token::Colon)?;

            // Parse value based on key
            match key.as_str() {
                "id" => {
                    let num = self.expect_number()?;
                    metadata.id = Some(num);
                }
                "name" => {
                    let name = self.expect_string()?;
                    metadata.name = Some(name);
                }
                "weight" => {
                    let weight = self.expect_float_or_number()?;
                    metadata.weight = Some(weight);
                }
                "group" => {
                    // Accept both identifier and string: `group: orthography` or `group: "orthography"`
                    let group = self.expect_identifier_or_string()?;
                    metadata.group = Some(group);
                }
                "enabled" => {
                    let enabled = self.expect_bool()?;
                    metadata.enabled = enabled;
                }
                _ => {
                    return Err(LLevError::new(LLevErrorKind::InvalidMetadataKey(key))
                        .at_position(self.lexer.position()));
                }
            }

            // Check for comma or end
            if self.check(&Token::Comma) {
                self.advance()?;
            } else if self.check(&Token::MetadataEnd) {
                self.advance()?;
                break;
            }
        }

        // Exit metadata mode
        self.lexer.exit_metadata();

        Ok(())
    }

    /// Skip whitespace and newline tokens.
    fn skip_whitespace_tokens(&mut self) {
        // Use raw lookahead to avoid consuming characters
        loop {
            let remaining = self.lexer.remaining_input();
            if remaining.starts_with('\n') {
                let _ = self.advance();
            } else {
                break;
            }
        }
    }

    /// Parse a rewrite rule `pattern -> replacement context?`.
    fn parse_rewrite_rule(&mut self) -> LLevResult<RewriteRuleAST> {
        // Parse pattern
        let pattern = self.parse_alternation()?;

        // Expect arrow
        self.expect(&Token::Arrow)?;

        // Parse replacement (can be empty)
        let replacement = if self.check_replacement_end() {
            Expression::Empty
        } else {
            self.parse_alternation()?
        };

        // Parse optional context (may include syllable clause)
        let context = if self.check(&Token::Slash) {
            self.advance()?;
            let (left, right, syllable) = self.parse_context_with_syllable()?;
            Some(ContextAST::new_with_syllable(left, right, syllable))
        } else {
            None
        };

        // Parse optional weight suffix
        let weight = if self.check(&Token::CharClassStart) {
            // Could be weight suffix [0.5] - but need to distinguish from char class
            // Weight suffix would be followed by a number
            // For simplicity, we'll skip weight suffix parsing in patterns
            // and rely on metadata block for weights
            None
        } else {
            None
        };

        Ok(RewriteRuleAST {
            pattern,
            replacement,
            context,
            weight,
        })
    }

    /// Check if we're at the end of a replacement (arrow, slash, semicolon, newline, eof).
    fn check_replacement_end(&mut self) -> bool {
        matches!(
            self.lexer.peek().ok(),
            Some(
                Token::Slash
                    | Token::Semicolon
                    | Token::Newline
                    | Token::Eof
                    | Token::CharClassStart
            )
        )
    }

    /// Parse context specification `/ left? _ right? syllable_clause?`.
    ///
    /// Returns (left_context, right_context, syllable_condition).
    fn parse_context_with_syllable(
        &mut self,
    ) -> LLevResult<(Option<ContextExpr>, Option<ContextExpr>, Option<SyllableExpr>)> {
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
        if !self.check_context_end() && !self.check(&Token::KeywordIf) {
            let expr = self.parse_context_or()?;
            right = Some(expr);
        }

        // Parse optional syllable clause
        // Switch to TopLevel mode for syllable keywords (identifiers)
        let syllable = if self.check(&Token::KeywordIf) {
            self.advance()?;
            self.lexer.enter_top_level();
            let result = self.parse_syllable_or()?;
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
        let expr = self.parse_concatenation()?;
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
        remaining.is_empty()
            || remaining.starts_with(';')
            || remaining.starts_with('\n')
            // Note: '[' is NOT an end marker - it can start a char class in the context
            // Note: 'if' is handled separately for syllable clause
    }

    // ==================== Expression Parsing ====================

    /// Parse alternation: `concat ("|" concat)*`
    fn parse_alternation(&mut self) -> LLevResult<Expression> {
        let mut left = self.parse_concatenation()?;

        while self.check(&Token::Pipe) {
            self.advance()?;
            let right = self.parse_concatenation()?;
            left = Expression::Alt(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse concatenation: `quantified*`
    fn parse_concatenation(&mut self) -> LLevResult<Expression> {
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

            Token::PhoneticShortcut { class_name, negated } => {
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
                            return Err(LLevError::new(LLevErrorKind::InvalidPattern(
                                format!("invalid character range starting with '{}'", c),
                            ))
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
                        let defined: Vec<&str> =
                            self.symbols.keys().map(|s| s.as_str()).collect();
                        return Err(LLevError::undefined_symbol_with_suggestion(
                            &name,
                            &defined,
                            self.lexer.position(),
                        ));
                    }
                }

                Token::PhoneticShortcut { class_name, negated } => {
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
                    return Err(
                        LLevError::new(LLevErrorKind::UnclosedCharClass)
                            .at_position(self.lexer.position()),
                    );
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
    fn parse_standalone_named_class(&mut self) -> LLevResult<Expression> {
        // Consume the leading ':'
        self.advance()?; // Colon

        // Collect chars that could be a name
        let mut name_chars = Vec::new();
        let mut found_closing_colon = false;

        loop {
            match self.advance()? {
                Token::Char(c) if c.is_alphanumeric() || c == '_' => {
                    name_chars.push(c);
                }
                Token::Colon => {
                    // Found closing colon - this is a valid named class
                    found_closing_colon = true;
                    break;
                }
                Token::CharClassEnd => {
                    // No closing colon found - treat ':' and collected chars as literals
                    self.lexer.exit_char_class();
                    let mut chars = vec![':'];
                    chars.extend(name_chars);
                    return Ok(Expression::CharClass {
                        chars,
                        negated: false,
                    });
                }
                Token::Dash => {
                    // Dash in the name area - treat as literals
                    // Need to collect remaining chars until ]
                    let mut chars = vec![':'];
                    chars.extend(name_chars);
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
                                return Err(
                                    LLevError::new(LLevErrorKind::UnclosedCharClass)
                                        .at_position(self.lexer.position()),
                                );
                            }
                            _ => {} // Ignore other tokens in fallback mode
                        }
                    }
                }
                other => {
                    return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                        expected: "identifier, ':', or ']'".to_string(),
                        found: format!("{:?}", other),
                    })
                    .at_position(self.lexer.position()));
                }
            }
        }

        if !found_closing_colon {
            // This shouldn't be reachable due to loop logic above
            return Err(LLevError::new(LLevErrorKind::InvalidPattern(
                "incomplete named class syntax".to_string(),
            ))
            .at_position(self.lexer.position()));
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

        let name: String = name_chars.iter().collect();
        if name.is_empty() {
            return Err(LLevError::new(LLevErrorKind::InvalidPattern(
                "empty named character class '[:]'".to_string(),
            ))
            .at_position(self.lexer.position()));
        }

        // Resolve the named class
        // User-defined symbols have precedence over built-in classes
        if let Some(expr) = self.symbols.get(&name) {
            // User-defined symbol
            match expr {
                Expression::CharClass { chars, negated } => Ok(Expression::CharClass {
                    chars: chars.clone(),
                    negated: *negated,
                }),
                _ => Err(LLevError::new(LLevErrorKind::TypeMismatch {
                    expected: "character class".to_string(),
                    found: format!("symbol '{}' is not a character class", name),
                })
                .at_position(self.lexer.position())),
            }
        } else if let Some(builtin_chars) = crate::phonetic::named_classes::get_chars_only(&name) {
            // Fallback to built-in classes
            Ok(Expression::CharClass {
                chars: builtin_chars,
                negated: false,
            })
        } else {
            // Not found - provide suggestions
            let mut defined: Vec<&str> = self.symbols.keys().map(|s| s.as_str()).collect();
            defined.extend(crate::phonetic::named_classes::all_builtin_class_names());
            Err(LLevError::undefined_symbol_with_suggestion(
                &name,
                &defined,
                self.lexer.position(),
            ))
        }
    }

    /// Parse nested character class content after '[' has been consumed.
    ///
    /// Handles:
    /// - `[[:NAME:]]` - nested named class
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

        // Check if first token is ':' - this indicates [:name:] pattern
        // which is only valid at the START of nested content
        if self.check(&Token::Colon) {
            self.advance()?; // consume ':'
            // Parse the name
            let mut name = String::new();
            loop {
                match self.advance()? {
                    Token::Char(nc) if nc.is_alphanumeric() || nc == '_' => {
                        name.push(nc);
                    }
                    Token::Colon => {
                        // End of name, expect ']' next
                        break;
                    }
                    other => {
                        return Err(LLevError::new(LLevErrorKind::ExpectedToken {
                            expected: "identifier or ':'".to_string(),
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

            if name.is_empty() {
                return Err(LLevError::new(LLevErrorKind::InvalidPattern(
                    "empty named character class '[:]'".to_string(),
                ))
                .at_position(self.lexer.position()));
            }

            // Resolve the named class
            nested_chars = self.resolve_named_class(&name, start_pos)?;

            // Apply negation if needed and return early
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
                            return Err(LLevError::new(LLevErrorKind::InvalidPattern(
                                format!("invalid character range starting with '{}'", c),
                            ))
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
                        let defined: Vec<&str> =
                            self.symbols.keys().map(|s| s.as_str()).collect();
                        return Err(LLevError::undefined_symbol_with_suggestion(
                            &name,
                            &defined,
                            self.lexer.position(),
                        ));
                    }
                }

                Token::PhoneticShortcut { class_name, negated } => {
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
                    return Err(
                        LLevError::new(LLevErrorKind::UnclosedCharClass)
                            .at_position(self.lexer.position()),
                    );
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

    /// Resolve a named class reference to its character set.
    ///
    /// Resolution order (user-defined symbols have precedence):
    /// 1. User-defined symbols (uppercase names like VOWEL, CONSONANT)
    /// 2. Built-in named classes (lowercase names like vowel, consonant)
    fn resolve_named_class(&self, name: &str, pos: Position) -> LLevResult<Vec<char>> {
        // First check user-defined symbols (they have precedence over built-ins)
        if is_user_symbol_name(name) {
            if let Some(expr) = self.symbols.get(name) {
                return match expr {
                    Expression::CharClass {
                        chars: ref class_chars,
                        ..
                    } => Ok(class_chars.clone()),
                    _ => Err(LLevError::new(LLevErrorKind::TypeMismatch {
                        expected: "character class".to_string(),
                        found: format!("symbol '{}' is not a character class", name),
                    })
                    .at_position(pos)),
                };
            }
            // User symbol name but not defined - fall through to check built-ins
        }

        // Then check built-in classes
        if let Some(builtin_chars) = crate::phonetic::named_classes::get_chars_only(name) {
            Ok(builtin_chars)
        } else if is_user_symbol_name(name) {
            // User symbol name but not defined and no matching built-in
            let defined: Vec<&str> = self.symbols.keys().map(|s| s.as_str()).collect();
            Err(LLevError::undefined_symbol_with_suggestion(name, &defined, pos))
        } else {
            // Unknown built-in class
            let builtin_names = crate::phonetic::named_classes::all_builtin_class_names();
            Err(LLevError::new(LLevErrorKind::UnknownNamedClass {
                name: name.to_string(),
                available: builtin_names.into_iter().map(|s| s.to_string()).collect(),
            })
            .at_position(pos))
        }
    }

    // ==================== Helper Methods ====================

    /// Advance and return the next token.
    fn advance(&mut self) -> LLevResult<Token> {
        self.lexer.next_token()
    }

    /// Check if the next token matches (without consuming).
    fn check(&mut self, expected: &Token) -> bool {
        self.lexer.peek().ok() == Some(expected)
    }

    /// Expect and consume a specific token.
    fn expect(&mut self, expected: &Token) -> LLevResult<Token> {
        let token = self.advance()?;
        if &token == expected {
            Ok(token)
        } else {
            Err(LLevError::expected_token(
                format!("{:?}", expected),
                format!("{:?}", token),
                self.lexer.position(),
            ))
        }
    }

    /// Expect and consume a string literal.
    fn expect_string(&mut self) -> LLevResult<String> {
        match self.advance()? {
            Token::String(s) => Ok(s),
            other => Err(LLevError::expected_token(
                "string".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume an identifier.
    fn expect_identifier(&mut self) -> LLevResult<String> {
        match self.advance()? {
            Token::Identifier(s) => Ok(s),
            other => Err(LLevError::expected_token(
                "identifier".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume an identifier or string literal.
    ///
    /// This is useful for metadata fields like `group` where users might
    /// write either `group: orthography` or `group: "orthography"`.
    fn expect_identifier_or_string(&mut self) -> LLevResult<String> {
        match self.advance()? {
            Token::Identifier(s) => Ok(s),
            Token::String(s) => Ok(s),
            other => Err(LLevError::expected_token(
                "identifier or string".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume a number.
    fn expect_number(&mut self) -> LLevResult<usize> {
        match self.advance()? {
            Token::Number(n) => Ok(n),
            other => Err(LLevError::expected_token(
                "number".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume a float or number (as f64).
    fn expect_float_or_number(&mut self) -> LLevResult<f64> {
        match self.advance()? {
            Token::Float(f) => Ok(f),
            Token::Number(n) => Ok(n as f64),
            other => Err(LLevError::expected_token(
                "number or float".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume a boolean (identifier "true" or "false").
    fn expect_bool(&mut self) -> LLevResult<bool> {
        match self.advance()? {
            Token::Identifier(s) if s == "true" => Ok(true),
            Token::Identifier(s) if s == "false" => Ok(false),
            other => Err(LLevError::expected_token(
                "true or false".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Skip optional terminator (semicolon or newline).
    fn skip_terminator(&mut self) {
        // Use raw lookahead to avoid consuming characters in TopLevel mode
        loop {
            let remaining = self.lexer.remaining_input();
            if remaining.starts_with(';') || remaining.starts_with('\n') {
                // We're in TopLevel mode, so advance will correctly handle these
                self.lexer.enter_top_level();
                let _ = self.advance();
            } else {
                break;
            }
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Parse a `.llev` file from a string.
pub fn parse_str(input: &str) -> LLevResult<LLevFile> {
    let mut parser = Parser::new(input);
    parser.parse_file()
}

/// Parse a `.llev` file from a string with pre-defined symbols.
///
/// The symbols map character class names to their character sets.
/// Symbol names should be UPPERCASE.
pub fn parse_str_with_symbols(
    input: &str,
    symbols: &HashMap<String, Vec<char>>,
) -> LLevResult<LLevFile> {
    let mut parser = Parser::new(input);
    // Convert Vec<char> symbols to Expression::CharClass
    for (name, chars) in symbols {
        parser.symbols.insert(
            name.clone(),
            Expression::CharClass {
                chars: chars.clone(),
                negated: false,
            },
        );
    }
    parser.parse_file()
}

/// Parse a single expression from a string.
pub fn parse_expression(input: &str) -> LLevResult<Expression> {
    let mut parser = Parser::new_pattern(input);
    parser.parse_expression()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_expression() {
        let expr = parse_expression("abc").unwrap();
        // Should be Concat(Concat(Char('a'), Char('b')), Char('c'))
        match expr {
            Expression::Concat(left, right) => {
                match (*left, *right) {
                    (Expression::Concat(ll, lr), Expression::Char('c')) => {
                        assert!(matches!(*ll, Expression::Char('a')));
                        assert!(matches!(*lr, Expression::Char('b')));
                    }
                    _ => panic!("unexpected structure"),
                }
            }
            _ => panic!("expected Concat"),
        }
    }

    #[test]
    fn test_parse_alternation() {
        let expr = parse_expression("a|b").unwrap();
        match expr {
            Expression::Alt(left, right) => {
                assert!(matches!(*left, Expression::Char('a')));
                assert!(matches!(*right, Expression::Char('b')));
            }
            _ => panic!("expected Alt"),
        }
    }

    #[test]
    fn test_parse_char_class() {
        let expr = parse_expression("[aeiou]").unwrap();
        match expr {
            Expression::CharClass { chars, negated } => {
                assert!(!negated);
                assert_eq!(chars, vec!['a', 'e', 'i', 'o', 'u']);
            }
            _ => panic!("expected CharClass"),
        }
    }

    #[test]
    fn test_parse_char_class_negated() {
        let expr = parse_expression("[^aeiou]").unwrap();
        match expr {
            Expression::CharClass { chars, negated } => {
                assert!(negated);
                assert_eq!(chars, vec!['a', 'e', 'i', 'o', 'u']);
            }
            _ => panic!("expected negated CharClass"),
        }
    }

    #[test]
    fn test_parse_char_range() {
        let expr = parse_expression("[a-c]").unwrap();
        match expr {
            Expression::CharClass { chars, negated } => {
                assert!(!negated);
                assert_eq!(chars, vec!['a', 'b', 'c']);
            }
            _ => panic!("expected CharClass with range"),
        }
    }

    #[test]
    fn test_parse_quantifiers() {
        let star = parse_expression("a*").unwrap();
        assert!(matches!(star, Expression::Star(_)));

        let plus = parse_expression("a+").unwrap();
        assert!(matches!(plus, Expression::Plus(_)));

        let opt = parse_expression("a?").unwrap();
        assert!(matches!(opt, Expression::Optional(_)));
    }

    #[test]
    fn test_parse_counted_quantifier() {
        let exact = parse_expression("a{3}").unwrap();
        match exact {
            Expression::RepeatExact(_, n) => {
                assert_eq!(n, 3);
            }
            _ => panic!("expected RepeatExact, got {:?}", exact),
        }

        let range = parse_expression("a{2,4}").unwrap();
        match range {
            Expression::RepeatRange { min, max, .. } => {
                assert_eq!(min, 2);
                assert_eq!(max, Some(4));
            }
            _ => panic!("expected RepeatRange, got {:?}", range),
        }

        let min_only = parse_expression("a{2,}").unwrap();
        match min_only {
            Expression::RepeatRange { min, max, .. } => {
                assert_eq!(min, 2);
                assert_eq!(max, None);
            }
            _ => panic!("expected RepeatRange, got {:?}", min_only),
        }
    }

    #[test]
    fn test_parse_group() {
        let expr = parse_expression("(ab)").unwrap();
        match expr {
            Expression::Concat(left, right) => {
                assert!(matches!(*left, Expression::Char('a')));
                assert!(matches!(*right, Expression::Char('b')));
            }
            _ => panic!("expected group contents"),
        }
    }

    #[test]
    fn test_parse_any() {
        let expr = parse_expression(".").unwrap();
        assert!(matches!(expr, Expression::Any));
    }

    #[test]
    fn test_parse_word_boundary() {
        let expr = parse_expression("#").unwrap();
        assert!(matches!(expr, Expression::WordBoundary));
    }

    #[test]
    fn test_parse_symbol_ref() {
        // Test symbol reference with defined symbol - expands to the definition
        let input = r#"
@define VOWEL = [aeiou]
$VOWEL -> x
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        // The symbol ref is expanded to the character class at parse time
        let rule = &file.rules[0].rule;
        match &rule.pattern {
            Expression::CharClass { chars, negated } => {
                assert!(!negated);
                assert_eq!(chars.len(), 5);
                assert!(chars.contains(&'a'));
                assert!(chars.contains(&'e'));
                assert!(chars.contains(&'i'));
                assert!(chars.contains(&'o'));
                assert!(chars.contains(&'u'));
            }
            _ => panic!("expected CharClass, got {:?}", rule.pattern),
        }
    }

    #[test]
    fn test_parse_empty_file() {
        let file = parse_str("").unwrap();
        assert!(file.rules.is_empty());
        assert!(file.symbols.is_empty());
        assert!(file.includes.is_empty());
    }

    #[test]
    fn test_parse_file_metadata() {
        let input = r#"
@name "Test Rules"
@version "1.0"
@author "Test Author"
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.metadata.name, Some("Test Rules".to_string()));
        assert_eq!(file.metadata.version, Some("1.0".to_string()));
        assert_eq!(file.metadata.author, Some("Test Author".to_string()));
    }

    #[test]
    fn test_parse_define_directive() {
        // Use MY_VOWEL to avoid conflict with built-in "vowel" class
        let input = "@define MY_VOWEL = [aeiou]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.symbols.len(), 1);
        assert_eq!(file.symbols[0].name, "MY_VOWEL");
    }

    #[test]
    fn test_parse_include_directive() {
        let input = r#"@include "other.llev""#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.includes.len(), 1);
        assert_eq!(file.includes[0].path, "other.llev");
    }

    #[test]
    fn test_parse_simple_rule() {
        let input = "ph -> f";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        // Pattern should be 'p' concat 'h'
        match &rule.pattern {
            Expression::Concat(left, right) => {
                assert!(matches!(**left, Expression::Char('p')));
                assert!(matches!(**right, Expression::Char('h')));
            }
            _ => panic!("expected Concat for pattern"),
        }
        // Replacement should be 'f'
        assert!(matches!(rule.replacement, Expression::Char('f')));
    }

    #[test]
    fn test_parse_rule_with_context() {
        let input = "c -> s / _[ei]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        assert!(matches!(rule.pattern, Expression::Char('c')));
        assert!(matches!(rule.replacement, Expression::Char('s')));
        assert!(rule.context.is_some());
        let ctx = rule.context.as_ref().expect("should have context");
        assert!(ctx.left.is_none());
        assert!(ctx.right.is_some());
    }

    #[test]
    fn test_parse_deletion_rule() {
        let input = "gh -> ;";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        assert!(matches!(rule.replacement, Expression::Empty));
    }

    #[test]
    fn test_parse_rule_with_metadata() {
        let input = r#"[id: 1, name: "ph to f"]
ph -> f"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let def = &file.rules[0];
        assert_eq!(def.metadata.id, Some(1));
        assert_eq!(def.metadata.name, Some("ph to f".to_string()));
    }

    #[test]
    fn test_parse_multiple_rules() {
        let input = r#"
ph -> f
gh ->
c -> k
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 3);
    }

    #[test]
    fn test_parse_complex_pattern() {
        let expr = parse_expression("(a|b)*c+d?").unwrap();
        // Should parse without error
        assert!(matches!(expr, Expression::Concat(_, _)));
    }

    #[test]
    fn test_parse_metadata_group_as_identifier() {
        let input = r#"[id: 1, name: "test", group: orthography]
ph -> f"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);
        let def = &file.rules[0];
        assert_eq!(def.metadata.group, Some("orthography".to_string()));
    }

    #[test]
    fn test_parse_metadata_group_as_string() {
        // This test verifies that group can be specified as a string literal
        let input = r#"[id: 1, name: "test", group: "orthography"]
ph -> f"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);
        let def = &file.rules[0];
        assert_eq!(def.metadata.group, Some("orthography".to_string()));
    }

    #[test]
    fn test_parse_escaped_uppercase_literal() {
        // Test that \A in a pattern becomes a literal 'A' character
        let input = r#"\A -> a"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        // Pattern should be literal 'A'
        assert!(matches!(rule.pattern, Expression::Char('A')));
        // Replacement should be literal 'a'
        assert!(matches!(rule.replacement, Expression::Char('a')));
    }

    #[test]
    fn test_parse_string_literal_for_uppercase() {
        // Test that "ABC" in a pattern works for literal uppercase strings
        // This is an alternative to using escape sequences
        let input = r#""ABC" -> abc"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        // Pattern should be Concat(Concat(Char('A'), Char('B')), Char('C'))
        match &rule.pattern {
            Expression::Concat(left, right) => {
                assert!(matches!(**right, Expression::Char('C')));
                match &**left {
                    Expression::Concat(ll, lr) => {
                        assert!(matches!(**ll, Expression::Char('A')));
                        assert!(matches!(**lr, Expression::Char('B')));
                    }
                    _ => panic!("expected nested Concat"),
                }
            }
            _ => panic!("expected Concat for pattern"),
        }
    }

    #[test]
    fn test_parse_mixed_escaped_and_regular() {
        // Test mixing escaped uppercase with regular patterns
        let input = r#"\Abc -> abc"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        // Pattern should be Concat(Concat(Char('A'), Char('b')), Char('c'))
        match &rule.pattern {
            Expression::Concat(left, right) => {
                assert!(matches!(**right, Expression::Char('c')));
                match &**left {
                    Expression::Concat(ll, lr) => {
                        assert!(matches!(**ll, Expression::Char('A')));
                        assert!(matches!(**lr, Expression::Char('b')));
                    }
                    _ => panic!("expected nested Concat"),
                }
            }
            _ => panic!("expected Concat for pattern"),
        }
    }

    // ==================== Compound Context Tests ====================

    #[test]
    fn test_parse_context_word_boundary_initial() {
        let input = "wr -> r / #_";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        // Left should be word boundary
        assert!(matches!(
            ctx.left.as_deref(),
            Some(ContextExpr::WordBoundary)
        ));
        assert!(ctx.right.is_none());
    }

    #[test]
    fn test_parse_context_word_boundary_final() {
        let input = "e -> / _#";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        assert!(ctx.left.is_none());
        // Right should be word boundary
        assert!(matches!(
            ctx.right.as_deref(),
            Some(ContextExpr::WordBoundary)
        ));
    }

    #[test]
    fn test_parse_context_not() {
        let input = "c -> k / _![ei]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        assert!(ctx.left.is_none());
        // Right should be NOT
        match ctx.right.as_deref() {
            Some(ContextExpr::Not(inner)) => {
                assert!(matches!(**inner, ContextExpr::Pattern(_)));
            }
            _ => panic!("expected Not context"),
        }
    }

    #[test]
    fn test_parse_context_and() {
        let input = "x -> gz / [aeiou]&[aeiou]_";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        // Left should be AND
        match ctx.left.as_deref() {
            Some(ContextExpr::And(_, _)) => {}
            _ => panic!("expected And context"),
        }
        assert!(ctx.right.is_none());
    }

    #[test]
    fn test_parse_context_or() {
        let input = "e -> / _([bcdfg]|#)";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        assert!(ctx.left.is_none());
        // Right should be OR (grouped)
        match ctx.right.as_deref() {
            Some(ContextExpr::Or(_, _)) => {}
            _ => panic!("expected Or context, got {:?}", ctx.right),
        }
    }

    #[test]
    fn test_parse_context_complex_compound() {
        // NOT has higher precedence than AND, which is higher than OR
        let input = "t -> d / ![x]&[aeiou]|[ou]_";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        // Left should be OR at the top level
        match ctx.left.as_deref() {
            Some(ContextExpr::Or(_, _)) => {}
            _ => panic!("expected Or context at top level"),
        }
    }

    // ==================== Syllable Condition Tests ====================

    #[test]
    fn test_parse_syllable_monosyllable() {
        let input = "y -> i / _# if monosyllable";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        match &ctx.syllable {
            Some(SyllableExpr::Cond(SyllableCondition::Monosyllable)) => {}
            _ => panic!("expected monosyllable condition"),
        }
    }

    #[test]
    fn test_parse_syllable_polysyllable() {
        let input = "y -> i / _# if polysyllable";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        match &ctx.syllable {
            Some(SyllableExpr::Cond(SyllableCondition::Polysyllable)) => {}
            _ => panic!("expected polysyllable condition"),
        }
    }

    #[test]
    fn test_parse_syllable_and() {
        let input = "y -> i / _# if polysyllable & final_syllable";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        match &ctx.syllable {
            Some(SyllableExpr::And(_, _)) => {}
            _ => panic!("expected And syllable condition"),
        }
    }

    #[test]
    fn test_parse_syllable_or() {
        let input = "a -> aa / _ if open_syllable | initial_syllable";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        match &ctx.syllable {
            Some(SyllableExpr::Or(_, _)) => {}
            _ => panic!("expected Or syllable condition"),
        }
    }

    #[test]
    fn test_parse_syllable_not() {
        let input = "e -> i / _[bcdfg] if !final_syllable";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        match &ctx.syllable {
            Some(SyllableExpr::Not(_)) => {}
            _ => panic!("expected Not syllable condition"),
        }
    }

    #[test]
    fn test_parse_syllable_complex() {
        let input = "t -> d / [aeiou]_[bcdfg] if !monosyllable & !final_syllable";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        // Should be And at top level with two Not conditions
        match &ctx.syllable {
            Some(SyllableExpr::And(left, right)) => {
                assert!(matches!(**left, SyllableExpr::Not(_)));
                assert!(matches!(**right, SyllableExpr::Not(_)));
            }
            _ => panic!("expected And of two Not conditions"),
        }
    }

    #[test]
    fn test_parse_syllable_with_parens() {
        let input = "y -> i / _# if (monosyllable | polysyllable) & final_syllable";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        // Should be And at top level with Or on left side
        match &ctx.syllable {
            Some(SyllableExpr::And(left, _)) => {
                assert!(matches!(**left, SyllableExpr::Or(_, _)));
            }
            _ => panic!("expected And with Or on left"),
        }
    }

    #[test]
    fn test_parse_all_syllable_keywords() {
        // Test all six syllable keywords parse correctly
        let keywords = [
            ("monosyllable", SyllableCondition::Monosyllable),
            ("polysyllable", SyllableCondition::Polysyllable),
            ("open_syllable", SyllableCondition::OpenSyllable),
            ("closed_syllable", SyllableCondition::ClosedSyllable),
            ("final_syllable", SyllableCondition::FinalSyllable),
            ("initial_syllable", SyllableCondition::InitialSyllable),
        ];

        for (kw, expected_cond) in keywords {
            let input = format!("a -> b / _ if {}", kw);
            let file = parse_str(&input).expect(&format!("failed to parse {}", kw));
            assert_eq!(file.rules.len(), 1);

            let rule = &file.rules[0].rule;
            let ctx = rule.context.as_ref().expect("should have context");
            match &ctx.syllable {
                Some(SyllableExpr::Cond(cond)) => {
                    assert_eq!(
                        std::mem::discriminant(cond),
                        std::mem::discriminant(&expected_cond),
                        "keyword {} didn't produce expected condition",
                        kw
                    );
                }
                _ => panic!("expected Cond for keyword {}", kw),
            }
        }
    }

    #[test]
    fn test_parse_context_only_no_syllable() {
        // Verify that rules without syllable clause still work
        let input = "c -> s / _[ei]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");
        assert!(ctx.syllable.is_none());
    }

    #[test]
    fn test_parse_no_context_no_syllable() {
        // Rules without context also shouldn't have syllable
        let input = "ph -> f";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        assert!(rule.context.is_none());
    }

    #[test]
    fn test_parse_user_symbol_in_char_class() {
        // $SYMBOL inside character classes expands the symbol
        // [$MY_VOWEL] = vowels (a, e, i, o, u)
        let input = r#"
@define MY_VOWEL = [aeiou]
x -> gz / [$MY_VOWEL]_[AB]
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        // Left context should be a char class with vowels from MY_VOWEL symbol
        if let Some(ref boxed) = ctx.left {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, negated } = expr {
                    assert!(!negated);
                    // Should contain vowels from the symbol
                    assert!(chars.contains(&'a'));
                    assert!(chars.contains(&'e'));
                    assert!(chars.contains(&'i'));
                    assert!(chars.contains(&'o'));
                    assert!(chars.contains(&'u'));
                    // Should NOT contain literal $, M, Y
                    assert!(!chars.contains(&'$'));
                    assert!(!chars.contains(&'M'));
                    assert!(!chars.contains(&'Y'));
                } else {
                    panic!("expected CharClass in left context");
                }
            } else {
                panic!("expected Pattern in left context");
            }
        } else {
            panic!("expected Some in left context");
        }
    }

    #[test]
    fn test_parse_negated_user_symbol_in_char_class() {
        // Use \$ for literal $ in negated char class: [^\$ab] = negated class with $, a, b
        let input = r#"
c -> k / _[^\$ab]
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        // Right context should be a negated char class with $, a, b
        if let Some(ref boxed) = ctx.right {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, negated } = expr {
                    assert!(negated);
                    assert!(chars.contains(&'$'));
                    assert!(chars.contains(&'a'));
                    assert!(chars.contains(&'b'));
                } else {
                    panic!("expected CharClass in right context");
                }
            } else {
                panic!("expected Pattern in right context");
            }
        } else {
            panic!("expected Some in right context");
        }
    }

    #[test]
    fn test_parse_mixed_symbol_and_chars_in_class() {
        // ${SYMBOL} inside character class expands the symbol
        // Define M symbol and test expansion
        let input = r#"
@define M = [mn]
a -> b / _[${M}xyz]
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        // Right context should have m, n (from M symbol), x, y, z
        if let Some(ref boxed) = ctx.right {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, negated } = expr {
                    assert!(!negated);
                    // From symbol M
                    assert!(chars.contains(&'m'));
                    assert!(chars.contains(&'n'));
                    // Literal chars
                    assert!(chars.contains(&'x'));
                    assert!(chars.contains(&'y'));
                    assert!(chars.contains(&'z'));
                    // Should NOT contain literal $, {, }
                    assert!(!chars.contains(&'$'));
                    assert!(!chars.contains(&'{'));
                    assert!(!chars.contains(&'}'));
                } else {
                    panic!("expected CharClass in right context");
                }
            } else {
                panic!("expected Pattern in right context");
            }
        } else {
            panic!("expected Some in right context");
        }
    }

    #[test]
    fn test_parse_dollar_as_literal_in_char_class() {
        // $ is a literal character inside character classes
        // This is standard regex convention
        let input = "a -> b / _[$]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        if let Some(ref boxed) = ctx.right {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, negated } = expr {
                    assert!(!negated);
                    assert_eq!(chars.len(), 1);
                    assert!(chars.contains(&'$'));
                } else {
                    panic!("expected CharClass");
                }
            } else {
                panic!("expected Pattern");
            }
        } else {
            panic!("expected right context");
        }
    }

    #[test]
    fn test_parse_empty_named_class_error() {
        // Test error for empty named class [::]
        // This tests POSIX syntax with empty name
        let input = "a -> b / _[[::]xyz]";
        let result = parse_str(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLevErrorKind::InvalidPattern(_)));
    }

    #[test]
    fn test_parse_symbol_reference_outside_char_class() {
        // Symbol references ($NAME) work outside character classes
        // but inside char classes, $ is literal
        let input = r#"
@define MY_VOWEL = [aeiou]
$MY_VOWEL -> X
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        // Pattern should be the vowels from MY_VOWEL
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'e'));
            assert!(chars.contains(&'i'));
            assert!(chars.contains(&'o'));
            assert!(chars.contains(&'u'));
        } else {
            panic!("expected CharClass");
        }
    }

    // ==================== Built-in Named Class Tests ====================

    #[test]
    fn test_parse_builtin_vowel_class() {
        // Test built-in [:vowel:] class
        let input = "c -> s / _[[:vowel:]]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        if let Some(ref boxed) = ctx.right {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, negated } = expr {
                    assert!(!negated);
                    // Should contain ASCII vowels
                    assert!(chars.contains(&'a'));
                    assert!(chars.contains(&'e'));
                    assert!(chars.contains(&'i'));
                    assert!(chars.contains(&'o'));
                    assert!(chars.contains(&'u'));
                    // Should contain uppercase
                    assert!(chars.contains(&'A'));
                    // Should contain IPA vowels
                    assert!(chars.contains(&'ə'));
                } else {
                    panic!("expected CharClass");
                }
            } else {
                panic!("expected Pattern");
            }
        } else {
            panic!("expected Some");
        }
    }

    #[test]
    fn test_parse_builtin_full_word_alias() {
        // Test full-word alias [:plosive:] (alias for stop)
        let input = "[[:plosive:]] -> 0";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, negated } = rule.pattern {
            assert!(!negated);
            // plosive should be alias for stop
            assert!(chars.contains(&'p'));
            assert!(chars.contains(&'t'));
            assert!(chars.contains(&'k'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_builtin_case_insensitive() {
        // Test case-insensitive lookup [:VOWEL:] and [:Vowel:]
        let inputs = vec![
            "c -> s / _[[:VOWEL:]]",
            "c -> s / _[[:Vowel:]]",
            "c -> s / _[[:vowel:]]",
        ];

        for input in inputs {
            let file = parse_str(input).expect(&format!("should parse: {}", input));
            assert_eq!(file.rules.len(), 1);

            let rule = &file.rules[0].rule;
            let ctx = rule.context.as_ref().expect("should have context");

            if let Some(ref boxed) = ctx.right {
                if let ContextExpr::Pattern(ref expr) = **boxed {
                    if let Expression::CharClass { ref chars, .. } = expr {
                        assert!(chars.contains(&'a'));
                    }
                }
            }
        }
    }

    #[test]
    fn test_parse_builtin_consonant_class() {
        // Test built-in [:consonant:] class
        let input = "[[:consonant:]] -> 0";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, negated } = rule.pattern {
            assert!(!negated);
            // Should contain ASCII consonants
            assert!(chars.contains(&'b'));
            assert!(chars.contains(&'c'));
            assert!(chars.contains(&'d'));
            // Should NOT contain vowels
            assert!(!chars.contains(&'a'));
            assert!(!chars.contains(&'e'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_builtin_posix_alpha() {
        // Test POSIX [:alpha:] class
        let input = "[[:alpha:]] -> x";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, .. } = rule.pattern {
            // Should have all letters
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'z'));
            assert!(chars.contains(&'A'));
            assert!(chars.contains(&'Z'));
            // Should NOT have digits
            assert!(!chars.contains(&'0'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_builtin_mixed_with_chars() {
        // Test mixing built-in class with extra chars: [[:vowel:]y]
        let input = "c -> s / _[[:vowel:]y]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        if let Some(ref boxed) = ctx.right {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, .. } = expr {
                    // Should have vowels AND 'y'
                    assert!(chars.contains(&'a'));
                    assert!(chars.contains(&'e'));
                    assert!(chars.contains(&'y'));
                } else {
                    panic!("expected CharClass");
                }
            }
        }
    }

    #[test]
    fn test_parse_define_lowercase_symbol_rejected() {
        // Test that lowercase symbol names are rejected
        let input = "@define vowel = [aeiou]";
        let result = parse_str(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLevErrorKind::SymbolNameMustBeUppercase { .. }));
    }

    #[test]
    fn test_parse_define_uppercase_required() {
        // Test that only UPPERCASE symbol names are accepted
        // These should fail (contain lowercase)
        let invalid_inputs = vec![
            "@define vowel = [aeiou]",  // all lowercase
            "@define Vowel = [aeiou]",  // mixed case
            "@define alpha = [abc]",    // all lowercase
        ];

        for input in invalid_inputs {
            let result = parse_str(input);
            assert!(result.is_err(), "should error on: {}", input);
            let err = result.unwrap_err();
            assert!(
                matches!(err.kind, LLevErrorKind::SymbolNameMustBeUppercase { .. }),
                "expected SymbolNameMustBeUppercase for: {}", input
            );
        }

        // These should succeed (all uppercase)
        let valid_inputs = vec![
            "@define VOWEL = [aeiou]",
            "@define V = [aeiou]",
            "@define MY_CLASS = [abc]",
        ];

        for input in valid_inputs {
            let result = parse_str(input);
            assert!(result.is_ok(), "should succeed for: {}, got: {:?}", input, result.err());
        }
    }

    #[test]
    fn test_parse_user_defined_non_conflicting() {
        // Test that user can define symbols that don't conflict with built-ins
        // User symbols use $ sigil
        let input = r#"
@define MY_VOWELS = [aeiou]
c -> s / _[$MY_VOWELS]
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.symbols.len(), 1);
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_parse_builtin_front_vowel() {
        // Test front vowel class (full name)
        let input = "c -> s / _[[:front_vowel:]]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        if let Some(ref boxed) = ctx.right {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, .. } = expr {
                    // Front vowels: e, i
                    assert!(chars.contains(&'e'));
                    assert!(chars.contains(&'i'));
                    // Should NOT have back vowels
                    assert!(!chars.contains(&'o'));
                    assert!(!chars.contains(&'u'));
                } else {
                    panic!("expected CharClass");
                }
            }
        }
    }

    #[test]
    fn test_parse_builtin_stop_consonant() {
        // Test stop consonant class (full name)
        let input = "[[:stop:]] -> 0";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, .. } = rule.pattern {
            // Stop consonants: p, b, t, d, k, g
            assert!(chars.contains(&'p'));
            assert!(chars.contains(&'b'));
            assert!(chars.contains(&'t'));
            assert!(chars.contains(&'d'));
            assert!(chars.contains(&'k'));
            assert!(chars.contains(&'g'));
            // Should NOT have fricatives
            assert!(!chars.contains(&'f'));
            assert!(!chars.contains(&'s'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_builtin_ascii_vowel() {
        // Test ASCII-only vowel subset
        let input = "[[:ascii_vowel:]] -> V";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, .. } = rule.pattern {
            // Should have ASCII vowels
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'e'));
            // Should NOT have IPA vowels
            assert!(!chars.contains(&'ə'));
            assert!(!chars.contains(&'ɪ'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_builtin_ipa_vowel() {
        // Test IPA-only vowel subset
        let input = "[[:ipa_vowel:]] -> V";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, .. } = rule.pattern {
            // Should have IPA vowels
            assert!(chars.contains(&'ə'));
            assert!(chars.contains(&'ɪ'));
            // Should NOT have ASCII vowels
            assert!(!chars.contains(&'a'));
            assert!(!chars.contains(&'e'));
        } else {
            panic!("expected CharClass");
        }
    }

    // ==================== Standalone Named Class Syntax Tests ====================

    #[test]
    fn test_parse_standalone_vowel_class() {
        // Test standalone [:vowel:] syntax (shorthand for [[:vowel:]])
        let input = "[:vowel:] -> V";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, negated } = rule.pattern {
            assert!(!negated);
            // Should contain vowels
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'e'));
            assert!(chars.contains(&'ə'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_standalone_vowel() {
        // Test standalone [:vowel:] syntax
        let input = "[:vowel:] -> V";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, .. } = rule.pattern {
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'e'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_standalone_in_context() {
        // Test standalone syntax in context
        let input = "c -> s / _[:front_vowel:]";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        let ctx = rule.context.as_ref().expect("should have context");

        if let Some(ref boxed) = ctx.right {
            if let ContextExpr::Pattern(ref expr) = **boxed {
                if let Expression::CharClass { ref chars, .. } = expr {
                    // front_vowel = front vowels (e, i)
                    assert!(chars.contains(&'e'));
                    assert!(chars.contains(&'i'));
                    assert!(!chars.contains(&'o'));
                } else {
                    panic!("expected CharClass");
                }
            } else {
                panic!("expected Pattern");
            }
        } else {
            panic!("expected right context");
        }
    }

    #[test]
    fn test_parse_standalone_consonant() {
        // Test standalone [:consonant:] in pattern
        let input = "[:consonant:][:vowel:] -> CV";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        // Pattern should be Concat(CharClass(consonants), CharClass(vowels))
        match &rule.pattern {
            Expression::Concat(left, right) => {
                if let Expression::CharClass { chars: l_chars, .. } = &**left {
                    assert!(l_chars.contains(&'b'));
                    assert!(l_chars.contains(&'c'));
                    assert!(!l_chars.contains(&'a'));
                } else {
                    panic!("expected left CharClass");
                }
                if let Expression::CharClass { chars: r_chars, .. } = &**right {
                    assert!(r_chars.contains(&'a'));
                    assert!(r_chars.contains(&'e'));
                    assert!(!r_chars.contains(&'b'));
                } else {
                    panic!("expected right CharClass");
                }
            }
            _ => panic!("expected Concat"),
        }
    }

    #[test]
    fn test_parse_standalone_posix_digit() {
        // Test standalone [:digit:] POSIX class
        let input = "[:digit:] -> 0";
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        if let Expression::CharClass { ref chars, .. } = rule.pattern {
            assert!(chars.contains(&'0'));
            assert!(chars.contains(&'5'));
            assert!(chars.contains(&'9'));
            assert!(!chars.contains(&'a'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_standalone_user_symbol() {
        // Test user symbol with $ sigil as pattern
        let input = r#"
@define MY_CHARS = [xyz]
$MY_CHARS -> 0
"#;
        let file = parse_str(input).unwrap();
        assert_eq!(file.rules.len(), 1);

        let rule = &file.rules[0].rule;
        // Pattern is expanded to CharClass at parse time
        if let Expression::CharClass { ref chars, .. } = rule.pattern {
            assert_eq!(chars.len(), 3);
            assert!(chars.contains(&'x'));
            assert!(chars.contains(&'y'));
            assert!(chars.contains(&'z'));
        } else {
            panic!("expected CharClass, got {:?}", rule.pattern);
        }
    }

    #[test]
    fn test_parse_standalone_undefined_symbol_error() {
        // Test error for undefined symbol with $ sigil
        let input = "$UNDEFINED_CLASS -> x";
        let result = parse_str(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLevErrorKind::UndefinedSymbol(_)));
    }

    #[test]
    fn test_parse_standalone_empty_error() {
        // Test error for empty standalone named class [::]
        let input = "[::]-> x";
        let result = parse_str(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLevErrorKind::InvalidPattern(_)));
    }

    #[test]
    fn test_parse_standalone_vs_posix() {
        // Test that both syntaxes produce the same result
        let standalone = parse_str("[:vowel:] -> V").unwrap();
        let posix = parse_str("[[:vowel:]] -> V").unwrap();

        let standalone_rule = &standalone.rules[0].rule;
        let posix_rule = &posix.rules[0].rule;

        if let (
            Expression::CharClass { chars: s_chars, .. },
            Expression::CharClass { chars: p_chars, .. },
        ) = (&standalone_rule.pattern, &posix_rule.pattern)
        {
            assert_eq!(s_chars.len(), p_chars.len());
            for c in s_chars {
                assert!(p_chars.contains(c));
            }
        } else {
            panic!("expected CharClass for both");
        }
    }

    #[test]
    fn test_parse_inline_named_class() {
        // [x:vowel:] is NOW parsed as literals: x, :, v, o, w, e, l, :
        // To union with named class, use: [x[[:vowel:]]]
        let file = parse_str("[x:vowel:] -> X").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            // All chars are literals
            assert!(chars.contains(&'x'));
            assert!(chars.contains(&':'));
            assert!(chars.contains(&'v'));
            assert!(chars.contains(&'o'));
            assert!(chars.contains(&'w'));
            assert!(chars.contains(&'e'));
            assert!(chars.contains(&'l'));
            // Should NOT contain 'a', 'i', 'u' (vowels not from named class)
            assert!(!chars.contains(&'a'));
            assert!(!chars.contains(&'i'));
            assert!(!chars.contains(&'u'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_inline_named_class_correct_syntax() {
        // [x[:vowel:]] is the correct syntax for x union with named class
        // Note: [x[:vowel:]] is equivalent to [x[[:vowel:]]]
        let file = parse_str("[x[:vowel:]] -> X").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            assert!(chars.contains(&'x'));
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'e'));
            assert!(chars.contains(&'i'));
            assert!(chars.contains(&'o'));
            assert!(chars.contains(&'u'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_inline_named_class_bracket_equivalence() {
        // All these are equivalent:
        // [x[:vowel:]] = [x[[:vowel:]]] = [x[[[:vowel:]]]]
        let file1 = parse_str("[x[:vowel:]] -> X").unwrap();
        let file2 = parse_str("[x[[:vowel:]]] -> X").unwrap();
        let file3 = parse_str("[x[[[:vowel:]]]] -> X").unwrap();
        if let (
            Expression::CharClass { chars: chars1, .. },
            Expression::CharClass { chars: chars2, .. },
            Expression::CharClass { chars: chars3, .. },
        ) = (
            &file1.rules[0].rule.pattern,
            &file2.rules[0].rule.pattern,
            &file3.rules[0].rule.pattern,
        ) {
            // All should have the same chars
            assert_eq!(chars1.len(), chars2.len());
            assert_eq!(chars1.len(), chars3.len());
            for c in chars1 {
                assert!(chars2.contains(c));
                assert!(chars3.contains(c));
            }
        } else {
            panic!("expected CharClass for all");
        }
    }

    #[test]
    fn test_parse_inline_named_class_with_symbol() {
        // $SYMBOL inside character classes expands the symbol
        // Define ABC symbol and test that it expands
        let input = r#"
@define ABC = [abc]
[$ABC:xyz:] -> X
"#;
        let file = parse_str(input).unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            // a, b, c from ABC symbol
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'b'));
            assert!(chars.contains(&'c'));
            // : is literal
            assert!(chars.contains(&':'));
            // x, y, z from the middle part
            assert!(chars.contains(&'x'));
            assert!(chars.contains(&'y'));
            assert!(chars.contains(&'z'));
            // Should NOT contain literal $, A, B, C
            assert!(!chars.contains(&'$'));
            assert!(!chars.contains(&'A'));
            assert!(!chars.contains(&'B'));
            assert!(!chars.contains(&'C'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_chars_union_named_class() {
        // [xyz[:consonant:]] unions literal chars with named class
        let file = parse_str("[xyz[:consonant:]] -> X").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            // Should have literal x, y, z
            assert!(chars.contains(&'x'));
            assert!(chars.contains(&'y'));
            assert!(chars.contains(&'z'));
            // Should have consonants from named class
            assert!(chars.contains(&'b'));
            assert!(chars.contains(&'c'));
            assert!(chars.contains(&'d'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_negated_nested_class() {
        // Test [^[:vowel:]] syntax - negated vowels = consonants + other chars
        let file = parse_str("[[^:vowel:]] -> C").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            // The char class itself is not negated, but contains negated vowels
            assert!(!negated);
            // Should NOT contain vowels
            assert!(!chars.contains(&'a'));
            assert!(!chars.contains(&'e'));
            assert!(!chars.contains(&'i'));
            assert!(!chars.contains(&'o'));
            assert!(!chars.contains(&'u'));
            // Should contain consonants and other printable ASCII
            assert!(chars.contains(&'b'));
            assert!(chars.contains(&'c'));
            assert!(chars.contains(&'!'));
            assert!(chars.contains(&' '));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_nested_class_union() {
        // Test union of nested classes with literal chars + named class
        // Since $ is literal inside char classes, use [abc[:consonant:]] syntax
        let file = parse_str("[abc[:consonant:]] -> X").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            // Should have literal a, b, c
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'b'));
            assert!(chars.contains(&'c'));
            // Should have consonants from named class (includes b, c, d, etc.)
            assert!(chars.contains(&'d'));
            assert!(chars.contains(&'f'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_deeply_nested_class() {
        // Test arbitrary nesting: [[[[:vowel:]]]]
        let file = parse_str("[[[[:vowel:]]]] -> V").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'e'));
            assert!(chars.contains(&'i'));
            assert!(chars.contains(&'o'));
            assert!(chars.contains(&'u'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_colon_as_literal() {
        // Bare ':' without closing ':' should be treated as literal
        let file = parse_str("[:abc] -> X").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            // ':' 'a' 'b' 'c' should all be literals
            assert!(chars.contains(&':'));
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'b'));
            assert!(chars.contains(&'c'));
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_escaped_brackets_in_char_class() {
        // \[ and \] should be literal brackets inside character class
        let file = parse_str(r"[a\[\]b] -> X").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'['));
            assert!(chars.contains(&']'));
            assert!(chars.contains(&'b'));
            assert_eq!(chars.len(), 4);
        } else {
            panic!("expected CharClass");
        }
    }

    #[test]
    fn test_parse_escaped_vs_unescaped_bracket() {
        // [ starts nested class, \[ is literal
        let file = parse_str(r"[\[[[:vowel:]]] -> X").unwrap();
        let rule = &file.rules[0].rule;
        if let Expression::CharClass { chars, negated } = &rule.pattern {
            assert!(!negated);
            // Should have literal '[' from \[
            assert!(chars.contains(&'['));
            // Should have vowels from nested class
            assert!(chars.contains(&'a'));
            assert!(chars.contains(&'e'));
            assert!(chars.contains(&'i'));
            assert!(chars.contains(&'o'));
            assert!(chars.contains(&'u'));
        } else {
            panic!("expected CharClass");
        }
    }
}
