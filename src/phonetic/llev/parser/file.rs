//! File-level structural parsing.
//!
//! Implements the top-level `parse_file` driver: directives
//! (`@name`/`@version`/`@include`/`@define`), metadata blocks, and
//! rewrite-rule definitions. The actual expression and context
//! sub-grammars are dispatched to `expr.rs` and `context.rs`.

use std::collections::HashMap;

use super::super::ast::{
    ContextAST, Expression, IncludeDirective, LLevFile, RewriteRuleAST, RuleDefinition,
    RuleMetadata, SymbolDef,
};
use super::super::error::{LLevError, LLevErrorKind, LLevResult};
use super::super::lexer::{Lexer, Token};
use super::Parser;

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

    // ==================== Directive Parsing ====================

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
                "ipa" => {
                    // IPA transcription with bracket notation: "/ʃ/" (phonemic) or "[ʃ]" (phonetic)
                    let ipa = self.expect_string()?;
                    metadata.ipa = Some(ipa);
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
