//! NFA compiler for phonetic regular expressions.
//!
//! This module compiles parsed regex ASTs into NFAs using Thompson's construction.
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::nfa::compiler::compile;
//! use liblevenshtein::phonetic::regex::parse;
//!
//! // Compile a simple pattern
//! let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
//! assert!(nfa.accepts("phone"));
//! assert!(nfa.accepts("fone"));
//! assert!(!nfa.accepts("bone"));
//!
//! // Compile a rewrite rule
//! let rule = parse_rule("ph -> f").unwrap();
//! let rewrite = compile_rewrite(&rule).unwrap();
//! ```

use std::collections::HashMap;

use super::context::{BoundaryKind, ContextPattern, ContextPatternChar};
use super::nfa::{NFAChar, NFA};
use super::optimizer::{OptimizationConfig, NfaOptimizerChar};
use super::thompson::{ThompsonBuilder, ThompsonBuilderChar};
use crate::phonetic::regex::ast::{ContextExpr, ContextExprByte, Regex, RegexByte, RegexFlags, UnicodeNormalization};
use crate::phonetic::regex::error::{ParseError, ParseErrorKind, ParseResult, Position};
use crate::phonetic::regex::transform::{apply_flags, TransformResult};

/// Result of compiling a regex with flag support.
///
/// Contains both the compiled NFA and runtime settings extracted from flags.
#[derive(Debug, Clone)]
pub struct CompileResultChar {
    /// The compiled NFA.
    pub nfa: NFAChar,
    /// Unicode normalization to apply to input at runtime.
    pub unicode_normalization: Option<UnicodeNormalization>,
    /// Whether multiline mode is enabled (for `^` and `$`).
    pub multiline: bool,
    /// Whether dotall mode is enabled (`.` matches newlines).
    pub dotall: bool,
    /// Local Levenshtein distance override (from `(?;N)` syntax).
    ///
    /// When set, this distance limit should be used for fuzzy matching
    /// instead of the global distance parameter.
    pub local_distance: Option<u8>,
}

/// Compile a character-level regex to an NFA.
///
/// This applies regex flag transformations (like `(?i)` case-insensitive)
/// before compilation. For flags that affect runtime behavior (unicode
/// normalization, multiline, dotall), use [`compile_with_flags`] instead
/// to access those settings.
///
/// # Arguments
///
/// * `regex` - The regex AST to compile
///
/// # Returns
///
/// An NFA that accepts the language defined by the regex.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::phonetic::regex::parse;
/// use liblevenshtein::phonetic::nfa::compiler::compile;
///
/// let regex = parse("a+b*c").unwrap();
/// let nfa = compile(&regex).unwrap();
/// assert!(nfa.accepts("ac"));
/// assert!(nfa.accepts("aaac"));
/// assert!(nfa.accepts("abbc"));
///
/// // Case-insensitive matching via (?i) flag
/// let regex = parse("(?i:hello)").unwrap();
/// let nfa = compile(&regex).unwrap();
/// assert!(nfa.accepts("hello"));
/// assert!(nfa.accepts("HELLO"));
/// assert!(nfa.accepts("HeLLo"));
/// ```
pub fn compile(regex: &Regex) -> ParseResult<NFAChar> {
    let mut compiler = NFACompilerChar::new();
    compiler.compile(regex)
}

/// Compile a character-level regex to an NFA with full flag support.
///
/// This applies regex flag transformations and returns both the NFA and
/// runtime settings extracted from flags like `(?u:NFC)`, `(?m)`, and `(?s)`.
///
/// # Arguments
///
/// * `regex` - The regex AST to compile
///
/// # Returns
///
/// A [`CompileResultChar`] containing the NFA and runtime settings.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::phonetic::regex::parse;
/// use liblevenshtein::phonetic::nfa::compiler::compile_with_flags;
///
/// let regex = parse("(?iu:café)").unwrap();
/// let result = compile_with_flags(&regex).unwrap();
///
/// // NFA matches case-insensitively (transformed)
/// assert!(result.nfa.accepts("café"));
/// assert!(result.nfa.accepts("CAFÉ"));
///
/// // Unicode normalization available for runtime use
/// if let Some(norm) = result.unicode_normalization {
///     // Apply normalization to input before matching
/// }
/// ```
pub fn compile_with_flags(regex: &Regex) -> ParseResult<CompileResultChar> {
    let mut compiler = NFACompilerChar::new();
    compiler.compile_with_flags(regex)
}

/// Compile a byte-level regex to an NFA.
pub fn compile_bytes(regex: &RegexByte) -> ParseResult<NFA> {
    let mut compiler = NFACompilerByte::new();
    compiler.compile(regex)
}

/// A compiled rewrite rule.
///
/// Contains the source pattern NFA, replacement string, and optional context.
#[derive(Debug, Clone)]
pub struct CompiledRewriteChar {
    /// NFA matching the source pattern
    pub source: NFAChar,
    /// Characters to replace with
    pub replacement: Vec<char>,
    /// Optional left context (lookbehind)
    pub left_context: Option<ContextPatternChar>,
    /// Optional right context (lookahead)
    pub right_context: Option<ContextPatternChar>,
    /// Weight/cost for this rule
    pub weight: f64,
}

/// A compiled byte-level rewrite rule.
#[derive(Debug, Clone)]
pub struct CompiledRewrite {
    /// NFA matching the source pattern
    pub source: NFA,
    /// Bytes to replace with
    pub replacement: Vec<u8>,
    /// Optional left context (lookbehind)
    pub left_context: Option<ContextPattern>,
    /// Optional right context (lookahead)
    pub right_context: Option<ContextPattern>,
    /// Weight/cost for this rule
    pub weight: f64,
}

/// Compile a rewrite rule regex to a compiled rewrite structure.
pub fn compile_rewrite(regex: &Regex) -> ParseResult<CompiledRewriteChar> {
    let mut compiler = NFACompilerChar::new();
    compiler.compile_rewrite(regex)
}

/// Compile a byte-level rewrite rule.
pub fn compile_rewrite_bytes(regex: &RegexByte) -> ParseResult<CompiledRewrite> {
    let mut compiler = NFACompilerByte::new();
    compiler.compile_rewrite(regex)
}

// ============================================================================
// Trampolining support for stack-safe compilation
// ============================================================================

/// Work item for trampolined regex compilation.
///
/// This enum represents the operations that need to be performed during
/// iterative (trampolined) compilation. Using an explicit work stack instead
/// of the call stack allows compilation of deeply nested patterns without
/// risk of stack overflow.
#[derive(Debug)]
enum CompileWork<'a> {
    /// Compile this regex node and push result to value stack.
    Compile(&'a Regex),
    /// Pop two values, concatenate them, push result.
    DoConcatenate,
    /// Pop two values, create alternation, push result.
    DoAlternation,
    /// Pop one value, apply Kleene star, push result.
    DoStar,
    /// Pop one value, apply Kleene plus, push result.
    DoPlus,
    /// Pop one value, apply optional, push result.
    DoOptional,
    /// Pop one value, apply exact repetition, push result.
    DoRepeatExact(usize),
    /// Pop one value, apply range repetition, push result.
    DoRepeatRange(usize, Option<usize>),
}

/// Work item for trampolined literal extraction.
#[derive(Debug)]
enum LiteralWork<'a> {
    /// Process this regex node.
    Process(&'a Regex),
    /// Pop two char vectors, concatenate them, push result.
    DoConcat,
}

/// Work item for trampolined context expression compilation.
#[derive(Debug)]
enum ContextWork<'a> {
    /// Process this context expression.
    Process(&'a ContextExpr),
    /// Pop two context patterns, create And, push result.
    DoAnd,
    /// Pop two context patterns, create Or, push result.
    DoOr,
    /// Pop one context pattern, create Not, push result.
    DoNot,
}

/// Character-level NFA compiler.
pub struct NFACompilerChar {
    builder: ThompsonBuilderChar,
    /// Symbol table for pattern expansion
    symbols: HashMap<String, Vec<char>>,
    /// Regex flags (multiline, dotall, etc.)
    flags: RegexFlags,
    /// Use trampolining for stack-safe compilation of deep patterns.
    /// When false (default), uses recursive compilation which is faster
    /// but may stack overflow on deeply nested patterns.
    use_trampolining: bool,
    /// Optimization configuration. When Some, the NFA is optimized after compilation.
    /// Default: full optimization enabled.
    optimization: Option<OptimizationConfig>,
}

impl NFACompilerChar {
    /// Create a new compiler with recursive compilation and optimization enabled (default).
    pub fn new() -> Self {
        Self {
            builder: ThompsonBuilderChar::new(),
            symbols: HashMap::new(),
            flags: RegexFlags::default(),
            use_trampolining: false,
            optimization: Some(OptimizationConfig::full()),
        }
    }

    /// Enable trampolining for stack-safe compilation.
    ///
    /// Use this when compiling potentially deep patterns from untrusted input.
    /// Trampolining uses an explicit heap-allocated stack instead of the call stack,
    /// preventing stack overflow on deeply nested patterns like `((((a))))`.
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::phonetic::nfa::NFACompilerChar;
    ///
    /// // For untrusted input, use trampolining
    /// let compiler = NFACompilerChar::new().with_trampolining();
    /// ```
    pub fn with_trampolining(mut self) -> Self {
        self.use_trampolining = true;
        self
    }

    /// Set trampolining mode.
    ///
    /// When `enabled` is true, compilation uses an iterative trampoline-based
    /// approach that is safe for deeply nested patterns but slightly slower.
    /// When false (default), uses recursive compilation which is faster.
    pub fn set_trampolining(&mut self, enabled: bool) {
        self.use_trampolining = enabled;
    }

    /// Check if trampolining is enabled.
    pub fn is_trampolining(&self) -> bool {
        self.use_trampolining
    }

    /// Enable optimization with the given configuration.
    ///
    /// By default, full optimization is enabled. Use this to customize
    /// which optimization passes are applied.
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::phonetic::nfa::{NFACompilerChar, OptimizationConfig};
    ///
    /// // Use quick optimization (no epsilon elimination)
    /// let compiler = NFACompilerChar::new()
    ///     .with_optimization(OptimizationConfig::quick());
    /// ```
    pub fn with_optimization(mut self, config: OptimizationConfig) -> Self {
        self.optimization = Some(config);
        self
    }

    /// Disable optimization entirely.
    ///
    /// This is useful for debugging or when the NFA will be used only once.
    /// The resulting NFA may have epsilon transitions and unreachable states.
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::phonetic::nfa::NFACompilerChar;
    ///
    /// let compiler = NFACompilerChar::new().without_optimization();
    /// ```
    pub fn without_optimization(mut self) -> Self {
        self.optimization = None;
        self
    }

    /// Set the optimization configuration.
    ///
    /// Pass `None` to disable optimization, or `Some(config)` to enable it.
    pub fn set_optimization(&mut self, config: Option<OptimizationConfig>) {
        self.optimization = config;
    }

    /// Check if optimization is enabled.
    pub fn is_optimization_enabled(&self) -> bool {
        self.optimization.is_some()
    }

    /// Get the current optimization configuration.
    pub fn optimization_config(&self) -> Option<&OptimizationConfig> {
        self.optimization.as_ref()
    }

    /// Add a symbol to the symbol table.
    ///
    /// Symbols can be referenced in patterns and will be expanded during compilation.
    pub fn add_symbol(&mut self, name: impl Into<String>, chars: Vec<char>) {
        self.symbols.insert(name.into(), chars);
    }

    /// Set the regex flags for compilation.
    pub fn set_flags(&mut self, flags: RegexFlags) {
        self.flags = flags;
    }

    /// Get the current flags.
    pub fn flags(&self) -> &RegexFlags {
        &self.flags
    }

    /// Get a symbol by name.
    pub fn get_symbol(&self, name: &str) -> Option<&Vec<char>> {
        self.symbols.get(name)
    }

    /// Compile a regex AST to an NFA.
    ///
    /// This applies flag transformations (like `(?i)` case-insensitive) before
    /// compilation. If optimization is enabled (default), the NFA is automatically
    /// optimized after Thompson construction.
    ///
    /// For access to runtime flags (unicode normalization, multiline, dotall),
    /// use [`compile_with_flags`] instead.
    pub fn compile(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        // Apply flag transformations before compilation
        let transform_result = apply_flags(regex);
        let nfa = self.compile_regex(&transform_result.regex)?;

        // Apply optimization if configured
        let mut nfa = if let Some(ref config) = self.optimization {
            let optimizer = NfaOptimizerChar::new(config.clone());
            let (optimized, _stats) = optimizer.optimize(nfa);
            optimized
        } else {
            nfa
        };

        // H9: Finalize CSR transition table
        nfa.finalize();
        Ok(nfa)
    }

    /// Compile a regex AST to an NFA with full flag support.
    ///
    /// This applies flag transformations and returns both the NFA and runtime
    /// settings extracted from flags like `(?u:NFC)`, `(?m)`, and `(?s)`.
    pub fn compile_with_flags(&mut self, regex: &Regex) -> ParseResult<CompileResultChar> {
        // Apply flag transformations before compilation
        let transform_result = apply_flags(regex);
        let nfa = self.compile_regex(&transform_result.regex)?;

        // Apply optimization if configured
        let nfa = if let Some(ref config) = self.optimization {
            let optimizer = NfaOptimizerChar::new(config.clone());
            let (optimized, _stats) = optimizer.optimize(nfa);
            optimized
        } else {
            nfa
        };

        // H9: Finalize CSR transition table
        let mut nfa = nfa;
        nfa.finalize();

        Ok(CompileResultChar {
            nfa,
            unicode_normalization: transform_result.unicode_normalization,
            multiline: transform_result.multiline,
            dotall: transform_result.dotall,
            local_distance: transform_result.local_distance,
        })
    }

    /// Compile a rewrite rule.
    ///
    /// The source pattern NFA is automatically optimized if optimization is enabled.
    pub fn compile_rewrite(&mut self, regex: &Regex) -> ParseResult<CompiledRewriteChar> {
        match regex {
            Regex::RewriteRule {
                pattern,
                replacement,
                context,
                weight,
            } => {
                // Compile and optimize the source pattern
                let source = self.compile_regex(pattern)?;
                let mut source = if let Some(ref config) = self.optimization {
                    let optimizer = NfaOptimizerChar::new(config.clone());
                    let (optimized, _stats) = optimizer.optimize(source);
                    optimized
                } else {
                    source
                };
                // H9: Finalize CSR transition table
                source.finalize();

                let replacement_chars = self.regex_to_literal(replacement)?;

                let (left_context, right_context) = if let Some(ctx) = context {
                    (
                        ctx.left
                            .as_ref()
                            .map(|l| self.compile_context_expr(l))
                            .transpose()?,
                        ctx.right
                            .as_ref()
                            .map(|r| self.compile_context_expr(r))
                            .transpose()?,
                    )
                } else {
                    (None, None)
                };

                Ok(CompiledRewriteChar {
                    source,
                    replacement: replacement_chars,
                    left_context,
                    right_context,
                    weight: *weight,
                })
            }
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidRewriteRule("expected rewrite rule".to_string()),
                Position::start(),
            )),
        }
    }

    /// Compile a context expression to a context pattern.
    fn compile_context_expr(&mut self, expr: &ContextExpr) -> ParseResult<ContextPatternChar> {
        match expr {
            ContextExpr::Pattern(regex) => {
                let nfa = self.compile_regex(regex)?;
                // Apply optimization if configured
                let mut nfa = if let Some(ref config) = self.optimization {
                    let optimizer = NfaOptimizerChar::new(config.clone());
                    let (optimized, _stats) = optimizer.optimize(nfa);
                    optimized
                } else {
                    nfa
                };
                // H9: Finalize CSR transition table
                nfa.finalize();
                Ok(ContextPatternChar::Nfa(nfa))
            }
            ContextExpr::WordBoundary => {
                // Word boundary is represented as a boundary kind
                // The actual direction (start/end) is determined by position in the context
                Ok(ContextPatternChar::Boundary(BoundaryKind::WordStart))
            }
            ContextExpr::And(a, b) => {
                let left = self.compile_context_expr(a)?;
                let right = self.compile_context_expr(b)?;
                Ok(ContextPatternChar::And(Box::new(left), Box::new(right)))
            }
            ContextExpr::Or(a, b) => {
                let left = self.compile_context_expr(a)?;
                let right = self.compile_context_expr(b)?;
                Ok(ContextPatternChar::Or(Box::new(left), Box::new(right)))
            }
            ContextExpr::Not(inner) => {
                let pattern = self.compile_context_expr(inner)?;
                Ok(ContextPatternChar::Not(Box::new(pattern)))
            }
        }
    }

    /// Compile a regex to NFA.
    ///
    /// This method dispatches to either the recursive or trampolined
    /// implementation based on the `use_trampolining` flag.
    fn compile_regex(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        if self.use_trampolining {
            self.compile_regex_trampolined(regex)
        } else {
            self.compile_regex_recursive(regex)
        }
    }

    /// Recursive implementation of regex compilation.
    ///
    /// This is the default, faster implementation that uses the call stack.
    /// For deeply nested patterns, use `with_trampolining()` instead.
    #[allow(deprecated)]
    fn compile_regex_recursive(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        match regex {
            Regex::Empty => Ok(self.builder.epsilon()),
            Regex::Char(c) => Ok(self.builder.single_char(*c)),
            Regex::CharClass(class) => Ok(self.builder.char_class(class.clone())),
            Regex::Any => Ok(self.builder.any_char()),
            Regex::Concat(a, b) => {
                let nfa_a = self.compile_regex_recursive(a)?;
                let nfa_b = self.compile_regex_recursive(b)?;
                Ok(self.builder.concatenate(nfa_a, nfa_b))
            }
            Regex::Alt(a, b) => {
                let nfa_a = self.compile_regex_recursive(a)?;
                let nfa_b = self.compile_regex_recursive(b)?;
                Ok(self.builder.alternation(nfa_a, nfa_b))
            }
            Regex::Star(inner) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.kleene_star(nfa))
            }
            Regex::Plus(inner) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.kleene_plus(nfa))
            }
            Regex::Optional(inner) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.optional(nfa))
            }
            Regex::RepeatExact(inner, n) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.repeat_exact(nfa, *n))
            }
            Regex::RepeatRange(inner, min, max) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.repeat_range(nfa, *min, *max))
            }
            // Legacy group (deprecated)
            Regex::Group(inner) => {
                // Groups don't affect NFA structure (capturing is not implemented yet)
                self.compile_regex_recursive(inner)
            }
            // Capturing group: (...)
            Regex::CapturingGroup(_, inner) => {
                // Capturing groups compile like regular groups for now
                // (capture semantics would be added in a future phase)
                self.compile_regex_recursive(inner)
            }
            // Non-capturing group: (?:...)
            Regex::NonCapturingGroup(inner) => {
                self.compile_regex_recursive(inner)
            }
            // Named group: (?<name>...)
            Regex::NamedGroup(_, inner) => {
                // Named groups compile like regular groups for now
                self.compile_regex_recursive(inner)
            }
            // Group reference: (?&name)
            Regex::GroupRef(name) => {
                // Group references require the named group's pattern to be available.
                // For now, return an error - in a future phase, we would expand
                // the group reference inline or implement proper subroutine semantics.
                Err(ParseError::new(
                    ParseErrorKind::InvalidGroupReference(format!(
                        "group reference (?&{}) cannot be compiled - subroutine expansion not yet implemented",
                        name
                    )),
                    Position::start(),
                ))
            }
            // Flags group: (?flags:...) or (?flags)
            Regex::FlagsGroup { inner, .. } => {
                // For now, ignore flags and compile the inner pattern if present
                // Flag-based matching (case-insensitive, etc.) would be implemented
                // by transforming the AST before compilation in a future phase
                match inner {
                    Some(inner_regex) => self.compile_regex_recursive(inner_regex),
                    None => Ok(self.builder.epsilon()),
                }
            }
            Regex::WordBoundary => {
                // Word boundary is handled specially in context matching
                // For now, return epsilon (will be handled by context logic)
                Ok(self.builder.epsilon())
            }
            // Anchor assertions (zero-width)
            Regex::StartOfLine => Ok(self.builder.start_of_line()),
            Regex::EndOfLine => Ok(self.builder.end_of_line()),
            Regex::StartOfInput => Ok(self.builder.start_of_input()),
            Regex::EndOfInput => Ok(self.builder.end_of_input()),
            Regex::EndOfInputStrict => Ok(self.builder.end_of_input_strict()),
            Regex::RewriteRule { pattern, .. } => {
                // When compiling a rewrite rule as a pattern, just compile the pattern part
                self.compile_regex_recursive(pattern)
            }
        }
    }

    /// Trampolined (iterative) implementation of regex compilation.
    ///
    /// This implementation uses an explicit heap-allocated work stack instead
    /// of the call stack, making it safe for deeply nested patterns that would
    /// otherwise cause stack overflow.
    #[allow(deprecated)]
    fn compile_regex_trampolined(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        let mut work_stack: Vec<CompileWork<'_>> = vec![CompileWork::Compile(regex)];
        let mut value_stack: Vec<NFAChar> = Vec::new();

        while let Some(work) = work_stack.pop() {
            match work {
                CompileWork::Compile(node) => {
                    self.process_compile_node(node, &mut work_stack, &mut value_stack)?;
                }
                CompileWork::DoConcatenate => {
                    let b = value_stack.pop().expect("second operand for concat");
                    let a = value_stack.pop().expect("first operand for concat");
                    value_stack.push(self.builder.concatenate(a, b));
                }
                CompileWork::DoAlternation => {
                    let b = value_stack.pop().expect("second operand for alt");
                    let a = value_stack.pop().expect("first operand for alt");
                    value_stack.push(self.builder.alternation(a, b));
                }
                CompileWork::DoStar => {
                    let inner = value_stack.pop().expect("operand for star");
                    value_stack.push(self.builder.kleene_star(inner));
                }
                CompileWork::DoPlus => {
                    let inner = value_stack.pop().expect("operand for plus");
                    value_stack.push(self.builder.kleene_plus(inner));
                }
                CompileWork::DoOptional => {
                    let inner = value_stack.pop().expect("operand for optional");
                    value_stack.push(self.builder.optional(inner));
                }
                CompileWork::DoRepeatExact(n) => {
                    let inner = value_stack.pop().expect("operand for repeat");
                    value_stack.push(self.builder.repeat_exact(inner, n));
                }
                CompileWork::DoRepeatRange(min, max) => {
                    let inner = value_stack.pop().expect("operand for repeat range");
                    value_stack.push(self.builder.repeat_range(inner, min, max));
                }
            }
        }

        value_stack.pop().ok_or_else(|| {
            ParseError::new(
                ParseErrorKind::InternalError("empty value stack after compilation".into()),
                Position::start(),
            )
        })
    }

    /// Process a single regex node for trampolined compilation.
    ///
    /// This method handles each regex variant by either:
    /// - Pushing the result directly to the value stack (base cases)
    /// - Pushing work items for children to the work stack (recursive cases)
    #[allow(deprecated)]
    fn process_compile_node<'a>(
        &mut self,
        node: &'a Regex,
        work_stack: &mut Vec<CompileWork<'a>>,
        value_stack: &mut Vec<NFAChar>,
    ) -> ParseResult<()> {
        match node {
            // === Base cases: push result directly ===
            Regex::Empty => value_stack.push(self.builder.epsilon()),
            Regex::Char(c) => value_stack.push(self.builder.single_char(*c)),
            Regex::CharClass(class) => value_stack.push(self.builder.char_class(class.clone())),
            Regex::Any => value_stack.push(self.builder.any_char()),
            Regex::WordBoundary => value_stack.push(self.builder.epsilon()),
            Regex::StartOfLine => value_stack.push(self.builder.start_of_line()),
            Regex::EndOfLine => value_stack.push(self.builder.end_of_line()),
            Regex::StartOfInput => value_stack.push(self.builder.start_of_input()),
            Regex::EndOfInput => value_stack.push(self.builder.end_of_input()),
            Regex::EndOfInputStrict => value_stack.push(self.builder.end_of_input_strict()),

            // === Binary operations: push operation, then children in reverse ===
            Regex::Concat(a, b) => {
                work_stack.push(CompileWork::DoConcatenate);
                work_stack.push(CompileWork::Compile(b));
                work_stack.push(CompileWork::Compile(a));
            }
            Regex::Alt(a, b) => {
                work_stack.push(CompileWork::DoAlternation);
                work_stack.push(CompileWork::Compile(b));
                work_stack.push(CompileWork::Compile(a));
            }

            // === Unary operations: push operation, then child ===
            Regex::Star(inner) => {
                work_stack.push(CompileWork::DoStar);
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::Plus(inner) => {
                work_stack.push(CompileWork::DoPlus);
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::Optional(inner) => {
                work_stack.push(CompileWork::DoOptional);
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::RepeatExact(inner, n) => {
                work_stack.push(CompileWork::DoRepeatExact(*n));
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::RepeatRange(inner, min, max) => {
                work_stack.push(CompileWork::DoRepeatRange(*min, *max));
                work_stack.push(CompileWork::Compile(inner));
            }

            // === Group wrappers: pass-through to inner ===
            Regex::Group(inner)
            | Regex::CapturingGroup(_, inner)
            | Regex::NonCapturingGroup(inner)
            | Regex::NamedGroup(_, inner) => {
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::FlagsGroup { inner: Some(inner_regex), .. } => {
                work_stack.push(CompileWork::Compile(inner_regex));
            }
            Regex::FlagsGroup { inner: None, .. } => {
                value_stack.push(self.builder.epsilon());
            }
            Regex::RewriteRule { pattern, .. } => {
                work_stack.push(CompileWork::Compile(pattern));
            }

            // === Error case ===
            Regex::GroupRef(name) => {
                return Err(ParseError::new(
                    ParseErrorKind::InvalidGroupReference(format!(
                        "group reference (?&{}) cannot be compiled - subroutine expansion not yet implemented",
                        name
                    )),
                    Position::start(),
                ));
            }
        }
        Ok(())
    }

    /// Convert a regex to a literal string (for replacement).
    #[allow(deprecated)]
    fn regex_to_literal(&self, regex: &Regex) -> ParseResult<Vec<char>> {
        match regex {
            Regex::Empty => Ok(Vec::new()),
            Regex::Char(c) => Ok(vec![*c]),
            Regex::Concat(a, b) => {
                let mut chars = self.regex_to_literal(a)?;
                chars.extend(self.regex_to_literal(b)?);
                Ok(chars)
            }
            // All group types extract the inner literal
            Regex::Group(inner)
            | Regex::CapturingGroup(_, inner)
            | Regex::NonCapturingGroup(inner)
            | Regex::NamedGroup(_, inner) => self.regex_to_literal(inner),
            // Flags group with inner pattern
            Regex::FlagsGroup { inner: Some(inner_regex), .. } => {
                self.regex_to_literal(inner_regex)
            }
            // Flags group without inner (inline flags)
            Regex::FlagsGroup { inner: None, .. } => Ok(Vec::new()),
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidRewriteRule(
                    "replacement must be a literal string".to_string(),
                ),
                Position::start(),
            )),
        }
    }
}

impl Default for NFACompilerChar {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte-level NFA compiler.
pub struct NFACompilerByte {
    builder: ThompsonBuilder,
}

impl NFACompilerByte {
    /// Create a new compiler.
    pub fn new() -> Self {
        Self {
            builder: ThompsonBuilder::new(),
        }
    }

    /// Compile a regex AST to an NFA.
    pub fn compile(&mut self, regex: &RegexByte) -> ParseResult<NFA> {
        let mut nfa = self.compile_regex(regex)?;
        // H9: Finalize CSR transition table
        nfa.finalize();
        Ok(nfa)
    }

    /// Compile a rewrite rule.
    pub fn compile_rewrite(&mut self, regex: &RegexByte) -> ParseResult<CompiledRewrite> {
        match regex {
            RegexByte::RewriteRule {
                pattern,
                replacement,
                context,
                weight,
            } => {
                let mut source = self.compile_regex(pattern)?;
                // H9: Finalize CSR transition table
                source.finalize();
                let replacement_bytes = self.regex_to_literal(replacement)?;

                let (left_context, right_context) = if let Some(ctx) = context {
                    (
                        ctx.left
                            .as_ref()
                            .map(|l| self.compile_context_expr(l))
                            .transpose()?,
                        ctx.right
                            .as_ref()
                            .map(|r| self.compile_context_expr(r))
                            .transpose()?,
                    )
                } else {
                    (None, None)
                };

                Ok(CompiledRewrite {
                    source,
                    replacement: replacement_bytes,
                    left_context,
                    right_context,
                    weight: *weight,
                })
            }
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidRewriteRule("expected rewrite rule".to_string()),
                Position::start(),
            )),
        }
    }

    /// Compile a context expression to a context pattern.
    fn compile_context_expr(&mut self, expr: &ContextExprByte) -> ParseResult<ContextPattern> {
        match expr {
            ContextExprByte::Pattern(regex) => {
                let mut nfa = self.compile_regex(regex)?;
                // H9: Finalize CSR transition table
                nfa.finalize();
                Ok(ContextPattern::Nfa(nfa))
            }
            ContextExprByte::WordBoundary => {
                Ok(ContextPattern::Boundary(BoundaryKind::WordStart))
            }
            ContextExprByte::And(a, b) => {
                let left = self.compile_context_expr(a)?;
                let right = self.compile_context_expr(b)?;
                Ok(ContextPattern::And(Box::new(left), Box::new(right)))
            }
            ContextExprByte::Or(a, b) => {
                let left = self.compile_context_expr(a)?;
                let right = self.compile_context_expr(b)?;
                Ok(ContextPattern::Or(Box::new(left), Box::new(right)))
            }
            ContextExprByte::Not(inner) => {
                let pattern = self.compile_context_expr(inner)?;
                Ok(ContextPattern::Not(Box::new(pattern)))
            }
        }
    }

    /// Compile a regex to NFA.
    #[allow(deprecated)]
    fn compile_regex(&mut self, regex: &RegexByte) -> ParseResult<NFA> {
        match regex {
            RegexByte::Empty => Ok(self.builder.epsilon()),
            RegexByte::Byte(b) => Ok(self.builder.single_byte(*b)),
            RegexByte::ByteClass(class) => Ok(self.builder.byte_class(class.clone())),
            RegexByte::Any => Ok(self.builder.any_byte()),
            RegexByte::Concat(a, b) => {
                let nfa_a = self.compile_regex(a)?;
                let nfa_b = self.compile_regex(b)?;
                Ok(self.builder.concatenate(nfa_a, nfa_b))
            }
            RegexByte::Alt(a, b) => {
                let nfa_a = self.compile_regex(a)?;
                let nfa_b = self.compile_regex(b)?;
                Ok(self.builder.alternation(nfa_a, nfa_b))
            }
            RegexByte::Star(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.kleene_star(nfa))
            }
            RegexByte::Plus(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.kleene_plus(nfa))
            }
            RegexByte::Optional(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.optional(nfa))
            }
            RegexByte::RepeatExact(inner, n) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.repeat_exact(nfa, *n))
            }
            RegexByte::RepeatRange(inner, min, max) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.repeat_range(nfa, *min, *max))
            }
            // Legacy group (deprecated)
            RegexByte::Group(inner) => {
                self.compile_regex(inner)
            }
            // Capturing group: (...)
            RegexByte::CapturingGroup(_, inner) => {
                self.compile_regex(inner)
            }
            // Non-capturing group: (?:...)
            RegexByte::NonCapturingGroup(inner) => {
                self.compile_regex(inner)
            }
            // Named group: (?<name>...)
            RegexByte::NamedGroup(_, inner) => {
                self.compile_regex(inner)
            }
            // Group reference: (?&name)
            RegexByte::GroupRef(name) => {
                Err(ParseError::new(
                    ParseErrorKind::InvalidGroupReference(format!(
                        "group reference (?&{}) cannot be compiled - subroutine expansion not yet implemented",
                        name
                    )),
                    Position::start(),
                ))
            }
            // Flags group: (?flags:...) or (?flags)
            RegexByte::FlagsGroup { inner, .. } => {
                match inner {
                    Some(inner_regex) => self.compile_regex(inner_regex),
                    None => Ok(self.builder.epsilon()),
                }
            }
            RegexByte::WordBoundary => {
                Ok(self.builder.epsilon())
            }
            // Anchor assertions (zero-width)
            RegexByte::StartOfLine => Ok(self.builder.start_of_line()),
            RegexByte::EndOfLine => Ok(self.builder.end_of_line()),
            RegexByte::StartOfInput => Ok(self.builder.start_of_input()),
            RegexByte::EndOfInput => Ok(self.builder.end_of_input()),
            RegexByte::EndOfInputStrict => Ok(self.builder.end_of_input_strict()),
            RegexByte::RewriteRule { pattern, .. } => {
                self.compile_regex(pattern)
            }
        }
    }

    /// Convert a regex to a literal byte string (for replacement).
    #[allow(deprecated)]
    fn regex_to_literal(&self, regex: &RegexByte) -> ParseResult<Vec<u8>> {
        match regex {
            RegexByte::Empty => Ok(Vec::new()),
            RegexByte::Byte(b) => Ok(vec![*b]),
            RegexByte::Concat(a, b) => {
                let mut bytes = self.regex_to_literal(a)?;
                bytes.extend(self.regex_to_literal(b)?);
                Ok(bytes)
            }
            // All group types extract the inner literal
            RegexByte::Group(inner)
            | RegexByte::CapturingGroup(_, inner)
            | RegexByte::NonCapturingGroup(inner)
            | RegexByte::NamedGroup(_, inner) => self.regex_to_literal(inner),
            // Flags group with inner pattern
            RegexByte::FlagsGroup { inner: Some(inner_regex), .. } => {
                self.regex_to_literal(inner_regex)
            }
            // Flags group without inner (inline flags)
            RegexByte::FlagsGroup { inner: None, .. } => Ok(Vec::new()),
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidRewriteRule(
                    "replacement must be a literal string".to_string(),
                ),
                Position::start(),
            )),
        }
    }
}

impl Default for NFACompilerByte {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::regex::{parse, parse_rule};

    #[test]
    fn test_compile_literal() {
        let regex = parse("phone").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(!nfa.accepts("fone"));
    }

    #[test]
    fn test_compile_alternation() {
        let regex = parse("ph|f").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("ph"));
        assert!(nfa.accepts("f"));
        assert!(!nfa.accepts("g"));
    }

    #[test]
    fn test_compile_group() {
        let regex = parse("(ph|f)one").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("fone"));
        assert!(!nfa.accepts("bone"));
    }

    #[test]
    fn test_compile_star() {
        let regex = parse("a*").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("aaa"));
        assert!(!nfa.accepts("b"));
    }

    #[test]
    fn test_compile_plus() {
        let regex = parse("a+").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(!nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("aaa"));
    }

    #[test]
    fn test_compile_optional() {
        let regex = parse("a?b").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("b"));
        assert!(nfa.accepts("ab"));
        assert!(!nfa.accepts("aab"));
    }

    #[test]
    fn test_compile_char_class() {
        let regex = parse("[aeiou]").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("e"));
        assert!(!nfa.accepts("b"));
    }

    #[test]
    fn test_compile_any() {
        let regex = parse("a.c").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("abc"));
        assert!(nfa.accepts("axc"));
        assert!(!nfa.accepts("ac"));
    }

    #[test]
    fn test_compile_repeat_exact() {
        let regex = parse("a{3}").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(!nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(!nfa.accepts("aaaa"));
    }

    #[test]
    fn test_compile_repeat_range() {
        let regex = parse("a{2,4}").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(!nfa.accepts("a"));
        assert!(nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(nfa.accepts("aaaa"));
        assert!(!nfa.accepts("aaaaa"));
    }

    #[test]
    fn test_compile_rewrite_rule_simple() {
        let regex = parse_rule("ph -> f").unwrap();
        let rewrite = compile_rewrite(&regex).unwrap();
        assert!(rewrite.source.accepts("ph"));
        assert_eq!(rewrite.replacement, vec!['f']);
        assert!(rewrite.left_context.is_none());
        assert!(rewrite.right_context.is_none());
    }

    #[test]
    fn test_compile_rewrite_rule_with_context() {
        let regex = parse_rule("c -> s / _[ei]").unwrap();
        let rewrite = compile_rewrite(&regex).unwrap();
        assert!(rewrite.source.accepts("c"));
        assert_eq!(rewrite.replacement, vec!['s']);
        assert!(rewrite.left_context.is_none());
        assert!(rewrite.right_context.is_some());
        let right = rewrite.right_context.unwrap();
        assert!(right.accepts("e"));
        assert!(right.accepts("i"));
        assert!(!right.accepts("a"));
    }

    #[test]
    fn test_compile_rewrite_rule_empty_replacement() {
        let regex = parse_rule("e -> / _#").unwrap();
        let rewrite = compile_rewrite(&regex).unwrap();
        assert!(rewrite.source.accepts("e"));
        assert!(rewrite.replacement.is_empty());
    }

    #[test]
    fn test_compile_complex_pattern() {
        let regex = parse("(ph|f)one[s]?").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("phones"));
        assert!(nfa.accepts("fone"));
        assert!(nfa.accepts("fones"));
        assert!(!nfa.accepts("bone"));
    }

    // Byte-level tests

    #[test]
    fn test_compile_bytes_literal() {
        let regex = crate::phonetic::regex::parse_bytes(b"phone").unwrap();
        let nfa = compile_bytes(&regex).unwrap();
        assert!(nfa.accepts(b"phone"));
        assert!(!nfa.accepts(b"fone"));
    }

    #[test]
    fn test_compile_bytes_alternation() {
        let regex = crate::phonetic::regex::parse_bytes(b"ph|f").unwrap();
        let nfa = compile_bytes(&regex).unwrap();
        assert!(nfa.accepts(b"ph"));
        assert!(nfa.accepts(b"f"));
    }

    #[test]
    fn test_compile_bytes_rewrite() {
        let regex = crate::phonetic::regex::parse_rule_bytes(b"ph -> f").unwrap();
        let rewrite = compile_rewrite_bytes(&regex).unwrap();
        assert!(rewrite.source.accepts(b"ph"));
        assert_eq!(rewrite.replacement, vec![b'f']);
    }

    // --- Anchor tests ---

    #[test]
    fn test_compile_start_of_line() {
        let regex = parse("^hello").unwrap();
        let nfa = compile(&regex).unwrap();
        // NFA should have anchor transition followed by literal
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_end_of_line() {
        let regex = parse("hello$").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_anchored_pattern() {
        let regex = parse("^hello$").unwrap();
        let nfa = compile(&regex).unwrap();
        // start_of_line + hello (5 chars) + end_of_line
        assert!(nfa.state_count() >= 4);
    }

    #[test]
    fn test_compile_start_of_input() {
        let regex = parse(r"\Ahello").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_end_of_input() {
        let regex = parse(r"hello\Z").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_strict_end_of_input() {
        let regex = parse(r"hello\z").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_multiline_flags() {
        // Pattern with multiline flag
        let regex = parse("(?m)^line$").unwrap();
        let nfa = compile(&regex).unwrap();
        // Should compile with anchor transitions
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_trampolining_api() {
        // Test that the trampolining API works correctly
        let mut compiler = NFACompilerChar::new();
        assert!(!compiler.is_trampolining(), "default should be recursive");

        compiler.set_trampolining(true);
        assert!(compiler.is_trampolining(), "should be trampolining after set");

        compiler.set_trampolining(false);
        assert!(!compiler.is_trampolining(), "should be recursive after unset");

        let compiler2 = NFACompilerChar::new().with_trampolining();
        assert!(compiler2.is_trampolining(), "builder should enable trampolining");
    }

    #[test]
    fn test_trampolined_matches_recursive_simple() {
        // Simple patterns should produce equivalent NFAs
        let patterns = ["a", "ab", "a|b", "a*", "a+", "a?", "[aeiou]"];

        for pattern in patterns {
            let regex = parse(pattern).expect("parse");

            // Compile with recursive (default)
            let nfa_recursive = compile(&regex).expect("recursive compile");

            // Compile with trampolining
            let mut compiler = NFACompilerChar::new().with_trampolining();
            let nfa_trampolined = compiler.compile(&regex).expect("trampolined compile");

            // Both should have the same state count
            assert_eq!(
                nfa_recursive.state_count(),
                nfa_trampolined.state_count(),
                "state count mismatch for pattern '{}'",
                pattern
            );

            // Both should accept/reject the same inputs
            let test_inputs = ["", "a", "b", "ab", "aaa", "aeiou"];
            for input in test_inputs {
                assert_eq!(
                    nfa_recursive.accepts(input),
                    nfa_trampolined.accepts(input),
                    "behavior mismatch for pattern '{}' on input '{}'",
                    pattern,
                    input
                );
            }
        }
    }

    #[test]
    fn test_trampolined_matches_recursive_complex() {
        // More complex patterns
        let patterns = [
            "(ab)+",
            "a{2,5}",
            "(?:a|b)*c",
            "(ph|f)one",
            "[a-z]+",
            "a{3}",
        ];

        for pattern in patterns {
            let regex = parse(pattern).expect("parse");

            let nfa_recursive = compile(&regex).expect("recursive compile");

            let mut compiler = NFACompilerChar::new().with_trampolining();
            let nfa_trampolined = compiler.compile(&regex).expect("trampolined compile");

            assert_eq!(
                nfa_recursive.state_count(),
                nfa_trampolined.state_count(),
                "state count mismatch for pattern '{}'",
                pattern
            );
        }
    }

    #[test]
    fn test_deep_nesting_with_trampolining() {
        // Build deeply nested pattern: ((((((((((a))))))))))
        // Note: The depth is limited by the parser's recursion, not the compiler.
        // The trampolining helps with compilation, but parsing is still recursive.
        let depth = 50; // Parser can handle this depth
        let mut pattern = "a".to_string();
        for _ in 0..depth {
            pattern = format!("({})", pattern);
        }

        let regex = parse(&pattern).expect("parse deeply nested pattern");

        // Compile with trampolining
        let mut compiler = NFACompilerChar::new().with_trampolining();
        let nfa = compiler.compile(&regex).expect("trampolined compile of deep pattern");

        // Verify it works correctly
        assert!(nfa.accepts("a"), "should accept 'a'");
        assert!(!nfa.accepts("b"), "should reject 'b'");
        assert!(!nfa.accepts("aa"), "should reject 'aa'");

        // Also verify recursive compilation works at this depth
        let nfa_recursive = compile(&regex).expect("recursive compile");
        assert_eq!(nfa.state_count(), nfa_recursive.state_count());
    }

    #[test]
    fn test_deep_alternation_with_trampolining() {
        // Build deeply nested alternation: (((a|b)|c)|d)...
        // Parser depth limited
        let depth = 40;
        let mut pattern = "a".to_string();
        for i in 0..depth {
            let c = char::from(b'a' + ((i + 1) % 26) as u8);
            pattern = format!("({}|{})", pattern, c);
        }

        let regex = parse(&pattern).expect("parse deeply nested alternation");

        let mut compiler = NFACompilerChar::new().with_trampolining();
        let nfa = compiler.compile(&regex).expect("trampolined compile of deep alternation");

        // Should accept any single letter
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("b"));
        assert!(nfa.accepts("c"));

        // Verify equivalence with recursive
        let nfa_recursive = compile(&regex).expect("recursive compile");
        assert_eq!(nfa.state_count(), nfa_recursive.state_count());
    }

    #[test]
    fn test_deep_concat_with_trampolining() {
        // Build deeply concatenated pattern via groups
        // Parser depth limited
        let depth = 100;
        let mut pattern = String::new();
        for _ in 0..depth {
            pattern.push_str("(?:a)");
        }

        let regex = parse(&pattern).expect("parse deeply concatenated pattern");

        let mut compiler = NFACompilerChar::new().with_trampolining();
        let nfa = compiler.compile(&regex).expect("trampolined compile of deep concat");

        // Should accept exactly 'depth' 'a' characters
        let expected: String = std::iter::repeat('a').take(depth).collect();
        assert!(nfa.accepts(&expected), "should accept {} 'a' chars", depth);

        let too_short: String = std::iter::repeat('a').take(depth - 1).collect();
        assert!(!nfa.accepts(&too_short), "should reject {} 'a' chars", depth - 1);

        // Verify equivalence with recursive
        let nfa_recursive = compile(&regex).expect("recursive compile");
        assert_eq!(nfa.state_count(), nfa_recursive.state_count());
    }

    #[test]
    fn test_recursive_mode_still_works() {
        // Verify recursive mode (default) still works for normal patterns
        let patterns = ["hello", "world|earth", "a+b*c?", "[0-9]+"];

        for pattern in patterns {
            let regex = parse(pattern).expect("parse");

            // Default is recursive
            let nfa = compile(&regex).expect("recursive compile");
            assert!(nfa.state_count() > 0);
        }
    }
}
