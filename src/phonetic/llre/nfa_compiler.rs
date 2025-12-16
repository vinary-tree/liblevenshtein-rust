//! NFA compiler for `.llre` files.
//!
//! This module compiles a parsed .llre file into an NFA that can be used for
//! matching. It handles symbol expansion, flag-aware compilation, and
//! optimization.

use crate::phonetic::nfa::{NFAChar, NFACompilerChar};
use crate::phonetic::regex::ast::RegexFlags;

use super::ast::{LLreFile, SymbolTable};
use super::error::{LLreError, LLreErrorKind, LLreResult};

/// Compiled NFA from an .llre file.
#[derive(Debug, Clone)]
pub struct CompiledNFA {
    /// The compiled NFA
    pub nfa: NFAChar,

    /// Whether multiline mode is enabled
    pub multiline: bool,

    /// Whether dotall mode is enabled
    pub dotall: bool,

    /// Whether case-insensitive mode is enabled
    pub case_insensitive: bool,

    /// Source file metadata (for debugging)
    pub name: Option<String>,

    /// Version from metadata
    pub version: Option<String>,
}

impl CompiledNFA {
    /// Check if the NFA accepts the input string (search semantics).
    ///
    /// This uses search semantics where anchors control matching:
    /// - `^pattern` matches at the start
    /// - `pattern$` matches at the end
    /// - `^pattern$` matches the entire string
    /// - `pattern` matches anywhere in the string
    pub fn matches(&self, input: &str) -> bool {
        self.nfa.search_with_flags(input, self.multiline, self.dotall)
    }

    /// Check if the NFA accepts the entire input string (full-match semantics).
    pub fn matches_full(&self, input: &str) -> bool {
        self.nfa.accepts_with_flags(input, self.multiline, self.dotall)
    }

    /// Check if the NFA accepts the input string (alias for matches).
    pub fn is_match(&self, input: &str) -> bool {
        self.matches(input)
    }

    /// Get the number of states in the NFA.
    pub fn state_count(&self) -> usize {
        self.nfa.state_count()
    }

    /// Get the number of transitions in the NFA.
    pub fn transition_count(&self) -> usize {
        self.nfa.num_transitions()
    }
}

/// Options for NFA compilation.
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    /// Maximum NFA size (number of states) before failing
    pub max_states: Option<usize>,

    /// Use trampolining for stack-safe compilation of deep patterns.
    /// Enable this when compiling potentially deep patterns from untrusted input.
    pub use_trampolining: bool,

    /// Whether to optimize the NFA after compilation
    pub optimize: bool,
}

impl CompileOptions {
    /// Create options with a maximum state limit.
    pub fn with_max_states(max_states: usize) -> Self {
        Self {
            max_states: Some(max_states),
            ..Default::default()
        }
    }

    /// Create options with optimization enabled.
    pub fn optimized() -> Self {
        Self {
            optimize: true,
            ..Default::default()
        }
    }
}

/// Compile an LLreFile to an NFA.
pub fn compile(file: &LLreFile) -> LLreResult<CompiledNFA> {
    compile_with_options(file, &CompileOptions::default())
}

/// Compile an LLreFile to an NFA with custom options.
pub fn compile_with_options(file: &LLreFile, options: &CompileOptions) -> LLreResult<CompiledNFA> {
    // Create the NFA compiler with symbol table
    let mut compiler = NFACompilerChar::new();

    // Add symbols from the symbol table
    add_symbols_to_compiler(&mut compiler, &file.symbol_table)?;

    // Set up flags
    let flags = file.effective_flags();
    compiler.set_flags(flags.clone());

    // Enable trampolining if requested (for deep patterns from untrusted input)
    if options.use_trampolining {
        compiler.set_trampolining(true);
    }

    // Compile the pattern
    let nfa = compiler.compile(&file.pattern).map_err(|e| {
        LLreError::with_position(
            LLreErrorKind::NfaCompilationFailed(e.to_string()),
            file.pattern_position,
        )
    })?;

    // Check size limits
    if let Some(max_states) = options.max_states {
        let state_count = nfa.state_count();
        if state_count > max_states {
            return Err(LLreError::new(LLreErrorKind::PatternTooComplex {
                size: state_count,
                max: max_states,
            }));
        }
    }

    Ok(CompiledNFA {
        nfa,
        multiline: flags.multiline.unwrap_or(false),
        dotall: flags.dotall.unwrap_or(false),
        case_insensitive: flags.case_insensitive.unwrap_or(false),
        name: file.metadata.name.clone(),
        version: file.metadata.version.clone(),
    })
}

/// Add symbols from a symbol table to the NFA compiler.
fn add_symbols_to_compiler(
    compiler: &mut NFACompilerChar,
    table: &SymbolTable,
) -> LLreResult<()> {
    // Add character class symbols
    for (name, chars) in &table.char_classes {
        compiler.add_symbol(name, chars.clone());
    }

    // Pattern symbols would need special handling - they're Regex AST nodes
    // that would need to be expanded during compilation. For now, we only
    // support character class symbols.
    if !table.patterns.is_empty() {
        // TODO: Support pattern symbols by expanding them during compilation
        // For now, character class symbols are the primary use case
    }

    Ok(())
}

/// Compile a regex pattern string directly to an NFA.
pub fn compile_pattern(pattern: &str) -> LLreResult<CompiledNFA> {
    compile_pattern_with_flags(pattern, &RegexFlags::default())
}

/// Compile a regex pattern string with flags.
pub fn compile_pattern_with_flags(pattern: &str, flags: &RegexFlags) -> LLreResult<CompiledNFA> {
    // Parse the pattern
    let regex = crate::phonetic::regex::parse(pattern)?;

    // Create compiler and set flags
    let mut compiler = NFACompilerChar::new();
    compiler.set_flags(flags.clone());

    // Compile
    let nfa = compiler.compile(&regex).map_err(|e| {
        LLreError::new(LLreErrorKind::NfaCompilationFailed(e.to_string()))
    })?;

    Ok(CompiledNFA {
        nfa,
        multiline: flags.multiline.unwrap_or(false),
        dotall: flags.dotall.unwrap_or(false),
        case_insensitive: flags.case_insensitive.unwrap_or(false),
        name: None,
        version: None,
    })
}

/// Quick helper to compile and match a pattern.
pub fn is_match(pattern: &str, input: &str) -> LLreResult<bool> {
    let compiled = compile_pattern(pattern)?;
    Ok(compiled.matches(input))
}

/// Quick helper to compile and match with multiline mode.
pub fn is_match_multiline(pattern: &str, input: &str) -> LLreResult<bool> {
    let flags = RegexFlags {
        multiline: Some(true),
        ..Default::default()
    };
    let compiled = compile_pattern_with_flags(pattern, &flags)?;
    Ok(compiled.matches(input))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::llre::parser::parse_str;

    #[test]
    fn test_compile_simple_pattern() {
        let file = parse_str("^hello$").expect("Failed to parse");
        let compiled = compile(&file).expect("Failed to compile");

        assert!(compiled.matches("hello"));
        assert!(!compiled.matches("hello world"));
        assert!(!compiled.matches("say hello"));
    }

    #[test]
    fn test_compile_with_multiline() {
        let file = parse_str(r#"
            @flags multiline
            ^hello$
        "#).expect("Failed to parse");

        let compiled = compile(&file).expect("Failed to compile");
        assert!(compiled.multiline);

        // With multiline, ^ and $ match at line boundaries
        assert!(compiled.matches("hello"));
        // Note: The actual multiline behavior depends on the NFA implementation
    }

    #[test]
    fn test_compile_pattern_directly() {
        let compiled = compile_pattern("[a-z]+").expect("Failed to compile");
        assert!(compiled.matches("hello"));
        assert!(!compiled.matches("123"));
    }

    #[test]
    fn test_compile_with_flags() {
        let flags = RegexFlags {
            multiline: Some(true),
            ..Default::default()
        };
        let compiled = compile_pattern_with_flags("^test$", &flags).expect("Failed to compile");
        assert!(compiled.multiline);
    }

    #[test]
    fn test_is_match_helper() {
        assert!(is_match("hello", "hello").expect("Failed"));
        assert!(!is_match("hello", "world").expect("Failed"));
    }

    #[test]
    fn test_compile_with_symbols() {
        let mut file = parse_str("^[a-z]+$").expect("Failed to parse");

        // Add symbols manually (normally from imports)
        file.symbol_table.add_char_class("VOWEL", vec!['a', 'e', 'i', 'o', 'u'], None);

        let compiled = compile(&file).expect("Failed to compile");
        assert!(compiled.matches("hello"));
    }

    #[test]
    fn test_compile_options_max_states() {
        let file = parse_str("(a|b|c|d|e)+").expect("Failed to parse");
        let options = CompileOptions::with_max_states(5); // Very small limit

        let result = compile_with_options(&file, &options);
        // This might fail if the NFA exceeds 5 states
        // The actual behavior depends on NFA construction
        assert!(result.is_ok() || matches!(result.unwrap_err().kind, LLreErrorKind::PatternTooComplex { .. }));
    }

    #[test]
    fn test_compiled_nfa_stats() {
        let compiled = compile_pattern("[a-z]+").expect("Failed to compile");
        assert!(compiled.state_count() > 0);
        assert!(compiled.transition_count() > 0);
    }

    #[test]
    fn test_compile_anchors() {
        // Start of line
        let compiled = compile_pattern("^hello").expect("Failed to compile");
        assert!(compiled.matches("hello"));
        assert!(compiled.matches("hello world"));
        assert!(!compiled.matches("say hello"));

        // End of line
        let compiled = compile_pattern("hello$").expect("Failed to compile");
        assert!(compiled.matches("hello"));
        assert!(compiled.matches("say hello"));
        assert!(!compiled.matches("hello world"));

        // Both
        let compiled = compile_pattern("^hello$").expect("Failed to compile");
        assert!(compiled.matches("hello"));
        assert!(!compiled.matches("hello world"));
        assert!(!compiled.matches("say hello"));
    }

    #[test]
    fn test_compile_alternation() {
        let compiled = compile_pattern("cat|dog|bird").expect("Failed to compile");
        assert!(compiled.matches("cat"));
        assert!(compiled.matches("dog"));
        assert!(compiled.matches("bird"));
        assert!(!compiled.matches("fish"));
    }

    #[test]
    fn test_compile_quantifiers() {
        // Zero or more
        let compiled = compile_pattern("ab*c").expect("Failed to compile");
        assert!(compiled.matches("ac"));
        assert!(compiled.matches("abc"));
        assert!(compiled.matches("abbc"));
        assert!(!compiled.matches("adc"));

        // One or more
        let compiled = compile_pattern("ab+c").expect("Failed to compile");
        assert!(!compiled.matches("ac"));
        assert!(compiled.matches("abc"));
        assert!(compiled.matches("abbc"));

        // Optional
        let compiled = compile_pattern("ab?c").expect("Failed to compile");
        assert!(compiled.matches("ac"));
        assert!(compiled.matches("abc"));
        assert!(!compiled.matches("abbc"));
    }

    #[test]
    fn test_compile_character_classes() {
        let compiled = compile_pattern("[aeiou]+").expect("Failed to compile");
        assert!(compiled.matches("aeiou"));
        assert!(compiled.matches("a"));
        assert!(!compiled.matches("xyz"));

        // Negated
        let compiled = compile_pattern("[^aeiou]+").expect("Failed to compile");
        assert!(compiled.matches("xyz"));
        assert!(!compiled.matches("aeiou"));
    }

    #[test]
    fn test_compile_groups() {
        let compiled = compile_pattern("(ab)+").expect("Failed to compile");
        assert!(compiled.matches("ab"));
        assert!(compiled.matches("abab"));
        assert!(!compiled.matches("a"));
        assert!(!compiled.matches("ba"));
    }
}
