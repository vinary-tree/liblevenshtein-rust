//! Compile-time regex NFA generation macros for liblevenshtein.
//!
//! This crate provides proc macros that compile regex patterns to NFAs at compile time,
//! embedding the serialized NFA directly into your binary. This eliminates runtime
//! parsing and compilation overhead.
//!
//! # Usage
//!
//! ```rust,ignore
//! use liblevenshtein::llre;
//!
//! // Compile-time regex - NFA embedded in binary
//! let pattern = llre!(r"^hello$");
//! assert!(pattern.matches("hello"));
//!
//! // With multiline flag
//! let multiline = llre!(r"(?m)^line$");
//!
//! // From .llre file
//! let email = llre_file!("patterns/email.llre");
//! ```
//!
//! # Benefits
//!
//! | Aspect | Runtime (`compile()`) | Compile-time (`llre!`) |
//! |--------|----------------------|------------------------|
//! | Startup cost | Parse + compile NFA | Zero (already in binary) |
//! | Pattern errors | Runtime panic/error | Compile-time error |
//! | Binary size | Smaller | Larger (embedded NFA) |
//! | Use case | Dynamic patterns | Static patterns |

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitStr};

/// Compile a regex pattern to an NFA at compile time.
///
/// The NFA is serialized and embedded directly in the binary, eliminating
/// runtime parsing and compilation overhead. Pattern errors are caught at
/// compile time.
///
/// # Usage
///
/// ```rust,ignore
/// use liblevenshtein::llre;
///
/// // Simple pattern
/// let pattern = llre!(r"^hello$");
/// assert!(pattern.matches("hello"));
///
/// // With flags (using inline flag syntax)
/// let multiline = llre!(r"(?m)^line$");
/// let dotall = llre!(r"(?s).*");
/// let both = llre!(r"(?ms)^.*$");
/// ```
///
/// # Compile-time Errors
///
/// Invalid patterns will cause compile-time errors:
///
/// ```rust,ignore
/// // This won't compile:
/// let bad = llre!(r"[invalid");  // Compile error: unclosed character class
/// ```
///
/// # Performance
///
/// - **Zero startup cost**: NFA is already in binary
/// - **Pattern validation**: Errors caught at compile time
/// - **Trade-off**: Larger binary size vs. faster startup
#[proc_macro]
pub fn llre(input: TokenStream) -> TokenStream {
    let pattern_lit = parse_macro_input!(input as LitStr);
    let pattern_str = pattern_lit.value();

    // Parse the pattern at compile time
    let file = match liblevenshtein::phonetic::llre::parse_str(&pattern_str) {
        Ok(f) => f,
        Err(e) => {
            let msg = format!("Failed to parse regex pattern: {}", e);
            return syn::Error::new(pattern_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Compile to NFA at compile time
    let compiled = match liblevenshtein::phonetic::llre::compile(&file) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("Failed to compile regex to NFA: {}", e);
            return syn::Error::new(pattern_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Serialize to bytes at compile time
    let bytes = match liblevenshtein::phonetic::llre::to_bytes(&compiled) {
        Ok(b) => b,
        Err(e) => {
            let msg = format!("Failed to serialize NFA: {}", e);
            return syn::Error::new(pattern_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Generate code that embeds and deserializes the NFA
    let bytes_lit = syn::LitByteStr::new(&bytes, pattern_lit.span());

    quote! {
        {
            static BYTES: &[u8] = #bytes_lit;
            static COMPILED: std::sync::OnceLock<liblevenshtein::phonetic::llre::CompiledNFA> =
                std::sync::OnceLock::new();
            COMPILED.get_or_init(|| {
                liblevenshtein::phonetic::llre::from_bytes(BYTES)
                    .expect("Invalid embedded NFA - this is a bug in liblevenshtein-macros")
            })
        }
    }
    .into()
}

/// Compile a `.llre` file to an NFA at compile time.
///
/// The file is loaded, parsed, imports resolved, and compiled to an NFA at compile
/// time. The serialized NFA is embedded directly in the binary.
///
/// # Usage
///
/// ```rust,ignore
/// use liblevenshtein::llre_file;
///
/// // Load and compile at compile time
/// let email = llre_file!("patterns/email.llre");
/// assert!(email.matches("test@example.com"));
///
/// // With imports resolved
/// let phonetic = llre_file!("patterns/phonetic.llre");  // Can @import .llev files
/// ```
///
/// # File Format
///
/// ```text
/// # Example .llre file
/// @name "Email Validator"
/// @version "1.0"
///
/// # Import symbols from .llev files
/// @import "phonetic/symbols.llev"
///
/// # Global flags
/// @flags multiline
///
/// # The regex pattern
/// ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
/// ```
///
/// # Compile-time Errors
///
/// - File not found
/// - Parse errors in .llre file
/// - Import resolution failures
/// - NFA compilation errors
///
/// # Search Paths
///
/// Files are searched relative to `CARGO_MANIFEST_DIR` (the directory containing
/// your `Cargo.toml`).
#[proc_macro]
pub fn llre_file(input: TokenStream) -> TokenStream {
    let path_lit = parse_macro_input!(input as LitStr);
    let relative_path = path_lit.value();

    // Get the manifest directory for relative path resolution
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = std::path::Path::new(&manifest_dir).join(&relative_path);

    // Load the file at compile time
    let file = match liblevenshtein::phonetic::llre::load_file(&full_path) {
        Ok(f) => f,
        Err(e) => {
            let msg = format!(
                "Failed to load .llre file '{}': {}",
                full_path.display(),
                e
            );
            return syn::Error::new(path_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Compile to NFA at compile time
    let compiled = match liblevenshtein::phonetic::llre::compile(&file) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!(
                "Failed to compile .llre file '{}': {}",
                full_path.display(),
                e
            );
            return syn::Error::new(path_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Serialize to bytes at compile time
    let bytes = match liblevenshtein::phonetic::llre::to_bytes(&compiled) {
        Ok(b) => b,
        Err(e) => {
            let msg = format!("Failed to serialize NFA: {}", e);
            return syn::Error::new(path_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Generate code that embeds and deserializes the NFA
    let bytes_lit = syn::LitByteStr::new(&bytes, path_lit.span());
    let path_str = full_path.display().to_string();

    quote! {
        {
            // Include the file in dependency tracking so rebuilds happen on change
            const _: &str = include_str!(#path_str);

            static BYTES: &[u8] = #bytes_lit;
            static COMPILED: std::sync::OnceLock<liblevenshtein::phonetic::llre::CompiledNFA> =
                std::sync::OnceLock::new();
            COMPILED.get_or_init(|| {
                liblevenshtein::phonetic::llre::from_bytes(BYTES)
                    .expect("Invalid embedded NFA - this is a bug in liblevenshtein-macros")
            })
        }
    }
    .into()
}

/// Compile a regex pattern with symbol imports from a `.llev` file.
///
/// This macro allows you to import symbols from a `.llev` file and use them
/// in your pattern. Symbols are expanded at compile time.
///
/// # Usage
///
/// ```rust,ignore
/// use liblevenshtein::llre_with_symbols;
///
/// // Import symbols and use in pattern
/// let vowels = llre_with_symbols!(
///     import = "phonetic/symbols.llev",
///     pattern = r"$VOWEL+"  // $VOWEL expanded from imported symbols
/// );
/// assert!(vowels.matches("aeiou"));
/// ```
///
/// # Symbol Expansion
///
/// - `$SYMBOL` - Expanded outside character classes
/// - Inside character classes, use POSIX-style `[:CLASS:]` syntax
///
/// # Compile-time Errors
///
/// - Import file not found
/// - Undefined symbol references
/// - Parse/compile errors
#[proc_macro]
pub fn llre_with_symbols(input: TokenStream) -> TokenStream {
    // Parse the import = "...", pattern = "..." syntax
    let args = parse_macro_input!(input as MacroArgs);

    // Get the manifest directory for relative path resolution
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let import_path = std::path::Path::new(&manifest_dir).join(&args.import_path);

    // Load the .llev file for symbols
    let llev_file = match liblevenshtein::phonetic::llev::load_file(&import_path) {
        Ok(f) => f,
        Err(e) => {
            let msg = format!(
                "Failed to load symbol file '{}': {}",
                import_path.display(),
                e
            );
            return syn::Error::new(args.import_span, msg)
                .to_compile_error()
                .into();
        }
    };

    // Build an LLreFile with the symbols and pattern
    use liblevenshtein::phonetic::llre::ast::SymbolTable;

    // Parse the pattern
    let regex = match liblevenshtein::phonetic::regex::parse(&args.pattern) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("Failed to parse pattern: {}", e);
            return syn::Error::new(args.pattern_span, msg)
                .to_compile_error()
                .into();
        }
    };

    // Build symbol table from llev file
    let mut symbol_table = SymbolTable::new();
    for symbol in &llev_file.symbols {
        if let liblevenshtein::phonetic::llev::Expression::CharClass { chars, .. } =
            &symbol.value
        {
            symbol_table.add_char_class(&symbol.name, chars.clone(), None);
        }
    }

    // Create the LLreFile
    let llre_file = liblevenshtein::phonetic::llre::ast::LLreFile {
        metadata: Default::default(),
        imports: Vec::new(),
        global_flags: Default::default(),
        pattern: regex,
        pattern_source: args.pattern.clone(),
        pattern_position: liblevenshtein::phonetic::llre::Position::new(0, 1, 1),
        source_file: None,
        resolved_imports: Vec::new(),
        symbol_table,
    };

    // Compile to NFA
    let compiled = match liblevenshtein::phonetic::llre::compile(&llre_file) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("Failed to compile pattern: {}", e);
            return syn::Error::new(args.pattern_span, msg)
                .to_compile_error()
                .into();
        }
    };

    // Serialize to bytes
    let bytes = match liblevenshtein::phonetic::llre::to_bytes(&compiled) {
        Ok(b) => b,
        Err(e) => {
            let msg = format!("Failed to serialize NFA: {}", e);
            return syn::Error::new(args.pattern_span, msg)
                .to_compile_error()
                .into();
        }
    };

    // Generate code
    let bytes_lit = syn::LitByteStr::new(&bytes, args.pattern_span);
    let import_str = import_path.display().to_string();

    quote! {
        {
            // Include files in dependency tracking
            const _: &str = include_str!(#import_str);

            static BYTES: &[u8] = #bytes_lit;
            static COMPILED: std::sync::OnceLock<liblevenshtein::phonetic::llre::CompiledNFA> =
                std::sync::OnceLock::new();
            COMPILED.get_or_init(|| {
                liblevenshtein::phonetic::llre::from_bytes(BYTES)
                    .expect("Invalid embedded NFA - this is a bug in liblevenshtein-macros")
            })
        }
    }
    .into()
}

/// Parsed arguments for `llre_with_symbols!` macro.
struct MacroArgs {
    import_path: String,
    import_span: proc_macro2::Span,
    pattern: String,
    pattern_span: proc_macro2::Span,
}

impl syn::parse::Parse for MacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Parse: import = "path", pattern = "regex"

        // Parse "import"
        let import_ident: syn::Ident = input.parse()?;
        if import_ident != "import" {
            return Err(syn::Error::new(
                import_ident.span(),
                "expected 'import'",
            ));
        }

        // Parse "="
        let _: syn::Token![=] = input.parse()?;

        // Parse import path string
        let import_lit: LitStr = input.parse()?;
        let import_path = import_lit.value();
        let import_span = import_lit.span();

        // Parse ","
        let _: syn::Token![,] = input.parse()?;

        // Parse "pattern"
        let pattern_ident: syn::Ident = input.parse()?;
        if pattern_ident != "pattern" {
            return Err(syn::Error::new(
                pattern_ident.span(),
                "expected 'pattern'",
            ));
        }

        // Parse "="
        let _: syn::Token![=] = input.parse()?;

        // Parse pattern string
        let pattern_lit: LitStr = input.parse()?;
        let pattern = pattern_lit.value();
        let pattern_span = pattern_lit.span();

        Ok(MacroArgs {
            import_path,
            import_span,
            pattern,
            pattern_span,
        })
    }
}

// ============================================================================
// LLev Macros
// ============================================================================

/// Compile inline `.llev` content to a RuleSetChar at compile time.
///
/// The rule set is serialized and embedded directly in the binary, eliminating
/// runtime parsing and compilation overhead. Parse errors are caught at compile time.
///
/// # Usage
///
/// ```rust,ignore
/// use liblevenshtein::llev;
///
/// // Simple rules
/// let rules = llev!(r#"
///     ph -> f;
///     gh -> ;
///     kn -> n / #_;
/// "#);
///
/// // Use the rules
/// let normalized = rules.apply("phone"); // "fone"
/// ```
///
/// # Compile-time Errors
///
/// Invalid rules will cause compile-time errors:
///
/// ```rust,ignore
/// // This won't compile:
/// let bad = llev!("-> f;");  // Compile error: empty pattern
/// ```
///
/// # Performance
///
/// - **Zero startup cost**: RuleSet is already serialized in binary
/// - **Rule validation**: Errors caught at compile time
/// - **Trade-off**: Larger binary size vs. faster startup
#[proc_macro]
pub fn llev(input: TokenStream) -> TokenStream {
    let content_lit = parse_macro_input!(input as LitStr);
    let content = content_lit.value();

    // Parse .llev content at compile time
    let file = match liblevenshtein::phonetic::llev::parse_str(&content) {
        Ok(f) => f,
        Err(e) => {
            let msg = format!("Failed to parse llev content: {}", e);
            return syn::Error::new(content_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Compile to RuleSetChar at compile time
    let ruleset = match liblevenshtein::phonetic::llev::RuleSetChar::from_llev(&file) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("Failed to compile llev content: {}", e);
            return syn::Error::new(content_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Serialize to bytes at compile time
    let bytes = match liblevenshtein::phonetic::llev::to_bytes_char(&ruleset) {
        Ok(b) => b,
        Err(e) => {
            let msg = format!("Failed to serialize RuleSetChar: {}", e);
            return syn::Error::new(content_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    let bytes_lit = syn::LitByteStr::new(&bytes, content_lit.span());

    quote! {
        {
            static BYTES: &[u8] = #bytes_lit;
            static RULESET: std::sync::OnceLock<liblevenshtein::phonetic::llev::RuleSetChar> =
                std::sync::OnceLock::new();
            RULESET.get_or_init(|| {
                liblevenshtein::phonetic::llev::from_bytes_char(BYTES)
                    .expect("Invalid embedded RuleSetChar - this is a bug in liblevenshtein-macros")
            })
        }
    }
    .into()
}

/// Load and compile a `.llev` file to a RuleSetChar at compile time.
///
/// The file is loaded, parsed, and compiled to a RuleSetChar at compile time.
/// The serialized rule set is embedded directly in the binary.
///
/// # Usage
///
/// ```rust,ignore
/// use liblevenshtein::llev_file;
///
/// // Load and compile at compile time
/// let english = llev_file!("data/rules/english/zompist.llev");
/// let normalized = english.apply("phone"); // "fone"
/// ```
///
/// # File Format
///
/// ```text
/// # Example .llev file
/// @name "English Rules"
/// @version "1.0"
///
/// # Define reusable symbols
/// @define FRONT_VOWEL = [ei]
///
/// # Rules
/// ph -> f;
/// c -> s / _$FRONT_VOWEL;
/// c -> k;
/// ```
///
/// # Compile-time Errors
///
/// - File not found
/// - Parse errors in .llev file
/// - Rule compilation errors
///
/// # Search Paths
///
/// Files are searched relative to `CARGO_MANIFEST_DIR` (the directory containing
/// your `Cargo.toml`).
#[proc_macro]
pub fn llev_file(input: TokenStream) -> TokenStream {
    let path_lit = parse_macro_input!(input as LitStr);
    let relative_path = path_lit.value();

    // Get the manifest directory for relative path resolution
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let full_path = std::path::Path::new(&manifest_dir).join(&relative_path);

    // Load the file at compile time
    let file = match liblevenshtein::phonetic::llev::load_file(&full_path) {
        Ok(f) => f,
        Err(e) => {
            let msg = format!(
                "Failed to load .llev file '{}': {}",
                full_path.display(),
                e
            );
            return syn::Error::new(path_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Compile to RuleSetChar at compile time
    let ruleset = match liblevenshtein::phonetic::llev::RuleSetChar::from_llev(&file) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!(
                "Failed to compile .llev file '{}': {}",
                full_path.display(),
                e
            );
            return syn::Error::new(path_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Serialize to bytes at compile time
    let bytes = match liblevenshtein::phonetic::llev::to_bytes_char(&ruleset) {
        Ok(b) => b,
        Err(e) => {
            let msg = format!("Failed to serialize RuleSetChar: {}", e);
            return syn::Error::new(path_lit.span(), msg)
                .to_compile_error()
                .into();
        }
    };

    // Generate code that embeds and deserializes the RuleSetChar
    let bytes_lit = syn::LitByteStr::new(&bytes, path_lit.span());
    let path_str = full_path.display().to_string();

    quote! {
        {
            // Include the file in dependency tracking so rebuilds happen on change
            const _: &str = include_str!(#path_str);

            static BYTES: &[u8] = #bytes_lit;
            static RULESET: std::sync::OnceLock<liblevenshtein::phonetic::llev::RuleSetChar> =
                std::sync::OnceLock::new();
            RULESET.get_or_init(|| {
                liblevenshtein::phonetic::llev::from_bytes_char(BYTES)
                    .expect("Invalid embedded RuleSetChar - this is a bug in liblevenshtein-macros")
            })
        }
    }
    .into()
}
