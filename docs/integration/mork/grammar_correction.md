# Phase D: Grammar Correction via MORK Pattern Matching

**Last Updated**: 2025-12-21
**Version**: v0.8.0
**Status**: PROPOSED

> ⚠️ **PROPOSAL NOTICE**: This document describes a **proposed** `src/grammar/` module for CFG-based grammatical error correction. This structure is a design specification for future implementation.
>
> **Current Implementation**: liblevenshtein v0.8.0 provides phonetic rules (`english::base()`, `llev!` macro) that handle spelling-level corrections, but not grammar-level corrections.

This document describes how MORK's pattern matching capabilities can serve as the rule engine for CFG-based grammatical error correction, building on the WFST infrastructure from Phases A-C.

## Overview

**Goal**: Use MORK's `transform_multi_multi_()` function as the rule engine for CFG-based error correction, representing grammar rules as pattern/template pairs.

**Key Insight**: MORK's pattern matching (`match2()`, `unify()`) directly implements the core operations needed for grammar rule application, with efficient lattice handling via `query_multi_i()`.

---

## Current v0.8.0 Capabilities

Before implementing full grammar correction, liblevenshtein v0.8.0 provides:

### What's Available Now (Spelling-Level)

| Feature | API | Description |
|---------|-----|-------------|
| Phonetic rules | `english::base()` | 62 orthographic spelling rules |
| Homophone handling | `english::homophones()` | Homophone pair rules |
| Text-speak expansion | `english::text_speak()` | Text-speak to standard spelling |
| Rule compilation | `llev!` macro | Compile-time phonetic rules |
| NFA composition | `ProductAutomatonChar` | Phonetic + edit distance |

### Example: Spelling Correction (Current)

```rust
use liblevenshtein::phonetic::rules::english;
use liblevenshtein::phonetic::nfa::ProductAutomatonChar;
use liblevenshtein::phonetic::verified::rules_to_nfa_char;

// Phonetic rules for spelling correction
let rules = english::base();
let nfa = rules_to_nfa_char(&rules.rules);
let product = ProductAutomatonChar::new(nfa, 2);

// Corrects spelling: "fone" → "phone", "colour" → "color"
if product.accepts("phone") {
    println!("Phonetic match found");
}
```

### Gap: Grammar-Level Correction

The current implementation handles **spelling/phonetic** corrections but not **grammar** corrections like:
- Article errors: "a apple" → "an apple"
- Subject-verb agreement: "he go" → "he goes"
- Missing words: "I going" → "I am going"

Phase D proposes extending to grammar correction via MORK pattern matching.

---

## Architecture

### Three-Tier Pipeline with MORK

```
┌─────────────────────────────────────────────────────────┐
│ Tier 1: Lexical (liblevenshtein)                        │
│   Input: "teh quik brwn fox"                            │
│   FST/Levenshtein → Word lattice                        │
│   Output: Lattice with ~K candidates per position       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Tier 2: Syntactic (MORK)                                │
│   Input: Word lattice as MORK expressions               │
│   - CFG rules compiled to pattern/template pairs        │
│   - query_multi_i() matches against lattice             │
│   - transform_multi_multi_() applies corrections        │
│   Output: Valid parse forest + corrections              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Tier 3: Semantic (Type checker / LLM)                   │
│   Final ranking and validation                          │
└─────────────────────────────────────────────────────────┘
```

### Why MORK for Grammar Correction?

| MORK Feature | Grammar Correction Use |
|--------------|------------------------|
| `match2()` | Recursive structural matching of AST nodes |
| `unify()` | Variable binding for feature agreement |
| `query_multi_i()` | O(K×N) lattice processing (not O(K^N)) |
| `transform_multi_multi_()` | Apply pattern→template corrections |
| De Bruijn indices | Feature propagation without name lookup |
| PathMap backend | Efficient rule storage and lookup |

---

## CFG as Pattern/Template Pairs

### Rule Encoding

CFG production rules map directly to MORK's pattern/template mechanism:

```metta
; CFG Rule: S → NP VP
; Encoded as pattern matching rule
Pattern:  (s (np ?NP) (vp ?VP))
Template: (sentence ?NP ?VP)

; Error Production: Article error (a → an before vowel)
Pattern:  (np (dt "a") (n ?N))
Constraint: (is_vowel_initial ?N)
Template: (np (dt "an") (n ?N))
Cost: 0.5
```

### Error Grammar Example

```metta
; Subject-Verb Agreement Error
; Pattern: singular subject with plural verb
Pattern:  (s (np (det ?D) (n ?N :num singular))
             (vp (v ?V :num plural) ?Rest))
Template: (s (np (det ?D) (n ?N :num singular))
             (vp (v (correct_agreement ?V singular)) ?Rest))
Cost: 1.0

; Missing Article
Pattern:  (np (n ?N :countable true :definite true))
Template: (np (dt "the") (n ?N))
Cost: 0.5

; Double Negative
Pattern:  (vp (adv "not") (v ?V) (adv "never"))
Template: (vp (adv "not") (v ?V) (adv "ever"))
Cost: 1.0
```

---

## Implementation

### CFG Compiler

**File**: `liblevenshtein-rust/src/grammar/cfg_compiler.rs`

```text
//! Compiler from CFG rules to MORK pattern/template pairs.

use mork_expr::{Expr, ExprZipper};
use std::collections::HashMap;

/// A compiled grammar rule.
#[derive(Clone, Debug)]
pub struct CompiledRule {
    /// Pattern expression for matching
    pub pattern: Expr,

    /// Template expression for correction
    pub template: Expr,

    /// Optional constraint expression
    pub constraint: Option<Expr>,

    /// Error cost (lower = more likely correction)
    pub cost: f32,

    /// Rule identifier for debugging
    pub rule_id: String,
}

/// Compiler for CFG rules.
pub struct CfgCompiler {
    /// Variable name to de Bruijn index mapping
    var_indices: HashMap<String, (u8, u8)>,

    /// Next available variable index
    next_var: u8,

    /// Current de Bruijn level
    level: u8,
}

impl CfgCompiler {
    pub fn new() -> Self {
        Self {
            var_indices: HashMap::new(),
            next_var: 0,
            level: 0,
        }
    }

    /// Compile a CFG rule in textual format.
    ///
    /// # Format
    /// ```text
    /// RULE_ID: PATTERN -> TEMPLATE [COST]
    /// ```
    ///
    /// # Example
    /// ```rust
    /// let rule = compiler.compile_rule(
    ///     "article_error",
    ///     "(np (dt \"a\") (n ?N))",
    ///     "(np (dt \"an\") (n ?N))",
    ///     Some("(is_vowel_initial ?N)"),
    ///     0.5
    /// )?;
    /// ```
    pub fn compile_rule(
        &mut self,
        rule_id: &str,
        pattern_str: &str,
        template_str: &str,
        constraint_str: Option<&str>,
        cost: f32,
    ) -> Result<CompiledRule, CompileError> {
        // Reset variable state for each rule
        self.var_indices.clear();
        self.next_var = 0;
        self.level = 0;

        // Parse and compile pattern
        let pattern = self.compile_expr(pattern_str)?;

        // Compile template with same variable bindings
        let template = self.compile_expr(template_str)?;

        // Optionally compile constraint
        let constraint = constraint_str
            .map(|s| self.compile_expr(s))
            .transpose()?;

        Ok(CompiledRule {
            pattern,
            template,
            constraint,
            cost,
            rule_id: rule_id.to_string(),
        })
    }

    /// Compile an S-expression string to MORK Expr.
    fn compile_expr(&mut self, expr_str: &str) -> Result<Expr, CompileError> {
        let tokens = self.tokenize(expr_str)?;
        self.parse_tokens(&tokens)
    }

    /// Tokenize S-expression.
    ///
    /// The tokenizer returns `Open`, `Close`, `Symbol`, `Variable`, and
    /// `String` tokens. It is intentionally separate from parsing so malformed
    /// quoting and invalid variable spelling are reported before de Bruijn
    /// indices are allocated.
    fn tokenize(&self, s: &str) -> Result<Vec<Token>, CompileError> {
        Tokenizer::new(s).collect()
    }

    /// Parse tokens to MORK Expr.
    ///
    /// The recursive-descent parser accepts atoms, lists, variables (`?name`),
    /// and strings. Variable tokens are resolved through `get_var_index()` so
    /// the same source variable has a stable de Bruijn coordinate throughout a
    /// compiled rule.
    fn parse_tokens(&mut self, tokens: &[Token]) -> Result<Expr, CompileError> {
        Parser::new(tokens, |compiler, name| compiler.get_var_index(name)).parse(self)
    }

    /// Get or create de Bruijn index for variable.
    fn get_var_index(&mut self, name: &str) -> (u8, u8) {
        if let Some(&idx) = self.var_indices.get(name) {
            idx
        } else {
            let idx = (self.level, self.next_var);
            self.next_var += 1;
            self.var_indices.insert(name.to_string(), idx);
            idx
        }
    }
}

#[derive(Debug)]
pub enum CompileError {
    ParseError(String),
    InvalidVariable(String),
    NestedTooDeep,
}

enum Token {
    Open,
    Close,
    Symbol(String),
    Variable(String),
    String(String),
    Number(i64),
}
```

### Error Grammar

**File**: `liblevenshtein-rust/src/grammar/error_grammar.rs`

```rust
//! Pre-defined error grammars for common mistake patterns.

use super::cfg_compiler::{CfgCompiler, CompiledRule};

/// Collection of grammar error rules.
pub struct ErrorGrammar {
    /// All compiled rules
    pub rules: Vec<CompiledRule>,

    /// Index by error category
    pub by_category: HashMap<ErrorCategory, Vec<usize>>,
}

/// Categories of grammar errors.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum ErrorCategory {
    ArticleError,
    SubjectVerbAgreement,
    TenseConsistency,
    PronounCase,
    DoubleNegative,
    RunOnSentence,
    FragmentSentence,
    WordOrder,
    MissingWord,
    ExtraWord,
}

impl ErrorGrammar {
    /// Load the default English error grammar.
    pub fn english_default() -> Result<Self, CompileError> {
        let mut compiler = CfgCompiler::new();
        let mut rules = Vec::new();
        let mut by_category = HashMap::new();

        // Article errors
        let article_rules = Self::compile_article_rules(&mut compiler)?;
        let start_idx = rules.len();
        rules.extend(article_rules);
        by_category.insert(
            ErrorCategory::ArticleError,
            (start_idx..rules.len()).collect()
        );

        // Subject-verb agreement
        let agreement_rules = Self::compile_agreement_rules(&mut compiler)?;
        let start_idx = rules.len();
        rules.extend(agreement_rules);
        by_category.insert(
            ErrorCategory::SubjectVerbAgreement,
            (start_idx..rules.len()).collect()
        );

        // ... more categories ...

        Ok(Self { rules, by_category })
    }

    fn compile_article_rules(
        compiler: &mut CfgCompiler
    ) -> Result<Vec<CompiledRule>, CompileError> {
        vec![
            // "a" before vowel → "an"
            compiler.compile_rule(
                "a_before_vowel",
                "(np (dt \"a\") (n ?N))",
                "(np (dt \"an\") (n ?N))",
                Some("(starts_with_vowel ?N)"),
                0.5,
            )?,

            // "an" before consonant → "a"
            compiler.compile_rule(
                "an_before_consonant",
                "(np (dt \"an\") (n ?N))",
                "(np (dt \"a\") (n ?N))",
                Some("(starts_with_consonant ?N)"),
                0.5,
            )?,

            // Missing article before singular countable noun
            compiler.compile_rule(
                "missing_article",
                "(np (n ?N :countable true :number singular))",
                "(np (dt \"a\") (n ?N))",
                None,
                0.8,
            )?,
        ]
    }

    fn compile_agreement_rules(
        compiler: &mut CfgCompiler
    ) -> Result<Vec<CompiledRule>, CompileError> {
        vec![
            // Singular subject with plural verb
            compiler.compile_rule(
                "subj_verb_singular",
                "(s (np ?Subj :number singular) (vp (v ?V :number plural) ?Rest))",
                "(s (np ?Subj :number singular) (vp (v (singularize ?V)) ?Rest))",
                None,
                1.0,
            )?,

            // Plural subject with singular verb
            compiler.compile_rule(
                "subj_verb_plural",
                "(s (np ?Subj :number plural) (vp (v ?V :number singular) ?Rest))",
                "(s (np ?Subj :number plural) (vp (v (pluralize ?V)) ?Rest))",
                None,
                1.0,
            )?,
        ]
    }

    /// Get rules for a specific category.
    pub fn rules_for(&self, category: ErrorCategory) -> impl Iterator<Item = &CompiledRule> {
        self.by_category
            .get(&category)
            .into_iter()
            .flatten()
            .filter_map(|&idx| self.rules.get(idx))
    }

    /// Get all rules.
    pub fn all_rules(&self) -> impl Iterator<Item = &CompiledRule> {
        self.rules.iter()
    }
}
```

### Lattice Parser Integration

**File**: `liblevenshtein-rust/src/grammar/lattice_parser.rs`

```rust
//! Integration of MORK pattern matching with lattice parsing.

use crate::lattice::Lattice;
use super::error_grammar::{ErrorGrammar, CompiledRule};
use mork_kernel::Space;
use mork_expr::{Expr, ExprEnv};
use std::collections::BTreeMap;

/// Parser that applies grammar rules to a lattice.
pub struct LatticeParser<'a> {
    /// MORK space for pattern matching
    space: &'a mut Space,

    /// Grammar rules to apply
    grammar: &'a ErrorGrammar,
}

impl<'a> LatticeParser<'a> {
    pub fn new(space: &'a mut Space, grammar: &'a ErrorGrammar) -> Self {
        Self { space, grammar }
    }

    /// Parse a lattice using grammar rules.
    ///
    /// # Algorithm
    ///
    /// 1. Load lattice edges as MORK expressions
    /// 2. For each grammar rule:
    ///    a. Query space with rule pattern
    ///    b. Apply template to matches
    ///    c. Track corrections with costs
    /// 3. Return parse forest with corrections
    pub fn parse(&mut self, lattice: &Lattice) -> ParseResult {
        // Step 1: Load lattice into MORK space
        self.load_lattice(lattice);

        // Step 2: Apply grammar rules
        let mut corrections = Vec::new();

        for rule in self.grammar.all_rules() {
            let matches = self.apply_rule(rule);
            corrections.extend(matches);
        }

        // Step 3: Build parse forest
        self.build_parse_forest(corrections)
    }

    /// Load lattice edges as MORK expressions.
    fn load_lattice(&mut self, lattice: &Lattice) {
        for edge in lattice.edges.iter() {
            let label = lattice.label(edge.label).unwrap_or("");

            // Create MORK expression for edge
            // Format: (edge source target label weight)
            let expr_str = format!(
                "(edge {} {} \"{}\" {})",
                edge.source.0,
                edge.target.0,
                label,
                edge.weight
            );

            self.space.insert_sexpr(&expr_str);
        }
    }

    /// Apply a single grammar rule.
    fn apply_rule(&mut self, rule: &CompiledRule) -> Vec<Correction> {
        let mut corrections = Vec::new();

        // Use MORK's query_multi to find all matches
        let matches = self.space.query_multi_raw(rule.pattern);

        for (bindings, location) in matches {
            // Check constraint if present
            if let Some(ref constraint) = rule.constraint {
                if !self.check_constraint(constraint, &bindings) {
                    continue;
                }
            }

            // Apply template to generate correction
            let corrected = self.apply_template(&rule.template, &bindings);

            corrections.push(Correction {
                rule_id: rule.rule_id.clone(),
                original_location: location,
                corrected_expr: corrected,
                cost: rule.cost,
                bindings: bindings.clone(),
            });
        }

        corrections
    }

    /// Check if constraint is satisfied.
    fn check_constraint(
        &self,
        constraint: &Expr,
        bindings: &BTreeMap<(u8, u8), ExprEnv>,
    ) -> bool {
        // Evaluate constraint with bindings
        // Implementation depends on constraint language
        true  // Simplified
    }

    /// Apply template with bindings to produce corrected expression.
    fn apply_template(
        &self,
        template: &Expr,
        bindings: &BTreeMap<(u8, u8), ExprEnv>,
    ) -> Expr {
        // Substitute variables in template with bound values
        self.space.substitute_bindings(template, bindings)
    }

    /// Build parse forest from corrections.
    fn build_parse_forest(&self, corrections: Vec<Correction>) -> ParseResult {
        // Group corrections by location
        // Build forest of valid parses
        // Rank by total cost
        ParseResult {
            corrections,
            best_parse: None,  // Computed from corrections
        }
    }
}

/// A single correction found by applying a grammar rule.
#[derive(Clone, Debug)]
pub struct Correction {
    /// Rule that produced this correction
    pub rule_id: String,

    /// Location in original expression
    pub original_location: Vec<u8>,

    /// Corrected expression
    pub corrected_expr: Expr,

    /// Cost of this correction
    pub cost: f32,

    /// Variable bindings from pattern match
    pub bindings: BTreeMap<(u8, u8), ExprEnv>,
}

/// Result of parsing with grammar rules.
#[derive(Clone, Debug)]
pub struct ParseResult {
    /// All corrections found
    pub corrections: Vec<Correction>,

    /// Best parse (if unique)
    pub best_parse: Option<Expr>,
}
```

---

## Usage

### Compiling Grammar Rules

```rust
use liblevenshtein::grammar::{CfgCompiler, ErrorGrammar};

// Use pre-defined English grammar
let grammar = ErrorGrammar::english_default()?;

// Or compile custom rules
let mut compiler = CfgCompiler::new();
let rule = compiler.compile_rule(
    "custom_rule",
    "(foo ?X (bar ?Y))",
    "(corrected_foo ?X ?Y)",
    None,
    0.5,
)?;
```

### Applying Grammar to Lattice

```rust
use liblevenshtein::grammar::LatticeParser;
use mork_kernel::Space;

let mut space = Space::new();
let grammar = ErrorGrammar::english_default()?;

let mut parser = LatticeParser::new(&mut space, &grammar);
let result = parser.parse(&word_lattice);

for correction in result.corrections {
    println!(
        "Rule {}: {} (cost {:.2})",
        correction.rule_id,
        correction.corrected_expr,
        correction.cost
    );
}
```

### MORK Query with Grammar Correction

```metta
; Apply grammar correction to fuzzy-matched input
!(let $lattice (wfst-lattice "teh quik brwn fox" 2)
    (grammar-correct $lattice english-default))

; Returns corrected parses ranked by total cost
```

---

## MORK Functions Used

| Function | Location | Purpose |
|----------|----------|---------|
| `query_multi_raw` | kernel/src/space.rs:992 | Find all pattern matches in space |
| `transform_multi_multi_` | kernel/src/space.rs:1221 | Apply pattern→template corrections |
| `unify` | expr/src/lib.rs:1849 | Variable binding and constraint checking |
| `match2` | expr/src/lib.rs:921 | Recursive structural pattern matching |

---

## Performance

### Efficiency

MORK's `query_multi_i()` provides **O(K×N)** complexity where:
- K = average candidates per position
- N = number of positions

This avoids the **O(K^N)** path enumeration of naive approaches.

### Optimization Strategies

1. **Rule ordering**: Apply high-confidence rules first
2. **Category filtering**: Only apply relevant rule categories
3. **Early termination**: Stop when good-enough parse found
4. **Caching**: Memoize pattern matches across similar inputs

---

## References

- CFG Grammar Correction Design
- [Grammar Correction Main Design](../../design/grammar-correction/MAIN_DESIGN.md)
- [Phase C: WFST Composition](./wfst_composition.md)
- [MORK Space Documentation](../../../../MORK/kernel/src/space.rs)
