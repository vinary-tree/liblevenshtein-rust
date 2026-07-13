# Structural Repair for Programming Languages

**Status**: Design Document
**Phase**: Future (Post Phase D)
**Depends On**: Phase D (Grammar Correction via MORK)
**Last Updated**: 2024-12-05

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [MORK Pattern Matching for AST Repair](#mork-pattern-matching-for-ast-repair)
4. [Error Detection Patterns](#error-detection-patterns)
5. [Correction Templates](#correction-templates)
6. [Type-Aware Repair](#type-aware-repair)
7. [Integration with IDE/LSP](#integration-with-idelsp)
8. [Implementation Guide](#implementation-guide)
9. [Example Languages](#example-languages)

---

## Executive Summary

### The Problem

Programming language sources contain errors that go beyond simple spelling:

1. **Syntax errors**: Missing semicolons, unbalanced brackets, malformed expressions
2. **Structural errors**: Incorrect AST node types, missing required children
3. **Type errors**: Type mismatches requiring structural coercion
4. **Semantic errors**: Undefined variables, incorrect scoping

### Solution: MORK-Based Structural Repair

Use MORK's pattern matching (`match2()`) and transformation (`transform_multi_multi_()`) to:
- **Detect** structural errors via pattern matching on malformed ASTs
- **Correct** errors via template-based AST transformation
- **Validate** repairs with optional type checking integration

### Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Declarative rules** | AST repair rules as pattern/template pairs |
| **Language-agnostic** | Same MORK infrastructure for any language |
| **Composable** | Lattice of candidate repairs from Tier 1 |
| **Incremental** | Repair affected AST subtrees only |
| **Type-aware** | Optional semantic layer for type-guided repair |

---

## Architecture Overview

### Three-Tier Structural Repair Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Lexical (liblevenshtein)                            │
│   Input text → Token lattice                                │
│   - Identifier spelling correction                          │
│   - Keyword fuzzy matching                                  │
│   - Literal normalization                                   │
│   Output: Token lattice with alternatives                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Syntactic (MORK + Tree-sitter/Parser)              │
│   Token lattice → Parse forest                              │
│   - Error-recovery parsing                                  │
│   - MORK pattern matching on AST                           │
│   - Structural repair via templates                         │
│   Output: Candidate AST forest                              │
│   Files: src/grammar/ast_repair.rs, MORK patterns          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: Semantic (Type Checker / LLM)                       │
│   Parse forest → Valid program                              │
│   - Type inference and checking                             │
│   - Type-guided repair suggestions                          │
│   - LLM for complex semantic repairs                        │
│   Output: Best corrected program                            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Example

```
Input: "fn main() { let x = 42 println!(x) }"
                            ↑
                      Missing semicolon

Tier 1 (Lexical):
  Tokens: [fn, main, (, ), {, let, x, =, 42, println, !, (, x, ), }]
  Lattice: No spelling errors, pass through

Tier 2 (Syntactic):
  Error-recovery parse:
    function_definition
    ├── name: "main"
    ├── params: []
    └── body:
        ├── let_statement
        │   ├── name: "x"
        │   └── value: 42
        └── ERROR: expected semicolon
        └── macro_invocation
            └── name: "println"

  MORK pattern match:
    Pattern:  (let_statement ?name ?value) (ERROR "expected semicolon") ?next
    Template: (let_statement ?name ?value) (semicolon) ?next

  Repaired AST:
    function_definition
    ├── name: "main"
    ├── params: []
    └── body:
        ├── let_statement { name: "x", value: 42 }
        ├── semicolon
        └── macro_invocation { name: "println" }

Tier 3 (Semantic):
  Type check: OK (x is u32, println accepts any Debug type)

Output: "fn main() { let x = 42; println!(x) }"
```

---

## MORK Pattern Matching for AST Repair

### AST as S-Expression

MORK operates on S-expressions. Programming language ASTs map naturally:

```metta
; Rust AST nodes as S-expressions
(function_definition
    (name main)
    (params)
    (body
        (let_statement (name x) (value (integer 42)))
        (macro_invocation (name println) (args (identifier x)))))

; Python AST nodes
(function_def
    (name greet)
    (args (arg (name person) (annotation (type str))))
    (body
        (expr (call
            (func (name print))
            (args (string "Hello"))))))
```

### Pattern Matching with `match2()`

MORK's `match2()` function (expr/src/lib.rs:921) performs recursive structural matching:

```metta
; Pattern to detect missing semicolon after let
Pattern:
    (let_statement ?name ?value)
    (ERROR "expected semicolon")
    ?next_stmt

; Matches against AST:
(body
    (let_statement (name x) (value (integer 42)))
    (ERROR "expected semicolon")
    (macro_invocation ...))

Bindings:
    ?name → (name x)
    ?value → (value (integer 42))
    ?next_stmt → (macro_invocation ...)
```

### Template-Based Repair

Use MORK's `transform_multi_multi_()` (space.rs:1221) for AST transformation:

```metta
; Repair rule: Insert semicolon
(repair-rule
    (pattern (let_statement ?name ?value) (ERROR "expected semicolon") ?next)
    (template (let_statement ?name ?value) (semicolon) ?next)
    (cost 0.1))

; Apply repair
transform_multi_multi_(rule, ast) →
    (body
        (let_statement (name x) (value (integer 42)))
        (semicolon)
        (macro_invocation ...))
```

---

## Error Detection Patterns

### Category 1: Missing Delimiters

```metta
; Missing semicolon after statement (Rust, C, Java)
(repair-rule missing-semicolon
    (pattern (?stmt_type . ?children) (ERROR "expected ;") ?next)
    (template (?stmt_type . ?children) (semicolon) ?next)
    (cost 0.1)
    (languages rust c java javascript))

; Missing closing parenthesis
(repair-rule missing-close-paren
    (pattern (call (func ?f) (args ?a ...) (ERROR "expected )")))
    (template (call (func ?f) (args ?a ...)))
    (cost 0.2)
    (languages all))

; Missing closing brace
(repair-rule missing-close-brace
    (pattern (block ?stmts ... (ERROR "expected }")))
    (template (block ?stmts ...))
    (cost 0.3)
    (languages rust c java javascript))
```

### Category 2: Expression Errors

```metta
; Missing operand in binary expression
(repair-rule missing-right-operand
    (pattern (binary_expr (op ?op) (left ?l) (right (ERROR))))
    (template (binary_expr (op ?op) (left ?l) (right (completion-hole))))
    (cost 0.5)
    (languages all))

; Incorrect operator precedence (suggest parentheses)
(repair-rule precedence-warning
    (pattern (binary_expr (op "*") (left (binary_expr (op "+") ?a ?b)) (right ?c)))
    (template (binary_expr (op "*") (left (paren_expr (binary_expr (op "+") ?a ?b))) (right ?c)))
    (cost 0.0)  ; Warning only, no change unless requested
    (type warning)
    (languages all))
```

### Category 3: Declaration Errors

```metta
; Missing type annotation where required (Rust, TypeScript)
(repair-rule missing-type-annotation
    (pattern (let_statement (name ?n) (type (ERROR)) (value ?v)))
    (template (let_statement (name ?n) (type (infer)) (value ?v)))
    (cost 0.3)
    (requires type-inference)
    (languages rust typescript))

; Missing return type
(repair-rule missing-return-type
    (pattern (function_definition (name ?n) (params ?p) (return_type (ERROR)) (body ?b)))
    (template (function_definition (name ?n) (params ?p) (return_type (infer)) (body ?b)))
    (cost 0.4)
    (requires type-inference)
    (languages rust typescript c))
```

### Category 4: Control Flow Errors

```metta
; Missing else branch in exhaustive match (Rust)
(repair-rule non-exhaustive-match
    (pattern (match_expr (scrutinee ?s) (arms ?a ...) (ERROR "non-exhaustive")))
    (template (match_expr (scrutinee ?s) (arms ?a ... (arm (pattern _) (body (todo))))))
    (cost 0.5)
    (languages rust))

; If without else in expression position
(repair-rule if-missing-else
    (pattern (if_expr (condition ?c) (then ?t) (else (ERROR))))
    (template (if_expr (condition ?c) (then ?t) (else (unit))))
    (cost 0.3)
    (languages rust kotlin))
```

### Category 5: Import/Module Errors

```metta
; Missing import
(repair-rule suggest-import
    (pattern (identifier ?name))
    (where (unresolved ?name) (can-import ?name from ?module))
    (template (identifier ?name))
    (side-effect (add-import ?module ?name))
    (cost 0.2)
    (languages rust python javascript typescript))

; Unused import (cleanup)
(repair-rule remove-unused-import
    (pattern (use_statement ?path))
    (where (unused ?path))
    (template (removed))
    (cost 0.0)
    (type cleanup)
    (languages rust python javascript))
```

---

## Correction Templates

### Template Variables

| Variable | Meaning |
|----------|---------|
| `?name` | Single node binding |
| `?items ...` | Sequence binding (variadic) |
| `(infer)` | Type-inference hole |
| `(completion-hole)` | User-completion hole |
| `(work-marker)` | Explicit work-item marker in source languages that expose one |
| `(unit)` | Unit type/expression `()` |
| `(removed)` | Remove node entirely |

### Template Operators

```metta
; Conditionals in templates
(template
    (if (vowel-initial ?word)
        (article "an")
        (article "a")))

; Lookup in external table
(template
    (type (lookup-type ?expr)))  ; Invoke type inference

; Generate unique identifier
(template
    (name (gensym "temp")))  ; temp_1, temp_2, etc.
```

### Multi-Step Repairs

Some repairs require multiple transformations:

```metta
; Repair: Add type annotation + coerce value
(repair-sequence type-mismatch
    (step 1 (infer-type ?expr) → ?inferred)
    (step 2 (if (not-equal ?inferred ?expected)
                (coerce ?expr ?expected))))

; Example: let x: i32 = 3.14
; Step 1: Infer 3.14 → f64
; Step 2: f64 != i32 → coerce: let x: i32 = 3.14 as i32
```

---

## Type-Aware Repair

### Integration with Type Checker

For languages with static typing, semantic repair requires type information:

```rust
// Tier 3: Type-Aware Repair
pub struct TypeAwareRepair {
    type_checker: Box<dyn TypeChecker>,
    repair_rules: Vec<TypedRepairRule>,
}

pub struct TypedRepairRule {
    pattern: MorkPattern,
    template: MorkTemplate,
    type_constraint: TypeConstraint,
    cost: f64,
}

pub enum TypeConstraint {
    /// No type constraint
    None,
    /// Expression must have specific type
    ExprHasType(ExprPattern, Type),
    /// Two expressions must have compatible types
    TypesCompatible(ExprPattern, ExprPattern),
    /// Return type must match function signature
    ReturnTypeMatches,
}

impl TypeAwareRepair {
    pub fn repair(&self, ast: &Ast, diagnostics: &[Diagnostic]) -> Vec<RepairCandidate> {
        let mut candidates = Vec::new();

        for rule in &self.repair_rules {
            // Pattern match
            let matches = mork_match2(rule.pattern, ast);

            for m in matches {
                // Check type constraint
                if self.type_checker.satisfies(&m, &rule.type_constraint) {
                    // Apply template
                    let repaired = mork_transform(&m, &rule.template);

                    // Verify repair is type-correct
                    if self.type_checker.check(&repaired).is_ok() {
                        candidates.push(RepairCandidate {
                            ast: repaired,
                            cost: rule.cost,
                            description: rule.description.clone(),
                        });
                    }
                }
            }
        }

        candidates.sort_by_key(|c| OrderedFloat(c.cost));
        candidates
    }
}
```

### Type-Guided Repairs

```metta
; Type mismatch: expected i32, found String
(repair-rule type-mismatch-parse
    (pattern (assignment (target ?t) (value ?v)))
    (where (type-of ?t i32) (type-of ?v String))
    (template (assignment
        (target ?t)
        (value (method_call (receiver ?v) (method "parse") (args) (unwrap)))))
    (cost 0.5))

; Example:
; Before: let x: i32 = user_input;  // user_input is String
; After:  let x: i32 = user_input.parse().unwrap();

; Type mismatch: expected &str, found String
(repair-rule string-to-str
    (pattern (call (func ?f) (args ... ?arg ...)))
    (where (param-type ?f ?arg &str) (type-of ?arg String))
    (template (call (func ?f) (args ... (borrow ?arg) ...)))
    (cost 0.2))

; Example:
; Before: greet(name);  // name is String, greet expects &str
; After:  greet(&name);
```

### Ownership Repairs (Rust-Specific)

```metta
; Move error: value borrowed after move
(repair-rule use-after-move-clone
    (pattern (block
        (let_statement (name ?x) ?init)
        ?stmts ...
        (call (func ?f1) (args ... ?x ...))  ; Move occurs here
        ?more_stmts ...
        (expr ?x)))  ; Use after move
    (template (block
        (let_statement (name ?x) ?init)
        ?stmts ...
        (call (func ?f1) (args ... (method_call (receiver ?x) (method "clone") (args)) ...))
        ?more_stmts ...
        (expr ?x)))
    (cost 0.3)
    (languages rust))

; Borrow error: cannot borrow as mutable more than once
(repair-rule double-mut-borrow-refcell
    (pattern (block
        (let (name ?ref1) (mut_borrow ?x))
        ?stmts ...
        (let (name ?ref2) (mut_borrow ?x))))
    (template (block
        (let (name ?x) (refcell ?x))
        (let (name ?ref1) (method_call (receiver ?x) (method "borrow_mut") (args)))
        ?stmts ...
        (let (name ?ref2) (method_call (receiver ?x) (method "borrow_mut") (args)))))
    (cost 0.8)
    (languages rust))
```

---

## Integration with IDE/LSP

### LSP Code Actions

The repair system provides code actions via LSP:

```rust
pub struct StructuralRepairProvider {
    repair_engine: MorkRepairEngine,
}

impl CodeActionProvider for StructuralRepairProvider {
    fn provide_code_actions(
        &self,
        document: &Document,
        range: Range,
        context: &CodeActionContext,
    ) -> Vec<CodeAction> {
        let mut actions = Vec::new();

        // Get AST for document
        let ast = self.parse_document(document);

        // Get diagnostics in range
        let diagnostics: Vec<_> = context.diagnostics
            .iter()
            .filter(|d| overlaps(&d.range, &range))
            .collect();

        // Generate repair candidates
        let candidates = self.repair_engine.repair(&ast, &diagnostics);

        for (i, candidate) in candidates.iter().take(5).enumerate() {
            actions.push(CodeAction {
                title: format!("Fix: {}", candidate.description),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(diagnostics.clone()),
                edit: Some(WorkspaceEdit {
                    changes: Some(hashmap! {
                        document.uri.clone() => vec![
                            TextEdit {
                                range: candidate.affected_range,
                                new_text: candidate.repaired_text.clone(),
                            }
                        ]
                    }),
                    ..Default::default()
                }),
                is_preferred: Some(i == 0),  // Best candidate is preferred
                ..Default::default()
            });
        }

        actions
    }
}
```

### Quick Fix Example (VSCode)

```json
// When cursor is on error: "missing semicolon"
{
  "title": "Fix: Insert semicolon",
  "kind": "quickfix",
  "isPreferred": true,
  "edit": {
    "changes": {
      "file:///src/main.rs": [
        {
          "range": { "start": { "line": 5, "character": 20 }, "end": { "line": 5, "character": 20 } },
          "newText": ";"
        }
      ]
    }
  }
}
```

### Batch Repair Mode

For automated fixing (e.g., `cargo fix`, `eslint --fix`):

```rust
pub fn batch_repair(
    files: &[PathBuf],
    config: &RepairConfig,
) -> Result<BatchRepairReport, Error> {
    let mut report = BatchRepairReport::default();

    for file in files {
        let source = fs::read_to_string(file)?;
        let ast = parse(&source)?;
        let diagnostics = diagnose(&ast);

        // Get best repair for each diagnostic
        let repairs = repair_engine.repair_all(&ast, &diagnostics);

        // Apply non-conflicting repairs
        let (applied, skipped) = apply_repairs(&source, repairs);

        report.files_fixed.push(FileRepairReport {
            path: file.clone(),
            repairs_applied: applied.len(),
            repairs_skipped: skipped.len(),
        });

        // Write repaired file
        if !applied.is_empty() {
            let repaired_source = apply_edits(&source, &applied);
            fs::write(file, repaired_source)?;
        }
    }

    Ok(report)
}
```

---

## Implementation Guide

### Files to Create

| File | Purpose |
|------|---------|
| `liblevenshtein-rust/src/grammar/ast_repair.rs` | Core AST repair engine |
| `liblevenshtein-rust/src/grammar/repair_rules.rs` | Rule definitions |
| `liblevenshtein-rust/src/grammar/type_constraints.rs` | Type constraint system |
| `liblevenshtein-rust/src/grammar/languages/mod.rs` | Language-specific rules |
| `liblevenshtein-rust/src/grammar/languages/rust.rs` | Rust-specific patterns |
| `liblevenshtein-rust/src/grammar/languages/python.rs` | Python-specific patterns |
| `liblevenshtein-rust/src/grammar/languages/javascript.rs` | JavaScript patterns |

### Core Data Structures

```rust
// src/grammar/ast_repair.rs

use indexmap::IndexMap;

/// AST repair engine using MORK pattern matching
pub struct AstRepairEngine {
    /// Repair rules organized by language
    rules: IndexMap<Language, Vec<RepairRule>>,
    /// Optional type checker for semantic repairs
    type_checker: Option<Box<dyn TypeChecker>>,
    /// MORK space for pattern matching
    mork_space: MorkSpace,
}

/// A single repair rule
pub struct RepairRule {
    /// Rule identifier
    pub id: RuleId,
    /// Human-readable name
    pub name: String,
    /// Pattern to match (MORK S-expression)
    pub pattern: Expr,
    /// Template to apply (MORK S-expression)
    pub template: Expr,
    /// Optional type constraint
    pub type_constraint: Option<TypeConstraint>,
    /// Cost of this repair (lower = better)
    pub cost: f64,
    /// Error type this rule fixes
    pub error_type: ErrorType,
    /// Languages this rule applies to
    pub languages: Vec<Language>,
}

/// Repair candidate with metadata
pub struct RepairCandidate {
    /// Repaired AST
    pub ast: Ast,
    /// Text edits to apply
    pub edits: Vec<TextEdit>,
    /// Cost of repair
    pub cost: f64,
    /// Human-readable description
    pub description: String,
    /// Rule that generated this candidate
    pub rule_id: RuleId,
}

impl AstRepairEngine {
    /// Create new repair engine
    pub fn new() -> Self {
        Self {
            rules: IndexMap::new(),
            type_checker: None,
            mork_space: MorkSpace::new(),
        }
    }

    /// Load rules for a language
    pub fn load_language(&mut self, lang: Language) -> Result<(), Error> {
        let rules = match lang {
            Language::Rust => rust_rules(),
            Language::Python => python_rules(),
            Language::JavaScript => javascript_rules(),
            Language::TypeScript => typescript_rules(),
            _ => return Err(Error::UnsupportedLanguage(lang)),
        };
        self.rules.insert(lang, rules);
        Ok(())
    }

    /// Set type checker for semantic repairs
    pub fn with_type_checker(mut self, tc: Box<dyn TypeChecker>) -> Self {
        self.type_checker = Some(tc);
        self
    }

    /// Generate repair candidates for AST with diagnostics
    pub fn repair(
        &self,
        ast: &Ast,
        diagnostics: &[Diagnostic],
        lang: Language,
    ) -> Vec<RepairCandidate> {
        let rules = self.rules.get(&lang).unwrap_or(&Vec::new());
        let mut candidates = Vec::new();

        // Convert AST to MORK expression
        let ast_expr = ast_to_mork_expr(ast);

        for diagnostic in diagnostics {
            // Find applicable rules
            for rule in rules.iter().filter(|r| r.matches_diagnostic(diagnostic)) {
                // Pattern match
                let matches = self.mork_space.match2(&rule.pattern, &ast_expr);

                for m in matches {
                    // Check type constraint if present
                    if let Some(tc) = &rule.type_constraint {
                        if let Some(checker) = &self.type_checker {
                            if !checker.satisfies(&m, tc) {
                                continue;
                            }
                        }
                    }

                    // Apply template
                    let repaired_expr = self.mork_space.transform(&m, &rule.template);
                    let repaired_ast = mork_expr_to_ast(&repaired_expr);

                    // Verify repair is valid
                    if let Some(checker) = &self.type_checker {
                        if checker.check(&repaired_ast).is_err() {
                            continue;
                        }
                    }

                    // Generate text edits
                    let edits = diff_ast(ast, &repaired_ast);

                    candidates.push(RepairCandidate {
                        ast: repaired_ast,
                        edits,
                        cost: rule.cost,
                        description: rule.describe(&m),
                        rule_id: rule.id,
                    });
                }
            }
        }

        // Sort by cost
        candidates.sort_by_key(|c| OrderedFloat(c.cost));
        candidates
    }
}
```

### MORK Integration

```rust
// src/grammar/mork_bridge.rs

use mork_kernel::space::Space;
use mork_expr::Expr;

/// Bridge between AST and MORK expressions
pub struct MorkBridge {
    space: Space,
}

impl MorkBridge {
    /// Convert language AST to MORK S-expression
    pub fn ast_to_expr(&self, ast: &Ast) -> Expr {
        match ast {
            Ast::FunctionDef { name, params, body, return_type } => {
                Expr::list(vec![
                    Expr::symbol("function_definition"),
                    Expr::list(vec![Expr::symbol("name"), Expr::symbol(name)]),
                    Expr::list(vec![Expr::symbol("params"), self.params_to_expr(params)]),
                    Expr::list(vec![Expr::symbol("body"), self.block_to_expr(body)]),
                    Expr::list(vec![Expr::symbol("return_type"),
                        return_type.as_ref().map_or(Expr::nil(), |t| self.type_to_expr(t))]),
                ])
            }
            Ast::LetStatement { name, type_ann, value } => {
                Expr::list(vec![
                    Expr::symbol("let_statement"),
                    Expr::list(vec![Expr::symbol("name"), Expr::symbol(name)]),
                    Expr::list(vec![Expr::symbol("type"),
                        type_ann.as_ref().map_or(Expr::symbol("infer"), |t| self.type_to_expr(t))]),
                    Expr::list(vec![Expr::symbol("value"), self.expr_to_mork(value)]),
                ])
            }
            Ast::Error { expected, found } => {
                Expr::list(vec![
                    Expr::symbol("ERROR"),
                    Expr::string(&format!("expected {}", expected)),
                ])
            }
            // ... other AST node types
        }
    }

    /// Convert MORK expression back to AST
    pub fn expr_to_ast(&self, expr: &Expr) -> Result<Ast, ConversionError> {
        match expr {
            Expr::List(items) if !items.is_empty() => {
                let tag = items[0].as_symbol()?;
                match tag {
                    "function_definition" => {
                        let name = self.extract_field(&items[1], "name")?;
                        let params = self.extract_params(&items[2])?;
                        let body = self.extract_block(&items[3])?;
                        let return_type = self.extract_optional_type(&items[4])?;
                        Ok(Ast::FunctionDef { name, params, body, return_type })
                    }
                    "let_statement" => {
                        let name = self.extract_field(&items[1], "name")?;
                        let type_ann = self.extract_optional_type(&items[2])?;
                        let value = self.extract_expr(&items[3])?;
                        Ok(Ast::LetStatement { name, type_ann, value })
                    }
                    // ... other node types
                    _ => Err(ConversionError::UnknownTag(tag.to_string())),
                }
            }
            _ => Err(ConversionError::InvalidStructure),
        }
    }

    /// Pattern match and apply repair
    pub fn apply_repair(&mut self, ast_expr: &Expr, rule: &RepairRule) -> Vec<Expr> {
        let matches = self.space.query_multi_raw(&rule.pattern, ast_expr);

        matches.into_iter()
            .map(|bindings| {
                self.space.transform_with_bindings(&rule.template, &bindings)
            })
            .collect()
    }
}
```

### Language-Specific Rules

```rust
// src/grammar/languages/rust.rs

pub fn rust_rules() -> Vec<RepairRule> {
    vec![
        // Missing semicolon
        RepairRule {
            id: RuleId::new("rust-missing-semicolon"),
            name: "Insert semicolon".to_string(),
            pattern: parse_mork(r#"
                (let_statement ?name ?value)
                (ERROR "expected ;")
                ?next
            "#),
            template: parse_mork(r#"
                (let_statement ?name ?value)
                (semicolon)
                ?next
            "#),
            type_constraint: None,
            cost: 0.1,
            error_type: ErrorType::MissingSemicolon,
            languages: vec![Language::Rust],
        },

        // Missing type annotation
        RepairRule {
            id: RuleId::new("rust-infer-type"),
            name: "Infer type annotation".to_string(),
            pattern: parse_mork(r#"
                (let_statement (name ?n) (type (ERROR)) (value ?v))
            "#),
            template: parse_mork(r#"
                (let_statement (name ?n) (type (infer)) (value ?v))
            "#),
            type_constraint: Some(TypeConstraint::InferType("?v".into())),
            cost: 0.3,
            error_type: ErrorType::MissingType,
            languages: vec![Language::Rust],
        },

        // Borrow as mutable
        RepairRule {
            id: RuleId::new("rust-add-mut-borrow"),
            name: "Add mutable borrow".to_string(),
            pattern: parse_mork(r#"
                (call (func ?f) (args ... (borrow ?arg) ...))
            "#),
            template: parse_mork(r#"
                (call (func ?f) (args ... (mut_borrow ?arg) ...))
            "#),
            type_constraint: Some(TypeConstraint::ParamNeedsMut("?f".into(), "?arg".into())),
            cost: 0.2,
            error_type: ErrorType::BorrowMismatch,
            languages: vec![Language::Rust],
        },

        // Clone to avoid move
        RepairRule {
            id: RuleId::new("rust-add-clone"),
            name: "Clone to avoid move".to_string(),
            pattern: parse_mork(r#"
                (call (func ?f) (args ... ?moved_arg ...))
            "#),
            template: parse_mork(r#"
                (call (func ?f) (args ... (method_call (receiver ?moved_arg) (method clone) (args)) ...))
            "#),
            type_constraint: Some(TypeConstraint::And(vec![
                TypeConstraint::IsMove("?moved_arg".into()),
                TypeConstraint::ImplementsClone("?moved_arg".into()),
            ])),
            cost: 0.4,
            error_type: ErrorType::UseAfterMove,
            languages: vec![Language::Rust],
        },
    ]
}
```

---

## Example Languages

### Rust

**Common Errors and Repairs**:

| Error | Pattern | Repair |
|-------|---------|--------|
| Missing semicolon | `let x = 42 println!()` | Insert `;` |
| Missing mut | `let x = vec![]` | `let mut x` |
| Borrow checker | `&x` where `&mut x` needed | `&mut x` |
| Type mismatch | `String` vs `&str` | Add `.as_str()` or `&` |
| Missing lifetime | `fn foo(s: &str) -> &str` | Add lifetime `'a` |

**Rust-Specific Rule File**: See `src/grammar/languages/rust.rs`

### Python

**Common Errors and Repairs**:

| Error | Pattern | Repair |
|-------|---------|--------|
| Indentation | Inconsistent | Normalize to 4 spaces |
| Missing colon | `def foo()` | `def foo():` |
| f-string error | `f"Hello {name}"` with missing `f` | Add `f` prefix |
| Import order | Unordered imports | Sort by PEP8 |

**Python-Specific Rule File**: See `src/grammar/languages/python.rs`

### JavaScript/TypeScript

**Common Errors and Repairs**:

| Error | Pattern | Repair |
|-------|---------|--------|
| Missing semicolon | `const x = 42` | `const x = 42;` (if enabled) |
| Const reassignment | `const x = 1; x = 2` | Change to `let` |
| Type annotation | Missing in TS | Infer and add |
| Null check | `x.prop` where `x` nullable | `x?.prop` |

**JavaScript-Specific Rule File**: See `src/grammar/languages/javascript.rs`

---

## Performance Considerations

### Incremental Repair

Only re-repair affected subtrees:

```rust
pub fn incremental_repair(
    old_ast: &Ast,
    new_ast: &Ast,
    old_repairs: &[RepairCandidate],
) -> Vec<RepairCandidate> {
    // Find changed subtrees
    let changed_nodes = diff_ast_nodes(old_ast, new_ast);

    // Only re-run rules for changed regions
    let affected_rules: Vec<_> = old_repairs.iter()
        .filter(|r| changed_nodes.iter().any(|n| overlaps(&r.affected_range, n)))
        .collect();

    // Re-run affected rules only
    // ...
}
```

### Caching

```rust
pub struct RepairCache {
    // Cache pattern match results
    match_cache: LruCache<(PatternHash, AstHash), Vec<Bindings>>,
    // Cache type inference results
    type_cache: LruCache<ExprHash, Type>,
}
```

### Parallel Rule Application

```rust
pub fn parallel_repair(ast: &Ast, rules: &[RepairRule]) -> Vec<RepairCandidate> {
    rules.par_iter()
        .flat_map(|rule| apply_rule(ast, rule))
        .collect()
}
```

---

## Related Documentation

- [Grammar Correction via MORK](./grammar_correction.md) - Phase D foundation
- WFST Architecture - Three-tier pipeline
- [MORK Pattern Matching](https://github.com/trueagi-io/MORK/tree/main/expr) - MORK `match2()` reference (the `expr` crate)
- Tree-sitter Integration - Parser integration

---

## Future Work

1. **Machine Learning Rule Discovery**: Learn repair patterns from fix commits
2. **Cross-File Repairs**: Handle multi-file refactorings
3. **IDE Plugin Ecosystem**: VSCode, IntelliJ, Emacs, Vim plugins
4. **Confidence Scoring**: ML-based confidence for repair suggestions
5. **Natural Language Explanations**: LLM-generated repair explanations
