[← Documentation Index](../../README.md)

# MeTTaTron: F1R3FLY.io's MeTTa Compiler

This document provides a comprehensive analysis of MeTTaTron, F1R3FLY.io's MeTTa
compiler designed for high-performance evaluation and Rholang integration.

**Location**: `/home/dylon/Workspace/f1r3fly.io/MeTTa-Compiler/`

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Parser Pipeline](#parser-pipeline)
4. [Backend Components](#backend-components)
5. [MORK Integration](#mork-integration)
6. [PathMap Par Integration](#pathmap-par-integration)
7. [Evaluation Engine](#evaluation-engine)
8. [Type System](#type-system)
9. [Rholang Bridge](#rholang-bridge)
10. [Relevance to MeTTaIL](#relevance-to-mettail)

---

## Overview

MeTTaTron is an optimized MeTTa compiler built specifically for F1R3FLY.io's
ecosystem. It provides:

- **High-performance evaluation** via MORK pattern matching
- **Rholang integration** via PathMap Par conversion
- **Tree-sitter parsing** for robust syntax handling
- **Lazy evaluation** with iterative trampoline design

### Design Goals

1. **Performance**: Sub-millisecond pattern matching
2. **Integration**: Seamless MeTTa ↔ Rholang conversion
3. **Compatibility**: API-compatible with hyperon-experimental
4. **Scalability**: Efficient handling of large knowledge bases

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MeTTaTron Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MeTTa Source                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                Tree-Sitter Parser                          │ │
│  │  tree_sitter_parser.rs                                     │ │
│  │  ┌────────────────┐    ┌────────────────┐                 │ │
│  │  │ tree-sitter-   │ →  │   MettaExpr    │                 │ │
│  │  │ metta grammar  │    │   (IR)         │                 │ │
│  │  └────────────────┘    └────────────────┘                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Compiler                                 │ │
│  │  compile.rs                                                │ │
│  │  ┌────────────────┐    ┌────────────────┐                 │ │
│  │  │  MettaExpr     │ →  │  MettaValue    │                 │ │
│  │  │  (syntax)      │    │  (runtime)     │                 │ │
│  │  └────────────────┘    └────────────────┘                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Evaluation Engine                          │ │
│  │  backend/eval/*.rs                                         │ │
│  │  ┌────────────────┐    ┌────────────────┐                 │ │
│  │  │  Environment   │ ←→ │  MORK Space    │                 │ │
│  │  │  (bindings)    │    │  (patterns)    │                 │ │
│  │  └────────────────┘    └────────────────┘                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                       │                                          │
│                       ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                PathMap Integration                         │ │
│  │  pathmap_par_integration.rs                                │ │
│  │  ┌────────────────┐    ┌────────────────┐                 │ │
│  │  │  MettaState    │ ↔  │  Rholang Par   │                 │ │
│  │  └────────────────┘    └────────────────┘                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Parser Pipeline

### Tree-Sitter Grammar

MeTTaTron uses a custom Tree-sitter grammar for MeTTa:

```
tree-sitter-metta/
├── grammar.js          # Grammar definition
├── src/
│   ├── parser.c        # Generated parser
│   └── scanner.c       # Custom lexer
└── bindings/
    └── rust/lib.rs     # Rust bindings
```

### MettaExpr IR

The intermediate representation from parsing:

```rust
/// Parsed MeTTa expression with source spans
pub enum MettaExpr {
    /// Named atom (symbol, variable, operator)
    Atom(String, Span),

    /// String literal
    String(String, Span),

    /// Integer literal
    Integer(i64, Span),

    /// Float literal
    Float(f64, Span),

    /// S-expression (parenthesized list)
    List(Vec<MettaExpr>, Span),

    /// Quoted expression
    Quoted(Box<MettaExpr>, Span),
}
```

### Parsing Flow

```rust
/// Parse MeTTa source into IR
pub fn parse(&mut self, source: &str) -> Result<Vec<MettaExpr>, SyntaxError> {
    let tree = self.parser.parse(source, None)
        .ok_or(SyntaxError::parse_failed())?;

    let root = tree.root_node();
    self.convert_node(root, source)
}
```

---

## Backend Components

### Directory Structure

```
src/backend/
├── compile.rs           # MettaExpr → MettaValue compilation
├── environment.rs       # Runtime environment (bindings, space)
├── symbol.rs            # Symbol interning
├── grounded.rs          # Grounded function support
├── mork_convert.rs      # MettaValue ↔ MORK Expr conversion
├── fuzzy_match.rs       # Fuzzy pattern matching
├── varint_encoding.rs   # Variable-length integer encoding
├── models/
│   ├── metta_state.rs   # Evaluation state
│   ├── metta_value.rs   # Runtime value representation
│   ├── bindings.rs      # Variable bindings
│   └── space_handle.rs  # Space reference
├── eval/
│   ├── evaluation.rs    # Main evaluation loop
│   ├── control_flow.rs  # if/let/match
│   ├── bindings.rs      # let/unify/match
│   ├── types.rs         # Type checking
│   ├── space.rs         # Space operations
│   └── ...              # Other operations
└── modules/
    ├── loader.rs        # Module loading
    └── package.rs       # Package management
```

### MettaValue

The runtime value representation:

```rust
/// Runtime MeTTa value
#[derive(Clone, Debug, PartialEq)]
pub enum MettaValue {
    /// Named symbol/variable
    Atom(String),

    /// Boolean value
    Bool(bool),

    /// 64-bit integer
    Long(i64),

    /// 64-bit float
    Float(f64),

    /// String value
    String(String),

    /// Nil value
    Nil,

    /// S-expression (operator + arguments)
    SExpr(Vec<MettaValue>),

    /// Type annotation
    Type(Box<MettaValue>),

    /// Error with message and details
    Error(String, Box<MettaValue>),

    /// Conjunction of goals
    Conjunction(Vec<MettaValue>),
}
```

### MettaState

The evaluation state machine:

```rust
/// Complete MeTTa evaluation state
pub struct MettaState {
    /// Expressions pending evaluation
    pub pending: Vec<MettaValue>,

    /// Current environment (bindings + space)
    pub environment: Arc<Environment>,

    /// Accumulated results
    pub results: Vec<MettaValue>,

    /// Evaluation statistics
    pub stats: EvalStats,
}

impl MettaState {
    /// Create new state from compiled expressions
    pub fn new_compiled(exprs: Vec<MettaValue>) -> Self {
        Self {
            pending: exprs,
            environment: Arc::new(Environment::new()),
            results: Vec::new(),
            stats: EvalStats::default(),
        }
    }
}
```

---

## MORK Integration

MORK (MeTTa Optimal Reduction Kernel) provides high-performance pattern matching.

### Conversion Context

```rust
/// Track variables during MettaValue → MORK conversion
pub struct ConversionContext {
    /// Variable names → De Bruijn indices
    pub var_map: HashMap<String, u8>,

    /// De Bruijn indices → variable names
    pub var_names: Vec<String>,
}

impl ConversionContext {
    /// Get or create De Bruijn index for a variable
    pub fn get_or_create_var(&mut self, name: &str) -> Result<Option<u8>, String> {
        if let Some(&idx) = self.var_map.get(name) {
            Ok(Some(idx))  // Existing variable
        } else {
            if self.var_names.len() >= 64 {
                return Err("Too many variables (max 64)".to_string());
            }
            let idx = self.var_names.len() as u8;
            self.var_map.insert(name.to_string(), idx);
            self.var_names.push(name.to_string());
            Ok(None)  // New variable
        }
    }
}
```

### MettaValue to MORK Bytes

```rust
/// Convert MettaValue to MORK s-expression bytes
pub fn metta_to_mork_bytes(
    value: &MettaValue,
    space: &Space,
    ctx: &mut ConversionContext,
) -> Result<Vec<u8>, String> {
    const BUFFER_SIZE: usize = 262144;  // 256KB for large expressions
    let mut buffer = vec![0u8; BUFFER_SIZE];
    let expr = Expr { ptr: buffer.as_mut_ptr() };
    let mut ez = ExprZipper::new(expr);

    write_metta_value(value, space, ctx, &mut ez)?;

    Ok(buffer[..ez.loc].to_vec())
}

fn write_metta_value(
    value: &MettaValue,
    space: &Space,
    ctx: &mut ConversionContext,
    ez: &mut ExprZipper,
) -> Result<(), String> {
    match value {
        MettaValue::Atom(name) => {
            // Handle variables vs symbols
            if name.starts_with('$') || name.starts_with('&') {
                // Variable - use De Bruijn encoding
                let var_id = &name[1..];
                match ctx.get_or_create_var(var_id)? {
                    None => ez.write_new_var(),
                    Some(idx) => ez.write_var_ref(idx),
                }
            } else {
                // Symbol - write as bytes
                write_symbol(name.as_bytes(), space, ez)?;
            }
        }
        MettaValue::SExpr(items) => {
            ez.write_sexpr_start(items.len())?;
            for item in items {
                write_metta_value(item, space, ctx, ez)?;
            }
            ez.write_sexpr_end();
        }
        // ... other cases
    }
    Ok(())
}
```

### Pattern Query

```rust
/// Query MORK space with pattern
pub fn query_pattern(
    space: &Space,
    pattern: &MettaValue,
) -> Result<Vec<Bindings>, String> {
    let mut ctx = ConversionContext::new();
    let pattern_bytes = metta_to_mork_bytes(pattern, space, &mut ctx)?;

    let results = space.query_multi(&pattern_bytes);

    // Convert MORK bindings back to MettaValue bindings
    results.into_iter()
        .map(|mork_bindings| convert_bindings(&mork_bindings, &ctx))
        .collect()
}
```

---

## PathMap Par Integration

MeTTaTron bridges MeTTa and Rholang through PathMap-based Par conversion.

### MettaValue to Par

```rust
/// Convert MettaValue to Rholang Par
pub fn metta_value_to_par(value: &MettaValue) -> Par {
    match value {
        MettaValue::Atom(s) => {
            // Atoms as strings
            create_string_par(s.clone())
        }
        MettaValue::Bool(b) => {
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::GBool(*b)),
            }])
        }
        MettaValue::Long(n) => create_int_par(*n),
        MettaValue::String(s) => {
            // Escape and quote strings
            create_string_par(format!(
                "\"{}\"",
                s.replace("\\", "\\\\").replace("\"", "\\\"")
            ))
        }
        MettaValue::SExpr(items) => {
            // S-expressions as Rholang tuples
            let item_pars: Vec<Par> = items.iter()
                .map(metta_value_to_par)
                .collect();

            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::ETupleBody(ETuple {
                    ps: item_pars,
                    locally_free: Vec::new(),
                    connective_used: false,
                })),
            }])
        }
        MettaValue::Error(msg, details) => {
            // Errors as ("error", msg, details) tuples
            let tag_par = create_string_par("error".to_string());
            let msg_par = create_string_par(msg.clone());
            let details_par = metta_value_to_par(details);

            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(ExprInstance::ETupleBody(ETuple {
                    ps: vec![tag_par, msg_par, details_par],
                    locally_free: Vec::new(),
                    connective_used: false,
                })),
            }])
        }
        // ... other cases
    }
}
```

### Magic Numbers

Special byte array markers for MeTTa-specific data:

```rust
// MeTTa Environment byte array markers
const METTA_MULTIPLICITIES_MAGIC: &[u8] = b"MTTM";  // Multiplicities
const METTA_SPACE_MAGIC: &[u8] = b"MTTS";           // Space
const METTA_LARGE_EXPRS_MAGIC: &[u8] = b"MTTL";     // Large expressions
```

### State Conversion

```rust
/// Convert complete MettaState to Rholang Par
pub fn metta_state_to_par(state: &MettaState) -> Par {
    // Convert results
    let results_par = if state.results.len() == 1 {
        metta_value_to_par(&state.results[0])
    } else {
        let result_pars: Vec<Par> = state.results.iter()
            .map(metta_value_to_par)
            .collect();
        create_list_par(result_pars)
    };

    // Include environment if non-empty
    if state.environment.is_empty() {
        results_par
    } else {
        let env_par = environment_to_par(&state.environment);
        create_tuple_par(vec![results_par, env_par])
    }
}

/// Convert Par back to MettaState
pub fn par_to_metta_state(par: &Par) -> Result<MettaState, ConversionError> {
    // Reverse conversion...
}
```

---

## Evaluation Engine

### Trampoline Design

MeTTaTron uses an iterative trampoline for lazy evaluation:

```rust
/// Single evaluation step
pub enum EvalStep {
    /// Evaluation complete
    Done(MettaValue),

    /// Continue with new expression
    Continue(MettaValue, Environment),

    /// Fork into multiple alternatives
    Fork(Vec<(MettaValue, Environment)>),

    /// Evaluation error
    Error(String),
}

/// Main evaluation loop
pub fn evaluate(state: &mut MettaState) -> Result<(), EvalError> {
    while let Some(expr) = state.pending.pop() {
        match eval_step(&expr, &state.environment)? {
            EvalStep::Done(result) => {
                state.results.push(result);
            }
            EvalStep::Continue(next, env) => {
                state.environment = Arc::new(env);
                state.pending.push(next);
            }
            EvalStep::Fork(alternatives) => {
                for (alt_expr, alt_env) in alternatives {
                    // Handle nondeterminism...
                }
            }
            EvalStep::Error(msg) => {
                return Err(EvalError::Runtime(msg));
            }
        }
    }
    Ok(())
}
```

### Operation Dispatch

```rust
/// Dispatch evaluation based on expression head
fn eval_step(
    expr: &MettaValue,
    env: &Environment,
) -> Result<EvalStep, EvalError> {
    match expr {
        MettaValue::SExpr(items) if !items.is_empty() => {
            let head = &items[0];
            let args = &items[1..];

            match head {
                MettaValue::Atom(op) => dispatch_op(op, args, env),
                _ => {
                    // Try to evaluate head first
                    EvalStep::Continue(/* ... */)
                }
            }
        }
        MettaValue::Atom(name) if name.starts_with('$') => {
            // Variable lookup
            match env.lookup(name) {
                Some(value) => EvalStep::Done(value.clone()),
                None => EvalStep::Done(expr.clone()),
            }
        }
        _ => EvalStep::Done(expr.clone()),
    }
}

fn dispatch_op(
    op: &str,
    args: &[MettaValue],
    env: &Environment,
) -> Result<EvalStep, EvalError> {
    match op {
        "eval" => eval_eval(args, env),
        "chain" => eval_chain(args, env),
        "match" => eval_match(args, env),
        "let" => eval_let(args, env),
        "if" => eval_if(args, env),
        "+" | "add" => eval_add(args, env),
        "-" | "sub" => eval_sub(args, env),
        // ... other operations
        _ => {
            // Check for user-defined functions
            eval_user_function(op, args, env)
        }
    }
}
```

---

## Type System

### Type Declarations

```metta
; Type annotations
(: factorial (-> Nat Nat))
(: add (-> Number Number Number))

; Type-annotated definitions
(= (factorial 0) 1)
(= (factorial $n) (* $n (factorial (- $n 1))))
```

### Type Checking

```rust
// From backend/eval/types.rs

/// Check if value matches expected type
pub fn check_type(
    value: &MettaValue,
    expected: &MettaValue,
    env: &Environment,
) -> bool {
    match (value, expected) {
        (MettaValue::Atom(name), MettaValue::Atom(ty)) => {
            match ty.as_str() {
                "Atom" => true,
                "Symbol" => !name.starts_with('$'),
                "Variable" => name.starts_with('$'),
                "Number" => is_number_atom(name),
                _ => {
                    // Query type from space
                    env.has_type(value, expected)
                }
            }
        }
        (MettaValue::Long(_), MettaValue::Atom(ty)) => {
            matches!(ty.as_str(), "Number" | "Int" | "Long")
        }
        (MettaValue::Float(_), MettaValue::Atom(ty)) => {
            matches!(ty.as_str(), "Number" | "Float")
        }
        (MettaValue::SExpr(_), MettaValue::Atom(ty)) => {
            ty == "Expression"
        }
        _ => false,
    }
}
```

---

## Rholang Bridge

### System Contract

MeTTaTron integrates with Rholang via the `rho:metta:compile` contract:

```rholang
new compile(`rho:metta:compile`) in {
  compile!("(= (add $x $y) (+ $x $y))", *result) |
  for (@compiled <- result) {
    // compiled is the MeTTa knowledge base as Par
  }
}
```

### Integration Points

| Contract | Purpose |
|----------|---------|
| `rho:metta:compile` | Compile MeTTa source to Par |
| `rho:metta:eval` | Evaluate MeTTa expression |
| `rho:metta:query` | Query MeTTa knowledge base |

### FFI Templates

```rust
// From integration/templates/rholang_handler.rs

/// Handle MeTTa compilation request from Rholang
pub fn handle_compile(
    source: &str,
    return_channel: Par,
) -> Result<Par, Error> {
    let state = compile(source)?;
    let par = metta_state_to_par(&state);
    send_to_channel(return_channel, par)
}
```

---

## Relevance to MeTTaIL

### Integration Points

MeTTaTron provides key infrastructure for MeTTaIL:

| Component | MeTTaIL Use |
|-----------|-------------|
| MORK pattern matching | Predicate evaluation |
| PathMap conversion | Rholang type exchange |
| Type checking | Extend with OSLF predicates |
| Evaluation engine | Behavioral type verification |

### Extension Strategy

1. **Predicate Layer**: Add OSLF predicates atop existing type checks
2. **Behavioral Tracking**: Instrument evaluation for reduction graph
3. **PathMap Types**: Encode behavioral types in Par representation
4. **MORK Queries**: Use MORK for efficient predicate evaluation

### Code Integration Example

```rust
// Proposed MeTTaIL extension to MeTTaTron

impl MettaState {
    /// Check behavioral predicate
    pub fn satisfies_predicate(&self, pred: &Predicate) -> bool {
        match pred {
            Predicate::Terminates => {
                // Check if evaluation will terminate
                self.check_termination()
            }
            Predicate::TypeSafe(ty) => {
                // Check type safety
                self.check_type_safe(ty)
            }
            Predicate::Behavioral(spec) => {
                // Build reduction graph and check
                let graph = self.build_reduction_graph();
                graph.satisfies(spec)
            }
        }
    }
}
```

---

## Summary

MeTTaTron provides:

1. **Tree-sitter parsing** for robust MeTTa syntax handling
2. **MORK integration** for high-performance pattern matching
3. **PathMap conversion** for Rholang interoperability
4. **Trampoline evaluation** for lazy computation
5. **Type checking** infrastructure for extension

For MeTTaIL integration:
- Use MORK for efficient predicate queries
- Extend type checking with OSLF predicates
- Add behavioral tracking to evaluation engine
- Leverage PathMap for cross-language type exchange

---

## References

- Repository: `/home/dylon/Workspace/f1r3fly.io/MeTTa-Compiler/`
- See [02-hyperon-experimental.md](./02-hyperon-experimental.md) for comparison
- See [04-mork-pathmap-integration.md](./04-mork-pathmap-integration.md) for storage details
