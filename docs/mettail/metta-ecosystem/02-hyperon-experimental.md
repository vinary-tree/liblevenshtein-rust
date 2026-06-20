[← Documentation Index](../../README.md)

# Hyperon-Experimental: MeTTa Reference Implementation

This document provides a technical analysis of hyperon-experimental, the official
reference implementation of MeTTa in Rust. Understanding this codebase is essential
for implementing MeTTaIL's semantic type checking.

**Location**: `/home/dylon/Workspace/f1r3fly.io/hyperon-experimental/`

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Crate Organization](#crate-organization)
3. [Four Atom Meta-Types in Code](#four-atom-meta-types-in-code)
4. [Pattern Matching Implementation](#pattern-matching-implementation)
5. [Interpreter Architecture](#interpreter-architecture)
6. [Type System Implementation](#type-system-implementation)
7. [Space Operations](#space-operations)
8. [Grounding and External Functions](#grounding-and-external-functions)
9. [Relevance to MeTTaIL](#relevance-to-mettail)

---

## Repository Structure

```
hyperon-experimental/
├── hyperon-atom/           # Core atom types and matching
│   └── src/
│       ├── lib.rs          # Atom enum, SymbolAtom, VariableAtom
│       ├── matcher.rs      # Pattern matching, Bindings, BindingsSet
│       ├── gnd/            # Grounded atom implementations
│       │   ├── number.rs   # Numeric types
│       │   ├── str.rs      # String type
│       │   └── bool.rs     # Boolean type
│       └── serial.rs       # Serialization
├── hyperon-space/          # Space (Atomspace) implementations
│   └── src/
│       ├── lib.rs          # Space trait
│       └── index/          # Indexing structures
│           └── trie.rs     # Trie-based indexing
├── lib/                    # Main MeTTa library
│   └── src/
│       ├── metta/
│       │   ├── interpreter.rs  # Stack-based evaluator
│       │   ├── types.rs        # Type checking
│       │   ├── text.rs         # Text parsing
│       │   └── runner/         # Runtime components
│       │       ├── stdlib/     # Standard library
│       │       └── builtin_mods/
│       └── space/
├── c/                      # C API bindings
├── repl/                   # Interactive REPL
└── python/                 # Python bindings
```

---

## Crate Organization

### hyperon-atom

The foundational crate containing atom representations:

| Module | Purpose |
|--------|---------|
| `lib.rs` | `Atom` enum, `SymbolAtom`, `VariableAtom`, `ExpressionAtom` |
| `matcher.rs` | `Bindings`, `BindingsSet`, unification algorithm |
| `gnd/` | Grounded atom implementations |
| `serial.rs` | Atom serialization/deserialization |

### hyperon-space

Space (knowledge base) implementations:

| Module | Purpose |
|--------|---------|
| `lib.rs` | `Space` trait, `DynSpace` |
| `index/trie.rs` | Trie-based atom indexing |
| `grounding/` | Grounding space with type information |

### lib (hyperon)

The main MeTTa library:

| Module | Purpose |
|--------|---------|
| `metta/interpreter.rs` | Stack-based evaluation |
| `metta/types.rs` | Type checking and inference |
| `metta/runner/` | Runtime, modules, standard library |

---

## Four Atom Meta-Types in Code

The `Atom` enum in `hyperon-atom/src/lib.rs` defines the four meta-types:

```rust
/// The four fundamental atom meta-types
#[derive(Clone, Debug, PartialEq)]
pub enum Atom {
    /// Named constant (symbol)
    Symbol(SymbolAtom),

    /// Pattern variable
    Variable(VariableAtom),

    /// Ordered list of atoms
    Expression(ExpressionAtom),

    /// Foreign data/function
    Grounded(GroundedAtom),
}
```

### SymbolAtom

Represents named constants:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolAtom {
    name: UniqueString,  // Interned string for efficiency
}

impl SymbolAtom {
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
}

// Construction
let sym = Atom::sym("foo");          // Named symbol
let op = Atom::sym("+");             // Operator symbol
let ty = Atom::sym("Type");          // Type name
```

### VariableAtom

Represents pattern variables:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariableAtom {
    name: UniqueString,
    id: Option<u64>,  // For alpha-renaming (unique variables)
}

impl VariableAtom {
    /// Create a unique variable (for avoiding capture)
    pub fn make_unique(&self) -> Self {
        Self {
            name: self.name.clone(),
            id: Some(new_unique_id()),
        }
    }
}

// Construction
let var = Atom::var("x");            // Variable $x
let anon = Atom::var("_");           // Anonymous variable
```

### ExpressionAtom

Represents ordered lists of atoms:

```rust
#[derive(Debug, Clone)]
pub struct ExpressionAtom {
    children: CowArray<Atom>,  // Copy-on-write for efficiency
}

impl ExpressionAtom {
    pub fn children(&self) -> &[Atom] {
        &self.children
    }

    pub fn into_children(self) -> Vec<Atom> {
        self.children.into_vec()
    }
}

// Construction
let expr = Atom::expr(vec![
    Atom::sym("+"),
    Atom::sym("1"),
    Atom::sym("2"),
]);  // (+ 1 2)
```

### GroundedAtom

Represents foreign data and functions:

```rust
/// Trait for grounded atom behavior
pub trait Grounded: Display + Debug {
    /// Type of this grounded atom
    fn type_(&self) -> Atom;

    /// Custom matching logic (optional)
    fn match_(&self, other: &Atom) -> matcher::MatchResultIter {
        match_by_equality(self, other)
    }

    /// Execute as function (optional)
    fn execute(&self, args: &[Atom]) -> Result<Vec<Atom>, ExecError> {
        Err(ExecError::NoReduce)
    }
}

// Example: grounded number
struct Number(i64);

impl Grounded for Number {
    fn type_(&self) -> Atom {
        Atom::sym("Number")
    }

    fn execute(&self, args: &[Atom]) -> Result<Vec<Atom>, ExecError> {
        // Numbers are not executable
        Err(ExecError::NoReduce)
    }
}
```

---

## Pattern Matching Implementation

The pattern matching algorithm is in `hyperon-atom/src/matcher.rs`.

### Bindings

A single variable assignment:

```rust
/// Map from variables to atoms
#[derive(Clone, Debug, Default)]
pub struct Bindings {
    map: HashMap<VariableAtom, Atom>,
}

impl Bindings {
    /// Add a binding (returns None if conflict)
    pub fn add_var_binding(&mut self, var: &VariableAtom, atom: Atom)
        -> Option<&mut Self>
    {
        match self.map.entry(var.clone()) {
            Entry::Occupied(e) => {
                if e.get() == &atom { Some(self) } else { None }
            }
            Entry::Vacant(e) => {
                e.insert(atom);
                Some(self)
            }
        }
    }

    /// Merge two binding sets
    pub fn merge(&self, other: &Bindings) -> Option<Bindings> {
        // Returns None if bindings conflict
    }
}
```

### BindingsSet

Set of all possible unification results:

```rust
/// Set of possible bindings (for nondeterministic matching)
#[derive(Clone, Debug, Default)]
pub struct BindingsSet(Vec<Bindings>);

impl BindingsSet {
    pub fn single() -> Self {
        Self(vec![Bindings::new()])
    }

    pub fn empty() -> Self {
        Self(vec![])
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
```

### Matching Algorithm

```rust
/// Match pattern against target, returning all valid bindings
pub fn match_atoms(pattern: &Atom, target: &Atom) -> MatchResultIter {
    match (pattern, target) {
        // Variable matches anything
        (Atom::Variable(var), _) => {
            let mut bindings = Bindings::new();
            bindings.add_var_binding(var, target.clone());
            MatchResultIter::from_bindings(bindings)
        }

        // Symbols must be equal
        (Atom::Symbol(p), Atom::Symbol(t)) if p == t => {
            MatchResultIter::from_bindings(Bindings::new())
        }

        // Expressions match recursively
        (Atom::Expression(p_expr), Atom::Expression(t_expr)) => {
            match_expressions(p_expr.children(), t_expr.children())
        }

        // Grounded atoms use custom match
        (Atom::Grounded(p), target) => {
            p.match_(target)
        }

        // Otherwise no match
        _ => MatchResultIter::empty()
    }
}

/// Match expression children pairwise
fn match_expressions(pattern: &[Atom], target: &[Atom]) -> MatchResultIter {
    if pattern.len() != target.len() {
        return MatchResultIter::empty();
    }

    pattern.iter().zip(target.iter())
        .fold(MatchResultIter::from_bindings(Bindings::new()),
              |acc, (p, t)| acc.flat_map(|b|
                  match_atoms(p, t).filter_map(|b2| b.merge(&b2))))
}
```

### Substitution

Apply bindings to an atom:

```rust
/// Replace variables with their bound values
pub fn apply_bindings_to_atom_move(atom: Atom, bindings: &Bindings) -> Atom {
    match atom {
        Atom::Variable(ref var) => {
            bindings.resolve(var).cloned().unwrap_or(atom)
        }
        Atom::Expression(expr) => {
            Atom::Expression(ExpressionAtom::new(
                expr.into_children()
                    .into_iter()
                    .map(|a| apply_bindings_to_atom_move(a, bindings))
                    .collect()
            ))
        }
        _ => atom
    }
}
```

---

## Interpreter Architecture

The interpreter in `lib/src/metta/interpreter.rs` uses a stack-based trampoline
design for lazy evaluation.

### Stack Structure

```rust
#[derive(Debug, Clone)]
struct Stack {
    prev: Option<Rc<RefCell<Self>>>,  // Parent frame
    atom: Atom,                        // Current expression
    ret: ReturnHandler,                // Continuation
    finished: bool,                    // Is this frame done?
    vars: Variables,                   // Scoped variables
    depth: usize,                      // Stack depth (for limits)
}

/// Handler called when nested operation completes
type ReturnHandler = fn(
    Rc<RefCell<Stack>>,  // Current stack
    Atom,                // Result atom
    Bindings             // Variable bindings
) -> Option<(Stack, Bindings)>;
```

### InterpreterState

```rust
#[derive(Debug)]
pub struct InterpreterState {
    /// Alternatives to evaluate (nondeterminism)
    plan: Vec<InterpretedAtom>,

    /// Completed results
    finished: Vec<Atom>,

    /// Evaluation context (space reference)
    context: InterpreterContext,

    /// Maximum stack depth
    max_stack_depth: usize,
}
```

### Evaluation Step

```rust
impl InterpreterState {
    /// Execute one step of interpretation
    pub fn interpret_step(&mut self) -> bool {
        if let Some(interpreted) = self.plan.pop() {
            let results = self.step(interpreted);

            for result in results {
                if result.is_finished() {
                    self.finished.push(result.into_atom());
                } else {
                    self.plan.push(result);
                }
            }

            true  // More work to do
        } else {
            false // Finished
        }
    }

    fn step(&self, interpreted: InterpretedAtom) -> Vec<InterpretedAtom> {
        // Handle each operation type...
    }
}
```

### Core Operations

The seven minimal operations:

| Operation | Implementation |
|-----------|----------------|
| `eval` | `interpret_eval()` - evaluate expression |
| `evalc` | Conditional evaluation |
| `chain` | `interpret_chain()` - sequential composition |
| `function`/`return` | `interpret_function()` - abstraction |
| `unify` | `interpret_unify()` - pattern matching |
| `cons-atom`/`decons-atom` | List operations |
| `collapse-bind`/`superpose-bind` | Nondeterminism |

---

## Type System Implementation

Type checking is in `lib/src/metta/types.rs`.

### Type Declarations

Types are declared using the `:` symbol:

```metta
; Type annotation
(: x Int)
(: add (-> Int Int Int))
```

### Type Queries

```rust
/// Query if atom has type
fn query_has_type(space: &DynSpace, atom: &Atom, typ: &Atom) -> BindingsSet {
    space.borrow().query(&typeof_query(atom, typ))
}

fn typeof_query(atom: &Atom, typ: &Atom) -> Atom {
    Atom::expr(vec![HAS_TYPE_SYMBOL, atom.clone(), typ.clone()])
}
```

### Subtyping

```rust
/// Query super-types of a type
fn query_super_types(space: &DynSpace, sub_type: &Atom) -> Vec<Atom> {
    let var_x = VariableAtom::new("X").make_unique();
    let super_types = space.borrow().query(
        &isa_query(&sub_type, &Atom::Variable(var_x.clone()))
    );
    // Extract bound values...
}

fn isa_query(sub_type: &Atom, super_type: &Atom) -> Atom {
    Atom::expr(vec![SUB_TYPE_SYMBOL, sub_type.clone(), super_type.clone()])
}
```

### Function Type Checking

```rust
/// Check if type is a function type (-> A B)
pub fn is_func(typ: &Atom) -> bool {
    match typ {
        Atom::Expression(expr) => {
            expr.children().first() == Some(&ARROW_SYMBOL)
                && expr.children().len() > 1
        },
        _ => false,
    }
}

/// Check argument types match expected
fn check_arg_types(
    actual: &[Vec<AtomType>],    // Actual arg types
    expected: &[Atom],           // Expected types from signature
    ret_typ: &Atom,              // Return type
    ...
) {
    // Recursively check each argument
}
```

### Meta-Types

Five special meta-types (checked but not explicitly assignable):

```rust
// Meta-type constants
pub const ATOM_TYPE_ATOM: Atom = sym!("Atom");
pub const ATOM_TYPE_SYMBOL: Atom = sym!("Symbol");
pub const ATOM_TYPE_VARIABLE: Atom = sym!("Variable");
pub const ATOM_TYPE_EXPRESSION: Atom = sym!("Expression");
pub const ATOM_TYPE_GROUNDED: Atom = sym!("Grounded");

// The undefined type (matches anything)
pub const ATOM_TYPE_UNDEFINED: Atom = sym!("%Undefined%");
```

---

## Space Operations

### Space Trait

```rust
pub trait Space {
    /// Add atom to space
    fn add(&mut self, atom: Atom);

    /// Remove atom from space
    fn remove(&mut self, atom: &Atom) -> bool;

    /// Query with pattern, return all matching bindings
    fn query(&self, pattern: &Atom) -> BindingsSet;

    /// Replace matching atoms
    fn replace(&mut self, pattern: &Atom, replacement: Atom);
}
```

### GroundingSpace

The default space implementation with type information:

```rust
pub struct GroundingSpace {
    atoms: AtomIndex,
    types: HashMap<Atom, Vec<Atom>>,
}

impl GroundingSpace {
    /// Get types of an atom
    pub fn get_types(&self, atom: &Atom) -> &[Atom] {
        self.types.get(atom).map_or(&[], |v| v.as_slice())
    }
}
```

### Trie Indexing

For efficient pattern matching:

```rust
// In hyperon-space/src/index/trie.rs
pub struct TrieIndex {
    root: TrieNode,
}

struct TrieNode {
    children: HashMap<AtomKey, TrieNode>,
    values: Vec<Atom>,
}
```

---

## Grounding and External Functions

### AutoGroundedType

Automatic grounding for simple types:

```rust
/// Automatically implemented for T: 'static + PartialEq + Clone + Debug
pub trait AutoGroundedType {}

impl<T: 'static + PartialEq + Clone + Debug> AutoGroundedType for T {}

// Usage
let num = Atom::value(42i64);  // Auto-grounded integer
```

### Custom Grounded Functions

```rust
use hyperon_atom::gnd::GroundedFunctionAtom;

// Create a grounded function
let add_op = GroundedFunctionAtom::new(
    "+".into(),
    expr!("->" "Number" "Number" "Number"),  // Type
    |args: &[Atom]| -> Result<Vec<Atom>, ExecError> {
        // Extract numbers from args
        let a = extract_number(&args[0])?;
        let b = extract_number(&args[1])?;
        Ok(vec![Atom::value(a + b)])
    }
);
```

### Execution Flow

```rust
impl Grounded for MyOperation {
    fn execute(&self, args: &[Atom]) -> Result<Vec<Atom>, ExecError> {
        // Validate arguments
        if args.len() != self.expected_args() {
            return Err(ExecError::Runtime(
                format!("Expected {} args, got {}",
                        self.expected_args(), args.len())
            ));
        }

        // Perform operation
        let result = self.compute(args)?;

        Ok(vec![result])
    }
}
```

---

## Relevance to MeTTaIL

### What MeTTaIL Can Reuse

| Component | Reuse Strategy |
|-----------|----------------|
| `Atom` enum | Direct import or compatible definition |
| `matcher.rs` | Use for predicate unification |
| Type queries | Extend with OSLF predicates |
| `Space` trait | Add behavioral type storage |

### Extension Points for MeTTaIL

1. **Predicate Types**: Extend `AtomType` with predicates
2. **Behavioral Checking**: Add behavioral analysis to `InterpreterState`
3. **Graph Representation**: Track reduction graph alongside evaluation
4. **OSLF Integration**: Implement presheaf queries using space infrastructure

### Key Interfaces

```rust
// MeTTaIL could implement a type-checking wrapper
pub trait TypedSpace: Space {
    /// Check if atom satisfies behavioral predicate
    fn satisfies_predicate(&self, atom: &Atom, pred: &Predicate) -> bool;

    /// Get behavioral type of process
    fn behavioral_type(&self, proc: &Atom) -> BehavioralType;
}

// Predicate evaluation using MeTTa infrastructure
pub fn evaluate_predicate(
    space: &DynSpace,
    predicate: &Atom,
    bindings: &Bindings
) -> BindingsSet {
    // Use existing query infrastructure
    space.borrow().query(&predicate)
}
```

---

## Summary

Hyperon-experimental provides:

1. **Atom representation**: Four meta-types with efficient implementations
2. **Pattern matching**: Sound, complete unification with occurs check
3. **Stack-based interpreter**: Lazy evaluation with nondeterminism
4. **Type system**: Gradual typing with meta-types
5. **Space abstraction**: Extensible knowledge store interface
6. **Grounding**: Foreign function interface

For MeTTaIL integration:
- Reuse atom types and matching infrastructure
- Extend type system with OSLF predicates
- Add behavioral analysis layer
- Track reduction graphs for behavioral types

---

## References

- Repository: `/home/dylon/Workspace/f1r3fly.io/hyperon-experimental/`
- Minimal MeTTa docs: `docs/minimal-metta.md` in repository
- See [01-opencog-hyperon.md](./01-opencog-hyperon.md) for theoretical foundations
- See [03-mettatron.md](./03-mettatron.md) for F1R3FLY.io's implementation
