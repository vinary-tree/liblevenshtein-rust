# Layer 1: Analysis Layer

The analysis layer computes program properties needed by simplification rules using Ascent datalog.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Overview

The analysis layer runs before rule application to compute facts that rules can query. This enables analysis-dependent simplifications like dead code elimination and constant propagation.

```
Input: Proc + TheoryDef
         ↓
┌──────────────────────────────────────┐
│         ANALYSIS LAYER               │
│                                      │
│  ┌────────────────────────────────┐  │
│  │     Ascent Datalog Program     │  │
│  │                                │  │
│  │  ┌──────────┐  ┌──────────┐   │  │
│  │  │Reachable │  │  Live    │   │  │
│  │  │  Facts   │  │  Facts   │   │  │
│  │  └──────────┘  └──────────┘   │  │
│  │                                │  │
│  │  ┌──────────┐  ┌──────────┐   │  │
│  │  │ Constant │  │   Cost   │   │  │
│  │  │  Values  │  │ Estimates│   │  │
│  │  └──────────┘  └──────────┘   │  │
│  │                                │  │
│  └────────────────────────────────┘  │
│                                      │
└──────────────────────────────────────┘
         ↓
Output: AnalysisFacts
```

---

## Module Structure

```
simplification/analysis/
├── mod.rs              # Analysis orchestration
├── relations.rs        # Ascent relation declarations
├── reachability.rs     # What code is reachable from entry
├── liveness.rs         # What variables are live at each point
├── dataflow.rs         # Value propagation (constant folding)
└── cost.rs             # Static cost estimation
```

---

## Ascent Relation Declarations

### Base Relations (Input)

```rust
// relations.rs

ascent! {
    // Program structure relations (populated from AST)
    relation entry_point(ProcId);
    relation step(ProcId, ProcId);           // Control flow edge
    relation contains_var(ProcId, Var);       // Variable usage
    relation defines_var(ProcId, Var);        // Variable definition
    relation expr_value(ExprId, ProcId);      // Expression location
    relation literal_value(ExprId, Value);    // Constant literals

    // Subterm decomposition
    relation subterm(ProcId, ProcId);         // Parent-child in AST
    relation par_left(ProcId, ProcId);        // P | Q -> P
    relation par_right(ProcId, ProcId);       // P | Q -> Q
}
```

### Derived Relations (Computed)

```rust
// Reachability analysis
relation reachable(ProcId);

reachable(p) <-- entry_point(p);
reachable(q) <-- reachable(p), step(p, q);

// Dead code detection
relation dead_code(ProcId);

dead_code(p) <-- subterm(_, p), !reachable(p);
```

```rust
// Liveness analysis (backward dataflow)
relation live_at(Var, ProcId);

// Variable is live where it's used
live_at(v, p) <-- contains_var(p, v), reachable(p);

// Propagate liveness backward through control flow
live_at(v, p) <-- live_at(v, q), step(p, q), !defines_var(p, v);
```

```rust
// Constant propagation (forward dataflow)
relation constant_value(ExprId, Value);

// Base case: literal values
constant_value(e, val) <-- literal_value(e, val);

// Propagate through operations
constant_value(e, fold(op, v1, v2)) <--
    binary_expr(e, op, e1, e2),
    constant_value(e1, v1),
    constant_value(e2, v2);
```

---

## Analysis Passes

### 1. Reachability Analysis

**Purpose**: Determine which code is reachable from the entry point.

```rust
// reachability.rs

impl ReachabilityAnalysis {
    /// Compute reachable set from entry points
    pub fn analyze(proc: &Proc) -> HashSet<ProcId> {
        let mut program = AscentProgram::new();

        // Populate base relations from AST
        self.populate_entry_points(&mut program, proc);
        self.populate_control_flow(&mut program, proc);

        // Run Ascent to fixpoint
        program.run();

        // Extract reachable set
        program.reachable.iter().map(|(p,)| *p).collect()
    }

    fn populate_control_flow(&self, program: &mut AscentProgram, proc: &Proc) {
        // Walk AST and add step(from, to) edges
        match proc {
            Proc::Seq(p1, p2) => {
                program.step.push((p1.id(), p2.id()));
            }
            Proc::If(cond, then_branch, else_branch) => {
                program.step.push((cond.id(), then_branch.id()));
                program.step.push((cond.id(), else_branch.id()));
            }
            // ... other cases
        }
    }
}
```

### 2. Liveness Analysis

**Purpose**: Determine which variables are live (will be used) at each program point.

```rust
// liveness.rs

impl LivenessAnalysis {
    /// Compute live variables at each program point
    pub fn analyze(proc: &Proc) -> HashMap<ProcId, HashSet<Var>> {
        let mut program = AscentProgram::new();

        // Populate use/def information
        self.populate_uses(&mut program, proc);
        self.populate_defs(&mut program, proc);
        self.populate_control_flow(&mut program, proc);

        program.run();

        // Convert to map form
        let mut result = HashMap::new();
        for (var, proc_id) in program.live_at.iter() {
            result.entry(*proc_id).or_insert_with(HashSet::new).insert(*var);
        }
        result
    }
}
```

### 3. Constant Propagation

**Purpose**: Track constant values through the program for constant folding.

```rust
// dataflow.rs

impl ConstantPropagation {
    /// Compute constant values for expressions
    pub fn analyze(proc: &Proc) -> HashMap<ExprId, Value> {
        let mut program = AscentProgram::new();

        // Populate literal values
        self.populate_literals(&mut program, proc);
        self.populate_operations(&mut program, proc);

        program.run();

        program.constant_value.iter()
            .map(|(e, v)| (*e, v.clone()))
            .collect()
    }

    fn fold(op: &BinaryOp, v1: &Value, v2: &Value) -> Option<Value> {
        match (op, v1, v2) {
            (BinaryOp::Plus, Value::Int(a), Value::Int(b)) => Some(Value::Int(a + b)),
            (BinaryOp::Mult, Value::Int(a), Value::Int(b)) => Some(Value::Int(a * b)),
            (BinaryOp::And, Value::Bool(a), Value::Bool(b)) => Some(Value::Bool(*a && *b)),
            // ... other operations
            _ => None,
        }
    }
}
```

### 4. Cost Estimation

**Purpose**: Estimate the cost of program fragments for optimization decisions.

```rust
// cost.rs

impl CostEstimation {
    /// Estimate cost of each program fragment
    pub fn analyze(proc: &Proc) -> HashMap<ProcId, Cost> {
        let mut costs = HashMap::new();
        self.estimate_recursive(proc, &mut costs);
        costs
    }

    fn estimate_recursive(&self, proc: &Proc, costs: &mut HashMap<ProcId, Cost>) {
        let cost = match proc {
            Proc::Nil => Cost::ZERO,
            Proc::Num(_) | Proc::Bool(_) | Proc::Var(_) => Cost::unit(1),
            Proc::Par(p1, p2) => {
                self.estimate_recursive(p1, costs);
                self.estimate_recursive(p2, costs);
                costs[&p1.id()] + costs[&p2.id()] + Cost::unit(1)
            }
            Proc::If(cond, then_b, else_b) => {
                self.estimate_recursive(cond, costs);
                self.estimate_recursive(then_b, costs);
                self.estimate_recursive(else_b, costs);
                // Take max of branches for worst-case
                costs[&cond.id()]
                    + Cost::max(costs[&then_b.id()], costs[&else_b.id()])
                    + Cost::unit(2)  // Branch overhead
            }
            // ... other cases
        };
        costs.insert(proc.id(), cost);
    }
}
```

---

## AnalysisFacts Structure

All analysis results are collected into a single structure:

```rust
/// Combined analysis results
pub struct AnalysisFacts {
    /// Set of reachable program points
    pub reachable: HashSet<ProcId>,

    /// Live variables at each program point
    pub live: HashMap<ProcId, HashSet<Var>>,

    /// Expressions with known constant values
    pub constant_values: HashMap<ExprId, Value>,

    /// Program points identified as dead code
    pub dead_code: HashSet<ProcId>,

    /// Cost estimates for program fragments
    pub cost_estimates: HashMap<ProcId, Cost>,
}

impl AnalysisFacts {
    /// Check if a program point is reachable
    pub fn is_reachable(&self, id: ProcId) -> bool {
        self.reachable.contains(&id)
    }

    /// Check if a variable is live at a program point
    pub fn is_live(&self, var: &Var, at: ProcId) -> bool {
        self.live.get(&at)
            .map(|vars| vars.contains(var))
            .unwrap_or(false)
    }

    /// Get constant value if known
    pub fn get_constant(&self, expr: ExprId) -> Option<&Value> {
        self.constant_values.get(&expr)
    }

    /// Check if a program point is dead code
    pub fn is_dead(&self, id: ProcId) -> bool {
        self.dead_code.contains(&id)
    }

    /// Get estimated cost
    pub fn cost(&self, id: ProcId) -> Cost {
        self.cost_estimates.get(&id).cloned().unwrap_or(Cost::UNKNOWN)
    }
}
```

---

## Integration with MORK

Analysis facts can be queried during MORK pattern matching via guard predicates:

```rust
// In rule guard implementation
impl SimplificationRule for DeadCodeElimination {
    fn guard(&self, term: &Proc, facts: &AnalysisFacts) -> bool {
        // Only apply if the term contains dead code
        term.subterms().any(|s| facts.is_dead(s.id()))
    }
}

impl SimplificationRule for ConstantFolding {
    fn guard(&self, term: &Proc, facts: &AnalysisFacts) -> bool {
        // Only apply if we have constant values for operands
        match term {
            Proc::BinaryExpr(_, lhs, rhs) => {
                facts.get_constant(lhs.id()).is_some()
                    && facts.get_constant(rhs.id()).is_some()
            }
            _ => false,
        }
    }
}
```

---

## Performance Considerations

### Incremental Analysis

For large programs, support incremental updates:

```rust
impl AnalysisLayer {
    /// Incrementally update analysis after local change
    pub fn update_incremental(
        &mut self,
        changed_proc: ProcId,
        new_proc: &Proc,
    ) -> AnalysisFacts {
        // Only re-analyze affected portion
        let affected = self.compute_affected_set(changed_proc);
        self.reanalyze_subset(&affected, new_proc)
    }
}
```

### Memoization via PathMap

Cache analysis results in PathMap:

```rust
pub struct AnalysisCache {
    cache: PathMap<CachedAnalysis>,
}

impl AnalysisCache {
    pub fn get_or_compute(
        &mut self,
        proc: &Proc,
        analyzer: &AnalysisLayer,
    ) -> AnalysisFacts {
        let hash = proc.content_hash();
        if let Some(cached) = self.cache.get(&hash) {
            return cached.facts.clone();
        }

        let facts = analyzer.analyze(proc);
        self.cache.insert(hash, CachedAnalysis { facts: facts.clone() });
        facts
    }
}
```

---

## Example: Dead Code Analysis

```rust
// Example program (Rholang-style)
// x!("hello") | 0 | (for (@msg <- y) { z!("received") } | 0)

// After analysis:
// reachable: {x_send, par1, par2, for_recv, z_send}
// dead_code: {}  // No unreachable code in this example
// But nil processes can be eliminated by nil-identity rule
```

---

## Next Steps

- [Rule Application](03-rule-application.md) - Using analysis facts in rules
- [Strategy Selection](04-strategy-selection.md) - Analysis-driven phase

---

## Changelog

- **2025-12-06**: Initial analysis layer documentation
