# Bisimulation-Based Optimization Strategies

Strategies for program optimization that use bisimulation as the soundness criterion.

**Status**: Design Documentation
**Last Updated**: 2025-12-17

---

## Overview

Bisimulation provides a rigorous foundation for program optimization:
- **Soundness**: If $`P \approx  Q`$, replacing P with Q preserves all observable behaviors
- **Completeness**: Bisimulation captures exactly the observable distinctions
- **Compositionality**: Bisimilar subterms can be replaced in any context

This document describes optimization strategies that exploit these properties to reduce code size, improve performance, and optimize for various resource constraints.

---

## Core Principle: Bisimulation as Soundness Criterion

### Optimization Soundness Theorem

**Statement**: An optimization `opt: P → Q` is sound if $`P \approx  Q`$.

**Implication**:
- Any behavior exhibited by P is also exhibited by Q
- Any behavior exhibited by Q is also exhibited by P
- No observable difference exists between P and Q

### Optimization Categories

| Category | Goal | Bisimulation Role |
|----------|------|-------------------|
| Size Reduction | Smaller AST | Replace with bisimilar smaller term |
| Latency Reduction | Faster execution | Replace with bisimilar faster term |
| Parallelism | Better concurrency | Replace with bisimilar parallel term |
| Resource Efficiency | Lower memory/bandwidth | Replace with bisimilar leaner term |

---

## Strategy 1: Dead Code Elimination

### Principle

Remove code that doesn't contribute to observable behavior.

### Applicable Laws

1. **Nil Identity**: $`P | 0 \equiv  P`$
   - Nil processes contribute nothing
   - Safe to remove

2. **Dead Scope Elimination**: $`\text{new} x.P \equiv  P`$ when $`x \notin  \text{FV}(P)`$
   - Unused bindings contribute nothing
   - Safe to remove

3. **Unreachable Code**: Processes that can never execute
   - Dead branches in conditionals
   - Send on channels that are never received

### Bisimulation Justification

**Claim**: Dead code removal preserves bisimilarity.

**Proof Sketch**:
- Dead code has no transitions (never executes)
- Removing it doesn't change the LTS
- Bisimulation relation trivially includes the dead-code-free version

### Implementation

```rust
pub struct DeadCodeEliminator {
    /// Analysis facts for liveness
    liveness: LivenessAnalysis,
}

impl DeadCodeEliminator {
    /// Eliminate dead code while preserving semantics
    pub fn eliminate(&self, proc: &Proc) -> Proc {
        match proc {
            // Nil identity: P | 0 → P
            Proc::Par(p, q) if matches!(q.as_ref(), Proc::Nil) => {
                self.eliminate(p)
            }
            Proc::Par(p, q) if matches!(p.as_ref(), Proc::Nil) => {
                self.eliminate(q)
            }

            // Dead scope: new x.P → P when x not free in P
            Proc::New(x, body) if !body.free_vars().contains(x) => {
                self.eliminate(body)
            }

            // Unreachable send: send on dead channel
            Proc::Send(chan, _, cont) if self.liveness.is_dead_channel(chan) => {
                self.eliminate(cont)
            }

            // Recursive cases
            Proc::Par(p, q) => {
                Proc::Par(
                    Box::new(self.eliminate(p)),
                    Box::new(self.eliminate(q)),
                )
            }
            Proc::New(x, body) => {
                Proc::New(x.clone(), Box::new(self.eliminate(body)))
            }

            // Base case: no elimination possible
            _ => proc.clone()
        }
    }
}
```

### Constraint: Space Optimization

**Metric**: AST node count reduction

**Target**: 10-30% reduction for typical programs

**Trade-off**: Analysis time vs. optimization benefit

---

## Strategy 2: Parallel Fusion

### Principle

Flatten nested parallel compositions for more efficient representation.

### Applicable Laws

1. **Associativity**: $`(P | Q) | R \equiv  P | (Q | R)`$
   - Nested Par can be flattened to list
   - Canonical form: right-associative

2. **Commutativity**: $`P | Q \equiv  Q | P`$
   - Enables reordering for cache locality
   - Enables grouping related processes

### Bisimulation Justification

**Claim**: Parallel fusion preserves bisimilarity.

**Proof**: See [09-rpo-congruence-proofs.md](09-rpo-congruence-proofs.md) for associativity and commutativity proofs.

### Implementation

```rust
pub struct ParallelFusion;

impl ParallelFusion {
    /// Flatten parallel compositions into canonical form
    pub fn fuse(&self, proc: &Proc) -> Proc {
        // 1. Flatten to list
        let components = self.flatten(proc);

        // 2. Filter nil
        let non_nil: Vec<_> = components
            .into_iter()
            .filter(|p| !matches!(p, Proc::Nil))
            .collect();

        // 3. Sort for canonical form
        let mut sorted = non_nil;
        sorted.sort_by_key(|p| canonical_order(p));

        // 4. Rebuild right-associative
        self.rebuild(sorted)
    }

    fn flatten(&self, proc: &Proc) -> Vec<Proc> {
        match proc {
            Proc::Par(p, q) => {
                let mut result = self.flatten(p);
                result.extend(self.flatten(q));
                result
            }
            _ => vec![self.fuse_subterm(proc)]
        }
    }

    fn fuse_subterm(&self, proc: &Proc) -> Proc {
        match proc {
            Proc::New(x, body) => Proc::New(x.clone(), Box::new(self.fuse(body))),
            Proc::Send(c, d, k) => Proc::Send(c.clone(), d.clone(), Box::new(self.fuse(k))),
            Proc::Receive(p, c, b) => Proc::Receive(p.clone(), c.clone(), Box::new(self.fuse(b))),
            _ => proc.clone()
        }
    }

    fn rebuild(&self, procs: Vec<Proc>) -> Proc {
        if procs.is_empty() {
            Proc::Nil
        } else {
            procs.into_iter().reduce(|acc, p| {
                Proc::Par(Box::new(acc), Box::new(p))
            }).unwrap_or(Proc::Nil)
        }
    }
}
```

### Constraint: Time Optimization

**Metric**: Tree traversal depth reduction

**Target**: $`\mathcal{O}(n)`$ traversal instead of $`\mathcal{O}(n^{2})`$ for nested structures

**Trade-off**: Flattening time vs. subsequent operation efficiency

---

## Strategy 3: Scope Minimization

### Principle

Push bindings as deep as possible to enable more dead code elimination.

### Applicable Laws

1. **Scope Extrusion** (reverse direction):
   - `(new x.P) | Q → new x.(P | Q)` when $`x \notin  \text{FV}(Q)`$
   - Push scope inward

2. **Scope Splitting**:
   - `new x.(P | Q)` where only P uses x
   - Can extrude Q then push x deeper into P

### Bisimulation Justification

**Claim**: Scope minimization preserves bisimilarity.

**Proof**: Scope extrusion and its reverse are bisimilarity-preserving (see RPO proofs). Minimized scope is bisimilar to expanded scope.

### Implementation

```rust
pub struct ScopeMinimizer;

impl ScopeMinimizer {
    /// Push scopes as deep as possible
    pub fn minimize(&self, proc: &Proc) -> Proc {
        match proc {
            // Scope can be pushed into one branch of Par
            Proc::New(x, body) => {
                let minimized_body = self.minimize(body);

                if let Proc::Par(p, q) = &minimized_body {
                    let x_in_p = p.free_vars().contains(x);
                    let x_in_q = q.free_vars().contains(x);

                    match (x_in_p, x_in_q) {
                        // x only in p: push into p
                        (true, false) => {
                            Proc::Par(
                                Box::new(Proc::New(x.clone(), p.clone())),
                                q.clone(),
                            )
                        }
                        // x only in q: push into q
                        (false, true) => {
                            Proc::Par(
                                p.clone(),
                                Box::new(Proc::New(x.clone(), q.clone())),
                            )
                        }
                        // x in both: cannot minimize further
                        (true, true) => {
                            Proc::New(x.clone(), Box::new(minimized_body))
                        }
                        // x in neither: dead scope, eliminate
                        (false, false) => minimized_body
                    }
                } else {
                    Proc::New(x.clone(), Box::new(minimized_body))
                }
            }

            // Recursive cases
            Proc::Par(p, q) => {
                Proc::Par(
                    Box::new(self.minimize(p)),
                    Box::new(self.minimize(q)),
                )
            }

            _ => proc.clone()
        }
    }
}
```

### Constraint: Space Optimization

**Metric**: Scope tree depth reduction

**Target**: Enable subsequent dead code elimination

**Trade-off**: Transformation passes vs. optimization benefit

---

## Strategy 4: Communication Fusion

### Principle

Identify matching send/receive pairs that can be fused for direct execution.

### Applicable Pattern

When a send and receive on the same channel are composed:

```
out(x, v) | in(x, λy.P) → P[v/y]
```

This is the **comm rule** - the fundamental reduction of process calculi.

### Bisimulation Justification

**Claim**: Communication fusion produces a bisimilar (actually, more reduced) process.

**Proof**: The comm rule is a **reduction**, not just a congruence. It's the defining operational semantics. The reduced process is bisimilar to the unreduced one since:
- The unreduced version can take a τ-step to the reduced
- Weak bisimilarity equates processes related by τ-steps

### Implementation

```rust
pub struct CommunicationFusion {
    /// Whether to fuse eagerly (may change non-determinism)
    eager: bool,
}

impl CommunicationFusion {
    /// Find and fuse matching communications
    pub fn fuse(&self, proc: &Proc) -> Proc {
        let flattened = flatten_par(proc);
        let mut result = Vec::new();
        let mut consumed = HashSet::new();

        for (i, p) in flattened.iter().enumerate() {
            if consumed.contains(&i) {
                continue;
            }

            // Look for matching communication
            if let Proc::Send(chan, data, cont) = p {
                for (j, q) in flattened.iter().enumerate() {
                    if i == j || consumed.contains(&j) {
                        continue;
                    }

                    if let Proc::Receive(pattern, recv_chan, body) = q {
                        if chan == recv_chan {
                            // Found match! Fuse the communication
                            let fused = substitute(body, pattern, data);
                            result.push(self.fuse(&fused));
                            result.push(self.fuse(cont));
                            consumed.insert(i);
                            consumed.insert(j);
                            break;
                        }
                    }
                }
            }

            if !consumed.contains(&i) {
                result.push(self.fuse_subterm(p));
            }
        }

        rebuild_par(result)
    }
}
```

### Constraint: Time Optimization

**Metric**: Synchronization overhead reduction

**Target**: Direct evaluation instead of message passing

**Trade-off**: May change non-deterministic behavior (only valid when deterministic)

**Warning**: Communication fusion changes the reduction strategy. It's only valid when:
1. The program is deterministic, OR
2. The specific execution order doesn't matter

---

## Strategy 5: Resource-Constrained Optimization

### Principle

Optimize for specific resource constraints while maintaining bisimilarity.

### Constraint Matrix

| Constraint | Primary Strategy | Secondary Strategy | Guarantee |
|------------|-----------------|-------------------|-----------|
| **Memory** | Dead code elimination | Scope minimization | $`P \approx  \text{simplify}(P)`$ |
| **Latency** | Communication fusion | Inline expansion | $`P \approx  \text{optimize}(P)`$ |
| **Parallelism** | Scope extrusion | Parallel fusion | $`P \approx  \text{parallelize}(P)`$ |
| **Bandwidth** | Message coalescing | Batch sends | $`P \approx  \text{coalesce}(P)`$ |

### Memory Optimization

```rust
pub fn optimize_memory(proc: &Proc) -> Proc {
    let p1 = DeadCodeEliminator::new().eliminate(proc);
    let p2 = ScopeMinimizer::new().minimize(&p1);
    let p3 = DeadCodeEliminator::new().eliminate(&p2);  // Second pass
    p3
}
```

### Latency Optimization

```rust
pub fn optimize_latency(proc: &Proc) -> Proc {
    let p1 = CommunicationFusion::new().fuse(proc);
    let p2 = inline_small_continuations(&p1);
    p2
}
```

### Parallelism Optimization

```rust
pub fn optimize_parallelism(proc: &Proc) -> Proc {
    let p1 = ParallelFusion::new().fuse(proc);
    let p2 = extrude_independent_scopes(&p1);
    p2
}
```

### Bandwidth Optimization

```rust
pub fn optimize_bandwidth(proc: &Proc) -> Proc {
    let p1 = coalesce_messages(proc);
    let p2 = batch_sends(&p1);
    p2
}
```

---

## Multi-Objective Optimization

### Pareto Frontier

When optimizing for multiple constraints, find the Pareto-optimal solutions:

```rust
pub struct MultiObjectiveOptimizer {
    memory_weight: f64,
    latency_weight: f64,
    parallelism_weight: f64,
}

impl MultiObjectiveOptimizer {
    pub fn optimize(&self, proc: &Proc) -> Vec<OptimizedProgram> {
        let candidates = vec![
            self.optimize_for_memory(proc),
            self.optimize_for_latency(proc),
            self.optimize_for_parallelism(proc),
            self.optimize_balanced(proc),
        ];

        // Filter to Pareto-optimal set
        pareto_filter(candidates)
    }

    fn score(&self, proc: &Proc) -> f64 {
        self.memory_weight * memory_score(proc)
            + self.latency_weight * latency_score(proc)
            + self.parallelism_weight * parallelism_score(proc)
    }
}
```

### User-Directed Optimization

Allow users to specify optimization priorities:

```metta
(optimize-for memory
    (program my-proc)
    (target-reduction 30%))

(optimize-for latency
    (program my-proc)
    (allow-reordering true))

(optimize-for parallelism
    (program my-proc)
    (max-parallel-branches 8))
```

---

## Correctness Verification

### Post-Optimization Verification

Every optimization must maintain bisimilarity:

```rust
pub fn verify_optimization(original: &Proc, optimized: &Proc) -> Result<(), OptimizationError> {
    // Quick check: structural equivalence
    if structural_equivalent(original, optimized) {
        return Ok(());
    }

    // Medium check: congruence derivability
    if derivable_by_congruence(original, optimized) {
        return Ok(());
    }

    // Slow check: full bisimulation
    if check_bisimilar(original, optimized) {
        return Ok(());
    }

    Err(OptimizationError::NotBisimilar {
        original: original.clone(),
        optimized: optimized.clone(),
    })
}
```

### Optimization Evidence

Track how optimizations were derived:

```rust
#[derive(Clone, Debug)]
pub struct OptimizationEvidence {
    /// Original program
    pub original: Proc,

    /// Optimized program
    pub optimized: Proc,

    /// Sequence of transformations applied
    pub transformations: Vec<TransformationStep>,

    /// Resource metrics
    pub metrics: OptimizationMetrics,
}

#[derive(Clone, Debug)]
pub struct TransformationStep {
    pub rule: String,
    pub location: AstPath,
    pub before: Proc,
    pub after: Proc,
}

#[derive(Clone, Debug)]
pub struct OptimizationMetrics {
    pub ast_size_before: usize,
    pub ast_size_after: usize,
    pub estimated_latency_before: Duration,
    pub estimated_latency_after: Duration,
    pub parallelism_before: usize,
    pub parallelism_after: usize,
}
```

---

## Related Documentation

- [RPO Congruence Proofs](09-rpo-congruence-proofs.md) - Bisimilarity proofs for congruence laws
- [Transparency Guarantees](10-transparency-guarantees.md) - Phase transparency
- [Up-To Verification](12-up-to-verification.md) - Efficient bisimulation checking
- [Performance Targets](08-performance-targets.md) - Benchmark goals

---

## References

1. Milner, "Communicating and Mobile Systems: the Pi-Calculus" - Comm rule
2. Sangiorgi & Walker, "The Pi-Calculus" - Optimization techniques
3. Wells & Stay, "Behavior in Higher-Order Languages" - Soundness framework

---

## Changelog

- **2025-12-17**: Initial optimization strategies documentation
