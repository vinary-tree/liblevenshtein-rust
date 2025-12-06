# Layer 3: Strategy Selection Layer

The strategy selection layer controls rule application order, termination, and conflict resolution.

**Status**: Design Documentation
**Last Updated**: 2025-12-06

---

## Overview

This layer orchestrates the simplification process by determining which rules to apply, in what order, and when to stop. It ensures termination while maximizing the effectiveness of simplification.

```
Input: Proc + AnalysisFacts + RuleRegistry
         ↓
┌──────────────────────────────────────┐
│      STRATEGY SELECTION LAYER        │
│                                      │
│  ┌────────────────────────────────┐  │
│  │      Phase Orchestration       │  │
│  │  Local → Analysis → Struct →   │  │
│  │              Type              │  │
│  └────────────────────────────────┘  │
│                 ↓                    │
│  ┌────────────────────────────────┐  │
│  │      Priority Ordering         │  │
│  │   Sort rules by weight/cost    │  │
│  └────────────────────────────────┘  │
│                 ↓                    │
│  ┌────────────────────────────────┐  │
│  │     Termination Control        │  │
│  │   Fixpoint + cycle detection   │  │
│  └────────────────────────────────┘  │
│                 ↓                    │
│  ┌────────────────────────────────┐  │
│  │     Conflict Resolution        │  │
│  │   When multiple rules apply    │  │
│  └────────────────────────────────┘  │
│                                      │
└──────────────────────────────────────┘
         ↓
Output: Transformation Schedule → Simplified Proc
```

---

## Module Structure

```
simplification/strategies/
├── mod.rs              # Strategy orchestration
├── priority.rs         # Rule priority ordering
├── termination.rs      # Fixpoint detection, cycle prevention
├── conflict.rs         # Resolve multiple applicable rules
└── metrics.rs          # Track optimization effectiveness
```

---

## Simplification Phases

Rules are organized into phases that execute in a specific order:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimplificationPhase {
    /// Phase 1: Fast local rewrites (algebraic identities)
    /// No analysis required, always applicable
    LocalSimplification = 0,

    /// Phase 2: Analysis-dependent rewrites
    /// Requires reachability, liveness, constant propagation
    AnalysisDriven = 1,

    /// Phase 3: Rholang structural normalization
    /// Applies congruence rules to reach canonical form
    StructuralNormalization = 2,

    /// Phase 4: Type-directed simplifications
    /// Requires full type information
    TypeDirected = 3,
}

impl SimplificationPhase {
    /// Phases execute in order
    pub fn all_phases() -> &'static [SimplificationPhase] {
        &[
            SimplificationPhase::LocalSimplification,
            SimplificationPhase::AnalysisDriven,
            SimplificationPhase::StructuralNormalization,
            SimplificationPhase::TypeDirected,
        ]
    }

    /// Some phases may need to repeat after later phases make changes
    pub fn triggers_reanalysis(&self) -> bool {
        match self {
            SimplificationPhase::AnalysisDriven => true,
            SimplificationPhase::TypeDirected => true,
            _ => false,
        }
    }
}
```

### Phase Descriptions

| Phase | Purpose | Example Rules | Analysis Required |
|-------|---------|---------------|-------------------|
| `LocalSimplification` | Fast algebraic rewrites | x+0→x, x*1→x, --x→x | None |
| `AnalysisDriven` | Context-dependent simplification | Dead code elimination, constant folding | Reachability, constants |
| `StructuralNormalization` | Canonical form for comparison | Rholang congruence (P\|Q → Q\|P) | None (term ordering) |
| `TypeDirected` | Type-aware transformations | Redundant cast removal, inlining | Type information |

---

## Strategy Context

```rust
/// Maintains state across the simplification process
pub struct StrategyContext {
    /// Current phase
    phase: SimplificationPhase,

    /// Iteration count within current phase
    phase_iteration: usize,

    /// Total iterations across all phases
    total_iterations: usize,

    /// Maximum iterations per phase
    max_phase_iterations: usize,

    /// Maximum total iterations (absolute termination bound)
    max_total_iterations: usize,

    /// Visited term hashes for cycle detection
    visited: HashSet<u64>,

    /// Metrics collection
    metrics: SimplificationMetrics,
}

impl StrategyContext {
    pub fn new(config: &SimplificationConfig) -> Self {
        Self {
            phase: SimplificationPhase::LocalSimplification,
            phase_iteration: 0,
            total_iterations: 0,
            max_phase_iterations: config.max_iterations,
            max_total_iterations: config.max_iterations * 4, // 4 phases
            visited: HashSet::new(),
            metrics: SimplificationMetrics::new(),
        }
    }

    /// Check if we should continue simplifying
    pub fn should_continue(&self) -> bool {
        self.total_iterations < self.max_total_iterations
    }

    /// Check if current phase should continue
    pub fn phase_should_continue(&self) -> bool {
        self.phase_iteration < self.max_phase_iterations
    }

    /// Record a term visit (returns false if already visited = cycle)
    pub fn visit(&mut self, term: &Proc) -> bool {
        let hash = term.content_hash();
        self.visited.insert(hash)
    }

    /// Advance to next phase
    pub fn advance_phase(&mut self) -> Option<SimplificationPhase> {
        let phases = SimplificationPhase::all_phases();
        let current_idx = self.phase as usize;

        if current_idx + 1 < phases.len() {
            self.phase = phases[current_idx + 1];
            self.phase_iteration = 0;
            // Keep visited set across phases
            Some(self.phase)
        } else {
            None
        }
    }

    /// Record a successful rule application
    pub fn record_application(&mut self, rule: &dyn SimplificationRule, old: &Proc, new: &Proc) {
        self.phase_iteration += 1;
        self.total_iterations += 1;
        self.metrics.record_rule_application(rule.name(), old.size(), new.size());
    }
}
```

---

## Priority Ordering

Rules within a phase are ordered by priority:

```rust
/// Rule priority determines application order
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RulePriority {
    /// Lower weight = higher priority
    weight: i32,
    /// Tie-breaker: more specific patterns first
    specificity: u32,
}

impl RulePriority {
    pub fn from_rule(rule: &dyn SimplificationRule) -> Self {
        Self {
            weight: rule.termination_weight(),
            specificity: rule.pattern_specificity(),
        }
    }
}

pub trait SimplificationRule {
    // ... other methods ...

    /// Termination weight - more negative = higher priority
    /// Size-reducing rules should have negative weights
    fn termination_weight(&self) -> i32;

    /// Pattern specificity - more specific = higher priority
    /// Counts non-variable nodes in pattern
    fn pattern_specificity(&self) -> u32 {
        self.pattern().count_non_variables()
    }
}

/// Order rules for application
pub fn order_rules<'a>(
    rules: &[&'a dyn SimplificationRule],
    phase: SimplificationPhase,
) -> Vec<&'a dyn SimplificationRule> {
    let mut phase_rules: Vec<_> = rules.iter()
        .filter(|r| r.phase() == phase)
        .copied()
        .collect();

    // Sort by priority (lowest weight first, then highest specificity)
    phase_rules.sort_by_key(|r| {
        let priority = RulePriority::from_rule(*r);
        (priority.weight, -(priority.specificity as i32))
    });

    phase_rules
}
```

### Priority Examples

| Rule | Termination Weight | Specificity | Priority Order |
|------|-------------------|-------------|----------------|
| mul-zero | -2 | 3 | 1 (highest) |
| add-zero | -1 | 3 | 2 |
| double-neg | -1 | 2 | 3 |
| par-commute | 0 | 2 | 4 (lowest) |

---

## Termination Control

### Fixpoint Detection

```rust
pub struct TerminationController {
    /// Maximum iterations
    max_iterations: usize,

    /// Visited terms (hash-based)
    visited: HashSet<u64>,

    /// Size history for monotonicity checking
    size_history: Vec<usize>,
}

impl TerminationController {
    /// Check if simplification has reached a fixpoint
    pub fn is_fixpoint(&self, term: &Proc, changed: bool) -> bool {
        if !changed {
            return true;  // No rule applied = fixpoint
        }

        let hash = term.content_hash();
        if self.visited.contains(&hash) {
            return true;  // Already seen this term = cycle
        }

        false
    }

    /// Check monotonicity (size should generally decrease)
    pub fn check_monotonicity(&self) -> bool {
        if self.size_history.len() < 2 {
            return true;
        }

        // Allow small temporary increases for normalization
        let window = &self.size_history[self.size_history.len().saturating_sub(10)..];
        let first = window.first().unwrap_or(&0);
        let last = window.last().unwrap_or(&0);

        // Overall trend should be non-increasing
        last <= first
    }
}
```

### Cycle Prevention

```rust
/// Detect and prevent infinite loops in simplification
pub struct CycleDetector {
    /// Term hashes seen during this simplification run
    seen: HashSet<u64>,

    /// Recent term sequence for pattern detection
    recent: VecDeque<u64>,

    /// Maximum sequence length to check
    max_sequence: usize,
}

impl CycleDetector {
    pub fn new(max_sequence: usize) -> Self {
        Self {
            seen: HashSet::new(),
            recent: VecDeque::new(),
            max_sequence,
        }
    }

    /// Returns true if this term creates a cycle
    pub fn would_cycle(&mut self, term: &Proc) -> bool {
        let hash = term.content_hash();

        // Simple cycle: exact term seen before
        if self.seen.contains(&hash) {
            return true;
        }

        // Check for short cycles (A→B→A)
        if self.recent.contains(&hash) {
            return true;
        }

        false
    }

    /// Record a new term
    pub fn record(&mut self, term: &Proc) {
        let hash = term.content_hash();
        self.seen.insert(hash);
        self.recent.push_back(hash);

        if self.recent.len() > self.max_sequence {
            self.recent.pop_front();
        }
    }
}
```

---

## Conflict Resolution

When multiple rules apply to the same term:

```rust
pub enum ConflictStrategy {
    /// Apply highest priority rule only
    HighestPriority,

    /// Apply rule that reduces size most
    MaximalReduction,

    /// Apply most specific pattern match
    MostSpecific,

    /// Apply all non-overlapping rules
    AllCompatible,
}

pub struct ConflictResolver {
    strategy: ConflictStrategy,
}

impl ConflictResolver {
    /// Select which rules to apply when multiple match
    pub fn resolve<'a>(
        &self,
        applicable: &[(&'a dyn SimplificationRule, Proc)],
        term: &Proc,
    ) -> Vec<(&'a dyn SimplificationRule, Proc)> {
        match self.strategy {
            ConflictStrategy::HighestPriority => {
                // Already sorted by priority, take first
                applicable.first().cloned().into_iter().collect()
            }

            ConflictStrategy::MaximalReduction => {
                // Choose rule that reduces size most
                applicable.iter()
                    .max_by_key(|(_, new)| term.size() as i64 - new.size() as i64)
                    .cloned()
                    .into_iter()
                    .collect()
            }

            ConflictStrategy::MostSpecific => {
                // Choose rule with most specific pattern
                applicable.iter()
                    .max_by_key(|(rule, _)| rule.pattern_specificity())
                    .cloned()
                    .into_iter()
                    .collect()
            }

            ConflictStrategy::AllCompatible => {
                // Apply all rules whose patterns don't overlap
                self.select_non_overlapping(applicable)
            }
        }
    }

    fn select_non_overlapping<'a>(
        &self,
        applicable: &[(&'a dyn SimplificationRule, Proc)],
    ) -> Vec<(&'a dyn SimplificationRule, Proc)> {
        let mut selected = Vec::new();
        let mut modified_positions = HashSet::new();

        for (rule, new_term) in applicable {
            let positions = rule.affected_positions();
            if positions.iter().all(|p| !modified_positions.contains(p)) {
                selected.push((*rule, new_term.clone()));
                modified_positions.extend(positions);
            }
        }

        selected
    }
}
```

---

## Complete Strategy Orchestration

```rust
pub struct SimplificationStrategy {
    registry: RuleRegistry,
    context: StrategyContext,
    termination: TerminationController,
    conflict: ConflictResolver,
    cycle_detector: CycleDetector,
}

impl SimplificationStrategy {
    /// Main simplification loop
    pub fn simplify(
        &mut self,
        initial: Proc,
        facts: &AnalysisFacts,
    ) -> SimplificationResult {
        let mut current = initial.clone();
        let mut analysis_facts = facts.clone();

        // Iterate through phases
        for phase in SimplificationPhase::all_phases() {
            self.context.phase = *phase;

            // Get rules for this phase, ordered by priority
            let rules = order_rules(
                &self.registry.all_rules().collect::<Vec<_>>(),
                *phase,
            );

            // Apply rules until fixpoint within phase
            current = self.apply_phase(current, &rules, &analysis_facts);

            // Some phases may trigger reanalysis
            if phase.triggers_reanalysis() && self.context.phase_iteration > 0 {
                // Recompute analysis facts if term changed
                analysis_facts = AnalysisLayer::new(&self.theory).analyze(&current);
            }
        }

        SimplificationResult {
            original: initial,
            simplified: current,
            metrics: self.context.metrics.clone(),
        }
    }

    /// Apply rules within a single phase
    fn apply_phase(
        &mut self,
        initial: Proc,
        rules: &[&dyn SimplificationRule],
        facts: &AnalysisFacts,
    ) -> Proc {
        let mut current = initial;
        self.context.phase_iteration = 0;

        loop {
            if !self.context.phase_should_continue() {
                break;
            }

            // Check for cycles
            if self.cycle_detector.would_cycle(&current) {
                break;
            }
            self.cycle_detector.record(&current);

            // Find all applicable rules
            let applicable = self.find_applicable(rules, &current, facts);

            if applicable.is_empty() {
                break;  // Fixpoint reached
            }

            // Resolve conflicts and select rules to apply
            let selected = self.conflict.resolve(&applicable, &current);

            if selected.is_empty() {
                break;
            }

            // Apply selected rules
            let (rule, new_term) = &selected[0];
            self.context.record_application(*rule, &current, new_term);
            current = new_term.clone();

            // Check termination conditions
            if !self.context.should_continue() {
                break;
            }
        }

        current
    }

    /// Find all rules that can apply to the current term
    fn find_applicable<'a>(
        &self,
        rules: &[&'a dyn SimplificationRule],
        term: &Proc,
        facts: &AnalysisFacts,
    ) -> Vec<(&'a dyn SimplificationRule, Proc)> {
        let mut applicable = Vec::new();

        for rule in rules {
            // Check guard predicate
            if !rule.guard(term, facts) {
                continue;
            }

            // Try pattern match via MORK
            if let Some(new_term) = self.try_apply(*rule, term) {
                // Check if transformation is beneficial
                if rule.is_beneficial(term, &new_term, facts) {
                    applicable.push((*rule, new_term));
                }
            }
        }

        applicable
    }

    fn try_apply(&self, rule: &dyn SimplificationRule, term: &Proc) -> Option<Proc> {
        let term_expr = term.to_mork_expr();

        let (_, changed) = self.space.transform_multi_multi_(
            rule.pattern(),
            rule.template(),
            term_expr,
        );

        if changed {
            Some(self.extract_result())
        } else {
            None
        }
    }
}
```

---

## Metrics Tracking

```rust
pub struct SimplificationMetrics {
    /// Rules applied and their counts
    rule_counts: HashMap<String, usize>,

    /// Size reduction per rule
    size_reductions: HashMap<String, i64>,

    /// Total size before/after
    initial_size: usize,
    final_size: usize,

    /// Time spent in each phase
    phase_times: HashMap<SimplificationPhase, Duration>,

    /// Total iterations
    total_iterations: usize,
}

impl SimplificationMetrics {
    pub fn record_rule_application(&mut self, rule_name: &str, old_size: usize, new_size: usize) {
        *self.rule_counts.entry(rule_name.to_string()).or_insert(0) += 1;
        *self.size_reductions.entry(rule_name.to_string()).or_insert(0) +=
            old_size as i64 - new_size as i64;
        self.total_iterations += 1;
    }

    pub fn size_reduction_percent(&self) -> f64 {
        if self.initial_size == 0 {
            return 0.0;
        }
        ((self.initial_size - self.final_size) as f64 / self.initial_size as f64) * 100.0
    }

    pub fn report(&self) -> String {
        format!(
            "Simplification: {} iterations, {:.1}% size reduction\n\
             Rules applied: {:?}",
            self.total_iterations,
            self.size_reduction_percent(),
            self.rule_counts
        )
    }
}
```

---

## Configuration

```rust
pub struct StrategyConfig {
    /// Maximum iterations per phase
    pub max_phase_iterations: usize,

    /// Maximum total iterations
    pub max_total_iterations: usize,

    /// Conflict resolution strategy
    pub conflict_strategy: ConflictStrategy,

    /// Enable cycle detection
    pub enable_cycle_detection: bool,

    /// Phases to run (for selective simplification)
    pub enabled_phases: Vec<SimplificationPhase>,

    /// Enable metrics collection
    pub collect_metrics: bool,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            max_phase_iterations: 100,
            max_total_iterations: 1000,
            conflict_strategy: ConflictStrategy::HighestPriority,
            enable_cycle_detection: true,
            enabled_phases: SimplificationPhase::all_phases().to_vec(),
            collect_metrics: true,
        }
    }
}
```

---

## Next Steps

- [Verification](05-verification.md) - Semantic preservation validation
- [Termination Proof](07-termination-proof.md) - Formal termination argument

---

## Changelog

- **2025-12-06**: Initial strategy selection layer documentation
