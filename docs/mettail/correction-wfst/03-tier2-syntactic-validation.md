# Tier 2: Syntactic Validation

This document describes Tier 2 of the correction WFST architecture: syntactic
validation via context-free grammars and MORK/PathMap.

**Sources**:
- MORK: `/home/dylon/Workspace/f1r3fly.io/MORK/`
- PathMap: `/home/dylon/Workspace/f1r3fly.io/PathMap/`

**Related Integration Docs**:
- [Grammar Correction](../../integration/mork/grammar_correction.md) - Phase D implementation
- [Lattice Integration](../../integration/mork/lattice_integration.md) - Phase B details
- [WFST Composition](../../integration/mork/wfst_composition.md) - Phase C details
- [MORK Integration Overview](../../integration/mork/README.md) - Complete phase documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Context-Free Grammar Integration](#context-free-grammar-integration)
3. [Lattice Parsing](#lattice-parsing)
4. [Grammar Rules in MORK Space](#grammar-rules-in-mork-space)
5. [Weighted Productions](#weighted-productions)
6. [Error Recovery Strategies](#error-recovery-strategies)
7. [Programming Language Awareness](#programming-language-awareness)
8. [Performance Optimizations](#performance-optimizations)

---

## Overview

Tier 2 filters lexical candidates by syntactic validity, using context-free
grammars stored in MORK Space with PathMap backend.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Tier 2: Syntactic Validation                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Candidate Lattice from Tier 1                           │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                   │
│  │ the │──│ tea │──│ ten │──│ tee │──│tech │                   │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                   │
│     │        │        │        │        │                        │
│     ▼        ▼        ▼        ▼        ▼                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   MORK Grammar Space                        ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  S → NP VP                                          │   ││
│  │  │  NP → Det N | N                                     │   ││
│  │  │  VP → V NP | V                                      │   ││
│  │  │  Det → "the" | "a" | ...                           │   ││
│  │  │  N → "cat" | "dog" | "tea" | ...                   │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Lattice Parser (GLR)                       ││
│  │  • Parse lattice against grammar                           ││
│  │  • Weight paths by production probabilities                ││
│  │  • Filter invalid candidates                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  Output: Syntactically Valid Candidates                         │
│  ┌─────┐  ┌─────┐                                              │
│  │ the │  │ tea │  (valid as Det or N in context)              │
│  └─────┘  └─────┘                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Context-Free Grammar Integration

Grammars define the syntactic structure of the target language:

### Grammar Representation

```rust
/// Context-free grammar
pub struct Grammar {
    /// Start symbol
    start: Symbol,
    /// Production rules
    productions: Vec<Production>,
    /// Terminal symbols
    terminals: HashSet<Symbol>,
    /// Non-terminal symbols
    nonterminals: HashSet<Symbol>,
}

/// Production rule: A → α
pub struct Production {
    /// Left-hand side (non-terminal)
    lhs: Symbol,
    /// Right-hand side (sequence of symbols)
    rhs: Vec<Symbol>,
    /// Production weight (for probabilistic grammars)
    weight: f64,
}

/// Symbol in grammar
#[derive(Clone, Eq, PartialEq, Hash)]
pub enum Symbol {
    Terminal(String),
    NonTerminal(String),
}
```

### Grammar Loading

```rust
impl Grammar {
    /// Load grammar from BNF-style file
    pub fn from_bnf(path: &Path) -> Result<Self, GrammarError> {
        let content = std::fs::read_to_string(path)?;
        let parser = BnfParser::new();
        parser.parse(&content)
    }

    /// Load grammar from MORK Space
    pub fn from_mork(space: &Space) -> Result<Self, GrammarError> {
        let productions = space.query_pattern(b"production/*")?
            .map(|(path, bytes)| Production::from_bytes(&bytes))
            .collect::<Result<Vec<_>, _>>()?;

        Self::from_productions(productions)
    }
}
```

---

## Lattice Parsing

Lattice parsing processes all paths through the candidate lattice efficiently:

### GLR Lattice Parser

```rust
/// GLR parser for lattice input
pub struct LatticeParse {
    /// Grammar
    grammar: Grammar,
    /// LR parsing table
    table: LRTable,
}

impl LatticeParser {
    /// Parse lattice against grammar
    pub fn parse<W: Semiring>(
        &self,
        lattice: &CandidateLattice<W>,
    ) -> ParseResult<W> {
        // Initialize with start states for all lattice entry points
        let mut states = self.initialize_states(lattice);

        // Process lattice in topological order
        for node in lattice.topological_order() {
            for edge in lattice.outgoing_edges(node) {
                // Try all parsing actions for this candidate
                let actions = self.table.actions(&states[node], &edge.candidate);

                for action in actions {
                    match action {
                        Action::Shift(next_state) => {
                            states[edge.to].push(next_state);
                        }
                        Action::Reduce(production) => {
                            self.reduce(&mut states, edge.to, production);
                        }
                        Action::Accept => {
                            // Valid parse found
                        }
                    }
                }
            }
        }

        self.extract_valid_paths(&states, lattice)
    }
}
```

### Speedup Analysis

Lattice parsing achieves 3-10x speedup over exhaustive enumeration:

| Lattice Size | Exhaustive | Lattice Parsing | Speedup |
|--------------|------------|-----------------|---------|
| 10 candidates | 10 parses | 1 parse | 10x |
| 100 candidates | 100 parses | 10-20 states | 5-10x |
| 1000 candidates | 1000 parses | 50-100 states | 10-20x |

The key insight: shared prefixes in the lattice share parse states.

---

## Grammar Rules in MORK Space

Grammar rules are stored in MORK Space for efficient pattern matching. For
complete implementation details including CFG-to-pattern compilation and MORK
function usage, see [Phase D: Grammar Correction](../../integration/mork/grammar_correction.md).

**Key MORK Functions**:
| Function | Location | Purpose |
|----------|----------|---------|
| `match2()` | expr/src/lib.rs:921 | Recursive structural matching |
| `unify()` | expr/src/lib.rs:1849 | Variable binding + constraints |
| `query_multi_i()` | kernel/src/space.rs:992 | $`\mathcal{O}(K\times N)`$ lattice queries |
| `transform_multi_multi_()` | kernel/src/space.rs:1221 | Pattern→template application |

### Storage Schema

```
MORK Space Grammar Layout:
├── grammar/
│   ├── productions/
│   │   ├── S/              → [NP, VP]
│   │   ├── NP/             → [[Det, N], [N]]
│   │   └── VP/             → [[V, NP], [V]]
│   ├── terminals/
│   │   ├── the/            → Det
│   │   ├── cat/            → N
│   │   └── sat/            → V
│   └── metadata/
│       ├── start/          → S
│       └── weights/        → production probabilities
```

### MORK Encoding

```rust
/// Encode production as MORK bytes
fn encode_production(prod: &Production) -> Vec<u8> {
    let mut bytes = Vec::new();

    // LHS symbol
    bytes.push(TAG_SYMBOL);
    bytes.extend(prod.lhs.as_bytes());

    // RHS symbols
    bytes.push(TAG_SEQUENCE);
    bytes.push(prod.rhs.len() as u8);
    for sym in &prod.rhs {
        bytes.push(TAG_SYMBOL);
        bytes.extend(sym.as_bytes());
    }

    // Weight
    bytes.extend(prod.weight.to_le_bytes());

    bytes
}

/// Query productions for non-terminal
fn query_productions(space: &Space, nonterminal: &str) -> Vec<Production> {
    let pattern = format!("grammar/productions/{}/", nonterminal);
    space.query_pattern(pattern.as_bytes())
        .map(|(_, bytes)| decode_production(&bytes))
        .collect()
}
```

---

## Weighted Productions

Productions carry weights for probabilistic parsing:

### Probabilistic CFG (PCFG)

```rust
/// Probabilistic production: P(A → α)
pub struct ProbabilisticProduction {
    pub production: Production,
    /// P(A → α | A)
    pub probability: f64,
}

impl ProbabilisticProduction {
    /// Validate: sum of probabilities for each LHS = 1
    pub fn validate_normalized(productions: &[Self]) -> bool {
        let mut sums: HashMap<Symbol, f64> = HashMap::new();
        for prod in productions {
            *sums.entry(prod.production.lhs.clone()).or_default() += prod.probability;
        }
        sums.values().all(|&sum| (sum - 1.0).abs() < 1e-6)
    }
}
```

### Weight Propagation

```rust
/// Propagate weights through parse
fn propagate_weights<W: Semiring>(
    parse_tree: &ParseTree,
    lattice: &CandidateLattice<W>,
) -> W {
    match parse_tree {
        ParseTree::Leaf { edge_id } => {
            // Terminal: use lattice edge weight
            lattice.edge(*edge_id).weight.clone()
        }
        ParseTree::Node { production, children } => {
            // Non-terminal: multiply production weight with children
            let prod_weight = W::from_f64(production.weight);
            children.iter()
                .map(|child| propagate_weights(child, lattice))
                .fold(prod_weight, |acc, w| acc.mul(&w))
        }
    }
}
```

---

## Error Recovery Strategies

When the input contains errors, the parser needs recovery strategies:

### Panic Mode Recovery

Skip tokens until a synchronization point:

```rust
/// Panic mode error recovery
fn panic_mode_recovery(
    &mut self,
    current_state: &ParseState,
    input: &[Token],
) -> RecoveryAction {
    // Find synchronization tokens (e.g., ';', '}', newline)
    let sync_tokens = &[Token::Semicolon, Token::RBrace, Token::Newline];

    for (i, token) in input.iter().enumerate() {
        if sync_tokens.contains(token) {
            return RecoveryAction::Skip(i + 1);
        }
    }

    RecoveryAction::Fail
}
```

### Phrase-Level Recovery

Insert or delete tokens to match expected pattern:

```rust
/// Phrase-level error recovery
fn phrase_level_recovery(
    &self,
    state: &ParseState,
    next_token: &Token,
) -> Vec<RecoveryAction> {
    let mut actions = Vec::new();

    // Try deletion
    actions.push(RecoveryAction::Delete);

    // Try insertion of expected tokens
    for expected in self.table.expected(state) {
        actions.push(RecoveryAction::Insert(expected.clone()));
    }

    actions
}
```

### Error Productions

Special productions that match error patterns:

```rust
// Grammar with error productions
S → NP VP
S → error VP     // Recover from bad NP
S → NP error     // Recover from bad VP
NP → Det N
NP → Det error   // Recover from missing N
```

---

## Programming Language Awareness

For code correction, integrate with programming language infrastructure:

### Tree-sitter Integration

```rust
/// Tree-sitter parser for programming languages
pub struct TreeSitterGrammar {
    language: Language,
    parser: Parser,
}

impl TreeSitterGrammar {
    /// Parse source with error recovery
    pub fn parse_with_errors(&self, source: &str) -> ParseResult {
        let tree = self.parser.parse(source, None).unwrap();

        let errors: Vec<_> = tree.root_node()
            .descendant_for_byte_range(0, source.len())
            .filter(|n| n.is_error() || n.is_missing())
            .map(|n| SyntaxError {
                span: n.byte_range(),
                kind: if n.is_error() { ErrorKind::Unexpected } else { ErrorKind::Missing },
            })
            .collect();

        ParseResult { tree, errors }
    }
}
```

### Language-Specific Grammars

```rust
/// Load grammar for specific language
pub fn grammar_for_language(lang: &str) -> Result<Grammar, GrammarError> {
    match lang {
        "rust" => Grammar::from_tree_sitter(tree_sitter_rust::language()),
        "python" => Grammar::from_tree_sitter(tree_sitter_python::language()),
        "rholang" => Grammar::from_tree_sitter(tree_sitter_rholang::language()),
        "metta" => Grammar::from_mork(&load_metta_grammar_space()?),
        _ => Err(GrammarError::UnsupportedLanguage(lang.to_string())),
    }
}
```

---

## Performance Optimizations

### LRU Cache for Hot Rules

```rust
/// Cache for frequently used grammar rules
pub struct GrammarCache {
    /// Production cache by LHS
    production_cache: LruCache<Symbol, Vec<Production>>,
    /// Parse table cache
    table_cache: LruCache<(usize, Symbol), Vec<Action>>,
}

impl GrammarCache {
    pub fn get_productions(&mut self, lhs: &Symbol) -> &[Production] {
        if !self.production_cache.contains(lhs) {
            let prods = self.space.query_productions(lhs);
            self.production_cache.put(lhs.clone(), prods);
        }
        self.production_cache.get(lhs).unwrap()
    }
}
```

### Incremental Parsing

Reuse partial parses for nearby edits:

```rust
/// Incremental parser state
pub struct IncrementalParser {
    /// Previous parse tree
    tree: ParseTree,
    /// Edit positions
    edits: Vec<Edit>,
}

impl IncrementalParser {
    /// Update parse after edit
    pub fn apply_edit(&mut self, edit: Edit) -> ParseResult {
        // Find affected subtree
        let affected = self.find_affected_subtree(&edit);

        // Re-parse only affected region
        let new_subtree = self.parse_region(edit.range())?;

        // Splice into existing tree
        self.splice_subtree(affected, new_subtree);

        Ok(())
    }
}
```

### Parallel Candidate Processing

```rust
/// Process candidates in parallel
pub fn filter_syntactic_parallel<W: Semiring + Send + Sync>(
    lattice: &CandidateLattice<W>,
    grammar: &Grammar,
) -> CandidateLattice<W> {
    use rayon::prelude::*;

    let valid_edges: Vec<_> = lattice.edges()
        .par_iter()
        .filter(|edge| grammar.can_derive(&edge.candidate))
        .cloned()
        .collect();

    CandidateLattice::from_edges(valid_edges)
}
```

---

## Summary

Tier 2 provides:

1. **CFG Integration**: Grammar representation and loading
2. **Lattice Parsing**: 3-10x speedup via shared parse states
3. **MORK Storage**: Efficient grammar rule storage and querying
4. **Weighted Productions**: Probabilistic parsing support
5. **Error Recovery**: Panic mode, phrase-level, error productions
6. **Language Awareness**: Tree-sitter and language-specific grammars
7. **Performance**: LRU caching, incremental parsing, parallelism

The filtered candidate set feeds into Tier 3 for semantic type checking.

---

## References

- MORK: `/home/dylon/Workspace/f1r3fly.io/MORK/`
- PathMap: `/home/dylon/Workspace/f1r3fly.io/PathMap/`
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- See [04-tier3-semantic-type-checking.md](./04-tier3-semantic-type-checking.md) for next tier
- See [bibliography.md](../reference/bibliography.md) for complete references
