# NFA Phonetic Regular Expressions

**Status**: Design Document
**Last Updated**: 2025-11-20
**Purpose**: NFA-based phonetic pattern matching for text normalization

---

## Table of Contents

1. [Introduction](#introduction)
2. [Regex Syntax for Phonetic Patterns](#regex-syntax-for-phonetic-patterns)
3. [NFA Construction Algorithms](#nfa-construction-algorithms)
4. [Composition with Levenshtein Automata](#composition-with-levenshtein-automata)
5. [Integration with Verified Phonetic Rules](#integration-with-verified-phonetic-rules)
6. [Performance Considerations](#performance-considerations)
7. [Implementation Examples](#implementation-examples)
8. [References](#references)

---

## Introduction

### Motivation

**Problem**: Phonetic misspellings require pattern-based matching beyond simple edit distance.

**Examples**:
```
"fone" → "phone"    (ph ↔ f substitution)
"nite" → "night"    (gh deletion)
"tuff" → "tough"    (ough ↔ uff substitution)
"sity" → "city"     (c ↔ s before i/y)
```

**Current Approach** (liblevenshtein-rust):
- Fixed table of 13 orthography rules (Zompist phonetic spelling)
- Sequential application: `apply_rules_seq(rules, input, fuel)`
- Formally verified in Coq (5 theorems proven)

**Limitation**: Rules are hardcoded, not composable as automata.

**Proposed Enhancement**: Compile phonetic patterns to NFAs
- **Expressiveness**: Regex-like syntax for patterns
- **Composability**: NFAs intersect with Levenshtein automata
- **Performance**: Pre-compiled NFAs (no runtime interpretation)
- **Verification**: Phonetic rules remain Coq-proven, NFA is compiled representation

---

### Benefits of NFA Representation

**1. Composability**:
```rust
let phonetic_nfa = PhoneticRegex::compile("(ph|f)(o|oa)(n|ne)")?;
let lev_fst = LevenshteinAutomaton::new("phone", 2)?;
let composed = phonetic_nfa.intersect(lev_fst);  // NFA ∩ FST

// Accepts: "phone", "fone", "foan", "phones" (within edit distance)
```

**2. Declarative Syntax**:
```rust
// Old: Imperative rules
fn apply_ph_to_f(input: &[Phone]) -> Vec<Phone> { ... }

// New: Declarative regex
let rule = PhoneticRegex::compile("ph → f")?;
```

**3. Pre-compilation**:
```rust
// Compile once
let rules = PhoneticRuleSet::compile_all(ZOMPIST_RULES)?;

// Use many times (no runtime overhead)
for query in queries {
    let matches = rules.apply(&query);
}
```

**4. Formal Verification Bridge**:
```rust
// Coq-verified rules export to NFA
let verified_rules = load_coq_verified_rules("orthography.v")?;
let nfa = PhoneticRegex::from_verified_rules(verified_rules)?;

// NFA preserves verified properties (soundness, completeness)
```

---

## Regex Syntax for Phonetic Patterns

### Basic Patterns

**1. Literal Characters**:
```regex
phone    # Matches exactly "phone"
cat      # Matches exactly "cat"
```

**2. Alternation** (OR):
```regex
ph|f           # Matches "ph" or "f"
c|k            # Matches "c" or "k"
ough|uff|off   # Matches "ough", "uff", or "off"
```

**3. Optionality** (?):
```regex
gh?      # Matches "gh" or ε (empty string)
e?       # Matches "e" or nothing
```

**4. Repetition**:
```regex
a*       # Zero or more "a" (a, aa, aaa, ...)
a+       # One or more "a" (a, aa, aaa, ...)
a{2}     # Exactly 2 "a"s (aa)
a{2,4}   # 2 to 4 "a"s (aa, aaa, aaaa)
```

**5. Groups**:
```regex
(ph|f)one     # "phone" or "fone"
(c|k)at       # "cat" or "kat"
```

**6. Character Classes**:
```regex
[aeiou]       # Any vowel
[bcdfg]       # Any consonant from set
[a-z]         # Any lowercase letter
[^aeiou]      # Any non-vowel (negation)
```

### Phonetic Extensions

**7. Contextual Rules** (/ for context):
```regex
c → s / _[ei]      # c becomes s before e or i
                    # "city" → "sity", "cent" → "sent"

c → k / _[aou]     # c becomes k before a, o, u
                    # "cat" → "kat", "cot" → "kot"

gh → ∅ / _#         # gh deletes at end of word
                    # "high" → "hi", "night" → "nite"
```

**8. Rewrite Rules** (→):
```regex
ph → f             # "phone" → "fone"
ough → uff         # "tough" → "tuff"
kn → n / #_        # "knight" → "night" (word-initial)
```

**9. Weighted Transitions** (cost):
```regex
ph → f [0.1]       # Low cost (very similar sounds)
c → s [0.2]        # Moderate cost
arbitrary [1.0]    # Default cost
```

**10. Phone Types** (from Coq verification):
```regex
V = [aeiou]        # Vowel
C = [bcdfghjklmnpqrstvwxyz]  # Consonant
D = ph|th|ch|sh    # Digraph
S = gh             # Silent
```

### Grammar (BNF)

```bnf
<pattern>   ::= <expr>
             | <expr> "→" <expr>
             | <expr> "→" <expr> "/" <context>
             | <expr> "→" <expr> "[" <cost> "]"

<expr>      ::= <term>
             | <term> "|" <expr>

<term>      ::= <factor>
             | <factor> <term>

<factor>    ::= <atom>
             | <atom> "*"
             | <atom> "+"
             | <atom> "?"
             | <atom> "{" <count> "}"
             | <atom> "{" <min> "," <max> "}"

<atom>      ::= <char>
             | "[" <char-class> "]"
             | "[^" <char-class> "]"
             | "(" <expr> ")"
             | "∅"
             | "V"
             | "C"
             | "D"
             | "S"

<context>   ::= <position> "_" <position>
<position>  ::= "#" | <char-class> | ε
<cost>      ::= <float>
```

### Examples from Zompist Rules

**Rule 1: ph → f**:
```regex
ph → f [0.1]
```

**Rule 2: c → s/k based on context**:
```regex
c → s / _[eiy] [0.2]   # Before e, i, y
c → k / _[aou] [0.2]   # Before a, o, u
```

**Rule 3: gh → ∅ (silent)**:
```regex
gh → ∅ / _# [0.3]      # Word-final (high, night)
gh → ∅ / _C [0.3]      # Before consonant (daughter)
```

**Rule 4: ough patterns**:
```regex
ough → uff [0.2]       # tough
ough → aw [0.2]        # thought
ough → oh [0.2]        # though
ough → ow [0.2]        # bough
ough → oo [0.2]        # through
```

**Rule 5: kn → n (word-initial)**:
```regex
kn → n / #_ [0.3]      # knight, know
```

**Rule 6: wr → r (word-initial)**:
```regex
wr → r / #_ [0.3]      # write, wrong
```

---

## NFA Construction Algorithms

### Thompson's Construction

**Overview**: Converts regex to NFA with ε-transitions.

**Algorithm**: Recursive construction based on regex operators.

**Base Cases**:

**1. Empty String (ε)**:
```
  (start) --ε--> ((accept))
```

**2. Single Character (a)**:
```
  (start) --a--> ((accept))
```

**Inductive Cases**:

**3. Concatenation (AB)**:
```
  (start) --A--> (mid) --B--> ((accept))
```
Merge accept state of A with start state of B.

**4. Alternation (A|B)**:
```
         ┌--ε--> [A] --ε--┐
  (start)                 ((accept))
         └--ε--> [B] --ε--┘
```
Add ε-transitions from start to both A and B, and from both to accept.

**5. Kleene Star (A*)**:
```
         ┌--------ε--------┐
         ↓                 |
  (start) --ε--> [A] --ε--+-> ((accept))
         └-----------ε-----------┘
```
Add ε-loop from accept(A) to start(A), and bypass ε-transition.

**Example: (a|b)*c**

**Step 1**: Base NFAs for a, b, c
```
a: (0) --a--> (1)
b: (2) --b--> (3)
c: (4) --c--> (5)
```

**Step 2**: Alternation (a|b)
```
     ┌--ε--> (0) --a--> (1) --ε--┐
(6)                               (7)
     └--ε--> (2) --b--> (3) --ε--┘
```

**Step 3**: Kleene Star (a|b)*
```
     ┌------------ε-------------┐
     ↓                          |
(8) --ε--> [a|b NFA from step 2] --ε--> (9)
     └-----------ε------------------------┘
```

**Step 4**: Concatenation (a|b)*c
```
(8) --[(a|b)* NFA]--> (9) --ε--> (4) --c--> (5)
```

**Final NFA**: 10 states, accepts strings like "c", "ac", "bc", "aac", "abc", "bbc", etc.

### Subset Construction (NFA → DFA)

**Purpose**: Convert NFA to DFA (no ε-transitions, deterministic).

**Algorithm**: Powerset construction.

**Steps**:

1. **ε-closure**: For each state, compute ε-closure(s) = set of states reachable via ε-transitions.

2. **Subset states**: Each DFA state corresponds to a set of NFA states.

3. **Transition function**: δ_DFA(S, a) = ε-closure(∪ δ_NFA(s, a) for s ∈ S)

4. **Start state**: ε-closure(start_NFA)

5. **Accept states**: Any subset containing an NFA accept state.

**Example**: (a|b)*

**NFA** (from Thompson's construction):
```
States: {0, 1, 2, 3, 4, 5}
Start: 0
Accept: 5
Transitions:
  0 --ε--> 1, 5
  1 --ε--> 2, 4
  2 --a--> 3
  3 --ε--> 1, 5
  4 --b--> 5
  5 --ε--> 1, 5
```

**DFA** (after subset construction):
```
States: {{0,1,2,4,5}, {1,2,3,4,5}, {1,2,4,5}}
Start: {0,1,2,4,5}
Accept: All states (contain 5)
Transitions:
  {0,1,2,4,5} --a--> {1,2,3,4,5}
  {0,1,2,4,5} --b--> {1,2,4,5}
  {1,2,3,4,5} --a--> {1,2,3,4,5}
  {1,2,3,4,5} --b--> {1,2,4,5}
  {1,2,4,5} --a--> {1,2,3,4,5}
  {1,2,4,5} --b--> {1,2,4,5}
```

**Simplification** (state renaming):
```
States: {A, B, C}
Start: A = {0,1,2,4,5}
Accept: {A, B, C}
Transitions:
  A --a--> B
  A --b--> C
  B --a--> B
  B --b--> C
  C --a--> B
  C --b--> C
```

**Minimization** (Hopcroft's algorithm):
```
States: {S}  (all states equivalent, all accepting)
Start: S
Accept: {S}
Transitions:
  S --a--> S
  S --b--> S
```

### Contextual Rules Implementation

**Challenge**: Context like `c → s / _[ei]` requires lookahead/lookbehind.

**Solution**: Augmented NFA with context predicates.

**State Representation**:
```rust
struct ContextualState {
    id: StateId,
    left_context: Option<CharClass>,   // Lookbehind
    right_context: Option<CharClass>,  // Lookahead
}
```

**Example: c → s / _[ei]**

**Interpretation**: "c" becomes "s" when followed by "e" or "i".

**NFA Construction**:
```
States: {0, 1, 2, 3}
Start: 0
Accept: 3

Transitions:
  0 --c--> 1  (consume 'c')
  1 --ε[lookahead=[ei]]--> 2  (check next char is e or i)
  2 --s--> 3  (emit 's')

Alternative (if context doesn't match):
  0 --c--> 3  (emit 'c' unchanged)
```

**Runtime Execution**:
```rust
fn transition(&self, state: StateId, input: char, lookahead: Option<char>) -> Option<StateId> {
    let trans = self.transitions.get(&(state, input))?;

    match trans.context {
        Some(Context::RightContext(chars)) => {
            if let Some(next) = lookahead {
                if chars.contains(&next) {
                    Some(trans.target)
                } else {
                    None  // Context doesn't match
                }
            } else {
                None  // No lookahead available
            }
        }
        None => Some(trans.target),  // No context constraint
    }
}
```

---

## Composition with Levenshtein Automata

### Product Automaton Construction

**Goal**: Intersect NFA (phonetic patterns) with Levenshtein FST (edit distance).

**Intuition**: Accept strings that are:
1. Phonetically plausible (match NFA pattern)
2. Within edit distance n of dictionary word (match Levenshtein automaton)

**State Space**: Product of NFA and Levenshtein states.

**Product State**:
```rust
struct ProductState {
    nfa_state: StateId,
    lev_state: StateId,
    edit_count: usize,
}
```

**Transition Function**:
```
δ_product((q_nfa, q_lev, e), a) = {
    (q'_nfa, q'_lev, e') :
        q'_nfa ∈ δ_nfa(q_nfa, a) ∧
        (q'_lev, e') ∈ δ_lev(q_lev, a, e)
}
```

**Accept Condition**:
- `q_nfa` is accepting in NFA
- `q_lev` is accepting in Levenshtein
- `edit_count ≤ max_distance`

### Example: Phonetic + Edit Distance

**Input**: "fone"
**Target**: "phone"
**Max edit distance**: 2

**NFA**: Phonetic pattern `(ph|f)(o|oa)(n|ne)`
```
States: {0, 1, 2, 3, 4}
Transitions:
  0 --ph--> 1
  0 --f--> 1
  1 --o--> 2
  1 --oa--> 2
  2 --n--> 3
  2 --ne--> 3
  3 --> ACCEPT
```

**Levenshtein**: edit_distance("fone", dictionary) ≤ 2

**Product Automaton** (simplified):
```
Start: (0_nfa, 0_lev, 0_edits)

Path for "fone" → "phone":
  (0, 0, 0) --f--> (1, 1, 0)    # Match 'f' (phonetic variant of 'ph')
  (1, 1, 0) --o--> (2, 2, 0)    # Match 'o'
  (2, 2, 0) --n--> (3, 3, 0)    # Match 'n'
  (3, 3, 0) --e--> (ACCEPT, ACCEPT, 1)  # Insert 'e' (edit cost +1)

Total: NFA accepts (phonetic), Levenshtein accepts (edit=1), VALID
```

**Alternative Path** (direct match):
```
Start: (0_nfa, 0_lev, 0_edits)

Path for "phone" → "phone":
  (0, 0, 0) --ph--> (1, 2, 0)   # Match 'ph' (2 chars in NFA, 2 in Lev)
  (1, 2, 0) --o--> (2, 3, 0)    # Match 'o'
  (2, 3, 0) --n--> (3, 4, 0)    # Match 'n'
  (3, 4, 0) --e--> (ACCEPT, ACCEPT, 0)  # Match 'e'

Total: Exact match (edit=0), VALID
```

### Weighted Product

**Combine Costs**: Phonetic cost + Edit distance cost

**Transition Weight**:
```rust
struct WeightedTransition {
    target: ProductState,
    phonetic_cost: f64,  // From NFA
    edit_cost: f64,      // From Levenshtein
    total_cost: f64,     // phonetic_cost + edit_cost
}
```

**Example**:
```
NFA: "ph" → "f" [cost = 0.1]
Levenshtein: substitution [cost = 1.0]

Combined: total_cost = 0.1 + 1.0 = 1.1
```

**Shortest Path**: Use Dijkstra's algorithm to find minimum-cost path.

---

## Integration with Verified Phonetic Rules

### Coq-Verified Rules

**Current System** (liblevenshtein-rust):

**Coq Definition**:
```coq
Inductive PhRule : Type :=
| RulePh_f : PhRule
| RuleC_s : PhRule
| RuleGh_silent : PhRule
(* ... 13 rules total *)

Definition apply_rule (r : PhRule) (phones : list Phone) : list Phone :=
  match r with
  | RulePh_f => replace_ph_with_f phones
  | RuleC_s => replace_c_with_s_before_ei phones
  | RuleGh_silent => delete_gh_word_final phones
  | ...
  end.
```

**Verified Properties**:
1. **Well-formedness**: All rules preserve phone sequence validity
2. **Bounded expansion**: Output length ≤ input length + k
3. **Termination**: Rule application always terminates
4. **Idempotence**: Applying rule twice = applying once (for some rules)
5. **Non-confluence**: Order matters (some rules don't commute)

### NFA Compilation from Verified Rules

**Strategy**: Export Coq rules to NFA representation.

**Coq → JSON → Rust**:

**Step 1: Coq Extraction**:
```coq
(* Extract rules to JSON *)
Definition rule_to_json (r : PhRule) : string :=
  match r with
  | RulePh_f => "{ ""pattern"": ""ph → f"", ""cost"": 0.1 }"
  | RuleC_s => "{ ""pattern"": ""c → s / _[ei]"", ""cost"": 0.2 }"
  | ...
  end.

Definition all_rules_json : string :=
  "[" ++ String.concat ", " (map rule_to_json all_rules) ++ "]".
```

**Step 2: Rust Import**:
```rust
#[derive(Deserialize)]
struct VerifiedRule {
    pattern: String,
    cost: f64,
}

pub fn load_verified_rules(path: &Path) -> Result<Vec<VerifiedRule>, Error> {
    let json = fs::read_to_string(path)?;
    let rules: Vec<VerifiedRule> = serde_json::from_str(&json)?;
    Ok(rules)
}
```

**Step 3: NFA Compilation**:
```rust
pub fn compile_verified_rules(rules: &[VerifiedRule]) -> Result<PhoneticNFA, Error> {
    let mut nfa = NFA::new();

    for rule in rules {
        let pattern_nfa = PhoneticRegex::compile(&rule.pattern)?;

        // Add to combined NFA with alternation
        nfa = nfa.union(pattern_nfa.with_cost(rule.cost));
    }

    Ok(PhoneticNFA { nfa })
}
```

**Preservation of Verified Properties**:

**Claim**: NFA preserves Coq-verified properties.

**Argument**:
1. **Well-formedness**: NFA construction preserves language structure
2. **Bounded expansion**: NFA states encode length constraints
3. **Termination**: NFA recognition is always O(n), guaranteed termination
4. **Idempotence**: Can encode as NFA property (check if L(NFA) = L(NFA ∘ NFA))
5. **Non-confluence**: NFA alternation explicitly models rule ordering

**Testing Equivalence**:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_nfa_matches_coq() {
        let coq_result = apply_rules_seq(&orthography_rules(), input, fuel);
        let nfa_result = phonetic_nfa.apply(input);

        assert_eq!(coq_result, nfa_result, "NFA must match Coq verified implementation");
    }
}
```

---

## Performance Considerations

### NFA vs DFA Trade-offs

| Aspect | NFA | DFA |
|--------|-----|-----|
| States | O(\|pattern\|) | O(2^{\|pattern\|}) worst case |
| Construction | O(\|pattern\|) | O(2^{\|pattern\|}) worst case |
| Recognition | O(\|input\| · \|states\|) | O(\|input\|) |
| Memory | Small | Large (but constant per input) |
| Parallelism | Multiple active states | Single active state |

**Recommendation**: Use NFA for pattern compilation, lazy DFA for recognition.

### Lazy DFA Construction

**Strategy**: Build DFA states on-demand during recognition.

**Algorithm**:
```rust
struct LazyDFA {
    nfa: NFA,
    dfa_cache: HashMap<Set<StateId>, DFAState>,
}

impl LazyDFA {
    fn transition(&mut self, state_set: &Set<StateId>, input: char) -> Set<StateId> {
        // Check cache
        if let Some(cached) = self.dfa_cache.get(&(state_set.clone(), input)) {
            return cached.clone();
        }

        // Compute ε-closure of all transitions
        let mut next_states = Set::new();
        for state in state_set {
            if let Some(targets) = self.nfa.transitions.get(&(*state, input)) {
                for target in targets {
                    next_states.extend(self.epsilon_closure(*target));
                }
            }
        }

        // Cache result
        self.dfa_cache.insert((state_set.clone(), input), next_states.clone());
        next_states
    }
}
```

**Advantages**:
- Only creates DFA states actually reached
- Amortizes construction cost over multiple queries
- Avoids exponential blowup for unused parts of state space

### Incremental Matching

**Streaming**: Process input incrementally (useful for real-time chat).

**Algorithm**:
```rust
pub struct IncrementalMatcher {
    nfa: PhoneticNFA,
    current_states: Set<StateId>,
}

impl IncrementalMatcher {
    pub fn new(nfa: PhoneticNFA) -> Self {
        let start_states = nfa.epsilon_closure(nfa.start);
        Self { nfa, current_states: start_states }
    }

    pub fn feed(&mut self, input: char) {
        let mut next_states = Set::new();

        for state in &self.current_states {
            if let Some(targets) = self.nfa.transitions.get(&(*state, input)) {
                for target in targets {
                    next_states.extend(self.nfa.epsilon_closure(*target));
                }
            }
        }

        self.current_states = next_states;
    }

    pub fn is_accepting(&self) -> bool {
        self.current_states.iter().any(|s| self.nfa.is_final(*s))
    }

    pub fn reset(&mut self) {
        self.current_states = self.nfa.epsilon_closure(self.nfa.start);
    }
}
```

**Usage**:
```rust
let mut matcher = IncrementalMatcher::new(phonetic_nfa);

for ch in input.chars() {
    matcher.feed(ch);

    if matcher.is_accepting() {
        // Found match at current position
        candidates.push(current_prefix.clone());
    }
}
```

### Memoization

**Cache NFA Intersection Results**:
```rust
pub struct MemoizedIntersection {
    cache: HashMap<(String, usize), Vec<String>>,
}

impl MemoizedIntersection {
    pub fn intersect(&mut self, phonetic: &NFA, lev: &Levenshtein, query: &str, max_dist: usize) -> Vec<String> {
        let key = (query.to_string(), max_dist);

        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        let result = product_automaton(phonetic, lev, query, max_dist);
        self.cache.insert(key, result.clone());
        result
    }
}
```

**When to Memoize**:
- Repeated queries (e.g., autocomplete, spell checking UI)
- Limited query space (bounded by user input patterns)
- Memory available (cache can grow large)

### Complexity Analysis

**NFA Construction** (Thompson's):
- **Time**: O(|pattern|) (linear in pattern length)
- **Space**: O(|pattern|) states

**NFA Recognition**:
- **Time**: O(|input| · |states|) = O(|input| · |pattern|)
- **Space**: O(|states|) for current state set

**NFA → DFA** (Subset construction):
- **Time**: O(2^|states|) worst case, O(|states|²) average
- **Space**: O(2^|states|) worst case

**NFA ∩ Levenshtein** (Product automaton):
- **Time**: O(|input| · |NFA states| · |Lev states| · max_distance)
- **Space**: O(|NFA states| · |Lev states| · max_distance)

**Practical Performance** (benchmarks needed):
- **NFA compilation**: <1ms per rule (one-time cost)
- **Recognition**: <10ms for typical queries (<50 chars)
- **Intersection**: <50ms with memoization

---

## NFA Optimization

The NFA module includes automatic optimization that runs after Thompson construction.
This improves both size (fewer states/transitions) and matching performance (no runtime
epsilon closure computation).

### Optimization Passes

The optimizer applies four passes in order:

1. **Epsilon Elimination** - O(|Q|² × |δ|)
   - Computes epsilon closure for all states
   - Adds direct transitions bypassing epsilon edges
   - Marks states as final if their epsilon closure contains a final state
   - Preserves anchor transitions (anchors are not epsilon transitions)

2. **Unreachable State Removal** - O(|Q| + |δ|)
   - BFS from start state to find reachable states
   - Removes unreachable states and their transitions
   - Renumbers states to maintain contiguous IDs

3. **Dead State Removal** - O(|Q| + |δ|)
   - Builds reverse transition graph
   - Backward BFS from all final states
   - Removes states that cannot reach any final state

4. **Transition Deduplication**
   - Removes duplicate transitions created during epsilon elimination
   - Uses hash set for O(1) duplicate detection

### Configuration

```rust
use liblevenshtein::phonetic::nfa::{compile, OptimizationConfig, NFACompilerChar};
use liblevenshtein::phonetic::regex::parse;

// Full optimization (default)
let regex = parse("(ph|f)one")?;
let nfa = compile(&regex)?;  // Automatically optimized

// Custom configuration
let mut compiler = NFACompilerChar::new()
    .with_optimization(OptimizationConfig::quick());  // No epsilon elimination
let nfa = compiler.compile(&regex)?;

// Disable optimization (for debugging)
let mut compiler = NFACompilerChar::new()
    .without_optimization();
let unoptimized = compiler.compile(&regex)?;

// Manual optimization with statistics
let (optimized, stats) = unoptimized.optimize_with(OptimizationConfig::full());
println!("States: {} → {}", stats.original_states, stats.final_states);
println!("Transitions: {} → {}", stats.original_transitions, stats.final_transitions);
println!("Epsilon transitions eliminated: {}", stats.epsilon_transitions_eliminated);
```

### Configuration Presets

| Preset | Epsilon Elimination | Remove Unreachable | Remove Dead | Deduplicate |
|--------|:------------------:|:------------------:|:-----------:|:-----------:|
| `full()` | ✓ | ✓ | ✓ | ✓ |
| `quick()` | ✗ | ✓ | ✓ | ✗ |
| `none()` | ✗ | ✗ | ✗ | ✗ |

- **`full()`**: Maximum optimization, best for production use
- **`quick()`**: Fast optimization without expensive epsilon elimination
- **`none()`**: No optimization, useful for debugging Thompson construction

### Expected Impact

| Metric | Before Optimization | After Optimization |
|--------|--------------------|--------------------|
| Epsilon transitions | O(n) | 0 |
| State count | 100% | 70-90% |
| Transition count | 100% | 70-85% |
| Match runtime | Epsilon closure per step | Direct transitions |

---

## Implementation Examples

### Example 1: Basic Phonetic Regex

**Pattern**: `(ph|f)one`

**Rust Implementation**:
```rust
use liblevenshtein::phonetic::PhoneticRegex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile regex
    let regex = PhoneticRegex::compile("(ph|f)one")?;

    // Test matches
    assert!(regex.is_match("phone"));
    assert!(regex.is_match("fone"));
    assert!(!regex.is_match("bone"));

    // Find all matches in dictionary
    let dict = vec!["phone", "fone", "bone", "tone"];
    let matches: Vec<_> = dict.iter()
        .filter(|w| regex.is_match(w))
        .collect();

    assert_eq!(matches, vec![&"phone", &"fone"]);

    Ok(())
}
```

### Example 2: Contextual Rule

**Pattern**: `c → s / _[ei]`

**Rust Implementation**:
```rust
use liblevenshtein::phonetic::{PhoneticRegex, Context};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile contextual rule
    let rule = PhoneticRegex::compile("c → s / _[ei]")?;

    // Apply rule
    assert_eq!(rule.apply("city"), "sity");
    assert_eq!(rule.apply("cent"), "sent");
    assert_eq!(rule.apply("cat"), "cat");   // No change (a, not e/i)
    assert_eq!(rule.apply("cot"), "cot");   // No change (o, not e/i)

    Ok(())
}
```

### Example 3: Weighted Phonetic + Levenshtein

**Goal**: Find phonetically similar words within edit distance 2.

**Rust Implementation**:
```rust
use liblevenshtein::phonetic::PhoneticRegex;
use liblevenshtein::transducer::{Transducer, Algorithm};
use liblevenshtein::dawg::DoubleArrayTrie;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load dictionary
    let dict = DoubleArrayTrie::from_file("dict.txt")?;
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Compile phonetic pattern
    let phonetic = PhoneticRegex::compile("(ph|f)(o|oa)(n|ne)")?;

    // Query with combined phonetic + edit distance
    let query = "fone";
    let max_distance = 2;

    // Method 1: Sequential (phonetic then Levenshtein)
    let phonetic_variants = phonetic.expand(query);
    let mut all_matches = Vec::new();

    for variant in phonetic_variants {
        let matches: Vec<_> = transducer
            .query(&variant, max_distance)
            .collect();
        all_matches.extend(matches);
    }

    // Method 2: Product automaton (simultaneous)
    let product = phonetic.intersect_with_levenshtein(&transducer, query, max_distance)?;
    let matches: Vec<_> = product.collect();

    // Expected: ["phone", "phones", "phoned", "fone", ...]
    assert!(matches.contains(&"phone".to_string()));

    Ok(())
}
```

### Example 4: Verified Rules Integration

**Load Coq-Verified Rules**:
```rust
use liblevenshtein::phonetic::{PhoneticNFA, VerifiedRules};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load verified rules from Coq extraction
    let rules = VerifiedRules::load("orthography_rules.json")?;

    // Compile to NFA
    let nfa = PhoneticNFA::from_verified_rules(&rules)?;

    // Test equivalence with Coq implementation
    let test_cases = vec![
        ("phone", "fone"),
        ("knight", "nite"),
        ("tough", "tuff"),
        ("city", "sity"),
    ];

    for (input, expected) in test_cases {
        let nfa_result = nfa.apply(input)?;
        let coq_result = rules.apply_coq(input)?;  // Call Coq-verified impl

        assert_eq!(nfa_result, coq_result,
            "NFA must match Coq verified implementation for input: {}", input);
    }

    Ok(())
}
```

### Example 5: Incremental Streaming

**Real-time Chat Normalization**:
```rust
use liblevenshtein::phonetic::{PhoneticNFA, IncrementalMatcher};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nfa = PhoneticNFA::compile("(ph|f)one")?;
    let mut matcher = IncrementalMatcher::new(nfa);

    let input = "I saw a fone yesterday";
    let mut candidates = Vec::new();
    let mut current_word = String::new();

    for ch in input.chars() {
        if ch.is_whitespace() {
            // End of word
            if matcher.is_accepting() {
                candidates.push(current_word.clone());
            }
            matcher.reset();
            current_word.clear();
        } else {
            matcher.feed(ch);
            current_word.push(ch);
        }
    }

    // Check last word
    if matcher.is_accepting() {
        candidates.push(current_word);
    }

    assert_eq!(candidates, vec!["fone"]);

    Ok(())
}
```

---

## References

### Key Papers

1. **Thompson, K.** (1968). Programming Techniques: Regular expression search algorithm. Communications of the ACM, 11(6), 419-422.
   - Original Thompson's construction algorithm

2. **Hopcroft, J.E., Ullman, J.D.** (1979). Introduction to Automata Theory, Languages, and Computation. Addison-Wesley.
   - Comprehensive treatment of NFA, DFA, minimization

3. **Aho, A.V., Sethi, R., Ullman, J.D.** (1986). Compilers: Principles, Techniques, and Tools (Dragon Book). Addison-Wesley.
   - Chapter 3: Lexical Analysis (regex to NFA/DFA)

4. **Schulz, K.U., Mihov, S.** (2002). Fast String Correction with Levenshtein-Automata. IJDAR, 5, 67-85.
   - Levenshtein automaton construction

5. **Brill, E., Moore, R.C.** (2000). An Improved Error Model for Noisy Channel Spelling Correction. ACL 2000.
   - Phonetic patterns in spelling correction

### Tools

- **Regex Crate** (Rust): https://docs.rs/regex/
  - Production-quality regex engine (reference implementation)

- **RE2** (Google): https://github.com/google/re2
  - DFA-based regex engine (guaranteed linear time)

- **Ragel**: http://www.colm.net/open-source/ragel/
  - State machine compiler for lexical analysis

### Additional Resources

- **RegexOne**: https://regexone.com/
  - Interactive regex tutorial

- **Debuggex**: https://www.debuggex.com/
  - Visualize regex as NFA/DFA

- **Regex101**: https://regex101.com/
  - Test regex patterns online
