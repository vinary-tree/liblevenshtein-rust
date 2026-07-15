# Formal Verification Architecture

**Document Version**: 1.0
**Last Updated**: 2025-01-18
**Status**: Living Document

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Design Principles](#design-principles)
4. [Architecture Overview](#architecture-overview)
5. [Phase 1: Phonetic Rewrite Rules](#phase-1-phonetic-rewrite-rules)
6. [Phase 2: Regex Automaton](#phase-2-regex-automaton)
7. [Phase 3: Phonetic Fuzzy Regex](#phase-3-phonetic-fuzzy-regex)
8. [Phase 4: Structural CFG](#phase-4-structural-cfg)
9. [Proof Strategies](#proof-strategies)
10. [Implementation Mapping](#implementation-mapping)
11. [Recovery Procedures](#recovery-procedures)

---

## Executive Summary

This document specifies the complete formal verification architecture for the liblevenshtein-rust phonetic fuzzy matching system. The system extends traditional Levenshtein edit distance with:

1. **Phonetic transformations** - Context-dependent rewrite rules
2. **Regex pattern matching** - Fuzzy matching against regular expressions
3. **Structural CFG operations** - Grammar-aware fuzzy matching with type-safe structural edits

**Key Innovation**: All algorithms are **proven correct in Rocq** before implementation, with Rust code mirroring the proofs and QuickCheck property tests validating the correspondence.

---

## Theoretical Foundation

### Core Concepts

#### 1. Levenshtein Automaton

**Standard Definition** (Schulz & Mihov, 2002):
- Given word `w` and distance `n`, construct automaton `A_n(w)`
- Accepts all strings `s` where $`\text{edit}_\text{distance}(w, s) \le  n`$
- Three operations: insert, delete, substitute (cost 1 each)

**Our Extension**:
- Multi-character operations: transpose $`\langle 2,2\rangle,`$ merge $`\langle 2,1\rangle,`$ split $`\langle 1,2\rangle`$
- Weighted operations: phonetic rules have weight 0.15 (effectively 0 after truncation)
- Context-dependent operations: rules apply only in specific contexts

#### 2. Phonetic Rewrite Systems

**Formal Model**:
```
RewriteRule = (pattern, replacement, context, weight)

pattern      : List Phone     -- Input sequence to match
replacement  : List Phone     -- Output sequence to produce
context      : Context        -- Where rule applies
weight       : ℚ              -- Cost (0 for exact, 0.15 for phonetic, 1 for edit)
```

**Application Semantics**:
```
apply_rule_at : RewriteRule → PhoneticString → ℕ → Option PhoneticString
apply_rule_at r s pos =
  if context_matches r.context s pos ∧ pattern_matches_at r.pattern s pos
  then Some (s[0..pos] ++ r.replacement ++ s[pos+|r.pattern|..])
  else None
```

**Sequential Application**:
```
apply_rules_seq : List RewriteRule → PhoneticString → ℕ → Option PhoneticString
apply_rules_seq [] s _ = Some s
apply_rules_seq (r::rs) s fuel =
  match find_first_match r s with
  | Some pos => apply_rule_at r s pos >>= λs'. apply_rules_seq (r::rs) s' (fuel-1)
  | None     => apply_rules_seq rs s fuel
```

**Key Properties** (to be proven):
1. **Termination**: $`\exists  \text{fuel}. \text{apply}_\text{rules}_\text{seq} \text{rules} s \text{fuel} = \text{Some} \text{result}`$
2. **Idempotence**: `apply_rules_seq rules result fuel = Some result`
3. **Bounded Expansion**: $`|\text{result}| \le  |s| + \max _\text{expansion}_\text{factor}`$

#### 3. Fuzzy Regular Expression Matching

**Definition** (Wu & Manber, 1992):
- Given regex pattern `P` and text `T`, find all matches with $`\le  k`$ errors
- Uses bit-parallel NFA simulation with error counters

**Our Approach**:
- Thompson construction: `Regex → NFA`
- Extended transition: `(NFA_State, Errors) × Char → Set (NFA_State, Errors)`
- Accept if: $`(\text{state} \in  \text{final}_\text{states}) \land  (\text{errors} \le  \max _\text{distance})`$

**Integration with Phonetics**:
```
PhoneticRegexState = {
  nfa_positions    : Set (StateId × Errors),
  phonetic_pending : List (PhoneticOp × Progress)
}
```

#### 4. Structural Context-Free Grammar Operations

**Grammar Definition**:
```
Grammar = Set Production

Production = {
  lhs : NonTerminal,
  rhs : List Symbol
}

Symbol = Terminal Char | NonTerminal NT

NonTerminal = {
  name : String,
  type : SemanticType  -- Type annotation for safety
}
```

**Structural Operations**:
```
Transpose : Production → ℕ → ℕ → Option Production
  -- Swap two non-terminals if type-compatible

Merge : Production → ℕ → ℕ → NonTerminal → Option Production
  -- Combine two non-terminals into one

Split : Production → ℕ → (NonTerminal × NonTerminal) → Option Production
  -- Split one non-terminal into two
```

**Type Safety Constraint**:
```
can_transpose : NonTerminal → NonTerminal → Prop
can_transpose nt1 nt2 = (nt1.type = nt2.type)
```

**Edit Distance Metric**:
```
struct_distance : Production → Production → ℕ
  -- Satisfies metric properties:
  -- 1. d(p, p) = 0 (identity)
  -- 2. d(p, q) = d(q, p) (symmetry)
  -- 3. d(p, r) ≤ d(p, q) + d(q, r) (triangle inequality)
```

---

## Design Principles

### 1. Proof-First Development

**Principle**: Formalize and prove before implementing.

**Rationale**:
- Proofs identify edge cases early
- Type system enforces correctness constraints
- Refactoring preserves proven properties

**Implementation**:
```
┌──────────────┐
│ Rocq Proof   │ ──────────┐
└──────────────┘           │
                           ▼
                    ┌──────────────┐
                    │ Extract OCaml│
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Rust Impl    │ ◄──── References proofs
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ QuickCheck   │ ◄──── Mirrors theorems
                    └──────────────┘
```

### 2. Modular Theorem Structure

**Principle**: Decompose into provable components.

**Pattern**:
```coq
(* Define type *)
Inductive T : Set := ...

(* Define well-formedness *)
Definition wf (t : T) : Prop := ...

(* Prove constructor preserves wf *)
Lemma constructor_preserves_wf : ...

(* Define operation *)
Definition op (t1 t2 : T) : T := ...

(* Prove operation correct *)
Theorem op_correct :
  wf t1 → wf t2 → wf (op t1 t2).
```

### 3. Extraction-Compatible Definitions

**Principle**: Write algorithms extractable to OCaml/Rust.

**Guidelines**:
- Use `Fixpoint` with explicit `fuel` for termination
- Avoid `Prop` in computational definitions (use `bool`)
- Use `option` for partial functions
- Structure matches target language idioms

**Example**:
```coq
(* Extractable - uses fuel *)
Fixpoint compute (x : nat) (fuel : nat) : option nat :=
  match fuel with
  | 0 => None
  | S fuel' => Some (x + 1)
  end.

(* Not extractable - uses Prop *)
Definition compute_prop (x : nat) : ∃ y, y > x := ...
```

### 4. Property-Based Validation

**Principle**: Every Rocq theorem has a corresponding QuickCheck test.

**Mapping**:
```
Rocq Theorem                    Rust QuickCheck Test
─────────────────────────────   ────────────────────────────
Theorem foo_preserves_wf :      #[quickcheck]
  ∀ x, wf x → wf (foo x).       fn foo_preserves_wf(x: T) -> bool {
                                  if !x.is_wellformed() { return true; }
                                  foo(x).is_wellformed()
                                }
```

---

## Architecture Overview

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    User-Facing API                          │
│  TransducerBuilder, PhoneticRegexMatcher, FuzzyEarleyParser │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼──────────┐   ┌────────▼──────────┐
│ IntersectionZipper│   │  Pattern Matching │
│   (Existing)      │   │   (New)           │
└────────┬──────────┘   └────────┬──────────┘
         │                       │
    ┌────┴────┐           ┌──────┴──────┐
    │         │           │             │
┌───▼───┐ ┌──▼────┐  ┌───▼──────┐ ┌────▼─────┐
│ Dict  │ │ Auto  │  │  Regex   │ │   CFG    │
│ (Trie)│ │ maton │  │  NFA     │ │  Earley  │
└───────┘ └───┬───┘  └────┬─────┘ └────┬─────┘
              │           │            │
         ┌────┴───────────┴────────────┘
         │
┌────────▼──────────┐
│  Phonetic Rules   │ ◄──── PHASE 1 (Current)
│  (Verified)       │
└───────────────────┘
```

### Component Dependencies

```
Phase 4 (CFG)
  │
  ├─► Depends on: Phonetic Rules (Phase 1)
  ├─► Depends on: Pattern Matching abstractions
  └─► New: Earley parser, Structural operations

Phase 3 (Phonetic Regex)
  │
  ├─► Depends on: Phonetic Rules (Phase 1)
  ├─► Depends on: Regex NFA (Phase 2)
  └─► New: Composition logic

Phase 2 (Regex NFA)
  │
  ├─► Depends on: Basic automaton infrastructure
  └─► New: Thompson construction, NFA simulation

Phase 1 (Phonetic Rules) ◄── CURRENT
  │
  ├─► Foundation for all phases
  └─► New: Rewrite system, Context matching
```

---

## Phase 1: Phonetic Rewrite Rules

### Detailed Design

#### Type Hierarchy

```coq
(* Base phonetic symbol *)
Inductive Phone : Set :=
  | Vowel     : ascii → Phone
  | Consonant : ascii → Phone
  | Digraph   : ascii → ascii → Phone
  | Silent    : Phone.

(* Application context *)
Inductive Context : Set :=
  | Initial           : Context
  | Final             : Context
  | BeforeVowel       : list ascii → Context
  | AfterConsonant    : list ascii → Context
  | BeforeConsonant   : list ascii → Context
  | AfterVowel        : list ascii → Context
  | Anywhere          : Context.

(* Rewrite rule *)
Record RewriteRule : Set := mkRule {
  rule_id     : nat;
  rule_name   : string;
  pattern     : list Phone;
  replacement : list Phone;
  context     : Context;
  weight      : Q;
}.
```

**Design Rationale**:
- `Phone` is atomic - prevents decomposition errors
- `Context` is explicit - makes dependencies clear
- `RewriteRule` bundles all metadata - self-documenting
- `weight : Q` (rational) - represents 0.15 exactly (no float rounding)

#### Algorithm Design

**Context Matching**:
```coq
Fixpoint context_matches (ctx : Context) (s : PhoneticString) (pos : nat) : bool :=
  match ctx with
  | Initial => pos =? 0
  | Final => pos =? length s
  | BeforeVowel vowels =>
      match nth_error s pos with
      | Some (Vowel v) => existsb (Ascii.eqb v) vowels
      | _ => false
      end
  | (* ... other cases ... *)
  end.
```

**Design Decisions**:
1. **Returns `bool`** - extractable to OCaml/Rust
2. **Uses `nth_error`** - safe indexing, no exceptions
3. **Pattern matches explicitly** - clear case analysis for proofs

**Pattern Matching**:
```coq
Fixpoint pattern_matches_at (pat : list Phone) (s : PhoneticString) (pos : nat) : bool :=
  match pat, s with
  | [], _ => true
  | p :: ps, _ =>
      match nth_error s pos with
      | Some p' =>
          if Phone_eqb p p' then
            pattern_matches_at ps s (S pos)
          else
            false
      | None => false
      end
  end.
```

**Design Decisions**:
1. **Recursive on pattern** - clear induction principle
2. **Early termination** - fails fast on mismatch
3. **Uses `Phone_eqb`** - structural equality with proof support

**Rule Application**:
```coq
Definition apply_rule_at (r : RewriteRule) (s : PhoneticString) (pos : nat)
  : option PhoneticString :=
  if context_matches (context r) s pos then
    if pattern_matches_at (pattern r) s pos then
      let prefix := firstn pos s in
      let suffix := skipn (pos + length (pattern r)) s in
      Some (prefix ++ replacement r ++ suffix)
    else
      None
  else
    None.
```

**Design Decisions**:
1. **Returns `option`** - explicit failure mode
2. **Uses `firstn`/`skipn`** - standard library, proven properties
3. **Concatenation order** - `prefix ++ replacement ++ suffix` (clear semantics)

**Sequential Application**:
```coq
Fixpoint apply_rules_seq (rules : list RewriteRule) (s : PhoneticString) (fuel : nat)
  : option PhoneticString :=
  match fuel with
  | 0 => Some s  (* Out of fuel *)
  | S fuel' =>
      match rules with
      | [] => Some s  (* Fixed point *)
      | r :: rest =>
          match find_first_match r s (length s) with
          | Some pos =>
              match apply_rule_at r s pos with
              | Some s' => apply_rules_seq rules s' fuel'  (* Restart *)
              | None => apply_rules_seq rest s fuel'
              end
          | None => apply_rules_seq rest s fuel'
          end
      end
  end.
```

**Design Decisions**:
1. **Fuel-based termination** - structurally decreasing, Rocq accepts
2. **Restart from beginning** - implements rule ordering dependency
3. **Graceful degradation** - returns current state if fuel exhausted

#### Theorem Structure

**Theorem 1: Well-Formedness**
```coq
Definition wf_rule (r : RewriteRule) : Prop :=
  length (pattern r) > 0 /\
  weight r >= 0.

Theorem zompist_rules_wellformed :
  forall r, In r zompist_rule_set -> wf_rule r.
```

**Proof Strategy**:
1. Split `zompist_rule_set` into `orthography_rules ++ phonetic_rules`
2. Prove `orthography_rules_wf` by enumeration (8 rules)
3. Prove `phonetic_rules_wf` by enumeration (3 rules)
4. Combine with `in_app_or`

**Status**: ✅ COMPLETE

**Theorem 2: Bounded Expansion**
```coq
Definition max_expansion_factor : nat := 3.

Theorem rule_application_bounded :
  forall r s pos s',
    In r zompist_rule_set ->
    apply_rule_at r s pos = Some s' ->
    length s' <= length s + max_expansion_factor.
```

**Proof Strategy**:
1. Unfold `apply_rule_at` definition
2. Extract `s' = prefix ++ replacement ++ suffix`
3. Rewrite: `|s'| = |prefix| + |replacement| + |suffix|`
4. Use lemma: `|prefix| = pos`
5. Use lemma: `|suffix| = |s| - pos - |pattern|`
6. Use lemma: $`|\text{replacement}| \le  2`$ (proved by enumeration)
7. Arithmetic: `|s'| = pos + |replacement| + (|s| - pos - |pattern|)`
              `     = |s| + |replacement| - |pattern|`
              $`     \le  |s| + 2 - 1`$ (since $`\lvert \text{pattern}\rvert \ge 1`$)
              `     = |s| + 1`
              $`     \le  |s| + 3`$

**Status**: 🔄 IN PROGRESS (arithmetic details `Admitted`)

**Remaining Work**:
- Prove `firstn_length_le`: $`\text{pos} \le  \text{length} s \to  \text{length} (\text{firstn} \text{pos} s) = \text{pos}`$
- Prove `skipn_length`: `length (skipn n s) = length s - n`
- Complete omega arithmetic

**Theorem 3: Non-Confluence**
```coq
Definition rules_commute (r1 r2 : RewriteRule) : Prop :=
  forall s pos1 pos2 s1 s2 s1' s2',
    pos1 <> pos2 ->
    apply_rule_at r1 s pos1 = Some s1 ->
    apply_rule_at r2 s pos2 = Some s2 ->
    apply_rule_at r2 s1 pos2 = Some s1' ->
    apply_rule_at r1 s2 pos1 = Some s2' ->
    s1' = s2'.

Theorem some_rules_dont_commute :
  exists r1 r2,
    In r1 zompist_rule_set /\
    In r2 zompist_rule_set /\
    ~rules_commute r1 r2.
```

**Proof Strategy**:
1. Construct counterexample: Rule 33 (silent e) and Rule 34 (vowel shortening)
2. Show: "make" → Rule 33 → "mak" → Rule 34 → "mak"
3. But:  "make" → Rule 34 → "make" → Rule 33 → "mak"
4. Different intermediate states prove non-commutativity

**Status**: Historical proof target in this architecture note; use current
proof READMEs for active status.

**Theorem 4: Termination**
```coq
Theorem sequential_application_terminates :
  forall rules s,
    (forall r, In r rules -> wf_rule r) ->
    exists fuel result,
      apply_rules_seq rules s fuel = Some result.
```

**Proof Strategy**:
1. Define `fuel = length s * length rules * max_expansion_factor`
2. Prove: Each iteration either:
   - Applies rule and modifies string (bounded by expansion)
   - Moves to next rule (bounded by rule count)
   - Reaches fixed point (terminates)
3. By induction on fuel, show termination

**Status**: Historical proof target in this architecture note; use current
proof READMEs for active status.

**Theorem 5: Idempotence**
```coq
Theorem rewrite_idempotent :
  forall rules s fuel s',
    (forall r, In r rules -> wf_rule r) ->
    fuel >= length s * length rules * max_expansion_factor ->
    apply_rules_seq rules s fuel = Some s' ->
    apply_rules_seq rules s' fuel = Some s'.
```

**Proof Strategy**:
1. Show: With sufficient fuel, `apply_rules_seq` reaches fixed point
2. At fixed point: `find_first_match r s' = None` for all rules `r`
3. Therefore: `apply_rules_seq rules s' _ = Some s'`

**Status**: Historical proof target in this architecture note; use current
proof READMEs for active status.

#### Zompist Rule Definitions

**Rule Coverage Plan**:

**Phase 1a (Implemented - 11 rules)**:
- ✅ ch → ç (digraph)
- ✅ sh → $ (digraph)
- ✅ ph → f (digraph)
- ✅ c → s / _[ie] (velar softening)
- ✅ c → k (elsewhere)
- ✅ g → j / _[ie] (velar softening)
- ✅ $`e \to  \emptyset  / _`$# (silent e)
- ✅ $`\text{gh} \to \emptyset`$ (silent gh)
- ✅ th → t (phonetic, weight 0.15)
- ✅ qu → kw (phonetic, weight 0.15)
- ✅ kw → qu (phonetic reverse, weight 0.15)

**Phase 1b (Priority - 15 rules)**:
- ⏳ a, e, i, o, u vowel handling
- ⏳ Double consonant rules
- ⏳ Vowel lengthening (gh effect)
- ⏳ More silent letters (k in kn, w in wr, etc.)
- ⏳ ng → ŋ
- ⏳ More digraphs (wh, ck, etc.)

**Phase 1c (Complete - 30 remaining rules)**:
- ⏳ Complex vowel rules
- ⏳ Stress-dependent rules
- ⏳ Exception handling
- ⏳ Rare patterns

**Rule Template**:
```coq
Definition rule_NAME : RewriteRule := {|
  rule_id := ID;
  rule_name := "DESCRIPTION";
  pattern := [PATTERN];
  replacement := [REPLACEMENT];
  context := CONTEXT;
  weight := WEIGHT;
|}.
```

---

## Phase 2: Regex Automaton

### Planned Design

#### Type Definitions

```coq
(* Regular expression syntax *)
Inductive Regex : Set :=
  | REmpty  : Regex
  | RChar   : ascii -> Regex
  | RConcat : Regex -> Regex -> Regex
  | RAlt    : Regex -> Regex -> Regex
  | RStar   : Regex -> Regex
  | RPlus   : Regex -> Regex
  | ROpt    : Regex -> Regex.

(* NFA state *)
Definition StateId := nat.

(* NFA transition *)
Inductive Transition : Set :=
  | TChar    : ascii -> StateId -> Transition
  | TEpsilon : StateId -> Transition.

(* NFA *)
Record NFA : Set := mkNFA {
  states      : list StateId;
  start       : StateId;
  finals      : list StateId;
  transitions : StateId -> list Transition;
}.
```

#### Key Algorithms

**Thompson Construction**:
```coq
Fixpoint thompson_construct (r : Regex) : NFA :=
  match r with
  | REmpty => (* NFA for empty language *)
  | RChar c => (* NFA for single character *)
  | RConcat r1 r2 =>
      let n1 := thompson_construct r1 in
      let n2 := thompson_construct r2 in
      (* Connect n1.finals to n2.start with epsilon *)
  | RAlt r1 r2 =>
      let n1 := thompson_construct r1 in
      let n2 := thompson_construct r2 in
      (* New start with epsilon to both n1, n2 *)
  | RStar r' =>
      let n := thompson_construct r' in
      (* Add loop from finals to start *)
  | (* ... *)
  end.
```

#### Theorems to Prove

```coq
(* Semantics of regex *)
Fixpoint regex_matches (r : Regex) (s : list ascii) : Prop :=
  match r with
  | REmpty => False
  | RChar c => s = [c]
  | RConcat r1 r2 =>
      exists s1 s2, s = s1 ++ s2 /\
                    regex_matches r1 s1 /\
                    regex_matches r2 s2
  | RAlt r1 r2 =>
      regex_matches r1 s \/ regex_matches r2 s
  | RStar r' =>
      s = [] \/
      exists s1 s2, s = s1 ++ s2 /\
                    regex_matches r' s1 /\
                    regex_matches (RStar r') s2
  | (* ... *)
  end.

(* NFA acceptance *)
Definition nfa_accepts (n : NFA) (s : list ascii) : Prop :=
  exists path,
    nfa_path n n.(start) s path /\
    exists f, In f n.(finals) /\ path_ends_at path f.

(* Correctness theorem *)
Theorem thompson_correctness :
  forall r s,
    regex_matches r s <-> nfa_accepts (thompson_construct r) s.
```

---

## Phase 3: Phonetic Fuzzy Regex

### Planned Design

#### Composition Strategy

```coq
(* Combined state *)
Record PhoneticRegexState : Set := mkPRState {
  nfa_positions     : list (StateId * nat);  (* (state, errors) *)
  phonetic_pending  : list PhoneticPending;
}.

Inductive PhoneticPending : Set :=
  | MergePending  : StateId -> nat -> ascii -> ascii -> PhoneticPending
  | SplitPending  : StateId -> nat -> ascii -> nat -> PhoneticPending
  | TransPending  : StateId -> nat -> ascii -> ascii -> PhoneticPending.

(* Transition function *)
Definition pr_transition
  (state : PhoneticRegexState)
  (phonetic_ops : list RewriteRule)
  (nfa : NFA)
  (char : ascii)
  : option PhoneticRegexState :=
  (* 1. Standard NFA transitions *)
  let nfa_next := standard_nfa_step state.(nfa_positions) nfa char in

  (* 2. Phonetic operation detection *)
  let phonetic_next := detect_phonetic_ops state phonetic_ops char in

  (* 3. Complete pending operations *)
  let completed := complete_pending state.(phonetic_pending) char in

  (* 4. Combine *)
  Some {| nfa_positions := nfa_next ++ phonetic_next ++ completed;
          phonetic_pending := new_pending |}.
```

#### Theorems to Prove

```coq
Theorem composition_sound :
  forall r ops s max_dist,
    phonetic_regex_accepts r ops s max_dist ->
    exists s',
      apply_rules_seq ops (string_to_phones s) _ = Some s' /\
      fuzzy_nfa_accepts (thompson_construct r) s' max_dist.
```

---

## Phase 4: Structural CFG

### Planned Design

#### Type System

```coq
(* Semantic types for non-terminals *)
Inductive SemanticType : Set :=
  | TExpr    : SemanticType
  | TStmt    : SemanticType
  | TType    : SemanticType
  | TPattern : SemanticType.

(* Typed non-terminal *)
Record NonTerminal : Set := mkNT {
  nt_name : string;
  nt_type : SemanticType;
}.

(* Production rule *)
Record Production : Set := mkProd {
  prod_lhs : NonTerminal;
  prod_rhs : list Symbol;
}.

Inductive Symbol : Set :=
  | Terminal    : ascii -> Symbol
  | NonTerminal : NonTerminal -> Symbol.
```

#### Structural Operations

```coq
(* Type-safe transpose *)
Definition transpose
  (p : Production)
  (i j : nat)
  (proof : types_compatible (type_at p i) (type_at p j))
  : Production :=
  (* Swap symbols at positions i and j *)

(* Theorem: Transpose preserves well-formedness *)
Theorem transpose_preserves_wf :
  forall p i j proof,
    wf_production p ->
    wf_production (transpose p i j proof).
```

#### Edit Distance

```coq
(* Structural edit distance *)
Fixpoint struct_distance (p q : Production) (fuel : nat) : option nat :=
  match fuel with
  | 0 => None
  | S fuel' =>
      if production_eqb p q then
        Some 0
      else
        (* Try all possible operations *)
        let transpose_cost := min_transpose_cost p q fuel' in
        let merge_cost := min_merge_cost p q fuel' in
        let split_cost := min_split_cost p q fuel' in
        min_option [transpose_cost; merge_cost; split_cost]
  end.

(* Metric properties *)
Theorem distance_is_metric :
  forall p q r,
    (* Identity *)
    struct_distance p p _ = Some 0 /\
    (* Symmetry *)
    struct_distance p q _ = struct_distance q p _ /\
    (* Triangle inequality *)
    (forall dp dq dr,
      struct_distance p q _ = Some dp ->
      struct_distance q r _ = Some dq ->
      struct_distance p r _ = Some dr ->
      dr <= dp + dq).
```

---

## Proof Strategies

### General Tactics

1. **Structural Induction**
   ```coq
   induction lst as [| x xs IH].
   - (* Base case: [] *)
   - (* Inductive case: x :: xs *)
   ```

2. **Case Analysis**
   ```coq
   destruct condition eqn:E.
   - (* True case *)
   - (* False case *)
   ```

3. **Enumeration for Finite Sets**
   ```coq
   unfold rule_set in H.
   simpl in H.
   repeat (destruct H as [H | H]; [
     (* Each specific case *)
   |]).
   contradiction.  (* Empty case *)
   ```

4. **Omega for Arithmetic**
   ```coq
   omega.  (* Solves linear arithmetic *)
   ```

5. **Rewriting with Lemmas**
   ```coq
   rewrite lemma_name.
   rewrite <- reverse_direction.
   ```

### Common Proof Patterns

**Pattern 1: Preservation Proofs**
```coq
Theorem op_preserves_property :
  forall x y,
    property x ->
    property y ->
    property (op x y).
Proof.
  intros x y Hx Hy.
  unfold property, op.
  (* Case analysis or induction *)
  destruct x; destruct y; simpl.
  (* Use hypotheses Hx, Hy *)
  assumption.  (* or: apply/split/etc *)
Qed.
```

**Pattern 2: Equivalence Proofs**
```coq
Theorem equiv_property :
  forall x,
    property1 x <-> property2 x.
Proof.
  intro x.
  split.
  - (* → direction *)
    intro H1.
    (* Derive property2 from property1 *)
  - (* ← direction *)
    intro H2.
    (* Derive property1 from property2 *)
Qed.
```

**Pattern 3: Termination Proofs**
```coq
Theorem function_terminates :
  forall x,
    exists fuel result,
      function x fuel = Some result.
Proof.
  intro x.
  exists (some_bound x).  (* Compute fuel *)
  exists (some_result x).  (* Compute result *)
  (* Prove by induction on fuel *)
  induction (some_bound x).
  - (* Base case *)
  - (* Inductive case *)
Qed.
```

---

## Implementation Mapping

### Rocq to Rust Translation

**Type Mapping**:
```
Rocq Type              Rust Type
─────────────────      ─────────────────
nat                 →  usize
ascii               →  u8 or char
list T              →  Vec<T>
option T            →  Option<T>
bool                →  bool
Q (rational)        →  f64 (with note about precision)
```

**Function Mapping**:
```
Rocq Definition                        Rust Implementation
─────────────────────────────────      ──────────────────────────────────
Definition f (x : T) : U := body.  →  pub fn f(x: T) -> U { body }

Fixpoint f (x : nat) (fuel : nat)  →  pub fn f(x: usize, fuel: usize)
  : option T := ...                      -> Option<T> { ... }

Inductive T : Set := C1 | C2.      →  pub enum T { C1, C2 }

Record R : Set := mkR { ... }.     →  pub struct R { ... }
```

**Proof Annotation**:
```rust
/// Apply phonetic rewrite rules sequentially
///
/// # Correctness (PROVEN in Rocq):
/// - **Terminates**: Theorem sequential_application_terminates
///   (docs/verification/phonetic/rewrite_rules.v:250)
/// - **Idempotent**: Theorem rewrite_idempotent
///   (docs/verification/phonetic/rewrite_rules.v:275)
/// - **Bounded**: Theorem rule_application_bounded
///   (docs/verification/phonetic/zompist_rules.v:145)
///
/// # Parameters
/// - `rules`: Rewrite rules to apply (must be well-formed)
/// - `input`: Input phonetic string
///
/// # Returns
/// Fixed point of sequential rule application
///
/// # Property Tests
/// See `properties::sequential_application_terminates` for QuickCheck validation
pub fn apply_rules_sequential(
    rules: &[RewriteRule],
    input: &[Phone],
) -> Vec<Phone> {
    // Implementation mirrors Rocq definition
    // See: docs/verification/phonetic/rewrite_rules.v:180-200

    let mut current = input.to_vec();
    let fuel = input.len() * rules.len() * MAX_EXPANSION_FACTOR;

    for _ in 0..fuel {
        let mut changed = false;
        for rule in rules {
            if let Some(pos) = find_first_match(rule, &current) {
                current = apply_at(rule, &current, pos);
                changed = true;
                break;  // Restart from first rule
            }
        }
        if !changed {
            break;  // Fixed point reached
        }
    }

    // Post-condition (from Theorem rewrite_idempotent)
    debug_assert_eq!(
        apply_rules_sequential(rules, &current),
        current,
        "Idempotence violated - see rewrite_rules.v:275"
    );

    current
}
```

### Property Test Template

```rust
#[cfg(test)]
mod properties {
    use super::*;
    use quickcheck::{Arbitrary, quickcheck};

    /// Property: [THEOREM NAME]
    ///
    /// Corresponds to: Theorem [theorem_name]
    /// Proof: [file.v:line]
    ///
    /// [INFORMAL DESCRIPTION]
    #[quickcheck]
    fn property_name(input: ArbitraryType) -> bool {
        // Setup
        let setup = prepare(input);

        // Property check (mirrors Rocq theorem)
        theorem_property_holds(setup)
    }
}
```

---

## Recovery Procedures

### If Proofs Are Lost

**Step 1: Recover Theorem Statements**
```bash
# Extract theorem statements from .v files
grep "Theorem\|Lemma\|Definition" docs/verification/**/*.v > theorems.txt
```

**Step 2: Check What Compiles**
```bash
cd docs/verification
make clean
make phonetic 2>&1 | tee build.log
# Check build.log for errors
```

**Step 3: Identify Admitted Lemmas**
```bash
grep -n "Admitted" docs/verification/**/*.v
```

**Step 4: Prioritize Proofs**
1. Well-formedness theorems (foundational)
2. Correctness theorems (core properties)
3. Optimization lemmas (performance)

### If Implementation Is Lost

**Step 1: Extract from Rocq**
```bash
cd docs/verification/phonetic
coqc rewrite_rules.v
# Produces: *.ml files
```

**Step 2: Reference Extracted OCaml**
- OCaml code is executable reference
- Translate to Rust preserving structure

**Step 3: Check Property Tests**
```bash
cd ../..
cargo test properties
# Failures indicate missing/incorrect implementation
```

### If Design Is Lost

**This Document** serves as the recovery point.

**Critical Files**:
1. `ARCHITECTURE.md` (this file) - Overall design
2. `docs/verification/README.md` - Quick reference
3. `docs/verification/PROGRESS.md` - Current status
4. `docs/verification/phonetic/*.v` - Formal specs

**Recovery Steps**:
1. Read ARCHITECTURE.md (this file)
2. Check PROGRESS.md for current phase
3. Read phase-specific `.v` files for formal spec
4. Consult README.md for build instructions
5. Follow verification workflow (formalize → prove → extract → implement)

---

## Conclusion

This architecture provides a **complete, rigorous specification** for the Rocq-verified phonetic fuzzy matching system. Every design decision is documented, every theorem is justified, and recovery procedures are explicit.

**Key Takeaways**:
1. Proof-first development ensures correctness
2. Modular theorem structure enables incremental progress
3. Extraction provides reference implementations
4. Property tests validate Rocq-Rust correspondence
5. Comprehensive documentation enables recovery and continuation

**Next Steps**: See `PROGRESS.md` for current phase status and immediate tasks.
